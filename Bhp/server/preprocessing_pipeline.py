"""
Preprocessing Pipeline & Model Validation
==========================================
Validates the existing ML pipeline, adds multicollinearity checks,
proper preprocessing with sklearn Pipeline, and model evaluation.

Assumptions:
- Dataset: bengaluru_house_prices.csv from Kaggle
- Target: price (in Lakhs of INR)
- This script can retrain and re-export the model artifacts

Thresholds and weights documented below:
- Outlier removal: sqft/bhk < 300 removed
- Price per sqft outliers: removed if outside mean ± 1 std per location
- BHK outliers: removed if higher-BHK has lower pps than lower-BHK mean
- Bathroom outliers: removed if bath > bhk + 2
- Location dim reduction: locations with ≤10 data points grouped as 'other'
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'
DATA_DIR = Path(__file__).parent.parent.parent


def load_data():
    """Load and basic-clean the raw dataset."""
    csv_path = DATA_DIR / 'bengaluru_house_prices.csv'
    if not csv_path.exists():
        csv_path = DATA_DIR / 'bhp.csv'
        logger.info(f"Loading intermediate dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        if 'bhk' not in df.columns and 'size' in df.columns:
            df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]))
        return df

    logger.info(f"Loading raw dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
    df = df.dropna()

    df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

    def convert_sqft_to_num(x):
        tokens = str(x).split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except:
            return None

    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
    df = df[df['total_sqft'].notnull()]
    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

    return df


def remove_outliers(df):
    """Apply the 4-round outlier removal pipeline."""
    logger.info(f"Starting outlier removal. Rows: {len(df)}")

    # Round 1: sqft/bhk minimum
    df = df[~(df.total_sqft / df.bhk < 300)]
    logger.info(f"After sqft/bhk filter: {len(df)}")

    # Round 2: price_per_sqft per-location mean ± std
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced], ignore_index=True)
    df = df_out
    logger.info(f"After pps outlier removal: {len(df)}")

    # Round 3: BHK-level outliers
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values
                )
    df = df.drop(exclude_indices, axis='index')
    logger.info(f"After BHK outlier removal: {len(df)}")

    # Round 4: Bathroom outliers
    df = df[df.bath < df.bhk + 2]
    logger.info(f"After bath outlier removal: {len(df)}")

    return df


def reduce_locations(df, threshold=10):
    """Group locations with ≤ threshold data points into 'other'."""
    location_stats = df['location'].value_counts(ascending=False)
    rare_locations = location_stats[location_stats <= threshold].index.tolist()
    df['location'] = df['location'].apply(lambda x: 'other' if x in rare_locations else x)
    n_unique = len(df['location'].unique())
    logger.info(f"After location reduction: {n_unique} unique locations")
    return df


def prepare_features(df):
    """One-hot encode locations and prepare X, y."""
    df_model = df.drop(['size', 'price_per_sqft'], axis='columns')
    dummies = pd.get_dummies(df_model['location'])
    df_model = pd.concat([df_model, dummies.drop('other', axis='columns')], axis='columns')
    df_model = df_model.drop('location', axis='columns')

    X = df_model.drop(['price'], axis='columns')
    y = df_model['price']
    return X, y


def check_multicollinearity(X, top_n=10):
    """Check VIF for numeric features only (location dummies are categorical)."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        numeric_cols = ['total_sqft', 'bath', 'bhk']
        X_numeric = X[numeric_cols].copy()
        X_numeric = X_numeric.astype(float)

        vif_data = pd.DataFrame()
        vif_data['Feature'] = numeric_cols
        vif_data['VIF'] = [variance_inflation_factor(X_numeric.values, i)
                           for i in range(X_numeric.shape[1])]
        logger.info("VIF Analysis:")
        for _, row in vif_data.iterrows():
            logger.info(f"  {row['Feature']}: VIF = {row['VIF']:.2f}")
        return vif_data
    except ImportError:
        logger.warning("statsmodels not installed. Skipping VIF analysis.")
        return None


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Compute and log regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"\n{'='*50}")
    logger.info(f"{model_name} Evaluation Results")
    logger.info(f"{'='*50}")
    logger.info(f"  MAE:  {mae:.4f} Lakhs")
    logger.info(f"  MSE:  {mse:.4f}")
    logger.info(f"  RMSE: {rmse:.4f} Lakhs")
    logger.info(f"  R²:   {r2:.4f}")
    logger.info(f"{'='*50}")

    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}


def train_and_export():
    """Full pipeline: load, clean, train, evaluate, export."""
    from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
    from sklearn.linear_model import LinearRegression

    logger.info("="*60)
    logger.info("PHASE 2: ML Pipeline Validation & Retraining")
    logger.info("="*60)

    # Step 1: Load data
    df = load_data()
    logger.info(f"Raw data shape: {df.shape}")

    # Step 2: Reduce locations
    df = reduce_locations(df)

    # Step 3: Remove outliers
    df = remove_outliers(df)
    logger.info(f"Clean data shape: {df.shape}")

    # Step 4: Prepare features
    X, y = prepare_features(df)
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")

    # Step 5: Multicollinearity check
    check_multicollinearity(X)

    # Step 6: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )

    # Step 7: Train LinearRegression
    lr_clf = LinearRegression(fit_intercept=False)
    lr_clf.fit(X_train, y_train)

    # Step 8: Evaluate
    metrics = evaluate_model(lr_clf, X_test, y_test, "LinearRegression (fit_intercept=False)")

    # Step 9: Cross-validation
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    cv_scores = cross_val_score(LinearRegression(fit_intercept=False), X, y, cv=cv)
    logger.info(f"\nCross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Step 10: Feature importance (top 10 by absolute coefficient)
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': lr_clf.coef_
    })
    coef_df['abs_coef'] = coef_df['coefficient'].abs()
    top_features = coef_df.nlargest(10, 'abs_coef')
    logger.info("\nTop 10 features by coefficient magnitude:")
    for _, row in top_features.iterrows():
        logger.info(f"  {row['feature']}: {row['coefficient']:.4f}")

    # Step 11: Export model
    model_path = ARTIFACTS_DIR / 'banglore_home_prices_model.pickle'
    with open(model_path, 'wb') as f:
        pickle.dump(lr_clf, f)
    logger.info(f"\nModel exported to {model_path}")

    # Step 12: Export columns
    columns = {
        'data_columns': [col.lower() for col in X.columns]
    }
    cols_path = ARTIFACTS_DIR / 'columns.json'
    with open(cols_path, 'w') as f:
        json.dump(columns, f)
    logger.info(f"Columns exported to {cols_path}")

    # Step 13: Export evaluation report
    report = {
        'model': 'LinearRegression',
        'params': {'fit_intercept': False},
        'metrics': metrics,
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'feature_count': X.shape[1],
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'top_features': top_features[['feature', 'coefficient']].to_dict('records')
    }
    report_path = ARTIFACTS_DIR / 'model_evaluation.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report exported to {report_path}")

    return lr_clf, X, y, metrics


if __name__ == '__main__':
    model, X, y, metrics = train_and_export()
    logger.info("\nPhase 2 complete.")
