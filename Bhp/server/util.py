import pickle
import json
import numpy as np
from risk_engine import RiskEngine, get_comprehensive_risk_report as _engine_report
from risk_matrix import get_comprehensive_risk_report_with_pi as _engine_report_pi

__locations = None
__data_columns = None
__model = None
__location_risk_data = None
__risk_engine = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


def _get_location_info(location):
    if __location_risk_data is None:
        return None
    loc = location.lower().strip()
    risk_data = __location_risk_data.get("location_risk", {})
    defaults = __location_risk_data.get("defaults", {})
    return risk_data.get(loc, defaults)


def get_investment_risk_score(location, sqft, bhk, bath, price):
    info = _get_location_info(location)
    if info is None:
        return {"score": 5, "level": "Medium", "message": "Insufficient data for risk assessment"}

    score = 5.0
    reasons = []

    price_per_sqft = (price * 100000) / sqft if sqft > 0 else 0
    rent_per_sqft = info.get("rent_per_sqft", 20)
    annual_rent_per_sqft = rent_per_sqft * 12
    expected_price_per_sqft = annual_rent_per_sqft / 0.03
    if price_per_sqft > expected_price_per_sqft * 1.3:
        score += 2
        reasons.append("Property is significantly overpriced for the area")
    elif price_per_sqft > expected_price_per_sqft * 1.15:
        score += 1
        reasons.append("Property is slightly above market rate")

    safety = info.get("safety", 7.0)
    if safety < 5.0:
        score += 2
        reasons.append("Low safety index in this location")
    elif safety < 6.5:
        score += 1
        reasons.append("Moderate safety concerns")

    flood = info.get("flood", "Medium")
    if flood == "High":
        score += 1.5
        reasons.append("High flood risk zone")
    elif flood == "Medium":
        score += 0.5
        reasons.append("Moderate flood risk")

    liquidity = info.get("liquidity", "Medium")
    if liquidity == "Low":
        score += 1
        reasons.append("Low market liquidity - hard to resell")

    crime = info.get("crime", "Low")
    if crime == "High":
        score += 1.5
        reasons.append("High crime rate area")
    elif crime == "Medium":
        score += 0.5
        reasons.append("Moderate crime rate")

    bath_bhk_ratio = bath / bhk if bhk > 0 else 1
    if bath_bhk_ratio > 1.5:
        score += 0.5
        reasons.append("Unusually high bath-to-BHK ratio")

    if sqft / bhk < 300 if bhk > 0 else False:
        score += 1
        reasons.append("Very small carpet area per bedroom")

    score = max(1, min(10, round(score, 1)))
    if score <= 3:
        level = "Low"
    elif score <= 5:
        level = "Medium"
    elif score <= 7:
        level = "High"
    else:
        level = "Very High"

    if not reasons:
        reasons.append("Property parameters are within normal range for this location")

    return {
        "score": score,
        "level": level,
        "reasons": reasons
    }


def get_overpricing_status(location, sqft, bhk, bath, price):
    info = _get_location_info(location)
    if info is None:
        return {"status": "Unknown", "deviation": "0%", "message": "Insufficient data for comparison"}

    price_per_sqft = (price * 100000) / sqft if sqft > 0 else 0
    rent_per_sqft = info.get("rent_per_sqft", 20)
    annual_rent_per_sqft = rent_per_sqft * 12
    estimated_price_per_sqft = annual_rent_per_sqft / 0.03

    bhk_adjustment = 1 + (bhk - 2) * 0.05
    bath_adjustment = 1 + (bath - 2) * 0.02
    adjusted_expected = estimated_price_per_sqft * bhk_adjustment * bath_adjustment

    deviation_pct = ((price_per_sqft - adjusted_expected) / adjusted_expected) * 100 if adjusted_expected > 0 else 0

    if deviation_pct > 20:
        status = "Overpriced"
        message = f"Property is {abs(deviation_pct):.1f}% above market rate. Consider negotiating."
    elif deviation_pct > 10:
        status = "Slightly High"
        message = f"Property is {abs(deviation_pct):.1f}% above market rate. Price is on the higher side."
    elif deviation_pct < -20:
        status = "Underpriced"
        message = f"Property is {abs(deviation_pct):.1f}% below market rate. Good deal potential."
    elif deviation_pct < -10:
        status = "Slightly Low"
        message = f"Property is {abs(deviation_pct):.1f}% below market rate. Slightly favorable."
    else:
        status = "Fair"
        message = f"Property is priced within {abs(deviation_pct):.1f}% of market rate. Fair deal."

    return {
        "status": status,
        "deviation": f"{deviation_pct:+.1f}%",
        "market_rate": round(adjusted_expected, 0),
        "your_rate": round(price_per_sqft, 0),
        "message": message
    }


def get_location_risk_rating(location):
    info = _get_location_info(location)
    if info is None:
        return {"grade": "N/A", "breakdown": {}, "message": "Location data not available"}

    breakdown = {
        "safety_index": {
            "score": info.get("safety", 7.0),
            "rating": "Excellent" if info.get("safety", 7) >= 8.5 else
                     "Good" if info.get("safety", 7) >= 7.0 else
                     "Average" if info.get("safety", 7) >= 5.5 else "Poor"
        },
        "flood_risk": {
            "level": info.get("flood", "Medium"),
            "impact": "Minimal" if info.get("flood") == "Low" else
                     "Moderate" if info.get("flood") == "Medium" else "Significant"
        },
        "infrastructure_score": {
            "score": info.get("infra", 7.0),
            "rating": "Excellent" if info.get("infra", 7) >= 8.5 else
                     "Good" if info.get("infra", 7) >= 7.0 else
                     "Average" if info.get("infra", 7) >= 5.5 else "Poor"
        },
        "market_liquidity": {
            "level": info.get("liquidity", "Medium"),
            "impact": "Easy to sell" if info.get("liquidity") == "High" else
                     "Moderate effort" if info.get("liquidity") == "Medium" else "Difficult to sell"
        },
        "crime_rate": {
            "level": info.get("crime", "Low"),
            "impact": "Safe area" if info.get("crime") == "Low" else
                     "Some caution needed" if info.get("crime") == "Medium" else "High alert area"
        }
    }

    grade = info.get("grade", "B")
    grade_desc = {
        "A": "Premium location with excellent infrastructure and high demand",
        "B": "Good location with solid amenities and stable market",
        "C": "Developing location with moderate infrastructure",
        "D": "Emerging or high-risk location - invest with caution"
    }

    return {
        "grade": grade,
        "description": grade_desc.get(grade, "Location assessment unavailable"),
        "breakdown": breakdown
    }


def get_roi_estimate(location, sqft, bhk, price):
    info = _get_location_info(location)
    if info is None:
        return {"monthly_rent": 0, "rental_yield": "0%", "appreciation": "0%", "total_roi": "0%", "payback_years": "N/A"}

    rent_per_sqft = info.get("rent_per_sqft", 20)
    monthly_rent = rent_per_sqft * sqft
    annual_rent = monthly_rent * 12
    price_in_rupees = price * 100000
    rental_yield = (annual_rent / price_in_rupees * 100) if price_in_rupees > 0 else 0

    appreciation = info.get("appreciation", 6.0)
    total_roi = rental_yield + appreciation

    payback_years = (price_in_rupees / annual_rent) if annual_rent > 0 else 0

    return {
        "monthly_rent": round(monthly_rent, 0),
        "annual_rent": round(annual_rent, 0),
        "rental_yield": f"{rental_yield:.1f}%",
        "expected_appreciation": f"{appreciation:.1f}%",
        "total_roi": f"{total_roi:.1f}%",
        "payback_years": f"{payback_years:.1f} years"
    }


def get_comprehensive_risk_report(location, sqft, bhk, bath, price):
    estimated_price = get_estimated_price(location, sqft, bhk, bath)

    if __risk_engine is not None:
        return _engine_report_pi(location, sqft, bhk, bath, price, ml_price=estimated_price)

    return {
        "estimated_price": estimated_price,
        "investment_risk": get_investment_risk_score(location, sqft, bhk, bath, price),
        "overpricing_analysis": get_overpricing_status(location, sqft, bhk, bath, price),
        "location_rating": get_location_risk_rating(location),
        "roi_estimate": get_roi_estimate(location, sqft, bhk, price)
    }


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model
    if __model is None:
        with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)

    global __location_risk_data
    if __location_risk_data is None:
        try:
            with open('./artifacts/location_risk_data.json', 'r') as f:
                __location_risk_data = json.load(f)
            print("Location risk data loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load risk data: {e}")
            __location_risk_data = {"location_risk": {}, "defaults": {
                "safety": 7.0, "flood": "Medium", "infra": 6.5,
                "liquidity": "Medium", "crime": "Low", "grade": "B",
                "rent_per_sqft": 20, "appreciation": 6.0
            }}

    global __risk_engine
    if __risk_engine is None:
        try:
            __risk_engine = RiskEngine()
            print("Risk engine initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize risk engine: {e}")
            __risk_engine = None

    print("loading saved artifacts...done")


def get_location_names():
    return __locations


def get_data_columns():
    return __data_columns


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names()[:5])
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_comprehensive_risk_report('1st Phase JP Nagar', 1000, 2, 2, 83.87))
    print(get_comprehensive_risk_report('Koramangala', 1000, 2, 2, 150))
    print(get_comprehensive_risk_report('Electronic City', 1000, 2, 2, 50))
