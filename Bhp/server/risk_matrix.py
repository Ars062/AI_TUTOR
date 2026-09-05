"""
Phase 4: Probability × Impact Risk Matrix
==========================================
Adds P×I assessment layer on top of Phase 3 risk engine.

Each risk category gets:
  - Probability (1-5): How likely is this risk to materialize?
  - Impact (1-5): If it materializes, how bad is it?
  - Risk Score: Probability × Impact (1-25)
  - Risk Level: Configurable thresholds

These thresholds are the project's selected analytical framework,
NOT universal industry standards. They are based on the Bengaluru
residential real estate context and the available data sources.

Probability Scale (1-5):
  1 = Very Unlikely (<20% chance)
  2 = Unlikely (20-40%)
  3 = Possible (40-60%)
  4 = Likely (60-80%)
  5 = Very Likely (>80%)

Impact Scale (1-5):
  1 = Negligible (minimal financial/quality impact)
  2 = Minor (small loss or inconvenience)
  3 = Moderate (noticeable loss, requires attention)
  4 = Major (significant loss, requires action)
  5 = Severe (critical loss, threatens investment viability)

Risk Score = Probability × Impact (range: 1-25)

Risk Level Thresholds (configurable):
  1-4:   Low      (monitor, no immediate action)
  5-9:   Medium   (awareness, plan mitigation)
  10-15: High     (active monitoring, mitigation needed)
  16-25: Critical (immediate action required)
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from pathlib import Path
from risk_engine import RiskEngine, RiskResult, _clamp

ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'


# ─── Configurable Thresholds ────────────────────────────────────────────────────

RISK_LEVEL_THRESHOLDS = {
    "Low": (1, 4),
    "Medium": (5, 9),
    "High": (10, 15),
    "Critical": (16, 25),
}


def _risk_level_from_pi(score: int) -> str:
    """Convert P×I score to risk level using configurable thresholds."""
    for level, (lo, hi) in RISK_LEVEL_THRESHOLDS.items():
        if lo <= score <= hi:
            return level
    return "Critical" if score > 25 else "Low"


# ─── P×I Data Classes ───────────────────────────────────────────────────────────

@dataclass
class PIAssessment:
    """Probability × Impact assessment for a single risk category."""
    risk_category: str
    probability: int          # 1-5
    probability_label: str    # description of probability level
    impact: int               # 1-5
    impact_label: str         # description of impact level
    risk_score: int           # probability × impact (1-25)
    risk_level: str           # Low / Medium / High / Critical
    probability_reasoning: str  # why this probability
    impact_reasoning: str       # why this impact

    def to_dict(self):
        return asdict(self)


# ─── Financial Risk P×I ─────────────────────────────────────────────────────────

def assess_financial_pi(price: float, sqft: float, bhk: int, bath: int,
                        rent_per_sqft: float) -> PIAssessment:
    """
    Financial Risk P×I: Overpricing probability and impact.
    
    Probability mapping (based on deviation from market):
      deviation < -10%: 1 (Very Unlikely overpriced)
      deviation -10% to 5%: 2 (Unlikely)
      deviation 5% to 15%: 3 (Possible)
      deviation 15% to 30%: 4 (Likely)
      deviation > 30%: 5 (Very Likely)
    
    Impact mapping (based on financial loss magnitude):
      deviation < 5%: 1 (Negligible)
      deviation 5-10%: 2 (Minor)
      deviation 10-20%: 3 (Moderate)
      deviation 20-30%: 4 (Major)
      deviation > 30%: 5 (Severe)
    """
    price_per_sqft = (price * 100000) / sqft if sqft > 0 else 0
    market_rate = (rent_per_sqft * 12) / 0.03
    bhk_adj = 1 + (bhk - 2) * 0.05
    bath_adj = 1 + (bath - 2) * 0.02
    adjusted_market = market_rate * bhk_adj * bath_adj
    deviation = ((price_per_sqft - adjusted_market) / adjusted_market * 100) if adjusted_market > 0 else 0

    # Probability
    if deviation > 30:
        prob, prob_label = 5, "Very Likely"
        prob_reason = f"Deviation {deviation:+.1f}% strongly indicates overpricing"
    elif deviation > 15:
        prob, prob_label = 4, "Likely"
        prob_reason = f"Deviation {deviation:+.1f}% suggests overpricing"
    elif deviation > 5:
        prob, prob_label = 3, "Possible"
        prob_reason = f"Deviation {deviation:+.1f}% indicates possible overpricing"
    elif deviation > -10:
        prob, prob_label = 2, "Unlikely"
        prob_reason = f"Deviation {deviation:+.1f}% within normal range"
    else:
        prob, prob_label = 1, "Very Unlikely"
        prob_reason = f"Deviation {deviation:+.1f}% below market — unlikely overpriced"

    # Impact
    abs_dev = abs(deviation)
    if abs_dev > 30:
        imp, imp_label = 5, "Severe"
        imp_reason = f"Overpricing >30% means significant financial loss"
    elif abs_dev > 20:
        imp, imp_label = 4, "Major"
        imp_reason = f"Overpricing 20-30% means notable financial loss"
    elif abs_dev > 10:
        imp, imp_label = 3, "Moderate"
        imp_reason = f"Overpricing 10-20% means moderate financial impact"
    elif abs_dev > 5:
        imp, imp_label = 2, "Minor"
        imp_reason = f"Overpricing 5-10% is a minor concern"
    else:
        imp, imp_label = 1, "Negligible"
        imp_reason = f"Overpricing <5% has minimal financial impact"

    score = prob * imp
    return PIAssessment(
        risk_category="Financial",
        probability=prob,
        probability_label=prob_label,
        impact=imp,
        impact_label=imp_label,
        risk_score=score,
        risk_level=_risk_level_from_pi(score),
        probability_reasoning=prob_reason,
        impact_reasoning=imp_reason
    )


# ─── Market Risk P×I ────────────────────────────────────────────────────────────

def assess_market_pi(price: float, sqft: float,
                     rent_per_sqft: float, appreciation: float) -> PIAssessment:
    """
    Market Risk P×I: Market underperformance probability and impact.
    
    Probability mapping (based on yield and appreciation):
      yield >5% and appreciation >8%: 1 (Very Unlikely)
      yield 3-5% or appreciation 6-8%: 2 (Unlikely)
      yield 2-3% or appreciation 4-6%: 3 (Possible)
      yield <2% or appreciation <4%: 4 (Likely)
      yield <1.5% and appreciation <3%: 5 (Very Likely)
    
    Impact mapping (based on combined market weakness):
      yield >4% and appreciation >6%: 1 (Negligible)
      yield >3% or appreciation >5%: 2 (Minor)
      yield 2-3% or appreciation 4-5%: 3 (Moderate)
      yield <2% or appreciation <4%: 4 (Major)
      yield <1.5% and appreciation <3%: 5 (Severe)
    """
    price_rupees = price * 100000
    annual_rent = rent_per_sqft * sqft * 12
    rental_yield = (annual_rent / price_rupees * 100) if price_rupees > 0 else 0

    # Probability
    if rental_yield < 1.5 and appreciation < 3:
        prob, prob_label = 5, "Very Likely"
        prob_reason = f"Yield {rental_yield:.1f}% and appreciation {appreciation:.1f}% both very low"
    elif rental_yield < 2 or appreciation < 4:
        prob, prob_label = 4, "Likely"
        prob_reason = f"Yield {rental_yield:.1f}% or appreciation {appreciation:.1f}% below threshold"
    elif rental_yield < 3 or appreciation < 6:
        prob, prob_label = 3, "Possible"
        prob_reason = f"Yield {rental_yield:.1f}% or appreciation {appreciation:.1f}% moderate"
    elif rental_yield < 5 and appreciation < 8:
        prob, prob_label = 2, "Unlikely"
        prob_reason = f"Yield {rental_yield:.1f}% and appreciation {appreciation:.1f}% adequate"
    else:
        prob, prob_label = 1, "Very Unlikely"
        prob_reason = f"Yield {rental_yield:.1f}% and appreciation {appreciation:.1f}% strong"

    # Impact
    if rental_yield > 4 and appreciation > 6:
        imp, imp_label = 1, "Negligible"
        imp_reason = "Strong market fundamentals minimize impact"
    elif rental_yield > 3 or appreciation > 5:
        imp, imp_label = 2, "Minor"
        imp_reason = "Adequate market conditions reduce impact"
    elif rental_yield > 2 or appreciation > 4:
        imp, imp_label = 3, "Moderate"
        imp_reason = "Moderate market conditions affect returns"
    elif rental_yield > 1.5 or appreciation > 3:
        imp, imp_label = 4, "Major"
        imp_reason = "Weak market conditions significantly affect returns"
    else:
        imp, imp_label = 5, "Severe"
        imp_reason = "Very weak market threatens investment viability"

    score = prob * imp
    return PIAssessment(
        risk_category="Market",
        probability=prob,
        probability_label=prob_label,
        impact=imp,
        impact_label=imp_label,
        risk_score=score,
        risk_level=_risk_level_from_pi(score),
        probability_reasoning=prob_reason,
        impact_reasoning=imp_reason
    )


# ─── Location Risk P×I ──────────────────────────────────────────────────────────

def assess_location_pi(location_info: dict) -> PIAssessment:
    """
    Location Risk P×I: Location issues probability and impact.
    
    Probability mapping (based on grade, safety, crime):
      Grade A, safety >8, crime Low: 1 (Very Unlikely)
      Grade B, safety >7: 2 (Unlikely)
      Grade B/C, safety 6-7: 3 (Possible)
      Grade C, safety <6, crime Medium: 4 (Likely)
      Grade D, safety <5, crime High: 5 (Very Likely)
    
    Impact mapping (based on quality of life / safety):
      Grade A, safety >8: 1 (Negligible)
      Grade B, safety >7: 2 (Minor)
      Grade B/C, safety 5-7: 3 (Moderate)
      Grade C/D, safety <6: 4 (Major)
      Grade D, safety <5, crime High: 5 (Severe)
    """
    safety = location_info.get("safety", 7.0)
    crime = location_info.get("crime", "Low")
    grade = location_info.get("grade", "B")

    # Probability
    if grade == "A" and safety >= 8 and crime == "Low":
        prob, prob_label = 1, "Very Unlikely"
        prob_reason = f"Grade {grade}, safety {safety}, crime {crime} — premium location"
    elif grade == "B" and safety >= 7:
        prob, prob_label = 2, "Unlikely"
        prob_reason = f"Grade {grade}, safety {safety} — solid location"
    elif grade in ("B", "C") and 5 <= safety < 7:
        prob, prob_label = 3, "Possible"
        prob_reason = f"Grade {grade}, safety {safety} — moderate concerns possible"
    elif grade == "C" and safety < 6:
        prob, prob_label = 4, "Likely"
        prob_reason = f"Grade {grade}, safety {safety}, crime {crime} — issues likely"
    else:
        prob, prob_label = 5, "Very Likely"
        prob_reason = f"Grade {grade}, safety {safety}, crime {crime} — high-risk location"

    # Impact
    if grade == "A" and safety >= 8:
        imp, imp_label = 1, "Negligible"
        imp_reason = "Premium location minimizes impact"
    elif grade == "B" and safety >= 7:
        imp, imp_label = 2, "Minor"
        imp_reason = "Good location reduces impact"
    elif grade in ("B", "C") and safety >= 5:
        imp, imp_label = 3, "Moderate"
        imp_reason = "Average location — moderate impact"
    elif grade == "D" and safety < 5 and crime == "High":
        imp, imp_label = 5, "Severe"
        imp_reason = "Poor location critically affects safety and quality of life"
    elif grade in ("C", "D") and safety < 6:
        imp, imp_label = 4, "Major"
        imp_reason = "Below-average location significantly affects livability"
    else:
        imp, imp_label = 3, "Moderate"
        imp_reason = "Location conditions have moderate impact"

    score = prob * imp
    return PIAssessment(
        risk_category="Location",
        probability=prob,
        probability_label=prob_label,
        impact=imp,
        impact_label=imp_label,
        risk_score=score,
        risk_level=_risk_level_from_pi(score),
        probability_reasoning=prob_reason,
        impact_reasoning=imp_reason
    )


# ─── Environmental Risk P×I ─────────────────────────────────────────────────────

def assess_environmental_pi(location_info: dict) -> PIAssessment:
    """
    Environmental Risk P×I: Flood/natural hazard probability and impact.
    
    Probability mapping (based on flood zone):
      Flood Low: 1 (Very Unlikely)
      Flood Low-Medium: 2 (Unlikely)
      Flood Medium: 3 (Possible)
      Flood Medium-High: 4 (Likely)
      Flood High: 5 (Very Likely)
    
    Impact mapping (based on potential damage):
      Flood Low: 1 (Negligible)
      Flood Low-Medium: 2 (Minor)
      Flood Medium: 3 (Moderate)
      Flood Medium-High: 4 (Major)
      Flood High: 5 (Severe)
    """
    flood = location_info.get("flood", "Medium")

    # Probability
    if flood == "Low":
        prob, prob_label = 1, "Very Unlikely"
        prob_reason = "Low flood zone — minimal probability of flood event"
    elif flood == "Medium":
        prob, prob_label = 3, "Possible"
        prob_reason = "Medium flood zone — flood event possible"
    else:
        prob, prob_label = 5, "Very Likely"
        prob_reason = "High flood zone — flood event likely"

    # Impact
    if flood == "Low":
        imp, imp_label = 1, "Negligible"
        imp_reason = "Low flood risk — minimal potential damage"
    elif flood == "Medium":
        imp, imp_label = 3, "Moderate"
        imp_reason = "Medium flood risk — moderate potential damage"
    else:
        imp, imp_label = 5, "Severe"
        imp_reason = "High flood risk — significant potential damage"

    score = prob * imp
    return PIAssessment(
        risk_category="Environmental",
        probability=prob,
        probability_label=prob_label,
        impact=imp,
        impact_label=imp_label,
        risk_score=score,
        risk_level=_risk_level_from_pi(score),
        probability_reasoning=prob_reason,
        impact_reasoning=imp_reason
    )


# ─── Liquidity Risk P×I ─────────────────────────────────────────────────────────

def assess_liquidity_pi(location_info: dict) -> PIAssessment:
    """
    Liquidity Risk P×I: Illiquidity probability and impact.
    
    Probability mapping (based on liquidity level):
      Liquidity High: 1 (Very Unlikely)
      Liquidity High-Medium: 2 (Unlikely)
      Liquidity Medium: 3 (Possible)
      Liquidity Medium-Low: 4 (Likely)
      Liquidity Low: 5 (Very Likely)
    
    Impact mapping (based on exit difficulty):
      Liquidity High: 1 (Negligible)
      Liquidity High-Medium: 2 (Minor)
      Liquidity Medium: 3 (Moderate)
      Liquidity Medium-Low: 4 (Major)
      Liquidity Low: 5 (Severe)
    """
    liquidity = location_info.get("liquidity", "Medium")

    # Probability
    if liquidity == "High":
        prob, prob_label = 1, "Very Unlikely"
        prob_reason = "High liquidity — easy to sell"
    elif liquidity == "Medium":
        prob, prob_label = 3, "Possible"
        prob_reason = "Medium liquidity — selling may take time"
    else:
        prob, prob_label = 5, "Very Likely"
        prob_reason = "Low liquidity — difficulty selling expected"

    # Impact
    if liquidity == "High":
        imp, imp_label = 1, "Negligible"
        imp_reason = "High liquidity — quick exit possible"
    elif liquidity == "Medium":
        imp, imp_label = 3, "Moderate"
        imp_reason = "Medium liquidity — moderate exit time"
    else:
        imp, imp_label = 5, "Severe"
        imp_reason = "Low liquidity — extended exit time, may need price reduction"

    score = prob * imp
    return PIAssessment(
        risk_category="Liquidity",
        probability=prob,
        probability_label=prob_label,
        impact=imp,
        impact_label=imp_label,
        risk_score=score,
        risk_level=_risk_level_from_pi(score),
        probability_reasoning=prob_reason,
        impact_reasoning=imp_reason
    )


# ─── Rental/Income Risk P×I ─────────────────────────────────────────────────────

def assess_rental_income_pi(price: float, sqft: float, bhk: int, bath: int,
                            rent_per_sqft: float) -> PIAssessment:
    """
    Rental/Income Risk P×I: Rental underperformance probability and impact.
    
    Probability mapping (based on payback and yield):
      payback <15yr and yield >5%: 1 (Very Unlikely)
      payback 15-20yr or yield 3-5%: 2 (Unlikely)
      payback 20-25yr or yield 2-3%: 3 (Possible)
      payback >25yr or yield <2%: 4 (Likely)
      payback >30yr and yield <1.5%: 5 (Very Likely)
    
    Impact mapping (based on cash flow effect):
      yield >5% and payback <15yr: 1 (Negligible)
      yield >4% or payback <18yr: 2 (Minor)
      yield 3-4% or payback 18-22yr: 3 (Moderate)
      yield 2-3% or payback 22-28yr: 4 (Major)
      yield <2% or payback >28yr: 5 (Severe)
    """
    monthly_rent = rent_per_sqft * sqft
    annual_rent = monthly_rent * 12
    price_rupees = price * 100000
    rental_yield = (annual_rent / price_rupees * 100) if price_rupees > 0 else 0
    payback_years = (price_rupees / annual_rent) if annual_rent > 0 else 99

    # Probability
    if payback_years > 30 and rental_yield < 1.5:
        prob, prob_label = 5, "Very Likely"
        prob_reason = f"Payback {payback_years:.0f}yr and yield {rental_yield:.1f}% — very poor rental economics"
    elif payback_years > 25 or rental_yield < 2:
        prob, prob_label = 4, "Likely"
        prob_reason = f"Payback {payback_years:.0f}yr or yield {rental_yield:.1f}% — poor rental economics"
    elif payback_years > 20 or rental_yield < 3:
        prob, prob_label = 3, "Possible"
        prob_reason = f"Payback {payback_years:.0f}yr or yield {rental_yield:.1f}% — moderate concern"
    elif payback_years > 15 and rental_yield < 5:
        prob, prob_label = 2, "Unlikely"
        prob_reason = f"Payback {payback_years:.0f}yr and yield {rental_yield:.1f}% — adequate"
    else:
        prob, prob_label = 1, "Very Unlikely"
        prob_reason = f"Payback {payback_years:.0f}yr and yield {rental_yield:.1f}% — strong rental economics"

    # Impact
    if rental_yield > 5 and payback_years < 15:
        imp, imp_label = 1, "Negligible"
        imp_reason = "Strong rental economics minimize impact"
    elif rental_yield > 4 or payback_years < 18:
        imp, imp_label = 2, "Minor"
        imp_reason = "Adequate rental income reduces impact"
    elif rental_yield > 3 or payback_years < 22:
        imp, imp_label = 3, "Moderate"
        imp_reason = "Moderate rental income affects returns"
    elif rental_yield > 2 or payback_years < 28:
        imp, imp_label = 4, "Major"
        imp_reason = "Weak rental income significantly affects returns"
    else:
        imp, imp_label = 5, "Severe"
        imp_reason = "Very weak rental income threatens holding viability"

    score = prob * imp
    return PIAssessment(
        risk_category="Rental/Income",
        probability=prob,
        probability_label=prob_label,
        impact=imp,
        impact_label=imp_label,
        risk_score=score,
        risk_level=_risk_level_from_pi(score),
        probability_reasoning=prob_reason,
        impact_reasoning=imp_reason
    )


# ─── Risk Matrix ────────────────────────────────────────────────────────────────

@dataclass
class RiskMatrix:
    """Full P×I risk matrix for a property."""
    assessments: List[PIAssessment]
    overall_probability: float
    overall_impact: float
    overall_risk_score: float
    overall_risk_level: str
    matrix: Dict[str, Dict]  # category -> {probability, impact, score}

    def to_dict(self):
        return asdict(self)


def build_risk_matrix(location: str, sqft: float, bhk: int, bath: int,
                      price: float) -> RiskMatrix:
    """
    Build complete P×I risk matrix for a property.
    
    Overall scores are weighted averages:
      Overall Probability = weighted avg of category probabilities
      Overall Impact = weighted avg of category impacts
      Overall Risk Score = Overall Probability × Overall Impact
    
    Weights (same as Phase 3):
      Financial: 0.25, Market: 0.15, Location: 0.20
      Environmental: 0.10, Liquidity: 0.15, Rental/Income: 0.15
    """
    engine = RiskEngine()
    info = engine._get_location_info(location)
    rent_per_sqft = info.get("rent_per_sqft", 20)
    appreciation = info.get("appreciation", 6.0)

    assessments = [
        assess_financial_pi(price, sqft, bhk, bath, rent_per_sqft),
        assess_market_pi(price, sqft, rent_per_sqft, appreciation),
        assess_location_pi(info),
        assess_environmental_pi(info),
        assess_liquidity_pi(info),
        assess_rental_income_pi(price, sqft, bhk, bath, rent_per_sqft),
    ]

    weights = {
        "Financial": 0.25,
        "Market": 0.15,
        "Location": 0.20,
        "Environmental": 0.10,
        "Liquidity": 0.15,
        "Rental/Income": 0.15,
    }

    overall_prob = sum(a.probability * weights[a.risk_category] for a in assessments)
    overall_imp = sum(a.impact * weights[a.risk_category] for a in assessments)
    overall_score = round(overall_prob * overall_imp, 1)

    matrix = {}
    for a in assessments:
        matrix[a.risk_category] = {
            "probability": a.probability,
            "impact": a.impact,
            "score": a.risk_score,
            "level": a.risk_level,
        }

    return RiskMatrix(
        assessments=assessments,
        overall_probability=round(overall_prob, 2),
        overall_impact=round(overall_imp, 2),
        overall_risk_score=overall_score,
        overall_risk_level=_risk_level_from_pi(int(overall_score)),
        matrix=matrix
    )


# ─── Integration with RiskEngine ────────────────────────────────────────────────

def assess_property_with_pi(location: str, sqft: float, bhk: int, bath: int,
                            price: float) -> dict:
    """
    Full property assessment including P×I risk matrix.
    Returns combined Phase 3 + Phase 4 output.
    """
    engine = RiskEngine()
    phase3 = engine.assess_property(location, sqft, bhk, bath, price)
    phase4 = build_risk_matrix(location, sqft, bhk, bath, price)

    return {
        **phase3,
        "risk_matrix": phase4.to_dict(),
    }


# ─── Backward-Compatible API ────────────────────────────────────────────────────

def get_comprehensive_risk_report_with_pi(location, sqft, bhk, bath, price, ml_price=None):
    """
    Backward-compatible risk report with P×I matrix added.
    Preserves all existing fields from Phase 3.
    """
    engine = RiskEngine()
    assessment = engine.assess_property(location, sqft, bhk, bath, price)
    pi_matrix = build_risk_matrix(location, sqft, bhk, bath, price)

    # Build overpricing analysis from valuation
    val = assessment["valuation"]
    if val["status"] == "Overpriced":
        message = f"Property is {abs(val['deviation_pct']):.1f}% above market rate. Consider negotiating."
    elif val["status"] == "Slightly High":
        message = f"Property is {abs(val['deviation_pct']):.1f}% above market rate. Price is on the higher side."
    elif val["status"] == "Underpriced":
        message = f"Property is {abs(val['deviation_pct']):.1f}% below market rate. Good deal potential."
    elif val["status"] == "Slightly Low":
        message = f"Property is {abs(val['deviation_pct']):.1f}% below market rate. Slightly favorable."
    else:
        message = f"Property is priced within {abs(val['deviation_pct']):.1f}% of market rate. Fair deal."

    overpricing = {
        "status": val["status"],
        "deviation": f"{val['deviation_pct']:+.1f}%",
        "market_rate": val["market_rate"],
        "your_rate": val["price_per_sqft"],
        "message": message
    }

    # Build location rating
    info = engine._get_location_info(location)
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

    location_rating = {
        "grade": grade,
        "description": grade_desc.get(grade, "Location assessment unavailable"),
        "breakdown": breakdown
    }

    # ROI estimate
    rent_per_sqft = info.get("rent_per_sqft", 20)
    monthly_rent = rent_per_sqft * sqft
    annual_rent = monthly_rent * 12
    price_in_rupees = price * 100000
    rental_yield = (annual_rent / price_in_rupees * 100) if price_in_rupees > 0 else 0
    appreciation = info.get("appreciation", 6.0)
    total_roi = rental_yield + appreciation
    payback_years = (price_in_rupees / annual_rent) if annual_rent > 0 else 0

    roi = {
        "monthly_rent": round(monthly_rent, 0),
        "annual_rent": round(annual_rent, 0),
        "rental_yield": f"{rental_yield:.1f}%",
        "expected_appreciation": f"{appreciation:.1f}%",
        "total_roi": f"{total_roi:.1f}%",
        "payback_years": f"{payback_years:.1f} years"
    }

    overall = {
        "score": assessment["overall_risk_score"],
        "level": assessment["overall_risk_level"],
        "attractiveness": assessment["attractiveness"],
        "category_scores": {r["risk_category"]: r["risk_score"] for r in assessment["risks"]},
    }

    return {
        "estimated_price": ml_price or 0,
        "investment_risk": {
            "score": assessment["overall_risk_score"],
            "level": assessment["overall_risk_level"],
            "reasons": [d for r in assessment["risks"] for d in r["drivers"]],
        },
        "overpricing_analysis": overpricing,
        "location_rating": location_rating,
        "roi_estimate": roi,
        "risk_categories": assessment["risks"],
        "attractiveness": assessment["attractiveness"],
        "overall_risk": overall,
        "risk_matrix": pi_matrix.to_dict(),
    }


if __name__ == '__main__':
    result = build_risk_matrix('Koramangala', 1000, 2, 2, 150)
    print(json.dumps(result.to_dict(), indent=2))
