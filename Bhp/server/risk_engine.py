"""
Phase 3: Structured Multi-Category Risk Engine
================================================
Modular, deterministic risk assessment engine that separates:
  1. Property Valuation (price estimation)
  2. Risk Assessment (multi-category risk analysis)
  3. Investment Attractiveness (combined scoring)

Risk Categories (data-supported):
  1. Financial Risk — overpricing relative to market
  2. Market Risk — market conditions, price-to-rent ratio
  3. Location Risk — safety, crime, infrastructure, grade
  4. Environmental Risk — flood, natural hazards
  5. Liquidity Risk — ease of resale
  6. Rental/Income Risk — rental yield, payback period

All thresholds and weights documented below.
No arbitrary scores — every value traceable to source data.

Threshold Documentation:
─────────────────────────
Financial Risk Thresholds:
  - Overpricing deviation >30%: score +2.5 (Very High)
  - Overpricing deviation >15%: score +1.5 (High)
  - Overpricing deviation >5%: score +0.5 (Medium)
  - Overpricing deviation <-20%: score -1.0 (Low, good deal)
  - Base score: 5.0

Market Risk Thresholds:
  - Rental yield <2%: score +2.0 (Very High risk)
  - Rental yield <3%: score +1.0 (High risk)
  - Rental yield >5%: score -1.0 (Low risk)
  - Price-to-rent ratio >25: score +1.5 (overvalued)
  - Price-to-rent ratio >20: score +0.5
  - Base score: 5.0

Location Risk Thresholds:
  - Safety <5.0: score +2.0
  - Safety <6.5: score +1.0
  - Safety >=8.0: score -1.0
  - Crime High: score +1.5
  - Crime Medium: score +0.5
  - Grade D: score +2.0
  - Grade C: score +1.0
  - Grade A: score -1.0
  - Base score: 5.0

Environmental Risk Thresholds:
  - Flood High: score +2.0
  - Flood Medium: score +0.5
  - Flood Low: score -0.5
  - Base score: 5.0

Liquidity Risk Thresholds:
  - Liquidity Low: score +2.0
  - Liquidity Medium: score +0.5
  - Liquidity High: score -1.0
  - Base score: 5.0

Rental/Income Risk Thresholds:
  - Payback >25 years: score +2.0
  - Payback >20 years: score +1.0
  - Payback <15 years: score -1.0
  - Bath/BHK ratio >1.5: score +0.5
  - sqft/BHK <300: score +1.0
  - Base score: 5.0
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'


# ─── Data Classes ────────────────────────────────────────────────────────────────

@dataclass
class RiskResult:
    """Structured output for a single risk category."""
    risk_category: str
    risk_score: float        # 1.0 - 10.0
    risk_level: str          # Low / Medium / High / Very High
    drivers: List[str]       # what drives this risk
    evidence: List[str]      # supporting data points
    mitigation: List[str]    # suggested mitigation actions

    def to_dict(self):
        return asdict(self)


@dataclass
class InvestmentAttractiveness:
    """Combined investment attractiveness score (separate from risk)."""
    score: float             # 1.0 - 10.0
    level: str               # Poor / Fair / Good / Excellent
    factors: List[str]       # contributing factors
    recommendation: str      # Buy / Hold / Avoid

    def to_dict(self):
        return asdict(self)


@dataclass
class PropertyValuation:
    """Property valuation result (separate from risk)."""
    estimated_price: float   # Lakhs
    price_per_sqft: float
    market_rate: float       # estimated market rate per sqft
    deviation_pct: float     # % deviation from market
    status: str              # Overpriced / Fair / Underpriced

    def to_dict(self):
        return asdict(self)


# ─── Risk Level Mapping ─────────────────────────────────────────────────────────

def _risk_level(score: float) -> str:
    """Convert numeric risk score to level string.
    
    Thresholds (documented):
      <=3.0: Low
      <=5.0: Medium
      <=7.0: High
      >7.0: Very High
    """
    if score <= 3.0:
        return "Low"
    elif score <= 5.0:
        return "Medium"
    elif score <= 7.0:
        return "High"
    else:
        return "Very High"


def _clamp(score: float, lo: float = 1.0, hi: float = 10.0) -> float:
    """Clamp score to [lo, hi]."""
    return max(lo, min(hi, round(score, 1)))


# ─── Risk Category: Financial Risk ──────────────────────────────────────────────

def assess_financial_risk(price: float, sqft: float, bhk: int, bath: int,
                          rent_per_sqft: float) -> RiskResult:
    """
    Financial Risk: Overpricing relative to market rate.
    
    Source data:
      - price: user-provided (Lakhs)
      - sqft: user-provided
      - rent_per_sqft: from location_risk_data.json
    
    Market rate formula:
      market_rate = (rent_per_sqft * 12) / 0.03
      This assumes 3% annual rental yield (cap rate).
    
    BHK/bath adjustments (existing util.py logic):
      adjusted = market_rate * (1 + (bhk-2) * 0.05) * (1 + (bath-2) * 0.02)
    
    Thresholds:
      deviation > 30%: Very High (+2.5)
      deviation > 15%: High (+1.5)
      deviation > 5%: Medium (+0.5)
      deviation < -20%: Low (-1.0, good deal)
    """
    score = 5.0
    drivers = []
    evidence = []
    mitigation = []

    price_per_sqft = (price * 100000) / sqft if sqft > 0 else 0
    market_rate = (rent_per_sqft * 12) / 0.03

    bhk_adj = 1 + (bhk - 2) * 0.05
    bath_adj = 1 + (bath - 2) * 0.02
    adjusted_market = market_rate * bhk_adj * bath_adj

    deviation = ((price_per_sqft - adjusted_market) / adjusted_market * 100) if adjusted_market > 0 else 0

    evidence.append(f"Price per sqft: Rs.{price_per_sqft:.0f}")
    evidence.append(f"Market rate (adjusted): Rs.{adjusted_market:.0f}")
    evidence.append(f"Deviation: {deviation:+.1f}%")

    if deviation > 30:
        score += 2.5
        drivers.append("Significant overpricing (>30% above market)")
        mitigation.append("Negotiate price down or look for alternatives")
        mitigation.append("Verify property condition justifies premium")
    elif deviation > 15:
        score += 1.5
        drivers.append("Above-market pricing (>15%)")
        mitigation.append("Negotiate closer to market rate")
    elif deviation > 5:
        score += 0.5
        drivers.append("Slightly above market rate")
    elif deviation < -20:
        score -= 1.0
        drivers.append("Potential undervaluation (>20% below market)")
        evidence.append("This may indicate a good deal or hidden issues")
    elif deviation < -10:
        drivers.append("Slightly below market rate")

    if not drivers:
        drivers.append("Price is within market range")

    return RiskResult(
        risk_category="Financial",
        risk_score=_clamp(score),
        risk_level=_risk_level(_clamp(score)),
        drivers=drivers,
        evidence=evidence,
        mitigation=mitigation
    )


# ─── Risk Category: Market Risk ─────────────────────────────────────────────────

def assess_market_risk(price: float, sqft: float,
                       rent_per_sqft: float, appreciation: float) -> RiskResult:
    """
    Market Risk: Market conditions, rental yield, price-to-rent ratio.
    
    Source data:
      - price: user-provided (Lakhs)
      - sqft: user-provided
      - rent_per_sqft: from location_risk_data.json
      - appreciation: from location_risk_data.json
    
    Formulas:
      annual_rent = rent_per_sqft * sqft * 12
      rental_yield = (annual_rent / (price * 100000)) * 100
      price_to_rent = (price * 100000) / annual_rent if annual_rent > 0 else inf
    
    Thresholds:
      Rental yield <2%: Very High risk (+2.0)
      Rental yield <3%: High risk (+1.0)
      Rental yield >5%: Low risk (-1.0)
      Price-to-rent >25: Overvalued (+1.5)
      Price-to-rent >20: Slightly overvalued (+0.5)
      Appreciation <4%: High risk (+1.0)
      Appreciation >8%: Low risk (-0.5)
    """
    score = 5.0
    drivers = []
    evidence = []
    mitigation = []

    price_rupees = price * 100000
    annual_rent = rent_per_sqft * sqft * 12
    rental_yield = (annual_rent / price_rupees * 100) if price_rupees > 0 else 0
    price_to_rent = (price_rupees / annual_rent) if annual_rent > 0 else float('inf')

    evidence.append(f"Annual rent: Rs.{annual_rent:,.0f}")
    evidence.append(f"Rental yield: {rental_yield:.2f}%")
    evidence.append(f"Price-to-rent ratio: {price_to_rent:.1f}")
    evidence.append(f"Expected appreciation: {appreciation:.1f}%")

    # Rental yield thresholds
    if rental_yield < 2:
        score += 2.0
        drivers.append("Very low rental yield (<2%)")
        mitigation.append("Property may not generate adequate rental income")
    elif rental_yield < 3:
        score += 1.0
        drivers.append("Below-average rental yield (<3%)")
    elif rental_yield > 5:
        score -= 1.0
        drivers.append("Strong rental yield (>5%)")

    # Price-to-rent thresholds
    if price_to_rent > 25:
        score += 1.5
        drivers.append("High price-to-rent ratio (>25) — overvalued market")
        mitigation.append("Consider waiting for market correction")
    elif price_to_rent > 20:
        score += 0.5
        drivers.append("Elevated price-to-rent ratio (>20)")

    # Appreciation thresholds
    if appreciation < 4:
        score += 1.0
        drivers.append("Low expected appreciation (<4%)")
    elif appreciation > 8:
        score -= 0.5
        drivers.append("Strong expected appreciation (>8%)")

    if not drivers:
        drivers.append("Market conditions are within normal range")

    return RiskResult(
        risk_category="Market",
        risk_score=_clamp(score),
        risk_level=_risk_level(_clamp(score)),
        drivers=drivers,
        evidence=evidence,
        mitigation=mitigation
    )


# ─── Risk Category: Location Risk ───────────────────────────────────────────────

def assess_location_risk(location_info: dict) -> RiskResult:
    """
    Location Risk: Safety, crime, infrastructure, grade.
    
    Source data: all from location_risk_data.json (hand-curated, synthetic)
    
    Thresholds:
      Safety <5.0: Very High risk (+2.0)
      Safety <6.5: High risk (+1.0)
      Safety >=8.0: Low risk (-1.0)
      Crime High: Very High risk (+1.5)
      Crime Medium: Medium risk (+0.5)
      Grade D: Very High risk (+2.0)
      Grade C: High risk (+1.0)
      Grade A: Low risk (-1.0)
      Infra <5.0: High risk (+1.0)
    """
    score = 5.0
    drivers = []
    evidence = []
    mitigation = []

    safety = location_info.get("safety", 7.0)
    crime = location_info.get("crime", "Low")
    infra = location_info.get("infra", 6.5)
    grade = location_info.get("grade", "B")

    evidence.append(f"Safety index: {safety}/10")
    evidence.append(f"Crime rate: {crime}")
    evidence.append(f"Infrastructure score: {infra}/10")
    evidence.append(f"Location grade: {grade}")

    # Safety thresholds
    if safety < 5.0:
        score += 2.0
        drivers.append("Low safety index (<5.0)")
        mitigation.append("Verify security arrangements in the property")
    elif safety < 6.5:
        score += 1.0
        drivers.append("Moderate safety concerns")
    elif safety >= 8.0:
        score -= 1.0
        drivers.append("Excellent safety index (>=8.0)")

    # Crime thresholds
    if crime == "High":
        score += 1.5
        drivers.append("High crime rate area")
        mitigation.append("Research specific crime types and frequency")
    elif crime == "Medium":
        score += 0.5
        drivers.append("Moderate crime rate")

    # Grade thresholds
    if grade == "D":
        score += 2.0
        drivers.append("Location grade D — emerging/high-risk")
        mitigation.append("Invest with caution; verify development plans")
    elif grade == "C":
        score += 1.0
        drivers.append("Location grade C — developing area")
    elif grade == "A":
        score -= 1.0
        drivers.append("Premium location grade A")

    # Infrastructure thresholds
    if infra < 5.0:
        score += 1.0
        drivers.append("Below-average infrastructure (<5.0)")
        mitigation.append("Check connectivity, schools, hospitals nearby")

    if not drivers:
        drivers.append("Location parameters are within normal range")

    return RiskResult(
        risk_category="Location",
        risk_score=_clamp(score),
        risk_level=_risk_level(_clamp(score)),
        drivers=drivers,
        evidence=evidence,
        mitigation=mitigation
    )


# ─── Risk Category: Environmental Risk ──────────────────────────────────────────

def assess_environmental_risk(location_info: dict) -> RiskResult:
    """
    Environmental Risk: Flood, natural hazards.
    
    Source data: from location_risk_data.json (hand-curated, synthetic)
    
    Thresholds:
      Flood High: Very High risk (+2.0)
      Flood Medium: Medium risk (+0.5)
      Flood Low: Low risk (-0.5)
    """
    score = 5.0
    drivers = []
    evidence = []
    mitigation = []

    flood = location_info.get("flood", "Medium")

    evidence.append(f"Flood risk: {flood}")

    if flood == "High":
        score += 2.0
        drivers.append("High flood risk zone")
        mitigation.append("Check if property is in flood-prone area")
        mitigation.append("Verify flood insurance availability")
        mitigation.append("Check historical flood data for the area")
    elif flood == "Medium":
        score += 0.5
        drivers.append("Moderate flood risk")
        mitigation.append("Verify drainage infrastructure")
    elif flood == "Low":
        score -= 0.5
        drivers.append("Low flood risk")

    if not drivers:
        drivers.append("No significant environmental risks identified")

    return RiskResult(
        risk_category="Environmental",
        risk_score=_clamp(score),
        risk_level=_risk_level(_clamp(score)),
        drivers=drivers,
        evidence=evidence,
        mitigation=mitigation
    )


# ─── Risk Category: Liquidity Risk ──────────────────────────────────────────────

def assess_liquidity_risk(location_info: dict) -> RiskResult:
    """
    Liquidity Risk: Ease of resale.
    
    Source data: from location_risk_data.json (hand-curated, synthetic)
    
    Thresholds:
      Liquidity Low: High risk (+2.0)
      Liquidity Medium: Medium risk (+0.5)
      Liquidity High: Low risk (-1.0)
    """
    score = 5.0
    drivers = []
    evidence = []
    mitigation = []

    liquidity = location_info.get("liquidity", "Medium")

    evidence.append(f"Market liquidity: {liquidity}")

    if liquidity == "Low":
        score += 2.0
        drivers.append("Low market liquidity — hard to resell")
        mitigation.append("Be prepared for longer selling time")
        mitigation.append("Consider properties in higher-liquidity areas")
    elif liquidity == "Medium":
        score += 0.5
        drivers.append("Moderate market liquidity")
    elif liquidity == "High":
        score -= 1.0
        drivers.append("High market liquidity — easy to resell")

    if not drivers:
        drivers.append("Liquidity conditions are normal")

    return RiskResult(
        risk_category="Liquidity",
        risk_score=_clamp(score),
        risk_level=_risk_level(_clamp(score)),
        drivers=drivers,
        evidence=evidence,
        mitigation=mitigation
    )


# ─── Risk Category: Rental/Income Risk ──────────────────────────────────────────

def assess_rental_income_risk(price: float, sqft: float, bhk: int, bath: int,
                               rent_per_sqft: float) -> RiskResult:
    """
    Rental/Income Risk: Rental yield, payback period, space efficiency.
    
    Source data:
      - price, sqft, bhk, bath: user-provided
      - rent_per_sqft: from location_risk_data.json
    
    Formulas:
      monthly_rent = rent_per_sqft * sqft
      annual_rent = monthly_rent * 12
      payback_years = (price * 100000) / annual_rent
      bath_bhk_ratio = bath / bhk
      sqft_per_bhk = sqft / bhk
    
    Thresholds:
      Payback >25 years: Very High risk (+2.0)
      Payback >20 years: High risk (+1.0)
      Payback <15 years: Low risk (-1.0)
      Bath/BHK >1.5: Medium risk (+0.5)
      sqft/BHK <300: High risk (+1.0)
    """
    score = 5.0
    drivers = []
    evidence = []
    mitigation = []

    monthly_rent = rent_per_sqft * sqft
    annual_rent = monthly_rent * 12
    price_rupees = price * 100000
    payback_years = (price_rupees / annual_rent) if annual_rent > 0 else float('inf')
    bath_bhk_ratio = bath / bhk if bhk > 0 else 1
    sqft_per_bhk = sqft / bhk if bhk > 0 else sqft

    evidence.append(f"Monthly rent: Rs.{monthly_rent:,.0f}")
    evidence.append(f"Payback period: {payback_years:.1f} years")
    evidence.append(f"Bath/BHK ratio: {bath_bhk_ratio:.1f}")
    evidence.append(f"Sqft per BHK: {sqft_per_bhk:.0f}")

    # Payback period thresholds
    if payback_years > 25:
        score += 2.0
        drivers.append(f"Very long payback period ({payback_years:.0f} years)")
        mitigation.append("Consider properties with better rental potential")
    elif payback_years > 20:
        score += 1.0
        drivers.append(f"Long payback period ({payback_years:.0f} years)")
    elif payback_years < 15:
        score -= 1.0
        drivers.append(f"Good payback period ({payback_years:.0f} years)")

    # Space efficiency thresholds
    if bath_bhk_ratio > 1.5:
        score += 0.5
        drivers.append("Unusually high bath-to-BHK ratio")
    if sqft_per_bhk < 300:
        score += 1.0
        drivers.append("Very small carpet area per bedroom (<300 sqft)")
        mitigation.append("Verify actual usable area")

    if not drivers:
        drivers.append("Rental income parameters are within normal range")

    return RiskResult(
        risk_category="Rental/Income",
        risk_score=_clamp(score),
        risk_level=_risk_level(_clamp(score)),
        drivers=drivers,
        evidence=evidence,
        mitigation=mitigation
    )


# ─── Investment Attractiveness ───────────────────────────────────────────────────

def assess_investment_attractiveness(risk_results: List[RiskResult],
                                     valuation: PropertyValuation,
                                     location_grade: str) -> InvestmentAttractiveness:
    """
    Combined investment attractiveness score (SEPARATE from risk).
    
    Formula:
      risk_avg = mean of all category risk_scores
      attractiveness = 10 - risk_avg (inverted: low risk = high attractiveness)
      Bonus: Grade A +0.5, Grade D -0.5
      Bonus: Underpriced +0.5, Overpriced -0.5
    
    Thresholds:
      >=8.0: Excellent
      >=6.0: Good
      >=4.0: Fair
      <4.0: Poor
    
    Recommendation:
      >=7.0: Buy
      >=5.0: Hold (proceed with caution)
      <5.0: Avoid
    """
    scores = [r.risk_score for r in risk_results]
    risk_avg = sum(scores) / len(scores) if scores else 5.0

    attractiveness = 10.0 - risk_avg

    factors = []

    # Grade bonus
    if location_grade == "A":
        attractiveness += 0.5
        factors.append("Premium location grade A (+0.5)")
    elif location_grade == "D":
        attractiveness -= 0.5
        factors.append("High-risk location grade D (-0.5)")

    # Valuation bonus
    if valuation.deviation_pct < -10:
        attractiveness += 0.5
        factors.append("Below-market pricing (+0.5)")
    elif valuation.deviation_pct > 15:
        attractiveness -= 0.5
        factors.append("Above-market pricing (-0.5)")

    attractiveness = _clamp(attractiveness)

    if attractiveness >= 8.0:
        level = "Excellent"
        recommendation = "Buy"
    elif attractiveness >= 6.0:
        level = "Good"
        recommendation = "Buy"
    elif attractiveness >= 4.0:
        level = "Fair"
        recommendation = "Hold"
    else:
        level = "Poor"
        recommendation = "Avoid"

    factors.insert(0, f"Average risk score: {risk_avg:.1f}/10")

    return InvestmentAttractiveness(
        score=attractiveness,
        level=level,
        factors=factors,
        recommendation=recommendation
    )


# ─── Main Risk Engine ───────────────────────────────────────────────────────────

class RiskEngine:
    """
    Structured risk assessment engine.
    
    Separates:
      1. PropertyValuation — price estimation
      2. RiskAssessment — multi-category risk analysis
      3. InvestmentAttractiveness — combined scoring
    """

    def __init__(self):
        self._location_risk_data = None
        self._load_location_data()

    def _load_location_data(self):
        """Load location risk data from JSON."""
        try:
            with open(ARTIFACTS_DIR / 'location_risk_data.json', 'r') as f:
                self._location_risk_data = json.load(f)
        except Exception:
            self._location_risk_data = {
                "location_risk": {},
                "defaults": {
                    "safety": 7.0, "flood": "Medium", "infra": 6.5,
                    "liquidity": "Medium", "crime": "Low", "grade": "B",
                    "rent_per_sqft": 20, "appreciation": 6.0
                }
            }

    def _get_location_info(self, location: str) -> dict:
        """Get location info with defaults for unknown locations."""
        loc = location.lower().strip()
        risk_data = self._location_risk_data.get("location_risk", {})
        defaults = self._location_risk_data.get("defaults", {})
        return risk_data.get(loc, defaults)

    def assess_property(self, location: str, sqft: float, bhk: int,
                        bath: int, price: float) -> dict:
        """
        Full property assessment: valuation + risk + attractiveness.
        
        Returns dict with:
          - valuation: PropertyValuation
          - risks: list of RiskResult
          - attractiveness: InvestmentAttractiveness
          - overall_risk_score: float
          - overall_risk_level: str
        """
        info = self._get_location_info(location)
        rent_per_sqft = info.get("rent_per_sqft", 20)
        appreciation = info.get("appreciation", 6.0)
        grade = info.get("grade", "B")

        # 1. Valuation
        price_per_sqft = (price * 100000) / sqft if sqft > 0 else 0
        market_rate = (rent_per_sqft * 12) / 0.03
        bhk_adj = 1 + (bhk - 2) * 0.05
        bath_adj = 1 + (bath - 2) * 0.02
        adjusted_market = market_rate * bhk_adj * bath_adj
        deviation = ((price_per_sqft - adjusted_market) / adjusted_market * 100) if adjusted_market > 0 else 0

        if deviation > 20:
            val_status = "Overpriced"
        elif deviation > 10:
            val_status = "Slightly High"
        elif deviation < -20:
            val_status = "Underpriced"
        elif deviation < -10:
            val_status = "Slightly Low"
        else:
            val_status = "Fair"

        valuation = PropertyValuation(
            estimated_price=0,  # filled by caller from ML model
            price_per_sqft=round(price_per_sqft, 0),
            market_rate=round(adjusted_market, 0),
            deviation_pct=round(deviation, 1),
            status=val_status
        )

        # 2. Risk Assessment (6 categories)
        risks = [
            assess_financial_risk(price, sqft, bhk, bath, rent_per_sqft),
            assess_market_risk(price, sqft, rent_per_sqft, appreciation),
            assess_location_risk(info),
            assess_environmental_risk(info),
            assess_liquidity_risk(info),
            assess_rental_income_risk(price, sqft, bhk, bath, rent_per_sqft),
        ]

        # 3. Overall risk score (weighted average)
        # Weights (documented):
        #   Financial: 0.25 (price is primary concern)
        #   Market: 0.15 (market conditions matter but less than price)
        #   Location: 0.20 (fundamental to property value)
        #   Environmental: 0.10 (important but location-specific)
        #   Liquidity: 0.15 (affects exit strategy)
        #   Rental/Income: 0.15 (affects holding cost)
        weights = {
            "Financial": 0.25,
            "Market": 0.15,
            "Location": 0.20,
            "Environmental": 0.10,
            "Liquidity": 0.15,
            "Rental/Income": 0.15,
        }
        overall_score = sum(
            r.risk_score * weights.get(r.risk_category, 0.15) for r in risks
        )
        overall_score = _clamp(overall_score)

        # 4. Investment Attractiveness
        attractiveness = assess_investment_attractiveness(risks, valuation, grade)

        return {
            "valuation": valuation.to_dict(),
            "risks": [r.to_dict() for r in risks],
            "attractiveness": attractiveness.to_dict(),
            "overall_risk_score": overall_score,
            "overall_risk_level": _risk_level(overall_score),
        }


# ─── Backward-Compatible API ────────────────────────────────────────────────────

def get_comprehensive_risk_report(location, sqft, bhk, bath, price, ml_price=None):
    """
    Backward-compatible risk report matching existing API contract.
    
    Returns the same structure as util.py:get_comprehensive_risk_report
    but powered by the new risk engine.
    """
    engine = RiskEngine()
    assessment = engine.assess_property(location, sqft, bhk, bath, price)

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

    # Build location rating from location risk
    engine2 = RiskEngine()
    info = engine2._get_location_info(location)
    location_risk = next((r for r in assessment["risks"] if r["risk_category"] == "Location"), None)

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

    # Overall risk from engine
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
    }


if __name__ == '__main__':
    engine = RiskEngine()
    result = engine.assess_property('Koramangala', 1000, 2, 2, 150)
    print(json.dumps(result, indent=2))
