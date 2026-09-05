"""
Unit Tests for Phase 3 Risk Engine
====================================
Tests all 6 risk categories, valuation, attractiveness, and edge cases.
Run: python -m pytest test_risk_engine.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import unittest
from risk_engine import (
    assess_financial_risk, assess_market_risk, assess_location_risk,
    assess_environmental_risk, assess_liquidity_risk, assess_rental_income_risk,
    assess_investment_attractiveness, RiskEngine, PropertyValuation,
    _risk_level, _clamp
)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions _risk_level and _clamp."""

    def test_risk_level_low(self):
        self.assertEqual(_risk_level(1.0), "Low")
        self.assertEqual(_risk_level(3.0), "Low")

    def test_risk_level_medium(self):
        self.assertEqual(_risk_level(3.1), "Medium")
        self.assertEqual(_risk_level(5.0), "Medium")

    def test_risk_level_high(self):
        self.assertEqual(_risk_level(5.1), "High")
        self.assertEqual(_risk_level(7.0), "High")

    def test_risk_level_very_high(self):
        self.assertEqual(_risk_level(7.1), "Very High")
        self.assertEqual(_risk_level(10.0), "Very High")

    def test_clamp(self):
        self.assertEqual(_clamp(0.5), 1.0)
        self.assertEqual(_clamp(5.5), 5.5)
        self.assertEqual(_clamp(11.0), 10.0)


class TestFinancialRisk(unittest.TestCase):
    """Test Financial Risk category."""

    def test_fair_price(self):
        # Price at market rate (rent=20, market=8000, price=80L for 1000sqft)
        result = assess_financial_risk(80.0, 1000, 2, 2, rent_per_sqft=20)
        self.assertEqual(result.risk_category, "Financial")
        self.assertEqual(result.risk_score, 5.0)
        self.assertEqual(result.risk_level, "Medium")

    def test_overpriced(self):
        # Price 40% above market
        result = assess_financial_risk(112.0, 1000, 2, 2, rent_per_sqft=20)
        self.assertGreater(result.risk_score, 5.0)
        self.assertIn("overpric", result.drivers[0].lower())

    def test_underpriced(self):
        # Price 25% below market
        result = assess_financial_risk(60.0, 1000, 2, 2, rent_per_sqft=20)
        self.assertLess(result.risk_score, 5.0)

    def test_has_evidence(self):
        result = assess_financial_risk(80.0, 1000, 2, 2, rent_per_sqft=20)
        self.assertGreater(len(result.evidence), 0)

    def test_has_mitigation_when_risky(self):
        result = assess_financial_risk(120.0, 1000, 2, 2, rent_per_sqft=20)
        self.assertGreater(len(result.mitigation), 0)


class TestMarketRisk(unittest.TestCase):
    """Test Market Risk category."""

    def test_normal_market(self):
        # Price=80L, rent=40/sqft -> annual_rent=480000, yield=6%, price_to_rent=16.7
        result = assess_market_risk(80.0, 1000, rent_per_sqft=40, appreciation=6.0)
        self.assertEqual(result.risk_category, "Market")
        self.assertEqual(result.risk_score, 4.0)  # yield >5% -> -1.0

    def test_low_yield(self):
        # Very high price, low rent = low yield
        result = assess_market_risk(200.0, 1000, rent_per_sqft=10, appreciation=6.0)
        self.assertGreater(result.risk_score, 5.0)

    def test_high_yield(self):
        # Low price, high rent = high yield
        result = assess_market_risk(40.0, 1000, rent_per_sqft=30, appreciation=8.0)
        self.assertLess(result.risk_score, 5.0)


class TestLocationRisk(unittest.TestCase):
    """Test Location Risk category."""

    def test_premium_location(self):
        info = {"safety": 9.0, "crime": "Low", "infra": 9.0, "grade": "A"}
        result = assess_location_risk(info)
        self.assertLess(result.risk_score, 5.0)
        self.assertEqual(result.risk_category, "Location")

    def test_poor_location(self):
        info = {"safety": 4.0, "crime": "High", "infra": 4.0, "grade": "D"}
        result = assess_location_risk(info)
        self.assertGreater(result.risk_score, 7.0)

    def test_default_values(self):
        info = {}
        result = assess_location_risk(info)
        self.assertEqual(result.risk_score, 5.0)


class TestEnvironmentalRisk(unittest.TestCase):
    """Test Environmental Risk category."""

    def test_high_flood(self):
        info = {"flood": "High"}
        result = assess_environmental_risk(info)
        self.assertGreater(result.risk_score, 5.0)
        self.assertIn("flood", result.drivers[0].lower())

    def test_low_flood(self):
        info = {"flood": "Low"}
        result = assess_environmental_risk(info)
        self.assertLess(result.risk_score, 5.0)

    def test_medium_flood(self):
        info = {"flood": "Medium"}
        result = assess_environmental_risk(info)
        self.assertEqual(result.risk_score, 5.5)


class TestLiquidityRisk(unittest.TestCase):
    """Test Liquidity Risk category."""

    def test_low_liquidity(self):
        info = {"liquidity": "Low"}
        result = assess_liquidity_risk(info)
        self.assertGreater(result.risk_score, 5.0)

    def test_high_liquidity(self):
        info = {"liquidity": "High"}
        result = assess_liquidity_risk(info)
        self.assertLess(result.risk_score, 5.0)


class TestRentalIncomeRisk(unittest.TestCase):
    """Test Rental/Income Risk category."""

    def test_long_payback(self):
        # Very expensive property, low rent
        result = assess_rental_income_risk(200.0, 1000, 2, 2, rent_per_sqft=10)
        self.assertGreater(result.risk_score, 5.0)

    def test_short_payback(self):
        # Cheap property, high rent
        result = assess_rental_income_risk(40.0, 1000, 2, 2, rent_per_sqft=30)
        self.assertLess(result.risk_score, 5.0)

    def test_small_space(self):
        result = assess_rental_income_risk(80.0, 500, 3, 2, rent_per_sqft=20)
        self.assertGreater(result.risk_score, 5.0)
        self.assertTrue(any("small" in d.lower() or "sqft" in d.lower() for d in result.drivers))


class TestInvestmentAttractiveness(unittest.TestCase):
    """Test Investment Attractiveness scoring."""

    def test_low_risk_high_attractiveness(self):
        from risk_engine import RiskResult
        risks = [
            RiskResult("Financial", 3.0, "Low", [], [], []),
            RiskResult("Market", 3.0, "Low", [], [], []),
            RiskResult("Location", 3.0, "Low", [], [], []),
            RiskResult("Environmental", 3.0, "Low", [], [], []),
            RiskResult("Liquidity", 3.0, "Low", [], [], []),
            RiskResult("Rental/Income", 3.0, "Low", [], [], []),
        ]
        val = PropertyValuation(80, 8000, 8000, -5.0, "Fair")
        result = assess_investment_attractiveness(risks, val, "A")
        self.assertGreaterEqual(result.score, 7.0)
        self.assertEqual(result.recommendation, "Buy")

    def test_high_risk_low_attractiveness(self):
        from risk_engine import RiskResult
        risks = [
            RiskResult("Financial", 8.0, "Very High", [], [], []),
            RiskResult("Market", 8.0, "Very High", [], [], []),
            RiskResult("Location", 8.0, "Very High", [], [], []),
            RiskResult("Environmental", 8.0, "Very High", [], [], []),
            RiskResult("Liquidity", 8.0, "Very High", [], [], []),
            RiskResult("Rental/Income", 8.0, "Very High", [], [], []),
        ]
        val = PropertyValuation(200, 16000, 8000, 50.0, "Overpriced")
        result = assess_investment_attractiveness(risks, val, "D")
        self.assertLessEqual(result.score, 4.0)
        self.assertEqual(result.recommendation, "Avoid")


class TestRiskEngine(unittest.TestCase):
    """Test the full RiskEngine integration."""

    def setUp(self):
        self.engine = RiskEngine()

    def test_assess_property_returns_all_fields(self):
        result = self.engine.assess_property('Koramangala', 1000, 2, 2, 150)
        self.assertIn("valuation", result)
        self.assertIn("risks", result)
        self.assertIn("attractiveness", result)
        self.assertIn("overall_risk_score", result)
        self.assertIn("overall_risk_level", result)

    def test_assess_property_has_6_risk_categories(self):
        result = self.engine.assess_property('Koramangala', 1000, 2, 2, 150)
        self.assertEqual(len(result["risks"]), 6)
        categories = [r["risk_category"] for r in result["risks"]]
        self.assertIn("Financial", categories)
        self.assertIn("Market", categories)
        self.assertIn("Location", categories)
        self.assertIn("Environmental", categories)
        self.assertIn("Liquidity", categories)
        self.assertIn("Rental/Income", categories)

    def test_unknown_location_uses_defaults(self):
        result = self.engine.assess_property('UnknownLocation', 1000, 2, 2, 80)
        self.assertIn("valuation", result)
        self.assertEqual(result["risks"][0]["risk_category"], "Financial")

    def test_overall_score_in_range(self):
        result = self.engine.assess_property('Koramangala', 1000, 2, 2, 150)
        self.assertGreaterEqual(result["overall_risk_score"], 1.0)
        self.assertLessEqual(result["overall_risk_score"], 10.0)

    def test_category_scores_sum_to_weighted(self):
        result = self.engine.assess_property('Koramangala', 1000, 2, 2, 150)
        weights = {"Financial": 0.25, "Market": 0.15, "Location": 0.20,
                    "Environmental": 0.10, "Liquidity": 0.15, "Rental/Income": 0.15}
        weighted = sum(
            r["risk_score"] * weights[r["risk_category"]]
            for r in result["risks"]
        )
        self.assertAlmostEqual(result["overall_risk_score"], round(weighted, 1), places=0)


class TestBackwardCompatibility(unittest.TestCase):
    """Test that the new engine produces compatible output with old API."""

    def test_report_structure(self):
        from risk_engine import get_comprehensive_risk_report
        report = get_comprehensive_risk_report('Koramangala', 1000, 2, 2, 150)
        self.assertIn("estimated_price", report)
        self.assertIn("investment_risk", report)
        self.assertIn("overpricing_analysis", report)
        self.assertIn("location_rating", report)
        self.assertIn("roi_estimate", report)

    def test_investment_risk_has_score_and_level(self):
        from risk_engine import get_comprehensive_risk_report
        report = get_comprehensive_risk_report('Koramangala', 1000, 2, 2, 150)
        self.assertIn("score", report["investment_risk"])
        self.assertIn("level", report["investment_risk"])

    def test_overpricing_has_status(self):
        from risk_engine import get_comprehensive_risk_report
        report = get_comprehensive_risk_report('Koramangala', 1000, 2, 2, 150)
        self.assertIn("status", report["overpricing_analysis"])
        self.assertIn(report["overpricing_analysis"]["status"],
                      ["Overpriced", "Slightly High", "Fair", "Slightly Low", "Underpriced"])


if __name__ == '__main__':
    unittest.main(verbosity=2)
