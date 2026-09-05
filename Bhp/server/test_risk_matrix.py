"""
Unit Tests for Phase 4: Probability × Impact Risk Matrix
=========================================================
Tests P×I assessments for all 6 risk categories, risk matrix, and thresholds.
Run: python -m unittest test_risk_matrix -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import unittest
from risk_matrix import (
    assess_financial_pi, assess_market_pi, assess_location_pi,
    assess_environmental_pi, assess_liquidity_pi, assess_rental_income_pi,
    build_risk_matrix, assess_property_with_pi, _risk_level_from_pi,
    RISK_LEVEL_THRESHOLDS
)


class TestRiskLevelThresholds(unittest.TestCase):
    """Test configurable risk level thresholds."""

    def test_low_range(self):
        for score in [1, 2, 3, 4]:
            self.assertEqual(_risk_level_from_pi(score), "Low")

    def test_medium_range(self):
        for score in [5, 6, 7, 8, 9]:
            self.assertEqual(_risk_level_from_pi(score), "Medium")

    def test_high_range(self):
        for score in [10, 11, 12, 13, 14, 15]:
            self.assertEqual(_risk_level_from_pi(score), "High")

    def test_critical_range(self):
        for score in [16, 20, 25]:
            self.assertEqual(_risk_level_from_pi(score), "Critical")

    def test_thresholds_are_configurable(self):
        self.assertIn("Low", RISK_LEVEL_THRESHOLDS)
        self.assertIn("Medium", RISK_LEVEL_THRESHOLDS)
        self.assertIn("High", RISK_LEVEL_THRESHOLDS)
        self.assertIn("Critical", RISK_LEVEL_THRESHOLDS)


class TestFinancialPI(unittest.TestCase):
    """Test Financial Risk P×I assessment."""

    def test_underpriced(self):
        # 25% below market — very unlikely overpriced
        result = assess_financial_pi(60.0, 1000, 2, 2, rent_per_sqft=20)
        self.assertEqual(result.risk_category, "Financial")
        self.assertEqual(result.probability, 1)
        self.assertLessEqual(result.risk_score, 5)

    def test_fair_price(self):
        # At market rate
        result = assess_financial_pi(80.0, 1000, 2, 2, rent_per_sqft=20)
        self.assertEqual(result.probability, 2)
        self.assertLessEqual(result.risk_score, 10)

    def test_overpriced(self):
        # 40% above market
        result = assess_financial_pi(112.0, 1000, 2, 2, rent_per_sqft=20)
        self.assertEqual(result.probability, 5)
        self.assertEqual(result.impact, 5)
        self.assertEqual(result.risk_score, 25)
        self.assertEqual(result.risk_level, "Critical")

    def test_has_reasoning(self):
        result = assess_financial_pi(80.0, 1000, 2, 2, rent_per_sqft=20)
        self.assertGreater(len(result.probability_reasoning), 0)
        self.assertGreater(len(result.impact_reasoning), 0)

    def test_score_equals_prob_times_impact(self):
        result = assess_financial_pi(100.0, 1000, 2, 2, rent_per_sqft=20)
        self.assertEqual(result.risk_score, result.probability * result.impact)


class TestMarketPI(unittest.TestCase):
    """Test Market Risk P×I assessment."""

    def test_strong_market(self):
        # High yield, high appreciation
        result = assess_market_pi(40.0, 1000, rent_per_sqft=30, appreciation=10)
        self.assertEqual(result.probability, 1)
        self.assertLessEqual(result.risk_score, 5)

    def test_weak_market(self):
        # Very low yield, low appreciation
        result = assess_market_pi(200.0, 1000, rent_per_sqft=5, appreciation=2)
        self.assertEqual(result.probability, 5)
        self.assertEqual(result.impact, 5)
        self.assertEqual(result.risk_score, 25)

    def test_moderate_market(self):
        result = assess_market_pi(80.0, 1000, rent_per_sqft=20, appreciation=5)
        self.assertGreaterEqual(result.probability, 2)
        self.assertLessEqual(result.probability, 4)


class TestLocationPI(unittest.TestCase):
    """Test Location Risk P×I assessment."""

    def test_premium_location(self):
        info = {"safety": 9.0, "crime": "Low", "grade": "A"}
        result = assess_location_pi(info)
        self.assertEqual(result.probability, 1)
        self.assertEqual(result.impact, 1)
        self.assertEqual(result.risk_score, 1)

    def test_poor_location(self):
        info = {"safety": 4.0, "crime": "High", "grade": "D"}
        result = assess_location_pi(info)
        self.assertEqual(result.probability, 5)
        self.assertEqual(result.impact, 5)
        self.assertEqual(result.risk_score, 25)

    def test_defaults(self):
        info = {}
        result = assess_location_pi(info)
        self.assertGreaterEqual(result.probability, 1)
        self.assertLessEqual(result.probability, 5)


class TestEnvironmentalPI(unittest.TestCase):
    """Test Environmental Risk P×I assessment."""

    def test_low_flood(self):
        info = {"flood": "Low"}
        result = assess_environmental_pi(info)
        self.assertEqual(result.probability, 1)
        self.assertEqual(result.impact, 1)
        self.assertEqual(result.risk_score, 1)

    def test_high_flood(self):
        info = {"flood": "High"}
        result = assess_environmental_pi(info)
        self.assertEqual(result.probability, 5)
        self.assertEqual(result.impact, 5)
        self.assertEqual(result.risk_score, 25)

    def test_medium_flood(self):
        info = {"flood": "Medium"}
        result = assess_environmental_pi(info)
        self.assertEqual(result.probability, 3)
        self.assertEqual(result.impact, 3)
        self.assertEqual(result.risk_score, 9)


class TestLiquidityPI(unittest.TestCase):
    """Test Liquidity Risk P×I assessment."""

    def test_high_liquidity(self):
        info = {"liquidity": "High"}
        result = assess_liquidity_pi(info)
        self.assertEqual(result.probability, 1)
        self.assertEqual(result.impact, 1)
        self.assertEqual(result.risk_score, 1)

    def test_low_liquidity(self):
        info = {"liquidity": "Low"}
        result = assess_liquidity_pi(info)
        self.assertEqual(result.probability, 5)
        self.assertEqual(result.impact, 5)
        self.assertEqual(result.risk_score, 25)


class TestRentalIncomePI(unittest.TestCase):
    """Test Rental/Income Risk P×I assessment."""

    def test_strong_rental(self):
        # Low price, high rent — short payback
        result = assess_rental_income_pi(40.0, 1000, 2, 2, rent_per_sqft=30)
        self.assertEqual(result.probability, 1)
        self.assertLessEqual(result.risk_score, 5)

    def test_weak_rental(self):
        # High price, low rent — long payback
        result = assess_rental_income_pi(200.0, 1000, 2, 2, rent_per_sqft=5)
        self.assertEqual(result.probability, 5)
        self.assertEqual(result.impact, 5)
        self.assertEqual(result.risk_score, 25)


class TestRiskMatrix(unittest.TestCase):
    """Test full risk matrix construction."""

    def test_matrix_has_all_categories(self):
        matrix = build_risk_matrix('Koramangala', 1000, 2, 2, 150)
        self.assertEqual(len(matrix.assessments), 6)
        categories = [a.risk_category for a in matrix.assessments]
        self.assertIn("Financial", categories)
        self.assertIn("Market", categories)
        self.assertIn("Location", categories)
        self.assertIn("Environmental", categories)
        self.assertIn("Liquidity", categories)
        self.assertIn("Rental/Income", categories)

    def test_matrix_has_overall_scores(self):
        matrix = build_risk_matrix('Koramangala', 1000, 2, 2, 150)
        self.assertGreaterEqual(matrix.overall_probability, 1.0)
        self.assertLessEqual(matrix.overall_probability, 5.0)
        self.assertGreaterEqual(matrix.overall_impact, 1.0)
        self.assertLessEqual(matrix.overall_impact, 5.0)
        self.assertGreaterEqual(matrix.overall_risk_score, 1.0)
        self.assertLessEqual(matrix.overall_risk_score, 25.0)

    def test_matrix_has_category_dict(self):
        matrix = build_risk_matrix('Koramangala', 1000, 2, 2, 150)
        self.assertEqual(len(matrix.matrix), 6)
        for cat, data in matrix.matrix.items():
            self.assertIn("probability", data)
            self.assertIn("impact", data)
            self.assertIn("score", data)
            self.assertIn("level", data)

    def test_unknown_location_uses_defaults(self):
        matrix = build_risk_matrix('UnknownPlace', 1000, 2, 2, 80)
        self.assertEqual(len(matrix.assessments), 6)

    def test_score_equals_prob_times_impact(self):
        matrix = build_risk_matrix('Koramangala', 1000, 2, 2, 150)
        for a in matrix.assessments:
            self.assertEqual(a.risk_score, a.probability * a.impact)


class TestPropertyWithPI(unittest.TestCase):
    """Test full property assessment with P×I."""

    def test_returns_all_phases(self):
        result = assess_property_with_pi('Koramangala', 1000, 2, 2, 150)
        self.assertIn("valuation", result)
        self.assertIn("risks", result)
        self.assertIn("attractiveness", result)
        self.assertIn("risk_matrix", result)

    def test_risk_matrix_has_pi_fields(self):
        result = assess_property_with_pi('Koramangala', 1000, 2, 2, 150)
        matrix = result["risk_matrix"]
        self.assertIn("assessments", matrix)
        self.assertIn("overall_probability", matrix)
        self.assertIn("overall_impact", matrix)
        self.assertIn("overall_risk_score", matrix)
        self.assertIn("overall_risk_level", matrix)

    def test_each_assessment_has_pi(self):
        result = assess_property_with_pi('Koramangala', 1000, 2, 2, 150)
        for a in result["risk_matrix"]["assessments"]:
            self.assertIn("probability", a)
            self.assertIn("impact", a)
            self.assertIn("risk_score", a)
            self.assertIn("risk_level", a)
            self.assertIn("probability_reasoning", a)
            self.assertIn("impact_reasoning", a)


if __name__ == '__main__':
    unittest.main(verbosity=2)
