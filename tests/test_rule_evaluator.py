import unittest

from rules.rule_evaluator import RuleEvaluator


class TestRuleEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = RuleEvaluator("rules/eligibility_rules.yaml")

    def test_eligible_profile(self) -> None:
        report = self.evaluator.evaluate(
            {
                "annual_income": 120000,
                "monthly_debt": 1500,
                "credit_score": 720,
                "employment_status": "employed",
            }
        )
        self.assertEqual(report["status"], "Eligible")

    def test_ineligible_profile(self) -> None:
        report = self.evaluator.evaluate(
            {
                "annual_income": 20000,
                "monthly_debt": 3000,
                "credit_score": 580,
                "employment_status": "unemployed",
            }
        )
        self.assertEqual(report["status"], "Ineligible")
        self.assertGreater(len(report["failed_rules"]), 0)


if __name__ == "__main__":
    unittest.main()
