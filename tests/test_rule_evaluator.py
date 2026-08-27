import unittest

from rules.rule_evaluator import RuleEvaluator


BASE_ELIGIBLE_PROFILE = {
    "annual_income": 900000,
    "monthly_debt": 15000,
    "credit_score": 720,
    "employment_status": "employed",
    "employment_years": 4,
    "property_value": 5000000,
    "requested_loan_amount": 4000000,
    "down_payment": 1000000,
}


class TestRuleEvaluatorHappyPath(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = RuleEvaluator("rules/eligibility_rules.yaml")

    def test_all_rules_pass_is_eligible(self) -> None:
        report = self.evaluator.evaluate(BASE_ELIGIBLE_PROFILE)
        self.assertEqual(report["status"], "Eligible")
        self.assertEqual(report["failed_rules"], [])
        self.assertTrue(all(rule["passed"] for rule in report["rule_breakdown"]))

    def test_report_contains_all_seven_rules(self) -> None:
        report = self.evaluator.evaluate(BASE_ELIGIBLE_PROFILE)
        names = {rule["name"] for rule in report["rule_breakdown"]}
        self.assertEqual(
            names,
            {
                "Annual Income",
                "Debt-to-Income Ratio",
                "Credit Score",
                "Employment Status",
                "Job Stability",
                "Loan-to-Value Ratio",
                "Down Payment",
            },
        )


class TestRuleEvaluatorAllFail(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = RuleEvaluator("rules/eligibility_rules.yaml")

    def test_all_rules_fail_is_ineligible(self) -> None:
        profile = {
            "annual_income": 20000,
            "monthly_debt": 3000,
            "credit_score": 580,
            "employment_status": "unemployed",
            "employment_years": 0.5,
            "property_value": 2000000,
            "requested_loan_amount": 1950000,
            "down_payment": 50000,
        }
        report = self.evaluator.evaluate(profile)
        self.assertEqual(report["status"], "Ineligible")
        self.assertEqual(len(report["failed_rules"]), 7)


class TestRuleEvaluatorBoundaries(unittest.TestCase):
    """Each rule uses >= or <=, so the exact threshold value must PASS,
    and one unit past it must FAIL. These tests lock that behavior in."""

    def setUp(self) -> None:
        self.evaluator = RuleEvaluator("rules/eligibility_rules.yaml")

    def _rule(self, report, name):
        return next(r for r in report["rule_breakdown"] if r["name"] == name)

    # --- Income: threshold 50,000 ---
    def test_income_exactly_at_threshold_passes(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, annual_income=50000)
        report = self.evaluator.evaluate(profile)
        self.assertTrue(self._rule(report, "Annual Income")["passed"])

    def test_income_one_below_threshold_fails(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, annual_income=49999)
        report = self.evaluator.evaluate(profile)
        self.assertFalse(self._rule(report, "Annual Income")["passed"])

    # --- DTI: max 0.43 ---
    def test_dti_exactly_at_max_passes(self) -> None:
        # monthly_income = 900000/12 = 75000; 0.43 * 75000 = 32250
        profile = dict(BASE_ELIGIBLE_PROFILE, monthly_debt=32250)
        report = self.evaluator.evaluate(profile)
        self.assertTrue(self._rule(report, "Debt-to-Income Ratio")["passed"])

    def test_dti_just_over_max_fails(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, monthly_debt=32251)
        report = self.evaluator.evaluate(profile)
        self.assertFalse(self._rule(report, "Debt-to-Income Ratio")["passed"])

    # --- Credit score: min 650 ---
    def test_credit_score_exactly_at_min_passes(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, credit_score=650)
        report = self.evaluator.evaluate(profile)
        self.assertTrue(self._rule(report, "Credit Score")["passed"])

    def test_credit_score_one_below_min_fails(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, credit_score=649)
        report = self.evaluator.evaluate(profile)
        self.assertFalse(self._rule(report, "Credit Score")["passed"])

    # --- Employment status: case-insensitivity + rejection ---
    def test_employment_status_is_case_insensitive(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, employment_status="SELF-EMPLOYED")
        report = self.evaluator.evaluate(profile)
        self.assertTrue(self._rule(report, "Employment Status")["passed"])

    def test_unemployed_status_fails(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, employment_status="unemployed")
        report = self.evaluator.evaluate(profile)
        self.assertFalse(self._rule(report, "Employment Status")["passed"])

    def test_unrecognized_status_fails(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, employment_status="retired")
        report = self.evaluator.evaluate(profile)
        self.assertFalse(self._rule(report, "Employment Status")["passed"])

    # --- Job stability: min 2 years ---
    def test_employment_years_exactly_at_min_passes(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, employment_years=2)
        report = self.evaluator.evaluate(profile)
        self.assertTrue(self._rule(report, "Job Stability")["passed"])

    def test_employment_years_just_below_min_fails(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, employment_years=1.9)
        report = self.evaluator.evaluate(profile)
        self.assertFalse(self._rule(report, "Job Stability")["passed"])

    # --- LTV: max 0.95 ---
    def test_ltv_exactly_at_max_passes(self) -> None:
        # property_value 5,000,000 * 0.95 = 4,750,000
        profile = dict(BASE_ELIGIBLE_PROFILE, requested_loan_amount=4750000)
        report = self.evaluator.evaluate(profile)
        self.assertTrue(self._rule(report, "Loan-to-Value Ratio")["passed"])

    def test_ltv_just_over_max_fails(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, requested_loan_amount=4750001)
        report = self.evaluator.evaluate(profile)
        self.assertFalse(self._rule(report, "Loan-to-Value Ratio")["passed"])

    # --- Down payment: min 5% ---
    def test_down_payment_exactly_at_min_passes(self) -> None:
        # 5% of 5,000,000 = 250,000
        profile = dict(BASE_ELIGIBLE_PROFILE, down_payment=250000)
        report = self.evaluator.evaluate(profile)
        self.assertTrue(self._rule(report, "Down Payment")["passed"])

    def test_down_payment_just_below_min_fails(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, down_payment=249999)
        report = self.evaluator.evaluate(profile)
        self.assertFalse(self._rule(report, "Down Payment")["passed"])


class TestRuleEvaluatorEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = RuleEvaluator("rules/eligibility_rules.yaml")

    def test_zero_income_fails_income_rule_not_crash(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, annual_income=0)
        report = self.evaluator.evaluate(profile)
        self.assertEqual(report["status"], "Ineligible")

    def test_zero_down_payment_fails_without_property_value(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, down_payment=0)
        profile.pop("property_value")
        report = self.evaluator.evaluate(profile)
        down_rule = next(r for r in report["rule_breakdown"] if r["name"] == "Down Payment")
        self.assertFalse(down_rule["passed"])
        self.assertEqual(report["status"], "Ineligible")

    def test_zero_property_value_treated_as_not_evaluated(self) -> None:
        # property_value present but 0 -> LTV/down-payment can't be computed;
        # those two checks should be skipped (passed=True) rather than crash
        # on a divide-by-zero, while the other rules still evaluate normally.
        profile = dict(BASE_ELIGIBLE_PROFILE, property_value=0)
        report = self.evaluator.evaluate(profile)
        ltv_rule = next(r for r in report["rule_breakdown"] if r["name"] == "Loan-to-Value Ratio")
        down_rule = next(r for r in report["rule_breakdown"] if r["name"] == "Down Payment")
        self.assertTrue(ltv_rule["passed"])
        self.assertTrue(down_rule["passed"])

    def test_missing_optional_fields_do_not_crash(self) -> None:
        minimal_profile = {
            "annual_income": 900000,
            "monthly_debt": 15000,
            "credit_score": 720,
            "employment_status": "employed",
        }
        report = self.evaluator.evaluate(minimal_profile)
        self.assertIn(report["status"], ("Eligible", "Ineligible"))

    def test_negative_income_fails_gracefully(self) -> None:
        profile = dict(BASE_ELIGIBLE_PROFILE, annual_income=-5000)
        report = self.evaluator.evaluate(profile)
        self.assertEqual(report["status"], "Ineligible")


if __name__ == "__main__":
    unittest.main()
