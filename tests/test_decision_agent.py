import unittest

from agents.decision_agent import DecisionAgent


class StubRuleEvaluator:
    def __init__(self, result: dict):
        self.result = result
        self.called_with = None

    def evaluate(self, profile: dict) -> dict:
        self.called_with = profile
        return self.result


class TestDecisionAgent(unittest.TestCase):
    def test_missing_fields_below_max_turns_asks_followup(self) -> None:
        agent = DecisionAgent(rule_evaluator=StubRuleEvaluator({"status": "Eligible"}))
        state = {
            "applicant_profile": {"annual_income": 90000},
            "turn_count": 1,
            "max_turns": 16,
        }
        result = agent(state)
        self.assertTrue(result["needs_followup"])
        self.assertEqual(result["decision_status"], "Requires More Info")
        self.assertIsNotNone(result["followup_field"])
        # Rule evaluator must NOT be called while info is still missing
        self.assertIsNone(agent.rule_evaluator.called_with)

    def test_missing_fields_at_max_turns_gives_up_gracefully(self) -> None:
        agent = DecisionAgent(rule_evaluator=StubRuleEvaluator({"status": "Eligible"}))
        state = {
            "applicant_profile": {"annual_income": 90000},
            "turn_count": 16,
            "max_turns": 16,
        }
        result = agent(state)
        self.assertEqual(result["decision_status"], "Requires More Info")
        self.assertFalse(result["needs_followup"])
        self.assertIn("missing_fields", result["final_report"])
        self.assertIsNone(agent.rule_evaluator.called_with)

    def test_all_fields_present_calls_rule_evaluator(self) -> None:
        full_profile = {
            "annual_income": 900000,
            "monthly_debt": 15000,
            "credit_score": 720,
            "employment_status": "employed",
            "employment_years": 4,
            "property_value": 5000000,
            "requested_loan_amount": 4000000,
            "down_payment": 1000000,
            "total_savings": 1500000,
        }
        agent = DecisionAgent(
            rule_evaluator=StubRuleEvaluator(
                {"status": "Eligible", "summary": "All rules passed"}
            )
        )
        state = {"applicant_profile": full_profile, "turn_count": 8, "max_turns": 16}
        result = agent(state)
        self.assertEqual(result["decision_status"], "Eligible")
        self.assertFalse(result["needs_followup"])
        self.assertEqual(agent.rule_evaluator.called_with, full_profile)

    def test_savings_is_optional_for_the_eligibility_decision(self) -> None:
        profile = {
            "annual_income": 90000,
            "monthly_debt": 400,
            "credit_score": 589,
            "employment_status": "employed",
            "employment_years": 0,
            "down_payment": 0,
        }
        agent = DecisionAgent(
            rule_evaluator=StubRuleEvaluator(
                {"status": "Ineligible", "summary": "Known rules failed", "rule_breakdown": []}
            )
        )

        result = agent({"applicant_profile": profile, "skipped_fields": ["total_savings"], "turn_count": 8})

        self.assertEqual(result["decision_status"], "Ineligible")
        self.assertFalse(result["needs_followup"])
        self.assertEqual(agent.rule_evaluator.called_with, profile)

    def test_skipped_required_field_is_named_in_final_summary(self) -> None:
        agent = DecisionAgent(rule_evaluator=StubRuleEvaluator({"status": "Eligible"}))
        profile = {
            "annual_income": 90000,
            "monthly_debt": 0,
            "credit_score": 710,
            "employment_status": "employed",
            "employment_years": 5,
        }
        result = agent(
            {"applicant_profile": profile, "skipped_fields": ["down_payment"], "turn_count": 8}
        )
        self.assertEqual(result["decision_status"], "Requires More Info")
        self.assertIn("Down Payment", result["decision_summary"])

    def test_unemployed_status_ends_precheck_without_asking_past_job_questions(self) -> None:
        evaluator = StubRuleEvaluator({"status": "Eligible"})
        evaluator.rules = {
            "income_threshold": 50000,
            "min_credit_score": 650,
        }
        agent = DecisionAgent(rule_evaluator=evaluator)
        profile = {
            "down_payment": 0,
            "credit_score": 800,
            "employment_status": "unemployed",
        }

        result = agent({"applicant_profile": profile, "turn_count": 3, "max_turns": 16})

        self.assertEqual(result["decision_status"], "Ineligible")
        self.assertFalse(result["needs_followup"])
        self.assertIsNone(result["followup_field"])
        self.assertIn("Employment Status", result["final_report"]["failed_rules"])
        self.assertNotIn("employment_years", result["applicant_profile"])
        self.assertNotIn("annual_income", result["applicant_profile"])
        self.assertIsNone(evaluator.called_with)

    def test_followup_field_is_first_missing_field_in_schema_order(self) -> None:
        agent = DecisionAgent(rule_evaluator=StubRuleEvaluator({"status": "Eligible"}))
        state = {
            "applicant_profile": {
                "annual_income": 90000,
                "monthly_debt": 1000,
            },
            "turn_count": 2,
            "max_turns": 16,
        }
        result = agent(state)
        self.assertEqual(result["followup_field"], "down_payment")


if __name__ == "__main__":
    unittest.main()
