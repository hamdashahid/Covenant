import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.decision_agent import CONFIDENCE_THRESHOLD, DecisionAgent


class StubRuleEvaluator:
    def evaluate(self, profile):
        return {
            "status": "Eligible",
            "summary": "All rules passed",
            "failed_rules": [],
            "rule_trace": [],
            "metrics": {},
        }


class TestDecisionAgent(unittest.TestCase):
    def test_low_confidence_triggers_retry_for_current_field(self) -> None:
        agent = DecisionAgent(rule_evaluator=StubRuleEvaluator())

        state = agent(
            {
                "applicant_profile": {
                    "annual_income": 90000,
                    "monthly_debt": 1200,
                    "credit_score": 710,
                    "employment_status": "employed",
                },
                "current_question_field": "credit_score",
                "last_extraction": {"confidence": CONFIDENCE_THRESHOLD - 0.1},
                "turn_count": 2,
                "max_turns": 8,
            }
        )

        self.assertTrue(state["needs_followup"])
        self.assertEqual(state["followup_field"], "credit_score")
        self.assertEqual(state["decision_status"], "Requires More Info")
        self.assertIn("higher-confidence answer for credit_score", state["decision_summary"])

    def test_low_confidence_blocks_final_evaluation_at_turn_limit(self) -> None:
        agent = DecisionAgent(rule_evaluator=StubRuleEvaluator())

        state = agent(
            {
                "applicant_profile": {
                    "annual_income": 90000,
                    "monthly_debt": 1200,
                    "credit_score": 710,
                    "employment_status": "employed",
                },
                "current_question_field": "credit_score",
                "last_extraction": {"confidence": CONFIDENCE_THRESHOLD - 0.1},
                "turn_count": 8,
                "max_turns": 8,
            }
        )

        self.assertFalse(state["needs_followup"])
        self.assertEqual(state["decision_status"], "Requires More Info")
        self.assertIn("low confidence in credit_score", state["decision_summary"])
        self.assertEqual(state["final_report"]["low_confidence_fields"], ["credit_score"])


if __name__ == "__main__":
    unittest.main()