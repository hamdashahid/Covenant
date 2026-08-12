import json
import unittest

from agents.extraction_validation import ExtractionValidationNode
from core.context_builder import ContextBuilder
from core.profile_updater import ProfileUpdater
from core.schemas import EXTRACTION_SCHEMA


class StubLLM:
    def __init__(self, payload):
        self.payload = payload

    def extract_structured(self, prompt: str, latest_response: str) -> str:
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload)


def make_node(payload) -> ExtractionValidationNode:
    return ExtractionValidationNode(
        llm_client=StubLLM(payload),
        context_builder=ContextBuilder(),
        profile_updater=ProfileUpdater(),
        extraction_schema=EXTRACTION_SCHEMA,
    )


def base_state(response_text: str) -> dict:
    return {
        "conversation_history": [],
        "applicant_profile": {},
        "current_question": "Tell me about your finances",
        "latest_user_response": response_text,
    }


class TestExtractionHappyPath(unittest.TestCase):
    def test_valid_extraction_updates_profile(self) -> None:
        node = make_node({
            "fields": {
                "annual_income": 90000,
                "monthly_debt": 1200,
                "credit_score": 710,
                "employment_status": "employed",
            },
            "confidence": 0.95,
            "issues": [],
        })
        state = node(base_state("My annual income is 90,000."))
        self.assertEqual(state["applicant_profile"]["annual_income"], 90000.0)
        self.assertEqual(state["applicant_profile"]["credit_score"], 710)

    def test_multiple_fields_extracted_from_one_message(self) -> None:
        node = make_node({
            "fields": {
                "property_value": 5000000,
                "requested_loan_amount": 4000000,
                "down_payment": 1000000,
            },
            "confidence": 0.9,
            "issues": [],
        })
        state = node(base_state("The house is 5,000,000, I need 4,000,000 loan, and can put 1,000,000 down"))
        profile = state["applicant_profile"]
        self.assertEqual(profile["property_value"], 5000000.0)
        self.assertEqual(profile["requested_loan_amount"], 4000000.0)
        self.assertEqual(profile["down_payment"], 1000000.0)


class TestExtractionMalformedResponses(unittest.TestCase):
    def test_invalid_json_gracefully_handled(self) -> None:
        node = make_node("not-json")
        state = node(base_state("I earn well"))
        self.assertEqual(state["applicant_profile"], {})
        self.assertIn("Extractor response was not valid JSON", state["last_extraction"]["issues"])

    def test_accepts_above_800_credit_score(self) -> None:
        node = make_node({
            "fields": {"credit_score": "above 800"},
            "confidence": 0.9,
            "issues": [],
        })
        state = node(base_state("Above 800"))
        self.assertEqual(state["applicant_profile"]["credit_score"], 800)

    def test_accepts_125k_income(self) -> None:
        node = make_node({
            "fields": {"annual_income": "125k"},
            "confidence": 0.9,
            "issues": [],
        })
        state = node(base_state("About 125k"))
        self.assertEqual(state["applicant_profile"]["annual_income"], 125000.0)

    def test_clear_numeric_credit_score_vs_vague_answer(self) -> None:
        clear_node = make_node({
            "fields": {"credit_score": "above 800"},
            "confidence": 0.9,
            "issues": [],
        })
        clear_state = clear_node(
            {
                "conversation_history": [],
                "applicant_profile": {},
                "current_question": "What is your credit score?",
                "latest_user_response": "Above 800",
            }
        )
        self.assertEqual(clear_state["applicant_profile"]["credit_score"], 800)

        vague_node = make_node({
            "fields": {"credit_score": "close to perfect"},
            "confidence": 0.9,
            "issues": [],
        })
        vague_state = vague_node(
            {
                "conversation_history": [],
                "applicant_profile": {},
                "current_question": "What is your credit score?",
                "latest_user_response": "Close to perfect",
            }
        )
        self.assertNotIn("credit_score", vague_state["applicant_profile"])
        self.assertIn("credit_score must be between 300 and 850", vague_state["last_extraction"]["issues"])

    def test_missing_fields_key_does_not_crash(self) -> None:
        node = make_node({"confidence": 0.5})
        state = node(base_state("hello"))
        self.assertEqual(state["applicant_profile"], {})

    def test_low_confidence_flagged_but_still_merges(self) -> None:
        node = make_node({
            "fields": {"annual_income": 60000},
            "confidence": 0.1,
            "issues": [],
        })
        state = node(base_state("maybe around 60000?"))
        self.assertIn("Extraction confidence too low", state["last_extraction"]["issues"])
        # low confidence is a warning, not a hard block -> value still lands
        self.assertEqual(state["applicant_profile"]["annual_income"], 60000.0)


class TestExtractionValidationRejectsOutOfRangeValues(unittest.TestCase):
    def test_credit_score_above_valid_range_rejected(self) -> None:
        node = make_node({
            "fields": {"credit_score": 90000},
            "confidence": 0.9,
            "issues": [],
        })
        state = node(base_state("90000"))
        self.assertNotIn("credit_score", state["applicant_profile"])
        self.assertIn("credit_score must be between 300 and 850", state["last_extraction"]["issues"])

    def test_credit_score_non_numeric_rejected(self) -> None:
        node = make_node({
            "fields": {"credit_score": "excellent"},
            "confidence": 0.7,
            "issues": [],
        })
        state = node(base_state("excellent"))
        self.assertNotIn("credit_score", state["applicant_profile"])

    def test_negative_annual_income_rejected(self) -> None:
        node = make_node({
            "fields": {"annual_income": -5000},
            "confidence": 0.9,
            "issues": [],
        })
        state = node(base_state("-5000"))
        self.assertNotIn("annual_income", state["applicant_profile"])
        self.assertIn("annual_income must be > 0", state["last_extraction"]["issues"])

    def test_negative_monthly_debt_rejected(self) -> None:
        node = make_node({
            "fields": {"monthly_debt": -100},
            "confidence": 0.9,
            "issues": [],
        })
        state = node(base_state("-100"))
        self.assertNotIn("monthly_debt", state["applicant_profile"])

    def test_zero_property_value_rejected(self) -> None:
        node = make_node({
            "fields": {"property_value": 0},
            "confidence": 0.9,
            "issues": [],
        })
        state = node(base_state("0"))
        self.assertNotIn("property_value", state["applicant_profile"])

    def test_negative_employment_years_rejected(self) -> None:
        node = make_node({
            "fields": {"employment_years": -1},
            "confidence": 0.9,
            "issues": [],
        })
        state = node(base_state("-1"))
        self.assertNotIn("employment_years", state["applicant_profile"])

    def test_zero_employment_years_is_valid(self) -> None:
        # brand-new hires (0 years) should be accepted by validation - the
        # eligibility RULE may still fail it, but it's not an invalid value
        node = make_node({
            "fields": {"employment_years": 0},
            "confidence": 0.9,
            "issues": [],
        })
        state = node(base_state("just started"))
        self.assertEqual(state["applicant_profile"]["employment_years"], 0.0)

    def test_zero_monthly_debt_is_valid(self) -> None:
        node = make_node({
            "fields": {"monthly_debt": 0},
            "confidence": 0.9,
            "issues": [],
        })
        state = node(base_state("no debt at all"))
        self.assertEqual(state["applicant_profile"]["monthly_debt"], 0.0)


class TestExtractionConflictDetection(unittest.TestCase):
    def test_conflicting_answer_is_logged_but_overwrites(self) -> None:
        node = make_node({
            "fields": {"annual_income": 60000},
            "confidence": 0.9,
            "issues": [],
        })
        state = base_state("actually it's 60000")
        state["applicant_profile"] = {"annual_income": 50000.0}
        result = node(state)
        self.assertEqual(result["applicant_profile"]["annual_income"], 60000.0)
        self.assertTrue(any("Conflict" in c for c in result["profile_conflicts"]))


if __name__ == "__main__":
    unittest.main()