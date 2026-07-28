import json
import unittest

from agents.extraction_validation import ExtractionValidationNode
from core.context_builder import ContextBuilder
from core.profile_updater import ProfileUpdater
from core.schemas import EXTRACTION_SCHEMA


class StubLLM:
    def __init__(self, payload: dict):
        self.payload = payload

    def extract_structured(self, prompt: str, latest_response: str) -> str:
        return json.dumps(self.payload)


class TestExtractionValidationNode(unittest.TestCase):
    def test_valid_extraction_updates_profile(self) -> None:
        node = ExtractionValidationNode(
            llm_client=StubLLM(
                {
                    "fields": {
                        "annual_income": 90000,
                        "monthly_debt": 1200,
                        "credit_score": 710,
                        "employment_status": "employed",
                    },
                    "confidence": 0.95,
                    "issues": [],
                }
            ),
            context_builder=ContextBuilder(),
            profile_updater=ProfileUpdater(),
            extraction_schema=EXTRACTION_SCHEMA,
        )

        state = node(
            {
                "conversation_history": [],
                "applicant_profile": {},
                "current_question": "What is your annual income?",
                "latest_user_response": "My annual income is 90,000.",
            }
        )

        self.assertEqual(state["applicant_profile"]["annual_income"], 90000.0)
        self.assertEqual(state["applicant_profile"]["credit_score"], 710)

    def test_invalid_json_gracefully_handled(self) -> None:
        class BadStub:
            def extract_structured(self, prompt: str, latest_response: str) -> str:
                return "not-json"

        node = ExtractionValidationNode(
            llm_client=BadStub(),
            context_builder=ContextBuilder(),
            profile_updater=ProfileUpdater(),
            extraction_schema=EXTRACTION_SCHEMA,
        )

        state = node(
            {
                "conversation_history": [],
                "applicant_profile": {},
                "current_question": "What is your annual income?",
                "latest_user_response": "I earn well",
            }
        )

        self.assertEqual(state["applicant_profile"], {})
        self.assertIn("Extractor response was not valid JSON", state["last_extraction"]["issues"])


if __name__ == "__main__":
    unittest.main()
