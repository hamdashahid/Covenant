from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents.decision_agent import DecisionAgent
from agents.extraction_validation import ExtractionValidationNode
from core.context_builder import ContextBuilder
from core.profile_updater import ProfileUpdater
from core.schemas import EXTRACTION_SCHEMA
from core.session_manager import SessionManager
from persistence.sqlite_store import SQLiteStore
from rules.rule_evaluator import RuleEvaluator


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def rules_path(repo_root: Path) -> str:
    return str(repo_root / "rules" / "eligibility_rules.yaml")


@pytest.fixture
def evaluator(rules_path: str) -> RuleEvaluator:
    return RuleEvaluator(rules_path)


@pytest.fixture
def base_profile() -> dict[str, Any]:
    return {
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


@pytest.fixture
def temp_db_path(tmp_path) -> str:
    return str(tmp_path / "ciap_test.db")


@pytest.fixture
def session_store(temp_db_path: str) -> SQLiteStore:
    return SQLiteStore(temp_db_path)


@pytest.fixture
def session_manager(session_store: SQLiteStore) -> SessionManager:
    return SessionManager(session_store, default_model_id="test-model")


class StubLLM:
    def __init__(self, payloads: list[dict[str, Any]] | None = None, error: Exception | None = None) -> None:
        self.payloads = payloads or []
        self.error = error
        self.calls = 0

    def extract_structured(self, prompt: str, latest_response: str) -> str:
        if self.error is not None:
            raise self.error
        if self.calls >= len(self.payloads):
            payload = self.payloads[-1] if self.payloads else {"fields": {}, "confidence": 0.9, "issues": []}
        else:
            payload = self.payloads[self.calls]
        self.calls += 1
        return json.dumps(payload)


@pytest.fixture
def make_extraction_node() -> callable:
    def _build(payloads: list[dict[str, Any]] | None = None, error: Exception | None = None) -> ExtractionValidationNode:
        return ExtractionValidationNode(
            llm_client=StubLLM(payloads=payloads, error=error),
            context_builder=ContextBuilder(),
            profile_updater=ProfileUpdater(),
            extraction_schema=EXTRACTION_SCHEMA,
        )

    return _build


@pytest.fixture
def decision_agent() -> DecisionAgent:
    return DecisionAgent(rule_evaluator=RuleEvaluator("rules/eligibility_rules.yaml"))