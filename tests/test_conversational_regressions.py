from __future__ import annotations

import json

import agents.interview_agent as interview_agent_module
import pytest

from agents.decision_agent import DecisionAgent
from agents.extraction_validation import ExtractionValidationNode
from agents.interview_agent import InterviewAgent
from core.context_builder import ContextBuilder
from core.profile_updater import ProfileUpdater
from core.schemas import EXTRACTION_SCHEMA
from graph.ciap_graph import build_ciap_graph


class Extractor:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def extract_structured(self, prompt: str, response: str) -> str:
        return json.dumps(self.payload)


def make_node(payload: dict) -> ExtractionValidationNode:
    return ExtractionValidationNode(
        Extractor(payload), ContextBuilder(), ProfileUpdater(), EXTRACTION_SCHEMA
    )


@pytest.mark.parametrize(
    "answer",
    ["retired officer", "I'm on pension", "not working anymore", "I used to be a bank officer, now retired", "unemployed due to retirement"],
)
def test_explicit_retirement_language_is_persisted_as_unemployed(answer: str) -> None:
    state = {
        "current_question": "What is your current employment status?",
        "latest_user_response": answer,
        "applicant_profile": {},
    }
    updated = make_node({"fields": {}, "confidence": 0.1, "issues": []})(state)
    assert updated["applicant_profile"]["employment_status"] == "unemployed"
    assert "employment_status" in updated["answered_fields"]


@pytest.mark.parametrize(
    ("field", "clarification", "expected"),
    [
        ("annual_income", "what range do you mean?", "yearly income before tax"),
        ("credit_score", "like what?", "three-digit score"),
    ],
)
def test_clarifying_question_reasks_same_field_with_helpful_answer(
    monkeypatch, field: str, clarification: str, expected: str
) -> None:
    questions: list[str] = []
    replies = iter([clarification, "100000"])
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: next(replies))
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(
        interview_agent_module.terminal_ui,
        "print_agent_message",
        lambda question, **kwargs: questions.append(question),
    )
    agent = InterviewAgent([{ "field": field, "question": f"Provide your {field}." }], "sys")
    state = {"conversation_history": [], "applicant_profile": {}, "max_turns": 5}

    first = agent(state)
    assert first["skip_extraction"] is True
    assert first["followup_field"] == field
    # Simulate the graph's decision route-back; the follow-up remains this field.
    second = agent(first)
    assert questions[1] != questions[0]
    assert expected in questions[1].lower()
    assert second["skip_extraction"] is False


def test_greeting_skip_is_limited_to_one_turn(monkeypatch) -> None:
    replies = iter(["hi", "720"])
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: next(replies))
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda *args, **kwargs: None)
    agent = InterviewAgent([{ "field": "credit_score", "question": "What is your credit score?" }], "sys")
    state = agent({"conversation_history": [], "applicant_profile": {}})
    assert state["skip_extraction"] is True
    state["followup_field"] = "credit_score"
    state = agent(state)
    assert state["skip_extraction"] is False
    assert state["latest_user_response"] == "720"


def test_off_topic_answer_is_not_mistaken_for_a_clarifying_question(monkeypatch) -> None:
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "What is the weather today?")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda *args, **kwargs: None)
    agent = InterviewAgent([{ "field": "annual_income", "question": "What is your annual income?" }], "sys")
    state = agent({"conversation_history": [], "applicant_profile": {}})
    assert state["skip_extraction"] is False


def test_graph_persists_retired_status_and_routes_to_rule_evaluator(monkeypatch) -> None:
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "retired officer")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda *args, **kwargs: None)
    agent = InterviewAgent([{ "field": "employment_status", "question": "What is your current employment status?" }], "sys")
    evaluator = type("Evaluator", (), {"rules": {}, "evaluate": lambda self, profile: {"status": "Ineligible", "summary": "", "rule_breakdown": []}})()
    final = build_ciap_graph(agent, make_node({"fields": {}, "confidence": 0.1, "issues": []}), DecisionAgent(evaluator), lambda state: None, lambda state: None).invoke(
        {"conversation_history": [], "applicant_profile": {}, "max_turns": 1}
    )
    assert final["applicant_profile"]["employment_status"] == "unemployed"

