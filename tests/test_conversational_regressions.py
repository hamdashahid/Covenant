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


def test_down_payment_no_hurdle_gets_natural_amount_followup(monkeypatch) -> None:
    questions: list[str] = []
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "100000")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(
        interview_agent_module.terminal_ui,
        "print_agent_message",
        lambda question, **kwargs: questions.append(question),
    )
    agent = InterviewAgent(
        [{"field": "down_payment", "question": "Is a down payment your biggest hurdle right now? If so, roughly how much are you able to put down?"}],
        "sys",
    )
    state = {
        "conversation_history": [
            {"role": "assistant", "content": "Is a down payment your biggest hurdle right now? If so, roughly how much are you able to put down?"},
            {"role": "user", "content": "no hurdle"},
        ],
        "applicant_profile": {},
        "latest_user_response": "no hurdle",
        "followup_field": "down_payment",
    }

    updated = agent(state)

    assert "isn't your main obstacle" in questions[0]
    assert "how much" in questions[0].lower()
    assert questions[0].count("?") == 1
    assert updated["current_question_field"] == "down_payment"


def test_down_payment_hurdle_answer_asks_only_for_amount(monkeypatch) -> None:
    questions: list[str] = []
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "around 30k")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda q, **kwargs: questions.append(q))
    agent = InterviewAgent([{"field": "down_payment", "question": "Is a down payment your biggest hurdle? If so, how much?"}], "sys")
    state = {
        "conversation_history": [{"role": "user", "content": "yes it is hurdle"}],
        "applicant_profile": {},
        "latest_user_response": "yes it is hurdle",
        "followup_field": "down_payment",
    }

    agent(state)

    assert "can be difficult" in questions[0]
    assert "how much" in questions[0].lower()
    assert questions[0].count("?") == 1


def test_invalid_credit_score_explains_valid_range(monkeypatch) -> None:
    questions: list[str] = []
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "720")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda q, **kwargs: questions.append(q))
    agent = InterviewAgent([{"field": "credit_score", "question": "What is your credit score?"}], "sys")
    state = {
        "conversation_history": [{"role": "user", "content": "0"}],
        "applicant_profile": {},
        "latest_user_response": "0",
        "followup_field": "credit_score",
        "last_extraction": {"issues": ["credit_score must be between 300 and 850"]},
    }

    agent(state)

    assert "between 300 and 850" in questions[0]


def test_move_next_question_defers_current_field(monkeypatch) -> None:
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "move next ques")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda *args, **kwargs: None)
    agent = InterviewAgent([{"field": "credit_score", "question": "What is your credit score?"}], "sys")

    updated = agent({"conversation_history": [], "applicant_profile": {}})

    assert updated["skip_extraction"] is True
    assert updated["skipped_fields"] == ["credit_score"]
    assert updated["followup_field"] is None


def test_move_next_quest_typo_defers_current_field(monkeypatch) -> None:
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "move to next quest")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda *args, **kwargs: None)
    agent = InterviewAgent([{"field": "monthly_debt", "question": "What are your monthly debt payments?"}], "sys")
    updated = agent({"conversation_history": [], "applicant_profile": {"annual_income": 99000}})
    assert updated["skip_extraction"] is True
    assert updated["skipped_fields"] == ["monthly_debt"]


@pytest.mark.parametrize(
    "answer",
    ["i dont want to tell", "i don't want to tell u", "no idea", "not remember", "no pass"],
)
def test_refusal_and_unknown_phrases_defer_field(monkeypatch, answer: str) -> None:
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: answer)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda *args, **kwargs: None)
    agent = InterviewAgent([{"field": "monthly_debt", "question": "What are your monthly debt payments?"}], "sys")
    updated = agent({"conversation_history": [], "applicant_profile": {"annual_income": 80000}})
    assert updated["skip_extraction"] is True
    assert updated["skipped_fields"] == ["monthly_debt"]


def test_decision_routes_past_a_deferred_field() -> None:
    evaluator = type("Evaluator", (), {"rules": {}, "evaluate": lambda self, profile: {"status": "Requires More Info", "summary": "", "rule_breakdown": []}})()
    agent = DecisionAgent(evaluator)

    updated = agent(
        {
            "applicant_profile": {"down_payment": 30000},
            "skipped_fields": ["credit_score"],
            "turn_count": 2,
            "max_turns": 16,
        }
    )

    assert updated["needs_followup"] is True
    assert updated["followup_field"] == "employment_status"


def test_short_affirmative_is_understood() -> None:
    agent = InterviewAgent([], "sys")
    assert "ye" in agent.AFFIRMATIVE
    assert "yep" in agent.AFFIRMATIVE


def test_fallback_question_uses_collected_context() -> None:
    agent = InterviewAgent([{"field": "credit_score", "question": "Do you know your approximate credit score?"}], "sys")

    question = agent._fallback_question(
        "credit_score",
        {"applicant_profile": {"down_payment": 30000}, "latest_user_response": "around 30k"},
        False,
    )

    assert "30,000" in question
    assert question.count("?") == 1


def test_generated_response_does_not_invent_currency() -> None:
    agent = InterviewAgent([], "sys")
    cleaned = agent._sanitize_generated_response(
        "Great, $30k sounds good. What is your credit score?",
        {"latest_user_response": "around 30k"},
        False,
    )
    assert "$" not in cleaned
    assert "30k" in cleaned


def test_opening_is_reduced_to_one_interview_question() -> None:
    agent = InterviewAgent([], "sys")
    cleaned = agent._sanitize_generated_response(
        "Hi, I'm Alex. Is this your first home, and are you buying on your own or with someone?",
        {},
        True,
    )
    assert cleaned.endswith("Will this be your first home?")
    assert "buying" not in cleaned.lower()
    assert cleaned.count("?") == 1


def test_generated_response_keeps_only_one_question() -> None:
    agent = InterviewAgent([], "sys")
    cleaned = agent._sanitize_generated_response(
        "That's helpful. How are you doing today? What is your approximate credit score?",
        {"latest_user_response": "30k"},
        False,
    )
    assert "How are you" not in cleaned
    assert cleaned.endswith("What is your approximate credit score?")
    assert cleaned.count("?") == 1


def test_generated_reply_without_a_question_is_rejected() -> None:
    agent = InterviewAgent([], "sys")
    assert not agent._response_matches_target(
        "You may want to check with your bank and let me know.",
        "credit_score",
        False,
    )


def test_generated_question_must_match_selected_topic() -> None:
    agent = InterviewAgent([], "sys")
    assert not agent._response_matches_target(
        "Just to confirm, have you been self-employed for 3 years?",
        "annual_income",
        False,
    )
    assert agent._response_matches_target(
        "What is your annual income from self-employment?",
        "annual_income",
        False,
    )


def test_no_to_first_home_question_does_not_finalize(monkeypatch) -> None:
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "no")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda *args, **kwargs: None)
    agent = InterviewAgent(
        [{"field": "down_payment", "question": "How much can you put down?"}],
        "sys",
        llm_client=None,
    )

    updated = agent({"conversation_history": [], "applicant_profile": {}})

    assert updated.get("user_requested_finalize") is not True
    assert updated["latest_user_response"] == "no"


def test_explicit_closing_phrase_still_finalizes() -> None:
    agent = InterviewAgent([], "sys")
    assert agent._detect_finalize_intent(
        "No, that's all the information I have",
        "Is there anything else you'd like to add?",
        None,
    )


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

