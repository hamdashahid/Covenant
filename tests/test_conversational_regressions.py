from __future__ import annotations

import json

import agents.interview_agent as interview_agent_module
import pytest

from agents.decision_agent import DecisionAgent
from agents.extraction_validation import ExtractionValidationNode
from agents.interview_agent import InterviewAgent
from core.context_builder import ContextBuilder
from core.conversation_intent import Intent, classify_input
from core.profile_updater import ProfileUpdater
from core.schemas import EXTRACTION_SCHEMA
from graph.ciap_graph import build_ciap_graph
from core.terminal_ui import _closing_transition, _conversational_message, print_summary


class Extractor:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def extract_structured(self, prompt: str, response: str) -> str:
        return json.dumps(self.payload)


class GeneratedReply:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate_reply(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        return self.reply


def make_node(payload: dict) -> ExtractionValidationNode:
    return ExtractionValidationNode(
        Extractor(payload), ContextBuilder(), ProfileUpdater(), EXTRACTION_SCHEMA
    )


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("retired officer", "retired"),
        ("I'm on pension", "retired"),
        ("not working anymore", "unemployed"),
        ("I used to be a bank officer, now retired", "retired"),
        ("unemployed due to retirement", "retired"),
    ],
)
def test_employment_language_preserves_retirement_meaning(answer: str, expected: str) -> None:
    state = {
        "current_question": "What is your current employment status?",
        "latest_user_response": answer,
        "applicant_profile": {},
    }
    updated = make_node({"fields": {}, "confidence": 0.1, "issues": []})(state)
    assert updated["applicant_profile"]["employment_status"] == expected
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


@pytest.mark.parametrize("answer", ["it's close to perfect", "excellent", "pretty good", "poor"])
def test_qualitative_credit_answer_is_understood_without_robotic_repetition(
    monkeypatch, answer: str
) -> None:
    questions: list[str] = []
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "780")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda q, **kwargs: questions.append(q))
    agent = InterviewAgent([{"field": "credit_score", "question": "What is your credit score?"}], "sys")
    state = {
        "conversation_history": [{"role": "user", "content": answer}],
        "applicant_profile": {},
        "latest_user_response": answer,
        "followup_field": "credit_score",
        "current_question_field": "credit_score",
        "last_extraction": {"issues": ["credit_score is unclear"]},
    }

    agent(state)

    assert answer not in questions[0]
    assert "general sense" in questions[0].lower()
    assert "approximate range" in questions[0].lower()
    assert questions[0].count("?") == 1


def test_down_payment_sentence_is_not_described_as_credit(monkeypatch) -> None:
    questions: list[str] = []
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "780")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda q, **kwargs: questions.append(q))
    agent = InterviewAgent(
        [
            {"field": "down_payment", "question": "How much can you put down?"},
            {"field": "credit_score", "question": "What is your approximate credit score?"},
        ],
        "sys",
    )
    state = {
        "conversation_history": [],
        "applicant_profile": {"down_payment": 0.0},
        "latest_user_response": "Yes, but I don't have a down payment.",
        "current_question_field": "down_payment",
        "followup_field": "credit_score",
        "last_extraction": {"issues": ["Extraction confidence too low"]},
    }

    agent(state)

    assert "general description of your credit" not in questions[0].lower()
    assert "credit score" in questions[0].lower()


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
    ["shut up plz move on", "its not neg bro i just said move on"],
)
def test_frustrated_move_on_language_skips_instead_of_looping(monkeypatch, answer: str) -> None:
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: answer)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda *args, **kwargs: None)
    agent = InterviewAgent(
        [
            {"field": "employment_years", "question": "How long have you worked there?"},
            {"field": "annual_income", "question": "What is your annual income?"},
        ],
        "sys",
    )

    updated = agent({"conversation_history": [], "applicant_profile": {"employment_status": "employed"}})

    assert updated.get("user_requested_stop") is not True
    assert updated["skipped_fields"] == ["employment_years"]
    assert updated["followup_field"] is None


def test_uncertain_credit_gets_one_helpful_range_prompt_before_deferral(monkeypatch) -> None:
    questions: list[str] = []
    replies = iter(["i am not sure", "780"])
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: next(replies))
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(
        interview_agent_module.terminal_ui,
        "print_agent_message",
        lambda question, **kwargs: questions.append(question),
    )
    agent = InterviewAgent(
        [{"field": "credit_score", "question": "What is your credit score?"}],
        "sys",
        llm_client=None,
    )

    first = agent({"conversation_history": [], "applicant_profile": {}})
    first_skips = list(first.get("skipped_fields", []))
    first_followup = first.get("followup_field")
    second = agent(first)

    assert first_skips == []
    assert first_followup == "credit_score"
    assert "below 650" in questions[1]
    assert second["latest_user_response"] == "780"


def test_unclear_employment_years_does_not_claim_answer_was_negative(monkeypatch) -> None:
    questions: list[str] = []
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "skip")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda q, **kwargs: questions.append(q))
    agent = InterviewAgent([{"field": "employment_years", "question": "How long have you worked there?"}], "sys")

    agent(
        {
            "conversation_history": [],
            "applicant_profile": {},
            "followup_field": "employment_years",
            "current_question_field": "employment_years",
            "latest_user_response": "something unrelated",
            "last_extraction": {"issues": ["employment_years is unclear"]},
        }
    )

    assert "couldn't identify" in questions[0].lower()
    assert "negative" not in questions[0].lower()


def test_job_details_are_not_described_as_negative_years(monkeypatch) -> None:
    questions: list[str] = []
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "5")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda q, **kwargs: questions.append(q))
    agent = InterviewAgent([{"field": "employment_years", "question": "How long have you worked there?"}], "sys")
    agent(
        {
            "conversation_history": [],
            "applicant_profile": {},
            "followup_field": "employment_years",
            "current_question_field": "employment_years",
            "latest_user_response": "60k full time law office of James d hunter",
            "last_extraction": {"issues": ["employment_years must be between 0 and 80"]},
        }
    )
    assert "understood the work details" in questions[0].lower()
    assert "negative" not in questions[0].lower()


def test_negative_monthly_debt_gets_accurate_guidance(monkeypatch) -> None:
    questions: list[str] = []
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "0")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda q, **kwargs: questions.append(q))
    agent = InterviewAgent([{"field": "monthly_debt", "question": "What is your monthly debt?"}], "sys")
    agent(
        {
            "conversation_history": [],
            "applicant_profile": {},
            "followup_field": "monthly_debt",
            "current_question_field": "monthly_debt",
            "latest_user_response": "-1",
            "last_extraction": {"issues": ["monthly_debt must be >= 0"]},
        }
    )
    assert "can't be negative" in questions[0].lower()
    assert "enter 0" in questions[0].lower()


def test_skipped_field_is_acknowledged_before_next_question(monkeypatch) -> None:
    questions: list[str] = []
    replies = iter(["move on", "710"])
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: next(replies))
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda q, **kwargs: questions.append(q))
    agent = InterviewAgent(
        [
            {"field": "down_payment", "question": "How much can you put down?"},
            {"field": "credit_score", "question": "What is your credit score?"},
        ],
        "sys",
    )
    state = agent({
        "conversation_history": [{"role": "assistant", "content": "How much can you put down?"}],
        "applicant_profile": {},
    })
    state = agent(state)
    assert "we'll skip the down payment" in questions[1].lower()
    assert "credit score" in questions[1].lower()
    assert "thanks, that helps" not in questions[1].lower()
    assert state["recently_deferred_field"] == ""


def test_skip_acknowledgement_is_not_repeated_on_later_turns(monkeypatch) -> None:
    questions: list[str] = []
    replies = iter(["move on", "710", "employed"])
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: next(replies))
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda q, **kwargs: questions.append(q))
    agent = InterviewAgent(
        [
            {"field": "down_payment", "question": "How much can you put down?"},
            {"field": "credit_score", "question": "What is your credit score?"},
            {"field": "employment_status", "question": "What is your employment status?"},
        ],
        "sys",
    )
    state = agent({
        "conversation_history": [{"role": "assistant", "content": "How much can you put down?"}],
        "applicant_profile": {},
    })
    state = agent(state)
    state["applicant_profile"]["credit_score"] = 710
    state["answered_fields"] = ["credit_score"]
    state["followup_field"] = "employment_status"
    agent(state)

    assert "skip the down payment" in questions[1].lower()
    assert "skip the down payment" not in questions[2].lower()


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


@pytest.mark.parametrize("answer", ["shutup", "shut up", "please stop", "leave me alone"])
def test_hostile_or_explicit_stop_language_ends_conversation(monkeypatch, answer: str) -> None:
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: answer)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda *args, **kwargs: None)
    updated = InterviewAgent([{"field": "credit_score", "question": "What is your credit score?"}], "sys")(
        {"conversation_history": [], "applicant_profile": {}}
    )
    assert updated["user_requested_stop"] is True
    assert updated["needs_followup"] is False


@pytest.mark.parametrize(
    ("answer", "intent"),
    [
        ("skip", Intent.SKIP),
        ("I don't want to tell you", Intent.REFUSAL),
        ("no idea", Intent.UNKNOWN),
        ("what is debt payment?", Intent.CLARIFICATION),
        ("emp", Intent.ANSWER),
    ],
)
def test_central_input_classifier(answer: str, intent: Intent) -> None:
    assert classify_input(answer).intent == intent


def test_third_clarification_auto_defers_field(monkeypatch) -> None:
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "what do you mean?")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda *args, **kwargs: None)
    agent = InterviewAgent([{"field": "monthly_debt", "question": "What are your monthly debt payments?"}], "sys")
    state = {"conversation_history": [], "applicant_profile": {}, "field_attempts": {"monthly_debt": 2}}

    updated = agent(state)

    assert updated["skipped_fields"] == ["monthly_debt"]
    assert updated["deferred_reasons"]["monthly_debt"] == "too_many_clarification_requests"
    assert updated["followup_field"] is None


def test_third_invalid_answer_moves_to_next_field(monkeypatch) -> None:
    questions: list[str] = []
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "employed")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda q, **kwargs: questions.append(q))
    agent = InterviewAgent(
        [
            {"field": "credit_score", "question": "What is your credit score?"},
            {"field": "employment_status", "question": "What is your employment status?"},
        ],
        "sys",
    )
    state = {
        "conversation_history": [],
        "applicant_profile": {},
        "followup_field": "credit_score",
        "current_question_field": "credit_score",
        "latest_user_response": "-2",
        "turn_count": 3,
        "last_extraction": {"issues": ["credit_score must be between 300 and 850"]},
        "field_attempts": {"credit_score": 2},
    }

    updated = agent(state)

    assert "credit_score" in updated["skipped_fields"]
    assert updated["current_question_field"] == "employment_status"
    assert "leave the credit score unanswered" in questions[0].lower()


def test_yes_to_credit_prompt_asks_for_score_without_extraction(monkeypatch) -> None:
    questions: list[str] = []
    replies = iter(["yes i know", "720"])
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: next(replies))
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda q, **kwargs: questions.append(q))
    agent = InterviewAgent([{"field": "credit_score", "question": "What is your approximate credit score?"}], "sys")

    first = agent({"conversation_history": [], "applicant_profile": {}})
    first_skipped_extraction = first["skip_extraction"]
    second = agent(first)

    assert first_skipped_extraction is True
    assert "what is your approximate credit score" in questions[1].lower()
    assert second["skip_extraction"] is False
    assert second["latest_user_response"] == "720"


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


def test_fallback_question_does_not_repeat_collected_number() -> None:
    agent = InterviewAgent([{"field": "credit_score", "question": "Do you know your approximate credit score?"}], "sys")

    question = agent._fallback_question(
        "credit_score",
        {"applicant_profile": {"down_payment": 30000}, "latest_user_response": "around 30k"},
        False,
    )

    assert "30,000" not in question
    assert "noted" not in question.lower()
    assert question.count("?") == 1


def test_generated_response_echoing_latest_number_is_rejected() -> None:
    agent = InterviewAgent([], "sys")
    cleaned = agent._sanitize_generated_response(
        "I've noted 30,000. What is your approximate credit score?",
        {"latest_user_response": "around 30,000"},
        False,
        {"allow_value_echo": False},
    )
    assert cleaned == ""


def test_repeated_robotic_opener_is_removed() -> None:
    agent = InterviewAgent([], "sys")
    cleaned = agent._sanitize_generated_response(
        "Thanks for sharing that. What is your approximate credit score?",
        {
            "latest_user_response": "30k",
            "conversation_history": [
                {"role": "assistant", "content": "Thanks for sharing that. How much can you put down?"}
            ],
        },
        False,
        {"allow_value_echo": False},
    )
    assert cleaned == "What is your approximate credit score?"


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        (
            "That's exciting! How much are you planning to put down as a down payment?",
            "How much are you planning to put down as a down payment?",
        ),
        (
            "Being a freelancer is great. How long have you been freelancing?",
            "How long have you been freelancing?",
        ),
        (
            "Four years sounds like solid experience. What is your annual income?",
            "What is your annual income?",
        ),
        (
            "Thanks for sharing your income details. What are your monthly debt payments?",
            "What are your monthly debt payments?",
        ),
    ],
)
def test_routine_generated_praise_is_removed(generated: str, expected: str) -> None:
    agent = InterviewAgent([], "sys")
    cleaned = agent._sanitize_generated_response(
        generated,
        {"latest_user_response": "routine answer"},
        False,
        {"mode": "contextual_transition", "allow_value_echo": False},
    )
    assert cleaned == expected


@pytest.mark.parametrize(
    ("target", "previous", "generated", "expected"),
    [
        (
            "credit_score",
            "down_payment",
            "No worries about the down payment. Could you let me know your credit score?",
            "Could you let me know your credit score?",
        ),
        (
            "employment_years",
            "employment_status",
            "Being a freelancer is quite flexible! How long have you been doing freelance work?",
            "How long have you been doing freelance work?",
        ),
        (
            "annual_income",
            "employment_years",
            "Freelancing for four years is solid experience. Could you share your annual income?",
            "Could you share your annual income?",
        ),
    ],
)
def test_next_missing_field_is_not_mistaken_for_clarification(
    target: str, previous: str, generated: str, expected: str
) -> None:
    agent = InterviewAgent(
        [{"field": target, "question": f"What is your {target}?"}],
        "sys",
        llm_client=GeneratedReply(generated),
    )
    state = {
        "conversation_history": [{"role": "user", "content": "routine answer"}],
        "applicant_profile": {},
        "latest_user_response": "routine answer",
        "current_question_field": previous,
        "followup_field": target,
        "last_extraction": {"issues": []},
        "turn_count": 2,
    }

    question = agent._generate_conversational_question(
        state, [target], target, is_first_turn=False
    )

    assert question == expected


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


@pytest.mark.parametrize(
    "opening_answer",
    [
        "Yes",
        "Yes,",
        "Yes,\\",
        "yes it will be",
        "yes it will be inshallah",
        "yes it is",
        "yes first home",
        "yes my first property",
        "No",
        "nope",
        "no it won't be",
        "no it is not",
        "no its my second home",
        "it will be my second home",
        "this is my 2nd property",
        "not my first house",
    ],
)
def test_first_home_answer_advances_to_down_payment_without_skipping(
    monkeypatch, opening_answer: str
) -> None:
    questions: list[str] = []
    replies = iter([opening_answer, "30k"])
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: next(replies))
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda q, **kwargs: questions.append(q))
    agent = InterviewAgent(
        [{"field": "down_payment", "question": "How much can you put down?"}],
        "sys",
        llm_client=None,
    )

    first = agent({"conversation_history": [], "applicant_profile": {}})
    first_skipped_fields = list(first.get("skipped_fields", []))
    second = agent(first)

    assert first_skipped_fields == []
    assert second["current_question_field"] == "down_payment"
    assert second["latest_user_response"] == "30k"
    assert "how much" in questions[1].lower()
    assert second["home_purchase_context"]["raw_answer"] == opening_answer


def test_opening_answer_with_down_payment_information_is_still_extracted(monkeypatch) -> None:
    monkeypatch.setattr(
        interview_agent_module.terminal_ui,
        "get_answer_prompt",
        lambda: "Yes, but I don't have a down payment.",
    )
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_agent_message", lambda *args, **kwargs: None)
    agent = InterviewAgent(
        [{"field": "down_payment", "question": "How much can you put down?"}],
        "sys",
        llm_client=None,
    )

    updated = agent({"conversation_history": [], "applicant_profile": {}})

    assert updated["skip_extraction"] is False
    assert updated["current_question_field"] == "down_payment"


def test_unclear_opening_answer_reasks_first_home_without_touching_down_payment(monkeypatch) -> None:
    questions: list[str] = []
    replies = iter(["abcd", "yes", "30k"])
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: next(replies))
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(
        interview_agent_module.terminal_ui,
        "print_agent_message",
        lambda question, **kwargs: questions.append(question),
    )
    agent = InterviewAgent(
        [{"field": "down_payment", "question": "How much can you put down?"}],
        "sys",
        llm_client=None,
    )

    first = agent({"conversation_history": [], "applicant_profile": {}})
    first_pending = first["opening_context_pending"]
    first_skipped = list(first.get("skipped_fields", []))
    first_profile = dict(first["applicant_profile"])
    second = agent(first)
    second_home_context = dict(second["home_purchase_context"])
    third = agent(second)

    assert first_pending is True
    assert first_skipped == []
    assert "down_payment" not in first_profile
    assert "first home" in questions[1].lower()
    assert "down payment" not in questions[1].lower()
    assert second_home_context["is_first_home"] is True
    assert third["current_question_field"] == "down_payment"
    assert "how much" in questions[2].lower()


def test_two_unclear_opening_answers_continue_without_looping_or_skipping_down_payment(monkeypatch) -> None:
    questions: list[str] = []
    replies = iter(["abcd", "something random", "30k"])
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: next(replies))
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(
        interview_agent_module.terminal_ui,
        "print_agent_message",
        lambda question, **kwargs: questions.append(question),
    )
    agent = InterviewAgent(
        [{"field": "down_payment", "question": "How much can you put down?"}],
        "sys",
        llm_client=None,
    )

    first = agent({"conversation_history": [], "applicant_profile": {}})
    second = agent(first)
    second_pending = second["opening_context_pending"]
    second_home_context = dict(second["home_purchase_context"])
    second_skipped = list(second.get("skipped_fields", []))
    third = agent(second)

    assert second_pending is False
    assert second_home_context["is_first_home"] is None
    assert second_skipped == []
    assert third["current_question_field"] == "down_payment"
    assert "how much" in questions[2].lower()


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
    assert final["applicant_profile"]["employment_status"] == "retired"


def test_retired_applicant_skips_job_years_and_is_asked_current_retirement_income(monkeypatch) -> None:
    questions: list[str] = []
    monkeypatch.setattr(interview_agent_module.terminal_ui, "get_answer_prompt", lambda: "90000")
    monkeypatch.setattr(interview_agent_module.terminal_ui, "print_thinking", lambda: None)
    monkeypatch.setattr(
        interview_agent_module.terminal_ui,
        "print_agent_message",
        lambda question, **kwargs: questions.append(question),
    )
    agent = InterviewAgent(
        [
            {"field": "employment_years", "question": "How long have you worked there?"},
            {"field": "annual_income", "question": "What is your annual income?"},
        ],
        "sys",
        llm_client=None,
    )

    updated = agent(
        {
            "conversation_history": [{"role": "user", "content": "retired officer"}],
            "applicant_profile": {"employment_status": "retired"},
            "followup_field": "employment_years",
            "current_question_field": "employment_status",
        }
    )

    assert updated["current_question_field"] == "annual_income"
    assert questions[0].lower().startswith("since you're retired")
    assert "pensions, investments, or other sources" in questions[0].lower()
    assert "previous" not in questions[0].lower()


def test_unemployed_result_mentions_current_status_not_employment_history() -> None:
    message, _ = _conversational_message(
        "Ineligible",
        {
            "rule_breakdown": [
                {"name": "Employment Status", "passed": False, "evaluation_status": "failed"}
            ]
        },
    )
    assert "current employment status" in message
    assert "employment history" not in message


def test_zero_down_payment_result_is_specific_and_recognizes_strengths() -> None:
    message, _ = _conversational_message(
        "Ineligible",
        {
            "rule_breakdown": [
                {"name": "Annual Income", "passed": True},
                {"name": "Credit Score", "passed": True},
                {"name": "Debt-to-Income Ratio", "passed": True},
                {"name": "Down Payment", "passed": False},
            ]
        },
        {
            "down_payment": 0.0,
            "monthly_debt": 0.0,
            "employment_status": "self-employed",
        },
    )

    assert "credit profile looks encouraging" in message.lower()
    assert "no monthly debt payments" in message.lower()
    assert "requires an amount greater than zero" in message.lower()
    assert "a bit below" not in message.lower()


def test_retired_summary_uses_contextual_label_and_clean_money(capsys) -> None:
    print_summary(
        {
            "down_payment": 0.0,
            "credit_score": 800,
            "employment_status": "retired",
            "annual_income": 900000.0,
            "monthly_debt": 0.0,
        }
    )

    output = capsys.readouterr().out
    assert "Current Yearly Income" in output
    assert "900,000" in output
    assert "900000.0" not in output
    assert "Years at Current Job" not in output
    assert "Total Savings" not in output


@pytest.mark.parametrize(
    ("state", "target", "expected"),
    [
        (
            {
                "current_question_field": "opening_context",
                "home_purchase_context": {"is_first_home": True},
            },
            "down_payment",
            "buying your first home is a big step",
        ),
        (
            {
                "current_question_field": "credit_score",
                "applicant_profile": {"credit_score": 830},
            },
            "employment_status",
            "strong credit position",
        ),
        (
            {
                "current_question_field": "annual_income",
                "applicant_profile": {"annual_income": 120000},
            },
            "monthly_debt",
            "strong income base",
        ),
        (
            {
                "current_question_field": "annual_income",
                "applicant_profile": {"annual_income": 20000},
            },
            "monthly_debt",
            "clear starting point",
        ),
    ],
)
def test_value_aware_transition_reacts_to_meaning_without_echoing_value(
    state: dict, target: str, expected: str
) -> None:
    transition = InterviewAgent([], "sys")._value_aware_transition(state, target)

    assert expected in transition.lower()
    assert not any(character.isdigit() for character in transition)


def test_value_aware_transition_is_neutral_on_routine_middle_income() -> None:
    transition = InterviewAgent([], "sys")._value_aware_transition(
        {
            "current_question_field": "annual_income",
            "applicant_profile": {"annual_income": 70000},
        },
        "monthly_debt",
    )

    assert transition == ""


def test_vague_down_payment_is_clarified_instead_of_recorded_as_zero() -> None:
    class IncorrectZeroInterpreter:
        def understand_turn(self, *args, **kwargs):
            return {
                "intent": "answer",
                "value": 0,
                "fields": {"down_payment": 0},
                "confidence": 0.9,
            }

    agent = InterviewAgent([], "sys", llm_client=IncorrectZeroInterpreter())
    intent, understood = agent._interpret_input(
        {},
        "down_payment",
        "How much can you put down?",
        "I have not that much down payment",
    )

    assert intent == Intent.CLARIFICATION
    assert understood["value"] is None
    assert understood["fields"] == {}
    assert understood["reason"] == "vague monetary amount"


def test_credit_requirement_question_answers_before_reasking() -> None:
    question = InterviewAgent([], "sys")._clarifying_question("credit_score")

    assert "at least 650" in question
    assert question.endswith("?")


def test_monthly_debt_explanation_answers_user_and_reasks_amount() -> None:
    question = InterviewAgent([], "sys")._clarifying_question("monthly_debt")

    assert "credit cards" in question.lower()
    assert "personal loans" in question.lower()
    assert "say 0" in question.lower()
    assert "monthly total" in question.lower()
    assert question.endswith("?")


def test_closing_transition_acknowledges_zero_debt() -> None:
    transition = _closing_transition({"monthly_debt": 0.0})

    assert "no monthly debt payments" in transition.lower()
    assert "put everything together" in transition.lower()

