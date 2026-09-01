from __future__ import annotations

import json

import pytest

from agents.extraction_validation import ExtractionValidationNode
from core.context_builder import ContextBuilder
from core.conversation_intent import Intent, classify_input
from core.profile_updater import ProfileUpdater
from core.schemas import EXTRACTION_SCHEMA


class EmptyExtractor:
    def extract_structured(self, prompt: str, latest_response: str) -> str:
        return json.dumps({"fields": {}, "confidence": 0.1, "issues": []})


def node() -> ExtractionValidationNode:
    return ExtractionValidationNode(
        EmptyExtractor(), ContextBuilder(), ProfileUpdater(), EXTRACTION_SCHEMA
    )


@pytest.mark.parametrize(
    "text",
    [
        "skip", "skip this", "skip this question", "pass", "next", "next question",
        "move next ques", "move to next quest", "ask the next question", "no pass",
        "shut up plz move on", "its not neg bro i just said move on",
        "bro please skip this question", "can we go to the next question",
    ],
)
def test_skip_language_matrix(text: str) -> None:
    assert classify_input(text).intent == Intent.SKIP


@pytest.mark.parametrize(
    "text",
    [
        "I don't know", "I do not know", "don't know", "no idea", "I have no idea",
        "I can't remember", "I cannot remember", "not sure", "not certain", "unknown",
        "no answer", "not remember", "not pretty sure", "not preety sure",
        "I couldn’t estimate my score right now", "I could not estimate my score right now",
    ],
)
def test_unknown_language_matrix(text: str) -> None:
    assert classify_input(text).intent == Intent.UNKNOWN


@pytest.mark.parametrize(
    "text",
    [
        "I don't want to tell", "I dont want to tell you", "I do not want to share",
        "I don't want to answer", "I'd rather not", "I would rather not say",
        "prefer not to answer", "prefer not to share", "none of your business",
    ],
)
def test_refusal_language_matrix(text: str) -> None:
    assert classify_input(text).intent == Intent.REFUSAL


@pytest.mark.parametrize(
    "text",
    [
        "stop", "end", "/stop", "/end", "shutup", "shut up", "please stop",
        "leave me alone", "end this conversation", "stop contacting me",
        "don't contact me anymore", "I don't want to continue", "no longer interested",
    ],
)
def test_stop_language_matrix(text: str) -> None:
    assert classify_input(text).intent == Intent.STOP


@pytest.mark.parametrize(
    "text",
    [
        "what do you mean", "which amount do you mean", "like what", "how much do you mean",
        "could you explain", "can you clarify", "what is debt payment", "what is a credit score",
        "what is annual income", "I'm not sure what you mean",
    ],
)
def test_clarification_language_matrix(text: str) -> None:
    assert classify_input(text).intent == Intent.CLARIFICATION


@pytest.mark.parametrize(
    ("field", "question", "text"),
    [
        ("down_payment", "How much can you put down?", "no"),
        ("down_payment", "How much can you put down?", "none"),
        ("down_payment", "How much can you put down?", "nothing"),
        ("down_payment", "How much can you put down?", "zero"),
        ("down_payment", "How much can you put down?", "no down payment"),
        ("down_payment", "How much can you put down?", "no amount"),
        ("down_payment", "How much can you put down?", "i have no amount"),
        ("down_payment", "How much can you put down?", "I've got no amount"),
        ("down_payment", "How much can you put down?", "i have no payment"),
        ("down_payment", "How much can you put down?", "nothing for the down payment"),
        ("down_payment", "How much can you put down?", "I have nothing available for a down payment"),
        ("monthly_debt", "How much debt do you pay?", "no debt"),
        ("monthly_debt", "How much debt do you pay?", "no debts"),
        ("monthly_debt", "How much debt do you pay?", "i dont pay debt"),
        ("monthly_debt", "How much debt do you pay?", "I do not pay any loans"),
        ("monthly_debt", "How much debt do you pay?", "I have no loans"),
        ("monthly_debt", "How much debt do you pay?", "nothing toward debt"),
        ("monthly_debt", "How much debt do you pay?", "I don’t make any debt payments"),
        ("monthly_debt", "How much debt do you pay?", "i dont pay toward debts"),
        ("monthly_debt", "How much debt do you pay?", "bro i tell u i dont pay toward debts"),
        ("monthly_debt", "How much debt do you pay?", "I do not owe any loans at all"),
        ("monthly_debt", "How much debt do you pay?", "I never make payments toward loans"),
        ("down_payment", "How much can you put down?", "I cannot put anything down"),
        ("total_savings", "How much do you have saved?", "I have not saved anything"),
        ("total_savings", "How much do you have saved?", "no savings"),
        ("total_savings", "How much do you have saved?", "nothing"),
    ],
)
def test_zero_answer_matrix(field: str, question: str, text: str) -> None:
    state = node()(
        {
            "conversation_history": [],
            "applicant_profile": {},
            "current_question": question,
            "current_question_field": field,
            "latest_user_response": text,
        }
    )
    assert state["applicant_profile"][field] == 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("emp", "employed"), ("employed", "employed"), ("I am employed", "employed"),
        ("yes i am emp", "employed"), ("working full time", "employed"),
        ("I work part-time", "employed"), ("self", "self-employed"),
        ("self employed", "self-employed"), ("I am self-employed", "self-employed"),
        ("freelancer", "self-employed"), ("I am freelancer", "self-employed"),
        ("business man", "self-employed"), ("business owner", "self-employed"),
        ("I own a business", "self-employed"), ("contractor", "self-employed"),
        ("unemployed", "unemployed"), ("not working anymore", "unemployed"),
        ("retired", "unemployed"), ("I am on pension", "unemployed"),
        ("I have no job", "unemployed"),
        ("I've been recently laid-off and hope to get back to work soon", "unemployed"),
        ("I was laid off last week", "unemployed"),
        ("I'm currently between jobs", "unemployed"),
        ("I am looking for work", "unemployed"),
        ("I recently lost my job", "unemployed"),
        ("I was made redundant", "unemployed"),
        ("I was terminated", "unemployed"),
    ],
)
def test_employment_language_matrix(text: str, expected: str) -> None:
    state = node()(
        {
            "conversation_history": [],
            "applicant_profile": {},
            "current_question": "Are you employed, self-employed, or between jobs?",
            "current_question_field": "employment_status",
            "latest_user_response": text,
        }
    )
    assert state["applicant_profile"]["employment_status"] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ninety thousand annually", 90000.0),
        ("I earn close to ninety thousand annually", 90000.0),
        ("one hundred twenty five thousand per year", 125000.0),
        ("two million", 2000000.0),
    ],
)
def test_written_income_amounts(text: str, expected: float) -> None:
    state = node()(
        {
            "conversation_history": [],
            "applicant_profile": {},
            "current_question": "What is your annual income?",
            "current_question_field": "annual_income",
            "latest_user_response": text,
        }
    )
    assert state["applicant_profile"]["annual_income"] == expected
