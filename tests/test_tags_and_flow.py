from __future__ import annotations

import os
import tempfile

from persistence.sqlite_store import SQLiteStore
from agents.interview_agent import InterviewAgent
from agents.decision_agent import DecisionAgent
from graph.ciap_graph import build_ciap_graph


def test_replace_and_get_messages_with_tags(tmp_path):
    db_file = tmp_path / "test_core.db"
    store = SQLiteStore(db_path=str(db_file))
    session_id = "sess-tags-1"
    store.create_session(session_id, "model-x")
    msgs = [
        {"role": "assistant", "content": "Hello", "tags": ["greeting"]},
        {"role": "user", "content": "My income is 100000", "tags": ["income", "user-provided"]},
    ]
    store.replace_messages(session_id, msgs)
    got = store.get_messages(session_id)
    assert len(got) == 2
    assert got[0]["tags"] == ["greeting"]
    assert "income" in got[1]["tags"]


def test_interview_agent_stop_and_greeting(monkeypatch):
    # Monkeypatch terminal_ui prompt to simulate user input
    answers = ["hello", "stop"]

    def fake_input():
        return answers.pop(0)

    monkeypatch.setattr("core.terminal_ui.get_answer_prompt", lambda: fake_input())
    agent = InterviewAgent([], "system prompt", llm_client=None, greeting_text="Welcome!")
    state = {"conversation_history": []}
    out = agent(state)
    # First call should consume "hello" and store a question
    assert "current_question" in out
    # Second call should handle stop
    out2 = agent(out)
    assert out2.get("user_requested_stop") is True


def test_graph_stop_closes_session(monkeypatch):
    # Simulate a user typing 'stop' as their first response and ensure the graph finalizes
    monkeypatch.setattr("core.terminal_ui.get_answer_prompt", lambda: "stop")

    interview_agent = InterviewAgent([], "system prompt", llm_client=None)

    def extraction_node_should_not_be_called(state):
        raise AssertionError("Extraction should not be called after user stop")

    class DummyDecision:
        def __call__(self, state):
            raise AssertionError("Decision should not be called after user stop")

    completed = {}

    def on_turn_complete(s):
        completed["turn_saved"] = True

    def on_completed(s):
        # the interview_agent sets decision_status to 'Stopped by User'
        completed["closed_status"] = s.get("decision_status")

    graph = build_ciap_graph(
        interview_agent=interview_agent,
        extraction_validation_node=extraction_node_should_not_be_called,
        decision_agent=DummyDecision(),
        on_turn_complete=on_turn_complete,
        on_completed=on_completed,
    )

    state = {"session_id": "s-stop", "conversation_history": []}
    final = graph.invoke(state)
    assert completed.get("closed_status") == "Stopped by User"


def test_graph_finalize_phrase_leads_to_decision(monkeypatch):
    # Simulate a user typing a finalize phrase and ensure the graph proceeds to decision evaluation.
    responses = ["no, that's all the information I have"]
    monkeypatch.setattr("core.terminal_ui.get_answer_prompt", lambda: responses.pop(0))

    interview_agent = InterviewAgent(
        [],
        "system prompt",
        llm_client=None,
    )

    def extraction_node(state):
        state["last_extraction"] = {"issues": []}
        return state

    class DummyDecision:
        def __call__(self, state):
            assert state.get("user_requested_finalize") is True
            state["needs_followup"] = False
            state["decision_status"] = "Requires More Info"
            state["final_report"] = {
                "status": "Requires More Info",
                "summary": "User indicated no more information is available.",
            }
            return state

    completed = {}

    def on_turn_complete(s):
        completed["turn_saved"] = True

    def on_completed(s):
        completed["closed_status"] = s.get("decision_status")

    graph = build_ciap_graph(
        interview_agent=interview_agent,
        extraction_validation_node=extraction_node,
        decision_agent=DummyDecision(),
        on_turn_complete=on_turn_complete,
        on_completed=on_completed,
    )

    state = {"session_id": "s-finalize", "conversation_history": []}
    final = graph.invoke(state)
    assert final["decision_status"] == "Requires More Info"
    assert completed.get("closed_status") == "Requires More Info"


def test_graph_finalize_after_combined_multi_field_response(monkeypatch):
    # Simulate a single response providing multiple fields, then a finalize phrase.
    responses = [
        "Property value 500000, down payment 100000, loan amount 300000",
        "No, that's all the information I have",
    ]
    monkeypatch.setattr("core.terminal_ui.print_agent_message", lambda *args, **kwargs: None)
    monkeypatch.setattr("core.terminal_ui.print_thinking", lambda *args, **kwargs: None)
    monkeypatch.setattr("core.terminal_ui.get_answer_prompt", lambda: responses.pop(0))

    interview_agent = InterviewAgent(
        [
            {"field": "property_value", "question": "What is the property value?"},
            {"field": "requested_loan_amount", "question": "How much loan amount are you requesting?"},
            {"field": "down_payment", "question": "How much can you pay upfront as a down payment?"},
        ],
        "system prompt",
        llm_client=None,
    )

    class FakeEvaluator:
        def evaluate(self, profile):
            return {
                "status": "Ineligible",
                "summary": "Some summary",
                "rule_breakdown": [
                    {"name": f"R{i}", "passed": (i < 9), "value_display": "x", "threshold_display": "y"}
                    for i in range(10)
                ],
            }

    decision_agent = DecisionAgent(rule_evaluator=FakeEvaluator())
    decision_agent.set_early_termination_thresholds(0.5, 1.0)

    def extraction_node(state):
        latest = state.get("latest_user_response", "")
        if "Property value" in latest:
            state["applicant_profile"] = {
                **state.get("applicant_profile", {}),
                "property_value": 500000,
                "down_payment": 100000,
                "requested_loan_amount": 300000,
            }
        state["last_extraction"] = {"issues": []}
        return state

    completed = {}

    def on_turn_complete(s):
        completed["turns"] = completed.get("turns", 0) + 1

    def on_completed(s):
        completed["closed_status"] = s.get("decision_status")
        completed["final_report"] = s.get("final_report")

    graph = build_ciap_graph(
        interview_agent=interview_agent,
        extraction_validation_node=extraction_node,
        decision_agent=decision_agent,
        on_turn_complete=on_turn_complete,
        on_completed=on_completed,
    )

    state = {
        "session_id": "s-combined",
        "conversation_history": [],
        "applicant_profile": {
            "annual_income": 120000,
            "monthly_debt": 2000,
            "credit_score": 700,
            "employment_status": "employed",
            "employment_years": 5,
        },
        "turn_count": 0,
        "max_turns": 16,
    }

    final = graph.invoke(state)

    assert completed.get("closed_status") == "Ineligible"
    assert final["final_report"]["status"] == "Ineligible"
    assert final.get("user_requested_finalize") is True
    assert final.get("needs_followup") is False
    assert completed.get("turns") == 2


def test_early_termination_offer_and_confirm(monkeypatch):
    # create a decision agent with a fake evaluator that returns many passed rules
    class FakeEvaluator:
        def evaluate(self, profile):
            # 9 passed, 1 failed -> pass_ratio = 0.9
            return {
                "status": "Ineligible",
                "summary": "Some summary",
                "rule_breakdown": [{"name": f"R{i}", "passed": (i < 9)} for i in range(10)],
            }

    da = DecisionAgent(rule_evaluator=FakeEvaluator())
    da.set_early_termination_thresholds(0.5, 1.0)
    # Provide a complete profile so DecisionAgent proceeds to evaluation
    required_fields = [
        "annual_income",
        "monthly_debt",
        "credit_score",
        "employment_status",
        "employment_years",
        "property_value",
        "requested_loan_amount",
        "down_payment",
    ]
    state = {"applicant_profile": {f: 1 for f in required_fields}}
    updated = da(state)
    assert updated.get("offer_early_termination") is True

    # Now simulate InterviewAgent confirming the offer
    monkeypatch.setattr("core.terminal_ui.get_answer_prompt", lambda: "yes")
    ia = InterviewAgent([], "sys", llm_client=None)
    updated2 = ia(updated)
    # After confirmation, the interview agent should mark user_confirmed_early_end
    assert updated2.get("user_confirmed_early_end") is True or updated2.get("offer_early_termination") is False


def test_early_termination_offer_for_six_of_seven_rules():
    class FakeEvaluatorSixOfSeven:
        def evaluate(self, profile):
            return {
                "status": "Ineligible",
                "summary": "Almost there",
                "rule_breakdown": [
                    {"name": f"R{i}", "passed": (i < 6)} for i in range(7)
                ],
            }

    da = DecisionAgent(rule_evaluator=FakeEvaluatorSixOfSeven())
    da.set_early_termination_thresholds(0.85, 1.0)
    required_fields = [
        "annual_income",
        "monthly_debt",
        "credit_score",
        "employment_status",
        "employment_years",
        "property_value",
        "requested_loan_amount",
        "down_payment",
    ]
    state = {"applicant_profile": {f: 1 for f in required_fields}}
    updated = da(state)
    assert updated.get("offer_early_termination") is True
    assert updated.get("auto_terminated") is False


def test_greeting_is_not_treated_as_a_ciip_answer(monkeypatch):
    monkeypatch.setattr("core.terminal_ui.print_agent_message", lambda *args, **kwargs: None)
    monkeypatch.setattr("core.terminal_ui.print_thinking", lambda *args, **kwargs: None)
    monkeypatch.setattr("core.terminal_ui.get_answer_prompt", lambda: "Hi")

    interview_agent = InterviewAgent([], "system prompt", llm_client=None)

    def extraction_node_should_not_run(state):
        raise AssertionError("Greeting should not trigger extraction")

    graph = build_ciap_graph(
        interview_agent=interview_agent,
        extraction_validation_node=extraction_node_should_not_run,
        decision_agent=DecisionAgent(rule_evaluator=type("R", (), {"evaluate": lambda self, profile: {"status": "Requires More Info", "summary": "", "rule_breakdown": []}})()),
        on_turn_complete=lambda s: None,
        on_completed=lambda s: None,
    )

    final = graph.invoke({"session_id": "s-greeting", "conversation_history": []})
    assert final.get("current_question")
    assert final.get("applicant_profile", {}) == {}


def test_natural_opt_out_stops_the_flow(monkeypatch):
    monkeypatch.setattr("core.terminal_ui.print_agent_message", lambda *args, **kwargs: None)
    monkeypatch.setattr("core.terminal_ui.print_thinking", lambda *args, **kwargs: None)
    monkeypatch.setattr("core.terminal_ui.get_answer_prompt", lambda: "I don't want to continue this conversation.")

    interview_agent = InterviewAgent([], "system prompt", llm_client=None)

    def extraction_node_should_not_run(state):
        raise AssertionError("Stop intent should skip extraction")

    graph = build_ciap_graph(
        interview_agent=interview_agent,
        extraction_validation_node=extraction_node_should_not_run,
        decision_agent=DecisionAgent(rule_evaluator=type("R", (), {"evaluate": lambda self, profile: {"status": "Requires More Info", "summary": "", "rule_breakdown": []}})()),
        on_turn_complete=lambda s: None,
        on_completed=lambda s: None,
    )

    final = graph.invoke({"session_id": "s-stop", "conversation_history": []})
    assert final.get("user_requested_stop") is True
    assert final.get("decision_status") == "Stopped by User"


def test_deterministic_ineligibility_finalizes_without_more_questions():
    class FakeEvaluator:
        def evaluate(self, profile):
            return {
                "status": "Ineligible",
                "summary": "Income is too low",
                "rule_breakdown": [
                    {"name": "Annual Income", "passed": False, "value_display": "Rs 50000", "threshold_display": "minimum Rs 100000 required"},
                    {"name": "Credit Score", "passed": True, "value_display": "700", "threshold_display": "minimum 640 required"},
                ],
            }

    decision_agent = DecisionAgent(rule_evaluator=FakeEvaluator())
    state = {
        "applicant_profile": {
            "annual_income": 50000,
            "credit_score": 700,
        },
        "max_turns": 8,
        "turn_count": 0,
    }
    updated = decision_agent(state)
    assert updated.get("needs_followup") is False
    assert updated.get("decision_status") == "Ineligible"
    assert updated.get("final_report", {}).get("status") == "Ineligible"


def test_merge_session_tags_is_idempotent(tmp_path):
    db_file = tmp_path / "test_tags.db"
    store = SQLiteStore(db_path=str(db_file))
    session_id = "sess-tags-merge"
    store.create_session(session_id, "model-x")

    store.merge_session_tags(session_id, ["ciap-ready", "existing"])
    store.merge_session_tags(session_id, ["ciap-ready", "new-tag"])

    session = store.get_session(session_id)
    assert session["tags"] == ["ciap-ready", "existing", "new-tag"]
