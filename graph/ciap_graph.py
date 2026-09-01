from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class CIAPState(TypedDict, total=False):
    session_id: str
    model_id: str
    conversation_history: list[dict[str, str]]
    applicant_profile: dict[str, Any]
    profile_conflicts: list[str]
    current_question: str
    current_question_field: str
    # Conversational memory
    asked_fields: list[str]
    answered_fields: list[str]
    question_history: list[dict[str, str]]
    latest_user_response: str
    turn_count: int
    max_turns: int
    followup_field: str | None
    needs_followup: bool
    decision_status: str
    decision_summary: str
    final_report: dict[str, Any]
    last_extraction: dict[str, Any]
    user_requested_stop: bool
    user_requested_finalize: bool
    offer_early_termination: bool
    user_confirmed_early_end: bool
    early_offered_already: bool
    auto_terminated: bool
    early_termination_pass_ratio: float
    skip_extraction: bool
    greeting_detected: bool
    lead_step: str
    summary: str
    qualification_category: str
    conversation_status: str
    session_tags: list[str]
    conversation_tag: str | None
    clarification_context: str
    skipped_fields: list[str]
    field_attempts: dict[str, int]
    deferred_reasons: dict[str, str]
    counted_invalid_response: str
    auto_deferred_field: str
    interpreted_input: dict[str, Any]
    recently_deferred_field: str
    home_purchase_context: dict[str, Any]
    recent_profile_corrections: list[str]

def build_ciap_graph(
    interview_agent: Any,
    extraction_validation_node: Any,
    decision_agent: Any,
    on_turn_complete: Any,
    on_completed: Any,
):
    graph = StateGraph(CIAPState)

    def interview_node(state: dict[str, Any]) -> dict[str, Any]:
        updated = interview_agent(state)
        logger.debug("Interview node result: user_requested_stop=%s user_requested_finalize=%s needs_followup=%s latest=%r",
            updated.get("user_requested_stop"), updated.get("user_requested_finalize"), updated.get("needs_followup"), updated.get("latest_user_response"))
        return updated

    def extraction_node(state: dict[str, Any]) -> dict[str, Any]:
        latest = str(state.get("latest_user_response", "")).strip().lower()
        stop_cmds = {"stop", "end", "/stop", "/end"}
        skip_extraction = bool(state.get("skip_extraction")) or state.get("user_requested_stop") or latest in stop_cmds
        if skip_extraction:
            logger.debug(
                "Skipping extraction: stop detected latest=%r user_requested_stop=%s skip_extraction=%s",
                latest,
                state.get("user_requested_stop"),
                state.get("skip_extraction"),
            )
            on_turn_complete(state)
            return state
        updated = extraction_validation_node(state)
        logger.debug("Extraction node result: needs_followup=%s current_question_field=%r", updated.get("needs_followup"), updated.get("current_question_field"))
        on_turn_complete(updated)
        return updated

    def decision_node(state: dict[str, Any]) -> dict[str, Any]:
        latest = str(state.get("latest_user_response", "")).strip().lower()
        stop_cmds = {"stop", "end", "/stop", "/end"}
        if state.get("user_requested_stop") or latest in stop_cmds:
            logger.debug("Skipping decision: stop detected latest=%r user_requested_stop=%s", latest, state.get("user_requested_stop"))
            on_completed(state)
            return state

        updated = decision_agent(state)
        logger.debug("Decision node result: needs_followup=%s decision_status=%r", updated.get("needs_followup"), updated.get("decision_status"))
        if not updated.get("needs_followup", False):
            on_completed(updated)
        return updated

    graph.add_node("interview", interview_node)
    graph.add_node("extract_validate", extraction_node)
    graph.add_node("decision", decision_node)

    graph.add_edge(START, "interview")
    graph.add_edge("interview", "extract_validate")
    graph.add_edge("extract_validate", "decision")

    def route_after_decision(state: dict[str, Any]) -> str:
        return "interview" if state.get("needs_followup", False) else END

    graph.add_conditional_edges("decision", route_after_decision, {"interview": "interview", END: END})
    return graph.compile()
