from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class CIAPState(TypedDict, total=False):
    session_id: str
    model_id: str
    conversation_history: list[dict[str, str]]
    applicant_profile: dict[str, Any]
    profile_conflicts: list[str]
    current_question: str
    current_question_field: str
    latest_user_response: str
    turn_count: int
    max_turns: int
    followup_field: str | None
    needs_followup: bool
    decision_status: str
    decision_summary: str
    final_report: dict[str, Any]
    last_extraction: dict[str, Any]


def build_ciap_graph(
    interview_agent: Any,
    extraction_validation_node: Any,
    decision_agent: Any,
    on_turn_complete: Any,
    on_completed: Any,
):
    graph = StateGraph(CIAPState)

    def interview_node(state: dict[str, Any]) -> dict[str, Any]:
        return interview_agent(state)

    def extraction_node(state: dict[str, Any]) -> dict[str, Any]:
        updated = extraction_validation_node(state)
        on_turn_complete(updated)
        return updated

    def decision_node(state: dict[str, Any]) -> dict[str, Any]:
        updated = decision_agent(state)
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
