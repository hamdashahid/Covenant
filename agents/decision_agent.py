from __future__ import annotations

from typing import Any

from core.schemas import REQUIRED_FIELDS


class DecisionAgent:
    def __init__(self, rule_evaluator: Any) -> None:
        self.rule_evaluator = rule_evaluator

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        profile = state.get("applicant_profile", {})
        missing = [field for field in REQUIRED_FIELDS if field not in profile]
        max_turns = int(state.get("max_turns", 8))
        turn_count = int(state.get("turn_count", 0))

        if missing and turn_count < max_turns:
            state["followup_field"] = missing[0]
            state["needs_followup"] = True
            state["decision_status"] = "Requires More Info"
            state["decision_summary"] = f"Need more validated information: missing {', '.join(missing)}"
            return state

        if missing and turn_count >= max_turns:
            state["followup_field"] = None
            state["needs_followup"] = False
            state["decision_status"] = "Requires More Info"
            state["decision_summary"] = (
                "Max interview turns reached before collecting required validated data"
            )
            state["final_report"] = {
                "status": state["decision_status"],
                "summary": state["decision_summary"],
                "missing_fields": missing,
            }
            return state

        report = self.rule_evaluator.evaluate(profile)
        state["followup_field"] = None
        state["needs_followup"] = False
        state["decision_status"] = report["status"]
        state["decision_summary"] = report["summary"]
        state["final_report"] = report
        return state
