from __future__ import annotations

from typing import Any

from core.schemas import REQUIRED_FIELDS


CONFIDENCE_THRESHOLD = 0.7
# merge

class DecisionAgent:
    def __init__(self, rule_evaluator: Any) -> None:
        self.rule_evaluator = rule_evaluator

    def _low_confidence_field(self, state: dict[str, Any]) -> str | None:
        extraction = state.get("last_extraction", {})
        confidence = float(extraction.get("confidence", 0.0) or 0.0)
        field = str(state.get("current_question_field") or "").strip()
        if field and confidence < CONFIDENCE_THRESHOLD:
            return field
        return None

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        profile = state.get("applicant_profile", {})
        missing = [field for field in REQUIRED_FIELDS if field not in profile]
        max_turns = int(state.get("max_turns", 8))
        turn_count = int(state.get("turn_count", 0))
        low_confidence_field = self._low_confidence_field(state)
        needs_followup_field = low_confidence_field or (missing[0] if missing else None)

        # The extraction node is the single validation point, so the decision step
        # trusts the validated profile but still retries when the confidence score
        # for the most recent field is too low to be auditably reliable.
        if needs_followup_field and turn_count < max_turns:
            state["followup_field"] = needs_followup_field
            state["needs_followup"] = True
            state["decision_status"] = "Requires More Info"
            state["decision_summary"] = (
                f"Need a higher-confidence answer for {needs_followup_field}"
            )
            return state

        if needs_followup_field and turn_count >= max_turns:
            state["followup_field"] = None
            state["needs_followup"] = False
            state["decision_status"] = "Requires More Info"
            if low_confidence_field and missing:
                reason = (
                    f"missing {', '.join(missing)}; low confidence in {low_confidence_field}"
                )
            elif low_confidence_field:
                reason = f"low confidence in {low_confidence_field}"
            else:
                reason = f"missing {', '.join(missing)}"
            state["decision_summary"] = (
                "Max interview turns reached before collecting required validated data: "
                f"{reason}"
            )
            state["final_report"] = {
                "status": state["decision_status"],
                "summary": state["decision_summary"],
                "missing_fields": missing,
                "low_confidence_fields": [low_confidence_field] if low_confidence_field else [],
            }
            return state

        report = self.rule_evaluator.evaluate(profile)
        state["followup_field"] = None
        state["needs_followup"] = False
        state["decision_status"] = report["status"]
        state["decision_summary"] = report["summary"]
        state["final_report"] = report
        return state
