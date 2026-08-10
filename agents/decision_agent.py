from __future__ import annotations

from typing import Any

from core.schemas import REQUIRED_FIELDS


class DecisionAgent:
    def __init__(self, rule_evaluator: Any) -> None:
        self.rule_evaluator = rule_evaluator
        # Tunable thresholds (defaults can be overridden by caller)
        self.early_offer_pass_ratio = 0.85
        self.early_auto_pass_ratio = 1.0

    def set_early_termination_thresholds(self, offer_ratio: float, auto_ratio: float) -> None:
        self.early_offer_pass_ratio = float(offer_ratio)
        self.early_auto_pass_ratio = float(auto_ratio)

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        profile = state.get("applicant_profile", {})
        missing = [field for field in REQUIRED_FIELDS if field not in profile]
        max_turns = int(state.get("max_turns", 8))
        turn_count = int(state.get("turn_count", 0))

        if state.get("user_requested_finalize"):
            if missing:
                state["followup_field"] = None
                state["needs_followup"] = False
                state["decision_status"] = "Requires More Info"
                state["decision_summary"] = "User indicated no more information is available."
                state["final_report"] = {
                    "status": state["decision_status"],
                    "summary": state["decision_summary"],
                    "missing_fields": missing,
                }
                state["offer_early_termination"] = False
                state["auto_terminated"] = False
                return state

            report = self.rule_evaluator.evaluate(profile)
            state["followup_field"] = None
            state["needs_followup"] = False
            state["decision_status"] = report["status"]
            state["decision_summary"] = report["summary"]
            state["final_report"] = report
            state["offer_early_termination"] = False
            state["auto_terminated"] = False
            return state

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
        # If the user explicitly confirmed early termination, finalize now
        if state.get("user_confirmed_early_end"):
            state["needs_followup"] = False
            state["followup_field"] = None
            return state
        # Early-termination logic: compute pass ratio and set flags
        try:
            passed_count = sum(1 for r in report.get("rule_breakdown", []) if r.get("passed"))
            total = max(1, len(report.get("rule_breakdown", [])))
            pass_ratio = passed_count / total
        except Exception:
            pass_ratio = 0.0

        if pass_ratio >= self.early_auto_pass_ratio:
            # Auto-complete: no further followup
            state["offer_early_termination"] = False
            state["auto_terminated"] = True
            state["needs_followup"] = False
        elif pass_ratio >= self.early_offer_pass_ratio:
            # Offer early termination to the user (confirmation handled by InterviewAgent)
            state["offer_early_termination"] = True
            state["auto_terminated"] = False
            state["needs_followup"] = True
            state["early_termination_pass_ratio"] = pass_ratio
        else:
            state["offer_early_termination"] = False
            state["auto_terminated"] = False
        return state
