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

    def _deterministic_early_report(self, profile: dict[str, Any]) -> dict[str, Any] | None:
        def coerce_float(value: Any, default: float = 0.0) -> float:
            if value is None:
                return default
            if isinstance(value, bool):
                return float(value)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return default
                try:
                    return float(text.replace(",", ""))
                except (TypeError, ValueError, OverflowError):
                    return default
            return default

        def coerce_int(value: Any, default: int = 0) -> int:
            if value is None:
                return default
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return default
                try:
                    return int(text.replace(",", ""))
                except (TypeError, ValueError, OverflowError):
                    return default
            return default

        annual_income = coerce_float(profile.get("annual_income", 0))
        credit_score = coerce_int(profile.get("credit_score", 0))
        rules = getattr(self.rule_evaluator, "rules", {}) or {}
        income_threshold = float(rules.get("income_threshold", 0))
        min_credit_score = int(rules.get("min_credit_score", 0))

        if annual_income and credit_score:
            if annual_income < income_threshold and credit_score < min_credit_score:
                return {
                    "status": "Ineligible",
                    "eligible": False,
                    "summary": "Income and credit score are below the minimum thresholds, so no further questions can change the decision.",
                    "failed_rules": ["Annual Income", "Credit Score"],
                    "rule_breakdown": [
                        {"name": "Annual Income", "passed": False, "value_display": f"Rs {annual_income:,.0f}", "threshold_display": f"minimum Rs {income_threshold:,.0f} required"},
                        {"name": "Credit Score", "passed": False, "value_display": str(credit_score), "threshold_display": f"minimum {min_credit_score} required"},
                    ],
                    "metrics": {
                        "annual_income": annual_income,
                        "credit_score": credit_score,
                    },
                }
        return None

    def _derive_session_tags(self, report: dict[str, Any], state: dict[str, Any]) -> list[str]:
        if state.get("user_requested_stop"):
            return ["ciap-stopped"]
        if report.get("status") == "Eligible":
            return ["ciap-ready"]
        if report.get("status") == "Ineligible":
            return ["ciap-follow-up"]
        return []

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
                state["lead_step"] = "finalized"
                state["summary"] = state["decision_summary"]
                state["qualification_category"] = "requires_more_info"
                state["conversation_status"] = "in_progress"
                state["final_report"] = {
                    "status": state["decision_status"],
                    "summary": state["decision_summary"],
                    "missing_fields": missing,
                }
                state["offer_early_termination"] = False
                state["auto_terminated"] = False
                state["session_tags"] = self._derive_session_tags(state["final_report"], state)
                return state

            report = self.rule_evaluator.evaluate(profile)
            state["followup_field"] = None
            state["needs_followup"] = False
            state["decision_status"] = report["status"]
            state["decision_summary"] = report["summary"]
            state["lead_step"] = "finalized"
            state["summary"] = report["summary"]
            state["qualification_category"] = report["status"].lower()
            state["conversation_status"] = "completed"
            state["final_report"] = report
            state["offer_early_termination"] = False
            state["auto_terminated"] = False
            state["session_tags"] = self._derive_session_tags(report, state)
            return state

        deterministic_report = self._deterministic_early_report(profile)
        if deterministic_report is not None:
            state["followup_field"] = None
            state["needs_followup"] = False
            state["decision_status"] = deterministic_report["status"]
            state["decision_summary"] = deterministic_report["summary"]
            state["lead_step"] = "finalized"
            state["summary"] = deterministic_report["summary"]
            state["qualification_category"] = deterministic_report["status"].lower()
            state["conversation_status"] = "completed"
            state["final_report"] = deterministic_report
            state["offer_early_termination"] = False
            state["auto_terminated"] = True
            state["session_tags"] = self._derive_session_tags(deterministic_report, state)
            return state

        if missing and turn_count < max_turns:
            state["followup_field"] = missing[0]
            state["needs_followup"] = True
            state["decision_status"] = "Requires More Info"
            state["decision_summary"] = f"Need more validated information: missing {', '.join(missing)}"
            state["lead_step"] = "collecting"
            state["summary"] = state["decision_summary"]
            state["qualification_category"] = "requires_more_info"
            state["conversation_status"] = "in_progress"
            state["session_tags"] = []
            return state

        if missing and turn_count >= max_turns:
            state["followup_field"] = None
            state["needs_followup"] = False
            state["decision_status"] = "Requires More Info"
            state["decision_summary"] = (
                "Max interview turns reached before collecting required validated data"
            )
            state["lead_step"] = "finalized"
            state["summary"] = state["decision_summary"]
            state["qualification_category"] = "requires_more_info"
            state["conversation_status"] = "in_progress"
            state["final_report"] = {
                "status": state["decision_status"],
                "summary": state["decision_summary"],
                "missing_fields": missing,
            }
            state["session_tags"] = self._derive_session_tags(state["final_report"], state)
            return state

        report = self.rule_evaluator.evaluate(profile)
        state["followup_field"] = None
        state["needs_followup"] = False
        state["decision_status"] = report["status"]
        state["decision_summary"] = report["summary"]
        state["lead_step"] = "finalized"
        state["summary"] = report["summary"]
        state["qualification_category"] = report["status"].lower()
        state["conversation_status"] = "completed"
        state["final_report"] = report
        # If the user explicitly confirmed early termination, finalize now
        if state.get("user_confirmed_early_end"):
            state["needs_followup"] = False
            state["followup_field"] = None
            state["session_tags"] = self._derive_session_tags(report, state)
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
        state["session_tags"] = self._derive_session_tags(report, state)
        return state
