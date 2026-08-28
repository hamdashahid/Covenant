from __future__ import annotations

from typing import Any

from core.schemas import ELIGIBILITY_REQUIRED_FIELDS


class DecisionAgent:
    def __init__(self, rule_evaluator: Any) -> None:
        self.rule_evaluator = rule_evaluator
        # Tunable thresholds (defaults can be overridden by caller)
        self.early_offer_pass_ratio = 0.85
        self.early_auto_pass_ratio = 1.0

    def set_early_termination_thresholds(self, offer_ratio: float, auto_ratio: float) -> None:
        self.early_offer_pass_ratio = float(offer_ratio)
        self.early_auto_pass_ratio = float(auto_ratio)

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip().replace(",", "").replace("$", "").replace("₹", "").replace("£", "").replace("€", "")
            if not text:
                return default
            try:
                return float(text)
            except (TypeError, ValueError, OverflowError):
                return default
        return default

    def _coerce_int(self, value: Any, default: int = 0) -> int:
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            text = value.strip().replace(",", "").replace("$", "").replace("₹", "").replace("£", "").replace("€", "")
            if not text:
                return default
            try:
                return int(float(text))
            except (TypeError, ValueError, OverflowError):
                return default
        return default

    def _derive_session_tags(self, report: dict[str, Any], state: dict[str, Any]) -> list[str]:
        if state.get("user_requested_stop"):
            return ["ciap-stopped"]
        if report.get("status") == "Eligible":
            return ["ciap-ready"]
        if report.get("status") == "Ineligible":
            return ["ciap-follow-up"]
        if report.get("status") == "Requires More Info":
            return ["ciap-follow-up"]
        return []

    def _derive_conversation_tag(self, report: dict[str, Any], state: dict[str, Any]) -> str | None:
        if state.get("user_requested_stop"):
            return "Stopped"
        status = report.get("status")
        if status == "Eligible":
            return "Qualified - Ready"
        if status == "Ineligible":
            failed = {
                str(rule.get("name", "")).lower()
                for rule in report.get("rule_breakdown", [])
                if not rule.get("passed") and rule.get("evaluation_status") != "not_evaluated"
            }
            if "annual income" in failed and "credit score" in failed:
                return "Unqualified - Low Income and Low Credit"
            if "annual income" in failed:
                return "Unqualified - Low Income"
            if "credit score" in failed:
                return "Unqualified - Low Credit"
            if "debt-to-income ratio" in failed:
                return "Unqualified - High Debt"
            if "loan-to-value ratio" in failed:
                return "Unqualified - High Loan-to-Value"
            if "job stability" in failed:
                return "Unqualified - Low Stability"
            return "Unqualified - Needs Review"
        if status == "Requires More Info":
            return "Needs More Info"
        return None

    def _build_not_evaluated_report(self, profile: dict[str, Any], summary: str) -> dict[str, Any]:
        rules = self.rule_evaluator.rules if hasattr(self.rule_evaluator, "rules") else {}
        income_threshold = float(rules.get("income_threshold", 0))
        max_dti_ratio = float(rules.get("max_dti_ratio", 1))
        min_credit_score = int(rules.get("min_credit_score", 0))
        allowed_statuses = [str(s).lower() for s in rules.get("allowed_employment_statuses", [])]
        min_employment_years = float(rules.get("min_employment_years", 0))
        max_ltv_ratio = float(rules.get("max_ltv_ratio", 1))
        min_down_payment_percent = float(rules.get("min_down_payment_percent", 0))

        annual_income = self._coerce_float(profile.get("annual_income"), 0)
        monthly_debt = self._coerce_float(profile.get("monthly_debt"), 0)
        credit_score = self._coerce_int(profile.get("credit_score"), 0)
        employment_status = str(profile.get("employment_status", "")).strip().lower()
        employment_years = self._coerce_float(profile.get("employment_years"), 0)
        property_value = self._coerce_float(profile.get("property_value"), 0)
        requested_loan_amount = self._coerce_float(profile.get("requested_loan_amount"), 0)
        down_payment = self._coerce_float(profile.get("down_payment"), 0)

        def rule_item(
            name: str,
            passed: bool | None,
            value_display: str,
            threshold_display: str,
            explanation: str,
        ) -> dict[str, Any]:
            item = {
                "name": name,
                "value_display": value_display,
                "threshold_display": threshold_display,
                "explanation": explanation,
                "passed": passed if passed is not None else False,
            }
            if passed is None:
                item["evaluation_status"] = "not_evaluated"
            else:
                item["evaluation_status"] = "passed" if passed else "failed"
            return item

        report: list[dict[str, Any]] = []

        if "annual_income" in profile:
            income_passed = annual_income >= income_threshold
            report.append(
                rule_item(
                    "Annual Income",
                    income_passed,
                    f"Rs {annual_income:,.0f}",
                    f"minimum Rs {income_threshold:,.0f} required",
                    "This check was evaluated." if income_passed else "Your income is below the minimum requirement.",
                )
            )
        else:
            report.append(
                rule_item(
                    "Annual Income",
                    None,
                    "Not provided",
                    f"minimum Rs {income_threshold:,.0f} required",
                    "This check was not evaluated because income information was missing.",
                )
            )

        if "monthly_debt" in profile and "annual_income" in profile and annual_income > 0:
            dti_ratio = monthly_debt / (annual_income / 12)
            report.append(
                rule_item(
                    "Debt-to-Income Ratio",
                    dti_ratio <= max_dti_ratio,
                    f"{dti_ratio * 100:.1f}%",
                    f"must be {max_dti_ratio * 100:.0f}% or lower",
                    "This check was evaluated." if dti_ratio <= max_dti_ratio else "Your debt-to-income ratio is too high.",
                )
            )
        else:
            report.append(
                rule_item(
                    "Debt-to-Income Ratio",
                    None,
                    "Not evaluated",
                    f"must be {max_dti_ratio * 100:.0f}% or lower",
                    "This check was not evaluated because income or debt information was missing.",
                )
            )

        if "credit_score" in profile:
            credit_passed = credit_score >= min_credit_score
            report.append(
                rule_item(
                    "Credit Score",
                    credit_passed,
                    str(credit_score),
                    f"minimum {min_credit_score} required",
                    "This check was evaluated." if credit_passed else "Your credit score is below the minimum requirement.",
                )
            )
        else:
            report.append(
                rule_item(
                    "Credit Score",
                    None,
                    "Not provided",
                    f"minimum {min_credit_score} required",
                    "This check was not evaluated because credit score information was missing.",
                )
            )

        if "employment_status" in profile:
            status_passed = employment_status in allowed_statuses
            report.append(
                rule_item(
                    "Employment Status",
                    status_passed,
                    employment_status.title() if employment_status else "Unknown",
                    f"must be one of: {', '.join(s.title() for s in allowed_statuses)}",
                    "This check was evaluated." if status_passed else "Your employment status is not accepted.",
                )
            )
        else:
            report.append(
                rule_item(
                    "Employment Status",
                    None,
                    "Not provided",
                    f"must be one of: {', '.join(s.title() for s in allowed_statuses)}",
                    "This check was not evaluated because employment status information was missing.",
                )
            )

        if "employment_years" in profile:
            years_passed = employment_years >= min_employment_years
            report.append(
                rule_item(
                    "Job Stability",
                    years_passed,
                    f"{employment_years:.1f} years",
                    f"minimum {min_employment_years:.0f} years required",
                    "This check was evaluated." if years_passed else "Your job stability is below the requirement.",
                )
            )
        else:
            report.append(
                rule_item(
                    "Job Stability",
                    None,
                    "Not provided",
                    f"minimum {min_employment_years:.0f} years required",
                    "This check was not evaluated because job stability information was missing.",
                )
            )

        if "property_value" in profile and "requested_loan_amount" in profile and property_value > 0:
            ltv_ratio = requested_loan_amount / property_value
            report.append(
                rule_item(
                    "Loan-to-Value Ratio",
                    ltv_ratio <= max_ltv_ratio,
                    f"{ltv_ratio * 100:.1f}%",
                    f"must be {max_ltv_ratio * 100:.0f}% or lower",
                    "This check was evaluated." if ltv_ratio <= max_ltv_ratio else "Your loan-to-value ratio is too high.",
                )
            )
        else:
            report.append(
                rule_item(
                    "Loan-to-Value Ratio",
                    None,
                    "Not evaluated",
                    f"must be {max_ltv_ratio * 100:.0f}% or lower",
                    "This check was not evaluated because property or loan amount information was missing.",
                )
            )

        if "down_payment" in profile and "property_value" in profile and property_value > 0:
            down_payment_percent = down_payment / property_value
            report.append(
                rule_item(
                    "Down Payment",
                    down_payment_percent >= min_down_payment_percent,
                    f"Rs {down_payment:,.0f} ({down_payment_percent * 100:.1f}%)",
                    f"minimum {min_down_payment_percent * 100:.0f}% of property value required",
                    "This check was evaluated." if down_payment_percent >= min_down_payment_percent else "Your down payment is smaller than required.",
                )
            )
        else:
            report.append(
                rule_item(
                    "Down Payment",
                    None,
                    "Not evaluated",
                    f"minimum {min_down_payment_percent * 100:.0f}% of property value required",
                    "This check was not evaluated because down payment or property information was missing.",
                )
            )

        return {
            "status": "Requires More Info",
            "summary": summary,
            "rule_breakdown": report,
        }

    def _deterministic_early_report(self, profile: dict[str, Any]) -> dict[str, Any] | None:
        if not hasattr(self.rule_evaluator, "rules"):
            return None

        income_threshold = float(self.rule_evaluator.rules.get("income_threshold", 0))
        min_credit_score = int(self.rule_evaluator.rules.get("min_credit_score", 0))

        annual_income = self._coerce_float(profile.get("annual_income"), 0)
        credit_score = self._coerce_int(profile.get("credit_score"), 0)

        failures: list[dict[str, Any]] = []

        if annual_income and annual_income < income_threshold:
            failures.append(
                {
                    "name": "Annual Income",
                    "passed": False,
                    "evaluation_status": "failed",
                    "value_display": f"Rs {annual_income:,.0f}",
                    "threshold_display": f"minimum Rs {income_threshold:,.0f} required",
                    "explanation": f"Your yearly income of Rs {annual_income:,.0f} is below the minimum requirement of Rs {income_threshold:,.0f}.",
                }
            )

        if credit_score and credit_score < min_credit_score:
            failures.append(
                {
                    "name": "Credit Score",
                    "passed": False,
                    "evaluation_status": "failed",
                    "value_display": str(credit_score),
                    "threshold_display": f"minimum {min_credit_score} required",
                    "explanation": f"Your credit score of {credit_score} is below the minimum requirement of {min_credit_score}.",
                }
            )

        if len(failures) >= 2:
            return {
                "status": "Ineligible",
                "eligible": False,
                "summary": (
                    "Some known values already fail eligibility, but these are often"
                    " fixable with time or a stronger profile. Focus on the areas below"
                    " and revisit once they improve."
                ),
                "failed_rules": [rule["name"] for rule in failures],
                "rule_breakdown": failures,
                "metrics": {
                    "annual_income": annual_income,
                    "credit_score": credit_score,
                },
            }
        return None

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        profile = state.get("applicant_profile", {})
        missing = [field for field in ELIGIBILITY_REQUIRED_FIELDS if field not in profile]
        skipped_fields = set(state.get("skipped_fields", []))
        actionable_missing = [field for field in missing if field not in skipped_fields]
        max_turns = int(state.get("max_turns", 8))
        turn_count = int(state.get("turn_count", 0))

        if state.get("user_requested_finalize"):
            if missing:
                state["followup_field"] = None
                state["needs_followup"] = False
                state["decision_status"] = "Requires More Info"
                state["decision_summary"] = "User indicated no more information is available."
                report = self._build_not_evaluated_report(profile, state["decision_summary"])
                state["final_report"] = report
                state["lead_step"] = "finalized"
                state["summary"] = state["decision_summary"]
                state["qualification_category"] = "requires_more_info"
                state["conversation_status"] = "in_progress"
                state["offer_early_termination"] = False
                state["auto_terminated"] = False
                state["session_tags"] = self._derive_session_tags(report, state)
                state["conversation_tag"] = self._derive_conversation_tag(report, state)
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
            state["conversation_tag"] = self._derive_conversation_tag(report, state)
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
            state["conversation_tag"] = self._derive_conversation_tag(deterministic_report, state)
            return state

        if actionable_missing and turn_count < max_turns:
            state["followup_field"] = actionable_missing[0]
            state["needs_followup"] = True
            state["decision_status"] = "Requires More Info"
            state["decision_summary"] = f"Need more validated information: missing {', '.join(missing)}"
            report = {
                "status": state["decision_status"],
                "summary": state["decision_summary"],
                "missing_fields": missing,
            }
            state["final_report"] = report
            state["lead_step"] = "collecting"
            state["summary"] = state["decision_summary"]
            state["qualification_category"] = "requires_more_info"
            state["conversation_status"] = "in_progress"
            state["session_tags"] = self._derive_session_tags(report, state)
            state["conversation_tag"] = self._derive_conversation_tag(report, state)
            return state

        if missing and not actionable_missing:
            state["followup_field"] = None
            state["needs_followup"] = False
            state["decision_status"] = "Requires More Info"
            state["decision_summary"] = (
                "The pre-check could not be completed because some questions were skipped."
            )
            report = {
                "status": state["decision_status"],
                "summary": state["decision_summary"],
                "missing_fields": missing,
            }
            state["final_report"] = report
            state["lead_step"] = "finalized"
            state["summary"] = state["decision_summary"]
            state["qualification_category"] = "requires_more_info"
            state["conversation_status"] = "completed"
            state["session_tags"] = self._derive_session_tags(report, state)
            state["conversation_tag"] = self._derive_conversation_tag(report, state)
            return state

        if missing and turn_count >= max_turns:
            state["followup_field"] = None
            state["needs_followup"] = False
            state["decision_status"] = "Requires More Info"
            state["decision_summary"] = (
                "Max interview turns reached before collecting required validated data"
            )
            report = {
                "status": state["decision_status"],
                "summary": state["decision_summary"],
                "missing_fields": missing,
            }
            state["final_report"] = report
            state["lead_step"] = "finalized"
            state["summary"] = state["decision_summary"]
            state["qualification_category"] = "requires_more_info"
            state["conversation_status"] = "in_progress"
            state["session_tags"] = self._derive_session_tags(report, state)
            state["conversation_tag"] = self._derive_conversation_tag(report, state)
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
        state["session_tags"] = self._derive_session_tags(report, state)
        state["conversation_tag"] = self._derive_conversation_tag(report, state)
        # All required interview answers are present, so show the result
        # directly instead of adding a redundant confirmation turn.
        state["offer_early_termination"] = False
        state["auto_terminated"] = True
        state["needs_followup"] = False
        state["session_tags"] = self._derive_session_tags(report, state)
        state["conversation_tag"] = self._derive_conversation_tag(report, state)
        return state
