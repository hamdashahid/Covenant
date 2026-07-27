from __future__ import annotations

from typing import Any

import yaml


class RuleEvaluator:
    def __init__(self, rules_path: str) -> None:
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = yaml.safe_load(f) or {}

    def evaluate(self, profile: dict[str, Any]) -> dict[str, Any]:
        failures: list[str] = []
        rule_trace: list[dict[str, Any]] = []

        annual_income = float(profile.get("annual_income", 0) or 0)
        monthly_debt = float(profile.get("monthly_debt", 0) or 0)
        credit_score = int(profile.get("credit_score", 0) or 0)
        employment_status = str(profile.get("employment_status", "")).strip().lower()

        income_threshold = float(self.rules.get("income_threshold", 0))
        max_dti_ratio = float(self.rules.get("max_dti_ratio", 1))
        min_credit_score = int(self.rules.get("min_credit_score", 0))
        allowed_statuses = [str(s).lower() for s in self.rules.get("allowed_employment_statuses", [])]

        monthly_income = annual_income / 12 if annual_income > 0 else 0
        dti_ratio = (monthly_debt / monthly_income) if monthly_income > 0 else 1.0

        income_passed = annual_income >= income_threshold
        rule_trace.append(
            {
                "rule_name": "income_threshold",
                "passed": income_passed,
                "observed_value": annual_income,
                "threshold_value": income_threshold,
                "comparison": ">=",
                "details": f"annual_income {annual_income} >= {income_threshold}",
            }
        )
        if not income_passed:
            failures.append(f"Annual income below threshold ({annual_income} < {income_threshold})")

        dti_passed = dti_ratio <= max_dti_ratio
        rule_trace.append(
            {
                "rule_name": "max_dti_ratio",
                "passed": dti_passed,
                "observed_value": dti_ratio,
                "threshold_value": max_dti_ratio,
                "comparison": "<=",
                "details": f"dti_ratio {dti_ratio:.4f} <= {max_dti_ratio}",
            }
        )
        if not dti_passed:
            failures.append(f"Debt-to-income ratio too high ({dti_ratio:.2f} > {max_dti_ratio})")

        credit_passed = credit_score >= min_credit_score
        rule_trace.append(
            {
                "rule_name": "min_credit_score",
                "passed": credit_passed,
                "observed_value": credit_score,
                "threshold_value": min_credit_score,
                "comparison": ">=",
                "details": f"credit_score {credit_score} >= {min_credit_score}",
            }
        )
        if not credit_passed:
            failures.append(f"Credit score below minimum ({credit_score} < {min_credit_score})")

        employment_passed = employment_status in allowed_statuses
        rule_trace.append(
            {
                "rule_name": "allowed_employment_statuses",
                "passed": employment_passed,
                "observed_value": employment_status,
                "threshold_value": allowed_statuses,
                "comparison": "in",
                "details": f"employment_status {employment_status!r} in {allowed_statuses}",
            }
        )
        if not employment_passed:
            failures.append("Employment status not eligible")

        eligible = len(failures) == 0
        status = "Eligible" if eligible else "Ineligible"
        summary = "All rules passed" if eligible else "; ".join(failures)

        return {
            "status": status,
            "eligible": eligible,
            "summary": summary,
            "failed_rules": failures,
            "rule_trace": rule_trace,
            "metrics": {
                "annual_income": annual_income,
                "monthly_debt": monthly_debt,
                "credit_score": credit_score,
                "dti_ratio": round(dti_ratio, 4),
            },
        }
