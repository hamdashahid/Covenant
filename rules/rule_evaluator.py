from __future__ import annotations

from typing import Any

import yaml


class RuleEvaluator:
    def __init__(self, rules_path: str) -> None:
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = yaml.safe_load(f) or {}

    def evaluate(self, profile: dict[str, Any]) -> dict[str, Any]:
        failures: list[str] = []

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

        if annual_income < income_threshold:
            failures.append(f"Annual income below threshold ({annual_income} < {income_threshold})")
        if dti_ratio > max_dti_ratio:
            failures.append(f"Debt-to-income ratio too high ({dti_ratio:.2f} > {max_dti_ratio})")
        if credit_score < min_credit_score:
            failures.append(f"Credit score below minimum ({credit_score} < {min_credit_score})")
        if employment_status not in allowed_statuses:
            failures.append("Employment status not eligible")

        eligible = len(failures) == 0
        status = "Eligible" if eligible else "Ineligible"
        summary = "All rules passed" if eligible else "; ".join(failures)

        return {
            "status": status,
            "eligible": eligible,
            "summary": summary,
            "failed_rules": failures,
            "metrics": {
                "annual_income": annual_income,
                "monthly_debt": monthly_debt,
                "credit_score": credit_score,
                "dti_ratio": round(dti_ratio, 4),
            },
        }
