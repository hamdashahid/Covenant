from __future__ import annotations

from typing import Any

import yaml


class RuleEvaluator:
    def __init__(self, rules_path: str) -> None:
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = yaml.safe_load(f) or {}

    def evaluate(self, profile: dict[str, Any]) -> dict[str, Any]:
        # ---- Pull raw values from the applicant profile ----
        annual_income = float(profile.get("annual_income", 0) or 0)
        monthly_debt = float(profile.get("monthly_debt", 0) or 0)
        credit_score = int(profile.get("credit_score", 0) or 0)
        employment_status = str(profile.get("employment_status", "")).strip().lower()
        employment_years = float(profile.get("employment_years", 0) or 0)
        property_value = float(profile.get("property_value", 0) or 0)
        requested_loan_amount = float(profile.get("requested_loan_amount", 0) or 0)
        down_payment = float(profile.get("down_payment", 0) or 0)

        # ---- Pull thresholds from the rules file ----
        income_threshold = float(self.rules.get("income_threshold", 0))
        max_dti_ratio = float(self.rules.get("max_dti_ratio", 1))
        min_credit_score = int(self.rules.get("min_credit_score", 0))
        allowed_statuses = [str(s).lower() for s in self.rules.get("allowed_employment_statuses", [])]
        min_employment_years = float(self.rules.get("min_employment_years", 0))
        max_ltv_ratio = float(self.rules.get("max_ltv_ratio", 1))
        min_down_payment_percent = float(self.rules.get("min_down_payment_percent", 0))

        # ---- Derived metrics ----
        monthly_income = annual_income / 12 if annual_income > 0 else 0
        dti_ratio = (monthly_debt / monthly_income) if monthly_income > 0 else 1.0
        ltv_ratio = (requested_loan_amount / property_value) if property_value > 0 else 1.0
        down_payment_percent = (down_payment / property_value) if property_value > 0 else 0.0

        rule_breakdown: list[dict[str, Any]] = []

        # 1. Income
        income_passed = annual_income >= income_threshold
        rule_breakdown.append({
            "name": "Annual Income",
            "passed": income_passed,
            "value_display": f"Rs {annual_income:,.0f}",
            "threshold_display": f"minimum Rs {income_threshold:,.0f} required",
            "explanation": (
                f"Your yearly income of Rs {annual_income:,.0f} meets the minimum requirement of "
                f"Rs {income_threshold:,.0f}."
                if income_passed else
                f"Your yearly income of Rs {annual_income:,.0f} is below the minimum requirement of "
                f"Rs {income_threshold:,.0f}. Lenders need to see you earn enough to comfortably repay the loan."
            ),
        })

        # 2. Debt-to-Income Ratio
        dti_passed = dti_ratio <= max_dti_ratio
        rule_breakdown.append({
            "name": "Debt-to-Income Ratio",
            "passed": dti_passed,
            "value_display": f"{dti_ratio * 100:.1f}%",
            "threshold_display": f"must be {max_dti_ratio * 100:.0f}% or lower",
            "explanation": (
                f"Your monthly debt payments use {dti_ratio * 100:.1f}% of your monthly income, "
                f"which is within the {max_dti_ratio * 100:.0f}% limit lenders allow."
                if dti_passed else
                f"Your monthly debt payments use {dti_ratio * 100:.1f}% of your monthly income, "
                f"which is higher than the {max_dti_ratio * 100:.0f}% limit. This means too much of your "
                f"income already goes toward paying off existing debt."
            ),
        })

        # 3. Credit Score
        credit_passed = credit_score >= min_credit_score
        rule_breakdown.append({
            "name": "Credit Score",
            "passed": credit_passed,
            "value_display": str(credit_score),
            "threshold_display": f"minimum {min_credit_score} required",
            "explanation": (
                f"Your credit score of {credit_score} meets the minimum requirement of {min_credit_score}, "
                "showing a reliable repayment history."
                if credit_passed else
                f"Your credit score of {credit_score} is below the minimum requirement of {min_credit_score}. "
                "A low score suggests higher risk to lenders."
            ),
        })

        # 4. Employment Status
        status_passed = employment_status in allowed_statuses
        rule_breakdown.append({
            "name": "Employment Status",
            "passed": status_passed,
            "value_display": employment_status.title() if employment_status else "Not provided",
            "threshold_display": f"must be one of: {', '.join(s.title() for s in allowed_statuses)}",
            "explanation": (
                f"Your employment status ('{employment_status.title()}') is accepted by the lender."
                if status_passed else
                f"Your employment status ('{employment_status.title() or 'Unknown'}') is not accepted. "
                "Lenders typically require a steady income source (employed or self-employed)."
            ),
        })

        # 5. Employment Stability (years at job)
        years_passed = employment_years >= min_employment_years
        rule_breakdown.append({
            "name": "Job Stability",
            "passed": years_passed,
            "value_display": f"{employment_years:.1f} years",
            "threshold_display": f"minimum {min_employment_years:.0f} years required",
            "explanation": (
                f"You've been at your current job/business for {employment_years:.1f} years, showing stable "
                f"income history (minimum required: {min_employment_years:.0f} years)."
                if years_passed else
                f"You've been at your current job/business for only {employment_years:.1f} years, which is "
                f"below the {min_employment_years:.0f}-year stability lenders usually want to see."
            ),
        })

        # 6. Loan-to-Value Ratio
        ltv_passed = ltv_ratio <= max_ltv_ratio
        rule_breakdown.append({
            "name": "Loan-to-Value Ratio",
            "passed": ltv_passed,
            "value_display": f"{ltv_ratio * 100:.1f}%",
            "threshold_display": f"must be {max_ltv_ratio * 100:.0f}% or lower",
            "explanation": (
                f"You're asking to borrow {ltv_ratio * 100:.1f}% of the property's value, which is within "
                f"the {max_ltv_ratio * 100:.0f}% limit lenders allow."
                if ltv_passed else
                f"You're asking to borrow {ltv_ratio * 100:.1f}% of the property's value, which is more than "
                f"the {max_ltv_ratio * 100:.0f}% limit. This means the loan is too large relative to the "
                "property's worth."
            ),
        })

        # 7. Down Payment
        down_passed = down_payment_percent >= min_down_payment_percent
        rule_breakdown.append({
            "name": "Down Payment",
            "passed": down_passed,
            "value_display": f"Rs {down_payment:,.0f} ({down_payment_percent * 100:.1f}%)",
            "threshold_display": f"minimum {min_down_payment_percent * 100:.0f}% of property value required",
            "explanation": (
                f"Your down payment of Rs {down_payment:,.0f} ({down_payment_percent * 100:.1f}% of the "
                f"property value) meets the minimum requirement of {min_down_payment_percent * 100:.0f}%."
                if down_passed else
                f"Your down payment of Rs {down_payment:,.0f} ({down_payment_percent * 100:.1f}% of the "
                f"property value) is below the minimum requirement of {min_down_payment_percent * 100:.0f}%."
            ),
        })

        failures = [rule["name"] for rule in rule_breakdown if not rule["passed"]]
        eligible = len(failures) == 0
        status = "Eligible" if eligible else "Ineligible"
        summary = (
            "All eligibility rules passed."
            if eligible else
            f"{len(failures)} rule(s) not met: " + ", ".join(failures)
        )

        return {
            "status": status,
            "eligible": eligible,
            "summary": summary,
            "failed_rules": failures,
            "rule_breakdown": rule_breakdown,
            "metrics": {
                "annual_income": annual_income,
                "monthly_debt": monthly_debt,
                "credit_score": credit_score,
                "employment_status": employment_status,
                "employment_years": employment_years,
                "property_value": property_value,
                "requested_loan_amount": requested_loan_amount,
                "down_payment": down_payment,
                "dti_ratio": round(dti_ratio, 4),
                "ltv_ratio": round(ltv_ratio, 4),
                "down_payment_percent": round(down_payment_percent, 4),
            },
        }