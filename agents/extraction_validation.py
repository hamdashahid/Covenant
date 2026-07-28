from __future__ import annotations

import json
from typing import Any

from core.context_builder import ContextBuilder
from core.profile_updater import ProfileUpdater


class ExtractionValidationNode:
    def __init__(
        self,
        llm_client: Any,
        context_builder: ContextBuilder,
        profile_updater: ProfileUpdater,
        extraction_schema: dict[str, str],
    ) -> None:
        self.llm_client = llm_client
        self.context_builder = context_builder
        self.profile_updater = profile_updater
        self.extraction_schema = extraction_schema

    def _coerce_and_validate(self, fields: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        validated: dict[str, Any] = {}
        issues: list[str] = []

        if "annual_income" in fields:
            try:
                value = float(fields["annual_income"])
                if value > 0:
                    validated["annual_income"] = value
                else:
                    issues.append("annual_income must be > 0")
            except (TypeError, ValueError):
                issues.append("annual_income is invalid")

        if "monthly_debt" in fields:
            try:
                value = float(fields["monthly_debt"])
                if value >= 0:
                    validated["monthly_debt"] = value
                else:
                    issues.append("monthly_debt must be >= 0")
            except (TypeError, ValueError):
                issues.append("monthly_debt is invalid")

        if "credit_score" in fields:
            try:
                value = int(fields["credit_score"])
                if 300 <= value <= 850:
                    validated["credit_score"] = value
                else:
                    issues.append("credit_score must be between 300 and 850")
            except (TypeError, ValueError):
                issues.append("credit_score is invalid")

        if "employment_status" in fields:
            value = str(fields["employment_status"]).strip().lower()
            if value:
                validated["employment_status"] = value
            else:
                issues.append("employment_status is invalid")

        if "requested_loan_amount" in fields:
            try:
                value = float(fields["requested_loan_amount"])
                if value > 0:
                    validated["requested_loan_amount"] = value
                else:
                    issues.append("requested_loan_amount must be > 0")
            except (TypeError, ValueError):
                issues.append("requested_loan_amount is invalid")

        if "employment_years" in fields:
            try:
                value = float(fields["employment_years"])
                if value >= 0:
                    validated["employment_years"] = value
                else:
                    issues.append("employment_years must be >= 0")
            except (TypeError, ValueError):
                issues.append("employment_years is invalid")

        if "property_value" in fields:
            try:
                value = float(fields["property_value"])
                if value > 0:
                    validated["property_value"] = value
                else:
                    issues.append("property_value must be > 0")
            except (TypeError, ValueError):
                issues.append("property_value is invalid")

        if "down_payment" in fields:
            try:
                value = float(fields["down_payment"])
                if value >= 0:
                    validated["down_payment"] = value
                else:
                    issues.append("down_payment must be >= 0")
            except (TypeError, ValueError):
                issues.append("down_payment is invalid")

        return validated, issues

    def _parse_response(self, raw: str) -> tuple[dict[str, Any], float, list[str]]:
        try:
            parsed = json.loads(raw)
            fields = parsed.get("fields", {}) if isinstance(parsed, dict) else {}
            confidence = float(parsed.get("confidence", 0.0)) if isinstance(parsed, dict) else 0.0
            issues = parsed.get("issues", []) if isinstance(parsed, dict) else ["Invalid extractor response shape"]
            if not isinstance(fields, dict):
                fields = {}
                issues.append("fields must be an object")
            if not isinstance(issues, list):
                issues = ["issues must be a list"]
            return fields, confidence, [str(i) for i in issues]
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}, 0.0, ["Extractor response was not valid JSON"]

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        prompt = self.context_builder.build_extraction_prompt(
            conversation_history=state.get("conversation_history", []),
            profile=state.get("applicant_profile", {}),
            latest_question=state.get("current_question", ""),
            latest_user_response=state.get("latest_user_response", ""),
            schema=self.extraction_schema,
        )
        raw = self.llm_client.extract_structured(prompt, state.get("latest_user_response", ""))
        fields, confidence, parse_issues = self._parse_response(raw)
        validated_fields, validation_issues = self._coerce_and_validate(fields)

        if confidence < 0.30:
            validation_issues.append("Extraction confidence too low")

        merged, conflicts = self.profile_updater.merge(
            state.get("applicant_profile", {}),
            validated_fields,
        )

        state["applicant_profile"] = merged
        state["profile_conflicts"] = list(state.get("profile_conflicts", [])) + conflicts
        state["last_extraction"] = {
            "raw": raw,
            "fields": fields,
            "validated_fields": validated_fields,
            "confidence": confidence,
            "issues": parse_issues + validation_issues,
        }
        return state