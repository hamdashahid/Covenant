from __future__ import annotations

import json
import re
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

    def _normalize_numeric_text(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        # Remove currency symbols and comma grouping before parsing.
        text = re.sub(r"[\$₹£€]", "", text)
        text = text.replace(",", "")

        # Extract the first numeric token, allowing modifiers like k / m and
        # fuzzy prefixes such as above, below, around, or plus signs.
        match = re.search(r"([-+]?\d+(?:\.\d+)?)([kKmM]?)", text)
        if match:
            number_text = match.group(1)
            suffix = match.group(2).lower()
            try:
                value_float = float(number_text)
            except (TypeError, ValueError, OverflowError):
                return text
            if suffix == "k":
                value_float *= 1_000
            elif suffix == "m":
                value_float *= 1_000_000
            return str(value_float)

        # Preserve the cleaned string if no numeric token was found.
        return text

    def _parse_float(self, raw: Any) -> float | None:
        if raw is None:
            return None
        if isinstance(raw, bool):
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        text = self._normalize_numeric_text(raw)
        if not text:
            return None
        try:
            return float(text)
        except (TypeError, ValueError, OverflowError):
            return None

    def _parse_int(self, raw: Any) -> int | None:
        if raw is None:
            return None
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float) and raw.is_integer():
            return int(raw)
        text = self._normalize_numeric_text(raw)
        if not text:
            return None
        try:
            if "." in text:
                numeric = float(text)
                if numeric.is_integer():
                    return int(numeric)
                return None
            return int(text)
        except (TypeError, ValueError, OverflowError):
            return None

    def _infer_fields_from_question(self, raw: str, latest_question: str) -> dict[str, Any]:
        response = str(raw or "").strip()
        question = str(latest_question or "").lower()
        inferred: dict[str, Any] = {}

        if not response:
            return inferred

        if re.search(r"annual income|income", question):
            value = self._parse_float(response)
            if value is not None:
                inferred["annual_income"] = value

        if re.search(r"monthly debt|debt payment|debt", question):
            value = self._parse_float(response)
            if value is not None:
                inferred["monthly_debt"] = value

        if re.search(r"credit score|score", question):
            value = self._parse_int(response)
            if value is not None:
                inferred["credit_score"] = value

        if re.search(r"employment status", question):
            normalized = response.lower()
            if normalized in {"employed", "self-employed", "unemployed"}:
                inferred["employment_status"] = normalized

        if re.search(r"years|how long have you been|current job|business", question):
            value = self._parse_float(response)
            if value is not None:
                inferred["employment_years"] = value

        if re.search(r"property value|value of the property|property you want|price of the property", question):
            value = self._parse_float(response)
            if value is not None:
                inferred["property_value"] = value

        if re.search(r"loan amount|how much loan|requesting|loan you're requesting|loan you want", question):
            value = self._parse_float(response)
            if value is not None:
                inferred["requested_loan_amount"] = value

        if re.search(r"down payment|pay upfront|down payment", question):
            value = self._parse_float(response)
            if value is not None:
                inferred["down_payment"] = value

        # Property type inference: look for keywords
        if re.search(r"home|house|resid|live in|residential", response.lower()):
            inferred["property_type"] = "residential"
        elif re.search(r"shop|store|commercial|business|office|warehouse", response.lower()):
            inferred["property_type"] = "commercial"

        return inferred

    def _coerce_and_validate(self, fields: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        validated: dict[str, Any] = {}
        issues: list[str] = []

        if "annual_income" in fields:
            value = self._parse_float(fields["annual_income"])
            if value is not None and value > 0:
                validated["annual_income"] = value
            else:
                issues.append("annual_income must be > 0")

        if "monthly_debt" in fields:
            value = self._parse_float(fields["monthly_debt"])
            if value is not None and value >= 0:
                validated["monthly_debt"] = value
            else:
                issues.append("monthly_debt must be >= 0")

        if "credit_score" in fields:
            value = self._parse_int(fields["credit_score"])
            if value is not None and 300 <= value <= 850:
                validated["credit_score"] = value
            else:
                issues.append("credit_score must be between 300 and 850")

        if "employment_status" in fields:
            value = str(fields["employment_status"]).strip().lower()
            if value:
                validated["employment_status"] = value
            else:
                issues.append("employment_status is invalid")

        if "requested_loan_amount" in fields:
            value = self._parse_float(fields["requested_loan_amount"])
            if value is not None and value > 0:
                validated["requested_loan_amount"] = value
            else:
                issues.append("requested_loan_amount must be > 0")

        if "employment_years" in fields:
            value = self._parse_float(fields["employment_years"])
            if value is not None and value >= 0:
                validated["employment_years"] = value
            else:
                issues.append("employment_years must be >= 0")

        if "property_value" in fields:
            value = self._parse_float(fields["property_value"])
            if value is not None and value > 0:
                validated["property_value"] = value
            else:
                issues.append("property_value must be > 0")

        if "down_payment" in fields:
            value = self._parse_float(fields["down_payment"])
            if value is not None and value >= 0:
                validated["down_payment"] = value
            else:
                issues.append("down_payment must be >= 0")

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
        try:
            raw = self.llm_client.extract_structured(prompt, state.get("latest_user_response", ""))
            extraction_failed = False
        except Exception as exc:  # pragma: no cover - defensive guard
            raw = json.dumps(
                {
                    "fields": {},
                    "confidence": 0.0,
                    "issues": [f"LLM extraction failed: {exc}"],
                }
            )
            extraction_failed = True
        fields, confidence, parse_issues = self._parse_response(raw)
        validated_fields, validation_issues = self._coerce_and_validate(fields)

        if not extraction_failed and (confidence < 0.30 or (not validated_fields and not parse_issues)):
            inferred_fields = self._infer_fields_from_question(
                state.get("latest_user_response", ""),
                state.get("current_question", ""),
            )
            inferred_validated, inferred_issues = self._coerce_and_validate(inferred_fields)
            if inferred_validated and not validated_fields:
                validated_fields = inferred_validated
                validation_issues = inferred_issues
            elif not confidence < 0.30:
                validation_issues.extend(inferred_issues)

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