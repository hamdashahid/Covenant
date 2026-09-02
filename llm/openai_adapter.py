from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class OpenAIClientAdapter:
    """
    Drop-in replacement for ClaudeClientAdapter. Same public interface
    (extract_structured / _fallback_extract) so agents/extraction_validation.py
    and main.py need no changes beyond which adapter class they import.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self._client = OpenAI(api_key=self.api_key) if (OpenAI and self.api_key) else None
        self.last_interpretation_error = ""
        self._interpretation_warning_emitted = False

    def extract_structured(self, prompt: str, latest_response: str) -> str:
        if not self._client:
            return self._fallback_extract(latest_response, "OpenAI client unavailable")
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_id,
                    max_tokens=400,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # pragma: no cover
                if attempt < max_attempts - 1:
                    time.sleep(0.5 * (2**attempt))
                    continue
                return self._fallback_extract(
                    latest_response,
                    f"OpenAI API error after {max_attempts} attempts: {exc}",
                )
        return self._fallback_extract(latest_response, "OpenAI API error: unknown failure")

    def understand_turn(
        self,
        current_field: str | None,
        current_question: str,
        user_response: str,
        profile: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Understand one whole applicant turn using a single semantic contract."""
        if not self._client:
            return None
        field_properties = {
            "down_payment": {"type": ["number", "null"]},
            "credit_score": {"type": ["number", "null"]},
            "employment_status": {"type": ["string", "null"]},
            "employment_years": {"type": ["number", "null"]},
            "annual_income": {"type": ["number", "null"]},
            "total_savings": {"type": ["number", "null"]},
            "monthly_debt": {"type": ["number", "null"]},
            "property_value": {"type": ["number", "null"]},
            "requested_loan_amount": {"type": ["number", "null"]},
        }
        schema = {
            "name": "mortgage_turn_understanding",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["answer", "skip", "unknown", "refusal", "clarification", "stop", "greeting"],
                    },
                    "field": {"type": ["string", "null"]},
                    "value": {"type": ["number", "string", "null"]},
                    "fields": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": field_properties,
                        "required": list(field_properties),
                    },
                    "corrections": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(field_properties)},
                    },
                    "uncertainty": {
                        "type": "string",
                        "enum": ["exact", "estimate", "range", "unknown", "not_applicable"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "needs_clarification": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "intent", "field", "value", "fields", "corrections", "uncertainty",
                    "confidence", "needs_clarification", "reason"
                ],
            },
        }
        prompt = (
            "Understand the applicant's entire latest reply in the context of the mortgage interview. "
            "Return every explicitly supplied mortgage fact in fields, even when several facts appear in one reply. "
            "Use null for fields not explicitly supplied. The current field controls only ambiguous bare answers; "
            "explicitly named extra facts belong to their own fields. Identify corrections to existing profile values. "
            "Understand meaning rather than matching exact wording. A clear absence such as 'I have no "
            "down payment' or 'I do not pay debt' is an answer with numeric value 0. 'I do not know' is "
            "unknown. 'I do not want to share' is refusal. A request to explain the question is clarification. "
            "For monthly income convert it to annual income; for annual debt convert it to monthly debt; "
            "represent job months as fractional years. Never turn a negative number into zero. "
            "Return intent=answer when at least one usable fact is supplied. Set value to the current field's "
            "value when supplied, otherwise null. "
            f"Current field: {current_field}\nCurrent question: {current_question}\n"
            f"Existing profile: {json.dumps(profile)}\nApplicant reply: {user_response}"
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model_id,
                max_tokens=250,
                temperature=0,
                response_format={"type": "json_schema", "json_schema": schema},
                messages=[{"role": "user", "content": prompt}],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            if not isinstance(parsed, dict):
                return None
            parsed_fields = parsed.get("fields", {})
            if isinstance(parsed_fields, dict):
                parsed["fields"] = {
                    key: value for key, value in parsed_fields.items() if value is not None
                }
            parsed["understanding_version"] = 1
            return parsed
        except Exception as exc:  # pragma: no cover - deterministic fallback remains available
            self.last_interpretation_error = f"{type(exc).__name__}: {exc}"
            if not self._interpretation_warning_emitted:
                logger.warning(
                    "OpenAI interpretation is unavailable; using the limited local reader. "
                    "Check API billing/credits and connectivity."
                )
                self._interpretation_warning_emitted = True
            return None

    def interpret_input(
        self,
        current_field: str | None,
        current_question: str,
        user_response: str,
        profile: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Backward-compatible alias for callers using the older method name."""
        return self.understand_turn(current_field, current_question, user_response, profile)

    def generate_reply(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        """
        Free-form conversational reply (not JSON). Used by the InterviewAgent so the
        interaction feels like a natural chat instead of a rigid form.
        Falls back to a plain templated line if the API is unavailable.
        """
        if not self._client:
            return ""
        try:
            response = self._client.chat.completions.create(
                model=self.model_id,
                max_tokens=200,
                temperature=0.6,
                messages=[{"role": "system", "content": system_prompt}, *messages],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:  # pragma: no cover
            return ""

    def _fallback_extract(self, text: str, reason: str) -> str:
        # Identical logic to ClaudeClientAdapter's fallback — provider-agnostic,
        # kept in sync so degraded-mode behavior doesn't differ by adapter.
        lowered = text.lower()
        fields: dict[str, Any] = {}

        numbers = [float(n.replace(",", "")) for n in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text)]
        if "income" in lowered and numbers:
            fields["annual_income"] = max(numbers)
        if "debt" in lowered and numbers:
            fields["monthly_debt"] = min(numbers)

        score_match = re.search(r"credit\s*score\D*(\d{3})", lowered)
        if score_match:
            fields["credit_score"] = int(score_match.group(1))

        if re.fullmatch(r"\s*(?:self(?:[ -]+emplo\w*)?|business[ -]?man)\s*", lowered) or re.search(r"\b(self[ -]?employ\w*|freelanc\w*|contractor|business[ -]?man|own (?:a )?business|business owner)\b", lowered):
            fields["employment_status"] = "self-employed"
        elif re.search(r"\b(retir\w*|pension(?:er|ed)?)\b", lowered):
            fields["employment_status"] = "retired"
        elif re.search(r"\b(not working anymore|no longer working|unemploy\w*|out of work|no job|jobless|laid[ -]?off|between jobs|looking for (?:a )?(?:job|work)|lost my job|terminated|dismissed|made redundant)\b", lowered):
            fields["employment_status"] = "unemployed"
        elif re.fullmatch(r"\s*emp\s*", lowered) or re.search(r"\b(employ\w*|full[ -]?time|part[ -]?time|working)\b", lowered):
            fields["employment_status"] = "employed"

        return json.dumps(
            {
                "fields": fields,
                "confidence": 0.45,
                "issues": [f"Fallback extraction used: {reason}"],
            }
        )
