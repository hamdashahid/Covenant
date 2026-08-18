from __future__ import annotations

import json
import os
import re
import time
from typing import Any

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

        if "self-employed" in lowered:
            fields["employment_status"] = "self-employed"
        elif "unemployed" in lowered:
            fields["employment_status"] = "unemployed"
        elif "employed" in lowered:
            fields["employment_status"] = "employed"

        return json.dumps(
            {
                "fields": fields,
                "confidence": 0.45,
                "issues": [f"Fallback extraction used: {reason}"],
            }
        )