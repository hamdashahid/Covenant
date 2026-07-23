from __future__ import annotations

import json
import os
import re
import time
from typing import Any

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover
    Anthropic = None  # type: ignore


class ClaudeClientAdapter:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._client = Anthropic(api_key=self.api_key) if (Anthropic and self.api_key) else None

    def extract_structured(self, prompt: str, latest_response: str) -> str:
        if not self._client:
            return self._fallback_extract(latest_response, "Anthropic client unavailable")
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = self._client.messages.create(
                    model=self.model_id,
                    max_tokens=400,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text_parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
                return "\n".join(text_parts).strip()
            except Exception as exc:  # pragma: no cover
                if attempt < max_attempts - 1:
                    time.sleep(0.5 * (2**attempt))
                    continue
                return self._fallback_extract(
                    latest_response,
                    f"Anthropic API error after {max_attempts} attempts: {exc}",
                )
        return self._fallback_extract(latest_response, "Anthropic API error: unknown failure")

    def _fallback_extract(self, text: str, reason: str) -> str:
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
        elif "employed" in lowered:
            fields["employment_status"] = "employed"
        elif "unemployed" in lowered:
            fields["employment_status"] = "unemployed"

        return json.dumps(
            {
                "fields": fields,
                "confidence": 0.45,
                "issues": [f"Fallback extraction used: {reason}"],
            }
        )
