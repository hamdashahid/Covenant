from __future__ import annotations

import json
from typing import Any


class ContextBuilder:
    def build_extraction_prompt(
        self,
        conversation_history: list[dict[str, str]],
        profile: dict[str, Any],
        latest_question: str,
        latest_user_response: str,
        schema: dict[str, str],
    ) -> str:
        return (
            "You are a strict mortgage data extraction and validation assistant. "
            "Extract structured fields from the latest applicant response.\n"
            "CRITICAL: Only include a field in `fields` if the applicant's LATEST RESPONSE "
            "explicitly and unambiguously states that value. Do NOT infer, guess, assume, or "
            "fill in a field based on context, tone, other turns, or what seems likely (e.g. do "
            "not infer employment status from age alone; do recognize an explicit statement that "
            "the applicant is retired or on a pension as unemployed, and do not "
            "infer a value of 0 or 'none' for a field the applicant did not address). If the "
            "latest response does not clearly address a field, omit that field entirely — never "
            "use placeholder values like 0, 'none', 'unknown', or '' to represent an unanswered "
            "field.\n"
            "Return ONLY valid JSON with keys: fields (object), confidence (0-1 number), issues (array of strings).\n"
            f"Schema: {json.dumps(schema)}\n"
            f"Current profile: {json.dumps(profile)}\n"
            f"Conversation history: {json.dumps(conversation_history[-8:])}\n"
            f"Latest question: {latest_question}\n"
            f"Latest response: {latest_user_response}\n"
        )
