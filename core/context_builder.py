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
            "Return ONLY valid JSON with keys: fields (object), confidence (0-1 number), issues (array of strings).\n"
            f"Schema: {json.dumps(schema)}\n"
            f"Current profile: {json.dumps(profile)}\n"
            f"Conversation history: {json.dumps(conversation_history[-8:])}\n"
            f"Latest question: {latest_question}\n"
            f"Latest response: {latest_user_response}\n"
        )
