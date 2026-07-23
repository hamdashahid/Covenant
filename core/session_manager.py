from __future__ import annotations

import uuid
from typing import Any

from persistence.sqlite_store import SQLiteStore


class SessionManager:
    def __init__(self, store: SQLiteStore, default_model_id: str = "claude-3-5-sonnet-latest") -> None:
        self.store = store
        self.default_model_id = default_model_id

    def start_or_resume(
        self,
        session_id: str | None = None,
        model_id: str | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        resolved_model = model_id or self.default_model_id
        if session_id:
            existing = self.store.get_session(session_id)
            if existing and existing["status"] != "completed":
                return existing["session_id"], existing["model_id"], existing["state"]

        new_session_id = session_id or str(uuid.uuid4())
        state = {
            "session_id": new_session_id,
            "model_id": resolved_model,
            "conversation_history": [],
            "applicant_profile": {},
            "profile_conflicts": [],
            "turn_count": 0,
            "max_turns": 8,
            "decision_status": "Requires More Info",
            "decision_summary": "",
        }
        self.store.upsert_session(new_session_id, resolved_model, state, "in_progress")
        return new_session_id, resolved_model, state

    def save_state(self, session_id: str, model_id: str, state: dict[str, Any], completed: bool = False) -> None:
        self.store.upsert_session(
            session_id=session_id,
            model_id=model_id,
            state=state,
            status="completed" if completed else "in_progress",
        )
