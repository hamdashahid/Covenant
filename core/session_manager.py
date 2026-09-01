from __future__ import annotations

import uuid
from typing import Any

from persistence.sqlite_store import SQLiteStore


class SessionManager:
    RUNTIME_KEYS = (
        "skipped_fields",
        "deferred_reasons",
        "field_attempts",
        "asked_fields",
        "answered_fields",
        "question_history",
        "followup_field",
        "last_extraction",
        "home_purchase_context",
        "recent_profile_corrections",
        "turn_understanding",
    )

    def __init__(self, store: SQLiteStore, default_model_id: str = "claude-sonnet-4-6") -> None:
        self.store = store
        self.default_model_id = default_model_id

    def start_or_resume(
        self,
        session_id: str | None = None,
        model_id: str | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        resolved_model = model_id or self.default_model_id
        existing: dict[str, Any] | None = None
        if session_id:
            existing = self.store.get_session(session_id)
            if existing:
                profile, conflicts = self.store.get_profile(existing["session_id"])
                history = self.store.get_messages(existing["session_id"])
                turn_count = len([message for message in history if message.get("role") == "user"])
                state = {
                    "session_id": existing["session_id"],
                    "model_id": existing["model_id"],
                    "conversation_history": history,
                    "applicant_profile": profile,
                    "profile_conflicts": conflicts,
                    "turn_count": turn_count,
                    "max_turns": 16,
                    "decision_status": "Requires More Info",
                    "decision_summary": "",
                    "conversation_tag": existing.get("conversation_tag"),
                }
                saved_runtime = existing.get("runtime_state", {})
                if isinstance(saved_runtime, dict):
                    state.update({key: saved_runtime[key] for key in self.RUNTIME_KEYS if key in saved_runtime})
                return existing["session_id"], existing["model_id"], state

        new_session_id = session_id or str(uuid.uuid4())
        state = {
            "session_id": new_session_id,
            "model_id": resolved_model,
            "conversation_history": [],
            "applicant_profile": {},
            "profile_conflicts": [],
            "turn_count": 0,
            "max_turns": 16,
            "decision_status": "Requires More Info",
            "decision_summary": "",
            "conversation_tag": None,
        }
        self.store.create_session(new_session_id, resolved_model)
        return new_session_id, resolved_model, state

    def save_state(self, session_id: str, state: dict[str, Any], completed: bool = False) -> None:
        self.store.upsert_profile(
            session_id=session_id,
            profile=state.get("applicant_profile", {}),
            conflicts=state.get("profile_conflicts", []),
        )
        self.store.replace_messages(
            session_id=session_id,
            messages=state.get("conversation_history", []),
        )
        self.store.update_session_state(session_id, "closed" if completed else "in_progress")
        if hasattr(self.store, "update_runtime_state"):
            self.store.update_runtime_state(
                session_id,
                {key: state[key] for key in self.RUNTIME_KEYS if key in state},
            )
