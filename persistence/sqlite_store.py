from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteStore:
    def __init__(self, db_path: str = "core.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    session_state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    closed_at TEXT,
                    tags TEXT,
                    conversation_tag TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS applicant_profiles (
                    session_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    conflicts_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eligibility_reports (
                    session_id TEXT PRIMARY KEY,
                    decision TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    rule_trace_json TEXT NOT NULL,
                    finalized_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    message_timestamp TEXT NOT NULL,
                    tags TEXT
                )
                """
            )
            # Migration safety: ensure tags columns exist if upgrading an older DB
            cur = conn.execute("PRAGMA table_info(sessions)").fetchall()
            cols = [r[1] for r in cur]
            if "tags" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN tags TEXT")
            if "conversation_tag" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN conversation_tag TEXT")
            cur = conn.execute("PRAGMA table_info(messages)").fetchall()
            cols = [r[1] for r in cur]
            if "tags" not in cols:
                conn.execute("ALTER TABLE messages ADD COLUMN tags TEXT")

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, model_id, session_state, created_at, closed_at, tags, conversation_tag FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        tags = None
        try:
            tags = json.loads(row["tags"]) if row["tags"] else None
        except Exception:
            tags = None
        conversation_tag = row["conversation_tag"] if "conversation_tag" in row.keys() else None
        return {
            "session_id": row["session_id"],
            "model_id": row["model_id"],
            "session_state": row["session_state"],
            "created_at": row["created_at"],
            "closed_at": row["closed_at"],
            "tags": tags,
            "conversation_tag": conversation_tag,
        }

    def create_session(self, session_id: str, model_id: str, tags: list[str] | None = None) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, model_id, session_state, created_at, closed_at, tags, conversation_tag)
                VALUES (?, ?, ?, ?, NULL, ?, NULL)
                ON CONFLICT(session_id) DO UPDATE SET
                    model_id=excluded.model_id,
                    tags=COALESCE(excluded.tags, sessions.tags)
                """,
                (session_id, model_id, "in_progress", now, json.dumps(tags) if tags is not None else None),
            )

    def merge_session_tags(self, session_id: str, tags: list[str] | None) -> None:
        if not tags:
            return
        existing = self.get_session(session_id)
        existing_tags = existing.get("tags") if existing else []
        if not isinstance(existing_tags, list):
            existing_tags = []
        merged: list[str] = []
        for tag in [*existing_tags, *tags]:
            value = str(tag).strip()
            if value and value not in merged:
                merged.append(value)
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET tags = ? WHERE session_id = ?",
                (json.dumps(merged), session_id),
            )

    def set_conversation_tag(self, session_id: str, conversation_tag: str | None) -> None:
        if not conversation_tag:
            return
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET conversation_tag = ? WHERE session_id = ?",
                (conversation_tag, session_id),
            )

    def get_available_tags(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT conversation_tag FROM sessions WHERE conversation_tag IS NOT NULL AND trim(conversation_tag) <> '' ORDER BY conversation_tag"
            ).fetchall()
        tags = []
        for row in rows:
            tag = str(row["conversation_tag"]).strip()
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def get_conversations_by_tag(self, tag: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, conversation_tag, created_at, closed_at FROM sessions WHERE conversation_tag = ? ORDER BY created_at DESC",
                (tag,),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "conversation_tag": row["conversation_tag"],
                "created_at": row["created_at"],
                "closed_at": row["closed_at"],
            }
            for row in rows
        ]

    def update_session_state(self, session_id: str, session_state: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET session_state = ?
                WHERE session_id = ?
                """,
                (session_state, session_id),
            )

    def update_session_tags(self, session_id: str, tags: list[str] | None) -> None:
        if tags is None:
            return
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET tags = ?
                WHERE session_id = ?
                """,
                (json.dumps(tags), session_id),
            )

    def get_sessions_with_tags(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, model_id, session_state, created_at, closed_at, tags
                FROM sessions
                ORDER BY closed_at IS NULL, closed_at DESC, created_at DESC, session_id DESC
                """
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            tags = None
            try:
                tags = json.loads(row["tags"]) if row["tags"] else None
            except Exception:
                tags = None
            out.append(
                {
                    "session_id": row["session_id"],
                    "model_id": row["model_id"],
                    "session_state": row["session_state"],
                    "created_at": row["created_at"],
                    "closed_at": row["closed_at"],
                    "tags": tags,
                }
            )
        return out

    def upsert_profile(
        self,
        session_id: str,
        profile: dict[str, Any],
        conflicts: list[str],
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO applicant_profiles (session_id, profile_json, conflicts_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    profile_json=excluded.profile_json,
                    conflicts_json=excluded.conflicts_json,
                    updated_at=excluded.updated_at
                """,
                (session_id, json.dumps(profile), json.dumps(conflicts), now),
            )

    def get_profile(self, session_id: str) -> tuple[dict[str, Any], list[str]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT profile_json, conflicts_json
                FROM applicant_profiles
                WHERE session_id = ?
                """,
                (session_id,),
            )
            result = row.fetchone()
        if not result:
            return {}, []
        return json.loads(result["profile_json"]), json.loads(result["conflicts_json"])

    def replace_messages(self, session_id: str, messages: list[dict[str, str]]) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM messages
                WHERE session_id = ?
                """,
                (session_id,),
            )
            conn.executemany(
                """
                INSERT INTO messages (session_id, role, content, message_timestamp, tags)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        str(message.get("role", "user")),
                        str(message.get("content", "")),
                        now,
                        json.dumps(message.get("tags")) if message.get("tags") is not None else None,
                    )
                    for message in messages
                ],
            )

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, tags
                FROM messages
                WHERE session_id = ?
                ORDER BY message_id ASC
                """,
                (session_id,),
            ).fetchall()
        out: list[dict[str, str]] = []
        for row in rows:
            tags = None
            try:
                tags = json.loads(row["tags"]) if row["tags"] else None
            except Exception:
                tags = None
            out.append({"role": row["role"], "content": row["content"], "tags": tags})
        return out

    def close_session(self, session_id: str, report: dict[str, Any], tags: list[str] | None = None) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET session_state = ?, closed_at = ?, tags = COALESCE(?, tags)
                WHERE session_id = ?
                """,
                ("closed", now, json.dumps(tags) if tags is not None else None, session_id),
            )
            conn.execute(
                """
                INSERT INTO eligibility_reports (session_id, decision, reasoning, rule_trace_json, finalized_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    decision=excluded.decision,
                    reasoning=excluded.reasoning,
                    rule_trace_json=excluded.rule_trace_json,
                    finalized_at=excluded.finalized_at
                """,
                (
                    session_id,
                    str(report.get("status", "Requires More Info")),
                    str(report.get("summary", "")),
                    json.dumps(report.get("failed_rules", report.get("missing_fields", []))),
                    now,
                ),
            )
