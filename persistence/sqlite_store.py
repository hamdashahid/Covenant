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
                    state_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    session_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    conflicts_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, model_id, state_json, status FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "session_id": row["session_id"],
            "model_id": row["model_id"],
            "state": json.loads(row["state_json"]),
            "status": row["status"],
        }

    def upsert_session(
        self,
        session_id: str,
        model_id: str,
        state: dict[str, Any],
        status: str,
    ) -> None:
        now = _utc_now()
        state_json = json.dumps(state)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, model_id, state_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    model_id=excluded.model_id,
                    state_json=excluded.state_json,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (session_id, model_id, state_json, status, now, now),
            )

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
                INSERT INTO profiles (session_id, profile_json, conflicts_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    profile_json=excluded.profile_json,
                    conflicts_json=excluded.conflicts_json,
                    updated_at=excluded.updated_at
                """,
                (session_id, json.dumps(profile), json.dumps(conflicts), now),
            )

    def save_report(self, session_id: str, report: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reports (session_id, decision, summary, report_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    report.get("status", "Requires More Info"),
                    report.get("summary", ""),
                    json.dumps(report),
                    _utc_now(),
                ),
            )
