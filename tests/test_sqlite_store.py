import sqlite3
import shutil
import tempfile
import unittest
import sys
from pathlib import Path
import gc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persistence.sqlite_store import SQLiteStore
from rules.rule_evaluator import RuleEvaluator


class TestSQLiteStore(unittest.TestCase):
    def test_close_session_persists_rule_trace_rows(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            db_path = Path(temp_dir) / "core.db"
            store = SQLiteStore(db_path=str(db_path))
            report = RuleEvaluator("rules/eligibility_rules.yaml").evaluate(
                {
                    "annual_income": 120000,
                    "monthly_debt": 1500,
                    "credit_score": 720,
                    "employment_status": "employed",
                }
            )

            store.create_session("session-1", "claude-sonnet-4-6")
            store.close_session("session-1", report)

            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                report_row = conn.execute(
                    "SELECT decision, reasoning, rule_trace_json FROM eligibility_reports WHERE session_id = ?",
                    ("session-1",),
                ).fetchone()
                trace_rows = conn.execute(
                    "SELECT rule_name, passed, observed_value, threshold_value, comparison, details FROM eligibility_report_rules WHERE session_id = ? ORDER BY rule_name ASC",
                    ("session-1",),
                ).fetchall()

            self.assertIsNotNone(report_row)
            self.assertEqual(report_row["decision"], "Eligible")
            self.assertIn("rule_name", report_row["rule_trace_json"])
            self.assertEqual(len(trace_rows), 4)
            self.assertEqual(trace_rows[0]["rule_name"], "allowed_employment_statuses")
            self.assertIn("employed", trace_rows[0]["observed_value"])
            self.assertEqual(trace_rows[0]["passed"], 1)
        finally:
            gc.collect()
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()