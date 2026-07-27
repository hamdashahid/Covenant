import shutil
import tempfile
import unittest
from pathlib import Path

from core.session_manager import SessionManager
from persistence.sqlite_store import SQLiteStore


class TestSessionManager(unittest.TestCase):
    def test_default_model_uses_openai(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            db_path = Path(temp_dir) / "core.db"
            store = SQLiteStore(db_path=str(db_path))
            manager = SessionManager(store=store)

            session_id, model_id, state = manager.start_or_resume()

            self.assertEqual(model_id, "gpt-4o")
            self.assertEqual(state["model_id"], "gpt-4o")
            self.assertEqual(session_id, state["session_id"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
