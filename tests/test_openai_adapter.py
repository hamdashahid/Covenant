import os
import shutil
import tempfile
import unittest
from pathlib import Path

from llm.openai_adapter import OpenAIClientAdapter


class TestOpenAIAdapter(unittest.TestCase):
    def test_loads_openai_api_key_from_project_env_file(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env_path = repo_root / ".env"
        original_key = os.environ.get("OPENAI_API_KEY")
        original_exists = env_path.exists()
        original_contents = env_path.read_text(encoding="utf-8") if original_exists else None
        try:
            os.environ.pop("OPENAI_API_KEY", None)
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")

            adapter = OpenAIClientAdapter(model_id="gpt-4o")

            self.assertIsNotNone(adapter._client)
            self.assertEqual(adapter.api_key, "test-key")
        finally:
            if original_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_key
            if original_exists and original_contents is not None:
                env_path.write_text(original_contents, encoding="utf-8")
            else:
                env_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
