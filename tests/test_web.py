import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from rag_app.adapters.config import Settings
from rag_app.presentation.web import create_app


class WebSmokeTests(unittest.TestCase):
    def test_local_pages_health_and_state_start_without_ai_calls(self):
        with tempfile.TemporaryDirectory() as root:
            settings = Settings(database_path=Path(root) / "test.sqlite3", openrouter_api_key="")
            app = create_app(settings=settings, start_worker=False)
            with TestClient(app) as client:
                page = client.get("/")
                self.assertEqual(page.status_code, 200)
                self.assertIn("Habla con tus vídeos", page.text)
                self.assertEqual(client.get("/health").json()["retrieval"], "hybrid")
                state = client.get("/api/state").json()
                self.assertEqual(state["channels"], [])
                self.assertEqual(state["jobs"], [])
                self.assertEqual(client.get("/static/app.js").status_code, 200)


if __name__ == "__main__":
    unittest.main()
