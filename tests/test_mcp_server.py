import json
import tempfile
import unittest
from pathlib import Path

from mcp import Client

from rag_app.adapters.config import Settings
from rag_app.adapters.sqlite_store import SQLiteStore
from rag_app.application.catalog import CatalogService
from rag_app.presentation.mcp_server import create_mcp_server
from tests.test_store_and_services import FakeAI, FakeYouTube


class MCPServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = SQLiteStore(Path(self.temporary.name) / "mcp.sqlite3")
        self.store.initialize()
        self.settings = Settings(database_path=self.store.database_path,
                                 openrouter_api_key="", max_transcription_batch_usd=0.10)
        self.ai = FakeAI()
        self.server = create_mcp_server(self.settings, self.store, self.ai)

    async def asyncTearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def parse(result):
        return json.loads(result.content[0].text)

    async def test_catalog_tools_read_local_sqlite_without_ai_calls(self):
        channel, videos = CatalogService(self.store, FakeYouTube()).add_channel(
            "https://www.youtube.com/@sample")
        async with Client(self.server) as client:
            listed_channels = await client.call_tool("list_channels", {})
            listed_videos = await client.call_tool("list_channel_videos", {
                "channel_id": channel["id"]
            })
        self.assertEqual(self.parse(listed_channels)[0]["title"], "Canal de prueba")
        self.assertEqual(self.parse(listed_videos)["videos"][0]["id"], videos[0]["id"])
        self.assertEqual(self.ai.embed_calls, 0)
        self.assertEqual(self.ai.complete_calls, 0)

    async def test_transcript_tool_is_local_and_marks_truncation(self):
        _, videos = CatalogService(self.store, FakeYouTube()).add_channel(
            "https://www.youtube.com/@sample")
        self.store.save_transcript_and_chunks(videos[0]["id"], "Transcripción completa.", "es", [])
        async with Client(self.server) as client:
            result = await client.call_tool("get_video_transcript", {
                "video_id": videos[0]["id"], "max_chars": 5
            })
        self.assertEqual(self.parse(result)["transcript"], "Trans")
        self.assertTrue(self.parse(result)["truncated"])
        self.assertEqual(self.ai.embed_calls, 0)

    async def test_search_and_chat_use_existing_services_with_cost_metadata(self):
        _, videos = CatalogService(self.store, FakeYouTube()).add_channel(
            "https://www.youtube.com/@sample")
        self.store.save_transcript_and_chunks(videos[0]["id"], "La insulina ayuda.", "es", [
            {"chunk_index": 0, "content": "La insulina ayuda a regular la glucosa.",
             "timestamp_seconds": 11}
        ])
        self.store.save_embeddings([self.store.unembedded_chunks(videos[0]["id"])[0]["id"]],
                                   [[1.0, 0.0]])
        async with Client(self.server) as client:
            tools = await client.list_tools()
            search = await client.call_tool("search_transcripts", {
                "query": "insulina glucosa", "video_ids": [videos[0]["id"]]
            })
            answer = await client.call_tool("ask_video_library", {
                "question": "¿Qué hace la insulina?", "video_ids": [videos[0]["id"]],
                "verify_faithfulness": False
            })
        self.assertEqual({tool.name for tool in tools.tools}, {
            "list_channels", "list_channel_videos", "get_video_transcript",
            "search_transcripts", "ask_video_library"
        })
        search_tool = next(tool for tool in tools.tools if tool.name == "search_transcripts")
        self.assertIn("OpenRouter", search_tool.description)
        search_payload = self.parse(search)
        self.assertEqual(search_payload["embedding_model"], "baai/bge-m3")
        self.assertEqual(search_payload["sources"][0]["timestamp_seconds"], 11)
        self.assertAlmostEqual(search_payload["cost_usd"], 0.000001)
        answer_payload = self.parse(answer)
        self.assertIn("[S1]", answer_payload["answer"])
        self.assertEqual(answer_payload["answer_model"], "openai/gpt-6-luna")
        self.assertAlmostEqual(answer_payload["cost_usd"], 0.000021)
        self.assertFalse(answer_payload["faithfulness_check_requested"])
        self.assertEqual(self.ai.embed_calls, 2)
        self.assertEqual(self.ai.complete_calls, 1)

    async def test_empty_library_chat_abstains_without_model_generation(self):
        async with Client(self.server) as client:
            answer = await client.call_tool("ask_video_library", {
                "question": "¿Qué dice el canal?"
            })
        self.assertTrue(self.parse(answer)["abstained"])
        self.assertEqual(self.ai.complete_calls, 0)


if __name__ == "__main__":
    unittest.main()
