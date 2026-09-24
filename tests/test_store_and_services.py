import tempfile
import unittest
from pathlib import Path

from rag_app.adapters.sqlite_store import SQLiteStore
from rag_app.application.catalog import CatalogService
from rag_app.application.chat import ChatService
from rag_app.application.chunking import transcript_chunks
from rag_app.application.ingestion import IngestionService
from rag_app.domain.errors import CostLimitError, ValidationError
from rag_app.adapters.config import Settings


class FakeYouTube:
    def __init__(self):
        self.refresh_count = 0

    def discover(self, url):
        self.refresh_count += 1
        return ({"id": "channel-1", "url": url, "title": "Canal de prueba"}, [
            {"id": "video-1", "channel_id": "channel-1", "title": "Vídeo de prueba",
             "url": "https://www.youtube.com/watch?v=video-1", "duration_seconds": 120, "published_at": None}
        ])

    def download_audio(self, _video_url, output_dir):
        return Path(output_dir) / "source.webm"


class FakeAI:
    def __init__(self):
        self.transcribe_calls = 0
        self.embed_calls = 0
        self.complete_calls = 0

    def transcribe(self, path, language="es", duration_seconds=None):
        self.transcribe_calls += 1
        return {"text": "La insulina ayuda a regular la glucosa.", "segments": [], "cost_usd": 0.00036}

    def embed(self, texts):
        self.embed_calls += 1
        vectors = [[1.0, 0.0] for _ in texts]
        return {"vectors": vectors, "cost_usd": 0.000001}

    def complete(self, messages, max_tokens=512, json_mode=False, purpose="chat"):
        self.complete_calls += 1
        if json_mode:
            return {"text": '{"faithful": true, "reason": "supported"}', "cost_usd": 0.00001}
        return {"text": "La insulina ayuda a regular la glucosa [S1].", "cost_usd": 0.00002}

    def rerank_jev(self, query, passages):
        return {"scores": [1.0] * len(passages), "cost_usd": 0.00001}


class StoreAndServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = SQLiteStore(Path(self.temporary.name) / "app.sqlite3")
        self.store.initialize()
        self.ai = FakeAI()
        self.media = FakeYouTube()
        self.settings = Settings(database_path=self.store.database_path, openrouter_api_key="",
                                 max_transcription_batch_usd=0.10)

    def tearDown(self):
        self.temporary.cleanup()

    def add_video(self):
        return CatalogService(self.store, self.media).add_channel("https://www.youtube.com/@sample")

    def test_channel_refresh_discovers_but_does_not_enqueue(self):
        channel, videos = self.add_video()
        self.assertEqual(len(videos), 1)
        self.assertEqual(videos[0]["status"], "discovered")
        self.assertEqual(self.store.list_jobs(), [])
        CatalogService(self.store, self.media).refresh_channel(channel["id"])
        self.assertEqual(len(self.store.list_jobs()), 0)

    def test_explicit_approval_queues_and_cost_cap_rejects_first(self):
        _, videos = self.add_video()
        service = IngestionService(self.store, object(), self.ai, self.settings)
        with self.assertRaises(CostLimitError):
            service.approve([videos[0]["id"]], max_batch_cost_usd=0.0001)
        self.assertEqual(self.store.list_jobs(), [])
        result = service.approve([videos[0]["id"]])
        self.assertEqual(result["queued"], 1)
        self.assertAlmostEqual(result["estimated_cost_usd"], 0.00036)

    def test_unknown_duration_cannot_bypass_cost_preflight(self):
        _, videos = self.add_video()
        with self.store._connection() as db:
            db.execute("UPDATE videos SET duration_seconds=0 WHERE id='video-1'")
        service = IngestionService(self.store, object(), self.ai, self.settings)
        with self.assertRaises(ValidationError):
            service.approve(["video-1"])
        self.assertEqual(self.store.list_jobs(), [])

    def test_audio_segments_keep_source_timestamps(self):
        chunks = transcript_chunks({"text": "hola", "segments": [
            {"start": 12.8, "text": "Primera parte."}, {"start": 16, "text": "Segunda parte."}
        ]}, max_chars=800)
        self.assertEqual(chunks[0]["timestamp_seconds"], 12)
        self.assertIn("Primera parte", chunks[0]["content"])

    def test_explicit_retry_resumes_embedding_without_retranscribing_saved_text(self):
        _, videos = self.add_video()
        self.store.enqueue_videos(videos, {"video-1": 0.00036})
        self.store.claim_next_job()
        self.store.save_transcript_and_chunks("video-1", "insulina glucosa", "es", [
            {"chunk_index": 0, "content": "insulina glucosa", "timestamp_seconds": 5}
        ])
        self.store.mark_interrupted_jobs()
        IngestionService(self.store, object(), self.ai, self.settings).approve(["video-1"])
        ingestion = IngestionService(self.store, object(), self.ai, self.settings)
        self.assertTrue(ingestion.process_next())
        self.assertEqual(self.ai.transcribe_calls, 0)
        self.assertEqual(self.ai.embed_calls, 1)
        self.assertEqual(self.store.get_video("video-1")["status"], "transcribed")

    def test_worker_transcribes_chunks_embeds_and_finishes_job(self):
        _, videos = self.add_video()
        ingestion = IngestionService(self.store, self.media, self.ai, self.settings)
        ingestion.approve(["video-1"])
        self.assertTrue(ingestion.process_next())
        video = self.store.get_video("video-1")
        self.assertEqual(video["status"], "transcribed")
        self.assertEqual(video["transcript"], "La insulina ayuda a regular la glucosa.")
        chunks = self.store.list_chunks(video_ids=["video-1"])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_index"], 0)
        self.assertEqual(self.ai.transcribe_calls, 1)

    def test_chat_cites_sources_and_checks_faithfulness(self):
        _, videos = self.add_video()
        self.store.upsert_videos(videos)
        self.store.save_transcript_and_chunks("video-1", "La insulina ayuda.", "es", [
            {"chunk_index": 0, "content": "La insulina ayuda a regular la glucosa.", "timestamp_seconds": 11}
        ])
        self.store.save_embeddings([self.store.unembedded_chunks("video-1")[0]["id"]], [[1.0, 0.0]])
        result = ChatService(self.store, self.ai, self.settings).ask("¿Qué hace la insulina?", channel_ids=["channel-1"])
        self.assertIn("[S1]", result["answer"])
        self.assertTrue(result["verified"])
        self.assertEqual(result["sources"][0]["timestamp_seconds"], 11)
        self.assertEqual(self.ai.complete_calls, 2)

    def test_chat_abstains_on_empty_library_without_generation(self):
        result = ChatService(self.store, self.ai, self.settings).ask("¿Qué contiene el vídeo?")
        self.assertTrue(result["abstained"])
        self.assertEqual(self.ai.complete_calls, 0)


if __name__ == "__main__":
    unittest.main()
