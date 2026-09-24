import tempfile
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_app.adapters.config import Settings
from rag_app.adapters.openrouter import OpenRouterAdapter
from rag_app.adapters.youtube import YouTubeAdapter
from rag_app.domain.validation import validate_youtube_channel_url
from rag_app.domain.errors import ProviderError, ValidationError


class OpenRouterAdapterTests(unittest.TestCase):
    def settings(self, root, key="test-key"):
        return Settings(database_path=Path(root) / "app.sqlite3", openrouter_api_key=key)

    def test_remote_models_and_reasoning_effort_are_configured(self):
        with tempfile.TemporaryDirectory() as root:
            settings = self.settings(root)
            self.assertEqual(settings.transcription_model, "qwen/qwen3-asr-0.6b")
            self.assertEqual(settings.embedding_model, "baai/bge-m3")
            self.assertEqual(settings.chat_model, "openai/gpt-6-luna")
            self.assertEqual(settings.reasoning_effort, "low")

    def test_embedding_adapter_validates_and_orders_remote_vectors(self):
        with tempfile.TemporaryDirectory() as root:
            adapter = OpenRouterAdapter(self.settings(root))
            adapter._post_json = lambda *_args, **_kwargs: {"data": [
                {"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}
            ], "usage": {"cost": 0.000001}}
            result = adapter.embed(["uno", "dos"])
            self.assertEqual(result["vectors"], [[1, 0], [0, 1]])
            self.assertAlmostEqual(result["cost_usd"], 0.000001)

    def test_transcription_uses_remote_qwen_endpoint_with_estimated_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            adapter = OpenRouterAdapter(self.settings(root))
            audio = Path(root) / "sample.wav"
            audio.write_bytes(b"fake-audio")
            calls = []

            def fake_post(endpoint, payload, timeout=None):
                calls.append((endpoint, payload, timeout))
                return {"text": "transcripción de prueba", "segments": [{"start": 0, "text": "prueba"}]}

            adapter._post_json = fake_post
            result = adapter.transcribe(audio, language="es", duration_seconds=10)
            self.assertTrue(calls[0][0].endswith("/audio/transcriptions"))
            self.assertEqual(calls[0][1]["model"], "qwen/qwen3-asr-0.6b")
            self.assertEqual(calls[0][1]["input_audio"]["format"], "wav")
            self.assertEqual(result["text"], "transcripción de prueba")
            self.assertTrue(result["cost_is_estimate"])
            self.assertAlmostEqual(result["cost_usd"], 0.00003)

    def test_chat_and_jev_use_remote_models_and_jev_cache(self):
        with tempfile.TemporaryDirectory() as root:
            adapter = OpenRouterAdapter(self.settings(root))
            requests = []

            def fake_post(endpoint, payload, timeout=None):
                requests.append((endpoint, payload))
                if endpoint.endswith("/chat/completions"):
                    return {"choices": [{"message": {"content": "Respuesta [S1]."}}], "usage": {"cost": 0.0001}}
                return {"answers": {"candidate_0": {"noul": 0.9}}}

            adapter._post_json = fake_post
            result = adapter.complete([{"role": "user", "content": "hola"}], max_tokens=123)
            self.assertEqual(requests[0][1]["model"], "openai/gpt-6-luna")
            self.assertEqual(requests[0][1]["reasoning_effort"], "low")
            self.assertEqual(requests[0][1]["max_tokens"], 123)
            self.assertIn("Respuesta", result["text"])
            first = adapter.rerank_jev("pregunta", ["pasaje candidato"])
            cached = adapter.rerank_jev("pregunta", ["pasaje candidato"])
            self.assertAlmostEqual(first["scores"][0], 0.9)
            self.assertTrue(cached["cache_hit"])
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[1][1]["model"], "typesafe/jev-1.13")

    def test_no_key_fails_before_network(self):
        with tempfile.TemporaryDirectory() as root:
            adapter = OpenRouterAdapter(self.settings(root, key=""))
            with self.assertRaises(ProviderError):
                adapter._post_json("https://invalid.local", {})

    def test_adapter_imports_do_not_load_local_inference_frameworks(self):
        from rag_app.adapters import openrouter
        self.assertIsNotNone(openrouter.OpenRouterAdapter)
        self.assertFalse(any(name in sys.modules for name in ("torch", "transformers", "faster_whisper", "sentence_transformers")))

    def test_youtube_url_validation(self):
        self.assertEqual(validate_youtube_channel_url("https://www.youtube.com/@example/"),
                         "https://www.youtube.com/@example")
        with self.assertRaises(ValidationError):
            validate_youtube_channel_url("https://example.com/@channel")

    def test_youtube_catalog_adapter_maps_video_metadata_without_network(self):
        class FakeChannel:
            channel_id = "channel-id"
            channel_name = "Sample channel"
            videos = [SimpleNamespace(video_id="video-id", title="Sample video", length=42,
                                      publish_date=None)]

            def __init__(self, _url):
                pass

        fake_module = SimpleNamespace(Channel=FakeChannel)
        with patch.dict(sys.modules, {"pytubefix": fake_module}):
            channel, videos = YouTubeAdapter().discover("https://www.youtube.com/@example")
        self.assertEqual(channel["id"], "channel-id")
        self.assertEqual(channel["title"], "Sample channel")
        self.assertEqual(videos[0]["url"], "https://www.youtube.com/watch?v=video-id")
        self.assertEqual(videos[0]["duration_seconds"], 42)


if __name__ == "__main__":
    unittest.main()
