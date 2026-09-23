import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cli_interface import build_parser, main
from hybrid_retrieval import BM25Index, hybrid_search
from jev_reranker import (
    JEV_ENDPOINT,
    JEV_MODEL,
    JevReranker,
    JevRerankerError,
)


class _Response:
    def __init__(self, result):
        self.result = result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.result).encode("utf-8")


class JevRerankerTests(unittest.TestCase):
    def make_reranker(self, directory, **kwargs):
        return JevReranker(
            api_key="unit-test-secret",
            cache_path=Path(directory) / "jev-cache.json",
            **kwargs,
        )

    def test_posts_one_batch_and_maps_noul_scores_in_candidate_order(self):
        response = _Response({
            "answers": {
                "candidate_0": {"noul": 0.12},
                "candidate_1": {"noul": 0.91},
            },
            "usage": {"input_tokens": 100, "cost": 0.0000042},
        })
        with tempfile.TemporaryDirectory() as directory, patch(
            "jev_reranker.urllib.request.urlopen", return_value=response
        ) as urlopen:
            reranker = self.make_reranker(directory)
            scores = reranker("¿Qué pasó?", ["pasaje uno", "pasaje dos"])

            self.assertEqual(scores, [0.12, 0.91])
            self.assertEqual(urlopen.call_count, 1)
            request = urlopen.call_args.args[0]
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(request.full_url, JEV_ENDPOINT)
            self.assertEqual(payload["model"], JEV_MODEL)
            self.assertEqual(payload["state"], {"query": "¿Qué pasó?"})
            self.assertEqual(
                payload["questions"]["candidate_1"]["instructions"]["candidate_passage"],
                "pasaje dos",
            )
            self.assertEqual(urlopen.call_args.kwargs["timeout"], 45)
            self.assertEqual(reranker.last_cost_usd, 0.0000042)
            self.assertEqual(reranker.last_cost_source, "reported")
            self.assertFalse(reranker.last_cache_hit)

    def test_cache_reuses_scores_without_storing_query_or_passages(self):
        response = _Response({
            "answers": {"candidate_0": {"noul": 0.8}},
            "usage": {"cost": 0.000001},
        })
        with tempfile.TemporaryDirectory() as directory, patch(
            "jev_reranker.urllib.request.urlopen", return_value=response
        ) as urlopen:
            first = self.make_reranker(directory)
            self.assertEqual(first("pregunta privada", ["texto privado"]), [0.8])
            self.assertEqual(first.last_cost_usd, 0.000001)
            second = self.make_reranker(directory)
            self.assertEqual(second("pregunta privada", ["texto privado"]), [0.8])
            self.assertTrue(second.last_cache_hit)
            self.assertIsNone(second.last_cost_usd)
            self.assertEqual(urlopen.call_count, 1)

            cache_text = (Path(directory) / "jev-cache.json").read_text(encoding="utf-8")
            self.assertNotIn("pregunta privada", cache_text)
            self.assertNotIn("texto privado", cache_text)
            self.assertNotIn("unit-test-secret", cache_text)
            self.assertIn("0.8", cache_text)

    def test_estimated_cost_limit_stops_before_network(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "jev_reranker.urllib.request.urlopen"
        ) as urlopen:
            reranker = self.make_reranker(directory, max_estimated_cost_usd=0.0)
            with self.assertRaisesRegex(JevRerankerError, "límite por consulta"):
                reranker("consulta", ["evidencia"])
            urlopen.assert_not_called()

    def test_invalid_paid_response_is_cached_to_prevent_automatic_rebilling(self):
        response = _Response({
            "answers": {"candidate_0": {"noul": 1.5}},
            "usage": {"cost": 0.000001},
        })
        with tempfile.TemporaryDirectory() as directory, patch(
            "jev_reranker.urllib.request.urlopen", return_value=response
        ) as urlopen:
            reranker = self.make_reranker(directory)
            with self.assertRaisesRegex(JevRerankerError, "score inválido"):
                reranker("q", ["doc"])
            with self.assertRaisesRegex(JevRerankerError, "no se repite"):
                reranker("q", ["doc"])
            self.assertEqual(urlopen.call_count, 1)

    def test_http_error_is_not_retried(self):
        from urllib.error import HTTPError

        error = HTTPError(JEV_ENDPOINT, 429, "rate limited", {}, None)
        with tempfile.TemporaryDirectory() as directory, patch(
            "jev_reranker.urllib.request.urlopen", side_effect=error
        ) as urlopen:
            reranker = self.make_reranker(directory)
            with self.assertRaisesRegex(JevRerankerError, "HTTP 429"):
                reranker("q", ["doc"])
            self.assertEqual(urlopen.call_count, 1)

    def test_hybrid_search_reranks_only_configured_shortlist(self):
        documents = ["primero", "segundo", "tercero", "cuarto"]
        index = BM25Index(documents)
        seen = []

        def rerank(query, texts):
            seen.extend(texts)
            return list(reversed(range(len(texts))))

        results = hybrid_search(
            "consulta",
            documents,
            lambda _query, _limit: [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)],
            bm25_index=index,
            top_k=2,
            candidate_k=4,
            reranker=rerank,
            reranker_candidate_k=2,
        )
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(results), 2)

    def test_cli_keeps_jev_explicit_and_requires_hybrid_mode(self):
        defaults = build_parser().parse_args([])
        self.assertEqual(defaults.reranker, "none")
        with self.assertRaisesRegex(SystemExit, "requires --retrieval-mode hybrid"):
            main(["--reranker", "jev"])


if __name__ == "__main__":
    unittest.main()
