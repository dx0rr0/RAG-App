import unittest

from hybrid_retrieval import (
    BM25Index,
    hybrid_search,
    reciprocal_rank_fusion,
    tokenize,
)
from retrieval_evaluation import (
    evaluate_rankings,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from tests.retrieval_benchmark import run_benchmark


class BM25Tests(unittest.TestCase):
    def test_search_ranks_matching_document_and_normalizes_accents(self):
        index = BM25Index(["la concentracion ayuda", "vitamina y huesos"])
        results = index.search("concentración", top_k=2)
        self.assertEqual(results[0][0], 0)
        self.assertGreater(results[0][1], 0)

    def test_empty_and_zero_k_queries_return_no_results(self):
        self.assertEqual(BM25Index([]).search("consulta"), [])
        self.assertEqual(BM25Index(["texto"]).search("sin coincidencias", 3), [])
        self.assertEqual(BM25Index(["texto"]).search("texto", 0), [])
        self.assertEqual(tokenize("Árbol y concentración"), ["arbol", "y", "concentracion"])


class FusionAndRerankingTests(unittest.TestCase):
    def test_reciprocal_rank_fusion_combines_rankings_and_caps_results(self):
        fused = reciprocal_rank_fusion(
            [("a", 0.1), ("b", 0.0)],
            [("b", 0.8), ("c", 0.2)],
            k=1,
            top_k=2,
        )
        self.assertEqual([document_id for document_id, _ in fused], ["b", "a"])
        self.assertGreater(fused[0][1], fused[1][1])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(["a"], top_k=-1)

    def test_hybrid_search_applies_optional_reranker_without_loading_a_model(self):
        documents = ["primero", "segundo", "tercero"]
        index = BM25Index(documents)
        calls = []

        def vector_search(query, limit):
            calls.append((query, limit))
            return [(0, 0.8), (1, 0.7), (2, 0.1)]

        def stub_reranker(query, candidates):
            self.assertEqual(query, "consulta")
            return [0.1, 0.9, 0.05][:len(candidates)]

        results = hybrid_search(
            "consulta", documents, vector_search, index,
            top_k=2, candidate_k=3, reranker=stub_reranker,
        )
        self.assertEqual([hit.index for hit in results], [1, 0])
        self.assertEqual(calls, [("consulta", 3)])

    def test_bad_reranker_output_is_rejected(self):
        with self.assertRaises(ValueError):
            hybrid_search(
                "q", ["a"], lambda query, limit: [(0, 1.0)],
                reranker=lambda query, texts: [],
            )

    def test_scalar_reranker_score_is_supported_for_one_candidate(self):
        results = hybrid_search(
            "q",
            ["only document"],
            lambda query, limit: [(0, 0.9)],
            reranker=lambda query, texts: 0.7,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].score, 0.7)

    def test_reranker_candidate_limit_must_cover_top_k(self):
        with self.assertRaisesRegex(ValueError, "at least top_k"):
            hybrid_search(
                "q",
                ["a", "b", "c"],
                lambda query, limit: [(0, 0.9), (1, 0.8), (2, 0.7)],
                top_k=2,
                candidate_k=3,
                reranker=lambda query, texts: [0.1],
                reranker_candidate_k=1,
            )

    def test_reranker_candidate_limit_caps_the_shortlist(self):
        seen = []
        results = hybrid_search(
            "q",
            ["a", "b", "c"],
            lambda query, limit: [(0, 0.9), (1, 0.8), (2, 0.7)],
            top_k=2,
            candidate_k=3,
            reranker=lambda query, texts: seen.extend(texts) or [0.1, 0.9],
            reranker_candidate_k=2,
        )
        self.assertEqual(len(seen), 2)
        self.assertEqual([hit.index for hit in results], [1, 0])


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_mrr_and_graded_ndcg(self):
        ranking = ["b", "a"]
        relevance = {"a": 3, "b": 1}
        self.assertEqual(recall_at_k(ranking, relevance, 2), 1.0)
        self.assertEqual(reciprocal_rank(ranking, relevance), 1.0)
        self.assertAlmostEqual(ndcg_at_k(ranking, relevance, 2), 0.7098097, places=6)

    def test_fixture_metrics_match_saved_pre_hybrid_baseline(self):
        metrics = run_benchmark()
        self.assertAlmostEqual(metrics["bm25"]["recall_at_k"], 1.0)
        self.assertAlmostEqual(metrics["vector"]["mrr"], 0.8333333, places=6)
        self.assertAlmostEqual(metrics["vector"]["ndcg_at_k"], 0.7935502, places=6)
        self.assertAlmostEqual(metrics["rrf"]["ndcg_at_k"], 0.9907071, places=6)
        self.assertEqual(metrics["reranker_stub"]["ndcg_at_k"], 1.0)

    def test_evaluate_rankings_handles_unanswered_query(self):
        result = evaluate_rankings({}, {"q": {"doc": 1}}, k=3)
        self.assertEqual(result["recall_at_k"], 0.0)
        self.assertEqual(result["mrr"], 0.0)
        self.assertEqual(result["ndcg_at_k"], 0.0)


if __name__ == "__main__":
    unittest.main()
