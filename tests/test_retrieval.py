import unittest

from rag_app.domain.retrieval import BM25Index, evaluate_rankings, reciprocal_rank_fusion
from tests.retrieval_benchmark import run_benchmark


class RetrievalTests(unittest.TestCase):
    def test_bm25_finds_lexical_match(self):
        index = BM25Index(["la insulina regula glucosa", "el fútbol se juega en equipo"])
        self.assertEqual(index.search("insulina glucosa", 1)[0][0], 0)

    def test_rrf_combines_rankings_and_deduplicates(self):
        ranking = reciprocal_rank_fusion(["a", "b", "a"], ["b", "a"], k=60)
        self.assertEqual([item[0] for item in ranking], ["a", "b"])
        self.assertAlmostEqual(ranking[0][1], 1 / 61 + 1 / 62)

    def test_retrieval_metrics(self):
        scores = evaluate_rankings({"q": ["relevant", "other"]}, {"q": {"relevant": 2}}, k=1)
        self.assertEqual(scores["recall_at_k"], 1.0)
        self.assertEqual(scores["mrr"], 1.0)
        self.assertEqual(scores["ndcg_at_k"], 1.0)

    def test_hybrid_fixture_is_deterministic(self):
        scores = run_benchmark()
        self.assertEqual(scores["bm25"]["queries"], 3)
        self.assertEqual(scores["bm25"]["recall_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
