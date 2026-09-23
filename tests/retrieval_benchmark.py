"""Run the deterministic BM25/vector/fusion/stub-reranker fixture benchmark."""

import json

from hybrid_retrieval import BM25Index, reciprocal_rank_fusion, tokenize
from retrieval_evaluation import evaluate_rankings
from tests.retrieval_fixture import DOCUMENTS, DOCUMENT_IDS, QUERIES, VECTOR_SCORES


def _vector_ranking(query_id):
    scores = VECTOR_SCORES[query_id]
    return sorted(DOCUMENT_IDS, key=lambda document_id: (-scores[document_id], document_id))


def _lexical_reranker_stub(query, candidate_texts):
    query_terms = set(tokenize(query))
    return [len(query_terms & set(tokenize(text))) for text in candidate_texts]


def run_benchmark(k=3):
    bm25 = BM25Index([DOCUMENTS[document_id] for document_id in DOCUMENT_IDS])
    rankings_by_method = {name: {} for name in ("bm25", "vector", "rrf", "reranker_stub")}
    qrels = {}

    for query_id, item in QUERIES.items():
        query = item["text"]
        qrels[query_id] = item["relevance"]
        bm25_ranking = [
            DOCUMENT_IDS[index]
            for index, _ in bm25.search(query, top_k=len(DOCUMENT_IDS))
        ]
        vector_ranking = _vector_ranking(query_id)
        fused = reciprocal_rank_fusion(bm25_ranking, vector_ranking)
        fused_ranking = [document_id for document_id, _ in fused]

        candidates = fused_ranking[:k]
        candidate_texts = [DOCUMENTS[document_id] for document_id in candidates]
        rerank_scores = _lexical_reranker_stub(query, candidate_texts)
        reranked_top = [
            document_id
            for _, document_id in sorted(
                enumerate(candidates),
                key=lambda item: (-rerank_scores[item[0]], item[0]),
            )
        ]
        reranked = reranked_top + fused_ranking[k:]

        rankings_by_method["bm25"][query_id] = bm25_ranking
        rankings_by_method["vector"][query_id] = vector_ranking
        rankings_by_method["rrf"][query_id] = fused_ranking
        rankings_by_method["reranker_stub"][query_id] = reranked

    return {
        method: evaluate_rankings(rankings, qrels, k=k)
        for method, rankings in rankings_by_method.items()
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
