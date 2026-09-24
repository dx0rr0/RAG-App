"""Run a deterministic mechanics benchmark. It makes no network requests."""

import json

from rag_app.domain.retrieval import BM25Index, evaluate_rankings, reciprocal_rank_fusion, tokenize
from tests.retrieval_fixture import DOCUMENTS, DOCUMENT_IDS, QUERIES, VECTOR_SCORES


def _vector_ranking(query_id):
    scores = VECTOR_SCORES[query_id]
    return sorted(DOCUMENT_IDS, key=lambda document_id: (-scores[document_id], document_id))


def _overlap_reranker(query, candidate_texts):
    terms = set(tokenize(query))
    return [len(terms & set(tokenize(text))) for text in candidate_texts]


def run_benchmark(k=3):
    bm25_index = BM25Index([DOCUMENTS[key] for key in DOCUMENT_IDS])
    methods = {name: {} for name in ("bm25", "vector", "rrf", "reranker_stub")}
    qrels = {}
    for query_id, sample in QUERIES.items():
        query = sample["text"]
        qrels[query_id] = sample["relevance"]
        bm25 = [DOCUMENT_IDS[index] for index, _score in bm25_index.search(query, len(DOCUMENT_IDS))]
        vector = _vector_ranking(query_id)
        fused = [key for key, _score in reciprocal_rank_fusion(bm25, vector)]
        candidates = fused[:k]
        scores = _overlap_reranker(query, [DOCUMENTS[key] for key in candidates])
        reranked = [key for _, key in sorted(enumerate(candidates), key=lambda row: (-scores[row[0]], row[0]))] + fused[k:]
        methods["bm25"][query_id], methods["vector"][query_id] = bm25, vector
        methods["rrf"][query_id], methods["reranker_stub"][query_id] = fused, reranked
    return {name: evaluate_rankings(ranking, qrels, k=k) for name, ranking in methods.items()}


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
