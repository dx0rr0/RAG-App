"""Dependency-free ranking metrics for a small, labeled retrieval set."""

import math


def _document_id(result):
    if isinstance(result, (tuple, list)) and len(result) == 2:
        return result[0]
    document_index = getattr(result, "index", None)
    if document_index is not None and not callable(document_index):
        return document_index
    return result


def _unique_ranking(ranking):
    result = []
    seen = set()
    for item in ranking:
        document_id = _document_id(item)
        if document_id not in seen:
            seen.add(document_id)
            result.append(document_id)
    return result


def recall_at_k(ranking, relevance, k=10):
    if k < 0:
        raise ValueError("k must be greater than or equal to zero")
    relevant = {document for document, grade in relevance.items() if grade > 0}
    if not relevant:
        return 0.0
    retrieved = set(_unique_ranking(ranking)[:k])
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(ranking, relevance):
    relevant = {document for document, grade in relevance.items() if grade > 0}
    if not relevant:
        return 0.0
    for rank, document in enumerate(_unique_ranking(ranking), start=1):
        if document in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking, relevance, k=10):
    if k < 0:
        raise ValueError("k must be greater than or equal to zero")
    ranked = _unique_ranking(ranking)[:k]

    def dcg(grades):
        return sum(
            (2 ** grade - 1) / math.log2(rank + 2)
            for rank, grade in enumerate(grades)
        )

    actual = dcg([relevance.get(document, 0) for document in ranked])
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


def evaluate_rankings(rankings, qrels, k=10):
    query_ids = list(qrels)
    if not query_ids:
        return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "queries": 0}

    per_query = [
        (
            recall_at_k(rankings.get(query_id, []), qrels[query_id], k),
            reciprocal_rank(rankings.get(query_id, []), qrels[query_id]),
            ndcg_at_k(rankings.get(query_id, []), qrels[query_id], k),
        )
        for query_id in query_ids
    ]
    return {
        "recall_at_k": sum(row[0] for row in per_query) / len(per_query),
        "mrr": sum(row[1] for row in per_query) / len(per_query),
        "ndcg_at_k": sum(row[2] for row in per_query) / len(per_query),
        "queries": len(per_query),
    }
