import math
import re
import unicodedata
from collections import Counter


def tokenize(text):
    normalized = unicodedata.normalize("NFKD", str(text)).casefold()
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.findall(r"[\w]+", normalized, flags=re.UNICODE)


class BM25Index:
    """Small, deterministic in-memory BM25 index for a scoped video corpus."""

    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = [str(item) for item in documents]
        self.k1 = float(k1)
        self.b = float(b)
        self.tokens = [tokenize(item) for item in self.documents]
        self.lengths = [len(item) for item in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.term_frequencies = [Counter(item) for item in self.tokens]
        document_frequency = Counter()
        for terms in self.tokens:
            document_frequency.update(set(terms))
        count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query, top_k=20):
        if top_k <= 0 or not self.documents:
            return []
        query_terms = tokenize(query)
        scored = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length_norm = self.k1 * (
                1.0 - self.b + self.b * self.lengths[index] / self.average_length
            ) if self.average_length else 0.0
            for term in query_terms:
                tf = frequencies.get(term, 0)
                if tf:
                    score += self.idf.get(term, 0.0) * (
                        tf * (self.k1 + 1.0) / (tf + length_norm)
                    )
            if score > 0:
                scored.append((index, score))
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]


def _result_id(result):
    return result[0] if isinstance(result, (tuple, list)) else result


def reciprocal_rank_fusion(*rankings, k=60, top_k=None):
    if k < 1:
        raise ValueError("k must be at least 1")
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, result in enumerate(ranking, start=1):
            document_id = _result_id(result)
            if document_id in seen:
                continue
            seen.add(document_id)
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
    return ordered[:top_k] if top_k is not None else ordered


def cosine_similarity(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _document_id(result):
    if isinstance(result, (tuple, list)) and len(result) == 2:
        return result[0]
    if isinstance(result, (str, bytes, int, float)):
        return result
    return getattr(result, "index", result)


def _unique_ranking(ranking):
    values = []
    seen = set()
    for item in ranking:
        value = _document_id(item)
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def recall_at_k(ranking, relevance, k=10):
    if k < 0:
        raise ValueError("k must be non-negative")
    relevant = {key for key, grade in relevance.items() if grade > 0}
    return len(set(_unique_ranking(ranking)[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranking, relevance):
    relevant = {key for key, grade in relevance.items() if grade > 0}
    return next((1.0 / rank for rank, key in enumerate(_unique_ranking(ranking), 1) if key in relevant), 0.0)


def ndcg_at_k(ranking, relevance, k=10):
    if k < 0:
        raise ValueError("k must be non-negative")
    ranked = _unique_ranking(ranking)[:k]
    dcg = sum((2 ** relevance.get(key, 0) - 1) / math.log2(rank + 2) for rank, key in enumerate(ranked))
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    ideal = sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal_grades))
    return dcg / ideal if ideal else 0.0


def evaluate_rankings(rankings, qrels, k=10):
    if not qrels:
        return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "queries": 0}
    values = [
        (recall_at_k(rankings.get(query, []), grades, k),
         reciprocal_rank(rankings.get(query, []), grades),
         ndcg_at_k(rankings.get(query, []), grades, k))
        for query, grades in qrels.items()
    ]
    count = len(values)
    return {
        "recall_at_k": sum(row[0] for row in values) / count,
        "mrr": sum(row[1] for row in values) / count,
        "ndcg_at_k": sum(row[2] for row in values) / count,
        "queries": count,
    }
