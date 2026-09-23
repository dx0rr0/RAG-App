"""Optional, dependency-light BM25, reciprocal-rank fusion, and reranking."""

from dataclasses import dataclass
import math
import re
import unicodedata


def tokenize(text):
    normalized = unicodedata.normalize("NFKD", str(text).casefold())
    unaccented = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", unaccented)


class BM25Index:
    """Small in-memory BM25 index; it has no model or network dependencies."""

    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = [str(document) for document in documents]
        self.tokens = [tokenize(document) for document in self.documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )
        self.k1 = float(k1)
        self.b = float(b)
        self.document_frequency = {}
        self.term_frequencies = []
        for tokens in self.tokens:
            frequencies = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.term_frequencies.append(frequencies)
            for token in frequencies:
                self.document_frequency[token] = (
                    self.document_frequency.get(token, 0) + 1
                )

    def scores(self, query):
        query_tokens = tokenize(query)
        count = len(self.documents)
        scores = [0.0] * count
        if not count or not query_tokens:
            return scores

        for token in query_tokens:
            frequency_in_corpus = self.document_frequency.get(token, 0)
            if not frequency_in_corpus:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (count - frequency_in_corpus + 0.5)
                / (frequency_in_corpus + 0.5)
            )
            for index, frequencies in enumerate(self.term_frequencies):
                term_frequency = frequencies.get(token, 0)
                if not term_frequency:
                    continue
                length_norm = 1.0 - self.b + self.b * (
                    self.lengths[index] / self.average_length
                    if self.average_length
                    else 0.0
                )
                scores[index] += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1.0)
                ) / (term_frequency + self.k1 * length_norm)
        return scores

    def search(self, query, top_k=10):
        if top_k < 0:
            raise ValueError("top_k must be greater than or equal to zero")
        if top_k == 0:
            return []
        scores = self.scores(query)
        ranked = sorted(
            enumerate(scores),
            key=lambda result: (-result[1], result[0]),
        )
        return [
            (index, score)
            for index, score in ranked
            if score > 0.0
        ][:top_k]


def _result_index(result):
    if isinstance(result, (tuple, list)) and len(result) == 2:
        return result[0]
    document_index = getattr(result, "index", None)
    if document_index is not None and not callable(document_index):
        return document_index
    return result


def reciprocal_rank_fusion(*rankings, k=60, top_k=None):
    if k <= 0:
        raise ValueError("RRF constant k must be positive")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be greater than or equal to zero")
    scores = {}
    first_seen = {}
    for ranking in rankings:
        seen_in_ranking = set()
        for rank, result in enumerate(ranking, start=1):
            document_id = _result_index(result)
            if document_id in seen_in_ranking:
                continue
            seen_in_ranking.add(document_id)
            first_seen.setdefault(document_id, len(first_seen))
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(
        scores.items(),
        key=lambda result: (-result[1], first_seen[result[0]]),
    )
    return fused if top_k is None else fused[:max(0, top_k)]


@dataclass(frozen=True)
class RankedDocument:
    index: int
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    fusion_score: float = 0.0


def hybrid_search(
    query,
    documents,
    vector_search_fn,
    bm25_index=None,
    top_k=5,
    candidate_k=20,
    rrf_constant=60,
    reranker=None,
):
    """Fuse vector and lexical candidate ranks, then optionally rerank candidates.

    ``reranker`` is a callable ``(query, candidate_texts) -> scores``. It is
    invoked only when supplied, keeping the default path model-free.
    """
    if top_k < 0 or candidate_k < 0:
        raise ValueError("top_k and candidate_k must be non-negative")
    if top_k == 0 or not documents:
        return []

    candidate_limit = min(len(documents), max(top_k, candidate_k))
    vector_results = list(vector_search_fn(query, candidate_limit) or [])
    lexical_results = (
        bm25_index.search(query, top_k=candidate_limit)
        if bm25_index is not None
        else []
    )
    fused = reciprocal_rank_fusion(
        lexical_results,
        vector_results,
        k=rrf_constant,
        top_k=candidate_limit,
    )
    vector_scores = {int(index): float(score) for index, score in vector_results}
    lexical_scores = {int(index): float(score) for index, score in lexical_results}

    if reranker is not None and fused:
        candidate_indices = [int(index) for index, _ in fused]
        candidate_texts = [str(documents[index]) for index in candidate_indices]
        raw_scores = reranker(query, candidate_texts)
        try:
            rerank_scores = list(raw_scores)
        except TypeError:
            rerank_scores = [raw_scores]
        if len(rerank_scores) != len(candidate_indices):
            raise ValueError("Reranker must return one score per candidate")
        scored = []
        for order, (index, score) in enumerate(zip(candidate_indices, rerank_scores)):
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("Reranker scores must be finite numbers")
            scored.append((index, score, order))
        scored.sort(key=lambda item: (-item[1], item[2]))
        fused_score_map = dict(fused)
        return [
            RankedDocument(
                index=index,
                score=score,
                vector_score=vector_scores.get(index),
                bm25_score=lexical_scores.get(index),
                fusion_score=fused_score_map[index],
            )
            for index, score, _ in scored[:top_k]
        ]

    return [
        RankedDocument(
            index=int(index),
            score=float(score),
            vector_score=vector_scores.get(int(index)),
            bm25_score=lexical_scores.get(int(index)),
            fusion_score=float(score),
        )
        for index, score in fused[:top_k]
    ]


def load_cross_encoder(model_name, device="auto"):
    """Load an optional sentence-transformers cross-encoder on explicit request."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Cross-encoder reranking needs sentence-transformers; install requirements.txt."
        ) from exc

    from vector_db_manager import resolve_device

    model = CrossEncoder(model_name, device=resolve_device(device))

    def score_pairs(query, candidate_texts):
        pairs = [(query, text) for text in candidate_texts]
        return model.predict(pairs)

    return score_pairs
