from rag_app.domain.retrieval import BM25Index, cosine_similarity, reciprocal_rank_fusion


class RetrievalService:
    def __init__(self, store, ai, settings):
        self.store, self.ai, self.settings = store, ai, settings

    def retrieve(self, query, channel_ids=None, video_ids=None, use_jev=False):
        documents = self.store.list_chunks(channel_ids=channel_ids, video_ids=video_ids)
        if not documents:
            return {"items": [], "similarity": 0.0, "cost_usd": 0.0}
        query_vector = self.ai.embed([query])
        q = query_vector["vectors"][0]
        lexical = BM25Index([item["content"] for item in documents])
        bm25 = [index for index, _ in lexical.search(query, self.settings.candidate_k)]
        dense = sorted(range(len(documents)), key=lambda i: cosine_similarity(q, documents[i]["embedding"]), reverse=True)[:self.settings.candidate_k]
        fused = reciprocal_rank_fusion(bm25, dense, top_k=self.settings.candidate_k)
        ranked = [documents[index] for index, _score in fused]
        for item in ranked:
            item["similarity"] = cosine_similarity(q, item["embedding"])
        rerank_cost = 0.0
        rerank_is_estimate = False
        if use_jev and ranked:
            result = self.ai.rerank_jev(query, [item["content"] for item in ranked[:9]])
            rerank_cost = float(result.get("cost_usd") or 0.0)
            rerank_is_estimate = bool(result.get("cost_is_estimate"))
            scores = result["scores"]
            reranked = list(zip(ranked[:len(scores)], scores))
            reranked.sort(key=lambda pair: (-pair[1], -pair[0]["similarity"]))
            ranked = [item for item, _score in reranked] + ranked[len(scores):]
        selected = ranked[:self.settings.result_k]
        return {"items": selected, "similarity": max((item["similarity"] for item in selected), default=0.0),
                "cost_usd": float(query_vector.get("cost_usd") or 0.0) + rerank_cost,
                "cost_is_estimate": bool(query_vector.get("cost_is_estimate")) or rerank_is_estimate}
