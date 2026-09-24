"""MCP presentation for the local YouTube RAG library."""

import json

from mcp.server import MCPServer

from rag_app.adapters.config import get_settings
from rag_app.adapters.openrouter import OpenRouterAdapter
from rag_app.adapters.sqlite_store import SQLiteStore
from rag_app.application.chat import ChatService
from rag_app.application.retrieval import RetrievalService


def _json_result(payload):
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def create_mcp_server(settings=None, store=None, ai=None, chat=None):
    """Build an injectable MCP server; importing this module has no side effects."""
    settings = settings or get_settings()
    store = store or SQLiteStore(settings.database_path)
    store.initialize()
    ai = ai or OpenRouterAdapter(settings)
    chat = chat or ChatService(store, ai, settings)
    retrieval = RetrievalService(store, ai, settings)
    server = MCPServer("video-rag")

    @server.tool()
    def list_channels() -> str:
        """List saved YouTube channels and video counts. This reads local SQLite and makes no AI/API calls."""
        return _json_result(store.list_channels())

    @server.tool()
    def list_channel_videos(channel_id: str, limit: int = 50) -> str:
        """List videos saved for a channel (maximum 100). Local SQLite only; no AI/API calls."""
        if not channel_id or len(channel_id) > 200:
            raise ValueError("channel_id must contain 1 to 200 characters")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        channel = store.get_channel(channel_id)
        if channel is None:
            return _json_result({"error": "channel_not_found", "channel_id": channel_id})
        videos = store.list_videos(channel_id)[:limit]
        return _json_result({"channel": channel, "videos": videos})

    @server.tool()
    def get_video_transcript(video_id: str, max_chars: int = 20000) -> str:
        """Read a saved transcript from local SQLite (maximum 20,000 characters; no AI/API calls)."""
        if not video_id or len(video_id) > 200:
            raise ValueError("video_id must contain 1 to 200 characters")
        if max_chars < 1 or max_chars > 20000:
            raise ValueError("max_chars must be between 1 and 20,000")
        video = store.get_video(video_id)
        if video is None:
            return _json_result({"error": "video_not_found", "video_id": video_id})
        transcript = video.get("transcript")
        if transcript is None:
            return _json_result({"error": "transcript_not_available", "video_id": video_id,
                                 "status": video.get("status")})
        return _json_result({"video_id": video_id, "title": video["title"], "url": video["url"],
                             "transcript": transcript[:max_chars],
                             "truncated": len(transcript) > max_chars})

    @server.tool()
    def search_transcripts(query: str, channel_ids: list[str] | None = None,
                           video_ids: list[str] | None = None, limit: int = 5) -> str:
        """Search transcript chunks with BM25 + BGE-M3 + RRF.

        Uses one OpenRouter BGE-M3 embedding request, which may incur a small
        API charge. Optionally restrict the search to channel or video IDs.
        Returns source links and timestamps.
        """
        query = str(query or "").strip()
        if not query or len(query) > 2000:
            raise ValueError("query must contain 1 to 2,000 characters")
        if limit < 1 or limit > 10:
            raise ValueError("limit must be between 1 and 10")
        if channel_ids is not None and len(channel_ids) > 50:
            raise ValueError("channel_ids accepts at most 50 values")
        if video_ids is not None and len(video_ids) > 100:
            raise ValueError("video_ids accepts at most 100 values")
        result = retrieval.retrieve(query, channel_ids=channel_ids, video_ids=video_ids)
        sources = [{"title": item["title"], "url": item["url"],
                    "timestamp_seconds": item.get("timestamp_seconds"),
                    "similarity": round(float(item.get("similarity", 0.0)), 4),
                    "content": item["content"]}
                   for item in result["items"][:limit]]
        return _json_result({"query": query, "sources": sources,
                             "embedding_model": settings.embedding_model,
                             "cost_usd": result.get("cost_usd"),
                             "cost_is_estimate": result.get("cost_is_estimate", False)})

    @server.tool()
    def ask_video_library(question: str, channel_ids: list[str] | None = None,
                          video_ids: list[str] | None = None,
                          verify_faithfulness: bool = False) -> str:
        """Answer from transcripts with citations and abstain when evidence is insufficient.

        Uses OpenRouter BGE-M3 and GPT-6 Luna (low), which may incur API
        charges. Set verify_faithfulness=true for one additional model request.
        Optionally restrict the answer to channel or video IDs.
        """
        question = str(question or "").strip()
        if not question or len(question) > 2000:
            raise ValueError("question must contain 1 to 2,000 characters")
        if channel_ids is not None and len(channel_ids) > 50:
            raise ValueError("channel_ids accepts at most 50 values")
        if video_ids is not None and len(video_ids) > 100:
            raise ValueError("video_ids accepts at most 100 values")
        answer = chat.ask(question, channel_ids=channel_ids, video_ids=video_ids,
                          verify_faithfulness=verify_faithfulness)
        return _json_result({**answer, "answer_model": settings.chat_model,
                             "embedding_model": settings.embedding_model,
                             "faithfulness_check_requested": verify_faithfulness})

    return server


def run():
    """Run the local stdio server for an MCP host to launch as a subprocess."""
    create_mcp_server().run()


if __name__ == "__main__":
    run()
