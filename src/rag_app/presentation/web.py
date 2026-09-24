from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from rag_app.adapters.config import get_settings
from rag_app.adapters.openrouter import OpenRouterAdapter
from rag_app.adapters.sqlite_store import SQLiteStore
from rag_app.adapters.youtube import YouTubeAdapter
from rag_app.application.catalog import CatalogService
from rag_app.application.chat import ChatService
from rag_app.application.ingestion import IngestionService
from rag_app.application.worker import BackgroundWorker
from rag_app.domain.errors import ApplicationError


PRESENTATION_DIR = Path(__file__).resolve().parent


class AddChannelRequest(BaseModel):
    url: str = Field(min_length=8, max_length=500)


class ApproveRequest(BaseModel):
    video_ids: list[str] = Field(min_length=1, max_length=100)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    channel_ids: list[str] = Field(default_factory=list, max_length=50)
    video_ids: list[str] = Field(default_factory=list, max_length=100)
    use_jev: bool = False
    verify_faithfulness: bool = True


def create_app(settings=None, start_worker=True):
    settings = settings or get_settings()
    store = SQLiteStore(settings.database_path)
    store.initialize()
    ai = OpenRouterAdapter(settings)
    youtube = YouTubeAdapter()
    catalog = CatalogService(store, youtube)
    ingestion = IngestionService(store, youtube, ai, settings)
    chat = ChatService(store, ai, settings)
    worker = BackgroundWorker(ingestion)

    @asynccontextmanager
    async def lifespan(_app):
        if start_worker:
            worker.start()
        try:
            yield
        finally:
            worker.stop()

    app = FastAPI(title="Video RAG", version="0.2.0", lifespan=lifespan)
    app.state.store = store
    app.state.settings = settings
    app.mount("/static", StaticFiles(directory=PRESENTATION_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=PRESENTATION_DIR / "templates")

    @app.exception_handler(ApplicationError)
    async def application_error(_request: Request, exc: ApplicationError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse(request, "index.html", {"max_batch_usd": settings.max_transcription_batch_usd})

    @app.get("/api/state")
    async def state():
        channels = store.list_channels()
        return {"channels": [{**channel, "videos": store.list_videos(channel["id"])} for channel in channels],
                "jobs": store.list_jobs(50), "max_batch_usd": settings.max_transcription_batch_usd}

    @app.post("/api/channels")
    async def add_channel(payload: AddChannelRequest):
        channel, videos = catalog.add_channel(payload.url)
        return {"channel": channel, "videos": videos}

    @app.post("/api/channels/{channel_id}/refresh")
    async def refresh_channel(channel_id: str):
        try:
            channel, videos = catalog.refresh_channel(channel_id)
            return {"channel": channel, "videos": videos}
        except KeyError:
            raise HTTPException(status_code=404, detail="No se encontró el canal.") from None

    @app.post("/api/ingestion/approve")
    async def approve(payload: ApproveRequest):
        return ingestion.approve(payload.video_ids)

    @app.get("/api/jobs")
    async def jobs():
        return {"jobs": store.list_jobs(100)}

    @app.post("/api/chat")
    async def ask(payload: ChatRequest):
        return chat.ask(payload.question, payload.conversation_id, payload.channel_ids,
                        payload.video_ids, payload.use_jev, payload.verify_faithfulness)

    @app.get("/api/conversations/{conversation_id}")
    async def conversation(conversation_id: str):
        return {"conversation_id": conversation_id, "messages": store.conversation_messages(conversation_id)}

    @app.get("/health")
    async def health():
        return {"status": "ok", "database": "sqlite", "retrieval": "hybrid"}

    return app


def run():
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
