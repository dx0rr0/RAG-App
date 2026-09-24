import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_local_env():
    for path in (Path.cwd() / ".env", PROJECT_ROOT / ".env"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line in lines:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if value.startswith("export "):
                value = value[7:].lstrip()
            name, separator, content = value.partition("=")
            allowed = {"OPENROUTER_API_KEY", "OPENROUTER_APIKEY", "RAG_APP_DATABASE", "RAG_MAX_BATCH_USD"}
            if not separator or name.strip() not in allowed:
                continue
            content = content.strip()
            if len(content) >= 2 and content[0] == content[-1] and content[0] in "\"'":
                content = content[1:-1]
            os.environ.setdefault(name.strip(), content)


@dataclass(frozen=True)
class Settings:
    database_path: Path
    openrouter_api_key: str
    chat_model: str = "openai/gpt-6-luna"
    embedding_model: str = "baai/bge-m3"
    transcription_model: str = "qwen/qwen3-asr-0.6b"
    reasoning_effort: str = "low"
    max_transcription_batch_usd: float = 0.10
    asr_usd_per_second: float = 0.000003
    embedding_usd_per_million_tokens: float = 0.01
    max_audio_bytes: int = 25 * 1024 * 1024
    chunk_size: int = 800
    chunk_overlap: int = 80
    vector_threshold: float = 0.35
    candidate_k: int = 20
    result_k: int = 5
    jev_max_estimated_usd: float = 0.001


def get_settings():
    _load_local_env()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip() or os.environ.get("OPENROUTER_APIKEY", "").strip()
    database = Path(os.environ.get("RAG_APP_DATABASE", str(PROJECT_ROOT / "data" / "rag_app.sqlite3")))
    try:
        max_batch = float(os.environ.get("RAG_MAX_BATCH_USD", "0.10"))
    except ValueError:
        max_batch = 0.10
    return Settings(database_path=database, openrouter_api_key=key,
                    max_transcription_batch_usd=max(0.0, min(max_batch, 7.0)))
