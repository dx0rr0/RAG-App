from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Channel:
    id: str
    url: str
    title: str
    last_scanned_at: Optional[str] = None


@dataclass(frozen=True)
class Video:
    id: str
    channel_id: str
    title: str
    url: str
    duration_seconds: int
    status: str = "discovered"
    published_at: Optional[str] = None


@dataclass(frozen=True)
class Chunk:
    id: int
    video_id: str
    channel_id: str
    title: str
    url: str
    content: str
    chunk_index: int
    timestamp_seconds: Optional[int]
    embedding: Optional[list[float]]


@dataclass(frozen=True)
class AIResult:
    text: str
    cost_usd: Optional[float]
    usage: dict
    generation_id: Optional[str] = None

\n