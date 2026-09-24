import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SQLiteStore:
    """SQLite repositories for the catalog, durable job queue, corpus and chat."""

    def __init__(self, database_path):
        self.database_path = Path(database_path)

    @contextmanager
    def _connection(self):
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self):
        with self._connection() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS channels (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_scanned_at TEXT
                );
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    duration_seconds INTEGER NOT NULL DEFAULT 0,
                    published_at TEXT,
                    status TEXT NOT NULL DEFAULT 'discovered',
                    transcript TEXT,
                    transcript_language TEXT,
                    error_message TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_videos_channel_status
                    ON videos(channel_id, status);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                    status TEXT NOT NULL DEFAULT 'pending',
                    estimated_cost_usd REAL NOT NULL,
                    actual_cost_usd REAL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_jobs_status_created
                    ON jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                    channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    timestamp_seconds INTEGER,
                    embedding_json TEXT,
                    UNIQUE(video_id, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS ix_chunks_channel_video
                    ON chunks(channel_id, video_id);
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL,
                    scope_json TEXT NOT NULL DEFAULT '{}',
                    cost_usd REAL,
                    created_at TEXT NOT NULL
                );
                PRAGMA user_version = 1;
            """)

    @staticmethod
    def _dict(row):
        return dict(row) if row is not None else None

    def save_channel(self, channel):
        now = _now()
        with self._connection() as db:
            db.execute("""INSERT INTO channels(id,url,title,created_at,last_scanned_at)
                VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                url=excluded.url,title=excluded.title,last_scanned_at=excluded.last_scanned_at""",
                (channel["id"], channel["url"], channel["title"], now, now))
        return self.get_channel(channel["id"])

    def get_channel(self, channel_id):
        with self._connection() as db:
            return self._dict(db.execute("SELECT * FROM channels WHERE id=?", (channel_id,)).fetchone())

    def get_channel_by_url(self, url):
        with self._connection() as db:
            return self._dict(db.execute("SELECT * FROM channels WHERE url=?", (url,)).fetchone())

    def list_channels(self):
        with self._connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM videos v WHERE v.channel_id=c.id) AS video_count "
                "FROM channels c ORDER BY c.title COLLATE NOCASE")]

    def mark_channel_scanned(self, channel_id):
        with self._connection() as db:
            db.execute("UPDATE channels SET last_scanned_at=? WHERE id=?", (_now(), channel_id))

    def upsert_videos(self, videos):
        inserted = 0
        with self._connection() as db:
            for video in videos:
                cursor = db.execute("""INSERT OR IGNORE INTO videos(
                    id,channel_id,title,url,duration_seconds,published_at,status,updated_at
                    ) VALUES(?,?,?,?,?,?,'discovered',?)""", (
                    video["id"], video["channel_id"], video["title"], video["url"],
                    max(0, int(video.get("duration_seconds", 0))), video.get("published_at"), _now()))
                inserted += cursor.rowcount
        return inserted

    def get_video(self, video_id):
        with self._connection() as db:
            return self._dict(db.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone())

    def list_videos(self, channel_id):
        with self._connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM videos WHERE channel_id=? ORDER BY published_at DESC, title COLLATE NOCASE",
                (channel_id,))]

    def select_videos(self, video_ids):
        ids = list(dict.fromkeys(str(value) for value in video_ids))
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        with self._connection() as db:
            return [dict(row) for row in db.execute(f"SELECT * FROM videos WHERE id IN ({marks})", ids)]

    def enqueue_videos(self, videos, estimates):
        jobs = []
        with self._connection() as db:
            for video in videos:
                existing = db.execute(
                    "SELECT 1 FROM jobs WHERE video_id=? AND status IN ('pending','running') LIMIT 1",
                    (video["id"],)).fetchone()
                if existing or video["status"] == "transcribed":
                    continue
                job_id = str(uuid4())
                estimate = float(estimates.get(video["id"], 0.0))
                db.execute("INSERT INTO jobs(id,video_id,estimated_cost_usd,created_at) VALUES(?,?,?,?)",
                           (job_id, video["id"], estimate, _now()))
                db.execute("UPDATE videos SET status='queued',error_message=NULL,updated_at=? WHERE id=?",
                           (_now(), video["id"]))
                jobs.append({"id": job_id, "video_id": video["id"], "estimated_cost_usd": estimate, "status": "pending"})
        return jobs

    def list_jobs(self, limit=100):
        with self._connection() as db:
            return [dict(row) for row in db.execute("""SELECT j.*,v.title AS video_title,v.channel_id
                FROM jobs j JOIN videos v ON v.id=j.video_id
                ORDER BY j.created_at DESC LIMIT ?""", (int(limit),))]

    def claim_next_job(self):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                return None
            now = _now()
            db.execute("UPDATE jobs SET status='running',started_at=?,error_message=NULL WHERE id=?",
                       (now, row["id"]))
            db.execute("UPDATE videos SET status='processing',error_message=NULL,updated_at=? WHERE id=?",
                       (now, row["video_id"]))
            return dict(db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())

    def mark_interrupted_jobs(self):
        """Require user approval after an interrupted in-flight provider request."""
        with self._connection() as db:
            now = _now()
            message = "Trabajo interrumpido. Revisa y vuelve a aprobar para continuar desde los datos guardados."
            db.execute("UPDATE jobs SET status='failed',finished_at=?,error_message=? WHERE status='running'",
                       (now, message))
            db.execute("UPDATE videos SET status='failed',error_message=?,updated_at=? WHERE status='processing'",
                       (message, now))

    def save_transcript_and_chunks(self, video_id, transcript, language, chunks):
        with self._connection() as db:
            video = db.execute("SELECT channel_id FROM videos WHERE id=?", (video_id,)).fetchone()
            if video is None:
                raise KeyError(video_id)
            db.execute("UPDATE videos SET transcript=?,transcript_language=?,status='processing',updated_at=? WHERE id=?",
                       (transcript, language, _now(), video_id))
            db.execute("DELETE FROM chunks WHERE video_id=?", (video_id,))
            for chunk_index, item in enumerate(chunks):
                db.execute("""INSERT INTO chunks(video_id,channel_id,chunk_index,content,timestamp_seconds)
                    VALUES(?,?,?,?,?)""", (video_id, video["channel_id"], chunk_index,
                    item["content"], item.get("timestamp_seconds")))

    def unembedded_chunks(self, video_id, limit=32):
        with self._connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT id,content FROM chunks WHERE video_id=? AND embedding_json IS NULL ORDER BY chunk_index LIMIT ?",
                (video_id, limit))]

    def save_embeddings(self, chunk_ids, vectors):
        if len(chunk_ids) != len(vectors):
            raise ValueError("Embedding count does not match chunk count")
        with self._connection() as db:
            db.executemany("UPDATE chunks SET embedding_json=? WHERE id=?", [
                (json.dumps(vector, separators=(",", ":")), chunk_id)
                for chunk_id, vector in zip(chunk_ids, vectors)])

    def add_job_cost(self, job_id, cost_usd):
        if cost_usd is None:
            return
        with self._connection() as db:
            db.execute("UPDATE jobs SET actual_cost_usd=COALESCE(actual_cost_usd,0)+? WHERE id=?",
                       (float(cost_usd), job_id))

    def finish_job(self, job_id, video_id):
        with self._connection() as db:
            now = _now()
            db.execute("UPDATE jobs SET status='completed',finished_at=? WHERE id=?", (now, job_id))
            db.execute("UPDATE videos SET status='transcribed',error_message=NULL,updated_at=? WHERE id=?",
                       (now, video_id))

    def fail_job(self, job_id, video_id, message):
        safe_message = str(message)[:500]
        with self._connection() as db:
            now = _now()
            db.execute("UPDATE jobs SET status='failed',finished_at=?,error_message=? WHERE id=?",
                       (now, safe_message, job_id))
            db.execute("UPDATE videos SET status='failed',error_message=?,updated_at=? WHERE id=?",
                       (safe_message, now, video_id))

    def list_chunks(self, channel_ids=None, video_ids=None):
        clauses = ["c.embedding_json IS NOT NULL"]
        values = []
        if video_ids:
            ids = list(dict.fromkeys(map(str, video_ids)))
            clauses.append("c.video_id IN (" + ",".join("?" for _ in ids) + ")")
            values.extend(ids)
        if channel_ids:
            ids = list(dict.fromkeys(map(str, channel_ids)))
            clauses.append("c.channel_id IN (" + ",".join("?" for _ in ids) + ")")
            values.extend(ids)
        with self._connection() as db:
            rows = db.execute("""SELECT c.*,v.title,v.url,v.duration_seconds
                FROM chunks c JOIN videos v ON v.id=c.video_id WHERE """ + " AND ".join(clauses), values)
            result = []
            for row in rows:
                item = dict(row)
                item["embedding"] = json.loads(item.pop("embedding_json"))
                result.append(item)
            return result

    def create_conversation(self, title="Vídeos"):
        conversation_id = str(uuid4())
        now = _now()
        with self._connection() as db:
            db.execute("INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                       (conversation_id, title[:120], now, now))
        return conversation_id

    def ensure_conversation(self, conversation_id=None):
        with self._connection() as db:
            if conversation_id:
                row = db.execute("SELECT id FROM conversations WHERE id=?", (conversation_id,)).fetchone()
                if row:
                    return conversation_id
        return self.create_conversation()

    def conversation_history(self, conversation_id, limit=8):
        with self._connection() as db:
            rows = db.execute("SELECT role,content FROM messages WHERE conversation_id=? "
                              "ORDER BY id DESC LIMIT ?", (conversation_id, limit))
            return [dict(row) for row in reversed(list(rows))]

    def save_message(self, conversation_id, role, content, scope, cost_usd=None):
        with self._connection() as db:
            db.execute("INSERT INTO messages(conversation_id,role,content,scope_json,cost_usd,created_at) "
                       "VALUES(?,?,?,?,?,?)", (conversation_id, role, content,
                       json.dumps(scope, ensure_ascii=False), cost_usd, _now()))
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), conversation_id))

    def conversation_messages(self, conversation_id):
        with self._connection() as db:
            messages = [dict(row) for row in db.execute(
                "SELECT role,content,scope_json,cost_usd,created_at FROM messages "
                "WHERE conversation_id=? ORDER BY id", (conversation_id,))]
            for message in messages:
                message["scope"] = json.loads(message.pop("scope_json") or "{}")
            return messages
