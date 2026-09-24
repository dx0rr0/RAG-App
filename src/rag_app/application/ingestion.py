import tempfile
from pathlib import Path

from rag_app.domain.errors import CostLimitError, NotFoundError, ValidationError
from rag_app.application.chunking import transcript_chunks


class IngestionService:
    def __init__(self, store, media, ai, settings):
        self.store, self.media, self.ai, self.settings = store, media, ai, settings

    def estimate_video(self, video):
        if video.get("transcript"):
            return 0.0
        return round(max(0, int(video.get("duration_seconds") or 0)) * self.settings.asr_usd_per_second, 8)

    def approve(self, video_ids, max_batch_cost_usd=None):
        ids = list(dict.fromkeys(str(value) for value in video_ids))
        if not ids or len(ids) > 100:
            raise ValidationError("Selecciona entre 1 y 100 vídeos.")
        videos = self.store.select_videos(ids)
        if len(videos) != len(ids):
            raise NotFoundError("Algún vídeo seleccionado ya no existe.")
        if any(not video.get("transcript") and int(video.get("duration_seconds") or 0) <= 0 for video in videos):
            raise ValidationError("No se puede estimar el coste de un vídeo sin duración conocida; omítelo por seguridad.")
        cap = self.settings.max_transcription_batch_usd if max_batch_cost_usd is None else float(max_batch_cost_usd)
        if cap < 0 or cap > self.settings.max_transcription_batch_usd:
            raise ValidationError(f"El límite no puede superar ${self.settings.max_transcription_batch_usd:.2f} por lote.")
        estimates = {video["id"]: self.estimate_video(video) for video in videos}
        total = sum(estimates.values())
        if total > cap:
            raise CostLimitError(f"Estimación del lote ${total:.4f}; supera el límite ${cap:.2f}. Reduce la selección o ajusta RAG_MAX_BATCH_USD.")
        jobs = self.store.enqueue_videos(videos, estimates)
        return {"jobs": jobs, "estimated_cost_usd": total, "selected": len(videos), "queued": len(jobs)}

    def process_next(self):
        job = self.store.claim_next_job()
        if job is None:
            return False
        video_id = job["video_id"]
        actual_cost = 0.0
        try:
            video = self.store.get_video(video_id)
            if video is None:
                raise NotFoundError(f"No existe el vídeo {video_id}.")
            if not video.get("transcript"):
                with tempfile.TemporaryDirectory(prefix="rag-app-") as temporary_dir:
                    audio = self.media.download_audio(video["url"], Path(temporary_dir))
                    result = self.ai.transcribe(audio, language="es", duration_seconds=video["duration_seconds"])
                actual_cost += float(result.get("cost_usd") or self.estimate_video(video))
                chunks = transcript_chunks(result, self.settings.chunk_size, self.settings.chunk_overlap)
                self.store.save_transcript_and_chunks(video_id, result["text"], "es", chunks)
                self.store.add_job_cost(job["id"], actual_cost)
            while True:
                unembedded = self.store.unembedded_chunks(video_id, limit=32)
                if not unembedded:
                    break
                embedding_result = self.ai.embed([row["content"] for row in unembedded])
                vectors = embedding_result["vectors"]
                self.store.save_embeddings([row["id"] for row in unembedded], vectors)
                embedding_cost = embedding_result.get("cost_usd")
                if embedding_cost is not None:
                    actual_cost += float(embedding_cost)
                    self.store.add_job_cost(job["id"], float(embedding_cost))
            self.store.finish_job(job["id"], video_id)
        except Exception as exc:
            # Paid calls have no automatic retry. Saved transcript/embeddings are retained.
            self.store.fail_job(job["id"], video_id, str(exc))
        return True
