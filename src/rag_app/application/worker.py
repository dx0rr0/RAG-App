import logging
import threading
import time


logger = logging.getLogger(__name__)


class BackgroundWorker:
    def __init__(self, ingestion, idle_seconds=1.0):
        self.ingestion = ingestion
        self.idle_seconds = idle_seconds
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.ingestion.store.mark_interrupted_jobs()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rag-ingestion-worker", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            try:
                did_work = self.ingestion.process_next()
            except Exception:
                logger.exception("Ingestion worker iteration failed")
                did_work = False
            if not did_work:
                self._stop.wait(self.idle_seconds)
