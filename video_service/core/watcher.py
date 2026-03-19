"""
video_service/core/watcher.py
=============================
Watch-folder auto-ingest for local video files.

- Watches configured top-level folders for new video files.
- Waits for files to stabilize before enqueueing jobs.
- Includes path-safety checks to prevent symlink/path escapes.
"""

from __future__ import annotations

import os
import uuid
import threading
import time
import logging
from contextlib import closing
from typing import Optional

from video_service.app.models.job import JobSettings
from video_service.core.cluster import cluster
from video_service.db.database import get_db
from video_service.core.logging_setup import job_context

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_shutdown_event = threading.Event()

_WATCHDOG_IMPORT_ERROR: Exception | None = None
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception as exc:  # pragma: no cover - dependent on environment
    Observer = None  # type: ignore[assignment]
    FileSystemEventHandler = object  # type: ignore[assignment]
    _WATCHDOG_IMPORT_ERROR = exc


class _WatcherMaintenancePause(RuntimeError):
    pass


def _parse_watch_folders(raw: str) -> list[str]:
    return [folder.strip() for folder in (raw or "").split(",") if folder.strip()]


def _parse_stabilize_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 3.0
    return value if value > 0 else 3.0


def _is_safe_watch_path(file_path: str, watch_roots: list[str]) -> bool:
    """Ensure file resolves inside configured watch roots."""
    try:
        real_path = os.path.realpath(file_path)
    except (OSError, ValueError):
        return False

    for root in watch_roots:
        try:
            if os.path.commonpath([real_path, os.path.realpath(root)]) == os.path.realpath(root):
                return True
        except ValueError:
            continue
    return False


class _StabilizationTracker:
    """Tracks file sizes to detect when write operations are complete."""

    def __init__(self, stabilize_seconds: float = 3.0):
        self.stabilize_seconds = stabilize_seconds
        self._pending: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()

    def register(self, path: str) -> None:
        with self._lock:
            if path in self._pending:
                return
            try:
                size = float(os.path.getsize(path))
            except OSError:
                return
            self._pending[path] = {"size": size, "stable_since": time.monotonic()}

    def check_ready(self) -> list[str]:
        """Return paths that have remained size-stable long enough."""
        ready: list[str] = []
        now = time.monotonic()
        with self._lock:
            to_remove: list[str] = []
            for path, info in self._pending.items():
                if not os.path.exists(path):
                    to_remove.append(path)
                    continue

                try:
                    current_size = float(os.path.getsize(path))
                except OSError:
                    to_remove.append(path)
                    continue

                if current_size != info["size"]:
                    info["size"] = current_size
                    info["stable_since"] = now
                    continue

                if now - info["stable_since"] >= self.stabilize_seconds:
                    ready.append(path)
                    to_remove.append(path)

            for path in to_remove:
                self._pending.pop(path, None)
        return ready


def _build_watch_job_settings() -> JobSettings:
    return JobSettings(
        provider=os.environ.get("WATCH_DEFAULT_PROVIDER", "Ollama"),
        model_name=os.environ.get("WATCH_DEFAULT_MODEL", "qwen3-vl:8b-instruct"),
        ocr_engine=os.environ.get("WATCH_DEFAULT_OCR_ENGINE", "EasyOCR"),
        scan_mode=os.environ.get("WATCH_DEFAULT_SCAN_MODE", "Tail Only"),
    )


def _resolve_watch_mode() -> str:
    mode = (os.environ.get("WATCH_DEFAULT_MODE", "pipeline") or "pipeline").strip().lower()
    if mode not in {"pipeline", "agent"}:
        logger.warning("watcher: invalid WATCH_DEFAULT_MODE=%s, using pipeline", mode)
        return "pipeline"
    return mode


def _submit_watch_job(file_path: str) -> str:
    """Insert a queued job for a stabilized watch-folder file."""
    if not cluster.is_accepting_new_jobs(cluster.self_name):
        raise _WatcherMaintenancePause("node in maintenance mode")

    job_id = f"{cluster.self_name}-{uuid.uuid4()}"
    settings = _build_watch_job_settings()
    mode = _resolve_watch_mode()

    with closing(get_db()) as conn:
        with conn:
            conn.execute(
                "INSERT INTO jobs (id, status, stage, stage_detail, mode, settings, url, events) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    "queued",
                    "queued",
                    "auto-ingested from watch folder",
                    mode,
                    settings.model_dump_json(),
                    file_path,
                    "[]",
                ),
            )
    with job_context(job_id):
        logger.info("watch_job_submitted: job_id=%s file=%s", job_id, file_path)
    return job_id


class _VideoFileHandler(FileSystemEventHandler):
    def __init__(self, tracker: _StabilizationTracker, watch_roots: list[str]):
        super().__init__()
        self._tracker = tracker
        self._watch_roots = watch_roots

    def _maybe_track(self, path: str) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            return
        if not _is_safe_watch_path(path, self._watch_roots):
            logger.warning("watcher: rejected path outside roots: %s", path)
            return
        self._tracker.register(path)

    def on_created(self, event) -> None:  # type: ignore[override]
        if not getattr(event, "is_directory", False):
            self._maybe_track(str(getattr(event, "src_path", "")))

    def on_moved(self, event) -> None:  # type: ignore[override]
        if not getattr(event, "is_directory", False):
            self._maybe_track(str(getattr(event, "dest_path", "")))


def get_watcher_diagnostics() -> dict:
    raw_folders = os.environ.get("WATCH_FOLDERS", "").strip()
    folders = _parse_watch_folders(raw_folders)
    stabilize_seconds = _parse_stabilize_seconds(os.environ.get("WATCH_STABILIZE_SECONDS", "3"))
    return {
        "enabled": len(folders) > 0,
        "watch_folders": folders,
        "output_dir": os.environ.get("WATCH_OUTPUT_DIR", "").strip() or None,
        "default_mode": _resolve_watch_mode(),
        "stabilize_seconds": stabilize_seconds,
        "watchdog_available": _WATCHDOG_IMPORT_ERROR is None,
        "watchdog_error": str(_WATCHDOG_IMPORT_ERROR) if _WATCHDOG_IMPORT_ERROR else None,
    }


def start_watcher() -> Optional[Observer]:
    """Start watch-folder ingest. Returns observer instance, or None when disabled."""
    raw_folders = os.environ.get("WATCH_FOLDERS", "").strip()
    if not raw_folders:
        logger.info("watcher: disabled (WATCH_FOLDERS not set)")
        return None

    if _WATCHDOG_IMPORT_ERROR is not None:
        logger.warning("watcher: disabled (watchdog import failed: %s)", _WATCHDOG_IMPORT_ERROR)
        return None

    watch_roots_raw = _parse_watch_folders(raw_folders)
    valid_roots: list[str] = []
    for root in watch_roots_raw:
        real = os.path.realpath(root)
        if os.path.isdir(real):
            valid_roots.append(real)
        else:
            logger.warning("watcher: skipping non-existent folder: %s", root)

    if not valid_roots:
        logger.warning("watcher: no valid folders to watch")
        return None

    stabilize_seconds = _parse_stabilize_seconds(os.environ.get("WATCH_STABILIZE_SECONDS", "3"))
    tracker = _StabilizationTracker(stabilize_seconds=stabilize_seconds)
    handler = _VideoFileHandler(tracker, valid_roots)

    observer = Observer()  # type: ignore[operator]
    for root in valid_roots:
        observer.schedule(handler, root, recursive=False)
        logger.info("watcher: monitoring %s", root)

    _shutdown_event.clear()

    def _stabilization_loop() -> None:
        while not _shutdown_event.is_set():
            for path in tracker.check_ready():
                try:
                    _submit_watch_job(path)
                except _WatcherMaintenancePause:
                    logger.debug("watcher: deferred %s while node is in maintenance mode", path)
                    tracker.register(path)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error("watcher: submit failed for %s: %s", path, exc)
            _shutdown_event.wait(timeout=1.0)

    checker = threading.Thread(target=_stabilization_loop, daemon=True, name="watch-stabilizer")
    checker.start()

    observer.start()
    logger.info("watcher: started on %d folder(s)", len(valid_roots))
    return observer


def stop_watcher(observer: Optional[Observer]) -> None:
    _shutdown_event.set()
    if observer is not None:
        observer.stop()
        observer.join(timeout=5.0)
    logger.info("watcher: stopped")
