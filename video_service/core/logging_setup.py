import logging
import os
import asyncio
import threading
from pathlib import Path
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from contextvars import ContextVar, Token
from collections import deque
from typing import Callable

_job_id_var: ContextVar[str] = ContextVar("job_id", default="-")
_stage_var: ContextVar[str] = ContextVar("stage", default="-")
_stage_detail_var: ContextVar[str] = ContextVar("stage_detail", default="-")

_configured = False
_env_loaded = False
_debug_enabled = False
_memory_handler: "MemoryListHandler | None" = None
_file_handler: RotatingFileHandler | None = None

_NOISY_LOGGERS = (
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "httpx",
    "httpcore",
    "urllib3",
    "transformers",
    "sentence_transformers",
    "PIL",
    "matplotlib",
)
_FORCE_QUIET_LOGGERS = (
    "httpx",
    "httpcore",
    "httpcore.http11",
    "urllib3",
    "huggingface_hub",
    "transformers",
    "sentence_transformers",
)
_FORCE_ERROR_LOGGERS = (
    # Transformers emits a warning_once call in this module using a non-format
    # argument shape that can trigger logging TypeError on Python 3.14.
    "transformers.modeling_attn_mask_utils",
    # Florence remote model load report warnings are noisy and not actionable
    # for runtime job observability.
    "transformers.modeling_utils",
    # Tensor parallel warnings are noisy/non-actionable for single-node local runs.
    "transformers.integrations.tensor_parallel",
)


class MemoryListHandler(logging.Handler):
    def __init__(self, max_lines: int = 1000):
        super().__init__()
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._lines_lock = threading.Lock()
        self._subscribers: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue[str]]] = {}
        self._subscribers_lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            self.handleError(record)
            return

        with self._lines_lock:
            self._lines.append(line)
        self._publish(line)

    def recent(self, limit: int = 200) -> list[str]:
        bounded_limit = max(1, int(limit))
        with self._lines_lock:
            lines = list(self._lines)
        return lines[-bounded_limit:]

    def clear(self) -> None:
        with self._lines_lock:
            self._lines.clear()

    def subscribe(
        self,
        max_queue_size: int = 1000,
    ) -> tuple[asyncio.Queue[str], Callable[[], None]]:
        loop = asyncio.get_running_loop()
        # Initialize the queue bound to the current request's event loop
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max(1, max_queue_size))
        token = id(queue)
        with self._subscribers_lock:
            self._subscribers[token] = (loop, queue)

        def unsubscribe() -> None:
            with self._subscribers_lock:
                self._subscribers.pop(token, None)

        return queue, unsubscribe

    @staticmethod
    def _queue_put_latest(queue: asyncio.Queue[str], line: str) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except Exception:
                pass
        try:
            queue.put_nowait(line)
        except Exception:
            pass

    def _publish(self, line: str) -> None:
        with self._subscribers_lock:
            subscribers = list(self._subscribers.items())

        stale_tokens: list[int] = []
        for token, (loop, queue) in subscribers:
            try:
                # We specifically MUST use the queue from within its bound loop thread 
                # or asyncio throws RuntimeError in Python 3.10+
                loop.call_soon_threadsafe(self._queue_put_latest, queue, line)
            except (RuntimeError, Exception) as e:
                import sys
                print(f"DEBUG: Dropping log subscriber due to: {e}", file=sys.stderr)
                stale_tokens.append(token)

        if stale_tokens:
            with self._subscribers_lock:
                for token in stale_tokens:
                    self._subscribers.pop(token, None)


class ContextEnricherFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = _job_id_var.get() or "-"
        record.stage = _stage_var.get() or "-"
        record.stage_detail = _stage_detail_var.get() or "-"
        return True


class NoisyLibraryFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # In DEBUG mode, allow all logs through.
        if _debug_enabled:
            return True
        # In INFO/WARNING/ERROR modes, suppress verbose library chatter.
        if record.levelno >= logging.WARNING:
            return True
        return not any(
            record.name == name or record.name.startswith(f"{name}.")
            for name in _NOISY_LOGGERS
        )


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_log_file_path(repo_root: Path) -> Path:
    log_dir_raw = (os.environ.get("LOG_DIR") or "logs").strip() or "logs"
    log_filename = (os.environ.get("LOG_FILENAME") or "video_service.log").strip() or "video_service.log"

    log_dir = Path(log_dir_raw).expanduser()
    if not log_dir.is_absolute():
        log_dir = (repo_root / log_dir).resolve()

    return log_dir / log_filename


def _configure_file_handler(root: logging.Logger, formatter: logging.Formatter) -> None:
    global _file_handler

    managed_handlers = [
        handler
        for handler in list(root.handlers)
        if getattr(handler, "_video_service_file_handler", False)
    ]

    file_logging_enabled = _env_truthy("LOG_TO_FILE", default=False)
    if not file_logging_enabled:
        for handler in managed_handlers:
            root.removeHandler(handler)
            handler.close()
        _file_handler = None
        return

    log_path = _resolve_log_file_path(_repo_root())
    log_path.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = _env_int("LOG_MAX_BYTES", 10 * 1024 * 1024)
    backup_count = _env_int("LOG_BACKUP_COUNT", 5)

    current_path = None
    if _file_handler is not None:
        current_path = Path(_file_handler.baseFilename)

    if _file_handler is None or current_path != log_path:
        for handler in managed_handlers:
            root.removeHandler(handler)
            handler.close()
        _file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        setattr(_file_handler, "_video_service_file_handler", True)
        root.addHandler(_file_handler)
    else:
        _file_handler.maxBytes = max_bytes
        _file_handler.backupCount = backup_count

    _file_handler.setFormatter(formatter)


def configure_logging(force: bool = False) -> None:
    global _configured, _env_loaded, _debug_enabled, _memory_handler
    if _configured and not force:
        return

    # Load repository .env once and make it authoritative for local app startup.
    # This keeps behavior deterministic across shells that may have stale exports.
    if not _env_loaded:
        try:
            from dotenv import load_dotenv

            repo_root = Path(__file__).resolve().parents[2]
            load_dotenv(dotenv_path=repo_root / ".env", override=True)
        except Exception:
            # Keep logging setup resilient even if python-dotenv is unavailable.
            pass
        _env_loaded = True

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper().strip()
    level = getattr(logging, level_name, logging.INFO)
    _debug_enabled = level <= logging.DEBUG

    fmt = "%(asctime)s %(levelname)-8s job_id=%(job_id)s stage=%(stage)s %(name)s %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    else:
        root.setLevel(level)
        for handler in root.handlers:
            handler.setFormatter(formatter)

    if _memory_handler is None:
        _memory_handler = MemoryListHandler(max_lines=1000)
    if _memory_handler not in root.handlers:
        root.addHandler(_memory_handler)
    _memory_handler.setFormatter(formatter)

    _configure_file_handler(root, formatter)

    context_filter = ContextEnricherFilter()
    noisy_filter = NoisyLibraryFilter()
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, ContextEnricherFilter) for f in handler.filters):
            handler.addFilter(context_filter)
        if not any(isinstance(f, NoisyLibraryFilter) for f in handler.filters):
            handler.addFilter(noisy_filter)

    # Always hard-gate known broken/noisy external loggers, even in DEBUG mode.
    # This avoids Python 3.14 formatting crashes from third-party warning calls.
    for logger_name in _FORCE_ERROR_LOGGERS:
        err_logger = logging.getLogger(logger_name)
        err_logger.setLevel(logging.ERROR)
        err_logger.propagate = False
        err_logger.handlers.clear()
        err_handler = logging.StreamHandler()
        err_handler.setLevel(logging.ERROR)
        err_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        err_handler.addFilter(context_filter)
        err_logger.addHandler(err_handler)

    if level > logging.DEBUG:
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()
        except Exception:
            pass

        for logger_name in _NOISY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        # Hard-gate especially noisy HTTP/model loggers so DEBUG cannot leak through
        # from external logger reconfiguration later in startup.
        for logger_name in _FORCE_QUIET_LOGGERS:
            noisy_logger = logging.getLogger(logger_name)
            noisy_logger.setLevel(logging.WARNING)
            noisy_logger.propagate = False
            noisy_logger.handlers.clear()
            hard_handler = logging.StreamHandler()
            hard_handler.setLevel(logging.WARNING)
            hard_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
            hard_handler.addFilter(context_filter)
            noisy_logger.addHandler(hard_handler)

    _configured = True


def get_recent_log_lines(limit: int = 200) -> list[str]:
    if _memory_handler is None:
        return []
    return _memory_handler.recent(limit=limit)


def subscribe_log_stream(
    max_queue_size: int = 1000,
) -> tuple[asyncio.Queue[str], Callable[[], None]]:
    if _memory_handler is None:
        configure_logging()
    assert _memory_handler is not None
    return _memory_handler.subscribe(max_queue_size=max_queue_size)


def clear_recent_log_lines() -> None:
    if _memory_handler is None:
        return
    _memory_handler.clear()


def set_job_context(job_id: str) -> Token:
    return _job_id_var.set(job_id or "-")


def reset_job_context(token: Token | None = None) -> None:
    if token is not None:
        _job_id_var.reset(token)
    else:
        _job_id_var.set("-")


def set_stage_context(stage: str, stage_detail: str = "") -> tuple[Token, Token]:
    stage_token = _stage_var.set(stage or "-")
    detail_token = _stage_detail_var.set(stage_detail or "-")
    return stage_token, detail_token


def reset_stage_context(tokens: tuple[Token, Token] | None = None) -> None:
    if tokens is not None:
        stage_token, detail_token = tokens
        _stage_var.reset(stage_token)
        _stage_detail_var.reset(detail_token)
    else:
        _stage_var.set("-")
        _stage_detail_var.set("-")


@contextmanager
def job_context(job_id: str):
    token = set_job_context(job_id)
    try:
        yield
    finally:
        reset_job_context(token)
