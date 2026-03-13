import pytest
import pandas as pd
import threading
import logging
import io

from video_service.workers import worker
from video_service.core import logging_setup

pytestmark = pytest.mark.unit


def test_get_worker_process_count_defaults_to_one(monkeypatch):
    monkeypatch.delenv("WORKER_PROCESSES", raising=False)
    assert worker._get_worker_process_count() == 1


def test_get_worker_process_count_invalid_or_non_positive(monkeypatch):
    monkeypatch.setenv("WORKER_PROCESSES", "abc")
    assert worker._get_worker_process_count() == 1

    monkeypatch.setenv("WORKER_PROCESSES", "0")
    assert worker._get_worker_process_count() == 1

    monkeypatch.setenv("WORKER_PROCESSES", "-7")
    assert worker._get_worker_process_count() == 1


def test_get_worker_process_count_positive_integer(monkeypatch):
    monkeypatch.setenv("WORKER_PROCESSES", "4")
    assert worker._get_worker_process_count() == 4


def test_run_worker_uses_single_process_path(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(worker, "_get_worker_process_count", lambda: 1)
    monkeypatch.setattr(worker, "_run_single_worker", lambda: calls.append("single"))
    monkeypatch.setattr(worker, "_run_worker_supervisor", lambda count: calls.append(f"supervisor:{count}"))

    worker.run_worker()
    assert calls == ["single"]


def test_run_worker_uses_supervisor_path(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(worker, "_get_worker_process_count", lambda: 3)
    monkeypatch.setattr(worker, "_run_single_worker", lambda: calls.append("single"))
    monkeypatch.setattr(worker, "_run_worker_supervisor", lambda count: calls.append(f"supervisor:{count}"))

    worker.run_worker()
    assert calls == ["supervisor:3"]


def test_run_pipeline_uses_pipeline_threads_per_job_env(monkeypatch):
    captured = {}
    monkeypatch.setenv("PIPELINE_THREADS_PER_JOB", "3")
    monkeypatch.setattr(worker, "_stage_callback", lambda _job_id: (lambda _s, _d: None))

    def _fake_run_pipeline_job(**kwargs):
        captured["workers"] = kwargs["workers"]
        captured["category_embedding_model"] = kwargs["category_embedding_model"]
        yield ({}, "", "", [], pd.DataFrame([{"Brand": "BrandX"}]))

    monkeypatch.setattr(worker, "run_pipeline_job", _fake_run_pipeline_job)

    worker._run_pipeline(
        "job-1",
        "https://example.test/ad.mp4",
        {"category_embedding_model": "BAAI/bge-large-en-v1.5"},
    )
    assert captured["workers"] == 3
    assert captured["category_embedding_model"] == "BAAI/bge-large-en-v1.5"


def test_extract_agent_ocr_text_supports_observation_without_scene_prefix():
    events = [
        "2026-02-28T00:00:00Z agent:\n--- Step 1 ---\nAction: [TOOL: OCR]\nResult: Observation: VOLVO | XC90",
    ]
    assert worker._extract_agent_ocr_text(events) == "VOLVO | XC90"


def test_job_heartbeat_updates_updated_at_and_stops(monkeypatch):
    calls: list[tuple[str, tuple]] = []

    def _fake_execute(sql: str, params: tuple, *, attempts: int = 10):
        calls.append((sql, params))
        stop_event.set()

    stop_event = threading.Event()
    monkeypatch.setattr(worker, "_execute_job_update_with_retry", _fake_execute)
    hb_thread = worker._start_job_heartbeat(
        "node-a-heartbeat-test",
        stop_event,
        interval_seconds=0.01,
    )
    hb_thread.join(timeout=1.0)

    assert calls
    assert calls[0][1] == ("node-a-heartbeat-test",)


def test_job_heartbeat_logs_with_job_context(monkeypatch):
    logger = logging.getLogger(worker.__name__)
    original_level = logger.level
    original_propagate = logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("job_id=%(job_id)s %(message)s"))
    handler.addFilter(logging_setup.ContextEnricherFilter())
    logger.addHandler(handler)

    calls = {"count": 0}

    def _fake_execute(sql: str, params: tuple, *, attempts: int = 10):
        calls["count"] += 1
        stop_event.set()

    stop_event = threading.Event()
    monkeypatch.setattr(worker, "_execute_job_update_with_retry", _fake_execute)

    job_token = logging_setup.set_job_context("node-a-heartbeat-log")
    stage_tokens = logging_setup.set_stage_context("claim", "worker claimed job")
    try:
        hb_thread = worker._start_job_heartbeat(
            "node-a-heartbeat-log",
            stop_event,
            interval_seconds=0.01,
        )
        hb_thread.join(timeout=1.0)
    finally:
        logging_setup.reset_stage_context(stage_tokens)
        logging_setup.reset_job_context(job_token)
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate

    assert calls["count"] == 1
    assert "job_id=node-a-heartbeat-log" in stream.getvalue()


def test_set_stage_logs_with_job_context_without_ambient_state(monkeypatch):
    logger = logging.getLogger(worker.__name__)
    original_level = logger.level
    original_propagate = logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("job_id=%(job_id)s stage=%(stage)s %(message)s"))
    handler.addFilter(logging_setup.ContextEnricherFilter())
    logger.addHandler(handler)

    monkeypatch.setattr(worker, "_execute_job_update_with_retry", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_append_job_event", lambda *args, **kwargs: None)

    try:
        worker._set_stage("node-a-stage-log", "ocr", "ocr engine=easyocr")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate

    out = stream.getvalue()
    assert "job_id=node-a-stage-log" in out
    assert "stage=ocr" in out
    assert "ocr engine=easyocr" in out
