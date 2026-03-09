import sqlite3

import pytest

from video_service.core import benchmarking
from video_service.core.benchmarking import (
    jaccard_similarity,
    levenshtein_similarity,
    extract_stage_duration_seconds,
)

pytestmark = pytest.mark.unit


def test_jaccard_similarity_basic():
    score = jaccard_similarity(
        ["Automotive", "Road Safety"],
        ["Road Safety", "Insurance"],
    )
    assert round(score, 3) == round(1 / 3, 3)


def test_levenshtein_similarity_bounds():
    assert levenshtein_similarity("abc", "abc") == 1.0
    assert 0.0 <= levenshtein_similarity("abc", "xyz") <= 1.0


def test_extract_stage_duration_from_events():
    events = [
        "2026-02-28T12:00:00+00:00 frame_extract: extracted 5 frames",
        "2026-02-28T12:00:12+00:00 llm: calling provider",
        "2026-02-28T12:00:19+00:00 persist: persisting result payload",
    ]
    assert extract_stage_duration_seconds(events, fallback_duration=33.2) == 19.0


def _db_factory(path: str):
    def _open_db():
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

    return _open_db


def test_evaluate_benchmark_suite_aggregates_saved_processing_paths(monkeypatch, tmp_path):
    db_path = str(tmp_path / "benchmark_paths.db")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE benchmark_truth (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                expected_ocr_text TEXT DEFAULT '',
                expected_categories_json TEXT DEFAULT '[]'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE benchmark_suites (
                id TEXT PRIMARY KEY,
                truth_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                total_jobs INTEGER DEFAULT 0,
                completed_jobs INTEGER DEFAULT 0,
                failed_jobs INTEGER DEFAULT 0,
                evaluated_at TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE benchmark_result (
                id TEXT PRIMARY KEY,
                suite_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                duration_seconds REAL,
                classification_accuracy REAL,
                ocr_accuracy REAL,
                composite_accuracy REAL,
                params_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                benchmark_suite_id TEXT,
                status TEXT DEFAULT 'queued',
                result_json TEXT DEFAULT '',
                artifacts_json TEXT DEFAULT '',
                events TEXT DEFAULT '[]',
                duration_seconds REAL,
                benchmark_params_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "INSERT INTO benchmark_truth (id, name, expected_ocr_text, expected_categories_json) VALUES (?, ?, ?, ?)",
            ("truth-1", "Truth", "ocr text", '["Category One"]'),
        )
        conn.execute(
            "INSERT INTO benchmark_suites (id, truth_id, status, total_jobs, completed_jobs, failed_jobs) VALUES (?, ?, ?, ?, ?, ?)",
            ("suite-1", "truth-1", "running", 3, 2, 0),
        )

        trace_one = {
            "attempts": [
                {"attempt_type": "initial", "title": "Initial Tail Pass", "status": "rejected"},
                {"attempt_type": "express_rescue", "title": "Express Rescue", "status": "accepted"},
            ],
            "summary": {"accepted_attempt_type": "express_rescue"},
        }
        trace_two = {
            "attempts": [
                {"attempt_type": "initial", "title": "Initial Tail Pass", "status": "rejected"},
                {"attempt_type": "ocr_context_rescue", "title": "OCR Context Rescue", "status": "accepted"},
            ],
            "summary": {"accepted_attempt_type": "ocr_context_rescue"},
        }
        conn.execute(
            """
            INSERT INTO jobs (
                id, benchmark_suite_id, status, result_json, artifacts_json, events, duration_seconds, benchmark_params_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-1",
                "suite-1",
                "completed",
                '[{"Category":"Category One"}]',
                '{"ocr_text":{"text":"ocr text"},"processing_trace":' + str(trace_one).replace("'", '"') + "}",
                '["2026-02-28T12:00:00+00:00 frame_extract: extracted 5 frames","2026-02-28T12:00:03+00:00 persist: persisting result payload"]',
                3.0,
                '{"provider":"Llama Server"}',
            ),
        )
        conn.execute(
            """
            INSERT INTO jobs (
                id, benchmark_suite_id, status, result_json, artifacts_json, events, duration_seconds, benchmark_params_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-2",
                "suite-1",
                "completed",
                '[{"Category":"Category One"}]',
                '{"ocr_text":{"text":"ocr text"},"processing_trace":' + str(trace_two).replace("'", '"') + "}",
                '["2026-02-28T12:01:00+00:00 frame_extract: extracted 5 frames","2026-02-28T12:01:04+00:00 persist: persisting result payload"]',
                4.0,
                '{"provider":"Ollama"}',
            ),
        )
        conn.execute(
            """
            INSERT INTO jobs (
                id, benchmark_suite_id, status, result_json, artifacts_json, events, duration_seconds, benchmark_params_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-3",
                "suite-1",
                "failed",
                "",
                '{"processing_trace":{"attempts":[{"attempt_type":"initial","title":"Initial Tail Pass","status":"rejected"}],"summary":{"accepted_attempt_type":""}}}',
                "[]",
                None,
                '{}',
            ),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(benchmarking, "get_db", _db_factory(db_path))

    result = benchmarking.evaluate_benchmark_suite("suite-1")

    assert result["ok"] is True
    assert result["path_metrics"]["jobs_with_trace"] == 3
    assert result["path_metrics"]["accepted_paths"] == [
        {"attempt_type": "express_rescue", "title": "Express Rescue", "count": 1},
        {"attempt_type": "ocr_context_rescue", "title": "OCR Context Rescue", "count": 1},
    ]
    assert result["path_metrics"]["transit_paths"] == [
        {"attempt_type": "initial", "title": "Initial Tail Pass", "count": 3},
        {"attempt_type": "express_rescue", "title": "Express Rescue", "count": 1},
        {"attempt_type": "ocr_context_rescue", "title": "OCR Context Rescue", "count": 1},
    ]
