import sqlite3

import pytest

import video_service.app.main as main

pytestmark = pytest.mark.unit


def test_analytics_includes_duration_percentiles_and_series(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    with conn:
        conn.execute(
            """
            CREATE TABLE job_stats (
                job_id TEXT,
                status TEXT,
                mode TEXT,
                provider TEXT,
                scan_mode TEXT,
                brand TEXT,
                category TEXT,
                duration_seconds REAL,
                created_at TEXT,
                completed_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                status TEXT,
                artifacts_json TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO job_stats (
                job_id, status, mode, provider, scan_mode, brand, category, duration_seconds, created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "node-a-1",
                    "completed",
                    "pipeline",
                    "Ollama",
                    "Tail Only",
                    "Brand A",
                    "Category A",
                    10.0,
                    "2026-02-28 11:00:00",
                    "2026-02-28 11:00:30",
                ),
                (
                    "node-a-2",
                    "completed",
                    "pipeline",
                    "Ollama",
                    "Tail Only",
                    "Brand A",
                    "Category A",
                    20.0,
                    "2026-02-28 11:01:00",
                    "2026-02-28 11:01:35",
                ),
                (
                    "node-a-3",
                    "completed",
                    "agent",
                    "Ollama",
                    "Full Video",
                    "Brand B",
                    "Category B",
                    30.0,
                    "2026-02-28 11:02:00",
                    "2026-02-28 11:02:42",
                ),
            ],
        )

    monkeypatch.setattr(main, "get_db", lambda: conn)
    payload = main.get_analytics()

    assert "duration_percentiles" in payload
    assert payload["duration_percentiles"]["count"] == 3
    assert payload["duration_percentiles"]["p50"] == 20.0
    assert payload["duration_percentiles"]["p90"] is not None
    assert payload["duration_percentiles"]["p95"] is not None
    assert payload["duration_percentiles"]["p99"] is not None

    assert "duration_series" in payload
    assert payload["duration_series"]
    assert payload["duration_series"][0]["count"] >= 1
    assert payload["recent_duration_points"]
