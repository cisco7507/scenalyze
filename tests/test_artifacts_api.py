import asyncio
import sqlite3

import pytest

from video_service.app import main
from video_service.app.models.job import JobSettings

pytestmark = pytest.mark.unit


class _Req:
    query_params = {}


async def _no_proxy(_req, _job_id):
    return None


def test_job_artifacts_endpoint_returns_required_keys_when_empty(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, artifacts_json TEXT)")
        conn.execute(
            "INSERT INTO jobs (id, artifacts_json) VALUES (?, ?)",
            ("job-1", None),
        )

    monkeypatch.setattr(main, "get_db", lambda: conn)
    monkeypatch.setattr(main, "_maybe_proxy", _no_proxy)

    payload = asyncio.run(main.get_job_artifacts(_Req(), "job-1"))

    assert "artifacts" in payload
    assert "latest_frames" in payload
    assert "per_frame_vision" in payload
    assert "ocr_text" in payload
    assert "vision_board" in payload
    assert "category_mapper" in payload
    assert "latest_frames" in payload["artifacts"]
    assert "per_frame_vision" in payload["artifacts"]
    assert "ocr_text" in payload["artifacts"]
    assert "vision_board" in payload["artifacts"]
    assert "category_mapper" in payload["artifacts"]
    assert "vector_plot" in payload["artifacts"]["vision_board"]
    assert "vector_plot" in payload["artifacts"]["category_mapper"]


def test_job_settings_accept_enable_web_search_alias():
    settings = JobSettings.model_validate(
        {
            "categories": "",
            "provider": "Ollama",
            "model_name": "qwen3-vl:8b-instruct",
            "ocr_engine": "EasyOCR",
            "ocr_mode": "🚀 Fast",
            "scan_mode": "Tail Only",
            "override": False,
            "enable_web_search": True,
            "enable_vision": True,
            "context_size": 8192,
        }
    )
    assert settings.enable_search is True
    assert settings.enable_web_search is True
    assert settings.enable_vision_board is True
    assert settings.enable_llm_frame is True
