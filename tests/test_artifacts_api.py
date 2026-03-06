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
    assert "processing_trace" in payload["artifacts"]
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


def test_job_explanation_endpoint_returns_structured_trace(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                status TEXT,
                stage TEXT,
                stage_detail TEXT,
                mode TEXT,
                brand TEXT,
                category TEXT,
                category_id TEXT,
                result_json TEXT,
                artifacts_json TEXT,
                events TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO jobs (
                id, status, stage, stage_detail, mode, brand, category, category_id,
                result_json, artifacts_json, events
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-2",
                "completed",
                "completed",
                "done",
                "pipeline",
                "Brand X",
                "Category One",
                "101",
                '[{"Brand":"Brand X","Category":"Category One","Category ID":"101","Confidence":0.97}]',
                '{"latest_frames":[{"url":"/artifacts/job-2/latest_frames/frame_001.jpg","label":"[29.4s]"}],'
                '"ocr_text":{"text":"brand x save more"},"category_mapper":{"method":"embeddings","score":0.99},'
                '"processing_trace":{"attempts":[{"attempt_type":"initial","title":"Initial Tail Pass","status":"rejected","detail":"tail scan","trigger_reason":"blank_brand_and_category","frame_count":3,"frame_times":[27.0,28.2,29.4],"ocr_excerpt":"","result":{"brand":"","category":"","confidence":0.0}},{"attempt_type":"express_rescue","title":"Express Rescue","status":"accepted","detail":"image-first retry","trigger_reason":"blank_brand_and_category","frame_count":1,"frame_times":[29.4],"ocr_excerpt":"","result":{"brand":"Brand X","category":"Category One","confidence":0.97}}],'
                '"summary":{"headline":"Fallback ladder explanation.","attempt_count":2,"retry_count":1,"accepted_attempt_type":"express_rescue","trigger_reason":"blank_brand_and_category"}}}',
                '["2026-03-06T10:35:29Z llm: blank initial result","2026-03-06T10:35:31Z llm: express rescue accepted"]',
            ),
        )

    monkeypatch.setattr(main, "get_db", lambda: conn)
    monkeypatch.setattr(main, "_maybe_proxy", _no_proxy)

    payload = asyncio.run(main.get_job_explanation(_Req(), "job-2"))

    explanation = payload["explanation"]
    assert explanation["job_id"] == "job-2"
    assert explanation["summary"]["accepted_attempt_type"] == "express_rescue"
    assert len(explanation["attempts"]) == 2
    assert explanation["attempts"][1]["status"] == "accepted"
    assert explanation["final"]["brand"] == "Brand X"
    assert explanation["final"]["category_id"] == "101"
    assert explanation["evidence"]["event_count"] == 2
    assert explanation["evidence"]["latest_frames"][0]["url"].endswith("frame_001.jpg")
