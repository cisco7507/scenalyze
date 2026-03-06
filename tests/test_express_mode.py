import sys
import types

import pytest
from PIL import Image

# `video_service.core.llm` imports `ddgs`; stub it for unit tests.
if "ddgs" not in sys.modules:
    ddgs_stub = types.ModuleType("ddgs")
    ddgs_stub.DDGS = object
    sys.modules["ddgs"] = ddgs_stub

from video_service.core import pipeline as pipeline_module

pytestmark = pytest.mark.unit


def test_process_single_video_express_mode_bypasses_ocr(monkeypatch):
    captured: dict[str, object] = {}

    class _DummyMapper:
        @staticmethod
        def map_category(**kwargs):
            assert kwargs.get("ocr_summary") == ""
            return {
                "canonical_category": "Category One",
                "category_id": "42",
                "category_match_method": "embeddings",
                "category_match_score": 0.99,
            }

    class _DummyOCR:
        @staticmethod
        def extract_text(*args, **kwargs):
            raise AssertionError("OCR should be bypassed in express mode")

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            captured["force_multimodal"] = args[6]
            captured["tail_image"] = args[3]
            captured["ocr_text"] = args[2]
            captured["express_mode"] = kwargs.get("express_mode")
            return {
                "brand": "Brand Express",
                "category": "Raw Category",
                "confidence": 0.8,
                "reasoning": "ok",
            }

    monkeypatch.setattr(
        pipeline_module,
        "extract_express_brand_frame",
        lambda _url, **kwargs: Image.new("RGB", (96, 96), "white"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("regular extraction should not run")),
    )
    monkeypatch.setattr(pipeline_module, "extract_middle_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())
    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())

    vision_scores, per_frame_vision, ocr_text, _, gallery, row, _ = pipeline_module.process_single_video(
        url="https://example.test/ad.mp4",
        categories=[],
        p="Ollama",
        m="qwen3-vl:8b-instruct",
        oe="EasyOCR",
        om="🚀 Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision_board=False,
        enable_llm_frame=False,
        ctx=8192,
        express_mode=True,
        job_id="job-express-1",
        stage_callback=None,
    )

    assert vision_scores == {}
    assert per_frame_vision == []
    assert ocr_text == ""
    assert len(gallery) == 1
    assert row[1] == "Brand Express"
    assert row[2] == "42"
    assert row[3] == "Category One"
    assert captured["ocr_text"] == ""
    assert captured["express_mode"] is True
    assert captured["force_multimodal"] is True
    assert captured["tail_image"] is not None
