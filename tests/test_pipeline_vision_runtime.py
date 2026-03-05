import sys
import types
import logging

import cv2
import numpy as np
import pytest
import torch

# `video_service.core.llm` imports `ddgs`; stub it for unit tests.
if "ddgs" not in sys.modules:
    ddgs_stub = types.ModuleType("ddgs")
    ddgs_stub.DDGS = object
    sys.modules["ddgs"] = ddgs_stub

from video_service.core import pipeline as pipeline_module

pytestmark = pytest.mark.unit


def test_pipeline_vision_uses_runtime_siglip_handles_and_emits_top_matches(monkeypatch):
    class _DummyMapper:
        categories = ["Category One", "Category Two"]
        vision_text_features = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )

        @staticmethod
        def ensure_vision_text_features():
            return True, "ready"

        @staticmethod
        def map_category(**kwargs):
            # Regression assertion: suggested categories must not influence mapping.
            assert kwargs.get("suggested_categories_text") == ""
            return {
                "canonical_category": "Category One",
                "category_id": "101",
                "category_match_method": "embeddings",
                "category_match_score": 0.99,
            }

    class _DummyInputs(dict):
        def to(self, _device):
            return self

    class _DummyProcessor:
        def __call__(self, **kwargs):
            assert "images" in kwargs
            return _DummyInputs({"pixel_values": torch.tensor([[1.0, 1.0, 1.0]])})

    class _DummyModel:
        logit_scale = torch.tensor(0.0)
        logit_bias = torch.tensor(0.0)

        @staticmethod
        def get_image_features(**kwargs):
            # More aligned with Category One than Category Two.
            class _Output:
                pooler_output = torch.tensor([[2.0, 0.2, 0.0]], dtype=torch.float32)

            return _Output()

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "sample ocr"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Brand X",
                "category": "Raw Category",
                "confidence": 1.0,
                "reasoning": "ok",
            }

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: ([{"image": object(), "ocr_image": object(), "time": 1.5}], None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())
    monkeypatch.setattr(pipeline_module.categories_runtime, "siglip_model", _DummyModel())
    monkeypatch.setattr(
        pipeline_module.categories_runtime, "siglip_processor", _DummyProcessor()
    )

    stages = []

    def _stage_cb(stage, detail):
        stages.append((stage, detail))

    vision_scores, per_frame_vision, ocr_text, _, _, row = pipeline_module.process_single_video(
        url="https://example.test/ad.mp4",
        categories=[],
        p="Ollama",
        m="qwen3-vl:8b-instruct",
        oe="EasyOCR",
        om="🚀 Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=True,
        ctx=8192,
        job_id="job-vision-1",
        stage_callback=_stage_cb,
    )

    assert vision_scores
    assert len(per_frame_vision) == 1
    assert per_frame_vision[0]["top_category"] == "Category One"
    assert ocr_text == "sample ocr"
    assert list(vision_scores.keys())[0] == "Category One"
    assert row[1] == "Brand X"
    assert row[2] == "101"
    assert any(stage == "vision" for stage, _ in stages)


def test_pipeline_passes_canonical_fallback_categories_to_llm(monkeypatch):
    class _DummyMapper:
        categories = ["Category One", "Category Two"]

        @staticmethod
        def map_category(**kwargs):
            return {
                "canonical_category": "Category One",
                "category_id": "101",
                "category_match_method": "embeddings",
                "category_match_score": 0.99,
            }

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "sample ocr"

    llm_calls: list[dict] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            llm_calls.append({"args": args, "kwargs": kwargs})
            return {
                "brand": "Brand X",
                "category": "Raw Category",
                "confidence": 1.0,
                "reasoning": "ok",
            }

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: ([{"image": object(), "ocr_image": object(), "time": 1.5, "type": "tail"}], None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    pipeline_module.process_single_video(
        url="https://example.test/ad.mp4",
        categories=[],
        p="LM Studio",
        m="local-model",
        oe="EasyOCR",
        om="🚀 Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-lmstudio-1",
    )

    assert len(llm_calls) == 1
    assert llm_calls[0]["args"][2] == "sample ocr"
    assert len(llm_calls[0]["args"]) == 8


def test_pipeline_prefilters_visually_duplicate_tail_frames_before_ocr(monkeypatch):
    class _DummyMapper:
        categories = ["Category One"]

        @staticmethod
        def map_category(**kwargs):
            return {
                "canonical_category": "Category One",
                "category_id": "101",
                "category_match_method": "embeddings",
                "category_match_score": 0.99,
            }

    ocr_calls: list[float] = []

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            ocr_calls.append(float(image[0, 0, 0]))
            return "same endcard text"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Brand X",
                "category": "Raw Category",
                "confidence": 1.0,
                "reasoning": "ok",
            }

    identical_frame = np.full((48, 64, 3), 200, dtype=np.uint8)
    frames = [
        {"image": object(), "ocr_image": identical_frame.copy(), "time": 27.0 + (i * 0.6), "type": "tail"}
        for i in range(5)
    ]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (frames, None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    pipeline_module.process_single_video(
        url="https://example.test/ad.mp4",
        categories=[],
        p="Ollama",
        m="qwen3-vl:8b-instruct",
        oe="EasyOCR",
        om="🚀 Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-ocr-prefilter-1",
    )

    assert len(ocr_calls) == 1
    assert ocr_calls == [200.0]


def test_extract_ocr_focus_region_returns_smaller_crop_for_text_band():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(frame, "BRAND.COM", (34, 178), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "SAVE MORE", (52, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)

    roi, used = pipeline_module._extract_ocr_focus_region(frame)

    assert used is True
    assert roi.shape[0] < frame.shape[0]
    assert roi.shape[1] < frame.shape[1]


def test_pipeline_easyocr_roi_falls_back_to_full_frame_when_crop_text_is_weak(monkeypatch):
    class _DummyMapper:
        categories = ["Category One"]

        @staticmethod
        def map_category(**kwargs):
            return {
                "canonical_category": "Category One",
                "category_id": "101",
                "category_match_method": "embeddings",
                "category_match_score": 0.99,
            }

    ocr_shapes: list[tuple[int, int]] = []

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            ocr_shapes.append(tuple(image.shape[:2]))
            return "" if len(ocr_shapes) == 1 else "brand signal"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Brand X",
                "category": "Raw Category",
                "confidence": 1.0,
                "reasoning": "ok",
            }

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(frame, "BRAND.COM", (34, 178), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "SAVE MORE", (52, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    frames = [{"image": object(), "ocr_image": frame, "time": 29.4, "type": "tail"}]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (frames, None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    pipeline_module.process_single_video(
        url="https://example.test/ad.mp4",
        categories=[],
        p="Ollama",
        m="qwen3-vl:8b-instruct",
        oe="EasyOCR",
        om="🚀 Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-ocr-roi-1",
    )

    assert len(ocr_shapes) == 2
    assert ocr_shapes[0][0] < frame.shape[0]
    assert ocr_shapes[0][1] < frame.shape[1]
    assert ocr_shapes[1] == frame.shape[:2]


def test_pipeline_tail_ocr_stops_early_after_strong_signal_but_keeps_last_frame(monkeypatch):
    class _DummyMapper:
        categories = ["Category One"]

        @staticmethod
        def map_category(**kwargs):
            return {
                "canonical_category": "Category One",
                "category_id": "101",
                "category_match_method": "embeddings",
                "category_match_score": 0.99,
            }

    ocr_calls: list[int] = []

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            marker = int(image[0, 0, 0])
            ocr_calls.append(marker)
            if marker == 10:
                return "brand.com save more"
            if marker == 40:
                return "final frame"
            return "should not run"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Brand X",
                "category": "Raw Category",
                "confidence": 1.0,
                "reasoning": "ok",
            }

    frames = [
        {"image": object(), "ocr_image": np.full((32, 32, 3), fill_value=v, dtype=np.uint8), "time": 27.0 + i, "type": "tail"}
        for i, v in enumerate((10, 20, 30, 40))
    ]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (frames, None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())
    monkeypatch.setattr(
        pipeline_module,
        "_select_frames_for_ocr",
        lambda incoming_frames: (incoming_frames, 0),
    )

    pipeline_module.process_single_video(
        url="https://example.test/ad.mp4",
        categories=[],
        p="Ollama",
        m="qwen3-vl:8b-instruct",
        oe="EasyOCR",
        om="🚀 Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-ocr-early-stop-1",
    )

    assert ocr_calls == [10, 40]


def test_pipeline_tail_easyocr_skips_nonfinal_frame_when_no_text_roi(monkeypatch):
    class _DummyMapper:
        categories = ["Category One"]

        @staticmethod
        def map_category(**kwargs):
            return {
                "canonical_category": "Category One",
                "category_id": "101",
                "category_match_method": "embeddings",
                "category_match_score": 0.99,
            }

    ocr_calls: list[int] = []

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            marker = int(np.count_nonzero(image))
            ocr_calls.append(marker)
            return "brand signal"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Brand X",
                "category": "Raw Category",
                "confidence": 1.0,
                "reasoning": "ok",
            }

    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    text_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(text_frame, "BRAND.COM", (34, 178), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    frames = [
        {"image": object(), "ocr_image": blank, "time": 27.0, "type": "tail"},
        {"image": object(), "ocr_image": text_frame, "time": 29.4, "type": "tail"},
    ]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (frames, None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())
    monkeypatch.setattr(
        pipeline_module,
        "_select_frames_for_ocr",
        lambda incoming_frames: (incoming_frames, 0),
    )

    pipeline_module.process_single_video(
        url="https://example.test/ad.mp4",
        categories=[],
        p="Ollama",
        m="qwen3-vl:8b-instruct",
        oe="EasyOCR",
        om="🚀 Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-ocr-no-roi-1",
    )

    assert len(ocr_calls) == 1
    assert ocr_calls[0] > 0


def test_pipeline_logs_ocr_call_metrics(monkeypatch, caplog):
    class _DummyMapper:
        categories = ["Category One"]

        @staticmethod
        def map_category(**kwargs):
            return {
                "canonical_category": "Category One",
                "category_id": "101",
                "category_match_method": "embeddings",
                "category_match_score": 0.99,
            }

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return f"text-{int(image[0, 0, 0])}"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Brand X",
                "category": "Raw Category",
                "confidence": 1.0,
                "reasoning": "ok",
            }

    frames = [
        {"image": object(), "ocr_image": np.full((32, 32, 3), fill_value=v, dtype=np.uint8), "time": 27.0 + i, "type": "tail"}
        for i, v in enumerate((10, 20))
    ]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (frames, None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())
    monkeypatch.setattr(
        pipeline_module,
        "_select_frames_for_ocr",
        lambda incoming_frames: (incoming_frames, 0),
    )

    caplog.set_level(logging.INFO, logger="video_service.core")

    pipeline_module.process_single_video(
        url="https://example.test/ad.mp4",
        categories=[],
        p="Ollama",
        m="qwen3-vl:8b-instruct",
        oe="EasyOCR",
        om="🚀 Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-ocr-metrics-1",
    )

    summary_logs = [record.getMessage() for record in caplog.records if "ocr_dedup:" in record.getMessage()]
    assert summary_logs
    assert "ocr_calls=2" in summary_logs[-1]
    assert "ocr_elapsed_ms=" in summary_logs[-1]
    assert "avg_ocr_ms=" in summary_logs[-1]
