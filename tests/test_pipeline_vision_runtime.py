import sys
import types
import logging

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

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
        logit_scale = torch.tensor(2.0)
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

    vision_scores, per_frame_vision, ocr_text, _, _, row, signal_artifacts = pipeline_module.process_single_video(
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
    assert signal_artifacts["mapper_plot"] is None or signal_artifacts["mapper_plot"]["space"] == "mapper"
    assert signal_artifacts["visual_plot"] is None or signal_artifacts["visual_plot"]["space"] == "visual"
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


def test_real_category_mapper_skips_unknown_without_embeddings():
    mapper_cls = type(pipeline_module.category_mapper)
    mapper = mapper_cls.__new__(mapper_cls)

    result = mapper.map_category(raw_category="Unknown", job_id="job-unknown")

    assert result["canonical_category"] == "Unknown"
    assert result["category_id"] == ""
    assert result["category_match_method"] == "skipped_unknown"
    assert result["category_match_score"] is None
    assert result["mapping_query_text"] == ""


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


def test_pipeline_skips_ocr_when_multimodal_tail_is_high_confidence(monkeypatch):
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
            images = kwargs.get("images", [])
            count = max(1, len(images))
            return _DummyInputs(
                {"pixel_values": torch.ones((count, 3), dtype=torch.float32)}
            )

    class _DummyModel:
        logit_scale = torch.tensor(2.0)
        logit_bias = torch.tensor(0.0)

        @staticmethod
        def get_image_features(**kwargs):
            count = int(kwargs["pixel_values"].shape[0])

            class _Output:
                pooler_output = torch.tensor(
                    [[2.0, 0.1, 0.0]] * count,
                    dtype=torch.float32,
                )

            return _Output()

    ocr_calls: list[str] = []

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            ocr_calls.append("called")
            return "ocr should be skipped"

    llm_calls: list[str] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            llm_calls.append(args[2])
            return {
                "brand": "Brand X",
                "category": "Category One",
                "confidence": 0.95,
                "reasoning": "strong visual evidence",
            }

    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    frames = [
        {"image": None, "ocr_image": frame.copy(), "time": 27.0 + i, "type": "tail"}
        for i in range(2)
    ]

    monkeypatch.setenv("OCR_SKIP_HIGH_CONFIDENCE", "true")
    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (frames, None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())
    monkeypatch.setattr(pipeline_module, "_select_frames_for_ocr", lambda incoming: ([incoming[-1]], 1))
    monkeypatch.setattr(pipeline_module, "_frame_quality_allows_ocr_skip", lambda _frame: (True, "ok"))
    monkeypatch.setattr(pipeline_module.categories_runtime, "siglip_model", _DummyModel())
    monkeypatch.setattr(
        pipeline_module.categories_runtime, "siglip_processor", _DummyProcessor()
    )

    _, _, ocr_text, _, _, row, signal_artifacts = pipeline_module.process_single_video(
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
        job_id="job-ocr-skip-1",
    )

    assert ocr_calls == []
    assert llm_calls == [""]
    assert ocr_text == ""
    assert row[1] == "Brand X"
    assert row[2] == "101"


def test_pipeline_runs_ocr_when_multimodal_tail_confidence_is_too_low(monkeypatch):
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
            images = kwargs.get("images", [])
            count = max(1, len(images))
            return _DummyInputs(
                {"pixel_values": torch.ones((count, 3), dtype=torch.float32)}
            )

    class _DummyModel:
        logit_scale = torch.tensor(2.0)
        logit_bias = torch.tensor(0.0)

        @staticmethod
        def get_image_features(**kwargs):
            count = int(kwargs["pixel_values"].shape[0])

            class _Output:
                pooler_output = torch.tensor(
                    [[2.0, 0.1, 0.0]] * count,
                    dtype=torch.float32,
                )

            return _Output()

    ocr_calls: list[str] = []

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            ocr_calls.append("called")
            return "ocr fallback"

    llm_calls: list[str] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            llm_calls.append(args[2])
            if args[2] == "":
                return {
                    "brand": "Brand X",
                    "category": "Category One",
                    "confidence": 0.40,
                    "reasoning": "not enough confidence",
                }
            return {
                "brand": "Brand X",
                "category": "Category One",
                "confidence": 0.95,
                "reasoning": "ocr backed",
            }

    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    frames = [
        {"image": None, "ocr_image": frame.copy(), "time": 27.0 + i, "type": "tail"}
        for i in range(2)
    ]

    monkeypatch.setenv("OCR_SKIP_HIGH_CONFIDENCE", "true")
    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (frames, None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())
    monkeypatch.setattr(pipeline_module, "_select_frames_for_ocr", lambda incoming: ([incoming[-1]], 1))
    monkeypatch.setattr(pipeline_module, "_frame_quality_allows_ocr_skip", lambda _frame: (True, "ok"))
    monkeypatch.setattr(pipeline_module.categories_runtime, "siglip_model", _DummyModel())
    monkeypatch.setattr(
        pipeline_module.categories_runtime, "siglip_processor", _DummyProcessor()
    )

    _, _, ocr_text, _, _, row, signal_artifacts = pipeline_module.process_single_video(
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
        job_id="job-ocr-skip-2",
    )

    assert ocr_calls == ["called"]
    assert llm_calls == ["", "ocr fallback"]
    assert ocr_text == "ocr fallback"
    assert row[1] == "Brand X"


def test_pipeline_does_not_trigger_edge_rescue_when_initial_result_is_usable(monkeypatch):
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
            return "brand signal"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Brand X",
                "category": "Category One",
                "confidence": 0.95,
                "reasoning": "usable",
            }

    frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 29.4, "type": "tail"}]
    rescue_calls: list[str] = []

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (frames, None),
    )
    monkeypatch.setattr(
        pipeline_module,
        "extract_tail_rescue_frames",
        lambda _url, **kwargs: (rescue_calls.append("called"), (frames, None))[1],
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
        job_id="job-edge-rescue-skip-1",
    )

    assert rescue_calls == []


def test_pipeline_edge_rescue_retries_extended_tail_ocr_only_after_blank_initial_result(monkeypatch):
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

    ocr_modes: list[str] = []

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            ocr_modes.append(mode)
            marker = int(image[0, 0, 0])
            if marker == 10:
                return ""
            if marker == 20:
                return "ACME SAVE MORE"
            return ""

    llm_calls: list[tuple[str, bool]] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            text = args[2]
            llm_calls.append((text, bool(kwargs.get("express_mode", False))))
            if text == "":
                return {
                    "brand": "",
                    "category": "",
                    "confidence": 0.0,
                    "reasoning": "blank",
                }
            return {
                "brand": "Acme",
                "category": "Category One",
                "confidence": 0.93,
                "reasoning": "rescued by OCR",
            }

    initial_frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 29.4, "type": "tail"}]
    rescue_frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 20, dtype=np.uint8), "time": 23.0, "type": "tail_rescue"}]
    rescue_calls: list[str] = []

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (initial_frames, None),
    )
    monkeypatch.setattr(
        pipeline_module,
        "extract_tail_rescue_frames",
        lambda _url, **kwargs: (rescue_calls.append("called"), (rescue_frames, None))[1],
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, ocr_text, _, _, row, signal_artifacts = pipeline_module.process_single_video(
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
        job_id="job-edge-rescue-1",
    )

    assert rescue_calls == ["called"]
    assert ocr_modes == ["🚀 Fast", "🧠 Detailed"]
    assert llm_calls == [("", False), ("ACME SAVE MORE", False)]
    assert ocr_text == "ACME SAVE MORE"
    assert row[1] == "Acme"


def test_pipeline_edge_rescue_falls_back_to_image_first_when_rescue_ocr_is_blank(monkeypatch):
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
            return ""

    llm_calls: list[tuple[str, bool]] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            text = args[2]
            express_mode = bool(kwargs.get("express_mode", False))
            llm_calls.append((text, express_mode))
            if express_mode:
                return {
                    "brand": "Visual Brand",
                    "category": "Category One",
                    "confidence": 0.91,
                    "reasoning": "rescued from image",
                }
            return {
                "brand": "",
                "category": "",
                "confidence": 0.0,
                "reasoning": "blank",
            }

    initial_frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 29.4, "type": "tail"}]
    rescue_frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 20, dtype=np.uint8), "time": 23.0, "type": "tail_rescue"}]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (initial_frames, None),
    )
    monkeypatch.setattr(
        pipeline_module,
        "extract_tail_rescue_frames",
        lambda _url, **kwargs: (rescue_frames, None),
    )
    monkeypatch.setattr(
        pipeline_module,
        "extract_express_brand_frame",
        lambda _url, **kwargs: Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8)),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, ocr_text, _, _, row, signal_artifacts = pipeline_module.process_single_video(
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
        enable_llm_frame=True,
        ctx=8192,
        job_id="job-edge-rescue-image-1",
    )

    assert llm_calls == [("", False), ("", True)]
    assert ocr_text == ""
    assert row[1] == "Visual Brand"
    processing_trace = signal_artifacts["processing_trace"]
    assert processing_trace["summary"]["accepted_attempt_type"] == "express_rescue"
    assert [attempt["attempt_type"] for attempt in processing_trace["attempts"]] == [
        "initial",
        "ocr_rescue",
        "express_rescue",
    ]
    assert processing_trace["attempts"][-1]["status"] == "accepted"
    assert processing_trace["attempts"][-1]["elapsed_ms"] is not None


def test_pipeline_edge_rescue_runs_express_before_extended_tail(monkeypatch):
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
            marker = int(image[0, 0, 0])
            if marker == 10:
                return ""
            if marker == 20:
                return ""
            raise AssertionError(f"unexpected OCR marker {marker}")

    llm_calls: list[tuple[str, bool]] = []
    full_video_calls: list[str] = []
    rescue_calls: list[int | None] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            text = args[2]
            express_mode = bool(kwargs.get("express_mode", False))
            llm_calls.append((text, express_mode))
            if express_mode:
                return {
                    "brand": "Express Brand",
                    "category": "Category One",
                    "confidence": 0.94,
                    "reasoning": "rescued by express",
                }
            return {
                "brand": "",
                "category": "",
                "confidence": 0.0,
                "reasoning": "blank",
            }

    initial_frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 29.4, "type": "tail"}]
    rescue_frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 20, dtype=np.uint8), "time": 23.0, "type": "tail_rescue"}]

    def _extract_frames(_url, **kwargs):
        scan_mode = kwargs.get("scan_mode", "Tail Only")
        full_video_calls.append(scan_mode)
        if scan_mode == "Full Video":
            raise AssertionError("full-video rescue should not run when express rescue succeeds")
        return initial_frames, None

    def _extract_tail(_url, **kwargs):
        rescue_calls.append(kwargs.get("lookback_seconds"))
        if kwargs.get("lookback_seconds") is not None:
            raise AssertionError("extended tail should not run when express rescue succeeds")
        return rescue_frames, None

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(pipeline_module, "extract_frames_for_pipeline", _extract_frames)
    monkeypatch.setattr(pipeline_module, "extract_tail_rescue_frames", _extract_tail)
    monkeypatch.setattr(
        pipeline_module,
        "extract_express_brand_frame",
        lambda _url, **kwargs: Image.fromarray(np.full((16, 16, 3), 220, dtype=np.uint8)),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, ocr_text, _, _, row, _ = pipeline_module.process_single_video(
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
        enable_llm_frame=True,
        ctx=8192,
        job_id="job-edge-rescue-express-1",
    )

    assert rescue_calls == [None]
    assert full_video_calls == ["Tail Only"]
    assert llm_calls == [("", False), ("", True)]
    assert ocr_text == ""
    assert row[1] == "Express Brand"


def test_pipeline_edge_rescue_runs_extended_tail_before_full_video(monkeypatch):
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
            marker = int(image[0, 0, 0])
            if marker in {10, 20}:
                return ""
            if marker == 30:
                return "HISTORICA CANADA"
            raise AssertionError(f"unexpected OCR marker {marker}")

    llm_calls: list[tuple[str, bool]] = []
    full_video_calls: list[str] = []
    rescue_calls: list[int | None] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            text = args[2]
            express_mode = bool(kwargs.get("express_mode", False))
            llm_calls.append((text, express_mode))
            if express_mode:
                return {
                    "brand": "",
                    "category": "",
                    "confidence": 0.0,
                    "reasoning": "blank",
                }
            if text == "HISTORICA CANADA":
                return {
                    "brand": "Historica Canada",
                    "category": "Category One",
                    "confidence": 0.98,
                    "reasoning": "rescued by extended tail",
                }
            return {
                "brand": "",
                "category": "",
                "confidence": 0.0,
                "reasoning": "blank",
            }

    initial_frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 29.4, "type": "tail"}]
    first_rescue_frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 20, dtype=np.uint8), "time": 23.0, "type": "tail_rescue"}]
    extended_rescue_frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 30, dtype=np.uint8), "time": 18.0, "type": "tail_rescue"}]

    def _extract_frames(_url, **kwargs):
        scan_mode = kwargs.get("scan_mode", "Tail Only")
        full_video_calls.append(scan_mode)
        if scan_mode == "Full Video":
            raise AssertionError("full-video rescue should not run when extended tail succeeds")
        return initial_frames, None

    def _extract_tail(_url, **kwargs):
        lookback = kwargs.get("lookback_seconds")
        rescue_calls.append(lookback)
        if lookback is None:
            return first_rescue_frames, None
        return extended_rescue_frames, None

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(pipeline_module, "extract_frames_for_pipeline", _extract_frames)
    monkeypatch.setattr(pipeline_module, "extract_tail_rescue_frames", _extract_tail)
    monkeypatch.setattr(
        pipeline_module,
        "extract_express_brand_frame",
        lambda _url, **kwargs: Image.fromarray(np.full((16, 16, 3), 220, dtype=np.uint8)),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, ocr_text, _, _, row, _ = pipeline_module.process_single_video(
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
        enable_llm_frame=True,
        ctx=8192,
        job_id="job-edge-rescue-extended-1",
    )

    assert rescue_calls == [None, 30]
    assert full_video_calls == ["Tail Only"]
    assert llm_calls == [("", False), ("", True), ("HISTORICA CANADA", False)]
    assert ocr_text == "HISTORICA CANADA"
    assert row[1] == "Historica Canada"


def test_pipeline_edge_rescue_runs_full_video_last_and_limits_frames(monkeypatch):
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

    ocr_counts: dict[int, int] = {10: 0, 20: 0, 30: 0, 40: 0}

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            marker = int(image[0, 0, 0])
            ocr_counts[marker] = ocr_counts.get(marker, 0) + 1
            if marker == 40:
                return "FINAL RESCUE SIGNAL"
            return ""

    llm_calls: list[tuple[str, bool]] = []
    scan_mode_calls: list[str] = []
    rescue_calls: list[int | None] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            text = args[2]
            express_mode = bool(kwargs.get("express_mode", False))
            llm_calls.append((text, express_mode))
            if "FINAL RESCUE SIGNAL" in text:
                return {
                    "brand": "Final Rescue Brand",
                    "category": "Category One",
                    "confidence": 0.99,
                    "reasoning": "rescued by full video",
                }
            return {
                "brand": "",
                "category": "",
                "confidence": 0.0,
                "reasoning": "blank",
            }

    initial_frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 29.4, "type": "tail"}]
    default_rescue_frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 20, dtype=np.uint8), "time": 23.0, "type": "tail_rescue"}]
    extended_rescue_frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 30, dtype=np.uint8), "time": 18.0, "type": "tail_rescue"}]
    full_video_frames = [
        {"image": object(), "ocr_image": np.full((32, 32, 3), 40, dtype=np.uint8), "time": float(idx), "type": "scene"}
        for idx in range(40)
    ]

    def _extract_frames(_url, **kwargs):
        scan_mode = kwargs.get("scan_mode", "Tail Only")
        scan_mode_calls.append(scan_mode)
        if scan_mode == "Full Video":
            return full_video_frames, None
        return initial_frames, None

    def _extract_tail(_url, **kwargs):
        lookback = kwargs.get("lookback_seconds")
        rescue_calls.append(lookback)
        if lookback is None:
            return default_rescue_frames, None
        return extended_rescue_frames, None

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(pipeline_module, "extract_frames_for_pipeline", _extract_frames)
    monkeypatch.setattr(pipeline_module, "extract_tail_rescue_frames", _extract_tail)
    monkeypatch.setattr(
        pipeline_module,
        "extract_express_brand_frame",
        lambda _url, **kwargs: Image.fromarray(np.full((16, 16, 3), 220, dtype=np.uint8)),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, ocr_text, _, _, row, _ = pipeline_module.process_single_video(
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
        enable_llm_frame=True,
        ctx=8192,
        job_id="job-edge-rescue-full-video-1",
    )

    assert rescue_calls == [None, 30]
    assert scan_mode_calls == ["Tail Only", "Full Video"]
    assert llm_calls[0] == ("", False)
    assert llm_calls[1] == ("", True)
    assert llm_calls[2][1] is False
    assert "FINAL RESCUE SIGNAL" in llm_calls[2][0]
    assert ocr_counts[40] == pipeline_module._resolve_full_video_rescue_max_frames()
    assert "FINAL RESCUE SIGNAL" in ocr_text
    assert row[1] == "Final Rescue Brand"


def test_pipeline_runs_ocr_context_rescue_for_low_confidence_short_ocr(monkeypatch):
    class _DummyMapper:
        categories = ["Furniture", "Sleep Products"]

        @staticmethod
        def map_category(**kwargs):
            raw = kwargs.get("raw_category", "")
            ocr_summary = kwargs.get("ocr_summary", "")
            if raw == "unknown" and "base réglable" in ocr_summary:
                return {
                    "canonical_category": "Furniture",
                    "category_id": "201",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.91,
                }
            if raw == "Furniture":
                return {
                    "canonical_category": "Furniture",
                    "category_id": "201",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.97,
                }
            return {
                "canonical_category": "Sleep Products",
                "category_id": "202",
                "category_match_method": "embeddings",
                "category_match_score": 0.94,
            }

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            marker = int(image[0, 0, 0])
            if mode == "🚀 Fast":
                if marker == 10:
                    return "DoRMEZ-vous"
                return ""
            if marker == 10:
                return "DoRMEZ-vous"
            if marker == 20:
                return "Zéro gravité"
            if marker == 30:
                return "achat d'une base réglable"
            raise AssertionError(f"unexpected OCR marker {marker}")

    llm_calls: list[tuple[str, bool]] = []
    rescue_calls: list[str] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            text = args[2]
            llm_calls.append((text, bool(kwargs.get("express_mode", False))))
            if "base réglable" in text:
                return {
                    "brand": "Dormez-vous",
                    "category": "Furniture",
                    "confidence": 0.95,
                    "reasoning": "rescued by aggregated OCR context",
                }
            return {
                "brand": "Dormez-vous",
                "category": "Sleep Products",
                "confidence": 0.45,
                "reasoning": "too little OCR context",
            }

    initial_frames = [
        {"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 29.0, "type": "tail"},
        {"image": object(), "ocr_image": np.full((32, 32, 3), 20, dtype=np.uint8), "time": 29.4, "type": "tail"},
        {"image": object(), "ocr_image": np.full((32, 32, 3), 30, dtype=np.uint8), "time": 29.8, "type": "tail"},
    ]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (initial_frames, None),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_select_frames_for_ocr",
        lambda frames: ([frames[0]], len(frames) - 1),
    )
    monkeypatch.setattr(
        pipeline_module,
        "extract_tail_rescue_frames",
        lambda _url, **kwargs: (rescue_calls.append("called"), ([], None))[1],
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, ocr_text, _, _, row, signal_artifacts = pipeline_module.process_single_video(
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
        job_id="job-context-rescue-1",
    )

    assert rescue_calls == []
    assert llm_calls[0] == ("DoRMEZ-vous", False)
    assert "DoRMEZ-vous" in llm_calls[1][0]
    assert "Zéro gravité" in llm_calls[1][0]
    assert "base réglable" in llm_calls[1][0]
    assert llm_calls[1][1] is False
    assert "base réglable" in ocr_text
    assert row[1] == "Dormez-vous"
    assert row[3] == "Furniture"
    processing_trace = signal_artifacts["processing_trace"]
    assert [attempt["attempt_type"] for attempt in processing_trace["attempts"]] == [
        "initial",
        "ocr_context_rescue",
    ]
    assert processing_trace["summary"]["accepted_attempt_type"] == "ocr_context_rescue"


def test_pipeline_runs_ocr_context_rescue_on_short_ocr_when_mapper_score_is_weak(monkeypatch):
    class _DummyMapper:
        categories = ["Furniture", "Sleep Products"]
        vision_text_features = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        )

        @staticmethod
        def ensure_vision_text_features():
            return True, "ready"

        @staticmethod
        def map_category(**kwargs):
            raw = kwargs.get("raw_category", "")
            ocr_summary = kwargs.get("ocr_summary", "")
            if raw == "unknown" and "base réglable" in ocr_summary:
                return {
                    "canonical_category": "Furniture",
                    "category_id": "201",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.91,
                }
            if raw == "Furniture":
                return {
                    "canonical_category": "Furniture",
                    "category_id": "201",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.97,
                }
            return {
                "canonical_category": "Sleep Products",
                "category_id": "202",
                "category_match_method": "embeddings",
                "category_match_score": 0.73,
            }

    class _DummyInputs(dict):
        def to(self, _device):
            return self

    class _DummyProcessor:
        def __call__(self, **kwargs):
            return _DummyInputs({"pixel_values": torch.tensor([[1.0, 1.0]], dtype=torch.float32)})

    class _DummyModel:
        logit_scale = torch.tensor(2.0)
        logit_bias = torch.tensor(0.0)

        @staticmethod
        def get_image_features(**kwargs):
            class _Output:
                pooler_output = torch.tensor([[2.0, 0.1]], dtype=torch.float32)

            return _Output()

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            marker = int(image[0, 0, 0])
            if mode == "🚀 Fast":
                return "DoRMEZ-vous" if marker == 10 else ""
            if marker == 10:
                return "DoRMEZ-vous"
            if marker == 20:
                return "Zéro gravité"
            if marker == 30:
                return "achat d'une base réglable"
            raise AssertionError(f"unexpected OCR marker {marker}")

    llm_calls: list[tuple[str, bool]] = []
    rescue_calls: list[str] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            text = args[2]
            llm_calls.append((text, bool(kwargs.get("express_mode", False))))
            if "base réglable" in text:
                return {
                    "brand": "Dormez-vous",
                    "category": "Furniture",
                    "confidence": 0.96,
                    "reasoning": "weak mapper score corrected by richer OCR",
                }
            return {
                "brand": "Dormez-vous",
                "category": "Sleep Products",
                "confidence": 0.99,
                "reasoning": "brand meaning only",
            }

    frame_image = Image.fromarray(np.full((16, 16, 3), 180, dtype=np.uint8))
    initial_frames = [
        {"image": frame_image, "_pil_cache": frame_image, "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 29.0, "type": "tail"},
        {"image": frame_image, "_pil_cache": frame_image, "ocr_image": np.full((32, 32, 3), 20, dtype=np.uint8), "time": 29.4, "type": "tail"},
        {"image": frame_image, "_pil_cache": frame_image, "ocr_image": np.full((32, 32, 3), 30, dtype=np.uint8), "time": 29.8, "type": "tail"},
    ]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (initial_frames, None),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_select_frames_for_ocr",
        lambda frames: ([frames[0]], len(frames) - 1),
    )
    monkeypatch.setattr(
        pipeline_module,
        "extract_tail_rescue_frames",
        lambda _url, **kwargs: (rescue_calls.append("called"), ([], None))[1],
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())
    monkeypatch.setattr(pipeline_module.categories_runtime, "siglip_model", _DummyModel())
    monkeypatch.setattr(pipeline_module.categories_runtime, "siglip_processor", _DummyProcessor())

    _, _, ocr_text, _, _, row, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/ad.mp4",
        categories=[],
        p="Ollama",
        m="qwen3-vl:8b-instruct",
        oe="EasyOCR",
        om="🚀 Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision_board=True,
        enable_llm_frame=False,
        ctx=8192,
        job_id="job-context-rescue-vision-1",
    )

    assert rescue_calls == []
    assert llm_calls[0] == ("DoRMEZ-vous", False)
    assert "base réglable" in llm_calls[1][0]
    assert "Zéro gravité" in llm_calls[1][0]
    assert "base réglable" in ocr_text
    assert row[3] == "Furniture"
    processing_trace = signal_artifacts["processing_trace"]
    assert processing_trace["summary"]["accepted_attempt_type"] == "ocr_context_rescue"


def test_pipeline_rejects_context_rescue_when_ocr_supports_different_category(monkeypatch):
    class _DummyMapper:
        categories = ["Historical Programming", "Financial Services", "Sleep Products"]

        @staticmethod
        def map_category(**kwargs):
            raw = kwargs.get("raw_category", "")
            ocr_summary = kwargs.get("ocr_summary", "")
            if raw == "Historical Programming":
                return {
                    "canonical_category": "Historical Programming",
                    "category_id": "301",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.97,
                }
            if raw == "Financial Services":
                return {
                    "canonical_category": "Financial Services",
                    "category_id": "391",
                    "category_match_method": "embeddings",
                    "category_match_score": 1.0,
                }
            if raw == "Sleep Products":
                return {
                    "canonical_category": "Sleep Products",
                    "category_id": "5361",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.63,
                }
            if raw == "unknown" and "HERITAGE" in ocr_summary:
                return {
                    "canonical_category": "Historical Programming",
                    "category_id": "301",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.91,
                }
            return {
                "canonical_category": raw or "Historical Programming",
                "category_id": "301",
                "category_match_method": "embeddings",
                "category_match_score": 0.5,
            }

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            marker = int(image[0, 0, 0])
            if mode == "🚀 Fast":
                return "HISTORICA" if marker == 10 else ""
            if marker == 10:
                return "HISTORICA"
            if marker == 20:
                return "HERITAGE MINUTES"
            if marker == 30:
                return "THE CRB FOUNDATION"
            raise AssertionError(f"unexpected OCR marker {marker}")

    llm_calls: list[tuple[str, bool]] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            text = args[2]
            express_mode = bool(kwargs.get("express_mode", False))
            llm_calls.append((text, express_mode))
            if express_mode:
                return {
                    "brand": "Historica",
                    "category": "Historical Programming",
                    "confidence": 0.98,
                    "reasoning": "rescued by express fallback",
                }
            if "HERITAGE MINUTES" in text:
                return {
                    "brand": "Historica",
                    "category": "Financial Services",
                    "confidence": 0.95,
                    "reasoning": "confident but wrong",
                }
            return {
                "brand": "Historica",
                "category": "Sleep Products",
                "confidence": 0.99,
                "reasoning": "too little OCR context",
            }

    initial_frames = [
        {"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 57.8, "type": "tail"},
        {"image": object(), "ocr_image": np.full((32, 32, 3), 20, dtype=np.uint8), "time": 58.4, "type": "tail"},
        {"image": object(), "ocr_image": np.full((32, 32, 3), 30, dtype=np.uint8), "time": 59.0, "type": "tail"},
    ]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (initial_frames, None),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_select_frames_for_ocr",
        lambda frames: ([frames[0]], len(frames) - 1),
    )
    monkeypatch.setattr(
        pipeline_module,
        "extract_tail_rescue_frames",
        lambda _url, **kwargs: ([], None),
    )
    monkeypatch.setattr(
        pipeline_module,
        "extract_express_brand_frame",
        lambda _url, **kwargs: Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8)),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, ocr_text, _, _, row, signal_artifacts = pipeline_module.process_single_video(
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
        enable_llm_frame=True,
        ctx=8192,
        job_id="job-context-rescue-reject-1",
    )

    assert llm_calls[0] == ("HISTORICA", False)
    assert "HERITAGE MINUTES" in llm_calls[1][0]
    assert llm_calls[2] == ("", True)
    assert row[1] == "Historica"
    assert row[3] == "Historical Programming"
    assert ocr_text == ""
    processing_trace = signal_artifacts["processing_trace"]
    assert [attempt["attempt_type"] for attempt in processing_trace["attempts"]] == [
        "initial",
        "ocr_context_rescue",
        "ocr_rescue",
        "express_rescue",
    ]
    assert processing_trace["attempts"][1]["status"] == "rejected"
    assert processing_trace["summary"]["accepted_attempt_type"] == "express_rescue"


def test_pipeline_challenges_generic_context_rescue_with_express_when_ocr_is_non_commercial(monkeypatch):
    class _DummyMapper:
        categories = ["Historical Programming", "Financial Services", "Sleep Remedy"]

        @staticmethod
        def map_category(**kwargs):
            raw = kwargs.get("raw_category", "")
            ocr_summary = kwargs.get("ocr_summary", "")
            if raw == "Sleep Products":
                return {
                    "canonical_category": "Sleep Remedy",
                    "category_id": "5361",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.63,
                }
            if raw == "Financial Services":
                return {
                    "canonical_category": "Financial Services",
                    "category_id": "391",
                    "category_match_method": "embeddings",
                    "category_match_score": 1.0,
                }
            if raw == "Educational Media / Historical Programming":
                return {
                    "canonical_category": "Historical Programming",
                    "category_id": "301",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.78,
                }
            if raw == "unknown" and "HERITAGE MINUTES" in ocr_summary:
                return {
                    "canonical_category": "Financial Services",
                    "category_id": "391",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.86,
                }
            return {
                "canonical_category": raw or "Financial Services",
                "category_id": "391",
                "category_match_method": "embeddings",
                "category_match_score": 0.50,
            }

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            marker = int(image[0, 0, 0])
            if mode == "🚀 Fast":
                return "DoRMEZ-vous" if marker == 10 else ""
            if marker == 10:
                return "HISTORICA"
            if marker == 20:
                return "HERITAGE MINUTES"
            if marker == 30:
                return "THE CRB FOUNDATION"
            raise AssertionError(f"unexpected OCR marker {marker}")

    llm_calls: list[tuple[str, bool]] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            text = args[2]
            express_mode = bool(kwargs.get("express_mode", False))
            llm_calls.append((text, express_mode))
            if express_mode:
                return {
                    "brand": "Historica",
                    "category": "Educational Media / Historical Programming",
                    "confidence": 0.99,
                    "reasoning": "express rescue wins",
                }
            if "HERITAGE MINUTES" in text:
                return {
                    "brand": "Historica",
                    "category": "Financial Services",
                    "confidence": 0.95,
                    "reasoning": "generic but wrong sector guess",
                }
            return {
                "brand": "Dormez-vous",
                "category": "Sleep Products",
                "confidence": 0.99,
                "reasoning": "too little OCR context",
            }

    initial_frames = [
        {"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 57.8, "type": "tail"},
        {"image": object(), "ocr_image": np.full((32, 32, 3), 20, dtype=np.uint8), "time": 58.4, "type": "tail"},
        {"image": object(), "ocr_image": np.full((32, 32, 3), 30, dtype=np.uint8), "time": 59.0, "type": "tail"},
    ]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (initial_frames, None),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_select_frames_for_ocr",
        lambda frames: ([frames[0]], len(frames) - 1),
    )
    monkeypatch.setattr(
        pipeline_module,
        "extract_tail_rescue_frames",
        lambda _url, **kwargs: ([], None),
    )
    monkeypatch.setattr(
        pipeline_module,
        "extract_express_brand_frame",
        lambda _url, **kwargs: Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8)),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, ocr_text, _, _, row, signal_artifacts = pipeline_module.process_single_video(
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
        enable_llm_frame=True,
        ctx=8192,
        job_id="job-context-rescue-challenge-1",
    )

    assert llm_calls[0] == ("DoRMEZ-vous", False)
    assert "HERITAGE MINUTES" in llm_calls[1][0]
    assert llm_calls[2] == ("", True)
    assert row[1] == "Historica"
    assert row[3] == "Historical Programming"
    assert "HISTORICA" in ocr_text
    processing_trace = signal_artifacts["processing_trace"]
    assert processing_trace["summary"]["accepted_attempt_type"] == "express_rescue"
    assert processing_trace["attempts"][1]["attempt_type"] == "ocr_context_rescue"
    assert processing_trace["attempts"][1]["status"] == "rejected"


def test_pipeline_accepts_specificity_search_rescue_for_broad_financial_category(monkeypatch):
    class _DummyMapper:
        categories = ["Financial Services", "Credit Card"]

        @staticmethod
        def get_mapper_neighbor_categories(**kwargs):
            return [("Financial Services", 1.0), ("Credit Card", 0.84)]

        @staticmethod
        def map_category(**kwargs):
            raw = kwargs.get("raw_category", "")
            if raw == "Financial Services":
                return {
                    "canonical_category": "Financial Services",
                    "category_id": "391",
                    "category_match_method": "embeddings",
                    "category_match_score": 1.0,
                }
            if raw == "Credit Card":
                return {
                    "canonical_category": "Credit Card",
                    "category_id": "5028",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.84,
                }
            return {
                "canonical_category": raw or "Financial Services",
                "category_id": "391",
                "category_match_method": "embeddings",
                "category_match_score": 0.5,
            }

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "AM EX Visit Amex.ca/Plat"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "American Express",
                "category": "Financial Services",
                "confidence": 0.99,
                "reasoning": "broad but correct parent",
            }

        @staticmethod
        def query_specificity_rescue(*args, **kwargs):
            return (
                {
                    "brand": "American Express",
                    "category": "Credit Card",
                    "confidence": 0.96,
                    "reasoning": "web refinement found the Platinum card page",
                },
                "ok",
            )

    frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 29.4, "type": "tail"}]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (frames, None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    sorted_vision, _, _, _, _, row, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/amex.mp4",
        categories=[],
        p="Llama Server",
        m="Qwen/Qwen3-VL-8B-Instruct-GGUF",
        oe="EasyOCR",
        om="🚀 Fast",
        override=False,
        sm="Tail Only",
        enable_search=True,
        enable_vision_board=False,
        enable_llm_frame=False,
        ctx=8192,
        job_id="job-specificity-search-1",
    )

    assert sorted_vision == {}
    assert row[1] == "American Express"
    assert row[2] == "5028"
    assert row[3] == "Credit Card"
    processing_trace = signal_artifacts["processing_trace"]
    assert processing_trace["summary"]["accepted_attempt_type"] == "specificity_search_rescue"
    assert processing_trace["attempts"][-1]["attempt_type"] == "specificity_search_rescue"
    assert processing_trace["attempts"][-1]["status"] == "accepted"


def test_pipeline_fails_closed_when_specificity_search_is_unavailable(monkeypatch):
    class _DummyMapper:
        categories = ["Financial Services"]

        @staticmethod
        def get_mapper_neighbor_categories(**kwargs):
            return [("Financial Services", 1.0)]

        @staticmethod
        def map_category(**kwargs):
            return {
                "canonical_category": "Financial Services",
                "category_id": "391",
                "category_match_method": "embeddings",
                "category_match_score": 1.0,
            }

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "AM EX Visit Amex.ca/Plat"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "American Express",
                "category": "Financial Services",
                "confidence": 0.99,
                "reasoning": "broad but correct parent",
            }

        @staticmethod
        def query_specificity_rescue(*args, **kwargs):
            return None, "search_unavailable"

    frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 29.4, "type": "tail"}]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (frames, None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, _, _, _, row, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/amex.mp4",
        categories=[],
        p="Llama Server",
        m="Qwen/Qwen3-VL-8B-Instruct-GGUF",
        oe="EasyOCR",
        om="🚀 Fast",
        override=False,
        sm="Tail Only",
        enable_search=True,
        enable_vision_board=False,
        enable_llm_frame=False,
        ctx=8192,
        job_id="job-specificity-search-2",
    )

    assert row[2] == "391"
    assert row[3] == "Financial Services"
    processing_trace = signal_artifacts["processing_trace"]
    assert processing_trace["summary"]["accepted_attempt_type"] == "initial"
    assert processing_trace["attempts"][-1]["attempt_type"] == "specificity_search_rescue"
    assert processing_trace["attempts"][-1]["status"] == "rejected"


def test_pipeline_triggers_specificity_search_for_generic_raw_category_with_weak_mapper(monkeypatch):
    specificity_calls = []

    class _DummyMapper:
        categories = ["Comedy", "Action/Thriller Cinema"]

        @staticmethod
        def get_mapper_neighbor_categories(**kwargs):
            return [("Comedy", 0.6108), ("Action/Thriller Cinema", 0.89)]

        @staticmethod
        def map_category(**kwargs):
            raw = kwargs.get("raw_category", "")
            if raw == "Movie":
                return {
                    "canonical_category": "Comedy",
                    "category_id": "5297",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.6108,
                }
            if raw == "Action/Thriller Cinema":
                return {
                    "canonical_category": "Action/Thriller Cinema",
                    "category_id": "5281",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.89,
                }
            return {
                "canonical_category": raw or "Comedy",
                "category_id": "5297",
                "category_match_method": "embeddings",
                "category_match_score": 0.5,
            }

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "Mercy Movie.ca FILMED FOR IMAX NOW PLAYING"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Mercy",
                "category": "Movie",
                "confidence": 0.99,
                "reasoning": "generic film label",
            }

        @staticmethod
        def query_specificity_rescue(*args, **kwargs):
            specificity_calls.append(kwargs)
            return (
                {
                    "brand": "Mercy",
                    "category": "Action/Thriller Cinema",
                    "confidence": 0.94,
                    "reasoning": "search refinement found theatrical positioning",
                },
                "ok",
            )

    frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 29.4, "type": "tail"}]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (frames, None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, _, _, _, row, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/mercy.mp4",
        categories=[],
        p="Llama Server",
        m="Qwen/Qwen3-VL-8B-Instruct-GGUF",
        oe="EasyOCR",
        om="🚀 Fast",
        override=False,
        sm="Tail Only",
        enable_search=True,
        enable_vision_board=False,
        enable_llm_frame=False,
        ctx=8192,
        job_id="job-specificity-search-3",
    )

    assert row[2] == "5281"
    assert row[3] == "Action/Thriller Cinema"
    assert specificity_calls
    assert specificity_calls[0]["candidate_categories"] == ["Comedy", "Action/Thriller Cinema"]
    processing_trace = signal_artifacts["processing_trace"]
    assert processing_trace["summary"]["accepted_attempt_type"] == "specificity_search_rescue"
    assert processing_trace["attempts"][-1]["attempt_type"] == "specificity_search_rescue"
    assert processing_trace["attempts"][-1]["status"] == "accepted"
