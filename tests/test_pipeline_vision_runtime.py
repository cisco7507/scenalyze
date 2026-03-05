import sys
import types

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


def test_pipeline_skips_prompt_categories_for_forced_json_providers(monkeypatch):
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
    assert llm_calls[0]["kwargs"]["skip_prompt_categories"] is True
    assert llm_calls[0]["args"][3] == ["Category One", "Category Two"]
