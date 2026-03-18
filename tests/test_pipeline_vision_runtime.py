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


def test_extract_ocr_domains_ignores_malformed_two_label_www_anchor():
    domains = pipeline_module._extract_ocr_domains(
        "WWW.HISTORICACANADA Heritage Minute www.historicacanada.ca"
    )

    assert "WWW.HISTORICACANADA" not in domains
    assert "www.historicacanada.ca" in domains


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
        om="Fast",
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
        om="Fast",
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


def test_pipeline_configures_requested_category_embedding_model(monkeypatch):
    configured_models: list[str | None] = []

    class _DummyMapper:
        categories = ["Category One"]

        @staticmethod
        def configure_embedding_model(model_name=None):
            configured_models.append(model_name)
            return str(model_name or "default")

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
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        category_embedding_model="sentence-transformers/all-mpnet-base-v2",
        job_id="job-embedding-model-1",
    )

    assert configured_models == ["sentence-transformers/all-mpnet-base-v2"]


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
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-ocr-prefilter-1",
    )

    assert len(ocr_calls) == 2
    assert ocr_calls == [200.0, 200.0]


def test_select_llm_evidence_frames_keeps_product_frame_and_single_latest_logo_endcard():
    product_frame = np.full((72, 96, 3), (0, 140, 255), dtype=np.uint8)
    cv2.putText(product_frame, "iPhone", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    logo_frame = np.zeros((72, 96, 3), dtype=np.uint8)
    cv2.circle(logo_frame, (48, 24), 5, (255, 255, 255), -1)

    frames = [
        {"ocr_image": product_frame.copy(), "time": 10.0, "type": "tail"},
        {"ocr_image": logo_frame.copy(), "time": 10.5, "type": "tail"},
        {"ocr_image": logo_frame.copy(), "time": 11.0, "type": "tail"},
        {"ocr_image": logo_frame.copy(), "time": 11.5, "type": "tail"},
        {"ocr_image": logo_frame.copy(), "time": 12.0, "type": "tail"},
    ]

    selected = pipeline_module._select_llm_evidence_frames(frames, frame_limit=4)

    assert [round(frame["time"], 1) for frame in selected] == [10.0, 12.0]


def test_pipeline_passes_representative_frame_pack_to_llm(monkeypatch):
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
            return "ROGERS"

    product_frame = np.full((72, 96, 3), (0, 140, 255), dtype=np.uint8)
    cv2.putText(product_frame, "iPhone", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    logo_frame = np.zeros((72, 96, 3), dtype=np.uint8)
    cv2.circle(logo_frame, (48, 24), 5, (255, 255, 255), -1)

    frames = [
        {"ocr_image": product_frame.copy(), "time": 10.0, "type": "tail"},
        {"ocr_image": logo_frame.copy(), "time": 10.5, "type": "tail"},
        {"ocr_image": logo_frame.copy(), "time": 11.0, "type": "tail"},
        {"ocr_image": logo_frame.copy(), "time": 11.5, "type": "tail"},
        {"ocr_image": logo_frame.copy(), "time": 12.0, "type": "tail"},
    ]

    llm_calls: list[dict[str, object]] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            llm_calls.append({"args": args, "kwargs": kwargs})
            return {
                "brand": "Rogers",
                "category": "Telecommunications",
                "confidence": 1.0,
                "reasoning": "ok",
            }

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: (frames, None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, _, _, gallery, _, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/rogers-iphone.mp4",
        categories=[],
        p="LM Studio",
        m="local-model",
        oe="EasyOCR",
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        enable_llm_frame=True,
        ctx=8192,
        job_id="job-llm-frame-pack-1",
    )

    assert len(llm_calls) == 1
    evidence_images = llm_calls[0]["kwargs"]["evidence_images"]
    assert len(evidence_images) == 2
    first_mean = float(np.array(evidence_images[0]).mean())
    second_mean = float(np.array(evidence_images[1]).mean())
    assert first_mean > second_mean
    assert len(gallery) == 5
    assert [label for _, label in signal_artifacts["llm_evidence_gallery"]] == ["10.0s", "12.0s"]


def test_pipeline_category_rerank_corrects_weak_cross_domain_mapping(monkeypatch):
    class _DummyMapper:
        categories = [
            "Electric Power Generation",
            "Automotive",
            "Car/Van",
            "City Cars",
            "Micro Cars",
        ]

        @staticmethod
        def map_category(**kwargs):
            raw = str(kwargs.get("raw_category", "") or "")
            if raw == "Electric Vehicles":
                return {
                    "canonical_category": "Electric Power Generation",
                    "category_id": "329",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.5964,
                }
            if raw == "Automotive":
                return {
                    "canonical_category": "Automotive",
                    "category_id": "399",
                    "category_match_method": "embeddings",
                    "category_match_score": 1.0,
                }
            return {
                "canonical_category": raw or "Unknown",
                "category_id": "",
                "category_match_method": "embeddings",
                "category_match_score": 0.5,
            }

        @staticmethod
        def get_mapper_neighbor_categories(
            raw_category,
            predicted_brand="",
            ocr_summary="",
            reasoning_summary="",
            top_k=8,
        ):
            query = str(raw_category or "")
            if query == "Electric Vehicles":
                return [
                    ("Electric Power Generation", 0.5964),
                    ("Automotive", 0.5884),
                    ("Car/Van", 0.5852),
                    ("City Cars", 0.5595),
                    ("Micro Cars", 0.5577),
                ][:top_k]
            if "Toyota" in query or "bZ" in query:
                return [
                    ("Automotive", 0.6421),
                    ("Car/Van", 0.6310),
                    ("Electric Power Generation", 0.4110),
                ][:top_k]
            return [("Electric Power Generation", 0.5964), ("Automotive", 0.5884)][:top_k]

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "Toyota bZ"

    rerank_calls: list[dict[str, object]] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Toyota",
                "category": "Electric Vehicles",
                "confidence": 0.96,
                "reasoning": "Toyota bZ electric vehicle launch ad.",
            }

        @staticmethod
        def query_category_rerank(*args, **kwargs):
            rerank_calls.append({"args": args, "kwargs": kwargs})
            return (
                {
                    "brand": "Toyota",
                    "category": "Automotive",
                    "confidence": 0.93,
                    "reasoning": "Toyota bZ is an electric vehicle, so the ad belongs in the automotive family.",
                },
                "ok",
            )

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: ([{"image": object(), "ocr_image": object(), "time": 1.5, "type": "tail"}], None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, _, _, _, row, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/toyota-bz.mp4",
        categories=[],
        p="LM Studio",
        m="local-model",
        oe="EasyOCR",
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-category-rerank-1",
    )

    assert len(rerank_calls) == 1
    assert row[1] == "Toyota"
    assert row[2] == "399"
    assert row[3] == "Automotive"
    attempts = signal_artifacts["processing_trace"]["attempts"]
    assert any(
        attempt.get("attempt_type") == "category_rerank" and attempt.get("status") == "accepted"
        for attempt in attempts
    )


def test_pipeline_skips_category_rerank_for_confident_mapping(monkeypatch):
    class _DummyMapper:
        categories = ["Automotive", "Car/Van", "City Cars"]

        @staticmethod
        def map_category(**kwargs):
            raw = str(kwargs.get("raw_category", "") or "")
            if raw == "Automotive":
                return {
                    "canonical_category": "Automotive",
                    "category_id": "399",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.711,
                }
            return {
                "canonical_category": raw or "Unknown",
                "category_id": "",
                "category_match_method": "embeddings",
                "category_match_score": 0.5,
            }

        @staticmethod
        def get_mapper_neighbor_categories(
            raw_category,
            predicted_brand="",
            ocr_summary="",
            reasoning_summary="",
            top_k=8,
        ):
            return [
                ("Automotive", 0.7110),
                ("Car/Van", 0.6610),
                ("City Cars", 0.6400),
            ][:top_k]

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "Toyota"

    rerank_calls: list[dict[str, object]] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Toyota",
                "category": "Automotive",
                "confidence": 0.94,
                "reasoning": "Toyota automotive ad.",
            }

        @staticmethod
        def query_category_rerank(*args, **kwargs):
            rerank_calls.append({"args": args, "kwargs": kwargs})
            return None, "should_not_run"

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: ([{"image": object(), "ocr_image": object(), "time": 1.5, "type": "tail"}], None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, _, _, _, row, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/toyota.mp4",
        categories=[],
        p="LM Studio",
        m="local-model",
        oe="EasyOCR",
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-category-rerank-2",
    )

    assert rerank_calls == []
    assert row[2] == "399"
    assert row[3] == "Automotive"
    attempts = signal_artifacts["processing_trace"]["attempts"]
    assert not any(attempt.get("attempt_type") == "category_rerank" for attempt in attempts)


def test_pipeline_category_rerank_refines_broad_family_mapping_with_local_evidence(monkeypatch):
    class _DummyMapper:
        categories = [
            "Detergents",
            "Dry Cleaning and Laundry Facilities",
            "Household cleaners",
            "Household Products",
            "Laundry Care",
        ]

        @staticmethod
        def map_category(**kwargs):
            raw = str(kwargs.get("raw_category", "") or "")
            if raw == "Consumer Goods / Household Cleaning & Laundry":
                return {
                    "canonical_category": "Household cleaners",
                    "category_id": "5233",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.7430,
                }
            if raw == "Laundry Care":
                return {
                    "canonical_category": "Laundry Care",
                    "category_id": "9275",
                    "category_match_method": "embeddings",
                    "category_match_score": 1.0,
                }
            return {
                "canonical_category": raw or "Unknown",
                "category_id": "",
                "category_match_method": "embeddings",
                "category_match_score": 0.5,
            }

        @staticmethod
        def get_mapper_neighbor_categories(
            raw_category,
            predicted_brand="",
            ocr_summary="",
            reasoning_summary="",
            top_k=8,
        ):
            query = str(raw_category or "")
            if query == "Consumer Goods / Household Cleaning & Laundry":
                return [
                    ("Household cleaners", 0.7430),
                    ("Laundry Care", 0.6909),
                    ("Dry Cleaning and Laundry Facilities", 0.6820),
                    ("Household Products", 0.6591),
                    ("Detergents", 0.6300),
                ][:top_k]
            if "Tide" in query and "Downy" in query and "laundry" in query.lower():
                return [
                    ("Laundry Care", 0.6909),
                    ("Detergents", 0.6300),
                    ("Household cleaners", 0.5810),
                ][:top_k]
            return [
                ("Household cleaners", 0.7430),
                ("Laundry Care", 0.6909),
            ][:top_k]

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "Concours Tide & Downy pour la rentrée scolaire."

    rerank_calls: list[dict[str, object]] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Tide & Downy",
                "category": "Consumer Goods / Household Cleaning & Laundry",
                "confidence": 1.0,
                "reasoning": (
                    "The ad promotes Tide detergent and Downy fabric softener products "
                    "for back-to-school laundry use."
                ),
            }

        @staticmethod
        def query_category_rerank(*args, **kwargs):
            rerank_calls.append({"args": args, "kwargs": kwargs})
            return (
                {
                    "brand": "Tide & Downy",
                    "category": "Laundry Care",
                    "confidence": 0.95,
                    "reasoning": "Laundry Care is the strongest nearby in-family category.",
                },
                "ok",
            )

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: ([{"image": object(), "ocr_image": object(), "time": 1.5, "type": "tail"}], None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, _, _, _, row, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/tide-downy.mp4",
        categories=[],
        p="LM Studio",
        m="local-model",
        oe="EasyOCR",
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-category-rerank-4",
    )

    assert len(rerank_calls) == 1
    assert row[2] == "9275"
    assert row[3] == "Laundry Care"
    attempts = signal_artifacts["processing_trace"]["attempts"]
    assert any(
        attempt.get("attempt_type") == "category_rerank"
        and "family_evidence_prefers='Laundry Care'" in str(attempt.get("trigger_reason", ""))
        and attempt.get("status") == "accepted"
        for attempt in attempts
    )


def test_pipeline_category_rerank_preserves_exact_taxonomy_leaf_without_remapping(monkeypatch):
    class _DummyMapper:
        categories = [
            "Retail Stores/Chains",
            "Grocery Stores",
            "Grocery Stores and Supermarkets",
            "Supermarkets",
        ]
        cat_to_id = {
            "Retail Stores/Chains": "403",
            "Grocery Stores": "9355",
            "Grocery Stores and Supermarkets": "19",
            "Supermarkets": "9323",
        }

        @staticmethod
        def map_category(**kwargs):
            raw = str(kwargs.get("raw_category", "") or "")
            if raw == "Grocery":
                return {
                    "canonical_category": "Grocery Stores",
                    "category_id": "9355",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.7842,
                    "mapping_query_text": "Grocery",
                }
            if raw == "Grocery Stores":
                return {
                    "canonical_category": "Retail Stores/Chains",
                    "category_id": "403",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.5377,
                    "mapping_query_text": "No Frills Loblaws Inc Prices effect until February",
                }
            return {
                "canonical_category": raw or "Unknown",
                "category_id": "",
                "category_match_method": "embeddings",
                "category_match_score": 0.5,
                "mapping_query_text": raw,
            }

        @staticmethod
        def get_mapper_neighbor_categories(
            raw_category,
            predicted_brand="",
            ocr_summary="",
            reasoning_summary="",
            top_k=8,
        ):
            query = str(raw_category or "")
            if query == "Grocery":
                return [
                    ("Grocery Stores", 0.7842),
                    ("Supermarkets", 0.7826),
                    ("Grocery Stores and Supermarkets", 0.7637),
                    ("Grocery Subscriptions", 0.7152),
                ][:top_k]
            if "No Frills" in query:
                return [
                    ("Retail Stores/Chains", 0.5377),
                    ("Discount and Warehouse Sales", 0.5363),
                    ("Retail and General Merchandise", 0.5274),
                    ("Grocery Stores", 0.5176),
                ][:top_k]
            return [("Grocery Stores", 0.7842), ("Supermarkets", 0.7826)][:top_k]

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "NOFRILLS Loblaws Inc Prices in effect until February 2026 YES SAVINGS"

    rerank_calls: list[dict[str, object]] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "No Frills",
                "category": "Grocery",
                "confidence": 0.99,
                "reasoning": "No Frills grocery promotion.",
            }

        @staticmethod
        def query_category_rerank(*args, **kwargs):
            rerank_calls.append({"args": args, "kwargs": kwargs})
            return (
                {
                    "brand": "No Frills",
                    "category": "Grocery Stores",
                    "confidence": 0.99,
                    "reasoning": "Grocery Stores is the most precise supported category.",
                },
                "ok",
            )

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: ([{"image": object(), "ocr_image": object(), "time": 1.5, "type": "tail"}], None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, _, _, _, row, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/no-frills.mp4",
        categories=[],
        p="LM Studio",
        m="local-model",
        oe="EasyOCR",
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-category-rerank-grocery",
    )

    assert len(rerank_calls) == 1
    assert row[2] == "9355"
    assert row[3] == "Grocery Stores"
    attempts = signal_artifacts["processing_trace"]["attempts"]
    assert any(
        attempt.get("attempt_type") == "category_rerank"
        and attempt.get("status") == "rejected"
        and "unchanged_category='Grocery Stores'" in str(attempt.get("evidence_note", ""))
        for attempt in attempts
    )


def test_pipeline_category_rerank_triggers_for_weak_freeform_mapping_without_extra_probe_contradiction(monkeypatch):
    class _DummyMapper:
        categories = [
            "Electric Power Generation",
            "Automotive",
            "Car/Van",
            "City Cars",
        ]

        @staticmethod
        def map_category(**kwargs):
            raw = str(kwargs.get("raw_category", "") or "")
            if raw == "Electric Vehicles":
                return {
                    "canonical_category": "Electric Power Generation",
                    "category_id": "329",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.5964,
                }
            if raw == "Automotive":
                return {
                    "canonical_category": "Automotive",
                    "category_id": "399",
                    "category_match_method": "embeddings",
                    "category_match_score": 1.0,
                }
            return {
                "canonical_category": raw or "Unknown",
                "category_id": "",
                "category_match_method": "embeddings",
                "category_match_score": 0.5,
            }

        @staticmethod
        def get_mapper_neighbor_categories(
            raw_category,
            predicted_brand="",
            ocr_summary="",
            reasoning_summary="",
            top_k=8,
        ):
            query = str(raw_category or "")
            if query == "Electric Vehicles":
                return [
                    ("Electric Power Generation", 0.5964),
                    ("Automotive", 0.5884),
                    ("Car/Van", 0.5852),
                    ("City Cars", 0.5595),
                ][:top_k]
            return [
                ("Electric Power Generation", 0.6120),
                ("Automotive", 0.6040),
                ("Car/Van", 0.5980),
            ][:top_k]

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "bZ"

    rerank_calls: list[dict[str, object]] = []

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Toyota",
                "category": "Electric Vehicles",
                "confidence": 0.99,
                "reasoning": "Toyota bZ electric vehicle ad.",
            }

        @staticmethod
        def query_category_rerank(*args, **kwargs):
            rerank_calls.append({"args": args, "kwargs": kwargs})
            return (
                {
                    "brand": "Toyota",
                    "category": "Automotive",
                    "confidence": 0.92,
                    "reasoning": "Within the supplied candidates, Automotive is the safer in-family category.",
                },
                "ok",
            )

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: ([{"image": object(), "ocr_image": object(), "time": 1.5, "type": "tail"}], None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, _, _, _, row, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/toyota-bz-freeform.mp4",
        categories=[],
        p="LM Studio",
        m="local-model",
        oe="EasyOCR",
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-category-rerank-3",
    )

    assert len(rerank_calls) == 1
    assert row[2] == "399"
    assert row[3] == "Automotive"
    attempts = signal_artifacts["processing_trace"]["attempts"]
    assert any(
        attempt.get("attempt_type") == "category_rerank"
        and "freeform_category_with_weak_mapping" in str(attempt.get("trigger_reason", ""))
        for attempt in attempts
    )


def test_pipeline_category_rerank_selects_supported_family_before_leaf(monkeypatch):
    family_calls = {"count": 0}
    rerank_calls = {"count": 0}

    class _DummyMapper:
        categories = [
            "Book Publishers",
            "Bookstores",
            "Retail Subscription Services",
            "Household Appliance Manufacture",
            "Personal Audio",
        ]
        cat_to_id = {
            "Book Publishers": "290",
            "Bookstores": "200",
            "Retail Subscription Services": "9345",
            "Household Appliance Manufacture": "123",
            "Personal Audio": "9141",
        }
        mapping_state = types.SimpleNamespace(
            records=(
                types.SimpleNamespace(category_id="290", name="Book Publishers", path_names=("Book Publishers",), parent_id="0"),
                types.SimpleNamespace(category_id="200", name="Bookstores", path_names=("Bookstores",), parent_id="0"),
                types.SimpleNamespace(category_id="9345", name="Retail Subscription Services", path_names=("Retail Subscription Services",), parent_id="0"),
                types.SimpleNamespace(category_id="123", name="Household Appliance Manufacture", path_names=("Household Appliance Manufacture",), parent_id="0"),
                types.SimpleNamespace(category_id="9141", name="Personal Audio", path_names=("Household Appliance Manufacture", "Personal Audio"), parent_id="123"),
            )
        )

        @staticmethod
        def map_category(**kwargs):
            raw = str(kwargs.get("raw_category", "") or "")
            mapping = {
                "Audio Books": ("Bookstores", "200", 0.58),
                "Book Publishers": ("Book Publishers", "290", 0.99),
                "Personal Audio": ("Personal Audio", "9141", 0.99),
            }
            canonical, category_id, score = mapping.get(raw, (raw or "Bookstores", "", 0.5))
            return {
                "canonical_category": canonical,
                "category_id": category_id,
                "category_match_method": "embeddings",
                "category_match_score": score,
            }

        @staticmethod
        def get_mapper_neighbor_categories(raw_category, predicted_brand="", ocr_summary="", reasoning_summary="", top_k=8):
            query = str(raw_category or "")
            if query == "Audio Books":
                return [
                    ("Bookstores", 0.58),
                    ("Personal Audio", 0.57),
                    ("Book Publishers", 0.56),
                    ("Retail Subscription Services", 0.55),
                ][:top_k]
            if "Audible" in query or "audiobook" in query.lower():
                return [
                    ("Book Publishers", 0.62),
                    ("Bookstores", 0.60),
                    ("Retail Subscription Services", 0.59),
                    ("Personal Audio", 0.52),
                ][:top_k]
            return [("Bookstores", 0.58), ("Personal Audio", 0.57), ("Book Publishers", 0.56)][:top_k]

        @staticmethod
        def get_category_context_map(labels):
            return {
                "Book Publishers": "Book Publishers",
                "Bookstores": "Bookstores",
                "Retail Subscription Services": "Retail Subscription Services",
                "Household Appliance Manufacture": "Household Appliance Manufacture",
                "Personal Audio": "Household Appliance Manufacture : Personal Audio",
            }

        @staticmethod
        def get_category_path_text(label):
            return _DummyMapper.get_category_context_map([]).get(label, label)

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "Audible original audiobook listen now"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Audible",
                "category": "Audio Books",
                "confidence": 0.99,
                "reasoning": "The ad promotes audiobook listening through Audible.",
            }

        @staticmethod
        def query_category_family_selection(*args, **kwargs):
            family_calls["count"] += 1
            return (
                {
                    "family": "Book Publishers",
                    "family_index": 3,
                    "confidence": 0.94,
                    "reasoning": "The ad is about audiobook content, not audio hardware.",
                },
                "ok",
            )

        @staticmethod
        def query_category_rerank(*args, **kwargs):
            rerank_calls["count"] += 1
            return None, "should_not_run_after_family_constrains_to_single_candidate"

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: ([{"image": object(), "ocr_image": object(), "time": 1.5, "type": "tail"}], None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, _, _, _, row, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/audible.mp4",
        categories=[],
        p="LM Studio",
        m="local-model",
        oe="EasyOCR",
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-category-family-audible",
    )

    assert family_calls["count"] == 1
    assert rerank_calls["count"] == 0
    assert row[2] == "290"
    assert row[3] == "Book Publishers"
    attempts = signal_artifacts["processing_trace"]["attempts"]
    assert any(
        attempt.get("attempt_type") == "category_rerank"
        and attempt.get("status") == "accepted"
        and "Selected family 'Book Publishers'" in str(attempt.get("evidence_note", ""))
        for attempt in attempts
    )


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
        om="Fast",
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
        om="Fast",
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
        om="Fast",
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
        om="Fast",
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
        om="Fast",
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
        om="Fast",
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
        om="Fast",
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
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-edge-rescue-1",
    )

    assert rescue_calls == ["called"]
    assert ocr_modes == ["Fast", "Detailed"]
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
        om="Fast",
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
        om="Fast",
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
        om="Fast",
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
        om="Fast",
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
            if mode == "Fast":
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
        om="Fast",
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
            if mode == "Fast":
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
        om="Fast",
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
            if mode == "Fast":
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
        om="Fast",
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
            if mode == "Fast":
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
        om="Fast",
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
        om="Fast",
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
        om="Fast",
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


def test_pipeline_accepts_entity_search_rescue_for_movie_title(monkeypatch):
    grounding_calls = []
    entity_calls = []

    class _DummyMapper:
        categories = ["Comedy Cinema", "Action/Thriller Cinema", "Theater"]
        mapping_state = types.SimpleNamespace(
            records=(
                types.SimpleNamespace(
                    category_id="5280",
                    name="Cinema Genre",
                    path_names=("Movies & TV Production and Distribution", "Cinema Genre"),
                    parent_id="303",
                ),
                types.SimpleNamespace(
                    category_id="5281",
                    name="Action/Thriller Cinema",
                    path_names=(
                        "Movies & TV Production and Distribution",
                        "Cinema Genre",
                        "Action/Thriller Cinema",
                    ),
                    parent_id="5280",
                ),
                types.SimpleNamespace(
                    category_id="5282",
                    name="Comedy Cinema",
                    path_names=(
                        "Movies & TV Production and Distribution",
                        "Cinema Genre",
                        "Comedy Cinema",
                    ),
                    parent_id="5280",
                ),
                types.SimpleNamespace(
                    category_id="5101",
                    name="Theater",
                    path_names=("Entertainment and Performance Arts", "Event", "Theater"),
                    parent_id="5097",
                ),
            )
        )

        @staticmethod
        def get_mapper_neighbor_categories(**kwargs):
            return [("Comedy Cinema", 0.6108), ("Action/Thriller Cinema", 0.89)]

        @staticmethod
        def get_category_context_map(labels):
            return {
                "Comedy Cinema": "Movies & TV Production and Distribution : Cinema Genre : Comedy Cinema",
                "Action/Thriller Cinema": "Movies & TV Production and Distribution : Cinema Genre : Action/Thriller Cinema",
                "Theater": "Entertainment and Performance Arts : Event : Theater",
            }

        @staticmethod
        def get_category_path_text(label):
            return _DummyMapper.get_category_context_map([]).get(label, label)

        @staticmethod
        def map_category(**kwargs):
            raw = kwargs.get("raw_category", "")
            if raw == "Movie":
                return {
                    "canonical_category": "Comedy Cinema",
                    "category_id": "5282",
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
                "canonical_category": raw or "Comedy Cinema",
                "category_id": "5282",
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
        def query_entity_grounding(*args, **kwargs):
            grounding_calls.append(kwargs)
            return (
                {
                    "entity_name": "Mercy",
                    "entity_kind": "film_release",
                    "genres": ["action", "thriller"],
                    "confidence": 0.94,
                    "reasoning": "grounded as a theatrical film",
                    "_search_results": [
                        {
                            "title": "Mercy (2025 film)",
                            "href": "https://example.test/mercy",
                            "body": "Mercy is an action thriller film now playing in IMAX.",
                        }
                    ],
                },
                "ok",
            )

        @staticmethod
        def query_entity_search_rescue(*args, **kwargs):
            entity_calls.append(kwargs)
            return (
                {
                    "brand": "Mercy",
                    "entity_name": "Mercy",
                    "entity_kind": "film_release",
                    "genres": ["action", "thriller"],
                    "category": "Action/Thriller Cinema",
                    "confidence": 0.94,
                    "reasoning": "branch constrained selection landed on cinema action/thriller",
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
        om="Fast",
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
    assert grounding_calls
    assert entity_calls
    assert entity_calls[0]["branch_label"] == "Cinema Genre"
    assert "Cinema Genre" in entity_calls[0]["candidate_categories"]
    assert "Action/Thriller Cinema" in entity_calls[0]["candidate_categories"]
    assert "Theater" not in entity_calls[0]["candidate_categories"]
    processing_trace = signal_artifacts["processing_trace"]
    assert processing_trace["summary"]["accepted_attempt_type"] == "entity_search_rescue"
    assert processing_trace["attempts"][-1]["attempt_type"] == "entity_search_rescue"
    assert processing_trace["attempts"][-1]["status"] == "accepted"


def test_pipeline_runs_entity_search_before_category_rerank_for_movie_titles(monkeypatch):
    calls = {"entity_grounding": 0, "entity_rescue": 0, "category_rerank": 0}

    class _DummyMapper:
        categories = ["Cinema Genre - All else", "Comedy Cinema", "Drama", "Movie Theatres"]
        mapping_state = types.SimpleNamespace(
            records=(
                types.SimpleNamespace(
                    category_id="5280",
                    name="Cinema Genre",
                    path_names=("Movies & TV Production and Distribution", "Cinema Genre"),
                    parent_id="303",
                ),
                types.SimpleNamespace(
                    category_id="5287",
                    name="Cinema Genre - All else",
                    path_names=(
                        "Movies & TV Production and Distribution",
                        "Cinema Genre",
                        "Cinema Genre - All else",
                    ),
                    parent_id="5280",
                ),
                types.SimpleNamespace(
                    category_id="5282",
                    name="Comedy Cinema",
                    path_names=(
                        "Movies & TV Production and Distribution",
                        "Cinema Genre",
                        "Comedy Cinema",
                    ),
                    parent_id="5280",
                ),
                types.SimpleNamespace(
                    category_id="5290",
                    name="Drama",
                    path_names=(
                        "Movies & TV Production and Distribution",
                        "Cinema Genre",
                        "Drama",
                    ),
                    parent_id="5280",
                ),
                types.SimpleNamespace(
                    category_id="5300",
                    name="Movie Theatres",
                    path_names=("Movies & TV Production and Distribution", "Movie Theatres"),
                    parent_id="303",
                ),
            )
        )

        @staticmethod
        def get_mapper_neighbor_categories(**kwargs):
            return [("Drama", 0.7106), ("Cinema Genre - All else", 0.6914), ("Comedy Cinema", 0.67)]

        @staticmethod
        def get_category_context_map(labels):
            return {
                "Cinema Genre - All else": "Movies & TV Production and Distribution : Cinema Genre : Cinema Genre - All else",
                "Comedy Cinema": "Movies & TV Production and Distribution : Cinema Genre : Comedy Cinema",
                "Drama": "Movies & TV Production and Distribution : Cinema Genre : Drama",
                "Movie Theatres": "Movies & TV Production and Distribution : Movie Theatres",
            }

        @staticmethod
        def get_category_path_text(label):
            return _DummyMapper.get_category_context_map([]).get(label, label)

        @staticmethod
        def map_category(**kwargs):
            raw = kwargs.get("raw_category", "")
            mapping = {
                "Movie": ("Drama", "5290", 0.7106),
                "Comedy Cinema": ("Comedy Cinema", "5282", 0.99),
                "Cinema Genre - All else": ("Cinema Genre - All else", "5287", 0.99),
            }
            canonical, category_id, score = mapping.get(raw, (raw or "Drama", "5290", 0.5))
            return {
                "canonical_category": canonical,
                "category_id": category_id,
                "category_match_method": "embeddings",
                "category_match_score": score,
            }

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "LES FURIES AU CINEMA DES VENDREDI LESFURIES-FILM.COM"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Les Furies",
                "category": "Movie",
                "confidence": 1.0,
                "reasoning": "broad movie label",
            }

        @staticmethod
        def query_entity_grounding(*args, **kwargs):
            calls["entity_grounding"] += 1
            return (
                {
                    "entity_name": "Les Furies",
                    "entity_kind": "film_release",
                    "genres": ["comedy"],
                    "confidence": 0.95,
                    "reasoning": "grounded as a theatrical film",
                    "_search_results": [
                        {
                            "title": "Les Furies film",
                            "href": "https://example.test/les-furies",
                            "body": "Les Furies is a comedy film opening in cinemas.",
                        }
                    ],
                },
                "ok",
            )

        @staticmethod
        def query_entity_search_rescue(*args, **kwargs):
            calls["entity_rescue"] += 1
            return (
                {
                    "brand": "Les Furies",
                    "entity_name": "Les Furies",
                    "entity_kind": "film_release",
                    "genres": ["comedy"],
                    "category": "Comedy Cinema",
                    "confidence": 0.98,
                    "reasoning": "grounded title and web evidence support comedy cinema",
                },
                "ok",
            )

        @staticmethod
        def query_category_rerank(*args, **kwargs):
            calls["category_rerank"] += 1
            return (
                {
                    "brand": "Les Furies",
                    "category_index": 1,
                    "confidence": 0.98,
                    "reasoning": "should not be called when entity search already refined the title-driven movie",
                },
                "ok",
            )

    frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 14.4, "type": "tail"}]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(pipeline_module, "extract_frames_for_pipeline", lambda _url, **kwargs: (frames, None))
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, _, _, _, row, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/les-furies.mp4",
        categories=[],
        p="Llama Server",
        m="Qwen/Qwen3-VL-8B-Instruct-GGUF",
        oe="EasyOCR",
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=True,
        enable_vision_board=False,
        enable_llm_frame=False,
        ctx=8192,
        job_id="job-entity-search-before-rerank",
    )

    assert row[2] == "5282"
    assert row[3] == "Comedy Cinema"
    assert calls["entity_grounding"] == 1
    assert calls["entity_rescue"] == 1
    assert calls["category_rerank"] == 0
    processing_trace = signal_artifacts["processing_trace"]
    assert processing_trace["summary"]["accepted_attempt_type"] == "entity_search_rescue"
    assert [attempt["attempt_type"] for attempt in processing_trace["attempts"]] == [
        "initial",
        "entity_search_rescue",
    ]


def test_pipeline_skips_entity_search_for_non_media_domain_hints(monkeypatch):
    entity_calls = {"grounding": 0, "rescue": 0}

    class _DummyMapper:
        categories = ["Consumer Electronics Stores", "Furniture Stores", "Movie Theatres"]
        mapping_state = types.SimpleNamespace(
            records=(
                types.SimpleNamespace(category_id="205", name="Consumer Electronics Stores", path_names=("Consumer Electronics Stores",), parent_id="0"),
                types.SimpleNamespace(category_id="206", name="Furniture Stores", path_names=("Furniture Stores",), parent_id="0"),
                types.SimpleNamespace(category_id="178", name="Movie Theatres", path_names=("Movie Theatres",), parent_id="0"),
            )
        )

        @staticmethod
        def map_category(**kwargs):
            raw = str(kwargs.get("raw_category", "") or "")
            if raw == "Retail - Electronics & Home Goods":
                return {
                    "canonical_category": "Consumer Electronics Stores",
                    "category_id": "205",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.91,
                }
            return {
                "canonical_category": raw or "Consumer Electronics Stores",
                "category_id": "205",
                "category_match_method": "embeddings",
                "category_match_score": 0.5,
            }

        @staticmethod
        def get_mapper_neighbor_categories(**kwargs):
            return [
                ("Consumer Electronics Stores", 0.91),
                ("Furniture Stores", 0.80),
                ("Movie Theatres", 0.40),
            ]

        @staticmethod
        def get_category_context_map(labels):
            return {
                "Consumer Electronics Stores": "Consumer Electronics Stores",
                "Furniture Stores": "Furniture Stores",
                "Movie Theatres": "Movie Theatres",
            }

        @staticmethod
        def get_category_path_text(label):
            return _DummyMapper.get_category_context_map([]).get(label, label)

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "The Brick thebrick.com low prices on home electronics and furniture"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "The Brick",
                "category": "Retail - Electronics & Home Goods",
                "confidence": 0.99,
                "reasoning": "The ad promotes a retail chain selling electronics and home goods.",
            }

        @staticmethod
        def query_entity_grounding(*args, **kwargs):
            entity_calls["grounding"] += 1
            return None, "should_not_run"

        @staticmethod
        def query_entity_search_rescue(*args, **kwargs):
            entity_calls["rescue"] += 1
            return None, "should_not_run"

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: ([{"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 29.4, "type": "tail"}], None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, _, _, _, row, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/the-brick.mp4",
        categories=[],
        p="Llama Server",
        m="Qwen/Qwen3-VL-8B-Instruct-GGUF",
        oe="EasyOCR",
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=True,
        enable_vision_board=False,
        enable_llm_frame=False,
        ctx=8192,
        job_id="job-entity-search-non-media-skip",
    )

    assert entity_calls == {"grounding": 0, "rescue": 0}
    assert row[2] == "205"
    assert row[3] == "Consumer Electronics Stores"
    attempts = signal_artifacts["processing_trace"]["attempts"]
    assert not any(attempt.get("attempt_type") == "entity_search_rescue" for attempt in attempts)


def test_pipeline_accepts_entity_search_rescue_for_stage_presentation(monkeypatch):
    grounding_calls = []
    entity_calls = []

    class _DummyMapper:
        categories = ["Comedy Cinema", "Action/Thriller Cinema", "Theater", "Entertainment and Performance Arts - All else"]
        mapping_state = types.SimpleNamespace(
            records=(
                types.SimpleNamespace(
                    category_id="5280",
                    name="Cinema Genre",
                    path_names=("Movies & TV Production and Distribution", "Cinema Genre"),
                    parent_id="303",
                ),
                types.SimpleNamespace(
                    category_id="5281",
                    name="Action/Thriller Cinema",
                    path_names=(
                        "Movies & TV Production and Distribution",
                        "Cinema Genre",
                        "Action/Thriller Cinema",
                    ),
                    parent_id="5280",
                ),
                types.SimpleNamespace(
                    category_id="5282",
                    name="Comedy Cinema",
                    path_names=(
                        "Movies & TV Production and Distribution",
                        "Cinema Genre",
                        "Comedy Cinema",
                    ),
                    parent_id="5280",
                ),
                types.SimpleNamespace(
                    category_id="5101",
                    name="Theater",
                    path_names=("Entertainment and Performance Arts", "Event", "Theater"),
                    parent_id="5097",
                ),
                types.SimpleNamespace(
                    category_id="5107",
                    name="Entertainment and Performance Arts - All else",
                    path_names=("Entertainment and Performance Arts", "Entertainment and Performance Arts - All else"),
                    parent_id="385",
                ),
            )
        )

        @staticmethod
        def get_mapper_neighbor_categories(**kwargs):
            return [("Entertainment and Performance Arts - All else", 0.59), ("Theater", 0.58)]

        @staticmethod
        def get_category_context_map(labels):
            return {
                "Comedy Cinema": "Movies & TV Production and Distribution : Cinema Genre : Comedy Cinema",
                "Action/Thriller Cinema": "Movies & TV Production and Distribution : Cinema Genre : Action/Thriller Cinema",
                "Theater": "Entertainment and Performance Arts : Event : Theater",
                "Entertainment and Performance Arts - All else": "Entertainment and Performance Arts : Entertainment and Performance Arts - All else",
            }

        @staticmethod
        def get_category_path_text(label):
            return _DummyMapper.get_category_context_map([]).get(label, label)

        @staticmethod
        def map_category(**kwargs):
            raw = kwargs.get("raw_category", "")
            if raw == "Entertainment":
                return {
                    "canonical_category": "Entertainment and Performance Arts - All else",
                    "category_id": "5107",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.59,
                }
            if raw == "Theater":
                return {
                    "canonical_category": "Theater",
                    "category_id": "5101",
                    "category_match_method": "embeddings",
                    "category_match_score": 0.91,
                }
            return {
                "canonical_category": raw or "Entertainment and Performance Arts - All else",
                "category_id": "5107",
                "category_match_method": "embeddings",
                "category_match_score": 0.5,
            }

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "Hamlet tickets on stage this season"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Hamlet",
                "category": "Entertainment",
                "confidence": 0.99,
                "reasoning": "title-driven performance ad",
            }

        @staticmethod
        def query_entity_grounding(*args, **kwargs):
            grounding_calls.append(kwargs)
            return (
                {
                    "entity_name": "Hamlet",
                    "entity_kind": "stage_production",
                    "genres": ["drama"],
                    "confidence": 0.93,
                    "reasoning": "web results support a stage presentation",
                    "_search_results": [
                        {
                            "title": "Hamlet stage production",
                            "href": "https://example.test/hamlet",
                            "body": "Hamlet returns to the stage this season.",
                        }
                    ],
                },
                "ok",
            )

        @staticmethod
        def query_entity_search_rescue(*args, **kwargs):
            entity_calls.append(kwargs)
            return (
                {
                    "brand": "Hamlet",
                    "entity_name": "Hamlet",
                    "entity_kind": "stage_production",
                    "genres": ["drama"],
                    "category": "Theater",
                    "confidence": 0.93,
                    "reasoning": "branch constrained selection landed on theater",
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
        url="https://example.test/hamlet.mp4",
        categories=[],
        p="Llama Server",
        m="Qwen/Qwen3-VL-8B-Instruct-GGUF",
        oe="EasyOCR",
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=True,
        enable_vision_board=False,
        enable_llm_frame=False,
        ctx=8192,
        job_id="job-entity-search-stage",
    )

    assert row[2] == "5101"
    assert row[3] == "Theater"
    assert grounding_calls
    assert entity_calls
    assert entity_calls[0]["branch_label"] == "Theater"
    assert "Theater" in entity_calls[0]["candidate_categories"]
    assert "Comedy Cinema" not in entity_calls[0]["candidate_categories"]
    processing_trace = signal_artifacts["processing_trace"]
    assert processing_trace["summary"]["accepted_attempt_type"] == "entity_search_rescue"
    assert processing_trace["attempts"][-1]["attempt_type"] == "entity_search_rescue"
    assert processing_trace["attempts"][-1]["status"] == "accepted"


def test_pipeline_entity_search_rescue_can_fall_back_to_branch_parent(monkeypatch):
    class _DummyMapper:
        categories = ["Cinema Genre", "Comedy Cinema", "Action/Thriller Cinema"]
        mapping_state = types.SimpleNamespace(
            records=(
                types.SimpleNamespace(
                    category_id="5280",
                    name="Cinema Genre",
                    path_names=("Movies & TV Production and Distribution", "Cinema Genre"),
                    parent_id="303",
                ),
                types.SimpleNamespace(
                    category_id="5281",
                    name="Action/Thriller Cinema",
                    path_names=("Movies & TV Production and Distribution", "Cinema Genre", "Action/Thriller Cinema"),
                    parent_id="5280",
                ),
                types.SimpleNamespace(
                    category_id="5282",
                    name="Comedy Cinema",
                    path_names=("Movies & TV Production and Distribution", "Cinema Genre", "Comedy Cinema"),
                    parent_id="5280",
                ),
            )
        )

        @staticmethod
        def get_category_context_map(labels):
            return {
                "Cinema Genre": "Movies & TV Production and Distribution : Cinema Genre",
                "Comedy Cinema": "Movies & TV Production and Distribution : Cinema Genre : Comedy Cinema",
                "Action/Thriller Cinema": "Movies & TV Production and Distribution : Cinema Genre : Action/Thriller Cinema",
            }

        @staticmethod
        def get_category_path_text(label):
            return _DummyMapper.get_category_context_map([]).get(label, label)

        @staticmethod
        def map_category(**kwargs):
            raw = kwargs.get("raw_category", "")
            mapping = {
                "Movie": ("Comedy Cinema", "5282", 0.58),
                "Cinema Genre": ("Cinema Genre", "5280", 0.97),
                "Comedy Cinema": ("Comedy Cinema", "5282", 0.97),
            }
            canonical, category_id, score = mapping.get(raw, (raw or "Cinema Genre", "5280", 0.5))
            return {
                "canonical_category": canonical,
                "category_id": category_id,
                "category_match_method": "embeddings",
                "category_match_score": score,
            }

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "Mercy Movie.ca official trailer now playing"

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
        def query_entity_grounding(*args, **kwargs):
            return (
                {
                    "entity_name": "Mercy",
                    "entity_kind": "film_release",
                    "genres": [],
                    "confidence": 0.74,
                    "reasoning": "the ad is clearly a theatrical film but the exact genre is unclear",
                    "_search_results": [{"title": "Mercy film", "href": "https://example.test/mercy", "body": "Mercy opens in theatres soon."}],
                },
                "ok",
            )

        @staticmethod
        def query_entity_search_rescue(*args, **kwargs):
            return (
                {
                    "brand": "Mercy",
                    "entity_name": "Mercy",
                    "entity_kind": "film_release",
                    "genres": [],
                    "category": "Cinema Genre",
                    "confidence": 0.74,
                    "reasoning": "branch parent is safer than guessing the wrong leaf",
                },
                "ok",
            )

    frames = [{"image": object(), "ocr_image": np.full((32, 32, 3), 10, dtype=np.uint8), "time": 29.4, "type": "tail"}]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(pipeline_module, "extract_frames_for_pipeline", lambda _url, **kwargs: (frames, None))
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, _, _, _, row, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/mercy.mp4",
        categories=[],
        p="Llama Server",
        m="Qwen/Qwen3-VL-8B-Instruct-GGUF",
        oe="EasyOCR",
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=True,
        enable_vision_board=False,
        enable_llm_frame=False,
        ctx=8192,
        job_id="job-entity-search-parent",
    )

    assert row[2] == "5280"
    assert row[3] == "Cinema Genre"
    processing_trace = signal_artifacts["processing_trace"]
    assert processing_trace["summary"]["accepted_attempt_type"] == "entity_search_rescue"


def test_category_rerank_candidates_include_evidence_neighbors(monkeypatch):
    class _DummyMapper:
        @staticmethod
        def get_mapper_neighbor_categories(
            raw_category,
            predicted_brand="",
            ocr_summary="",
            reasoning_summary="",
            top_k=5,
        ):
            if raw_category == "Hair Care":
                if predicted_brand:
                    assert predicted_brand == "Pantene"
                    assert reasoning_summary == "Pantene Miracle Rescue shampoos and conditioners."
                return [
                    ("Haircare products", 0.5612),
                    ("Haircare products - All else", 0.5499),
                    ("Hair loss product", 0.5348),
                    ("Hair Care Services", 0.4964),
                    ("Hair removal product", 0.4816),
                ][:top_k]

            assert raw_category in {
                "Pantene Miracle Rescue shampoo conditioner",
                "Pantene hair garbled should dominate compact query",
            }
            return [
                ("Shampoo/Conditioner", 0.6707),
                ("Haircare products", 0.6112),
                ("Haircare products - All else", 0.6013),
            ][:top_k]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())

    candidates, evidence_neighbors, primary_candidates = pipeline_module._build_category_rerank_candidates(
        raw_category="Hair Care",
        current_match={
            "canonical_category": "Haircare products",
            "category_match_score": 0.5612,
        },
        predicted_brand="Pantene",
        ocr_text="garbled ocr that should not dominate the compact evidence query",
        reasoning="Pantene Miracle Rescue shampoos and conditioners.",
    )

    labels = [label for label, _score in candidates]
    assert labels[0] == "Shampoo/Conditioner"
    assert primary_candidates[0][0] == "Haircare products"
    assert "Shampoo/Conditioner" in labels
    assert evidence_neighbors[0][0] == "Shampoo/Conditioner"


def test_category_rerank_triggers_for_specific_freeform_mismatch(monkeypatch):
    class _DummyMapper:
        categories = [
            "Women's perfume",
            "Air fresheners",
            "Home Products",
            "Men's perfume",
        ]

        @staticmethod
        def get_mapper_neighbor_categories(
            raw_category,
            predicted_brand="",
            ocr_summary="",
            reasoning_summary="",
            top_k=5,
        ):
            if raw_category == "Home Fragrance Diffusers":
                return [
                    ("Women's perfume", 0.7400),
                    ("Air fresheners", 0.7210),
                    ("Home Products", 0.7090),
                    ("Men's perfume", 0.6880),
                ][:top_k]
            return [
                ("Air fresheners", 0.7010),
                ("Home Products", 0.6840),
                ("Women's perfume", 0.6300),
            ][:top_k]

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())

    should_rerank, reason, candidates, _visual_matches = pipeline_module._should_run_category_rerank(
        result_payload={
            "brand": "Febreze",
            "category": "Home Fragrance Diffusers",
            "reasoning": (
                "The frames show Febreze branding and a home fragrance diffuser with refill cartridges."
            ),
        },
        category_match={
            "canonical_category": "Women's perfume",
            "category_match_method": "embeddings",
            "category_match_score": 0.74,
        },
        ocr_text="Febreze diffuser refill home fragrance",
        sorted_vision={},
    )

    assert should_rerank is True
    assert "freeform_label_mismatch" in reason
    assert candidates[:4] == [
        "Women's perfume",
        "Air fresheners",
        "Home Products",
        "Men's perfume",
    ]


def test_category_rerank_triggers_for_broad_neighbor_dispersion(monkeypatch):
    class _DummyMapper:
        categories = [
            "Alcoholic beverages",
            "Non-alcoholic beverages",
            "Beverage Manufacture and Bottling",
            "Fast Food and Quickservice Restaurants",
            "Food Distributors",
            "Appetizers and Dips",
        ]

        @staticmethod
        def get_mapper_neighbor_categories(
            raw_category,
            predicted_brand="",
            ocr_summary="",
            reasoning_summary="",
            top_k=6,
        ):
            query = str(raw_category or "")
            if query == "Food & Beverage":
                return [
                    ("Alcoholic beverages", 0.8080),
                    ("Non-alcoholic beverages", 0.7757),
                    ("Beverage Manufacture and Bottling", 0.7706),
                    ("Fast Food and Quickservice Restaurants", 0.7505),
                    ("Food Distributors", 0.7476),
                ][:top_k]
            if "Avocados From Mexico" in query or "guac" in query.lower():
                return [
                    ("Appetizers and Dips", 0.7920),
                    ("Food Distributors", 0.7810),
                    ("Non-alcoholic beverages", 0.6400),
                ][:top_k]
            return []

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())

    should_rerank, reason, candidates, _visual_matches = pipeline_module._should_run_category_rerank(
        result_payload={
            "brand": "Avocados From Mexico",
            "category": "Food & Beverage",
            "reasoning": "The ad promotes avocados and guacamole as a food product.",
        },
        category_match={
            "canonical_category": "Alcoholic beverages",
            "category_match_method": "embeddings",
            "category_match_score": 0.8080,
        },
        ocr_text="FOOTBALL and GUAC Avocados From Mexico ALWAYS GOOD",
        sorted_vision={
            "Appetizers and Dips": 0.62,
            "Food Distributors": 0.54,
        },
    )

    assert should_rerank is True
    assert "broad_neighbor_dispersion" in reason
    assert "Appetizers and Dips" in candidates


def test_category_rerank_candidates_expand_supported_family_branch(monkeypatch):
    class _DummyMapper:
        categories = [
            "Healthcare Services",
            "Mental Healthcare",
            "Rehabilitation Therapy Practices",
            "Home Healthcare Services",
            "Spa Services",
            "Online Travel Services",
        ]
        cat_to_id = {
            "Healthcare Services": "10",
            "Mental Healthcare": "11",
            "Rehabilitation Therapy Practices": "12",
            "Home Healthcare Services": "13",
            "Spa Services": "188",
            "Online Travel Services": "200",
        }
        _parent_ids = {
            "Mental Healthcare": "10",
            "Rehabilitation Therapy Practices": "10",
            "Home Healthcare Services": "10",
        }
        _path_text = {
            "Healthcare Services": "Healthcare Services",
            "Mental Healthcare": "Healthcare Services : Mental Healthcare",
            "Rehabilitation Therapy Practices": "Healthcare Services : Rehabilitation Therapy Practices",
            "Home Healthcare Services": "Healthcare Services : Home Healthcare Services",
            "Spa Services": "Spa Services",
            "Online Travel Services": "Online Travel Services",
        }

        @staticmethod
        def get_category_parent_id(label):
            return _DummyMapper._parent_ids.get(label, "")

        @staticmethod
        def get_category_path_text(label):
            return _DummyMapper._path_text.get(label, label)

        @staticmethod
        def get_mapper_neighbor_categories(
            raw_category,
            predicted_brand="",
            ocr_summary="",
            reasoning_summary="",
            top_k=5,
        ):
            query = str(raw_category or "")
            if query == "Online Therapy Services" and not predicted_brand and not ocr_summary and not reasoning_summary:
                return [
                    ("Spa Services", 0.6699),
                    ("Online Travel Services", 0.6574),
                    ("Rehabilitation Therapy Practices", 0.6541),
                    ("Home Healthcare Services", 0.6450),
                ][:top_k]
            if query == "Online Therapy Services" and predicted_brand == "betterhelp":
                return [
                    ("Spa Services", 0.6699),
                    ("Rehabilitation Therapy Practices", 0.6541),
                    ("Home Healthcare Services", 0.6450),
                    ("Online Travel Services", 0.6420),
                ][:top_k]
            if "betterhelp" in query.lower() or "therapy" in query.lower() or "platform" in query.lower():
                return [
                    ("Rehabilitation Therapy Practices", 0.6610),
                    ("Mental Healthcare", 0.6522),
                    ("Home Healthcare Services", 0.6335),
                ][:top_k]
            return []

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())

    candidates, evidence_neighbors, primary_candidates = pipeline_module._build_category_rerank_candidates(
        raw_category="Online Therapy Services",
        current_match={
            "canonical_category": "Spa Services",
            "category_match_score": 0.6699,
        },
        predicted_brand="betterhelp",
        ocr_text="Discover the power of help on the world's largest online therapy platform. betterhelp.com",
        reasoning="The ad promotes BetterHelp as an online therapy platform and mental health service.",
    )

    labels = [label for label, _score in candidates]
    assert primary_candidates[0][0] == "Spa Services"
    assert evidence_neighbors[0][0] == "Rehabilitation Therapy Practices"
    assert "Mental Healthcare" in labels
    assert labels.index("Rehabilitation Therapy Practices") < labels.index("Spa Services")


def test_category_rerank_branch_expansion_does_not_flood_unrelated_siblings(monkeypatch):
    class _DummyMapper:
        categories = [
            "Pharmaceutical Manufacture and Sale - over the counter",
            "Birth Control Products",
            "Painkiller",
            "Nasal products",
            "Anti-Snoring",
            "Anti-Nausea",
            "Jam/preserves/honey",
        ]
        cat_to_id = {
            "Pharmaceutical Manufacture and Sale - over the counter": "373",
            "Birth Control Products": "384",
            "Painkiller": "5344",
            "Nasal products": "5357",
            "Anti-Snoring": "5300",
            "Anti-Nausea": "5301",
            "Jam/preserves/honey": "6000",
        }
        _parent_ids = {
            "Painkiller": "373",
            "Nasal products": "373",
            "Anti-Snoring": "373",
            "Anti-Nausea": "373",
        }
        _path_text = {
            "Pharmaceutical Manufacture and Sale - over the counter": "Pharmaceutical Manufacture and Sale - over the counter",
            "Birth Control Products": "Birth Control Products",
            "Painkiller": "Pharmaceutical Manufacture and Sale - over the counter : Painkiller",
            "Nasal products": "Pharmaceutical Manufacture and Sale - over the counter : Nasal products",
            "Anti-Snoring": "Pharmaceutical Manufacture and Sale - over the counter : Anti-Snoring",
            "Anti-Nausea": "Pharmaceutical Manufacture and Sale - over the counter : Anti-Nausea",
            "Jam/preserves/honey": "Food - Packaged : Jam/preserves/honey",
        }

        @staticmethod
        def get_category_parent_id(label):
            return _DummyMapper._parent_ids.get(label, "")

        @staticmethod
        def get_category_path_text(label):
            return _DummyMapper._path_text.get(label, label)

        @staticmethod
        def get_mapper_neighbor_categories(
            raw_category,
            predicted_brand="",
            ocr_summary="",
            reasoning_summary="",
            top_k=8,
        ):
            query = str(raw_category or "")
            if query == "Over-the-Counter Medication":
                return [
                    ("Birth Control Products", 0.7351),
                    ("Pharmaceutical Manufacture and Sale - over the counter", 0.7239),
                    ("Painkiller", 0.6670),
                    ("Nasal products", 0.6610),
                ][:top_k]
            if "SORE THROAT" in query or "VapoCOOL" in query:
                return [
                    ("Painkiller", 0.5813),
                    ("Nasal products", 0.5790),
                    ("Birth Control Products", 0.5684),
                ][:top_k]
            if "counter" in query.lower():
                return [
                    ("Jam/preserves/honey", 0.6190),
                    ("Pharmaceutical Manufacture and Sale - over the counter", 0.5475),
                    ("Birth Control Products", 0.5534),
                ][:top_k]
            return []

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())

    candidates, _evidence_neighbors, _primary_candidates = pipeline_module._build_category_rerank_candidates(
        raw_category="Over-the-Counter Medication",
        current_match={
            "canonical_category": "Birth Control Products",
            "category_match_score": 0.7351,
        },
        predicted_brand="Vicks",
        ocr_text="Vicks VapoCOOL VAPORIZE MAX Honey Lemon Chill SORE THROAT PAIN",
        reasoning="Over-the-counter Vicks medication for sore throat relief.",
        visual_matches=[("Nasal products", 0.0846)],
    )

    labels = [label for label, _score in candidates]
    assert "Nasal products" in labels
    assert "Painkiller" in labels
    assert "Anti-Snoring" not in labels
    assert "Anti-Nausea" not in labels


def test_pipeline_category_rerank_expands_selected_family_before_leaf_choice(monkeypatch):
    rerank_calls = {"count": 0}

    class _DummyMapper:
        categories = [
            "Pharmaceutical Manufacture and Sale - over the counter",
            "Birth Control Products",
            "Painkiller",
            "Nasal products",
            "Cough Flu-analgesics",
            "Optical",
        ]
        cat_to_id = {
            "Pharmaceutical Manufacture and Sale - over the counter": "373",
            "Birth Control Products": "384",
            "Painkiller": "5357",
            "Nasal products": "5362",
            "Cough Flu-analgesics": "5344",
            "Optical": "5354",
        }
        mapping_state = types.SimpleNamespace(
            records=(
                types.SimpleNamespace(category_id="373", name="Pharmaceutical Manufacture and Sale - over the counter", path_names=("Pharmaceutical Manufacture and Sale - over the counter",), parent_id="0"),
                types.SimpleNamespace(category_id="384", name="Birth Control Products", path_names=("Birth Control Products",), parent_id="0"),
                types.SimpleNamespace(category_id="5357", name="Painkiller", path_names=("Pharmaceutical Manufacture and Sale - over the counter", "Painkiller"), parent_id="373"),
                types.SimpleNamespace(category_id="5362", name="Nasal products", path_names=("Pharmaceutical Manufacture and Sale - over the counter", "Nasal products"), parent_id="373"),
                types.SimpleNamespace(category_id="5344", name="Cough Flu-analgesics", path_names=("Pharmaceutical Manufacture and Sale - over the counter", "Cough Flu-analgesics"), parent_id="373"),
                types.SimpleNamespace(category_id="5354", name="Optical", path_names=("Pharmaceutical Manufacture and Sale - over the counter", "Optical"), parent_id="373"),
            )
        )

        @staticmethod
        def get_category_parent_id(label):
            mapping = {
                "Painkiller": "373",
                "Nasal products": "373",
                "Cough Flu-analgesics": "373",
                "Optical": "373",
            }
            return mapping.get(label, "")

        @staticmethod
        def get_category_context_map(labels):
            return {
                "Pharmaceutical Manufacture and Sale - over the counter": "Pharmaceutical Manufacture and Sale - over the counter",
                "Birth Control Products": "Birth Control Products",
                "Painkiller": "Pharmaceutical Manufacture and Sale - over the counter : Painkiller",
                "Nasal products": "Pharmaceutical Manufacture and Sale - over the counter : Nasal products",
                "Cough Flu-analgesics": "Pharmaceutical Manufacture and Sale - over the counter : Cough Flu-analgesics",
                "Optical": "Pharmaceutical Manufacture and Sale - over the counter : Optical",
            }

        @staticmethod
        def get_category_path_text(label):
            return _DummyMapper.get_category_context_map([]).get(label, label)

        @staticmethod
        def map_category(**kwargs):
            raw = str(kwargs.get("raw_category", "") or "")
            mapping = {
                "Over-the-Counter Medication": ("Birth Control Products", "384", 0.7351),
                "Cough Flu-analgesics": ("Cough Flu-analgesics", "5344", 0.99),
                "Painkiller": ("Painkiller", "5357", 0.99),
            }
            canonical, category_id, score = mapping.get(raw, (raw or "Birth Control Products", "", 0.5))
            return {
                "canonical_category": canonical,
                "category_id": category_id,
                "category_match_method": "embeddings",
                "category_match_score": score,
            }

        @staticmethod
        def get_mapper_neighbor_categories(raw_category, predicted_brand="", ocr_summary="", reasoning_summary="", top_k=8):
            query = str(raw_category or "")
            if query == "Over-the-Counter Medication":
                return [
                    ("Birth Control Products", 0.7351),
                    ("Pharmaceutical Manufacture and Sale - over the counter", 0.7239),
                    ("Painkiller", 0.6670),
                    ("Optical", 0.6610),
                ][:top_k]
            if "SORE THROAT" in query or "VapoCOOL" in query:
                return [
                    ("Painkiller", 0.5813),
                    ("Nasal products", 0.5790),
                    ("Cough Flu-analgesics", 0.5775),
                ][:top_k]
            return [("Birth Control Products", 0.7351), ("Painkiller", 0.6670), ("Optical", 0.6610)][:top_k]

    class _DummyOCR:
        @staticmethod
        def extract_text(engine, image, mode):
            return "VICKS VapoCOOL MAX Honey Lemon Chill SORE THROAT PAIN"

    class _DummyLLM:
        @staticmethod
        def query_pipeline(*args, **kwargs):
            return {
                "brand": "Vicks",
                "category": "Over-the-Counter Medication",
                "confidence": 1.0,
                "reasoning": "The ad promotes an OTC Vicks sore-throat product.",
            }

        @staticmethod
        def query_category_family_selection(*args, **kwargs):
            return (
                {
                    "family": "Pharmaceutical Manufacture and Sale - over the counter",
                    "family_index": 1,
                    "confidence": 0.96,
                    "reasoning": "The ad is clearly for an OTC medication family.",
                },
                "ok",
            )

        @staticmethod
        def query_category_rerank(*args, **kwargs):
            rerank_calls["count"] += 1
            candidates = list(kwargs.get("candidate_categories") or [])
            assert "Nasal products" in candidates
            assert "Cough Flu-analgesics" in candidates
            return (
                {
                    "brand": "Vicks",
                    "category": "Cough Flu-analgesics",
                    "confidence": 0.98,
                    "reasoning": "The product is a cough and flu analgesic within OTC medication.",
                },
                "ok",
            )

    monkeypatch.setattr(pipeline_module, "category_mapper", _DummyMapper())
    monkeypatch.setattr(
        pipeline_module,
        "extract_frames_for_pipeline",
        lambda _url, **kwargs: ([{"image": object(), "ocr_image": object(), "time": 1.5, "type": "tail"}], None),
    )
    monkeypatch.setattr(pipeline_module, "ocr_manager", _DummyOCR())
    monkeypatch.setattr(pipeline_module, "llm_engine", _DummyLLM())

    _, _, _, _, _, row, signal_artifacts = pipeline_module.process_single_video(
        url="https://example.test/vicks.mp4",
        categories=[],
        p="LM Studio",
        m="local-model",
        oe="EasyOCR",
        om="Fast",
        override=False,
        sm="Tail Only",
        enable_search=False,
        enable_vision=False,
        ctx=8192,
        job_id="job-category-family-vicks",
    )

    assert rerank_calls["count"] == 1
    assert row[2] == "5344"
    assert row[3] == "Cough Flu-analgesics"
    attempts = signal_artifacts["processing_trace"]["attempts"]
    assert any(
        attempt.get("attempt_type") == "category_rerank"
        and attempt.get("status") == "accepted"
        and "Selected family 'Pharmaceutical Manufacture and Sale - over the counter'" in str(attempt.get("evidence_note", ""))
        for attempt in attempts
    )
