import os
import sys
import time
import threading
import torch
import numpy as np
import pytest
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from video_service.core import ocr as ocr_module

pytestmark = pytest.mark.unit


def test_florence_config_compat_shim_sets_forced_bos_token_id(monkeypatch):
    monkeypatch.delattr(PretrainedConfig, "forced_bos_token_id", raising=False)

    mgr = ocr_module.OCRManager()
    mgr._ensure_florence_config_compat()

    assert hasattr(PretrainedConfig, "forced_bos_token_id")
    assert getattr(PretrainedConfig, "forced_bos_token_id") is None


def test_florence_tokenizer_compat_shim_adds_additional_special_tokens(monkeypatch):
    monkeypatch.delattr(PreTrainedTokenizerBase, "additional_special_tokens", raising=False)

    mgr = ocr_module.OCRManager()
    mgr._ensure_florence_tokenizer_compat()
    assert hasattr(PreTrainedTokenizerBase, "additional_special_tokens")

    class _DummyTokenizer:
        def __init__(self):
            self.special_tokens_map = {}

        def add_special_tokens(self, payload):
            self.special_tokens_map.update(payload)

    tok = _DummyTokenizer()
    prop = PreTrainedTokenizerBase.additional_special_tokens
    assert prop.fget(tok) == []
    prop.fset(tok, ["<loc_1>", "<loc_2>"])
    assert prop.fget(tok) == ["<loc_1>", "<loc_2>"]


def test_florence_build_uses_eager_attention_and_flash_guard(monkeypatch):
    captured = {}
    state = {"processor_calls": 0}
    original_verify_tp_plan = ocr_module.hf_modeling_utils.verify_tp_plan

    class _DummyModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    def _fake_model_loader(*args, **kwargs):
        captured["model_args"] = args
        captured["model_kwargs"] = kwargs
        captured["flash_env_during_load"] = os.environ.get("FLASH_ATTN_DISABLED")
        captured["flash_module_during_load"] = sys.modules.get("flash_attn", "MISSING")
        captured["verify_tp_plan_during_load"] = ocr_module.hf_modeling_utils.verify_tp_plan
        return _DummyModel()

    def _fake_processor_loader(*args, **kwargs):
        state["processor_calls"] += 1
        if state["processor_calls"] == 1:
            raise RuntimeError("first attempt failed; retry fallback path")
        captured["processor_args"] = args
        captured["processor_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(ocr_module.AutoModelForCausalLM, "from_pretrained", _fake_model_loader)
    monkeypatch.setattr(ocr_module.AutoProcessor, "from_pretrained", _fake_processor_loader)
    monkeypatch.delenv("FLASH_ATTN_DISABLED", raising=False)
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)

    mgr = ocr_module.OCRManager()
    engine = mgr._build_florence_engine()

    assert engine["type"] == "florence2"
    assert captured["model_kwargs"]["attn_implementation"] == "eager"
    assert captured["model_kwargs"]["trust_remote_code"] is True
    assert captured["flash_env_during_load"] == "1"
    assert captured["flash_module_during_load"] is None
    assert captured["processor_kwargs"]["trust_remote_code"] is True
    assert state["processor_calls"] == 2
    assert "use_fast" not in captured["processor_kwargs"]
    assert os.environ.get("FLASH_ATTN_DISABLED") is None
    assert "flash_attn" not in sys.modules
    assert captured["verify_tp_plan_during_load"] is not original_verify_tp_plan
    assert captured["verify_tp_plan_during_load"]([], {}) is None


def test_florence_build_meta_linspace_guard_avoids_item_on_meta(monkeypatch):
    class _DummyModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    def _fake_model_loader(*args, **kwargs):
        # Simulate Florence remote init behavior that previously crashed under
        # HF's meta-device context.
        _ = torch.linspace(0, 1, 3, device="meta")[0].item()
        return _DummyModel()

    monkeypatch.setattr(ocr_module.AutoModelForCausalLM, "from_pretrained", _fake_model_loader)
    monkeypatch.setattr(ocr_module.AutoProcessor, "from_pretrained", lambda *a, **k: object())

    mgr = ocr_module.OCRManager()
    engine = mgr._build_florence_engine()
    assert engine["type"] == "florence2"


def test_florence_extract_text_disables_cache_for_generation(monkeypatch):
    captured = {}

    class _FakeProcessor:
        def __call__(self, text, images, return_tensors):
            assert text == "<OCR_WITH_REGION>"
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "pixel_values": torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            }

        def batch_decode(self, _generated_ids, skip_special_tokens=False):
            assert skip_special_tokens is False
            return ["<OCR_WITH_REGION> demo"]

        def post_process_generation(self, _text, task, image_size):
            assert task == "<OCR_WITH_REGION>"
            assert image_size == (16, 16)
            return {
                "<OCR_WITH_REGION>": {
                    "labels": ["demo-line"],
                    "quad_boxes": [[0, 0, 10, 0, 10, 2, 0, 2]],
                }
            }

    class _FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.tensor([[0, 1, 2]], dtype=torch.long)

    mgr = ocr_module.OCRManager()
    monkeypatch.setattr(
        mgr,
        "get_engine",
        lambda _name: {"type": "florence2", "model": _FakeModel(), "processor": _FakeProcessor()},
    )
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    text = mgr.extract_text("Florence-2 (Microsoft)", image, mode="🚀 Fast")

    assert text == "demo-line"
    assert captured["use_cache"] is False
    assert captured["num_beams"] == 1
    assert captured["max_new_tokens"] == 256


def test_florence_extract_text_serializes_generate_calls(monkeypatch):
    state = {
        "in_flight": 0,
        "max_in_flight": 0,
    }

    class _FakeProcessor:
        def __call__(self, text, images, return_tensors):
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "pixel_values": torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            }

        def batch_decode(self, _generated_ids, skip_special_tokens=False):
            return ["<OCR_WITH_REGION> demo"]

        def post_process_generation(self, _text, task, image_size):
            return {
                "<OCR_WITH_REGION>": {
                    "labels": ["line"],
                    "quad_boxes": [[0, 0, 10, 0, 10, 2, 0, 2]],
                }
            }

    class _FakeModel:
        def generate(self, **kwargs):
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            time.sleep(0.05)
            state["in_flight"] -= 1
            return torch.tensor([[0, 1, 2]], dtype=torch.long)

    mgr = ocr_module.OCRManager()
    monkeypatch.setattr(
        mgr,
        "get_engine",
        lambda _name: {"type": "florence2", "model": _FakeModel(), "processor": _FakeProcessor()},
    )

    image = np.zeros((16, 16, 3), dtype=np.uint8)

    def _run():
        text = mgr.extract_text("Florence-2 (Microsoft)", image, mode="🚀 Fast")
        assert text == "line"

    t1 = threading.Thread(target=_run)
    t2 = threading.Thread(target=_run)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert state["max_in_flight"] == 1


def test_florence_max_new_tokens_respects_mode_and_env(monkeypatch):
    mgr = ocr_module.OCRManager()
    monkeypatch.delenv("FLORENCE_MAX_NEW_TOKENS", raising=False)
    monkeypatch.delenv("FLORENCE_MAX_NEW_TOKENS_FAST", raising=False)
    assert mgr._resolve_florence_max_new_tokens("🚀 Fast") == 256
    assert mgr._resolve_florence_max_new_tokens("Detailed") == 1024

    monkeypatch.setenv("FLORENCE_MAX_NEW_TOKENS_FAST", "128")
    monkeypatch.setenv("FLORENCE_MAX_NEW_TOKENS", "768")
    assert mgr._resolve_florence_max_new_tokens("🚀 Fast") == 128
    assert mgr._resolve_florence_max_new_tokens("Detailed") == 768


def test_florence_init_failure_falls_back_to_easyocr(monkeypatch):
    class _DummyReader:
        def readtext(self, _image_rgb, detail=1):
            if detail == 0:
                return ["fallback-ocr"]
            return [(
                [(0, 0), (10, 0), (10, 10), (0, 10)],
                "fallback-ocr",
                0.99,
            )]

    monkeypatch.setattr(
        ocr_module.easyocr,
        "Reader",
        lambda *args, **kwargs: _DummyReader(),
    )
    monkeypatch.setattr(
        ocr_module.OCRManager,
        "_build_florence_engine",
        lambda self: (_ for _ in ()).throw(
            AttributeError("'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'")
        ),
    )

    mgr = ocr_module.OCRManager()
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    text = mgr.extract_text("Florence-2 (Microsoft)", image, mode="🚀 Fast")
    assert "fallback-ocr" in text
    assert mgr.florence_unavailable_reason is not None

    # Subsequent Florence requests should not attempt Florence init again.
    before = mgr.florence_unavailable_reason
    text2 = mgr.extract_text("Florence-2 (Microsoft)", image, mode="🚀 Fast")
    assert "fallback-ocr" in text2
    assert mgr.florence_unavailable_reason == before


def test_easyocr_mode_profiles_adjust_readtext_kwargs(monkeypatch):
    captured: list[dict] = []
    shapes: list[tuple[int, int]] = []

    class _DummyReader:
        def readtext(self, _image_rgb, **kwargs):
            shapes.append(tuple(_image_rgb.shape[:2]))
            captured.append(dict(kwargs))
            return [(
                [(0, 0), (10, 0), (10, 10), (0, 10)],
                "mode-line",
                0.99,
            )]

    mgr = ocr_module.OCRManager()
    monkeypatch.setattr(mgr, "get_engine", lambda _name: _DummyReader())
    monkeypatch.setenv("EASYOCR_MAX_DIMENSION_FAST", "64")
    monkeypatch.setenv("EASYOCR_MAX_DIMENSION_DETAILED", "0")
    image = np.zeros((80, 160, 3), dtype=np.uint8)

    fast_text = mgr.extract_text("EasyOCR", image, mode="🚀 Fast")
    detailed_text = mgr.extract_text("EasyOCR", image, mode="🧠 Detailed")

    assert "mode-line" in fast_text
    assert "mode-line" in detailed_text
    assert len(captured) == 2
    assert captured[0]["min_size"] > captured[1]["min_size"]
    assert captured[0]["text_threshold"] > captured[1]["text_threshold"]
    assert shapes[0][1] == 64
    assert shapes[1] == (80, 160)


def test_easyocr_mode_kwargs_fallback_on_type_error(monkeypatch):
    calls: list[dict] = []

    class _LegacyReader:
        def readtext(self, _image_rgb, **kwargs):
            calls.append(dict(kwargs))
            if len(calls) == 1:
                raise TypeError("unexpected keyword argument")
            return [(
                [(0, 0), (10, 0), (10, 10), (0, 10)],
                "legacy-line",
                0.88,
            )]

    mgr = ocr_module.OCRManager()
    monkeypatch.setattr(mgr, "get_engine", lambda _name: _LegacyReader())
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    text = mgr.extract_text("EasyOCR", image, mode="🚀 Fast")

    assert "legacy-line" in text
    assert calls[0]["detail"] == 0
    assert calls[0]["min_size"] == 20
    assert calls[1] == {"detail": 0}
