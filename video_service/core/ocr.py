import os
import sys
import torch
import easyocr
import cv2
import threading
import time
from contextlib import contextmanager
from typing import Any
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import get_imports
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from unittest.mock import patch
import transformers.modeling_utils as hf_modeling_utils
from video_service.core.utils import logger, device, TORCH_DTYPE

class OCRManager:
    def __init__(self):
        self.engines = {}
        self.current_engine = None
        self.current_name = ""
        self.lock = threading.Lock()
        self.florence_infer_lock = threading.Lock()
        self.florence_unavailable_reason = None

    @staticmethod
    def _build_easyocr_reader():
        use_gpu = (device == "cuda")
        logger.info(f"Initializing EasyOCR. GPU mode: {use_gpu}")
        return easyocr.Reader(['en', 'fr'], gpu=use_gpu, verbose=False)

    @staticmethod
    def _ensure_florence_config_compat():
        # Compatibility shim for newer transformers + Florence remote config.
        # Some versions access forced_bos_token_id before instance assignment.
        if not hasattr(PretrainedConfig, "forced_bos_token_id"):
            setattr(PretrainedConfig, "forced_bos_token_id", None)

    @staticmethod
    def _ensure_florence_tokenizer_compat():
        # Compatibility shim for newer tokenizers where Florence remote code
        # still expects tokenizer.additional_special_tokens.
        if hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
            return

        def _get_additional_special_tokens(tokenizer):
            special_tokens = getattr(tokenizer, "special_tokens_map", {}) or {}
            tokens = special_tokens.get("additional_special_tokens")
            return list(tokens) if tokens is not None else []

        def _set_additional_special_tokens(tokenizer, tokens):
            tokenizer.add_special_tokens({"additional_special_tokens": list(tokens or [])})

        setattr(
            PreTrainedTokenizerBase,
            "additional_special_tokens",
            property(_get_additional_special_tokens, _set_additional_special_tokens),
        )

    @staticmethod
    @contextmanager
    def _florence_flash_attn_guard():
        sentinel = object()
        prev_module = sys.modules.get("flash_attn", sentinel)
        prev_env = os.environ.get("FLASH_ATTN_DISABLED")
        os.environ["FLASH_ATTN_DISABLED"] = "1"
        # Some Florence remote-code paths attempt to import flash_attn.
        # On MPS/CPU runtimes this should be treated as unavailable.
        sys.modules["flash_attn"] = None
        try:
            yield
        finally:
            if prev_module is sentinel:
                sys.modules.pop("flash_attn", None)
            else:
                sys.modules["flash_attn"] = prev_module
            if prev_env is None:
                os.environ.pop("FLASH_ATTN_DISABLED", None)
            else:
                os.environ["FLASH_ATTN_DISABLED"] = prev_env

    @staticmethod
    @contextmanager
    def _florence_meta_linspace_guard():
        # Florence remote model init can call `.item()` on values from torch.linspace
        # while HF is temporarily in a meta-device context. Force CPU linspace in this
        # scoped block to prevent RuntimeError: Tensor.item() on meta tensors.
        original_linspace = torch.linspace

        def _safe_linspace(*args, **kwargs):
            requested_device = kwargs.get("device")
            if requested_device is None or str(requested_device) == "meta":
                kwargs["device"] = "cpu"
            return original_linspace(*args, **kwargs)

        with patch.object(torch, "linspace", _safe_linspace):
            yield

    @staticmethod
    def _resolve_florence_max_new_tokens(mode: str) -> int:
        # Keep detailed mode at parity with combined.py defaults.
        detailed_default = int(os.environ.get("FLORENCE_MAX_NEW_TOKENS", "1024"))
        # Fast mode should actually be fast on MPS/CPU; allow override via env.
        fast_default = int(os.environ.get("FLORENCE_MAX_NEW_TOKENS_FAST", "256"))
        return fast_default if "Fast" in mode else detailed_default

    @staticmethod
    def _resolve_easyocr_readtext_kwargs(mode: str) -> dict[str, Any]:
        # EasyOCR has no Florence-style beam toggle.
        # This is a best-effort heuristic split:
        # - Fast: stricter thresholds and larger min_size for speed.
        # - Detailed: lower thresholds to capture more text.
        mode_lower = (mode or "").lower()
        if "fast" in mode_lower:
            return {
                "detail": 0,
                "paragraph": False,
                "min_size": 20,
                "text_threshold": 0.8,
                "width_ths": 0.7,
            }
        return {
            "detail": 1,
            "paragraph": False,
            "min_size": 8,
            "text_threshold": 0.6,
            "width_ths": 0.5,
        }

    @staticmethod
    def _resolve_easyocr_max_dimension(mode: str) -> int:
        mode_lower = (mode or "").lower()
        if "fast" in mode_lower:
            raw = os.environ.get("EASYOCR_MAX_DIMENSION_FAST", "960")
        else:
            raw = os.environ.get("EASYOCR_MAX_DIMENSION_DETAILED", "0")
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            logger.warning("invalid_easyocr_max_dimension mode=%s value=%r fallback=0", mode, raw)
            return 0

    def _prepare_easyocr_image(self, image_rgb: Any, mode: str) -> Any:
        max_dimension = self._resolve_easyocr_max_dimension(mode)
        if max_dimension <= 0 or not hasattr(image_rgb, "shape"):
            return image_rgb

        height, width = image_rgb.shape[:2]
        current_max = max(height, width)
        if current_max <= max_dimension:
            return image_rgb

        scale = max_dimension / float(current_max)
        resized = cv2.resize(
            image_rgb,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        logger.debug(
            "easyocr_resize mode=%s original=%dx%d resized=%dx%d max_dimension=%d",
            mode,
            width,
            height,
            resized.shape[1],
            resized.shape[0],
            max_dimension,
        )
        return resized

    def _build_florence_engine(self):
        def fixed_get_imports(filename):
            imports = get_imports(filename)
            if "flash_attn" in imports:
                imports.remove("flash_attn")
            return imports
        
        def _noop_verify_tp_plan(*args, **kwargs):
            return None

        with (
            self._florence_flash_attn_guard(),
            self._florence_meta_linspace_guard(),
            patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports),
            patch.object(hf_modeling_utils, "verify_tp_plan", _noop_verify_tp_plan),
        ):
            self._ensure_florence_config_compat()
            self._ensure_florence_tokenizer_compat()
            model_id = "microsoft/Florence-2-base"
            logger.info(f"Initializing Florence-2 on {device} with dtype {TORCH_DTYPE}")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=TORCH_DTYPE,
                # Florence remote code on current transformers can trip SDPA checks
                # before language_model is initialized; eager attention avoids that path.
                attn_implementation="eager",
            ).to(device).eval()
            try:
                processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    use_fast=False,
                )
            except Exception:
                processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                )
            return {"type": "florence2", "model": model, "processor": processor}

    def get_engine(self, name):
        with self.lock:
            if name == self.current_name:
                return self.current_engine
            self.current_engine = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if name == "EasyOCR":
                self.current_engine = self._build_easyocr_reader()
            elif name == "Florence-2 (Microsoft)":
                if self.florence_unavailable_reason:
                    logger.warning(
                        "Florence-2 unavailable in this runtime; using EasyOCR fallback: %s",
                        self.florence_unavailable_reason,
                    )
                    self.current_engine = self._build_easyocr_reader()
                else:
                    try:
                        self.current_engine = self._build_florence_engine()
                    except Exception as exc:
                        self.florence_unavailable_reason = f"{type(exc).__name__}: {exc}"
                        logger.warning(
                            "Florence-2 init failed; falling back to EasyOCR: %s",
                            self.florence_unavailable_reason,
                        )
                        self.current_engine = self._build_easyocr_reader()
            self.current_name = name
            return self.current_engine

    def extract_text(self, engine_name: str, image_rgb: Any, mode: str = "Detailed") -> str:
        try:
            engine = self.get_engine(engine_name)

            if isinstance(engine, dict) and engine.get("type") == "florence2":
                # Florence inference is not reliably thread-safe on MPS across
                # concurrent pipeline threads; serialize access per process.
                with self.florence_infer_lock:
                    pil_img = Image.fromarray(image_rgb)
                    inputs = engine["processor"](text="<OCR_WITH_REGION>", images=pil_img, return_tensors="pt")
                    inputs = {k: v.to(device, dtype=TORCH_DTYPE) if torch.is_floating_point(v) and TORCH_DTYPE != torch.float32 else v.to(device) for k, v in inputs.items()}
                    max_new_tokens = self._resolve_florence_max_new_tokens(mode)
                    logger.debug(
                        "Florence OCR start mode=%s max_new_tokens=%d size=%dx%d",
                        mode,
                        max_new_tokens,
                        pil_img.width,
                        pil_img.height,
                    )
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        generated_ids = engine["model"].generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            num_beams=1 if "Fast" in mode else 3,
                            # Florence remote generation expects legacy tuple cache shape.
                            # On modern transformers, EncoderDecoderCache can break that path.
                            use_cache=False,
                        )
                    elapsed = time.perf_counter() - t0
                    logger.debug(
                        "Florence OCR done in %.2fs tokens=%d",
                        elapsed,
                        int(generated_ids.shape[-1]) if hasattr(generated_ids, "shape") else -1,
                    )
                    if elapsed > 30:
                        logger.warning("Florence OCR slow frame took %.2fs", elapsed)
                    parsed = engine["processor"].post_process_generation(engine["processor"].batch_decode(generated_ids, skip_special_tokens=False)[0], task="<OCR_WITH_REGION>", image_size=(pil_img.width, pil_img.height))
                    ocr_data = parsed.get("<OCR_WITH_REGION>", {})
                    annotated = [f"{'[HUGE] ' if (b[5]-b[1])/pil_img.height > 0.15 else ''}{l}" for l, b in zip(ocr_data.get("labels", []), ocr_data.get("quad_boxes", []))]
                    return " ".join(annotated)

            # EasyOCR path (selected explicitly or Florence fallback).
            if engine is None:
                return ""
            if hasattr(engine, "readtext"):
                easyocr_kwargs = self._resolve_easyocr_readtext_kwargs(mode)
                prepared_image = self._prepare_easyocr_image(image_rgb, mode)
                profile = "fast" if "fast" in (mode or "").lower() else "detailed"
                logger.debug(
                    "easyocr_mode profile=%s mode=%s kwargs=%s",
                    profile,
                    mode,
                    easyocr_kwargs,
                )
                try:
                    results = engine.readtext(prepared_image, **easyocr_kwargs)
                except TypeError as exc:
                    logger.warning(
                        "easyocr_mode_kwargs_unsupported mode=%s error=%s; falling back to defaults",
                        mode,
                        exc,
                    )
                    results = engine.readtext(
                        prepared_image,
                        detail=0 if "fast" in (mode or "").lower() else 1,
                    )
                if results and isinstance(results[0], str):
                    return " ".join(text for text in results if str(text).strip())
                annotated = [f"{'[HUGE] ' if (max(p[1] for p in b) - min(p[1] for p in b))/prepared_image.shape[0] > 0.15 else ''}{t}" for b, t, c in results]
                return " ".join(annotated)
            logger.warning("Unknown OCR engine payload type for %s: %s", engine_name, type(engine).__name__)
            return ""
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return ""

ocr_manager = OCRManager()
