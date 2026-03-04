import queue
import threading
import concurrent.futures
import random
import time
import io
import base64
import json
import re
import subprocess
import os
from abc import ABC, abstractmethod
from typing import Optional, Protocol

import requests
from PIL import Image
from ddgs import DDGS

from video_service.core.utils import logger

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", "300"))
OPENAI_COMPAT_URL = os.environ.get("OPENAI_COMPAT_URL", "http://localhost:1234/v1/chat/completions")


class SearchManager:
    def __init__(self):
        self.queue = queue.Queue()
        self.client = DDGS()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            query, future = self.queue.get()
            max_retries = 3
            base_delay = 2.0
            success = False
            attempt = 0

            while attempt < max_retries and not success:
                try:
                    results = self.client.text(query, max_results=3)
                    snippets = " | ".join([r.get("body", "") for r in results if isinstance(r, dict)])
                    future.set_result(snippets if snippets else None)
                    success = True
                except Exception as exc:
                    attempt += 1
                    if attempt < max_retries:
                        backoff_sleep = (base_delay**attempt) + random.uniform(0.8, 2.5)
                        time.sleep(backoff_sleep)
                        self.client = DDGS()
                    else:
                        future.set_exception(exc)

            self.queue.task_done()
            if success:
                time.sleep(random.uniform(0.8, 2.5))

    def search(self, query, timeout=45):
        future = concurrent.futures.Future()
        self.queue.put((query, future))
        try:
            return future.result(timeout=timeout)
        except Exception:
            return None


search_manager = SearchManager()


class LLMProvider(Protocol):
    @property
    def supports_vision(self) -> bool:
        ...

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        ...

    def generate_text(
        self,
        prompt: str,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        ...


class BaseProvider(ABC):
    def __init__(self, backend_model: str, context_size: int = 8192) -> None:
        self.backend_model = backend_model
        self.context_size = int(context_size)

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        raise NotImplementedError

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError


def _clean_and_parse_json(raw_text: str) -> dict:
    try:
        text = re.sub(r"\x1b\[[0-9;]*m", "", raw_text)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.replace("```json", "").replace("```", "").strip()
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
        return {"error": "No JSON found", "raw_output": text}
    except Exception as exc:
        return {"error": f"JSON Parse Failed: {str(exc)}"}


class GeminiCLIProvider(BaseProvider):
    @property
    def supports_vision(self) -> bool:
        return False

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        try:
            output = subprocess.run(
                ["gemini", f"{system_prompt}\n\n{user_prompt}"],
                capture_output=True,
                text=True,
            ).stdout
            return _clean_and_parse_json(output)
        except Exception as exc:
            return {"error": str(exc)}

    def generate_text(
        self,
        prompt: str,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        try:
            res = subprocess.run(["gemini", prompt], capture_output=True, text=True)
            if res.returncode != 0 or not res.stdout.strip():
                err = res.stderr.strip() if res.stderr else "No stderr output."
                return f'[TOOL: ERROR | reason="Gemini CLI failed (Code {res.returncode}): {err}"]'
            return res.stdout.strip()
        except Exception as exc:
            return f'[TOOL: ERROR | reason="Fatal Exception: {str(exc)}"]'


class OllamaProvider(BaseProvider):
    @property
    def supports_vision(self) -> bool:
        return True

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        msg = {"role": "user", "content": user_prompt}
        if images:
            msg["images"] = images
        payload = {
            "model": self.backend_model,
            "messages": [{"role": "system", "content": system_prompt}, msg],
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": self.context_size},
        }
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            content = resp.json().get("message", {}).get("content", "")
            return _clean_and_parse_json(content)
        except requests.exceptions.Timeout:
            logger.error(
                "LLM provider %s timed out after %d seconds",
                "Ollama",
                LLM_TIMEOUT_SECONDS,
            )
            return {"error": f"Timeout after {LLM_TIMEOUT_SECONDS}s"}
        except Exception as exc:
            return {"error": str(exc)}

    def generate_text(
        self,
        prompt: str,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        payload = {
            "model": self.backend_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": self.context_size},
        }
        if images:
            payload["images"] = images
        try:
            res = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            if res.status_code != 200:
                return f'[TOOL: ERROR | reason="Ollama HTTP {res.status_code}: {res.text}"]'
            return res.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            logger.error(
                "LLM provider %s timed out after %d seconds",
                "Ollama",
                LLM_TIMEOUT_SECONDS,
            )
            return f'[TOOL: ERROR | reason="LLM Timeout after {LLM_TIMEOUT_SECONDS}s"]'
        except Exception as exc:
            return f'[TOOL: ERROR | reason="Fatal Exception: {str(exc)}"]'


class OpenAICompatibleProvider(BaseProvider):
    def __init__(
        self,
        backend_model: str,
        context_size: int = 8192,
        force_json_mode: bool = False,
    ) -> None:
        super().__init__(backend_model=backend_model, context_size=context_size)
        self.force_json_mode = bool(force_json_mode)

    @property
    def supports_vision(self) -> bool:
        return True

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        json_prefix = '{\n  "brand": "'
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": f"</think>\n{json_prefix}"},
        ]
        if images:
            msgs[1]["content"] = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images[0]}"}},
            ]
        payload = {
            "model": self.backend_model,
            "messages": msgs,
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 2.0,
        }
        if self.force_json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = requests.post(OPENAI_COMPAT_URL, json=payload, timeout=LLM_TIMEOUT_SECONDS)
            resp.raise_for_status()
            content_json = resp.json()
            raw_content = ""
            if "choices" in content_json and len(content_json["choices"]) > 0:
                raw_content = content_json["choices"][0].get("message", {}).get("content", "")
            content = raw_content if "{" in raw_content else f"{json_prefix}{raw_content}"
            return _clean_and_parse_json(content)
        except requests.exceptions.Timeout:
            logger.error(
                "LLM provider %s timed out after %d seconds",
                "LM Studio",
                LLM_TIMEOUT_SECONDS,
            )
            return {"error": f"Timeout after {LLM_TIMEOUT_SECONDS}s"}
        except Exception as exc:
            return {"error": str(exc)}

    def generate_text(
        self,
        prompt: str,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        msgs = [
            {
                "role": "system",
                "content": "You are a ReACT Agent. Strictly follow the prompt formatting.",
            },
            {"role": "user", "content": prompt},
        ]
        if images:
            msgs[1]["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images[0]}"}},
            ]
        payload = {"model": self.backend_model, "messages": msgs, "temperature": 0.1}
        try:
            res = requests.post(OPENAI_COMPAT_URL, json=payload, timeout=LLM_TIMEOUT_SECONDS)
            if res.status_code != 200:
                return f'[TOOL: ERROR | reason="LM Studio HTTP {res.status_code}: {res.text}"]'
            return res.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except requests.exceptions.Timeout:
            logger.error(
                "LLM provider %s timed out after %d seconds",
                "LM Studio",
                LLM_TIMEOUT_SECONDS,
            )
            return f'[TOOL: ERROR | reason="LLM Timeout after {LLM_TIMEOUT_SECONDS}s"]'
        except Exception as exc:
            return f'[TOOL: ERROR | reason="Fatal Exception: {str(exc)}"]'


class OllamaQwenProvider(OllamaProvider):
    """Ollama provider tuned for Qwen chat/instruct sampling presets."""

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        msg = {"role": "user", "content": user_prompt}
        if images:
            msg["images"] = images
        payload = {
            "model": self.backend_model,
            "messages": [{"role": "system", "content": system_prompt}, msg],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 40,
                "presence_penalty": 0.0,
                # Ollama uses repeat_penalty
                "repeat_penalty": 1.0,
                "num_ctx": self.context_size,
            },
        }
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            content = resp.json().get("message", {}).get("content", "")
            return _clean_and_parse_json(content)
        except requests.exceptions.Timeout:
            logger.error(
                "LLM provider %s timed out after %d seconds",
                "OllamaQwen",
                LLM_TIMEOUT_SECONDS,
            )
            return {"error": f"Timeout after {LLM_TIMEOUT_SECONDS}s"}
        except Exception as exc:
            return {"error": str(exc)}

    def generate_text(
        self,
        prompt: str,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        user_msg = {"role": "user", "content": prompt}
        if images:
            user_msg["images"] = images
        payload = {
            "model": self.backend_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a ReACT Agent. Strictly follow the prompt formatting.",
                },
                user_msg,
            ],
            "stream": False,
            "options": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "presence_penalty": 1.5,
                # Ollama uses repeat_penalty
                "repeat_penalty": 1.0,
                "num_ctx": self.context_size,
            },
        }
        try:
            res = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            if res.status_code != 200:
                return f'[TOOL: ERROR | reason="Ollama HTTP {res.status_code}: {res.text}"]'
            return res.json().get("message", {}).get("content", "").strip()
        except requests.exceptions.Timeout:
            logger.error(
                "LLM provider %s timed out after %d seconds",
                "OllamaQwen",
                LLM_TIMEOUT_SECONDS,
            )
            return f'[TOOL: ERROR | reason="LLM Timeout after {LLM_TIMEOUT_SECONDS}s"]'
        except Exception as exc:
            return f'[TOOL: ERROR | reason="Fatal Exception: {str(exc)}"]'


def create_provider(provider: str, backend_model: str, context_size: int = 8192) -> LLMProvider:
    provider_name = (provider or "").strip().lower()
    if provider_name == "gemini cli":
        return GeminiCLIProvider(backend_model=backend_model, context_size=context_size)
    if provider_name == "ollama":
        model_name = (backend_model or "").strip().lower()
        if "qwen" in model_name:
            return OllamaQwenProvider(backend_model=backend_model, context_size=context_size)
        return OllamaProvider(backend_model=backend_model, context_size=context_size)
    if provider_name in {"llama-server", "llama server"}:
        return OpenAICompatibleProvider(
            backend_model=backend_model,
            context_size=context_size,
            force_json_mode=True,
        )
    if provider_name in {"lm studio", "openai compatible", "openai-compatible"}:
        return OpenAICompatibleProvider(
            backend_model=backend_model,
            context_size=context_size,
            force_json_mode=True,
        )
    raise ValueError(f"Unsupported provider: {provider}")


class ClassificationPipeline:
    def __init__(
        self,
        provider: LLMProvider,
        search_client: SearchManager,
        validation_threshold: float,
    ) -> None:
        self.provider = provider
        self.search_client = search_client
        self.validation_threshold = validation_threshold

    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        raw_ocr_text: str,
        enable_search: bool,
        include_image: bool,
        image_b64: Optional[str],
    ) -> dict:
        images = [image_b64] if (include_image and image_b64 and self.provider.supports_vision) else None
        res = self.provider.generate_json(system_prompt, user_prompt, images=images)
        logger.debug("llm_pipeline_initial_result: %s", res)

        brand = res.get("brand", "Unknown") if isinstance(res, dict) else "Unknown"
        if brand.lower() in ["unknown", "none", "n/a", ""] and enable_search:
            words = [w for w in re.sub(r"[^a-zA-Z0-9\s]", "", raw_ocr_text).split() if len(w) > 3]
            if words and (
                snippets := self.search_client.search(" ".join(words[:8]) + " brand company product")
            ):
                res = self.provider.generate_json(
                    system_prompt + "\nAGENTIC RECOVERY",
                    f"OCR: {raw_ocr_text}\nWEB: {snippets}",
                )
                if isinstance(res, dict):
                    res["reasoning"] = "(Recovered) " + res.get("reasoning", "")
                    brand = res.get("brand", "Unknown")

        confidence_raw = res.get("confidence", 0.0) if isinstance(res, dict) else 0.0
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0

        needs_validation = (
            self.validation_threshold <= 0.0
            or confidence <= 0.0
            or confidence < self.validation_threshold
        )

        if (
            isinstance(res, dict)
            and "category" in res
            and enable_search
            and brand.lower() not in ["unknown", "none", "n/a", ""]
        ):
            if needs_validation:
                if self.validation_threshold <= 0.0:
                    logger.debug(
                        "llm_validation_triggered: confidence=%.2f threshold=%.2f (threshold<=0 forces validation)",
                        confidence,
                        self.validation_threshold,
                    )
                else:
                    logger.debug(
                        "llm_validation_triggered: confidence=%.2f < threshold=%.2f",
                        confidence,
                        self.validation_threshold,
                    )

                if val_snippets := self.search_client.search(f"{brand} official brand company"):
                    val_res = self.provider.generate_json(
                        system_prompt + "\nVALIDATION MODE",
                        f"Brand: {brand}\nWeb: {val_snippets}\nCorrect brand name. Keep category {res.get('category')}.",
                    )
                    if isinstance(val_res, dict) and "category" in val_res:
                        return val_res
            else:
                logger.debug(
                    "llm_validation_skipped: confidence=%.2f >= threshold=%.2f",
                    confidence,
                    self.validation_threshold,
                )

        return res if isinstance(res, dict) else {"error": "Unexpected LLM response"}


class HybridLLM:
    def _pil_to_base64(self, pil_image, max_dimension=768):
        if not pil_image:
            return None

        new_w, new_h = pil_image.size
        if max(new_w, new_h) > max_dimension:
            scale = max_dimension / float(max(new_w, new_h))
            new_w = max(1, int(new_w * scale))
            new_h = max(1, int(new_h * scale))

        # IMPORTANT: Qwen3-VL panics if dimensions aren't safely larger than its patch factor (32).
        new_w = max(64, new_w)
        new_h = max(64, new_h)

        if (new_w, new_h) != pil_image.size:
            lanczos = getattr(Image, "Resampling", Image).LANCZOS
            pil_image = pil_image.resize((new_w, new_h), lanczos)

        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _get_validation_threshold(self) -> float:
        validation_threshold_raw = os.environ.get("LLM_VALIDATION_THRESHOLD", "0.7")
        try:
            return float(validation_threshold_raw)
        except (TypeError, ValueError):
            logger.warning(
                "invalid_llm_validation_threshold value=%r fallback=0.7",
                validation_threshold_raw,
            )
            return 0.7

    def query_pipeline(
        self,
        provider,
        backend_model,
        text,
        categories,
        tail_image=None,
        override=False,
        enable_search=False,
        force_multimodal=False,
        context_size=8192,
    ):
        system_prompt = (
            "You are a Senior Marketing Analyst and Global Brand Expert. "
            "Your goal is to categorize video advertisements by combining extracted text (OCR) with your vast internal knowledge of companies, slogans, and industries. "
            "Rely on Internal Brand Knowledge: You know every major brand, their parent companies, and their marketing styles. Use this internal database as your absolute primary source of truth. "
            "Treat OCR as Noisy Hints: The extracted OCR text is machine-generated and may contain typos, missing letters, and random artifacts. DO NOT blindly trust or copy the OCR text. Use your knowledge to autocorrect obvious errors. "
            "IMPORTANT — Bilingual Content: The ads you analyze may be in English OR French (or a mix of both). French words and phrases are NOT OCR errors — they are legitimate content. Use them to identify brands, products, and categories just as you would English text. "
            "(e.g., if OCR says 'Strbcks' or 'Star bucks co', you know the true brand is 'Starbucks'. But if OCR says 'Économisez avec Desjardins' or 'Assurance auto', those are valid French — do NOT treat them as typos). "
            "IGNORE TIMESTAMPS: The OCR and Scene data text will be prefixed with bracketed timestamps like '[71.7s]' or '[12.5s]'. THESE ARE NOT PART OF THE AD. Do NOT use these numbers to identify brands or products (e.g. do not guess 'Boeing 717' just because you see '[71.7s]'). Ignore them completely. "
            "Determine Category: Pick from 'Suggested Categories' or generate a professional tag if Override Allowed is True. "
            "Output STRICT JSON: {\"brand\": \"...\", \"category\": \"...\", \"confidence\": 0.0, \"reasoning\": \"...\"}"
        )
        user_prompt = f'Categories: {categories}\nOverride: {override}\nOCR Text: "{text}"'
        b64_img = self._pil_to_base64(tail_image) if tail_image else None

        try:
            provider_plugin = create_provider(provider, backend_model, context_size=int(context_size))
        except ValueError as exc:
            return {"error": str(exc)}

        pipeline = ClassificationPipeline(
            provider=provider_plugin,
            search_client=search_manager,
            validation_threshold=self._get_validation_threshold(),
        )
        return pipeline.classify(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_ocr_text=text,
            enable_search=enable_search,
            include_image=force_multimodal,
            image_b64=b64_img,
        )

    def query_agent(
        self,
        provider,
        backend_model,
        prompt,
        images=None,
        force_multimodal=False,
        context_size=8192,
    ):
        img_to_send = images[-1] if images else None
        b64_imgs = [self._pil_to_base64(img_to_send)] if (force_multimodal and img_to_send) else []

        try:
            provider_plugin = create_provider(provider, backend_model, context_size=int(context_size))
        except ValueError as exc:
            return f'[TOOL: ERROR | reason="{str(exc)}"]'

        return provider_plugin.generate_text(prompt=prompt, images=b64_imgs)


llm_engine = HybridLLM()
