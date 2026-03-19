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
from difflib import SequenceMatcher
from abc import ABC, abstractmethod
from typing import Optional, Protocol

import requests
from PIL import Image
from ddgs import DDGS

from video_service.core.logging_setup import (
    bind_current_log_context,
    capture_log_context,
    reset_log_fallback_context,
    set_log_fallback_context,
)
from video_service.core.utils import logger

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", "300"))
OPENAI_COMPAT_URL = os.environ.get("OPENAI_COMPAT_URL", "http://localhost:1234/v1/chat/completions")


def _classification_response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "brand": {"type": "string"},
            "category": {"type": "string"},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["brand", "category", "confidence", "reasoning"],
        "additionalProperties": True,
    }


def _category_index_response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "category_index": {"type": "integer"},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["category_index", "confidence", "reasoning"],
        "additionalProperties": True,
    }


def _family_index_response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "family_index": {"type": "integer"},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["family_index", "confidence", "reasoning"],
        "additionalProperties": True,
    }


def _entity_grounding_response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "entity_name": {"type": "string"},
            "entity_kind": {
                "type": "string",
                "enum": [
                    "film_release",
                    "stage_production",
                    "venue_exhibitor",
                    "tv_program",
                    "live_event",
                    "unknown",
                ],
            },
            "genres": {
                "type": "array",
                "items": {"type": "string"},
            },
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["entity_name", "entity_kind", "genres", "confidence", "reasoning"],
        "additionalProperties": True,
    }


class SearchManager:
    def __init__(self):
        self._ensure_ddgs_executor_context()
        self.queue = queue.Queue()
        self.client = DDGS()
        threading.Thread(target=self._worker, daemon=True).start()

    @staticmethod
    def _normalize_results(results) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for result in results or []:
            if not isinstance(result, dict):
                continue
            title = " ".join(str(result.get("title", "") or "").split()).strip()
            body = " ".join(str(result.get("body", "") or "").split()).strip()
            href = " ".join(
                str(
                    result.get("href")
                    or result.get("url")
                    or result.get("link")
                    or ""
                ).split()
            ).strip()
            if not any((title, body, href)):
                continue
            normalized.append({"title": title, "body": body, "href": href})
        return normalized

    @staticmethod
    def _ensure_ddgs_executor_context() -> None:
        if not hasattr(DDGS, "get_executor"):
            return

        original = getattr(DDGS, "get_executor")
        if getattr(original, "__name__", "") == "_context_get_executor":
            return

        class _ContextThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
            def submit(self, fn, /, *args, **kwargs):
                wrapped = bind_current_log_context(lambda: fn(*args, **kwargs))
                return super().submit(wrapped)

        def _shutdown_executor(executor) -> None:
            if executor is None or not hasattr(executor, "shutdown"):
                return
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            except Exception:
                pass

        def _context_get_executor(cls):
            executor = getattr(cls, "_executor", None)
            if not isinstance(executor, _ContextThreadPoolExecutor):
                _shutdown_executor(executor)
                cls._executor = _ContextThreadPoolExecutor(
                    max_workers=getattr(cls, "threads", None),
                    thread_name_prefix="DDGS",
                )
            return cls._executor

        DDGS.get_executor = classmethod(_context_get_executor)

    def _worker(self):
        while True:
            task, fallback_context, future = self.queue.get()
            max_retries = 3
            base_delay = 2.0
            success = False
            attempt = 0
            fallback_token = set_log_fallback_context(*fallback_context)

            try:
                while attempt < max_retries and not success:
                    try:
                        results = task()
                        future.set_result(results)
                        success = True
                    except Exception as exc:
                        attempt += 1
                        if attempt < max_retries:
                            backoff_sleep = (base_delay**attempt) + random.uniform(0.8, 2.5)
                            time.sleep(backoff_sleep)
                            self.client = DDGS()
                        else:
                            future.set_exception(exc)
            finally:
                reset_log_fallback_context(fallback_token)
                self.queue.task_done()

            if success:
                time.sleep(random.uniform(0.8, 2.5))

    def search_results(self, query, timeout=45, max_results=3):
        future = concurrent.futures.Future()
        fallback_context = capture_log_context()
        task = bind_current_log_context(lambda: list(self.client.text(query, max_results=max_results)))
        self.queue.put((task, fallback_context, future))
        try:
            results = future.result(timeout=timeout)
            return self._normalize_results(results)
        except Exception:
            return []

    def search(self, query, timeout=45):
        results = self.search_results(query, timeout=timeout, max_results=3)
        if not results:
            return None
        snippets = " | ".join(result.get("body", "") for result in results if result.get("body"))
        return snippets or None


search_manager = SearchManager()


def _is_valid_search_domain(domain: str) -> bool:
    labels = [label for label in (domain or "").strip().lower().split(".") if label]
    if len(labels) < 2:
        return False
    if labels[0] in {"www", "m", "amp"} and len(labels) < 3:
        return False
    tld = labels[-1]
    if not re.fullmatch(r"[a-z]{2,24}", tld):
        return False
    if len(labels) == 2 and len(tld) > 10:
        return False
    return True


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
        request_context_key: Optional[str] = None,
    ) -> None:
        super().__init__(backend_model=backend_model, context_size=context_size)
        self.force_json_mode = bool(force_json_mode)
        self.request_context_key = (request_context_key or "").strip() or None

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
        json_prefix = '{\n  "reasoning": "'
        response_schema = kwargs.get("response_schema") or _classification_response_schema()
        schema_name = str(kwargs.get("schema_name") or "ad_classification").strip() or "ad_classification"
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if images:
            content = [{"type": "text", "text": user_prompt}]
            content.extend(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                for image_b64 in images
                if image_b64
            )
            msgs[1]["content"] = content
        payload = {
            "model": self.backend_model,
            "messages": msgs,
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
        }
        if self.request_context_key:
            payload[self.request_context_key] = int(self.context_size)

        if self.force_json_mode:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": response_schema,
                },
            }
            logger.info(
                "JSON schema structured output active schema=%s required=%s",
                schema_name,
                response_schema.get("required"),
            )
        try:
            resp = requests.post(OPENAI_COMPAT_URL, json=payload, timeout=LLM_TIMEOUT_SECONDS)
            resp.raise_for_status()
            content_json = resp.json()
            raw_content = ""
            if "choices" in content_json and len(content_json["choices"]) > 0:
                raw_content = content_json["choices"][0].get("message", {}).get("content", "")
            content = raw_content if "{" in raw_content else f"{json_prefix}{raw_content}"
            logger.info(f"RAW LLAMA-SERVER OUTPUT: {content}")
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
            content = [{"type": "text", "text": prompt}]
            content.extend(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                for image_b64 in images
                if image_b64
            )
            msgs[1]["content"] = content
        payload = {"model": self.backend_model, "messages": msgs, "temperature": 0.1}
        if self.request_context_key:
            payload[self.request_context_key] = int(self.context_size)
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
            request_context_key="n_ctx",
        )
    if provider_name in {"lm studio", "openai compatible", "openai-compatible"}:
        return OpenAICompatibleProvider(
            backend_model=backend_model,
            context_size=context_size,
            force_json_mode=True,
        )
    raise ValueError(f"Unsupported provider: {provider}")


_BRAND_MEMORY_REASON_PATTERNS = (
    "famously associated",
    "typical of",
    "advertising style",
    "brand messaging",
    "core part of",
    "often emphasizes",
    "widely associated",
)
_GENERIC_SEARCH_STOPWORDS = {
    "official",
    "brand",
    "company",
    "product",
    "service",
    "services",
    "offer",
    "offers",
    "promotion",
    "promo",
}


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _normalize_brand_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _compact_brand_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _extract_domains(text: str) -> list[str]:
    return re.findall(r"\b([a-z0-9-]+\.[a-z]{2,}(?:/[a-z0-9/_-]+)?)\b", text or "", flags=re.IGNORECASE)


def _brand_tokens(brand: str) -> list[str]:
    return [token for token in _normalize_brand_text(brand).split() if len(token) >= 3]


def _ocr_tokens(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in re.findall(r"[A-Za-zÀ-ÿ0-9]{4,}", text or ""):
        normalized = token.lower()
        if normalized in _GENERIC_SEARCH_STOPWORDS or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(token)
    return ordered


def _has_exact_brand_anchor(text: str, brand: str) -> bool:
    normalized_text = set(_normalize_brand_text(text).split())
    brand_parts = _brand_tokens(brand)
    return bool(brand_parts and any(part in normalized_text for part in brand_parts))


def _has_domain_anchor(text: str) -> bool:
    return bool(_extract_domains(text))


def _has_market_cue(text: str) -> bool:
    normalized = _normalize_brand_text(text)
    if any(marker in normalized for marker in {" canada ", " canadian ", " australia ", " australian "}):
        return True
    for domain in _extract_domains(text):
        host = domain.split("/", 1)[0]
        tld = host.rsplit(".", 1)[-1]
        if len(tld) == 2:
            return True
    return False


def _ocr_is_sparse_or_slogan_like(text: str) -> tuple[bool, str]:
    tokens = _ocr_tokens(text)
    compact = _normalize_brand_text(text)
    if len(tokens) <= 6:
        return True, f"sparse_tokens={len(tokens)}"
    if len(compact) <= 80:
        return True, f"short_chars={len(compact)}"
    return False, "rich_ocr"


def _reasoning_looks_memory_led(reasoning: str) -> bool:
    normalized = (reasoning or "").lower()
    return any(pattern in normalized for pattern in _BRAND_MEMORY_REASON_PATTERNS)


def _brand_confirmed_by_web(brand: str, snippets: str) -> bool:
    normalized_snippets = set(_normalize_brand_text(snippets).split())
    brand_parts = _brand_tokens(brand)
    if not brand_parts:
        return False
    return any(part in normalized_snippets for part in brand_parts)


def _looks_like_ocr_brand_normalization(raw_ocr_text: str, brand: str) -> tuple[bool, str]:
    compact_brand = _compact_brand_text(brand)
    if len(compact_brand) < 4:
        return False, "brand_too_short"

    best_token = ""
    best_ratio = 0.0
    for token in _ocr_tokens(raw_ocr_text):
        compact_token = _compact_brand_text(token)
        if len(compact_token) < 4:
            continue
        ratio = SequenceMatcher(None, compact_brand, compact_token).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_token = token

    if not best_token:
        return False, "no_candidate_token"

    if abs(len(_compact_brand_text(best_token)) - len(compact_brand)) > 2:
        return False, f"length_gap token={best_token!r} ratio={best_ratio:.2f}"

    threshold = _env_float("BRAND_AMBIGUITY_OCR_NORMALIZATION_THRESHOLD", 0.84)
    if best_ratio >= threshold:
        return True, f"ocr_normalization token={best_token!r} ratio={best_ratio:.2f}"
    return False, f"ratio={best_ratio:.2f}<threshold={threshold:.2f}"


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

    def _brand_ambiguity_guard_enabled(self) -> bool:
        return _env_truthy("BRAND_AMBIGUITY_GUARD", default=True)

    def _brand_ambiguity_confidence_threshold(self) -> float:
        return _env_float("BRAND_AMBIGUITY_CONFIDENCE_THRESHOLD", 0.85)

    def _build_brand_disambiguation_query(self, raw_ocr_text: str, current_brand: str = "") -> str:
        query_parts: list[str] = []
        if current_brand:
            query_parts.append(f"\"{current_brand.strip()}\"")
        domains = _extract_domains(raw_ocr_text)
        if domains:
            first_domain = domains[0]
            host = first_domain.split("/", 1)[0]
            query_parts.extend([f"site:{host}", f"\"{host}\""])
        ocr_tokens = _ocr_tokens(raw_ocr_text)
        if ocr_tokens:
            query_parts.extend(f"\"{token}\"" if idx == 0 else token for idx, token in enumerate(ocr_tokens[:6]))
        query_parts.extend(["official", "brand", "slogan"])
        return " ".join(dict.fromkeys(part for part in query_parts if part))

    def _should_trigger_brand_ambiguity_guard(
        self,
        result_payload: dict,
        raw_ocr_text: str,
    ) -> tuple[bool, str]:
        if not self._brand_ambiguity_guard_enabled():
            return False, "disabled"

        brand = str(result_payload.get("brand", "") or "").strip()
        if brand.lower() in {"", "unknown", "none", "n/a"}:
            return False, "blank_brand"

        try:
            confidence = float(result_payload.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        threshold = self._brand_ambiguity_confidence_threshold()
        if confidence < threshold:
            return False, f"confidence={confidence:.2f}<threshold={threshold:.2f}"

        plausible_normalization, normalization_reason = _looks_like_ocr_brand_normalization(
            raw_ocr_text,
            brand,
        )
        if plausible_normalization:
            return False, normalization_reason

        if _has_exact_brand_anchor(raw_ocr_text, brand):
            return False, "exact_brand_anchor"
        if _has_domain_anchor(raw_ocr_text):
            return False, "domain_anchor"
        if _has_market_cue(raw_ocr_text):
            return False, "market_cue"

        sparse_ocr, sparse_reason = _ocr_is_sparse_or_slogan_like(raw_ocr_text)
        memory_led = _reasoning_looks_memory_led(str(result_payload.get("reasoning", "") or ""))
        if sparse_ocr or memory_led:
            reasons = [sparse_reason] if sparse_ocr else []
            if memory_led:
                reasons.append("memory_led_reasoning")
            return True, ";".join(reasons) or "weak_anchor"
        return False, "evidence_rich"

    def _attempt_brand_disambiguation(
        self,
        system_prompt: str,
        current_result: dict,
        raw_ocr_text: str,
        images: Optional[list[str]] = None,
    ) -> tuple[dict | None, str]:
        current_brand = str(current_result.get("brand", "") or "").strip()
        query = self._build_brand_disambiguation_query(raw_ocr_text, current_brand=current_brand)
        if not query:
            return None, "no_search_query"

        logger.info("brand_disambiguation_triggered query=%r current_brand=%r", query, current_brand)
        snippets = self.search_client.search(query)
        if not snippets:
            return None, "search_unavailable"

        val_res = self.provider.generate_json(
            system_prompt + "\nBRAND DISAMBIGUATION MODE",
            (
                f"OCR: {raw_ocr_text}\n"
                f"Current Brand Guess: {current_brand}\n"
                f"Current Category: {current_result.get('category', '')}\n"
                f"Web Evidence: {snippets}\n"
                "Attached images are chronological recent frames from the ad. Direct on-frame logos, branded product text, and explicit product naming outweigh loose slogan-only or token-only web matches. "
                "Confirm or correct the brand. Slogan-only matches are low-trust unless corroborated by exact brand text, a domain, market/country cues, or explicit web evidence. "
                "Keep the category unless a corrected brand clearly changes it."
            ),
            images=images,
        )
        if not isinstance(val_res, dict) or "brand" not in val_res:
            return None, "invalid_llm_response"

        resolved_brand = str(val_res.get("brand", "") or "").strip()
        if resolved_brand.lower() in {"", "unknown", "none", "n/a"}:
            return None, "blank_brand_response"

        current_confirmed = _brand_confirmed_by_web(current_brand, snippets)
        resolved_confirmed = _brand_confirmed_by_web(resolved_brand, snippets)
        plausible_normalization, normalization_reason = _looks_like_ocr_brand_normalization(
            raw_ocr_text,
            current_brand,
        )

        if resolved_brand.lower() == current_brand.lower():
            if not current_confirmed and not plausible_normalization:
                return None, f"web_unconfirmed_brand={resolved_brand!r}"
            val_res["reasoning"] = "(Brand confirmed) " + str(val_res.get("reasoning", "") or "")
            return val_res, "brand_confirmed_by_web" if current_confirmed else normalization_reason

        if plausible_normalization and not current_confirmed:
            return None, f"kept_plausible_ocr_normalization {normalization_reason}"

        if current_confirmed and resolved_confirmed:
            return None, f"ambiguous_web_brand_support current={current_brand!r} candidate={resolved_brand!r}"

        if not resolved_confirmed:
            return None, f"web_unconfirmed_brand={resolved_brand!r}"

        val_res["reasoning"] = "(Brand corrected) " + str(val_res.get("reasoning", "") or "")
        return val_res, "brand_corrected_by_web"

    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        raw_ocr_text: str,
        enable_search: bool,
        include_image: bool,
        image_b64: Optional[str],
        express_mode: bool = False,
    ) -> dict:
        images: Optional[list[str]] = None
        if include_image and self.provider.supports_vision and image_b64:
            if isinstance(image_b64, list):
                images = [value for value in image_b64 if value]
            elif isinstance(image_b64, str):
                images = [image_b64]
        res = self.provider.generate_json(system_prompt, user_prompt, images=images)
        logger.debug("llm_pipeline_initial_result: %s", res)
        if express_mode:
            return res if isinstance(res, dict) else {"error": "Unexpected LLM response"}

        if isinstance(res, dict):
            guard_triggered, guard_reason = self._should_trigger_brand_ambiguity_guard(res, raw_ocr_text)
            if guard_triggered:
                logger.info("brand_ambiguity_guard_triggered: %s brand=%r", guard_reason, res.get("brand", ""))
                res["brand_ambiguity_flag"] = True
                res["brand_ambiguity_reason"] = guard_reason
                res["brand_evidence_strength"] = "weak_anchor"
                if enable_search:
                    refined, refine_reason = self._attempt_brand_disambiguation(
                        system_prompt,
                        res,
                        raw_ocr_text,
                        images=images,
                    )
                    if refined is not None:
                        refined["brand_ambiguity_flag"] = True
                        refined["brand_ambiguity_reason"] = guard_reason
                        refined["brand_ambiguity_resolved"] = True
                        refined["brand_disambiguation_reason"] = refine_reason
                        logger.info(
                            "brand_disambiguation_accepted: %s old_brand=%r new_brand=%r",
                            refine_reason,
                            res.get("brand", ""),
                            refined.get("brand", ""),
                        )
                        res = refined
                    else:
                        res["brand_ambiguity_resolved"] = False
                        res["brand_disambiguation_reason"] = refine_reason
                        logger.info("brand_disambiguation_rejected: %s", refine_reason)
                else:
                    res["brand_ambiguity_resolved"] = False
                    res["brand_disambiguation_reason"] = "search_disabled"

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

    def _extract_search_domain(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        matches = re.findall(
            r"\b((?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[a-z0-9/_-]+)?)\b",
            text,
            flags=re.IGNORECASE,
        )
        for raw in matches:
            domain = raw.split("/", 1)[0].lower()
            if not _is_valid_search_domain(domain):
                continue
            path = raw.split("/", 1)[1] if "/" in raw else ""
            return domain, path
        return "", ""

    def _build_specificity_search_query(
        self,
        brand: str,
        current_category: str,
        ocr_text: str,
    ) -> str:
        domain, path = self._extract_search_domain(ocr_text)
        query_parts: list[str] = []
        brand_clean = (brand or "").strip()
        if domain:
            query_parts.append(f"site:{domain}")
        if brand_clean:
            query_parts.append(f"\"{brand_clean}\"")
        if domain:
            query_parts.append(f"\"{domain}\"")
        if path:
            path_tokens = [token for token in re.split(r"[/_-]+", path) if len(token) >= 3]
            if path_tokens:
                query_parts.extend(path_tokens[:3])
        ocr_tokens = [
            token
            for token in re.findall(r"[A-Za-zÀ-ÿ0-9]{4,}", ocr_text or "")
            if token.lower() not in {"brand", "offer", "official", "service", "services"}
        ]
        if ocr_tokens:
            query_parts.extend(ocr_tokens[:4])
        if current_category:
            query_parts.append(current_category)
        query_parts.append("official site")
        return " ".join(dict.fromkeys(part for part in query_parts if part))

    def _build_entity_search_query(
        self,
        brand: str,
        raw_category: str,
        ocr_text: str,
    ) -> str:
        domain, path = self._extract_search_domain(ocr_text)
        query_parts: list[str] = []
        brand_clean = (brand or "").strip()
        if domain:
            query_parts.append(f"site:{domain}")
        if brand_clean:
            query_parts.append(f"\"{brand_clean}\"")
        if domain:
            query_parts.append(f"\"{domain}\"")
        if path:
            path_tokens = [token for token in re.split(r"[/_-]+", path) if len(token) >= 3]
            if path_tokens:
                query_parts.extend(path_tokens[:3])
        ocr_tokens = [
            token
            for token in _ocr_tokens(ocr_text)
            if token.lower() not in {"online", "platform", "service", "services", "official"}
        ]
        if ocr_tokens:
            query_parts.extend(ocr_tokens[:5])
        raw_clean = str(raw_category or "").strip()
        if raw_clean and len(raw_clean.split()) <= 3:
            query_parts.append(raw_clean)
        query_parts.append("official")
        return " ".join(dict.fromkeys(part for part in query_parts if part))

    @staticmethod
    def _normalize_entity_kind(value: object) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {
            "film_release",
            "stage_production",
            "venue_exhibitor",
            "tv_program",
            "live_event",
            "unknown",
        }:
            return normalized
        return "unknown"

    @staticmethod
    def _normalize_entity_genres(value: object) -> list[str]:
        raw_values: list[str] = []
        if isinstance(value, (list, tuple)):
            raw_values = [str(item or "").strip() for item in value]
        else:
            text = str(value or "").strip()
            if text:
                raw_values = re.split(r"[,/|;]\s*", text)
        genres: list[str] = []
        for raw in raw_values:
            clean = " ".join(str(raw or "").strip().lower().split())
            if clean and clean not in genres:
                genres.append(clean)
        return genres[:5]

    @staticmethod
    def _format_search_results_for_prompt(results: list[dict[str, str]]) -> str:
        lines: list[str] = []
        for idx, result in enumerate(results[:5], start=1):
            title = str(result.get("title", "") or "").strip()
            body = str(result.get("body", "") or "").strip()
            href = str(result.get("href", "") or "").strip()
            lines.append(f"{idx}. Title: {title or 'None'}")
            if href:
                lines.append(f"   URL: {href}")
            if body:
                lines.append(f"   Snippet: {body}")
        return "\n".join(lines).strip()

    @staticmethod
    def _summarize_search_results_for_log(results: list[dict[str, str]]) -> list[str]:
        summarized: list[str] = []
        for result in results[:5]:
            title = str(result.get("title", "") or "").strip()
            href = str(result.get("href", "") or "").strip()
            body = str(result.get("body", "") or "").strip()
            entry = " | ".join(part for part in (title, href, body[:120]) if part)
            if entry:
                summarized.append(entry)
        return summarized

    def _build_product_focus_guidance(
        self,
        *,
        raw_category: str = "",
        mapped_category: str = "",
        ocr_text: str = "",
        reasoning: str = "",
        candidate_categories: list[str] | None = None,
    ) -> str:
        evidence_text = " ".join(
            part for part in (raw_category, mapped_category, ocr_text, reasoning) if part
        ).lower()
        service_evidence_text = " ".join(
            part for part in (ocr_text, reasoning) if part
        ).lower()
        if not evidence_text:
            return ""

        device_cues = {
            "iphone",
            "galaxy",
            "pixel",
            "smartphone",
            "handset",
            "camera",
            "compared to",
            "apple",
            "device",
            "phone",
        }
        service_cues = {
            "5g",
            "activation",
            "carrier",
            "coverage",
            "data",
            "monthly",
            "network",
            "plan",
            "plans",
            "provider",
            "roaming",
            "service",
            "services",
            "unlimited",
            "wireless",
        }
        provider_terms = {
            "provider",
            "providers",
            "service",
            "services",
            "telecommunication",
            "telecommunications",
            "wireless",
            "internet",
            "carrier",
        }
        device_candidate_terms = {
            "device",
            "devices",
            "handset",
            "mobile",
            "phone",
            "phones",
            "smartphone",
            "smartphones",
        }

        candidates = [str(candidate or "").strip() for candidate in (candidate_categories or []) if str(candidate or "").strip()]
        provider_candidate_present = any(
            any(term in candidate.casefold() for term in provider_terms)
            for candidate in candidates
        ) or any(term in evidence_text for term in provider_terms)
        device_candidate_present = any(
            any(term in candidate.casefold() for term in device_candidate_terms)
            for candidate in candidates
        )

        device_signal_count = sum(1 for cue in device_cues if cue in evidence_text)
        service_signal_count = sum(1 for cue in service_cues if cue in service_evidence_text)
        if provider_candidate_present and device_signal_count >= 2 and device_candidate_present and service_signal_count <= 2:
            return (
                "If the ad centers on a named phone/device model or visible handset comparison, "
                "choose the device/product category rather than the carrier/provider category "
                "unless the evidence is mainly about plans, network coverage, pricing, or service benefits."
            )
        return ""

    def query_category_rerank(
        self,
        provider,
        backend_model,
        brand: str,
        raw_category: str,
        mapped_category: str,
        ocr_text: str,
        reasoning: str,
        candidate_categories: list[str],
        candidate_category_contexts: dict[str, str] | None = None,
        visual_matches: list[tuple[str, float]] | None = None,
        context_size=8192,
        product_focus_guidance_enabled: bool = True,
    ) -> tuple[dict | None, str]:
        try:
            provider_plugin = create_provider(provider, backend_model, context_size=int(context_size))
        except ValueError as exc:
            return None, f"provider_error:{exc}"

        normalized_candidates = {
            str(candidate or "").strip().casefold(): str(candidate or "").strip()
            for candidate in candidate_categories
            if str(candidate or "").strip()
        }
        if len(normalized_candidates) < 2:
            return None, "insufficient_candidates"

        visual_hint_text = ""
        if visual_matches:
            visual_hint_text = " | ".join(
                f"{label} ({score:.4f})" for label, score in visual_matches[:4]
            )

        ocr_excerpt = " ".join((ocr_text or "").split())[:500]
        reasoning_excerpt = " ".join((reasoning or "").split())[:400]
        candidate_lines: list[str] = []
        for idx, candidate in enumerate(normalized_candidates.values(), start=1):
            candidate_lines.append(f"{idx}. {candidate}")
            path_text = ""
            if candidate_category_contexts:
                path_text = str(candidate_category_contexts.get(candidate, "") or "").strip()
            if path_text and path_text != candidate:
                candidate_lines.append(f"   Parent/Leaf Path: {path_text}")
        candidate_text = "\n".join(candidate_lines)
        product_focus_guidance = ""
        if product_focus_guidance_enabled:
            product_focus_guidance = self._build_product_focus_guidance(
                raw_category=raw_category,
                mapped_category=mapped_category,
                ocr_text=ocr_text,
                reasoning=reasoning,
                candidate_categories=list(normalized_candidates.values()),
            )
        system_prompt = (
            "You are resolving a taxonomy mapping ambiguity for a video ad classification system. "
            "Choose exactly one category from the supplied candidate categories only. "
            "Return the 1-based category_index for the chosen candidate. "
            "The category_index is authoritative. "
            "Use the brand, OCR evidence, prior category guess, model reasoning, and optional visual hints together. "
            "Prefer the category that best matches the ad's overall product or service domain, not a superficial keyword overlap. "
            + (
                "The advertiser or provider brand does not automatically determine category. "
                "If a seller, carrier, retailer, or provider is promoting a specific product model, classify the promoted product family unless the ad is primarily about the service/store/network offering itself. "
                if product_focus_guidance_enabled
                else ""
            )
            + (
            "Do not invent a label. Do not choose a candidate whose domain contradicts the overall ad evidence. "
            "If the evidence is mixed, choose the safer in-family category from the list. "
            "Output STRICT JSON: {\"category_index\": 0, \"confidence\": 0.0, \"reasoning\": \"...\"}"
            )
        )
        user_prompt = (
            f"Brand: {brand}\n"
            f"LLM Category Guess: {raw_category}\n"
            f"Current Mapped Category: {mapped_category}\n"
            f"Candidate Categories:\n{candidate_text}\n"
            f"OCR Evidence: {ocr_excerpt or 'None'}\n"
            f"Prior Reasoning: {reasoning_excerpt or 'None'}\n"
            f"Visual Hints: {visual_hint_text or 'None'}\n"
            f"Decision Guidance: {product_focus_guidance or 'None'}\n"
            "Return the single best-supported category from the candidate list."
        )
        ordered_candidates = list(normalized_candidates.values())

        def _resolve_index(payload: dict | None) -> tuple[str | None, int]:
            if not isinstance(payload, dict):
                return None, 0
            chosen_index = payload.get("category_index")
            try:
                candidate_index = int(chosen_index)
            except (TypeError, ValueError):
                return None, 0
            if 1 <= candidate_index <= len(ordered_candidates):
                return ordered_candidates[candidate_index - 1], candidate_index
            return None, 0

        def _resolve_exact_candidate_text(payload: dict | None) -> tuple[str | None, int]:
            if not isinstance(payload, dict):
                return None, 0
            chosen_text = str(payload.get("category", "") or "").strip()
            if not chosen_text:
                return None, 0
            canonical_candidate = normalized_candidates.get(chosen_text.casefold())
            if canonical_candidate is None:
                return None, 0
            for idx, candidate in enumerate(ordered_candidates, start=1):
                if candidate == canonical_candidate:
                    return canonical_candidate, idx
            return None, 0

        original_category_text = ""
        try:
            res = provider_plugin.generate_json(
                system_prompt,
                user_prompt,
                response_schema=_category_index_response_schema(),
                schema_name="category_rerank",
            )
        except Exception as exc:
            return None, f"provider_error:{exc}"
        if not isinstance(res, dict):
            return None, "invalid_llm_response"
        original_category_text = str(res.get("category", "") or "").strip()

        canonical_candidate, resolved_index = _resolve_index(res)
        resolution_source = f"category_index:{resolved_index}" if resolved_index else ""
        if canonical_candidate is None:
            logger.info(
                "category_rerank_retry_triggered reason=missing_category_index category_text=%r category_index=%r",
                str(res.get("category", "") or "").strip(),
                res.get("category_index"),
            )
            retry_system_prompt = (
                "You must select exactly one category from the numbered candidate list. "
                "Return STRICT JSON only as {\"category_index\": 0, \"confidence\": 0.0, \"reasoning\": \"...\"}. "
                "The category_index is required and must be the 1-based number of the chosen candidate. "
                "Do not return category text. Do not explain alternatives outside the list."
            )
            retry_user_prompt = (
                f"Candidate Categories:\n{candidate_text}\n"
                f"OCR Evidence: {ocr_excerpt or 'None'}\n"
                f"Prior Reasoning: {reasoning_excerpt or 'None'}\n"
                f"Visual Hints: {visual_hint_text or 'None'}\n"
                "Return only the single best category_index from the candidate list."
            )
            try:
                retry_res = provider_plugin.generate_json(
                    retry_system_prompt,
                    retry_user_prompt,
                    response_schema=_category_index_response_schema(),
                    schema_name="category_rerank_retry",
                )
            except Exception as exc:
                return None, f"provider_error:{exc}"
            if not isinstance(retry_res, dict):
                return None, "invalid_llm_response"
            original_category_text = str(retry_res.get("category", "") or "").strip()
            canonical_candidate, resolved_index = _resolve_index(retry_res)
            if canonical_candidate is None:
                canonical_candidate, resolved_index = _resolve_exact_candidate_text(retry_res)
                if canonical_candidate is None:
                    return None, "missing_category_index"
                resolution_source = f"retry_category_text_exact:{resolved_index}"
            else:
                resolution_source = f"retry_category_index:{resolved_index}"
            res = retry_res
        res["brand"] = brand
        res["category"] = canonical_candidate
        res["category_index"] = resolved_index
        logger.info(
            "category_rerank_resolved source=%s chosen=%r category_text=%r category_index=%r",
            resolution_source or "unknown",
            canonical_candidate,
            original_category_text,
            res.get("category_index"),
        )
        return res, "ok"

    def query_category_family_selection(
        self,
        provider,
        backend_model,
        brand: str,
        raw_category: str,
        mapped_category: str,
        ocr_text: str,
        reasoning: str,
        candidate_families: list[str] | None = None,
        candidate_family_contexts: dict[str, str] | None = None,
        family_members: dict[str, list[str]] | None = None,
        visual_matches: list[tuple[str, float]] | None = None,
        context_size=8192,
    ) -> tuple[dict | None, str]:
        try:
            provider_plugin = create_provider(provider, backend_model, context_size=int(context_size))
        except ValueError as exc:
            return None, f"provider_error:{exc}"

        normalized_families = {
            str(candidate or "").strip().casefold(): str(candidate or "").strip()
            for candidate in (candidate_families or [])
            if str(candidate or "").strip()
        }
        if len(normalized_families) < 2:
            return None, "insufficient_families"

        visual_hint_text = ""
        if visual_matches:
            visual_hint_text = " | ".join(
                f"{label} ({score:.4f})" for label, score in visual_matches[:4]
            )

        ordered_families = list(normalized_families.values())
        family_lines: list[str] = []
        for idx, family in enumerate(ordered_families, start=1):
            family_lines.append(f"{idx}. {family}")
            path_text = ""
            if candidate_family_contexts:
                path_text = str(candidate_family_contexts.get(family, "") or "").strip()
            if path_text and path_text != family:
                family_lines.append(f"   Family Path: {path_text}")
            members = [str(member or "").strip() for member in (family_members or {}).get(family, []) if str(member or "").strip()]
            if members:
                family_lines.append(f"   Candidate Labels: {', '.join(members[:6])}")
        family_text = "\n".join(family_lines)
        ocr_excerpt = " ".join((ocr_text or "").split())[:500]
        reasoning_excerpt = " ".join((reasoning or "").split())[:500]

        system_prompt = (
            "You are selecting the taxonomy family or branch before the final leaf category is chosen. "
            "Choose exactly one family from the supplied numbered family list only. "
            "Return the 1-based family_index for the chosen family. "
            "The family_index is authoritative. "
            "Prefer the family that best represents what is being advertised overall: the product, service, content, retailer, institution, or venue actually promoted. "
            "Do not choose an incidental device, channel, or medium unless the ad is primarily about that device, channel, or medium. "
            "Output STRICT JSON: {\"family_index\": 0, \"confidence\": 0.0, \"reasoning\": \"...\"}"
        )
        user_prompt = (
            f"Brand: {brand}\n"
            f"LLM Category Guess: {raw_category}\n"
            f"Current Mapped Category: {mapped_category}\n"
            f"Candidate Families:\n{family_text}\n"
            f"OCR Evidence: {ocr_excerpt or 'None'}\n"
            f"Prior Reasoning: {reasoning_excerpt or 'None'}\n"
            f"Visual Hints: {visual_hint_text or 'None'}\n"
            "Return the single best-supported family_index from the family list."
        )

        def _resolve_index(payload: dict | None) -> tuple[str | None, int]:
            if not isinstance(payload, dict):
                return None, 0
            chosen_index = payload.get("family_index")
            try:
                family_index = int(chosen_index)
            except (TypeError, ValueError):
                return None, 0
            if 1 <= family_index <= len(ordered_families):
                return ordered_families[family_index - 1], family_index
            return None, 0

        def _resolve_exact_family_text(payload: dict | None) -> tuple[str | None, int]:
            if not isinstance(payload, dict):
                return None, 0
            chosen_text = str(payload.get("family", "") or payload.get("category", "") or "").strip()
            if not chosen_text:
                return None, 0
            canonical_family = normalized_families.get(chosen_text.casefold())
            if canonical_family is None:
                return None, 0
            for idx, family in enumerate(ordered_families, start=1):
                if family == canonical_family:
                    return canonical_family, idx
            return None, 0

        try:
            res = provider_plugin.generate_json(
                system_prompt,
                user_prompt,
                response_schema=_family_index_response_schema(),
                schema_name="category_family_selection",
            )
        except Exception as exc:
            return None, f"provider_error:{exc}"
        if not isinstance(res, dict):
            return None, "invalid_llm_response"

        original_family_text = str(res.get("family", "") or res.get("category", "") or "").strip()
        canonical_family, resolved_index = _resolve_index(res)
        resolution_source = f"family_index:{resolved_index}" if resolved_index else ""
        if canonical_family is None:
            logger.info(
                "category_family_selection_retry_triggered reason=missing_family_index family_text=%r family_index=%r",
                original_family_text,
                res.get("family_index"),
            )
            retry_system_prompt = (
                "You must select exactly one family from the numbered family list. "
                "Return STRICT JSON only as {\"family_index\": 0, \"confidence\": 0.0, \"reasoning\": \"...\"}. "
                "The family_index is required and must be the 1-based number of the chosen family. "
                "Do not return family text. Do not explain alternatives outside the list."
            )
            retry_user_prompt = (
                f"Candidate Families:\n{family_text}\n"
                f"OCR Evidence: {ocr_excerpt or 'None'}\n"
                f"Prior Reasoning: {reasoning_excerpt or 'None'}\n"
                f"Visual Hints: {visual_hint_text or 'None'}\n"
                "Return only the single best family_index from the family list."
            )
            try:
                retry_res = provider_plugin.generate_json(
                    retry_system_prompt,
                    retry_user_prompt,
                    response_schema=_family_index_response_schema(),
                    schema_name="category_family_selection_retry",
                )
            except Exception as exc:
                return None, f"provider_error:{exc}"
            if not isinstance(retry_res, dict):
                return None, "invalid_llm_response"
            original_family_text = str(retry_res.get("family", "") or retry_res.get("category", "") or "").strip()
            canonical_family, resolved_index = _resolve_index(retry_res)
            if canonical_family is None:
                canonical_family, resolved_index = _resolve_exact_family_text(retry_res)
                if canonical_family is None:
                    return None, "missing_family_index"
                resolution_source = f"retry_family_text_exact:{resolved_index}"
            else:
                resolution_source = f"retry_family_index:{resolved_index}"
            res = retry_res

        res["family"] = canonical_family
        res["family_index"] = resolved_index
        logger.info(
            "category_family_selection_resolved source=%s family=%r family_text=%r family_index=%r",
            resolution_source or "unknown",
            canonical_family,
            original_family_text,
            res.get("family_index"),
        )
        return res, "ok"

    def query_specificity_rescue(
        self,
        provider,
        backend_model,
        brand: str,
        current_category: str,
        ocr_text: str,
        candidate_categories: list[str] | None = None,
        candidate_category_contexts: dict[str, str] | None = None,
        visual_matches: list[tuple[str, float]] | None = None,
        context_size=8192,
    ) -> tuple[dict | None, str]:
        try:
            provider_plugin = create_provider(provider, backend_model, context_size=int(context_size))
        except ValueError as exc:
            return None, f"provider_error:{exc}"

        query = self._build_specificity_search_query(
            brand=brand,
            current_category=current_category,
            ocr_text=ocr_text,
        )
        if not query.strip():
            return None, "no_search_query"

        snippets = search_manager.search(query)
        if not snippets:
            return None, "search_unavailable"

        visual_hint_text = ""
        if visual_matches:
            visual_hint_text = " | ".join(
                f"{label} ({score:.4f})" for label, score in visual_matches[:4]
            )
        candidate_text = ""
        if candidate_categories:
            candidate_lines: list[str] = []
            for idx, candidate in enumerate(candidate_categories, start=1):
                candidate_lines.append(f"{idx}. {candidate}")
                path_text = ""
                if candidate_category_contexts:
                    path_text = str(candidate_category_contexts.get(candidate, "") or "").strip()
                if path_text and path_text != candidate:
                    candidate_lines.append(f"   Parent/Leaf Path: {path_text}")
            candidate_text = "\n".join(candidate_lines)

        system_prompt = (
            "You are refining a broad video-ad category into the most specific supported product or service category. "
            "Use the existing brand, OCR, and web search evidence to narrow a broad parent category only when the evidence clearly supports a more specific leaf category. "
            "You must choose the best-supported category from the supplied candidate categories. "
            "Do not broaden the category. Do not invent a subtype that is not supported. "
            "If the evidence only supports the current broad category, keep it. "
            "Output STRICT JSON: {\"brand\": \"...\", \"category\": \"...\", \"confidence\": 0.0, \"reasoning\": \"...\"}"
        )
        user_prompt = (
            f"Brand: {brand}\n"
            f"Current Category: {current_category}\n"
            f"Candidate Categories:\n{candidate_text or current_category}\n"
            f"OCR Evidence: {ocr_text}\n"
            f"Visual Hints: {visual_hint_text or 'None'}\n"
            f"Web Search Snippets: {snippets}\n"
            "Return the most specific supported product or service category from the candidate list."
        )
        try:
            res = provider_plugin.generate_json(
                system_prompt,
                user_prompt,
            )
        except Exception as exc:
            return None, f"provider_error:{exc}"
        if not isinstance(res, dict) or "category" not in res:
            return None, "invalid_llm_response"
        res["brand"] = brand
        return res, "ok"

    def query_entity_grounding(
        self,
        provider,
        backend_model,
        brand: str,
        raw_category: str,
        ocr_text: str,
        visual_matches: list[tuple[str, float]] | None = None,
        context_size=8192,
    ) -> tuple[dict | None, str]:
        try:
            provider_plugin = create_provider(provider, backend_model, context_size=int(context_size))
        except ValueError as exc:
            return None, f"provider_error:{exc}"

        query = self._build_entity_search_query(
            brand=brand,
            raw_category=raw_category,
            ocr_text=ocr_text,
        )
        if not query.strip():
            return None, "no_search_query"

        search_results = search_manager.search_results(query, max_results=5)
        logger.info(
            "entity_grounding_query query=%r results=%r",
            query,
            self._summarize_search_results_for_log(search_results),
        )
        if not search_results:
            return None, "search_unavailable"

        visual_hint_text = ""
        if visual_matches:
            visual_hint_text = " | ".join(
                f"{label} ({score:.4f})" for label, score in visual_matches[:4]
            )

        search_text = self._format_search_results_for_prompt(search_results)
        ocr_excerpt = " ".join((ocr_text or "").split())[:500]
        raw_excerpt = " ".join((raw_category or "").split())[:120]

        system_prompt = (
            "You are grounding the real-world entity being advertised before taxonomy mapping. "
            "Use OCR evidence, visual hints, and web search results to identify the advertised entity independent of the taxonomy. "
            "Return the entity_name, entity_kind, genres, confidence, and reasoning. "
            "entity_kind must be one of: film_release, stage_production, venue_exhibitor, tv_program, live_event, unknown. "
            "genres should be a short list of broad genres like comedy, drama, action, thriller, horror, documentary, musical, family, or sports when supported. "
            "Do not choose a taxonomy label. "
            "Output STRICT JSON: {\"entity_name\": \"...\", \"entity_kind\": \"unknown\", \"genres\": [\"...\"], \"confidence\": 0.0, \"reasoning\": \"...\"}"
        )
        user_prompt = (
            f"Brand/Title Guess: {brand}\n"
            f"Raw Category Guess: {raw_excerpt or 'None'}\n"
            f"OCR Evidence: {ocr_excerpt or 'None'}\n"
            f"Visual Hints: {visual_hint_text or 'None'}\n"
            f"Web Search Results:\n{search_text}\n"
            "Return the grounded entity information only."
        )

        try:
            res = provider_plugin.generate_json(
                system_prompt,
                user_prompt,
                response_schema=_entity_grounding_response_schema(),
                schema_name="entity_grounding",
            )
        except Exception as exc:
            return None, f"provider_error:{exc}"
        if not isinstance(res, dict):
            return None, "invalid_llm_response"

        entity_name = (
            str(res.get("entity_name", "") or "").strip()
            or str(res.get("entity_title", "") or "").strip()
            or str(brand or "").strip()
        )
        entity_kind = self._normalize_entity_kind(res.get("entity_kind"))
        genres = self._normalize_entity_genres(res.get("genres"))
        res["entity_name"] = entity_name
        res["entity_kind"] = entity_kind
        res["genres"] = genres
        res["_search_query"] = query
        res["_search_results"] = search_results
        logger.info(
            "entity_grounding_resolved entity_name=%r entity_kind=%r genres=%r confidence=%r",
            entity_name,
            entity_kind,
            genres,
            res.get("confidence"),
        )
        return res, "ok"

    def query_entity_search_rescue(
        self,
        provider,
        backend_model,
        brand: str,
        raw_category: str,
        current_category: str,
        entity_name: str,
        entity_kind: str,
        ocr_text: str,
        genres: list[str] | None = None,
        branch_label: str = "",
        search_results: list[dict[str, str]] | None = None,
        candidate_categories: list[str] | None = None,
        candidate_category_contexts: dict[str, str] | None = None,
        visual_matches: list[tuple[str, float]] | None = None,
        context_size=8192,
    ) -> tuple[dict | None, str]:
        try:
            provider_plugin = create_provider(provider, backend_model, context_size=int(context_size))
        except ValueError as exc:
            return None, f"provider_error:{exc}"

        normalized_candidates = {
            str(candidate or "").strip().casefold(): str(candidate or "").strip()
            for candidate in (candidate_categories or [])
            if str(candidate or "").strip()
        }
        if not normalized_candidates:
            return None, "insufficient_candidates"

        normalized_search_results = list(search_results or [])
        logger.info(
            "entity_branch_selection entity_name=%r entity_kind=%r branch=%r genres=%r candidates=%r",
            entity_name,
            entity_kind,
            branch_label,
            list(genres or []),
            list(normalized_candidates.values())[:8],
        )
        if not normalized_search_results:
            return None, "search_unavailable"

        visual_hint_text = ""
        if visual_matches:
            visual_hint_text = " | ".join(
                f"{label} ({score:.4f})" for label, score in visual_matches[:4]
            )

        ordered_candidates = list(normalized_candidates.values())
        candidate_lines: list[str] = []
        for idx, candidate in enumerate(ordered_candidates, start=1):
            candidate_lines.append(f"{idx}. {candidate}")
            path_text = ""
            if candidate_category_contexts:
                path_text = str(candidate_category_contexts.get(candidate, "") or "").strip()
            if path_text and path_text != candidate:
                candidate_lines.append(f"   Parent/Leaf Path: {path_text}")
        candidate_text = "\n".join(candidate_lines)
        search_text = self._format_search_results_for_prompt(normalized_search_results)
        ocr_excerpt = " ".join((ocr_text or "").split())[:500]
        genre_text = ", ".join(genres or []) or "None"

        system_prompt = (
            "You are selecting a taxonomy category after the advertised entity has already been grounded. "
            "The entity grounding is authoritative context. "
            "Choose exactly one category from the supplied branch-constrained candidate categories only. "
            "Return the 1-based category_index for the chosen candidate. "
            "The category_index is authoritative. "
            "Do not invent a label. Do not choose a candidate outside the list. "
            "If the evidence supports the branch but not a specific leaf, choose the broader branch parent candidate instead of forcing the wrong leaf. "
            "Output STRICT JSON: {\"category_index\": 0, \"confidence\": 0.0, \"reasoning\": \"...\"}"
        )
        user_prompt = (
            f"Brand/Title Guess: {brand}\n"
            f"Raw Category Guess: {raw_category}\n"
            f"Current Category: {current_category}\n"
            f"Grounded Entity Name: {entity_name}\n"
            f"Grounded Entity Kind: {entity_kind}\n"
            f"Grounded Genres: {genre_text}\n"
            f"Selected Taxonomy Branch: {branch_label or 'None'}\n"
            f"Candidate Categories:\n{candidate_text}\n"
            f"OCR Evidence: {ocr_excerpt or 'None'}\n"
            f"Visual Hints: {visual_hint_text or 'None'}\n"
            f"Web Search Results:\n{search_text}\n"
            "Return the single best-supported category_index from the candidate list."
        )

        def _resolve_index(payload: dict | None) -> tuple[str | None, int]:
            if not isinstance(payload, dict):
                return None, 0
            chosen_index = payload.get("category_index")
            try:
                candidate_index = int(chosen_index)
            except (TypeError, ValueError):
                return None, 0
            if 1 <= candidate_index <= len(ordered_candidates):
                return ordered_candidates[candidate_index - 1], candidate_index
            return None, 0

        def _resolve_exact_candidate_text(payload: dict | None) -> tuple[str | None, int]:
            if not isinstance(payload, dict):
                return None, 0
            chosen_text = str(payload.get("category", "") or "").strip()
            if not chosen_text:
                return None, 0
            if chosen_text.isdigit():
                candidate_index = int(chosen_text)
                if 1 <= candidate_index <= len(ordered_candidates):
                    return ordered_candidates[candidate_index - 1], candidate_index
                return None, 0
            canonical_candidate = normalized_candidates.get(chosen_text.casefold())
            if canonical_candidate is None:
                return None, 0
            for idx, candidate in enumerate(ordered_candidates, start=1):
                if candidate == canonical_candidate:
                    return canonical_candidate, idx
            return None, 0

        try:
            res = provider_plugin.generate_json(
                system_prompt,
                user_prompt,
                response_schema=_category_index_response_schema(),
                schema_name="entity_branch_selection",
            )
        except Exception as exc:
            return None, f"provider_error:{exc}"
        if not isinstance(res, dict):
            return None, "invalid_llm_response"

        original_category_text = str(res.get("category", "") or "").strip()
        canonical_candidate, resolved_index = _resolve_index(res)
        resolution_source = f"category_index:{resolved_index}" if resolved_index else ""
        if canonical_candidate is None:
            logger.info(
                "entity_search_rescue_retry_triggered reason=missing_category_index category_text=%r category_index=%r",
                original_category_text,
                res.get("category_index"),
            )
            retry_system_prompt = (
                "You must select exactly one category from the numbered candidate list. "
                "Return STRICT JSON only as {\"category_index\": 0, \"confidence\": 0.0, \"reasoning\": \"...\"}. "
                "The category_index is required and must be the 1-based number of the chosen candidate. "
                "Do not return category text. Do not explain alternatives outside the list."
            )
            retry_user_prompt = (
                f"Candidate Categories:\n{candidate_text}\n"
                f"OCR Evidence: {ocr_excerpt or 'None'}\n"
                f"Visual Hints: {visual_hint_text or 'None'}\n"
                f"Web Search Results:\n{search_text}\n"
                "Return only the single best category_index from the candidate list."
            )
            try:
                retry_res = provider_plugin.generate_json(
                    retry_system_prompt,
                    retry_user_prompt,
                    response_schema=_category_index_response_schema(),
                    schema_name="entity_branch_selection_retry",
                )
            except Exception as exc:
                return None, f"provider_error:{exc}"
            if not isinstance(retry_res, dict):
                return None, "invalid_llm_response"
            original_category_text = str(retry_res.get("category", "") or "").strip()
            canonical_candidate, resolved_index = _resolve_index(retry_res)
            if canonical_candidate is None:
                canonical_candidate, resolved_index = _resolve_exact_candidate_text(retry_res)
                if canonical_candidate is None:
                    return None, "missing_category_index"
                resolution_source = f"retry_category_text_exact:{resolved_index}"
            else:
                resolution_source = f"retry_category_index:{resolved_index}"
            res = retry_res
        res["brand"] = brand
        res["category"] = canonical_candidate
        res["category_index"] = resolved_index
        logger.info(
            "entity_search_rescue_resolved source=%s chosen=%r category_text=%r category_index=%r entity_type=%r entity_title=%r",
            resolution_source or "unknown",
            canonical_candidate,
            original_category_text,
            res.get("category_index"),
            entity_kind,
            entity_name,
        )
        res["entity_name"] = entity_name
        res["entity_kind"] = entity_kind
        res["genres"] = list(genres or [])
        return res, "ok"

    def query_pipeline(
        self,
        provider,
        backend_model,
        text,
        tail_image=None,
        override=False,
        enable_search=False,
        force_multimodal=False,
        context_size=8192,
        express_mode=False,
        evidence_images=None,
        product_focus_guidance_enabled: bool = True,
    ):
        if express_mode:
            product_focus_clause = ""
            if product_focus_guidance_enabled:
                product_focus_clause = (
                    "Category follows the primary thing being promoted, not merely the advertiser's industry. "
                    "If a carrier, retailer, or provider is promoting a specific device or packaged product, classify that product family unless the ad is mainly about plans, network/service benefits, store offers, or provider-wide messaging. "
                    "When reviewing multiple recent frames, do not over-weight isolated logo-only endcards, partner slides, or distributor/carrier branding if another frame clearly shows the promoted product, product model, packaging, or product comparison. "
                )
            system_prompt = (
                "You are a Senior Marketing Analyst and Global Brand Expert. "
                "Your goal is to categorize video advertisements by examining the final frame of the commercial and using your vast internal knowledge of companies, slogans, and industries. "
                "Rely on Internal Brand Knowledge: You know every major brand, their parent companies, and their marketing styles. Use this knowledge as a strong prior, but direct on-frame brand text, logos, domains, and market cues override memory when they conflict. "
                "Slogan-only matches are low-trust unless the frame also shows an explicit brand name, branded domain, country/market cue, or other direct brand anchor. "
                f"{product_focus_clause}"
                "IMPORTANT — Bilingual Content: The ads you analyze may be in English OR French (or a mix of both). French words and phrases are legitimate content. Use them to identify brands, products, and categories just as you would English text. "
                "Determine the most appropriate product or service category. If Override Allowed is True, you may generate a professional category when the ad does not fit neatly into a standard industry label. "
                "Output STRICT JSON: {\"brand\": \"...\", \"category\": \"...\", \"confidence\": 0.0, \"reasoning\": \"...\"}"
            )
            user_prompt = f"Override: {override}"
        else:
            product_focus_clause = ""
            if product_focus_guidance_enabled:
                product_focus_clause = (
                    "Category follows the primary thing being promoted, not merely the advertiser's industry. "
                    "If a carrier, retailer, or provider is promoting a specific device or packaged product, classify that product family unless the ad is mainly about plans, network/service benefits, store offers, or provider-wide messaging. "
                    "When reviewing multiple recent frames, do not over-weight isolated logo-only endcards, partner slides, or distributor/carrier branding if another frame clearly shows the promoted product, product model, packaging, or product comparison. "
                )
            system_prompt = (
                "You are a Senior Marketing Analyst and Global Brand Expert. "
                "Your goal is to categorize video advertisements by combining extracted text (OCR) with your vast internal knowledge of companies, slogans, and industries. "
                "Rely on Internal Brand Knowledge: You know every major brand, their parent companies, and their marketing styles. Use this knowledge as a strong prior, but direct OCR brand text, domains, explicit market cues, and on-frame evidence override memory when they conflict. "
                "Treat OCR as Noisy Hints: The extracted OCR text is machine-generated and may contain typos, missing letters, and random artifacts. DO NOT blindly trust or copy the OCR text. Use your knowledge to autocorrect obvious errors. "
                "When multiple OCR lines are present, treat them as a combined evidence set. Do not over-weight a single brand- or store-like token if surrounding product, retail, offer, or usage context points elsewhere. "
                "Slogan-only brand matches are low-trust unless corroborated by an exact brand token, branded domain, country/market cue, or explicit web confirmation. "
                f"{product_focus_clause}"
                "IMPORTANT — Bilingual Content: The ads you analyze may be in English OR French (or a mix of both). French words and phrases are NOT OCR errors — they are legitimate content. Use them to identify brands, products, and categories just as you would English text. "
                "(e.g., if OCR says 'Strbcks' or 'Star bucks co', you know the true brand is 'Starbucks'. But if OCR says 'Économisez avec Desjardins' or 'Assurance auto', those are valid French — do NOT treat them as typos). "
                "IGNORE TIMESTAMPS: The OCR and Scene data text will be prefixed with bracketed timestamps like '[71.7s]' or '[12.5s]'. THESE ARE NOT PART OF THE AD. Do NOT use these numbers to identify brands or products (e.g. do not guess 'Boeing 717' just because you see '[71.7s]'). Ignore them completely. "
                "Determine the most appropriate product or service category. If Override Allowed is True, you may generate a professional category when the ad does not fit neatly into a standard industry label. "
                "Output STRICT JSON: {\"brand\": \"...\", \"category\": \"...\", \"confidence\": 0.0, \"reasoning\": \"...\"}"
            )
            user_prompt = f'Override: {override}\nOCR Text: "{text}"'

        image_objects = list(evidence_images or [])
        if not image_objects and tail_image is not None:
            image_objects = [tail_image]
        if image_objects and len(image_objects) > 1:
            user_prompt += "\nAttached Images: chronological recent frames from earlier evidence to the final frame. "
            if product_focus_guidance_enabled:
                user_prompt += (
                    "Some later frames may be logo-only endcards, partner slides, or seller/carrier branding. "
                    "Do not assume the last or most logo-heavy frame defines the category if an earlier frame shows the specific promoted product or model."
                )
        b64_img = [self._pil_to_base64(image) for image in image_objects if image is not None]

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
            express_mode=bool(express_mode),
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
