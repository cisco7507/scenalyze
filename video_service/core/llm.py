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
        json_prefix = '{\n  "reasoning": "'
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
            "presence_penalty": 0.0,
        }

        if self.force_json_mode:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ad_classification",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "brand": {"type": "string"},
                            "category": {"type": "string"},
                            "confidence": {"type": "number"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["brand", "category", "confidence", "reasoning"],
                    },
                },
            }
            logger.info("JSON schema structured output active (no enum constraint)")
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

    def _build_brand_disambiguation_query(self, raw_ocr_text: str) -> str:
        query_parts: list[str] = []
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
    ) -> tuple[dict | None, str]:
        current_brand = str(current_result.get("brand", "") or "").strip()
        query = self._build_brand_disambiguation_query(raw_ocr_text)
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
                "Confirm or correct the brand. Slogan-only matches are low-trust unless corroborated by exact brand text, a domain, market/country cues, or explicit web evidence. "
                "Keep the category unless a corrected brand clearly changes it."
            ),
        )
        if not isinstance(val_res, dict) or "brand" not in val_res:
            return None, "invalid_llm_response"

        resolved_brand = str(val_res.get("brand", "") or "").strip()
        if resolved_brand.lower() in {"", "unknown", "none", "n/a"}:
            return None, "blank_brand_response"

        if not _brand_confirmed_by_web(resolved_brand, snippets):
            return None, f"web_unconfirmed_brand={resolved_brand!r}"

        if resolved_brand.lower() == current_brand.lower():
            val_res["reasoning"] = "(Brand confirmed) " + str(val_res.get("reasoning", "") or "")
            return val_res, "brand_confirmed_by_web"

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
        images = [image_b64] if (include_image and image_b64 and self.provider.supports_vision) else None
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
                    refined, refine_reason = self._attempt_brand_disambiguation(system_prompt, res, raw_ocr_text)
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
        match = re.search(r"\b([a-z0-9-]+\.[a-z]{2,}(?:/[a-z0-9/_-]+)?)\b", text, flags=re.IGNORECASE)
        if not match:
            return "", ""
        raw = match.group(1)
        domain = raw.split("/", 1)[0].lower()
        path = raw.split("/", 1)[1] if "/" in raw else ""
        return domain, path

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

    def query_specificity_rescue(
        self,
        provider,
        backend_model,
        brand: str,
        current_category: str,
        ocr_text: str,
        candidate_categories: list[str] | None = None,
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
            candidate_text = " | ".join(candidate_categories)

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
            f"Candidate Categories: {candidate_text or current_category}\n"
            f"OCR Evidence: {ocr_text}\n"
            f"Visual Hints: {visual_hint_text or 'None'}\n"
            f"Web Search Snippets: {snippets}\n"
            "Return the most specific supported product or service category from the candidate list."
        )
        try:
            res = provider_plugin.generate_json(system_prompt, user_prompt)
        except Exception as exc:
            return None, f"provider_error:{exc}"
        if not isinstance(res, dict) or "category" not in res:
            return None, "invalid_llm_response"
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
    ):
        if express_mode:
            system_prompt = (
                "You are a Senior Marketing Analyst and Global Brand Expert. "
                "Your goal is to categorize video advertisements by examining the final frame of the commercial and using your vast internal knowledge of companies, slogans, and industries. "
                "Rely on Internal Brand Knowledge: You know every major brand, their parent companies, and their marketing styles. Use this knowledge as a strong prior, but direct on-frame brand text, logos, domains, and market cues override memory when they conflict. "
                "Slogan-only matches are low-trust unless the frame also shows an explicit brand name, branded domain, country/market cue, or other direct brand anchor. "
                "IMPORTANT — Bilingual Content: The ads you analyze may be in English OR French (or a mix of both). French words and phrases are legitimate content. Use them to identify brands, products, and categories just as you would English text. "
                "Determine the most appropriate product or service category. If Override Allowed is True, you may generate a professional category when the ad does not fit neatly into a standard industry label. "
                "Output STRICT JSON: {\"brand\": \"...\", \"category\": \"...\", \"confidence\": 0.0, \"reasoning\": \"...\"}"
            )
            user_prompt = f"Override: {override}"
        else:
            system_prompt = (
                "You are a Senior Marketing Analyst and Global Brand Expert. "
                "Your goal is to categorize video advertisements by combining extracted text (OCR) with your vast internal knowledge of companies, slogans, and industries. "
                "Rely on Internal Brand Knowledge: You know every major brand, their parent companies, and their marketing styles. Use this knowledge as a strong prior, but direct OCR brand text, domains, explicit market cues, and on-frame evidence override memory when they conflict. "
                "Treat OCR as Noisy Hints: The extracted OCR text is machine-generated and may contain typos, missing letters, and random artifacts. DO NOT blindly trust or copy the OCR text. Use your knowledge to autocorrect obvious errors. "
                "When multiple OCR lines are present, treat them as a combined evidence set. Do not over-weight a single brand- or store-like token if surrounding product, retail, offer, or usage context points elsewhere. "
                "Slogan-only brand matches are low-trust unless corroborated by an exact brand token, branded domain, country/market cue, or explicit web confirmation. "
                "IMPORTANT — Bilingual Content: The ads you analyze may be in English OR French (or a mix of both). French words and phrases are NOT OCR errors — they are legitimate content. Use them to identify brands, products, and categories just as you would English text. "
                "(e.g., if OCR says 'Strbcks' or 'Star bucks co', you know the true brand is 'Starbucks'. But if OCR says 'Économisez avec Desjardins' or 'Assurance auto', those are valid French — do NOT treat them as typos). "
                "IGNORE TIMESTAMPS: The OCR and Scene data text will be prefixed with bracketed timestamps like '[71.7s]' or '[12.5s]'. THESE ARE NOT PART OF THE AD. Do NOT use these numbers to identify brands or products (e.g. do not guess 'Boeing 717' just because you see '[71.7s]'). Ignore them completely. "
                "Determine the most appropriate product or service category. If Override Allowed is True, you may generate a professional category when the ad does not fit neatly into a standard industry label. "
                "Output STRICT JSON: {\"brand\": \"...\", \"category\": \"...\", \"confidence\": 0.0, \"reasoning\": \"...\"}"
            )
            user_prompt = f'Override: {override}\nOCR Text: "{text}"'
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
