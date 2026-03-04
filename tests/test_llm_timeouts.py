import sys
import types

import pytest
import requests

# `video_service.core.llm` imports `ddgs`; stub it for unit tests.
if "ddgs" not in sys.modules:
    ddgs_stub = types.ModuleType("ddgs")
    ddgs_stub.DDGS = object
    sys.modules["ddgs"] = ddgs_stub

from video_service.core.llm import (
    HybridLLM,
    OpenAICompatibleProvider,
    OllamaQwenProvider,
    create_provider,
)

pytestmark = pytest.mark.unit


class _DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}: {self.text}")


def test_query_pipeline_uses_timeout_300_for_remote_calls(monkeypatch):
    calls: list[dict] = []

    def _fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "timeout": timeout})
        if "api/generate" in url:
            return _DummyResponse({"response": '{"brand":"B","category":"C","confidence":0.9,"reasoning":"ok"}'})
        return _DummyResponse({"choices": [{"message": {"content": '{"brand":"B","category":"C","confidence":0.9,"reasoning":"ok"}'}}]})

    llm = HybridLLM()
    monkeypatch.setattr("video_service.core.llm.requests.post", _fake_post)

    llm.query_pipeline(
        provider="Ollama",
        backend_model="qwen3-vl:8b-instruct",
        text="sample",
        categories="Auto",
        enable_search=False,
    )
    llm.query_pipeline(
        provider="LM Studio",
        backend_model="local-model",
        text="sample",
        categories="Auto",
        enable_search=False,
    )

    assert len(calls) == 2
    assert all(call["timeout"] == 300 for call in calls)


def test_query_agent_uses_timeout_300_for_remote_calls(monkeypatch):
    calls: list[dict] = []

    def _fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "timeout": timeout})
        if "api/generate" in url:
            return _DummyResponse({"response": "[TOOL: FINAL | brand=\"Brand\" category=\"Cat\" reason=\"ok\"]"})
        return _DummyResponse({"choices": [{"message": {"content": "[TOOL: FINAL | brand=\"Brand\" category=\"Cat\" reason=\"ok\"]"}}]})

    llm = HybridLLM()
    monkeypatch.setattr("video_service.core.llm.requests.post", _fake_post)

    llm.query_agent(
        provider="Ollama",
        backend_model="qwen3-vl:8b-instruct",
        prompt="prompt",
    )
    llm.query_agent(
        provider="LM Studio",
        backend_model="local-model",
        prompt="prompt",
    )

    assert len(calls) == 2
    assert all(call["timeout"] == 300 for call in calls)


def test_query_pipeline_timeout_returns_structured_error(monkeypatch):
    llm = HybridLLM()

    def _timeout_post(url, json=None, timeout=None):
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr("video_service.core.llm.requests.post", _timeout_post)

    result = llm.query_pipeline(
        provider="Ollama",
        backend_model="qwen3-vl:8b-instruct",
        text="sample",
        categories="Auto",
        enable_search=False,
    )

    assert isinstance(result, dict)
    assert result.get("error") == "Timeout after 300s"


def test_query_agent_timeout_returns_tool_error(monkeypatch):
    llm = HybridLLM()

    def _timeout_post(url, json=None, timeout=None):
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr("video_service.core.llm.requests.post", _timeout_post)

    result = llm.query_agent(
        provider="LM Studio",
        backend_model="local-model",
        prompt="prompt",
    )

    assert result == '[TOOL: ERROR | reason="LLM Timeout after 300s"]'


def test_create_provider_routes_qwen_models_to_qwen_plugin():
    provider = create_provider("Ollama", "qwen3-vl:8b-instruct", context_size=8192)
    assert isinstance(provider, OllamaQwenProvider)


def test_create_provider_routes_llama_server_to_openai_compat_json_mode():
    provider = create_provider("llama-server", "unsloth/Qwen3.5-4B-GGUF", context_size=8192)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.force_json_mode is True


def test_qwen_ollama_agent_uses_chat_endpoint_and_qwen_options(monkeypatch):
    calls: list[dict] = []

    def _fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _DummyResponse({"message": {"content": "[TOOL: FINAL | reason=\"ok\"]"}})

    monkeypatch.setattr("video_service.core.llm.requests.post", _fake_post)
    llm = HybridLLM()
    result = llm.query_agent(
        provider="Ollama",
        backend_model="qwen3-vl:8b-instruct",
        prompt="test prompt",
    )

    assert result == '[TOOL: FINAL | reason="ok"]'
    assert len(calls) == 1
    call = calls[0]
    assert call["url"].endswith("/api/chat")
    assert call["timeout"] == 300
    options = call["json"]["options"]
    assert options["temperature"] == 1.0
    assert options["top_p"] == 0.95
    assert options["top_k"] == 20
    assert options["presence_penalty"] == 1.5
    assert options["repeat_penalty"] == 1.0


def test_lm_studio_pipeline_uses_assistant_prefill_and_reconstructs_json(monkeypatch):
    calls: list[dict] = []

    def _fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": 'BrandY",\n  "category":"Retail",\n  "confidence":0.88,\n  "reasoning":"ok"}'
                        }
                    }
                ]
            }
        )

    llm = HybridLLM()
    monkeypatch.setattr("video_service.core.llm.requests.post", _fake_post)

    result = llm.query_pipeline(
        provider="LM Studio",
        backend_model="local-model",
        text="sample",
        categories="Auto",
        enable_search=False,
    )

    assert result["brand"] == "BrandY"
    assert result["category"] == "Retail"
    assert len(calls) == 1
    call = calls[0]
    assert call["url"].endswith("/v1/chat/completions")
    assert call["timeout"] == 300
    assert call["json"]["messages"][1]["role"] == "user"
    assert call["json"]["messages"][2]["role"] == "assistant"
    assert call["json"]["messages"][2]["content"].startswith("</think>\n{\n  \"brand\": \"")
    assert call["json"]["temperature"] == 0.0
    assert call["json"]["top_p"] == 1.0
    assert call["json"]["presence_penalty"] == 2.0
    assert call["json"]["response_format"] == {"type": "json_object"}
