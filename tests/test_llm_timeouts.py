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
    ClassificationPipeline,
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
        enable_search=False,
    )
    llm.query_pipeline(
        provider="LM Studio",
        backend_model="local-model",
        text="sample",
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


def test_specificity_search_query_uses_anchor_terms_not_visual_labels():
    llm = HybridLLM()
    query = llm._build_specificity_search_query(
        brand="Mercy",
        current_category="Movie",
        ocr_text="Mercy Movie.ca FILMED FOR IMAX NOW PLAYING",
    )

    assert "Mercy" in query
    assert '"movie.ca"' in query
    assert "official site" in query
    assert "Action/thriller Cinema" not in query
    assert "Filmed Entertainment" not in query


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


def test_lm_studio_pipeline_uses_structure_only_json_schema(monkeypatch):
    calls: list[dict] = []

    def _fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"brand":"BrandY","category":"Retail","confidence":0.88,"reasoning":"ok"}'
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
        enable_search=False,
    )

    assert result["brand"] == "BrandY"
    assert result["category"] == "Retail"
    assert len(calls) == 1
    call = calls[0]
    assert call["url"].endswith("/v1/chat/completions")
    assert call["timeout"] == 300
    assert call["json"]["messages"][1]["role"] == "user"
    assert "Categories:" not in call["json"]["messages"][1]["content"]
    assert 'OCR Text: "sample"' in call["json"]["messages"][1]["content"]
    assert call["json"]["temperature"] == 0.0
    assert call["json"]["top_p"] == 1.0
    assert call["json"]["presence_penalty"] == 0.0
    assert call["json"]["response_format"]["type"] == "json_schema"
    category_schema = call["json"]["response_format"]["json_schema"]["schema"]["properties"]["category"]
    assert category_schema == {"type": "string"}


def test_llama_server_pipeline_omits_prompt_categories_and_uses_structure_only_schema(monkeypatch):
    calls: list[dict] = []

    def _fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"brand":"Heineken","category":"Beer","confidence":0.91,"reasoning":"ok"}'
                        }
                    }
                ]
            }
        )

    llm = HybridLLM()
    monkeypatch.setattr("video_service.core.llm.requests.post", _fake_post)

    result = llm.query_pipeline(
        provider="llama-server",
        backend_model="unsloth/Qwen3.5-4B-GGUF",
        text="sample",
        enable_search=False,
    )

    assert result["brand"] == "Heineken"
    assert len(calls) == 1
    call = calls[0]
    assert call["url"].endswith("/v1/chat/completions")
    assert "Categories:" not in call["json"]["messages"][1]["content"]
    assert 'OCR Text: "sample"' in call["json"]["messages"][1]["content"]
    category_schema = call["json"]["response_format"]["json_schema"]["schema"]["properties"]["category"]
    assert category_schema == {"type": "string"}


def test_ollama_pipeline_omits_categories_from_prompt_and_sends_no_schema(monkeypatch):
    calls: list[dict] = []

    def _fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _DummyResponse(
            {"message": {"content": '{"brand":"BrandX","category":"Auto","confidence":0.93,"reasoning":"ok"}'}}
        )

    llm = HybridLLM()
    monkeypatch.setattr("video_service.core.llm.requests.post", _fake_post)

    result = llm.query_pipeline(
        provider="Ollama",
        backend_model="qwen3-vl:8b-instruct",
        text="sample",
        enable_search=False,
    )

    assert result["brand"] == "BrandX"
    assert len(calls) == 1
    call = calls[0]
    assert call["url"].endswith("/api/chat")
    assert "Categories:" not in call["json"]["messages"][1]["content"]
    assert 'OCR Text: "sample"' in call["json"]["messages"][1]["content"]
    assert "response_format" not in call["json"]


class _FakeProvider:
    supports_vision = False

    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def generate_json(self, system_prompt: str, user_prompt: str, images=None, **kwargs) -> dict:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "images": images,
            }
        )
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


class _FakeSearchClient:
    def __init__(self, snippets: str | None) -> None:
        self.snippets = snippets
        self.queries: list[str] = []

    def search(self, query: str, timeout: int = 45):
        self.queries.append(query)
        return self.snippets


def test_brand_ambiguity_guard_triggers_web_disambiguation(monkeypatch):
    provider = _FakeProvider(
        [
            {
                "brand": "Telus",
                "category": "Telecommunications",
                "confidence": 0.99,
                "reasoning": "This slogan is famously associated with Telus and matches its advertising style.",
            },
            {
                "brand": "Telstra",
                "category": "Telecommunications",
                "confidence": 0.93,
                "reasoning": "Web snippets explicitly tie the slogan to Telstra.",
            },
        ]
    )
    search_client = _FakeSearchClient("Telstra wherever we go mobile network official slogan")
    pipeline = ClassificationPipeline(provider=provider, search_client=search_client, validation_threshold=0.7)

    monkeypatch.setenv("BRAND_AMBIGUITY_GUARD", "true")
    monkeypatch.setenv("BRAND_AMBIGUITY_CONFIDENCE_THRESHOLD", "0.85")

    result = pipeline.classify(
        system_prompt="sys",
        user_prompt="user",
        raw_ocr_text="whereverwego",
        enable_search=True,
        include_image=False,
        image_b64=None,
    )

    assert result["brand"] == "Telstra"
    assert result["brand_ambiguity_flag"] is True
    assert result["brand_ambiguity_resolved"] is True
    assert result["brand_disambiguation_reason"] == "brand_corrected_by_web"
    assert len(provider.calls) == 2
    assert len(search_client.queries) == 1
    assert "Telus" not in search_client.queries[0]
    assert "whereverwego" in search_client.queries[0]


def test_brand_ambiguity_guard_skips_when_exact_brand_anchor_present(monkeypatch):
    provider = _FakeProvider(
        [
            {
                "brand": "Telstra",
                "category": "Telecommunications",
                "confidence": 0.99,
                "reasoning": "Direct OCR brand anchor.",
            }
        ]
    )
    search_client = _FakeSearchClient("unused")
    pipeline = ClassificationPipeline(provider=provider, search_client=search_client, validation_threshold=0.7)

    monkeypatch.setenv("BRAND_AMBIGUITY_GUARD", "true")
    monkeypatch.setenv("BRAND_AMBIGUITY_CONFIDENCE_THRESHOLD", "0.85")

    result = pipeline.classify(
        system_prompt="sys",
        user_prompt="user",
        raw_ocr_text="Telstra whereverwego",
        enable_search=True,
        include_image=False,
        image_b64=None,
    )

    assert result["brand"] == "Telstra"
    assert "brand_ambiguity_flag" not in result
    assert len(provider.calls) == 1
    assert search_client.queries == []


def test_brand_ambiguity_guard_fails_closed_when_search_unavailable(monkeypatch):
    provider = _FakeProvider(
        [
            {
                "brand": "Telus",
                "category": "Telecommunications",
                "confidence": 0.99,
                "reasoning": "This slogan is famously associated with Telus and matches its advertising style.",
            }
        ]
    )
    search_client = _FakeSearchClient(None)
    pipeline = ClassificationPipeline(provider=provider, search_client=search_client, validation_threshold=0.7)

    monkeypatch.setenv("BRAND_AMBIGUITY_GUARD", "true")
    monkeypatch.setenv("BRAND_AMBIGUITY_CONFIDENCE_THRESHOLD", "0.85")

    result = pipeline.classify(
        system_prompt="sys",
        user_prompt="user",
        raw_ocr_text="whereverwego",
        enable_search=True,
        include_image=False,
        image_b64=None,
    )

    assert result["brand"] == "Telus"
    assert result["brand_ambiguity_flag"] is True
    assert result["brand_ambiguity_resolved"] is False
    assert result["brand_disambiguation_reason"] == "search_unavailable"
    assert len(provider.calls) == 1
    assert len(search_client.queries) == 1
