import sys
import types
import io
import logging
import concurrent.futures

import pytest
import requests
from PIL import Image

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
    SearchManager,
    create_provider,
)
from video_service.core import logging_setup

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


def test_query_pipeline_openai_compat_includes_all_recent_images(monkeypatch):
    calls: list[dict] = []

    def _fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _DummyResponse({"choices": [{"message": {"content": '{"brand":"B","category":"C","confidence":0.9,"reasoning":"ok"}'}}]})

    llm = HybridLLM()
    monkeypatch.setattr("video_service.core.llm.requests.post", _fake_post)

    image_one = Image.new("RGB", (80, 80), color="red")
    image_two = Image.new("RGB", (80, 80), color="blue")
    llm.query_pipeline(
        provider="LM Studio",
        backend_model="local-model",
        text="sample",
        evidence_images=[image_one, image_two],
        enable_search=False,
        force_multimodal=True,
    )

    payload = calls[0]["json"]
    content = payload["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert len([item for item in content if item["type"] == "image_url"]) == 2


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


def test_specificity_search_query_skips_malformed_www_anchor():
    llm = HybridLLM()
    query = llm._build_specificity_search_query(
        brand="Historica Canada",
        current_category="Non-profit / Educational Organization",
        ocr_text="WWW.HISTORICACANADA Heritage Minute brought to you by Historica Canada",
    )

    assert "site:www.historicacanada" not in query
    assert '"www.historicacanada"' not in query
    assert '"Historica Canada"' in query


def test_create_provider_routes_qwen_models_to_qwen_plugin():
    provider = create_provider("Ollama", "qwen3-vl:8b-instruct", context_size=8192)
    assert isinstance(provider, OllamaQwenProvider)


def test_create_provider_routes_llama_server_to_openai_compat_json_mode():
    provider = create_provider("llama-server", "unsloth/Qwen3.5-4B-GGUF", context_size=8192)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.force_json_mode is True


def test_search_manager_preserves_job_context_for_search_thread_logs(monkeypatch):
    class _DummyDDGS:
        _executor = None
        threads = None

        @classmethod
        def get_executor(cls):
            if cls._executor is None:
                cls._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="DDGS")
            return cls._executor

        def text(self, query, max_results=3):
            def _emit():
                logging.getLogger("primp").info("response: %s", query)
                return [{"body": "snippet"}]

            return self.get_executor().submit(_emit).result()

    monkeypatch.setattr("video_service.core.llm.DDGS", _DummyDDGS)

    primp_logger = logging.getLogger("primp")
    original_level = primp_logger.level
    original_propagate = primp_logger.propagate
    primp_logger.setLevel(logging.INFO)
    primp_logger.propagate = False

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("job_id=%(job_id)s stage=%(stage)s %(message)s"))
    handler.addFilter(logging_setup.ContextEnricherFilter())
    primp_logger.addHandler(handler)

    manager = SearchManager()
    job_token = logging_setup.set_job_context("node-a-search-job")
    stage_tokens = logging_setup.set_stage_context("llm", "search disambiguation")
    try:
        result = manager.search('"onparle" official brand slogan', timeout=2)
    finally:
        logging_setup.reset_stage_context(stage_tokens)
        logging_setup.reset_job_context(job_token)
        primp_logger.removeHandler(handler)
        primp_logger.setLevel(original_level)
        primp_logger.propagate = original_propagate

    assert result == "snippet"
    out = stream.getvalue()
    assert "job_id=node-a-search-job" in out
    assert "stage=llm" in out
    assert '"onparle" official brand slogan' in out

    if _DummyDDGS._executor is not None:
        _DummyDDGS._executor.shutdown(wait=False, cancel_futures=True)


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


def test_query_category_rerank_includes_device_over_provider_guidance(monkeypatch):
    captured: dict[str, str] = {}

    class _CaptureProvider:
        def generate_json(self, system_prompt: str, user_prompt: str, images=None, **kwargs) -> dict:
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            return {
                "brand": "TELUS",
                "category": "Handset/Mobile",
                "confidence": 0.92,
                "reasoning": "The ad is centered on the iPhone device, not the carrier service plan.",
            }

    monkeypatch.setattr("video_service.core.llm.create_provider", lambda *args, **kwargs: _CaptureProvider())

    llm = HybridLLM()
    result, status = llm.query_category_rerank(
        provider="LM Studio",
        backend_model="local-model",
        brand="TELUS",
        raw_category="Mobile Phone Service Provider",
        mapped_category="Wireless Telecommunications Services",
        ocr_text="TELUS iPhone 17e Compared to iPhone 16e",
        reasoning="The ad shows a TELUS-branded iPhone handset comparison.",
        candidate_categories=[
            "Wireless Telecommunications Services",
            "Telecommunication Services",
            "Handset/Mobile",
        ],
    )

    assert status == "ok"
    assert result["category"] == "Handset/Mobile"
    assert "advertiser or provider brand does not automatically determine category" in captured["system_prompt"]
    assert "Decision Guidance: If the ad centers on a named phone/device model" in captured["user_prompt"]


def test_query_pipeline_prompt_prefers_promoted_product_over_provider_industry(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_classify(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        raw_ocr_text: str,
        enable_search: bool,
        include_image: bool,
        image_b64,
        express_mode: bool,
    ):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return {
            "brand": "TELUS",
            "category": "Handset/Mobile",
            "confidence": 0.91,
            "reasoning": "Prompt guidance prefers the product being promoted.",
        }

    monkeypatch.setattr("video_service.core.llm.create_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr("video_service.core.llm.ClassificationPipeline.classify", _fake_classify)

    llm = HybridLLM()
    result = llm.query_pipeline(
        provider="LM Studio",
        backend_model="local-model",
        text="TELUS iPhone 17e Compared to iPhone 16e",
        enable_search=False,
    )

    assert result["category"] == "Handset/Mobile"
    assert "Category follows the primary thing being promoted" in str(captured["system_prompt"])
    assert 'OCR Text: "TELUS iPhone 17e Compared to iPhone 16e"' in str(captured["user_prompt"])


def test_query_pipeline_prompt_warns_against_logo_endcard_overweighting(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_classify(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        raw_ocr_text: str,
        enable_search: bool,
        include_image: bool,
        image_b64,
        express_mode: bool,
    ):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return {
            "brand": "Apple",
            "category": "Handset/Mobile",
            "confidence": 0.9,
            "reasoning": "The earlier frame shows the iPhone product even though later frames are logo endcards.",
        }

    monkeypatch.setattr("video_service.core.llm.create_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr("video_service.core.llm.ClassificationPipeline.classify", _fake_classify)

    llm = HybridLLM()
    result = llm.query_pipeline(
        provider="LM Studio",
        backend_model="local-model",
        text="ROGERS APPLE",
        evidence_images=[Image.new("RGB", (80, 80), color="orange"), Image.new("RGB", (80, 80), color="black")],
        enable_search=False,
        force_multimodal=True,
    )

    assert result["category"] == "Handset/Mobile"
    assert "do not over-weight isolated logo-only endcards" in str(captured["system_prompt"]).lower()
    assert "some later frames may be logo-only endcards" in str(captured["user_prompt"]).lower()


def test_product_focus_guidance_skips_non_product_or_service_only_ads():
    llm = HybridLLM()

    charity_guidance = llm._build_product_focus_guidance(
        raw_category="Charity / Nonprofit",
        mapped_category="Non-profit Organizations",
        ocr_text="Donate today to support local families",
        reasoning="The ad is a fundraising appeal for a charity campaign.",
        candidate_categories=["Non-profit Organizations", "Social Services"],
    )
    assert charity_guidance == ""

    service_guidance = llm._build_product_focus_guidance(
        raw_category="Mobile Phone Service Provider",
        mapped_category="Wireless Telecommunications Services",
        ocr_text="TELUS 5G unlimited data plan network coverage",
        reasoning="The ad emphasizes TELUS wireless plans and coverage.",
        candidate_categories=["Wireless Telecommunications Services", "Handset/Mobile"],
    )
    assert service_guidance == ""


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
    assert '"Telus"' in search_client.queries[0]
    assert "whereverwego" in search_client.queries[0]


def test_brand_ambiguity_guard_disambiguation_uses_current_brand_and_images(monkeypatch):
    provider = _FakeProvider(
        [
            {
                "brand": "Google",
                "category": "Technology / Internet Services",
                "confidence": 0.99,
                "reasoning": "The final frame shows the Google G logo.",
            },
            {
                "brand": "Google",
                "category": "Technology / Internet Services",
                "confidence": 0.95,
                "reasoning": "Recent frames and the logo confirm Google.",
            },
        ]
    )
    provider.supports_vision = True
    search_client = _FakeSearchClient("Google Pixel official slogan")
    pipeline = ClassificationPipeline(provider=provider, search_client=search_client, validation_threshold=0.7)

    monkeypatch.setenv("BRAND_AMBIGUITY_GUARD", "true")
    monkeypatch.setenv("BRAND_AMBIGUITY_CONFIDENCE_THRESHOLD", "0.85")

    result = pipeline.classify(
        system_prompt="sys",
        user_prompt="user",
        raw_ocr_text="onparle",
        enable_search=True,
        include_image=True,
        image_b64=["image-a", "image-b"],
    )

    assert result["brand"] == "Google"
    assert len(search_client.queries) == 1
    assert '"Google"' in search_client.queries[0]
    assert provider.calls[1]["images"] == ["image-a", "image-b"]


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


def test_brand_ambiguity_guard_does_not_override_plausible_ocr_normalization(monkeypatch):
    provider = _FakeProvider(
        [
            {
                "brand": "Graza",
                "category": "Coffee",
                "confidence": 0.95,
                "reasoning": "The OCR text is likely a stylized misspelling of Graza.",
            }
        ]
    )
    search_client = _FakeSearchClient("Grayza apparel hospitality tech")
    pipeline = ClassificationPipeline(provider=provider, search_client=search_client, validation_threshold=0.7)

    monkeypatch.setenv("BRAND_AMBIGUITY_GUARD", "true")
    monkeypatch.setenv("BRAND_AMBIGUITY_CONFIDENCE_THRESHOLD", "0.85")

    result = pipeline.classify(
        system_prompt="sys",
        user_prompt="user",
        raw_ocr_text="GRAYZA",
        enable_search=True,
        include_image=False,
        image_b64=None,
    )

    assert result["brand"] == "Graza"
    assert "brand_ambiguity_flag" not in result
    assert len(provider.calls) == 1
    assert search_client.queries == []
