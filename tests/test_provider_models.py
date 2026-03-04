import asyncio

import pytest

import video_service.app.main as main

pytestmark = pytest.mark.unit


class _DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


class _DummyAsyncClient:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self._status_code = status_code
        self.calls: list[tuple[str, float]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, timeout: float = 0.0):
        self.calls.append((url, timeout))
        return _DummyResponse(self._payload, status_code=self._status_code)


def test_list_llama_server_models_reads_openai_v1_models(monkeypatch):
    payload = {
        "object": "list",
        "data": [
            {"id": "unsloth/Qwen3.5-4B-GGUF", "created": 12345, "owned_by": "llama-server"},
            {"id": "another/model", "created": 99999, "owned_by": "llama-server"},
        ],
    }
    client = _DummyAsyncClient(payload=payload)

    monkeypatch.setenv("OPENAI_COMPAT_URL", "http://localhost:1234/v1/chat/completions")
    monkeypatch.setattr(main.httpx, "AsyncClient", lambda: client)

    models = asyncio.run(main.list_llama_server_models())

    assert [m["name"] for m in models] == ["unsloth/Qwen3.5-4B-GGUF", "another/model"]
    assert client.calls
    called_url, timeout = client.calls[0]
    assert called_url == "http://localhost:1234/v1/models"
    assert timeout == 5.0


def test_list_provider_models_routes_llama_server(monkeypatch):
    async def _fake_llama_models():
        return [{"name": "unsloth/Qwen3.5-4B-GGUF"}]

    monkeypatch.setattr(main, "list_llama_server_models", _fake_llama_models)

    models = asyncio.run(main.list_provider_models(provider="llama-server"))
    assert models == [{"name": "unsloth/Qwen3.5-4B-GGUF"}]
