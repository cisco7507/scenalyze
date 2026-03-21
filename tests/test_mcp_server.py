from __future__ import annotations

import asyncio
import json

from video_service.app.models.job import JobMode
from video_service.mcp.server import create_server
from video_service.mcp.service import HttpScenalyzeService, LocalScenalyzeService


class FakeService:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def submit_video_by_filepath(self, file_path: str, *, mode=JobMode.pipeline, settings=None):
        self.calls.append(("submit_video_by_filepath", file_path, mode, settings))
        return {
            "job_id": "node-a-123",
            "status": "queued",
            "resources": {"status": "scenalyze://jobs/node-a-123/status"},
        }

    def submit_video_by_url(self, url: str, *, mode=JobMode.pipeline, settings=None):
        self.calls.append(("submit_video_by_url", url, mode, settings))
        return {
            "job_id": "node-a-456",
            "status": "queued",
            "resources": {"status": "scenalyze://jobs/node-a-456/status"},
        }

    def get_job_status(self, job_id: str):
        return {"job_id": job_id, "status": "completed"}

    def list_recent_jobs(self, limit: int = 20):
        self.calls.append(("list_recent_jobs", limit))
        return [{"job_id": "node-a-123", "status": "completed"}]

    def get_job_result(self, job_id: str):
        return {"result": [{"brand": "Acme", "category_name": "Toiletries"}]}

    def get_job_artifacts(self, job_id: str):
        return {"artifacts": {"processing_trace": {"summary": {"headline": "ok"}}}}

    def get_job_events(self, job_id: str):
        return {"events": ["event-1"]}

    def get_job_explanation(self, job_id: str):
        return {"explanation": {"job_id": job_id, "summary": {"headline": "ok"}}}

    def get_taxonomy_explorer(self):
        return {"enabled": True, "items": []}

    def find_taxonomy_candidates(
        self,
        query: str,
        *,
        limit: int = 5,
        suggested_categories_text: str = "",
        predicted_brand: str = "",
        ocr_summary: str = "",
        reasoning_summary: str = "",
    ):
        return {"query": query, "top_matches": [{"label": "Toiletries", "score": 0.9}]}

    def get_cluster_nodes(self):
        return {"self": "node-a", "nodes": {"node-a": "http://localhost:8000"}}

    def get_device_diagnostics(self):
        return {"selected_device": "cpu"}

    def get_system_profile(self):
        return {"hardware": {"accelerator": "cpu"}}

    def get_concurrency_diagnostics(self):
        return {"workers": 1}

    def list_provider_models(self, provider: str = "ollama"):
        return [{"name": f"{provider}-model"}]


def _resource_text(blocks) -> str:
    assert blocks
    return "".join(getattr(block, "content", "") for block in blocks)


def test_create_server_exposes_expected_tools():
    server = create_server(FakeService())
    assert server.settings.stateless_http is True
    tools = {tool.name for tool in asyncio.run(server.list_tools())}
    assert {
        "submit_video_by_filepath",
        "submit_video_by_url",
        "get_job_status",
        "list_recent_jobs",
        "get_job_result",
        "get_job_artifacts",
        "get_job_explanation",
        "find_taxonomy_candidates",
        "get_cluster_nodes",
        "get_device_diagnostics",
        "get_system_profile",
        "get_concurrency_diagnostics",
        "list_provider_models",
    }.issubset(tools)


def test_create_server_can_enable_stateful_http():
    server = create_server(FakeService(), stateless_http=False)
    assert server.settings.stateless_http is False


def test_submit_tool_returns_job_resources_and_poll_hint():
    server = create_server(FakeService())
    _content, payload = asyncio.run(
        server.call_tool("submit_video_by_filepath", {"file_path": "/tmp/example.mov"})
    )
    assert payload["job_id"] == "node-a-123"
    assert "poll_hint" in payload
    assert payload["resources"]["status"] == "scenalyze://jobs/node-a-123/status"


def test_dynamic_resource_returns_json_text():
    server = create_server(FakeService())
    payload = _resource_text(
        asyncio.run(server.read_resource("scenalyze://jobs/node-a-123/status"))
    )
    decoded = json.loads(payload)
    assert decoded["job_id"] == "node-a-123"
    assert decoded["status"] == "completed"


def test_local_service_submit_video_by_filepath_uses_create_job(monkeypatch):
    captured = {}

    def fake_create_job(self, mode, settings, url=None):
        captured["mode"] = mode
        captured["settings"] = settings
        captured["url"] = url
        return "node-a-created"

    monkeypatch.setattr(LocalScenalyzeService, "_create_job", fake_create_job)
    service = LocalScenalyzeService()
    payload = service.submit_video_by_filepath("/tmp/sample.mov")
    assert payload["job_id"] == "node-a-created"
    assert payload["resources"]["result"] == "scenalyze://jobs/node-a-created/result"
    assert captured["mode"] == "pipeline"
    assert captured["url"] == "/tmp/sample.mov"


def test_http_service_submit_video_by_url_formats_payload(monkeypatch):
    captured = {}

    def fake_request(self, method, path, *, json_payload=None, params=None):
        captured["method"] = method
        captured["path"] = path
        captured["json_payload"] = json_payload
        captured["params"] = params
        return [{"job_id": "node-a-http", "status": "queued"}]

    monkeypatch.setattr(HttpScenalyzeService, "_request", fake_request)
    service = HttpScenalyzeService("http://127.0.0.1:8000")
    payload = service.submit_video_by_url("https://example.com/video.mp4")
    assert captured["method"] == "POST"
    assert captured["path"] == "/jobs/by-urls"
    assert captured["json_payload"]["urls"] == ["https://example.com/video.mp4"]
    assert payload["resources"]["status"] == "scenalyze://jobs/node-a-http/status"
