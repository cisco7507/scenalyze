import asyncio

import pytest

from video_service.app import main
from video_service.app.models.job import FolderRequest, JobMode, JobSettings, UrlBatchRequest


pytestmark = pytest.mark.unit


class _Req:
    def __init__(self, internal: bool = False):
        self.query_params = {"internal": "1"} if internal else {}


class _DummyResponse:
    def __init__(self, payload, status_code: int = 200):
        self.status_code = status_code
        self._payload = payload
        self.content = b"{}"
        self.text = str(payload)

    def json(self):
        return self._payload


class _DummyAsyncClient:
    def __init__(self, payloads_by_url):
        self.payloads_by_url = payloads_by_url

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url: str, json: dict, timeout: float):
        payload = self.payloads_by_url[url]
        return _DummyResponse(payload)


def test_create_job_urls_balances_each_url_across_cluster(monkeypatch):
    sequence = iter(["node-a", "node-b"])
    monkeypatch.setattr(main, "_rr_or_raise", lambda: asyncio.sleep(0, result=next(sequence)))
    monkeypatch.setattr(main.cluster, "self_name", "node-a", raising=False)
    monkeypatch.setattr(main.cluster, "get_node_url", lambda node: "http://node-b" if node == "node-b" else "http://node-a", raising=False)

    captured_local = []

    def _fake_create_job(mode: str, settings: JobSettings, url: str | None = None) -> str:
        captured_local.append(url)
        return "node-a-local-1"

    monkeypatch.setattr(main, "_create_job", _fake_create_job)
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda: _DummyAsyncClient(
            {
                "http://node-b/jobs/by-urls?internal=1": [
                    {"job_id": "node-b-remote-1", "status": "queued"},
                ]
            }
        ),
    )

    body = UrlBatchRequest(
        mode=JobMode.pipeline,
        urls=["https://example.com/a", "https://example.com/b"],
        settings=JobSettings(),
    )

    responses = asyncio.run(main.create_job_urls(_Req(), body))

    assert [response.job_id for response in responses] == ["node-a-local-1", "node-b-remote-1"]
    assert captured_local == ["https://example.com/a"]


def test_create_job_folder_balances_each_file_across_cluster(monkeypatch, tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"a")
    (tmp_path / "b.mov").write_bytes(b"b")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    monkeypatch.setattr(main.os, "listdir", lambda path: ["a.mp4", "notes.txt", "b.mov"])

    sequence = iter(["node-a", "node-b"])
    monkeypatch.setattr(main, "_rr_or_raise", lambda: asyncio.sleep(0, result=next(sequence)))
    monkeypatch.setattr(main.cluster, "self_name", "node-a", raising=False)
    monkeypatch.setattr(main.cluster, "get_node_url", lambda node: "http://node-b" if node == "node-b" else "http://node-a", raising=False)

    captured_local = []

    def _fake_create_job(mode: str, settings: JobSettings, url: str | None = None) -> str:
        captured_local.append(url)
        return "node-a-local-folder-1"

    monkeypatch.setattr(main, "_create_job", _fake_create_job)
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda: _DummyAsyncClient(
            {
                "http://node-b/jobs/by-filepath?internal=1": {
                    "job_id": "node-b-remote-folder-1",
                    "status": "queued",
                }
            }
        ),
    )

    body = FolderRequest(
        mode=JobMode.pipeline,
        folder_path=str(tmp_path),
        settings=JobSettings(),
    )

    responses = asyncio.run(main.create_job_folder(_Req(), body))

    assert [response.job_id for response in responses] == ["node-a-local-folder-1", "node-b-remote-folder-1"]
    assert captured_local == [str(tmp_path / "a.mp4")]
