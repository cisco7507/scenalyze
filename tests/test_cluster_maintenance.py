import asyncio
import json

import pytest
from fastapi import HTTPException

from video_service.app import main
from video_service.core.cluster import ClusterConfig


pytestmark = pytest.mark.unit


class _DummyResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.content = b"{}"
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class _DummyAsyncClient:
    def __init__(self, payload_by_url: dict[str, dict]):
        self.payload_by_url = payload_by_url

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url: str, timeout: float):
        payload = self.payload_by_url.get(url)
        if payload is None:
            return _DummyResponse(404, {"detail": "missing"})
        return _DummyResponse(200, payload)


def test_cluster_config_excludes_maintenance_nodes_from_round_robin(tmp_path, monkeypatch):
    config_path = tmp_path / "cluster_config.json"
    config_path.write_text(
        json.dumps(
            {
                "self_name": "node-a",
                "nodes": {
                    "node-a": "http://node-a",
                    "node-b": "http://node-b",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NODE_RUNTIME_STATE_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("NODE_NAME", "node-a")

    cluster = ClusterConfig(str(config_path))
    cluster.node_status = {"node-a": True, "node-b": True}
    cluster.node_maintenance["node-a"] = True
    cluster.node_maintenance["node-b"] = False

    assert cluster.select_rr_node() == "node-b"

    cluster.node_maintenance["node-b"] = True
    assert cluster.select_rr_node() is None


def test_cluster_config_persists_local_maintenance_state(tmp_path, monkeypatch):
    config_path = tmp_path / "cluster_config.json"
    config_path.write_text(
        json.dumps(
            {
                "self_name": "node-a",
                "nodes": {
                    "node-a": "http://node-a",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NODE_RUNTIME_STATE_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("NODE_NAME", "node-a")

    cluster = ClusterConfig(str(config_path))
    cluster.set_maintenance_mode(True)

    reloaded = ClusterConfig(str(config_path))

    assert reloaded.maintenance_mode is True
    assert reloaded.node_maintenance["node-a"] is True
    assert reloaded.is_accepting_new_jobs("node-a") is False


def test_rr_or_raise_distinguishes_maintenance_from_health(monkeypatch):
    monkeypatch.setattr(main.cluster, "select_rr_node", lambda: None, raising=False)
    monkeypatch.setattr(main.cluster, "get_rr_coordinator", lambda: "node-a", raising=False)
    monkeypatch.setattr(main.cluster, "self_name", "node-a", raising=False)
    monkeypatch.setattr(main.cluster, "get_healthy_nodes", lambda: ["node-a"], raising=False)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(main._rr_or_raise())

    assert exc.value.status_code == 503
    assert exc.value.detail == "No nodes accepting new jobs"


def test_cluster_nodes_exposes_maintenance_and_accepting_maps(monkeypatch):
    monkeypatch.setattr(main.cluster, "nodes", {"node-a": "http://node-a"}, raising=False)
    monkeypatch.setattr(main.cluster, "node_status", {"node-a": True}, raising=False)
    monkeypatch.setattr(main.cluster, "node_maintenance", {"node-a": True}, raising=False)
    monkeypatch.setattr(main.cluster, "self_name", "node-a", raising=False)
    monkeypatch.setattr(main.cluster, "is_accepting_new_jobs", lambda node=None: False, raising=False)

    payload = asyncio.run(main.cluster_nodes())

    assert payload["maintenance"]["node-a"] is True
    assert payload["accepting_new_jobs"]["node-a"] is False


def test_rr_or_raise_uses_remote_coordinator_for_cluster_wide_rr(monkeypatch):
    monkeypatch.setattr(main.cluster, "self_name", "node-b", raising=False)
    monkeypatch.setattr(main.cluster, "get_rr_coordinator", lambda: "node-a", raising=False)
    monkeypatch.setattr(main.cluster, "get_node_url", lambda name: "http://node-a" if name == "node-a" else None, raising=False)
    monkeypatch.setattr(main.cluster, "internal_timeout", 1.0, raising=False)
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda: _DummyAsyncClient(
            {"http://node-a/admin/cluster/rr-target?internal=1": {"target": "node-b", "coordinator": "node-a"}}
        ),
    )

    target = asyncio.run(main._rr_or_raise())

    assert target == "node-b"
