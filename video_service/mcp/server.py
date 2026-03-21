from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from video_service.app.models.job import JobMode, JobSettings
from video_service.core.logging_setup import configure_logging
from video_service.mcp.service import (
    HttpScenalyzeService,
    LocalScenalyzeService,
    ScenalyzeService,
    ScenalyzeServiceError,
    job_resource_uris,
)

configure_logging()
logger = logging.getLogger(__name__)

SERVER_INSTRUCTIONS = (
    "Scenalyze MCP exposes job-oriented video classification workflows, taxonomy inspection, "
    "and operational diagnostics. Prefer asynchronous usage: submit work, then poll or read "
    "job resources until the job reaches completed or failed."
)


def _json_resource(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _resolve_service(
    service_mode: str,
    *,
    api_base_url: str | None = None,
) -> ScenalyzeService:
    normalized = str(service_mode or "local").strip().lower()
    if normalized == "http":
        return HttpScenalyzeService(
            api_base_url or os.environ.get("SCENALYZE_API_BASE_URL", "http://127.0.0.1:8000")
        )
    if normalized != "local":
        raise ValueError(f"Unsupported Scenalyze MCP service mode: {service_mode}")
    return LocalScenalyzeService()


def create_server(
    service: ScenalyzeService | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    streamable_http_path: str | None = None,
    stateless_http: bool | None = None,
    log_level: str | None = None,
) -> FastMCP:
    svc = service or _resolve_service(os.environ.get("SCENALYZE_MCP_SERVICE_MODE", "local"))
    resolved_stateless_http = stateless_http
    if resolved_stateless_http is None:
        raw = (os.environ.get("SCENALYZE_MCP_STATELESS_HTTP", "1") or "").strip().lower()
        resolved_stateless_http = raw not in {"0", "false", "no", "off"}
    mcp = FastMCP(
        "Scenalyze MCP",
        instructions=SERVER_INSTRUCTIONS,
        host=host or os.environ.get("SCENALYZE_MCP_HOST", "127.0.0.1"),
        port=int(port or os.environ.get("SCENALYZE_MCP_PORT", "8766")),
        streamable_http_path=streamable_http_path or os.environ.get("SCENALYZE_MCP_PATH", "/mcp"),
        json_response=True,
        stateless_http=bool(resolved_stateless_http),
        log_level=(log_level or os.environ.get("SCENALYZE_MCP_LOG_LEVEL", "INFO")).upper(),
    )

    @mcp.tool(description="Queue a local video file for Scenalyze classification.")
    def submit_video_by_filepath(
        file_path: str,
        mode: JobMode = JobMode.pipeline,
        settings: JobSettings | None = None,
    ) -> dict[str, Any]:
        result = svc.submit_video_by_filepath(file_path, mode=mode, settings=settings)
        result["poll_hint"] = (
            f"Read {job_resource_uris(result['job_id'])['status']} or call get_job_status "
            "until status is completed or failed."
        )
        return result

    @mcp.tool(description="Queue a remote video URL for Scenalyze classification.")
    def submit_video_by_url(
        url: str,
        mode: JobMode = JobMode.pipeline,
        settings: JobSettings | None = None,
    ) -> dict[str, Any]:
        result = svc.submit_video_by_url(url, mode=mode, settings=settings)
        result["poll_hint"] = (
            f"Read {job_resource_uris(result['job_id'])['status']} or call get_job_status "
            "until status is completed or failed."
        )
        return result

    @mcp.tool(description="Return the current status row for a Scenalyze job.")
    def get_job_status(job_id: str) -> dict[str, Any]:
        return svc.get_job_status(job_id)

    @mcp.tool(description="List recent Scenalyze jobs ordered by creation time.")
    def list_recent_jobs(limit: int = 20) -> dict[str, Any]:
        jobs = svc.list_recent_jobs(limit)
        return {"jobs": jobs, "count": len(jobs)}

    @mcp.tool(description="Return the normalized classification result for a Scenalyze job.")
    def get_job_result(job_id: str) -> dict[str, Any]:
        return svc.get_job_result(job_id)

    @mcp.tool(description="Return artifacts captured during a Scenalyze job.")
    def get_job_artifacts(job_id: str) -> dict[str, Any]:
        return svc.get_job_artifacts(job_id)

    @mcp.tool(description="Return the event log recorded for a Scenalyze job.")
    def get_job_events(job_id: str) -> dict[str, Any]:
        return svc.get_job_events(job_id)

    @mcp.tool(description="Return the structured explanation payload for a Scenalyze job.")
    def get_job_explanation(job_id: str) -> dict[str, Any]:
        return svc.get_job_explanation(job_id)

    @mcp.tool(description="Map a freeform category phrase into the FreeWheel taxonomy and return nearby candidates.")
    def find_taxonomy_candidates(
        query: str,
        limit: int = 5,
        suggested_categories_text: str = "",
        predicted_brand: str = "",
        ocr_summary: str = "",
        reasoning_summary: str = "",
    ) -> dict[str, Any]:
        return svc.find_taxonomy_candidates(
            query,
            limit=limit,
            suggested_categories_text=suggested_categories_text,
            predicted_brand=predicted_brand,
            ocr_summary=ocr_summary,
            reasoning_summary=reasoning_summary,
        )

    @mcp.tool(description="Return Scenalyze cluster node state and maintenance flags.")
    def get_cluster_nodes() -> dict[str, Any]:
        return svc.get_cluster_nodes()

    @mcp.tool(description="Return Scenalyze device diagnostics and torch backend information.")
    def get_device_diagnostics() -> dict[str, Any]:
        return svc.get_device_diagnostics()

    @mcp.tool(description="Return Scenalyze system hardware and capability profile.")
    def get_system_profile() -> dict[str, Any]:
        return svc.get_system_profile()

    @mcp.tool(description="Return current Scenalyze concurrency settings and diagnostics.")
    def get_concurrency_diagnostics() -> dict[str, Any]:
        return svc.get_concurrency_diagnostics()

    @mcp.tool(description="List models available from the configured Ollama or llama-server provider.")
    def list_provider_models(provider: str = "ollama") -> dict[str, Any]:
        models = svc.list_provider_models(provider)
        return {"provider": provider, "models": models, "count": len(models)}

    @mcp.resource(
        "scenalyze://jobs/{job_id}/status",
        name="job-status",
        description="Normalized status row for a Scenalyze job.",
        mime_type="application/json",
    )
    def resource_job_status(job_id: str) -> str:
        return _json_resource(svc.get_job_status(job_id))

    @mcp.resource(
        "scenalyze://jobs/{job_id}/result",
        name="job-result",
        description="Normalized classification result payload for a Scenalyze job.",
        mime_type="application/json",
    )
    def resource_job_result(job_id: str) -> str:
        return _json_resource(svc.get_job_result(job_id))

    @mcp.resource(
        "scenalyze://jobs/{job_id}/artifacts",
        name="job-artifacts",
        description="Artifacts and trace metadata captured during a Scenalyze job.",
        mime_type="application/json",
    )
    def resource_job_artifacts(job_id: str) -> str:
        return _json_resource(svc.get_job_artifacts(job_id))

    @mcp.resource(
        "scenalyze://jobs/{job_id}/events",
        name="job-events",
        description="Event log captured during a Scenalyze job.",
        mime_type="application/json",
    )
    def resource_job_events(job_id: str) -> str:
        return _json_resource(svc.get_job_events(job_id))

    @mcp.resource(
        "scenalyze://jobs/{job_id}/explanation",
        name="job-explanation",
        description="Structured explanation for how a Scenalyze job reached its outcome.",
        mime_type="application/json",
    )
    def resource_job_explanation(job_id: str) -> str:
        return _json_resource(svc.get_job_explanation(job_id))

    @mcp.resource(
        "scenalyze://cluster/nodes",
        name="cluster-nodes",
        description="Current Scenalyze cluster node status and maintenance state.",
        mime_type="application/json",
    )
    def resource_cluster_nodes() -> str:
        return _json_resource(svc.get_cluster_nodes())

    @mcp.resource(
        "scenalyze://diagnostics/device",
        name="device-diagnostics",
        description="Current Scenalyze device diagnostics.",
        mime_type="application/json",
    )
    def resource_device_diagnostics() -> str:
        return _json_resource(svc.get_device_diagnostics())

    @mcp.resource(
        "scenalyze://system/profile",
        name="system-profile",
        description="Current Scenalyze system hardware profile.",
        mime_type="application/json",
    )
    def resource_system_profile() -> str:
        return _json_resource(svc.get_system_profile())

    @mcp.resource(
        "scenalyze://taxonomy/explorer",
        name="taxonomy-explorer",
        description="Expanded FreeWheel taxonomy explorer payload.",
        mime_type="application/json",
    )
    def resource_taxonomy_explorer() -> str:
        return _json_resource(svc.get_taxonomy_explorer())

    @mcp.resource(
        "scenalyze://models/{provider}",
        name="provider-models",
        description="Models available from a configured provider.",
        mime_type="application/json",
    )
    def resource_provider_models(provider: str) -> str:
        return _json_resource(
            {
                "provider": provider,
                "models": svc.list_provider_models(provider),
            }
        )

    @mcp.prompt(
        name="investigate_job",
        description="Guide an agent through a single Scenalyze job investigation.",
    )
    def investigate_job(job_id: str) -> str:
        resources = job_resource_uris(job_id)
        return (
            f"Investigate Scenalyze job `{job_id}`.\n"
            f"1. Read {resources['status']}.\n"
            f"2. Read {resources['result']}.\n"
            f"3. Read {resources['explanation']}.\n"
            f"4. Read {resources['artifacts']}.\n"
            f"5. Read {resources['events']} only if the explanation is incomplete.\n\n"
            "Summarize the final classification, the evidence used, any rerank or recovery path, "
            "and whether the result appears correct."
        )

    @mcp.prompt(
        name="compare_jobs",
        description="Guide an agent through a side-by-side comparison of two Scenalyze jobs.",
    )
    def compare_jobs(job_id_a: str, job_id_b: str) -> str:
        return (
            f"Compare Scenalyze jobs `{job_id_a}` and `{job_id_b}`.\n"
            f"Read scenalyze://jobs/{job_id_a}/status, scenalyze://jobs/{job_id_a}/result, scenalyze://jobs/{job_id_a}/explanation.\n"
            f"Read scenalyze://jobs/{job_id_b}/status, scenalyze://jobs/{job_id_b}/result, scenalyze://jobs/{job_id_b}/explanation.\n\n"
            "Explain differences in initial category guess, mapper behavior, rerank/family selection, "
            "and final category. Be explicit about which stage changed the outcome."
        )

    return mcp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Scenalyze MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default=os.environ.get("SCENALYZE_MCP_TRANSPORT", "stdio"),
        help="MCP transport to expose.",
    )
    parser.add_argument(
        "--service-mode",
        choices=("local", "http"),
        default=os.environ.get("SCENALYZE_MCP_SERVICE_MODE", "local"),
        help="How the MCP server accesses Scenalyze state.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("SCENALYZE_API_BASE_URL", "http://127.0.0.1:8000"),
        help="Base URL for the existing Scenalyze HTTP API when --service-mode=http.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("SCENALYZE_MCP_HOST", "127.0.0.1"),
        help="Bind host for Streamable HTTP transport.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SCENALYZE_MCP_PORT", "8766")),
        help="Bind port for Streamable HTTP transport.",
    )
    parser.add_argument(
        "--path",
        default=os.environ.get("SCENALYZE_MCP_PATH", "/mcp"),
        help="Streamable HTTP MCP mount path.",
    )
    parser.add_argument(
        "--stateful-http",
        action="store_true",
        help="Use session-based Streamable HTTP instead of the default stateless mode.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("SCENALYZE_MCP_LOG_LEVEL", "INFO"),
        help="MCP server log level.",
    )
    args = parser.parse_args(argv)

    service = _resolve_service(args.service_mode, api_base_url=args.api_base_url)
    server = create_server(
        service,
        host=args.host,
        port=args.port,
        streamable_http_path=args.path,
        stateless_http=not args.stateful_http,
        log_level=args.log_level,
    )
    logger.info(
        "starting Scenalyze MCP transport=%s service_mode=%s host=%s port=%s path=%s stateless_http=%s",
        args.transport,
        args.service_mode,
        args.host,
        args.port,
        args.path,
        not args.stateful_http,
    )
    try:
        server.run(args.transport)
    except ScenalyzeServiceError as exc:
        raise SystemExit(str(exc)) from exc
