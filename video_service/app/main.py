"""
video_service/app/main.py
==========================
FastAPI application — production-hardened.

Security additions
------------------
- CORS restricted to CORS_ORIGINS env var (default: localhost dashboard ports)
- URL validation on every /jobs/by-urls submission
- Path traversal guard on /jobs/by-folder
- Upload size limit (MAX_UPLOAD_MB env var, default 500 MB)
- Internal routing uses ?internal=1 (proxy recursion protection)

Observability additions
-----------------------
- Structured JSON logging via video_service.core.logging_setup
- /metrics endpoint with richer counters
- /health returns node + DB liveness
- /cluster/nodes shows per-node health with latency
"""

import os
import uuid
import json
import shutil
import logging
import mimetypes
import asyncio
import math
import time as _time
from datetime import datetime, timezone
from collections import defaultdict
from contextlib import asynccontextmanager, closing
from typing import List, Optional, AsyncGenerator
from pathlib import Path

import cv2
import httpx
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from video_service.core.logging_setup import (
    clear_recent_log_lines,
    configure_logging,
    get_recent_log_lines,
    subscribe_log_stream,
)
configure_logging()

from video_service.db.database import get_db, init_db
from video_service.app.models.job import (
    JobResponse, JobStatus, JobSettings,
    UrlBatchRequest, FolderRequest, FilePathRequest, BulkDeleteRequest, JobMode,
    BenchmarkTruthCreateRequest,
    BenchmarkRunRequest,
    BenchmarkSuiteUpdateRequest,
    BenchmarkTestUpdateRequest,
)
from video_service.core.device import get_diagnostics
from video_service.core.concurrency import get_concurrency_diagnostics
from video_service.core.cluster import cluster
from video_service.core.categories import category_mapper
from video_service.core.hardware_profiler import get_system_profile
from video_service.core.benchmarking import (
    evaluate_benchmark_suite,
    normalize_scan_mode,
    normalize_ocr_engine,
    normalize_ocr_mode,
)
from video_service.core.security import (
    validate_url, safe_folder_path, check_upload_size,
    MAX_UPLOAD_BYTES, MAX_UPLOAD_MB,
)
from video_service.core.cleanup import start_cleanup_thread
from video_service.core.abort import mark_job_aborted

logger = logging.getLogger(__name__)

# ── In-memory counters (per-process; reset on restart) ───────────────────────
_counters: dict[str, int] = defaultdict(int)
_start_time = _time.time()

# ── CORS config ──────────────────────────────────────────────────────────────
_CORS_ORIGINS_RAW = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173,http://127.0.0.1:5174"
)
CORS_ORIGINS: list[str] = [o.strip() for o in _CORS_ORIGINS_RAW.split(",") if o.strip()]

NODE_NAME = cluster.self_name
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/video_service_uploads")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/tmp/video_service_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def _round_or_none(value):
    return round(value, 1) if value is not None else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    q = max(0.0, min(1.0, float(quantile)))
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _compute_duration_analytics(points: list[dict], bucket_count: int = 48) -> tuple[dict, list[dict]]:
    durations = [
        float(p.get("duration_seconds"))
        for p in points
        if p.get("duration_seconds") is not None
    ]
    duration_percentiles = {
        "count": len(durations),
        "p50": _round_or_none(_percentile(durations, 0.50)),
        "p90": _round_or_none(_percentile(durations, 0.90)),
        "p95": _round_or_none(_percentile(durations, 0.95)),
        "p99": _round_or_none(_percentile(durations, 0.99)),
    }

    buckets: dict[str, list[float]] = defaultdict(list)
    for point in points:
        completed_at = str(point.get("completed_at") or "")
        duration = point.get("duration_seconds")
        if duration is None:
            continue
        if len(completed_at) >= 13:
            bucket = f"{completed_at[:13]}:00"
        else:
            bucket = completed_at or "unknown"
        buckets[bucket].append(float(duration))

    series = []
    for bucket in sorted(buckets.keys())[-max(1, int(bucket_count)):]:
        values = buckets[bucket]
        series.append(
            {
                "bucket": bucket,
                "count": len(values),
                "p50": _round_or_none(_percentile(values, 0.50)),
                "p90": _round_or_none(_percentile(values, 0.90)),
                "p95": _round_or_none(_percentile(values, 0.95)),
                "p99": _round_or_none(_percentile(values, 0.99)),
            }
        )

    return duration_percentiles, series


def _empty_path_metrics() -> dict:
    return {
        "jobs_with_trace": 0,
        "accepted_paths": [],
        "transit_paths": [],
    }


def _extract_processing_trace(artifacts_json: str | None) -> dict | None:
    if not artifacts_json:
        return None
    try:
        parsed = json.loads(artifacts_json)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    trace = parsed.get("processing_trace")
    return trace if isinstance(trace, dict) else None


def _humanize_attempt_type(value: str) -> str:
    labels = {
        "initial": "Initial Tail Pass",
        "ocr_context_rescue": "OCR Context Rescue",
        "ocr_rescue": "OCR Rescue",
        "express_rescue": "Express Rescue",
        "extended_tail": "Extended Tail",
        "full_video": "Full Video",
        "specificity_search_rescue": "Specificity Search Rescue",
    }
    key = str(value or "").strip().lower()
    return labels.get(key, value.replace("_", " ").title() if value else "Unknown")


def _build_path_metrics(artifact_rows: list) -> dict:
    accepted_path_counts: dict[str, int] = defaultdict(int)
    transit_path_counts: dict[str, int] = defaultdict(int)
    path_titles: dict[str, str] = {}
    jobs_with_trace = 0

    for row in artifact_rows:
        trace = _extract_processing_trace(row["artifacts_json"])
        if not trace:
            continue
        jobs_with_trace += 1
        summary = trace.get("summary")
        if isinstance(summary, dict):
            accepted_type = str(summary.get("accepted_attempt_type") or "").strip()
            if accepted_type:
                accepted_path_counts[accepted_type] += 1
                path_titles.setdefault(accepted_type, _humanize_attempt_type(accepted_type))
        attempts = trace.get("attempts")
        if isinstance(attempts, list):
            seen_attempt_types: set[str] = set()
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                attempt_type = str(attempt.get("attempt_type") or "").strip()
                if not attempt_type or attempt_type in seen_attempt_types:
                    continue
                seen_attempt_types.add(attempt_type)
                transit_path_counts[attempt_type] += 1
                title = str(attempt.get("title") or "").strip()
                path_titles.setdefault(
                    attempt_type,
                    title or _humanize_attempt_type(attempt_type),
                )

    return {
        "jobs_with_trace": jobs_with_trace,
        "accepted_paths": [
            {
                "attempt_type": attempt_type,
                "title": path_titles.get(attempt_type, _humanize_attempt_type(attempt_type)),
                "count": count,
            }
            for attempt_type, count in sorted(
                accepted_path_counts.items(),
                key=lambda item: (-item[1], path_titles.get(item[0], item[0])),
            )
        ],
        "transit_paths": [
            {
                "attempt_type": attempt_type,
                "title": path_titles.get(attempt_type, _humanize_attempt_type(attempt_type)),
                "count": count,
            }
            for attempt_type, count in sorted(
                transit_path_counts.items(),
                key=lambda item: (-item[1], path_titles.get(item[0], item[0])),
            )
        ],
    }


def _merge_analytics_payloads(payloads: list[dict]) -> dict:
    if not payloads:
        return {
            "top_brands": [],
            "categories": [],
            "avg_duration_by_mode": [],
            "avg_duration_by_scan": [],
            "daily_outcomes": [],
            "providers": [],
            "totals": {
                "total": 0,
                "completed": 0,
                "failed": 0,
                "avg_duration": None,
            },
            "duration_percentiles": {
                "count": 0,
                "p50": None,
                "p90": None,
                "p95": None,
                "p99": None,
            },
            "duration_series": [],
            "recent_duration_points": [],
            "path_metrics": _empty_path_metrics(),
        }

    brand_counts: dict[str, int] = defaultdict(int)
    category_counts: dict[str, int] = defaultdict(int)
    provider_counts: dict[str, int] = defaultdict(int)
    daily_counts: dict[tuple[str, str], int] = defaultdict(int)

    mode_accumulator: dict[str, dict[str, float]] = defaultdict(
        lambda: {"duration_sum": 0.0, "count": 0}
    )
    scan_accumulator: dict[str, dict[str, float]] = defaultdict(
        lambda: {"duration_sum": 0.0, "count": 0}
    )

    total_jobs = 0
    total_completed = 0
    total_failed = 0
    completed_duration_sum = 0.0
    completed_duration_count = 0

    merged_duration_points: list[dict] = []
    accepted_path_counts: dict[str, int] = defaultdict(int)
    transit_path_counts: dict[str, int] = defaultdict(int)
    path_titles: dict[str, str] = {}
    jobs_with_trace = 0

    for payload in payloads:
        for row in payload.get("top_brands", []):
            brand = str(row.get("brand") or "").strip()
            if brand:
                brand_counts[brand] += int(row.get("count") or 0)

        for row in payload.get("categories", []):
            category = str(row.get("category") or "").strip()
            if category:
                category_counts[category] += int(row.get("count") or 0)

        for row in payload.get("providers", []):
            provider = str(row.get("provider") or "").strip()
            if provider:
                provider_counts[provider] += int(row.get("count") or 0)

        for row in payload.get("daily_outcomes", []):
            day = str(row.get("day") or "").strip()
            status = str(row.get("status") or "").strip()
            if day and status:
                daily_counts[(day, status)] += int(row.get("count") or 0)

        for row in payload.get("avg_duration_by_mode", []):
            mode = str(row.get("mode") or "unknown").strip() or "unknown"
            avg = row.get("avg_duration")
            count = int(row.get("count") or 0)
            if avg is None or count <= 0:
                continue
            mode_accumulator[mode]["duration_sum"] += float(avg) * count
            mode_accumulator[mode]["count"] += count

        for row in payload.get("avg_duration_by_scan", []):
            scan = str(row.get("scan_mode") or "unknown").strip() or "unknown"
            avg = row.get("avg_duration")
            count = int(row.get("count") or 0)
            if avg is None or count <= 0:
                continue
            scan_accumulator[scan]["duration_sum"] += float(avg) * count
            scan_accumulator[scan]["count"] += count

        totals = payload.get("totals", {})
        total_jobs += int(totals.get("total") or 0)
        total_completed += int(totals.get("completed") or 0)
        total_failed += int(totals.get("failed") or 0)
        avg_duration = totals.get("avg_duration")
        completed_count = int(totals.get("completed") or 0)
        if avg_duration is not None and completed_count > 0:
            completed_duration_sum += float(avg_duration) * completed_count
            completed_duration_count += completed_count

        for point in payload.get("recent_duration_points", []):
            completed_at = str(point.get("completed_at") or "").strip()
            duration = point.get("duration_seconds")
            if completed_at and duration is not None:
                merged_duration_points.append(
                    {
                        "completed_at": completed_at,
                        "duration_seconds": float(duration),
                    }
                )

        path_metrics = payload.get("path_metrics") or {}
        jobs_with_trace += int(path_metrics.get("jobs_with_trace") or 0)
        for row in path_metrics.get("accepted_paths", []):
            attempt_type = str(row.get("attempt_type") or "").strip()
            if not attempt_type:
                continue
            accepted_path_counts[attempt_type] += int(row.get("count") or 0)
            title = str(row.get("title") or "").strip()
            path_titles.setdefault(
                attempt_type,
                title or _humanize_attempt_type(attempt_type),
            )
        for row in path_metrics.get("transit_paths", []):
            attempt_type = str(row.get("attempt_type") or "").strip()
            if not attempt_type:
                continue
            transit_path_counts[attempt_type] += int(row.get("count") or 0)
            title = str(row.get("title") or "").strip()
            path_titles.setdefault(
                attempt_type,
                title or _humanize_attempt_type(attempt_type),
            )

    merged_duration_points.sort(key=lambda row: row["completed_at"])
    merged_duration_points = merged_duration_points[-1200:]
    duration_percentiles, duration_series = _compute_duration_analytics(merged_duration_points)

    return {
        "top_brands": [
            {"brand": brand, "count": count}
            for brand, count in sorted(
                brand_counts.items(), key=lambda item: item[1], reverse=True
            )[:20]
        ],
        "categories": [
            {"category": category, "count": count}
            for category, count in sorted(
                category_counts.items(), key=lambda item: item[1], reverse=True
            )[:25]
        ],
        "avg_duration_by_mode": [
            {
                "mode": mode,
                "avg_duration": _round_or_none(
                    values["duration_sum"] / values["count"]
                    if values["count"] > 0
                    else None
                ),
                "count": int(values["count"]),
            }
            for mode, values in sorted(
                mode_accumulator.items(),
                key=lambda item: item[1]["count"],
                reverse=True,
            )
        ],
        "avg_duration_by_scan": [
            {
                "scan_mode": scan_mode,
                "avg_duration": _round_or_none(
                    values["duration_sum"] / values["count"]
                    if values["count"] > 0
                    else None
                ),
                "count": int(values["count"]),
            }
            for scan_mode, values in sorted(
                scan_accumulator.items(),
                key=lambda item: item[1]["count"],
                reverse=True,
            )
        ],
        "daily_outcomes": [
            {"day": day, "status": status, "count": count}
            for (day, status), count in sorted(daily_counts.items())
        ],
        "providers": [
            {"provider": provider, "count": count}
            for provider, count in sorted(
                provider_counts.items(), key=lambda item: item[1], reverse=True
            )
        ],
        "totals": {
            "total": total_jobs,
            "completed": total_completed,
            "failed": total_failed,
            "avg_duration": _round_or_none(
                completed_duration_sum / completed_duration_count
                if completed_duration_count > 0
                else None
            ),
        },
        "duration_percentiles": duration_percentiles,
        "duration_series": duration_series,
        "recent_duration_points": merged_duration_points,
        "path_metrics": {
            "jobs_with_trace": jobs_with_trace,
            "accepted_paths": [
                {
                    "attempt_type": attempt_type,
                    "title": path_titles.get(attempt_type, _humanize_attempt_type(attempt_type)),
                    "count": count,
                }
                for attempt_type, count in sorted(
                    accepted_path_counts.items(),
                    key=lambda item: (-item[1], path_titles.get(item[0], item[0])),
                )
            ],
            "transit_paths": [
                {
                    "attempt_type": attempt_type,
                    "title": path_titles.get(attempt_type, _humanize_attempt_type(attempt_type)),
                    "count": count,
                }
                for attempt_type, count in sorted(
                    transit_path_counts.items(),
                    key=lambda item: (-item[1], path_titles.get(item[0], item[0])),
                )
            ],
        },
    }


# ── App lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Re-apply logging gates after server bootstrap to suppress noisy libs
    # even if another component touched logger levels/handlers.
    configure_logging(force=True)
    cluster.start_health_checks()
    logger.info("startup: initialising DB (node=%s)", NODE_NAME)
    init_db()
    _recover_stale_jobs_on_startup()
    from video_service.core.stale_recovery import (
        start_stale_recovery_thread,
        stop_stale_recovery,
    )
    start_stale_recovery_thread()
    start_cleanup_thread()

    from video_service.core.watcher import start_watcher, stop_watcher
    watcher_observer = start_watcher()

    # Lazy import avoids pulling worker-side heavy deps during module import.
    from video_service.workers.embedded import (
        start as start_embedded_workers,
        shutdown as shutdown_embedded_workers,
    )
    worker_count = start_embedded_workers()
    if worker_count:
        logger.info("startup: %d embedded worker(s) active", worker_count)

    logger.info("startup: ready (node=%s, cors_origins=%s)", NODE_NAME, CORS_ORIGINS)
    try:
        yield
    finally:
        stop_watcher(watcher_observer)
        stop_stale_recovery()
        shutdown_embedded_workers()
        logger.info("shutdown: node=%s", NODE_NAME)


app = FastAPI(
    title="Video Ad Classification Service",
    version="1.0.0",
    description="HA cluster of workers that classify video advertisements.",
    lifespan=lifespan,
)
app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ── Health & diagnostics ─────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health_check():
    """Local node health. Checks DB is reachable."""
    try:
        with closing(get_db()) as conn:
            conn.execute("SELECT 1").fetchone()
        db_ok = True
    except Exception as exc:
        logger.error("health: DB check failed: %s", exc)
        db_ok = False

    status = "ok" if db_ok else "degraded"
    code   = 200  if db_ok else 503
    return JSONResponse({"status": status, "node": NODE_NAME, "db": db_ok}, status_code=code)


@app.get("/cluster/nodes", tags=["ops"])
async def cluster_nodes():
    """Returns all configured nodes + their last-known health state."""
    return {
        "nodes": cluster.nodes,
        "status": cluster.node_status,
        "self": cluster.self_name,
    }


@app.get("/ollama/models", tags=["ops"])
async def list_ollama_models():
    """Return locally available Ollama models; [] when Ollama is unreachable."""
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{ollama_host}/api/tags", timeout=5.0)
            if res.status_code == 200:
                payload = res.json()
                models = payload.get("models", [])
                return [
                    {
                        "name": model.get("name", ""),
                        "size": model.get("size"),
                        "modified_at": model.get("modified_at"),
                    }
                    for model in models
                    if isinstance(model, dict) and model.get("name")
                ]
    except Exception as exc:
        logger.warning("ollama_models: unreachable: %s", exc)
    return []


def _openai_compat_models_url() -> str:
    compat_url = os.environ.get(
        "OPENAI_COMPAT_URL", "http://localhost:1234/v1/chat/completions"
    ).rstrip("/")
    if compat_url.endswith("/v1/chat/completions"):
        base = compat_url[: -len("/v1/chat/completions")]
    elif compat_url.endswith("/chat/completions"):
        base = compat_url[: -len("/chat/completions")]
    elif "/v1/" in compat_url:
        base = compat_url.split("/v1/")[0]
    else:
        base = compat_url
    return f"{base}/v1/models"


@app.get("/llama-server/models", tags=["ops"])
async def list_llama_server_models():
    """Return models from OpenAI-compatible llama-server /v1/models endpoint."""
    models_url = _openai_compat_models_url()
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(models_url, timeout=5.0)
            if res.status_code == 200:
                payload = res.json()
                data = payload.get("data", []) if isinstance(payload, dict) else []
                return [
                    {
                        "name": model.get("id", ""),
                        "size": model.get("owned_by"),
                        "modified_at": model.get("created"),
                    }
                    for model in data
                    if isinstance(model, dict) and model.get("id")
                ]
    except Exception as exc:
        logger.warning("llama_server_models: unreachable: %s", exc)
    return []


@app.get("/api/v1/models", tags=["ops"])
async def list_provider_models(provider: str = "ollama"):
    provider_name = (provider or "").strip().lower()
    if provider_name == "ollama":
        return await list_ollama_models()
    if provider_name in {"llama-server", "llama server"}:
        return await list_llama_server_models()
    return []


@app.get("/cluster/jobs", tags=["ops"])
async def cluster_jobs():
    """Fan out /admin/jobs to all healthy nodes and merge."""
    if not cluster.enabled:
        return _get_jobs_from_db()

    aggr: list = []
    async with httpx.AsyncClient() as client:
        for node, url in cluster.nodes.items():
            if not cluster.node_status.get(node):
                continue
            try:
                res = await client.get(
                    f"{url}/admin/jobs?internal=1",
                    timeout=cluster.internal_timeout
                )
                if res.status_code == 200:
                    aggr.extend(res.json())
            except Exception as exc:
                logger.warning("cluster_jobs: node %s unreachable: %s", node, exc)

    deduped = _dedupe_jobs_by_id(aggr)
    if len(deduped) != len(aggr):
        logger.info(
            "cluster_jobs: deduped duplicate rows total=%d unique=%d",
            len(aggr),
            len(deduped),
        )
    return deduped


@app.get("/cluster/analytics", tags=["analytics"])
async def cluster_analytics():
    if not cluster.enabled:
        return get_analytics()

    payloads: list[dict] = []
    async with httpx.AsyncClient() as client:
        for node, url in cluster.nodes.items():
            if not cluster.node_status.get(node):
                continue
            try:
                res = await client.get(
                    f"{url}/analytics?internal=1",
                    timeout=cluster.internal_timeout,
                )
                if res.status_code == 200:
                    payload = res.json()
                    if isinstance(payload, dict):
                        payloads.append(payload)
            except Exception as exc:
                logger.warning("cluster_analytics: node %s unreachable: %s", node, exc)

    return _merge_analytics_payloads(payloads)


@app.get("/diagnostics/device", tags=["ops"])
def device_diagnostics():
    return get_diagnostics()


@app.get("/diagnostics/concurrency", tags=["ops"])
def concurrency_diagnostics():
    return get_concurrency_diagnostics()


@app.get("/diagnostics/watcher", tags=["ops"])
def watcher_diagnostics():
    from video_service.core.watcher import get_watcher_diagnostics

    return get_watcher_diagnostics()


def _get_category_mapping_diagnostics():
    return category_mapper.get_diagnostics()


@app.get("/diagnostics/category", tags=["ops"])
def category_mapping_diagnostics():
    return _get_category_mapping_diagnostics()


@app.get("/diagnostics/categories", tags=["ops"])
def category_mapping_diagnostics_legacy():
    return _get_category_mapping_diagnostics()


@app.get("/api/system/profile", tags=["ops"])
def system_profile():
    return get_system_profile()


@app.get("/metrics", tags=["ops"])
def get_metrics():
    """Basic prometheus-style counters (text format available via Accept header)."""
    stats = {}
    with closing(get_db()) as conn:
        for status in ("queued", "processing", "completed", "failed"):
            row = conn.execute(
                "SELECT COUNT(*) as c FROM jobs WHERE status = ?", (status,)
            ).fetchone()
            stats[f"jobs_{status}"] = row["c"]

    return {
        **stats,
        "jobs_submitted_this_process": _counters["submitted"],
        "uptime_seconds": round(_time.time() - _start_time),
        "node": NODE_NAME,
    }


@app.get("/analytics", tags=["analytics"])
def get_analytics():
    with closing(get_db()) as conn:
        top_brands = conn.execute(
            """
            SELECT brand, COUNT(*) as count
            FROM job_stats
            WHERE status = 'completed'
              AND TRIM(COALESCE(brand, '')) != ''
              AND LOWER(TRIM(brand)) NOT IN ('unknown', 'none', 'n/a')
            GROUP BY brand
            ORDER BY count DESC
            LIMIT 20
            """
        ).fetchall()

        categories = conn.execute(
            """
            SELECT category, COUNT(*) as count
            FROM job_stats
            WHERE status = 'completed'
              AND TRIM(COALESCE(category, '')) != ''
              AND LOWER(TRIM(category)) NOT IN ('unknown', 'none', 'n/a')
            GROUP BY category
            ORDER BY count DESC
            LIMIT 25
            """
        ).fetchall()

        avg_duration_by_mode = conn.execute(
            """
            SELECT mode, AVG(duration_seconds) as avg_dur, COUNT(*) as count
            FROM job_stats
            WHERE status = 'completed'
              AND duration_seconds IS NOT NULL
            GROUP BY mode
            ORDER BY count DESC
            """
        ).fetchall()

        avg_duration_by_scan = conn.execute(
            """
            SELECT scan_mode, AVG(duration_seconds) as avg_dur, COUNT(*) as count
            FROM job_stats
            WHERE status = 'completed'
              AND duration_seconds IS NOT NULL
            GROUP BY scan_mode
            ORDER BY count DESC
            """
        ).fetchall()

        daily_outcomes = conn.execute(
            """
            SELECT DATE(completed_at) as day, status, COUNT(*) as count
            FROM job_stats
            GROUP BY day, status
            ORDER BY day
            """
        ).fetchall()

        providers = conn.execute(
            """
            SELECT provider, COUNT(*) as count
            FROM job_stats
            WHERE status = 'completed'
              AND TRIM(COALESCE(provider, '')) != ''
            GROUP BY provider
            ORDER BY count DESC
            """
        ).fetchall()

        totals = conn.execute(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed,
                   SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed,
                   AVG(CASE WHEN status='completed' THEN duration_seconds END) as avg_duration
            FROM job_stats
            """
        ).fetchone()

        recent_duration_rows = conn.execute(
            """
            SELECT completed_at, duration_seconds
            FROM job_stats
            WHERE status = 'completed'
              AND duration_seconds IS NOT NULL
              AND completed_at IS NOT NULL
            ORDER BY completed_at DESC
            LIMIT 600
            """
        ).fetchall()
        try:
            artifact_rows = conn.execute(
                """
                SELECT artifacts_json
                FROM jobs
                WHERE status = 'completed'
                  AND TRIM(COALESCE(artifacts_json, '')) != ''
                """
            ).fetchall()
        except Exception:
            artifact_rows = []
    recent_duration_points = [
        {
            "completed_at": row["completed_at"],
            "duration_seconds": float(row["duration_seconds"]),
        }
        for row in reversed(recent_duration_rows)
        if row["completed_at"] and row["duration_seconds"] is not None
    ]
    duration_percentiles, duration_series = _compute_duration_analytics(
        recent_duration_points
    )
    path_metrics = _build_path_metrics(artifact_rows)

    return {
        "top_brands": [{"brand": r["brand"], "count": r["count"]} for r in top_brands],
        "categories": [{"category": r["category"], "count": r["count"]} for r in categories],
        "avg_duration_by_mode": [
            {
                "mode": r["mode"] or "unknown",
                "avg_duration": _round_or_none(r["avg_dur"]),
                "count": r["count"],
            }
            for r in avg_duration_by_mode
        ],
        "avg_duration_by_scan": [
            {
                "scan_mode": r["scan_mode"] or "unknown",
                "avg_duration": _round_or_none(r["avg_dur"]),
                "count": r["count"],
            }
            for r in avg_duration_by_scan
        ],
        "daily_outcomes": [
            {"day": r["day"], "status": r["status"], "count": r["count"]}
            for r in daily_outcomes
        ],
        "providers": [{"provider": r["provider"], "count": r["count"]} for r in providers],
        "totals": {
            "total": totals["total"] if totals else 0,
            "completed": totals["completed"] if totals and totals["completed"] is not None else 0,
            "failed": totals["failed"] if totals and totals["failed"] is not None else 0,
            "avg_duration": _round_or_none(totals["avg_duration"] if totals else None),
        },
        "duration_percentiles": duration_percentiles,
        "duration_series": duration_series,
        "recent_duration_points": recent_duration_points,
        "path_metrics": path_metrics,
    }


async def _resolve_benchmark_provider_models(
    requested_providers: list[str] | None = None,
    requested_models: list[str] | None = None,
) -> list[tuple[str, str]]:
    providers = [p.strip() for p in (requested_providers or []) if p and p.strip()]
    models = [m.strip() for m in (requested_models or []) if m and m.strip()]

    if providers and models:
        return [(provider, model) for provider in providers for model in models]

    pairs: list[tuple[str, str]] = []

    use_provider = lambda name: (not providers) or any(
        p.lower() == name.lower() for p in providers
    )

    if use_provider("Ollama"):
        try:
            ollama_models = await list_ollama_models()
        except Exception:
            ollama_models = []
        ollama_names = [m.get("name") for m in ollama_models if isinstance(m, dict) and m.get("name")]
        if not ollama_names:
            ollama_names = [os.environ.get("DEFAULT_MODEL", "qwen3-vl:8b-instruct")]
        for model in ollama_names[:3]:
            pairs.append(("Ollama", str(model)))

    if use_provider("LM Studio"):
        pairs.append(("LM Studio", os.environ.get("BENCHMARK_LM_STUDIO_MODEL", "local-model")))

    if use_provider("Llama Server"):
        try:
            llama_models = await list_llama_server_models()
        except Exception:
            llama_models = []
        llama_names = [
            m.get("name")
            for m in llama_models
            if isinstance(m, dict) and m.get("name")
        ]
        if not llama_names:
            llama_names = [
                os.environ.get(
                    "BENCHMARK_LLAMA_SERVER_MODEL",
                    os.environ.get("BENCHMARK_LM_STUDIO_MODEL", "local-model"),
                )
            ]
        for model in llama_names[:3]:
            pairs.append(("Llama Server", str(model)))

    if use_provider("Gemini CLI"):
        pairs.append(("Gemini CLI", os.environ.get("BENCHMARK_GEMINI_MODEL", "Gemini CLI Default")))

    if models:
        pairs = [(provider, model) for provider, _ in pairs for model in models]

    deduped: list[tuple[str, str]] = []
    seen = set()
    for provider, model in pairs:
        key = (provider.strip(), model.strip())
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _safe_json_list(raw: str | None) -> list:
    try:
        parsed = json.loads(raw or "[]")
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _safe_json_object(raw: str | None) -> dict:
    try:
        parsed = json.loads(raw or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _serialize_benchmark_truth_row(row) -> dict:
    expected_categories = _safe_json_list(row["expected_categories_json"])
    expected_category = str(row["expected_category"] or "").strip()
    if expected_category and expected_category not in expected_categories:
        expected_categories = [expected_category, *expected_categories]
    return {
        "test_id": row["id"],
        "truth_id": row["id"],
        "suite_id": row["suite_id"] or "",
        "name": row["name"],
        "source_url": row["video_url"],
        "video_url": row["video_url"],
        "expected_ocr_text": row["expected_ocr_text"] or "",
        "expected_categories": expected_categories,
        "expected_category": expected_category,
        "expected_brand": row["expected_brand"] or "",
        "expected_confidence": row["expected_confidence"],
        "expected_reasoning": row["expected_reasoning"] or "",
        "metadata": _safe_json_object(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _load_suite_tests(conn, suite_id: str, suite_truth_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, name, suite_id, video_url, expected_ocr_text, expected_categories_json,
               expected_brand, expected_category, expected_confidence, expected_reasoning,
               metadata_json, created_at, updated_at
        FROM benchmark_truth
        WHERE suite_id = ?
        ORDER BY created_at ASC
        """,
        (suite_id,),
    ).fetchall()
    if rows:
        return [_serialize_benchmark_truth_row(row) for row in rows]

    if not suite_truth_id:
        return []

    row = conn.execute(
        """
        SELECT id, name, suite_id, video_url, expected_ocr_text, expected_categories_json,
               expected_brand, expected_category, expected_confidence, expected_reasoning,
               metadata_json, created_at, updated_at
        FROM benchmark_truth
        WHERE id = ?
        """,
        (suite_truth_id,),
    ).fetchone()
    return [_serialize_benchmark_truth_row(row)] if row else []


@app.post("/api/benchmark/truths", tags=["benchmark"])
def create_benchmark_truth(body: BenchmarkTruthCreateRequest):
    truth_id = f"truth-{uuid.uuid4()}"
    expected_category = str(body.expected_category or "").strip()
    expected_categories = [c.strip() for c in (body.expected_categories or []) if c and c.strip()]
    if expected_category and expected_category not in expected_categories:
        expected_categories = [expected_category, *expected_categories]
    with closing(get_db()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO benchmark_truth (
                    id, name, suite_id, video_url, expected_ocr_text, expected_categories_json,
                    expected_brand, expected_category, expected_confidence, expected_reasoning, metadata_json
                ) VALUES (?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    truth_id,
                    body.name.strip(),
                    body.video_url.strip(),
                    body.expected_ocr_text or "",
                    json.dumps(expected_categories),
                    body.expected_brand or "",
                    expected_category,
                    body.expected_confidence,
                    body.expected_reasoning or "",
                    json.dumps(body.metadata or {}),
                ),
            )
    return {"truth_id": truth_id}


@app.get("/api/benchmark/truths", tags=["benchmark"])
def list_benchmark_truths():
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT id, name, suite_id, video_url, expected_ocr_text, expected_categories_json,
                   expected_brand, expected_category, expected_confidence, expected_reasoning,
                   metadata_json, created_at, updated_at
            FROM benchmark_truth
            WHERE COALESCE(suite_id, '') = ''
            ORDER BY created_at DESC
            """
        ).fetchall()
    return {"truths": [_serialize_benchmark_truth_row(row) for row in rows]}


@app.get("/api/benchmark/truths/{truth_id}", tags=["benchmark"])
def get_benchmark_truth(truth_id: str):
    with closing(get_db()) as conn:
        row = conn.execute(
            """
            SELECT id, name, suite_id, video_url, expected_ocr_text, expected_categories_json,
                   expected_brand, expected_category, expected_confidence, expected_reasoning,
                   metadata_json, created_at, updated_at
            FROM benchmark_truth
            WHERE id = ?
            """,
            (truth_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Benchmark truth not found")
    return _serialize_benchmark_truth_row(row)


@app.delete("/api/benchmark/truths/{truth_id}", tags=["benchmark"])
def delete_benchmark_truth(truth_id: str):
    """Delete a golden-truth record.  Rejects deletion if a running suite references it."""
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT id FROM benchmark_truth WHERE id = ?", (truth_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Benchmark truth not found")
        # Prevent deletion if an active suite points to this truth
        active = conn.execute(
            """
            SELECT id FROM benchmark_suites
            WHERE truth_id = ? AND status IN ('running', 'queued')
            LIMIT 1
            """,
            (truth_id,),
        ).fetchone()
        if active:
            raise HTTPException(
                status_code=409,
                detail="Cannot delete: an active benchmark suite references this truth",
            )
        with conn:
            conn.execute("DELETE FROM benchmark_truth WHERE id = ?", (truth_id,))
    return {"deleted": truth_id}


@app.post("/api/benchmark/run", tags=["benchmark"])
async def run_benchmark_suite(body: BenchmarkRunRequest):
    with closing(get_db()) as conn:
        truth = conn.execute(
            """
            SELECT id, name, video_url, expected_ocr_text, expected_categories_json,
                   expected_brand, expected_category, expected_confidence, expected_reasoning, metadata_json
            FROM benchmark_truth
            WHERE id = ?
            """,
            (body.truth_id,),
        ).fetchone()
    if not truth:
        raise HTTPException(status_code=404, detail="Benchmark truth not found")

    # If explicit combos are provided, use them directly; otherwise auto-resolve.
    if body.model_combos:
        provider_model_pairs = [
            (c.get("provider", ""), c.get("model", ""))
            for c in body.model_combos
            if c.get("provider") and c.get("model")
        ]
    else:
        provider_model_pairs = await _resolve_benchmark_provider_models(
            requested_providers=body.providers,
            requested_models=body.models,
        )
    if not provider_model_pairs:
        raise HTTPException(status_code=400, detail="No provider/model pairs available for benchmark")

    scan_options = ["tail", "full"]
    ocr_engines = ["easyocr", "microsoft"]
    ocr_modes = ["fast", "detailed"]

    permutations = []
    for scan in scan_options:
        for ocr_engine in ocr_engines:
            for ocr_mode in ocr_modes:
                for provider, model in provider_model_pairs:
                    permutations.append(
                        {
                            "scan_strategy": scan,
                            "ocr_engine": ocr_engine,
                            "ocr_mode": ocr_mode,
                            "provider": provider,
                            "model_name": model,
                        }
                    )

    suite_id = f"suite-{uuid.uuid4()}"
    suite_name = f"Suite for {truth['name']}"
    suite_description = f"Generated from truth {truth['id']}"
    suite_truth_id = f"test-{uuid.uuid4()}"
    matrix_payload = {
        "scan_strategy": scan_options,
        "ocr_engine": ocr_engines,
        "ocr_mode": ocr_modes,
        "provider_model_pairs": provider_model_pairs,
        "permutations": len(permutations),
    }

    with closing(get_db()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO benchmark_suites (
                    id, truth_id, name, description, status, matrix_json, created_by, total_jobs, completed_jobs, failed_jobs
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
                """,
                (
                    suite_id,
                    suite_truth_id,
                    suite_name,
                    suite_description,
                    "running",
                    json.dumps(matrix_payload),
                    "api",
                    len(permutations),
                ),
            )
            conn.execute(
                """
                INSERT INTO benchmark_truth (
                    id, name, suite_id, video_url, expected_ocr_text, expected_categories_json,
                    expected_brand, expected_category, expected_confidence, expected_reasoning, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    suite_truth_id,
                    truth["name"],
                    suite_id,
                    truth["video_url"],
                    truth["expected_ocr_text"] or "",
                    truth["expected_categories_json"] or "[]",
                    truth["expected_brand"] or "",
                    truth["expected_category"] or "",
                    truth["expected_confidence"],
                    truth["expected_reasoning"] or "",
                    truth["metadata_json"] or "{}",
                ),
            )

    benchmark_jobs = []
    for permutation in permutations:
        settings = JobSettings(
            categories=body.categories or "",
            provider=permutation["provider"],
            model_name=permutation["model_name"],
            ocr_engine=normalize_ocr_engine(permutation["ocr_engine"]),
            ocr_mode=normalize_ocr_mode(permutation["ocr_mode"]),
            scan_mode=normalize_scan_mode(permutation["scan_strategy"]),
            override=False,
            express_mode=body.express_mode,
            enable_search=True,
            enable_web_search=True,
            enable_agentic_search=True,
            enable_vision_board=True,
            enable_llm_frame=True,
            context_size=8192,
        )
        job_id = _create_job(
            JobMode.benchmark.value,
            settings,
            url=truth["video_url"],
            benchmark_suite_id=suite_id,
            benchmark_truth_id=suite_truth_id,
            benchmark_params=permutation,
        )
        benchmark_jobs.append(
            {
                "job_id": job_id,
                "params": permutation,
            }
        )

    logger.info(
        "benchmark_suite_started: suite_id=%s truth_id=%s jobs=%d",
        suite_id,
        suite_truth_id,
        len(benchmark_jobs),
    )

    return {
        "suite_id": suite_id,
        "truth_id": suite_truth_id,
        "source_truth_id": truth["id"],
        "jobs_enqueued": len(benchmark_jobs),
        "matrix": matrix_payload,
        "jobs": benchmark_jobs,
    }


@app.get("/api/benchmark/suites", tags=["benchmark"])
def list_benchmark_suites():
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.truth_id, s.name, s.description, s.status, s.total_jobs, s.completed_jobs, s.failed_jobs,
                   s.evaluated_at, s.created_at, s.updated_at, t.name as truth_name,
                   (
                     SELECT COUNT(*)
                     FROM benchmark_truth bt
                     WHERE bt.suite_id = s.id OR bt.id = s.truth_id
                   ) AS test_count
            FROM benchmark_suites s
            LEFT JOIN benchmark_truth t ON t.id = s.truth_id
            ORDER BY s.created_at DESC
            """
        ).fetchall()
    return {
        "suites": [
            {
                "suite_id": row["id"],
                "truth_id": row["truth_id"],
                "truth_name": row["truth_name"],
                "name": row["name"] or "",
                "description": row["description"] or "",
                "status": row["status"],
                "total_jobs": row["total_jobs"],
                "completed_jobs": row["completed_jobs"],
                "failed_jobs": row["failed_jobs"],
                "test_count": row["test_count"] or 0,
                "evaluated_at": row["evaluated_at"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
    }


@app.get("/api/benchmark/suites/{suite_id}", tags=["benchmark"])
def get_benchmark_suite(suite_id: str):
    with closing(get_db()) as conn:
        row = conn.execute(
            """
            SELECT s.*, t.name as truth_name, t.video_url
            FROM benchmark_suites s
            LEFT JOIN benchmark_truth t ON t.id = s.truth_id
            WHERE s.id = ?
            """,
            (suite_id,),
        ).fetchone()
        tests = _load_suite_tests(conn, suite_id, row["truth_id"] if row else "")
    if not row:
        raise HTTPException(status_code=404, detail="Benchmark suite not found")
    return {
        "suite_id": row["id"],
        "truth_id": row["truth_id"],
        "truth_name": row["truth_name"],
        "video_url": row["video_url"],
        "name": row["name"] or "",
        "description": row["description"] or "",
        "status": row["status"],
        "matrix": json.loads(row["matrix_json"] or "{}"),
        "total_jobs": row["total_jobs"],
        "completed_jobs": row["completed_jobs"],
        "failed_jobs": row["failed_jobs"],
        "tests": tests,
        "test_count": len(tests),
        "evaluated_at": row["evaluated_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@app.put("/benchmarks/suites/{suite_id}", tags=["benchmark"])
def update_benchmark_suite(suite_id: str, body: BenchmarkSuiteUpdateRequest):
    with closing(get_db()) as conn:
        with conn:
            existing = conn.execute(
                "SELECT id FROM benchmark_suites WHERE id = ?",
                (suite_id,),
            ).fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Benchmark suite not found")
            conn.execute(
                """
                UPDATE benchmark_suites
                SET name = ?, description = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (body.name.strip(), body.description.strip(), suite_id),
            )
        row = conn.execute(
            """
            SELECT s.id, s.truth_id, s.name, s.description, s.status, s.total_jobs, s.completed_jobs, s.failed_jobs,
                   s.evaluated_at, s.created_at, s.updated_at, t.name as truth_name
            FROM benchmark_suites s
            LEFT JOIN benchmark_truth t ON t.id = s.truth_id
            WHERE s.id = ?
            """,
            (suite_id,),
        ).fetchone()
    return {
        "suite_id": row["id"],
        "truth_id": row["truth_id"],
        "truth_name": row["truth_name"],
        "name": row["name"] or "",
        "description": row["description"] or "",
        "status": row["status"],
        "total_jobs": row["total_jobs"],
        "completed_jobs": row["completed_jobs"],
        "failed_jobs": row["failed_jobs"],
        "evaluated_at": row["evaluated_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@app.delete("/benchmarks/suites/{suite_id}", tags=["benchmark"])
def delete_benchmark_suite(suite_id: str):
    with closing(get_db()) as conn:
        with conn:
            suite = conn.execute(
                "SELECT id, truth_id FROM benchmark_suites WHERE id = ?",
                (suite_id,),
            ).fetchone()
            if not suite:
                raise HTTPException(status_code=404, detail="Benchmark suite not found")

            deleted_results = conn.execute(
                "DELETE FROM benchmark_result WHERE suite_id = ?",
                (suite_id,),
            ).rowcount
            deleted_jobs = conn.execute(
                "DELETE FROM jobs WHERE benchmark_suite_id = ?",
                (suite_id,),
            ).rowcount
            deleted_tests = conn.execute(
                "DELETE FROM benchmark_truth WHERE suite_id = ? OR id = ?",
                (suite_id, suite["truth_id"] or ""),
            ).rowcount
            conn.execute(
                "DELETE FROM benchmark_suites WHERE id = ?",
                (suite_id,),
            )

    return {
        "status": "deleted",
        "suite_id": suite_id,
        "deleted_tests": deleted_tests,
        "deleted_jobs": deleted_jobs,
        "deleted_results": deleted_results,
    }


@app.put("/benchmarks/tests/{test_id}", tags=["benchmark"])
def update_benchmark_test(test_id: str, body: BenchmarkTestUpdateRequest):
    with closing(get_db()) as conn:
        with conn:
            row = conn.execute(
                """
                SELECT id, name, suite_id, video_url, expected_ocr_text, expected_categories_json,
                       expected_brand, expected_category, expected_confidence, expected_reasoning,
                       metadata_json, created_at, updated_at
                FROM benchmark_truth
                WHERE id = ?
                """,
                (test_id,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Benchmark test not found")

            expected_categories = _safe_json_list(row["expected_categories_json"])
            if body.expected_categories is not None:
                expected_categories = [c.strip() for c in body.expected_categories if c and c.strip()]
            expected_category = (
                body.expected_category.strip()
                if isinstance(body.expected_category, str)
                else str(row["expected_category"] or "").strip()
            )
            if expected_category:
                expected_categories = [c for c in expected_categories if c != expected_category]
                expected_categories = [expected_category, *expected_categories]

            conn.execute(
                """
                UPDATE benchmark_truth
                SET video_url = ?,
                    expected_ocr_text = ?,
                    expected_categories_json = ?,
                    expected_brand = ?,
                    expected_category = ?,
                    expected_confidence = ?,
                    expected_reasoning = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    body.source_url.strip() if isinstance(body.source_url, str) else row["video_url"],
                    body.expected_ocr_text if body.expected_ocr_text is not None else row["expected_ocr_text"],
                    json.dumps(expected_categories),
                    body.expected_brand if body.expected_brand is not None else row["expected_brand"],
                    expected_category,
                    body.expected_confidence if body.expected_confidence is not None else row["expected_confidence"],
                    body.expected_reasoning if body.expected_reasoning is not None else row["expected_reasoning"],
                    test_id,
                ),
            )
        updated = conn.execute(
            """
            SELECT id, name, suite_id, video_url, expected_ocr_text, expected_categories_json,
                   expected_brand, expected_category, expected_confidence, expected_reasoning,
                   metadata_json, created_at, updated_at
            FROM benchmark_truth
            WHERE id = ?
            """,
            (test_id,),
        ).fetchone()
    return _serialize_benchmark_truth_row(updated)


@app.delete("/benchmarks/tests/{test_id}", tags=["benchmark"])
def delete_benchmark_test(test_id: str):
    with closing(get_db()) as conn:
        with conn:
            row = conn.execute(
                "SELECT id, suite_id FROM benchmark_truth WHERE id = ?",
                (test_id,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Benchmark test not found")

            suite_id = row["suite_id"] or ""
            conn.execute(
                "UPDATE benchmark_suites SET truth_id = '' WHERE truth_id = ?",
                (test_id,),
            )
            conn.execute(
                "DELETE FROM benchmark_result WHERE job_id IN (SELECT id FROM jobs WHERE benchmark_truth_id = ?)",
                (test_id,),
            )
            deleted_jobs = conn.execute(
                "DELETE FROM jobs WHERE benchmark_truth_id = ?",
                (test_id,),
            ).rowcount
            conn.execute(
                "DELETE FROM benchmark_truth WHERE id = ?",
                (test_id,),
            )
            if suite_id:
                conn.execute(
                    """
                    UPDATE benchmark_suites
                    SET total_jobs = (
                        SELECT COUNT(*) FROM jobs WHERE benchmark_suite_id = ?
                    ),
                    completed_jobs = (
                        SELECT COUNT(*) FROM jobs WHERE benchmark_suite_id = ? AND status = 'completed'
                    ),
                    failed_jobs = (
                        SELECT COUNT(*) FROM jobs WHERE benchmark_suite_id = ? AND status = 'failed'
                    ),
                    updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (suite_id, suite_id, suite_id, suite_id),
                )

    return {"status": "deleted", "test_id": test_id, "suite_id": suite_id, "deleted_jobs": deleted_jobs}


@app.get("/api/benchmark/suites/{suite_id}/results", tags=["benchmark"])
def get_benchmark_suite_results(suite_id: str):
    evaluation = evaluate_benchmark_suite(suite_id)
    if not evaluation.get("ok"):
        raise HTTPException(status_code=404, detail=evaluation.get("error", "benchmark_evaluation_failed"))

    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT job_id, duration_seconds, classification_accuracy, ocr_accuracy, composite_accuracy, params_json
            FROM benchmark_result
            WHERE suite_id = ?
            ORDER BY created_at ASC
            """,
            (suite_id,),
        ).fetchall()
        suite = conn.execute(
            "SELECT status, total_jobs, completed_jobs, failed_jobs FROM benchmark_suites WHERE id = ?",
            (suite_id,),
        ).fetchone()

    scatter_points = []
    for row in rows:
        params = json.loads(row["params_json"] or "{}")
        label = (
            f"{params.get('provider', 'unknown')} + "
            f"{params.get('model_name', 'model')} + "
            f"{params.get('ocr_engine', 'ocr')} + "
            f"{params.get('scan_strategy', 'scan')}"
        )
        composite_pct = round(float(row["composite_accuracy"] or 0.0) * 100.0, 2)
        scatter_points.append(
            {
                "job_id": row["job_id"],
                "x_duration_seconds": row["duration_seconds"],
                "y_composite_accuracy_pct": composite_pct,
                "classification_accuracy": row["classification_accuracy"],
                "ocr_accuracy": row["ocr_accuracy"],
                "params": params,
                "label": label,
            }
        )

    return {
        "suite_id": suite_id,
        "status": suite["status"] if suite else evaluation.get("status"),
        "total_jobs": suite["total_jobs"] if suite else evaluation.get("total_jobs"),
        "completed_jobs": suite["completed_jobs"] if suite else evaluation.get("completed_jobs"),
        "failed_jobs": suite["failed_jobs"] if suite else evaluation.get("failed_jobs"),
        "points": scatter_points,
        "path_metrics": evaluation.get("path_metrics", {}),
    }


# ── Internal helpers ─────────────────────────────────────────────────────────

def _create_job(
    mode: str,
    settings: JobSettings,
    url: str = None,
    *,
    benchmark_suite_id: str | None = None,
    benchmark_truth_id: str | None = None,
    benchmark_params: dict | None = None,
) -> str:
    job_id = f"{NODE_NAME}-{uuid.uuid4()}"
    benchmark_params_json = json.dumps(benchmark_params or {})
    with closing(get_db()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, status, stage, stage_detail, mode, settings, url, events,
                    benchmark_suite_id, benchmark_truth_id, benchmark_params_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    "queued",
                    "queued",
                    "waiting for worker claim",
                    mode,
                    settings.model_dump_json(),
                    url,
                    "[]",
                    benchmark_suite_id or "",
                    benchmark_truth_id or "",
                    benchmark_params_json,
                ),
            )
    _counters["submitted"] += 1
    logger.info(
        "job_created: job_id=%s mode=%s url=%s benchmark_suite_id=%s",
        job_id,
        mode,
        url,
        benchmark_suite_id or "-",
    )
    return job_id


def _default_job_artifacts(job_id: str) -> dict:
    return {
        "latest_frames": [],
        "per_frame_vision": [],
        "ocr_text": {
            "text": "",
            "lines": [],
            "url": None,
        },
        "vision_board": {
            "image_url": None,
            "plot_url": None,
            "top_matches": [],
            "metadata": {},
            "vector_plot": None,
        },
        "category_mapper": {
            "category": "",
            "category_id": "",
            "method": "",
            "score": None,
            "confidence": None,
            "vector_plot": None,
        },
        "processing_trace": None,
        "extras": {
            "events_url": f"/jobs/{job_id}/events",
        },
    }


def _normalize_job_artifacts(job_id: str, artifacts: Optional[dict]) -> dict:
    payload = _default_job_artifacts(job_id)
    if not isinstance(artifacts, dict):
        return payload

    payload["latest_frames"] = artifacts.get("latest_frames") or []
    per_frame_vision = artifacts.get("per_frame_vision")
    payload["per_frame_vision"] = per_frame_vision if isinstance(per_frame_vision, list) else []

    ocr_payload = artifacts.get("ocr_text")
    if isinstance(ocr_payload, dict):
        payload["ocr_text"]["text"] = ocr_payload.get("text") or ""
        payload["ocr_text"]["lines"] = ocr_payload.get("lines") or []
        payload["ocr_text"]["url"] = ocr_payload.get("url")
    elif isinstance(ocr_payload, str):
        payload["ocr_text"]["text"] = ocr_payload
        payload["ocr_text"]["lines"] = [line for line in ocr_payload.splitlines() if line.strip()]

    vision_payload = artifacts.get("vision_board")
    if isinstance(vision_payload, dict):
        payload["vision_board"]["image_url"] = vision_payload.get("image_url")
        payload["vision_board"]["plot_url"] = vision_payload.get("plot_url")
        payload["vision_board"]["top_matches"] = vision_payload.get("top_matches") or []
        payload["vision_board"]["metadata"] = vision_payload.get("metadata") or {}
        payload["vision_board"]["vector_plot"] = vision_payload.get("vector_plot")

    mapper_payload = artifacts.get("category_mapper")
    if isinstance(mapper_payload, dict):
        payload["category_mapper"]["category"] = mapper_payload.get("category") or ""
        payload["category_mapper"]["category_id"] = (
            str(mapper_payload.get("category_id") or "")
        )
        payload["category_mapper"]["method"] = mapper_payload.get("method") or ""
        payload["category_mapper"]["score"] = mapper_payload.get("score")
        payload["category_mapper"]["confidence"] = mapper_payload.get("confidence")
        payload["category_mapper"]["vector_plot"] = mapper_payload.get("vector_plot")

    if isinstance(artifacts.get("processing_trace"), dict):
        payload["processing_trace"] = artifacts.get("processing_trace")

    extras = artifacts.get("extras")
    if isinstance(extras, dict):
        payload["extras"].update(extras)

    for key, value in artifacts.items():
        if key not in payload:
            payload[key] = value
    return payload


def _build_job_explanation(job_id: str, job_row, artifacts: dict, result_payload: list | None, events: list[str]) -> dict:
    trace = artifacts.get("processing_trace")
    attempts = trace.get("attempts") if isinstance(trace, dict) else []
    summary = trace.get("summary") if isinstance(trace, dict) else {}
    if not isinstance(attempts, list):
        attempts = []
    if not isinstance(summary, dict):
        summary = {}

    first_result = result_payload[0] if isinstance(result_payload, list) and result_payload else {}
    if not isinstance(first_result, dict):
        first_result = {}

    category_mapper = artifacts.get("category_mapper") if isinstance(artifacts.get("category_mapper"), dict) else {}
    ocr_payload = artifacts.get("ocr_text") if isinstance(artifacts.get("ocr_text"), dict) else {}
    latest_frames = artifacts.get("latest_frames") if isinstance(artifacts.get("latest_frames"), list) else []
    row_brand = job_row["brand"] if "brand" in job_row.keys() else ""
    row_category = job_row["category"] if "category" in job_row.keys() else ""
    row_category_id = job_row["category_id"] if "category_id" in job_row.keys() else ""

    if attempts:
        headline = summary.get("headline") or "Processing explanation generated from structured execution trace."
    elif result_payload:
        headline = "This job completed before structured explanation traces were available."
    else:
        headline = "No structured explanation is available for this job yet."

    accepted_attempt = next(
        (
            attempt
            for attempt in reversed(attempts)
            if isinstance(attempt, dict) and attempt.get("status") == "accepted"
        ),
        None,
    )
    accepted_result = accepted_attempt.get("result") if isinstance(accepted_attempt, dict) else {}
    if not isinstance(accepted_result, dict):
        accepted_result = {}

    return {
        "job_id": job_id,
        "mode": job_row["mode"] if "mode" in job_row.keys() else None,
        "status": job_row["status"],
        "stage": job_row["stage"] if "stage" in job_row.keys() else None,
        "stage_detail": job_row["stage_detail"] if "stage_detail" in job_row.keys() else None,
        "summary": {
            "headline": headline,
            "attempt_count": summary.get("attempt_count", len(attempts)),
            "retry_count": summary.get("retry_count", max(0, len(attempts) - 1)),
            "accepted_attempt_type": summary.get("accepted_attempt_type", ""),
            "trigger_reason": summary.get("trigger_reason", ""),
        },
        "attempts": attempts,
        "final": {
            "brand": first_result.get("Brand") or first_result.get("brand") or row_brand,
            "category": first_result.get("Category") or first_result.get("category") or row_category,
            "category_id": first_result.get("Category ID") or first_result.get("category_id") or row_category_id,
            "confidence": first_result.get("Confidence") or first_result.get("confidence"),
            "mapper_method": category_mapper.get("method") if isinstance(category_mapper, dict) else "",
            "mapper_score": category_mapper.get("score") if isinstance(category_mapper, dict) else None,
            "brand_ambiguity_flag": bool(accepted_result.get("brand_ambiguity_flag", False)),
            "brand_ambiguity_reason": str(accepted_result.get("brand_ambiguity_reason") or ""),
            "brand_ambiguity_resolved": bool(accepted_result.get("brand_ambiguity_resolved", False)),
            "brand_disambiguation_reason": str(accepted_result.get("brand_disambiguation_reason") or ""),
            "brand_evidence_strength": str(accepted_result.get("brand_evidence_strength") or ""),
        },
        "evidence": {
            "ocr_excerpt": " ".join((ocr_payload.get("text") or "").split())[:400],
            "latest_frames": latest_frames[:4],
            "event_count": len(events),
            "recent_events": events[-8:],
        },
    }


def _append_recovery_event(conn, job_id: str, message: str) -> None:
    row = conn.execute("SELECT events FROM jobs WHERE id = ?", (job_id,)).fetchone()
    events: list[str] = []
    if row and row["events"]:
        try:
            parsed = json.loads(row["events"])
            if isinstance(parsed, list):
                events = [str(item) for item in parsed]
        except Exception:
            events = []
    events.append(f"{datetime.now(timezone.utc).isoformat()} recovery: {message}")
    events = events[-400:]
    conn.execute(
        "UPDATE jobs SET events = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (json.dumps(events), job_id),
    )


def _recover_stale_jobs_on_startup() -> int:
    """Reset any jobs left in processing state from a previous crash/restart."""
    with closing(get_db()) as conn:
        candidates = conn.execute(
            "SELECT id FROM jobs WHERE status = 'processing'"
        ).fetchall()
        job_ids = [row["id"] for row in candidates]
        if not job_ids:
            return 0

        placeholders = ",".join("?" for _ in job_ids)
        with conn:
            conn.execute(
                f"""
                UPDATE jobs
                SET status = 'queued',
                    stage = 'queued',
                    stage_detail = 'recovered after restart',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id IN ({placeholders})
                """,
                tuple(job_ids),
            )
            for job_id in job_ids:
                _append_recovery_event(conn, job_id, "recovered after process restart")

    logger.info(
        "startup_recovery: reset %d orphaned processing jobs to queued",
        len(job_ids),
    )
    return len(job_ids)


def _resolve_enable_web_search(
    enable_search: bool,
    enable_web_search: Optional[bool],
    enable_agentic_search: Optional[bool],
) -> bool:
    if enable_web_search is not None:
        return bool(enable_web_search)
    if enable_agentic_search is not None:
        return bool(enable_agentic_search)
    return bool(enable_search)


def _resolve_vision_flags(
    enable_vision_board: Optional[bool],
    enable_llm_frame: Optional[bool],
    enable_vision_legacy: Optional[bool],
) -> tuple[bool, bool]:
    board = enable_vision_board
    llm_frame = enable_llm_frame

    # Backward compatibility for legacy clients still posting `enable_vision`.
    if board is None and enable_vision_legacy is not None:
        board = bool(enable_vision_legacy)
    if llm_frame is None and enable_vision_legacy is not None:
        llm_frame = bool(enable_vision_legacy)

    if board is None:
        board = True
    if llm_frame is None:
        llm_frame = True
    return bool(board), bool(llm_frame)


async def _proxy_request(request: Request, target_url: str) -> Response:
    """Forward a request to another cluster node, adding ?internal=1."""
    async with httpx.AsyncClient() as client:
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)
        try:
            res = await client.request(
                method=request.method,
                url=f"{target_url}{request.url.path}?internal=1",
                content=body,
                headers=headers,
                timeout=cluster.internal_timeout,
            )
            return Response(content=res.content, status_code=res.status_code, headers=dict(res.headers))
        except Exception as exc:
            logger.error("proxy error → %s: %s", target_url, exc)
            raise HTTPException(status_code=503, detail=f"Proxy error: {exc}")


async def _maybe_proxy(req: Request, job_id: str) -> Response | None:
    """If the job belongs to another node, proxy the request there."""
    if req.query_params.get("internal"):
        return None
    target = None
    for node in cluster.nodes:
        if job_id.startswith(f"{node}-"):
            target = node
            break
    if target and target != cluster.self_name:
        url = cluster.get_node_url(target)
        if url:
            return await _proxy_request(req, url)
    return None


def _rr_or_raise() -> str:
    """Select a node via round-robin or raise 503."""
    node = cluster.select_rr_node()
    if not node:
        raise HTTPException(503, "No healthy nodes available")
    return node


# ── Job submission endpoints ─────────────────────────────────────────────────

@app.post("/jobs/by-urls", response_model=List[JobResponse], tags=["jobs"])
async def create_job_urls(request: Request, body: UrlBatchRequest):
    if not request.query_params.get("internal"):
        target = _rr_or_raise()
        if target != cluster.self_name:
            return await _proxy_request(request, cluster.get_node_url(target))

    responses = []
    for url in body.urls:
        safe_url = validate_url(url)
        job_id = _create_job(body.mode.value, body.settings, url=safe_url)
        responses.append(JobResponse(job_id=job_id, status="queued"))
    return responses


@app.post("/jobs/by-folder", response_model=List[JobResponse], tags=["jobs"])
async def create_job_folder(req: Request, request: FolderRequest):
    if not req.query_params.get("internal"):
        target = _rr_or_raise()
        if target != cluster.self_name:
            return await _proxy_request(req, cluster.get_node_url(target))

    safe_dir = safe_folder_path(request.folder_path)
    responses = []
    for fname in os.listdir(safe_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
            full_path = os.path.join(safe_dir, fname)
            job_id = _create_job(request.mode.value, request.settings, url=full_path)
            responses.append(JobResponse(job_id=job_id, status="queued"))
    return responses


@app.post("/jobs/by-filepath", response_model=JobResponse, tags=["jobs"])
async def create_job_filepath(req: Request, request: FilePathRequest):
    if not req.query_params.get("internal"):
        target = _rr_or_raise()
        if target != cluster.self_name:
            return await _proxy_request(req, cluster.get_node_url(target))

    file_path = (request.file_path or "").strip()
    if not file_path:
        raise HTTPException(status_code=400, detail="Empty file path")
    job_id = _create_job(request.mode.value, request.settings, url=file_path)
    return JobResponse(job_id=job_id, status="queued")


def _parse_settings(
    categories: str = Form(""),
    provider: str = Form("Ollama"),
    model_name: str = Form("qwen3-vl:8b-instruct"),
    ocr_engine: str = Form("EasyOCR"),
    ocr_mode: str = Form("🚀 Fast"),
    scan_mode: str = Form("Tail Only"),
    express_mode: bool = Form(False),
    override: bool = Form(False),
    enable_search: bool = Form(False),
    enable_web_search: Optional[bool] = Form(None),
    enable_agentic_search: Optional[bool] = Form(None),
    enable_vision_board: Optional[bool] = Form(None),
    enable_llm_frame: Optional[bool] = Form(None),
    enable_vision: Optional[bool] = Form(None),  # Deprecated alias
    context_size: int = Form(8192),
) -> JobSettings:
    resolved_search = _resolve_enable_web_search(
        enable_search=enable_search,
        enable_web_search=enable_web_search,
        enable_agentic_search=enable_agentic_search,
    )
    resolved_vision_board, resolved_llm_frame = _resolve_vision_flags(
        enable_vision_board=enable_vision_board,
        enable_llm_frame=enable_llm_frame,
        enable_vision_legacy=enable_vision,
    )
    return JobSettings(
        categories=categories, provider=provider, model_name=model_name,
        ocr_engine=ocr_engine, ocr_mode=ocr_mode, scan_mode=scan_mode,
        express_mode=express_mode,
        override=override,
        enable_search=resolved_search,
        enable_web_search=resolved_search,
        enable_agentic_search=resolved_search,
        enable_vision_board=resolved_vision_board,
        enable_llm_frame=resolved_llm_frame,
        context_size=context_size,
    )


@app.post("/jobs/upload", response_model=JobResponse, tags=["jobs"])
async def create_job_upload(
    req: Request,
    mode: JobMode = Form(JobMode.pipeline),
    settings: JobSettings = Depends(_parse_settings),
    file: UploadFile = File(...),
):
    """
    Direct file upload endpoint.
    NOTE: multipart bodies cannot be re-streamed by the proxy, so this
    endpoint always processes locally (internal=1 behaviour).
    Clients should POST directly to the node they want to own the job
    (or always use ?internal=1 to skip routing).
    """
    # Enforce upload size
    content_length = req.headers.get("content-length")
    check_upload_size(int(content_length) if content_length else None)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_filename = f"{uuid.uuid4()}_{os.path.basename(file.filename or 'upload.mp4')}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    # Stream with size guard
    written = 0
    try:
        with open(file_path, "wb") as buf:
            while chunk := await file.read(1 << 20):  # 1 MB chunks
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds {MAX_UPLOAD_MB:.0f} MB limit"
                    )
                buf.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("upload_write_error: %s", exc)
        raise HTTPException(500, "Failed to save uploaded file")

    job_id = _create_job(mode.value, settings, url=file_path)
    return JobResponse(job_id=job_id, status="queued")


# ── Job read endpoints ───────────────────────────────────────────────────────

def _get_jobs_from_db(limit: int = 100) -> list:
    def row_value(r, key: str, default=None):
        return r[key] if key in r.keys() else default

    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        JobStatus(
            job_id=r["id"], status=r["status"],
            stage=row_value(r, "stage"), stage_detail=row_value(r, "stage_detail"),
            duration_seconds=row_value(r, "duration_seconds"),
            created_at=r["created_at"], updated_at=r["updated_at"],
            progress=r["progress"], error=row_value(r, "error"),
            settings=JobSettings.model_validate_json(r["settings"]) if r["settings"] else None,
            mode=row_value(r, "mode"), url=row_value(r, "url"),
            brand=row_value(r, "brand"), category=row_value(r, "category"), category_id=row_value(r, "category_id"),
        )
        for r in rows
    ]


def _dedupe_jobs_by_id(jobs: list[dict]) -> list[dict]:
    deduped: dict[str, dict] = {}
    for job in jobs:
        job_id = job.get("job_id")
        if not job_id:
            continue
        current = deduped.get(job_id)
        incoming_updated = job.get("updated_at") or ""
        current_updated = (current or {}).get("updated_at") or ""
        if current is None or incoming_updated > current_updated:
            deduped[job_id] = job

    return sorted(deduped.values(), key=lambda x: x.get("created_at", ""), reverse=True)


@app.get("/jobs", response_model=List[JobStatus], tags=["jobs"])
def get_jobs_recent():
    return _get_jobs_from_db()


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["jobs"])
async def get_job(req: Request, job_id: str):
    proxy = await _maybe_proxy(req, job_id)
    if proxy:
        return proxy
    with closing(get_db()) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Job not found")
    def row_value(r, key: str, default=None):
        return r[key] if key in r.keys() else default
    return JobStatus(
        job_id=row["id"], status=row["status"],
        stage=row_value(row, "stage"), stage_detail=row_value(row, "stage_detail"),
        duration_seconds=row_value(row, "duration_seconds"),
        created_at=row["created_at"], updated_at=row["updated_at"],
        progress=row["progress"], error=row_value(row, "error"),
        settings=JobSettings.model_validate_json(row["settings"]) if row["settings"] else None,
        mode=row_value(row, "mode"), url=row_value(row, "url"),
        brand=row_value(row, "brand"), category=row_value(row, "category"), category_id=row_value(row, "category_id"),
    )


@app.get("/jobs/{job_id}/result", tags=["jobs"])
async def get_job_result(req: Request, job_id: str):
    proxy = await _maybe_proxy(req, job_id)
    if proxy:
        return proxy
    with closing(get_db()) as conn:
        row = conn.execute("SELECT result_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row or not row["result_json"]:
        return {"result": None}
    return {"result": json.loads(row["result_json"])}


@app.get("/jobs/{job_id}/video", tags=["jobs"])
async def stream_job_video(req: Request, job_id: str):
    """Stream source video for a job. Serves local files only."""
    proxy = await _maybe_proxy(req, job_id)
    if proxy:
        return proxy

    with closing(get_db()) as conn:
        row = conn.execute("SELECT url FROM jobs WHERE id = ?", (job_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    source_url = (row["url"] or "").strip()
    if not source_url:
        raise HTTPException(status_code=404, detail="Source video not configured")

    if source_url.startswith(("http://", "https://")):
        return JSONResponse({"type": "remote", "url": source_url})

    try:
        video_path = Path(source_url).expanduser().resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid video path: {exc}") from exc

    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Source video file not found on server")

    media_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    return FileResponse(path=str(video_path), media_type=media_type, filename=video_path.name)


@app.get("/jobs/{job_id}/video-poster", tags=["jobs"])
async def get_job_video_poster(req: Request, job_id: str):
    """Return a JPEG poster frame for a local source video."""
    proxy = await _maybe_proxy(req, job_id)
    if proxy:
        return proxy

    with closing(get_db()) as conn:
        row = conn.execute("SELECT url FROM jobs WHERE id = ?", (job_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    source_url = (row["url"] or "").strip()
    if not source_url:
        raise HTTPException(status_code=404, detail="Source video not configured")

    if source_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=404, detail="Poster preview is only available for local videos")

    try:
        video_path = Path(source_url).expanduser().resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid video path: {exc}") from exc

    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Source video file not found on server")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open source video")

        frame = None
        ok = False
        for _ in range(5):
            ok, candidate = cap.read()
            if ok and candidate is not None and getattr(candidate, "size", 0) > 0:
                frame = candidate
                break

        if frame is None:
            raise HTTPException(status_code=500, detail="Could not decode poster frame")

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )
        if not ok:
            raise HTTPException(status_code=500, detail="Could not encode poster frame")

        return Response(
            content=encoded.tobytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=300"},
        )
    finally:
        cap.release()


@app.get("/jobs/{job_id}/artifacts", tags=["jobs"])
async def get_job_artifacts(req: Request, job_id: str):
    proxy = await _maybe_proxy(req, job_id)
    if proxy:
        return proxy
    with closing(get_db()) as conn:
        row = conn.execute("SELECT artifacts_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row or not row["artifacts_json"]:
        payload = _default_job_artifacts(job_id)
    else:
        try:
            parsed = json.loads(row["artifacts_json"])
        except Exception:
            parsed = None
        payload = _normalize_job_artifacts(job_id, parsed)
    return {"artifacts": payload, **payload}


@app.get("/jobs/{job_id}/events", tags=["jobs"])
async def get_job_events(req: Request, job_id: str):
    proxy = await _maybe_proxy(req, job_id)
    if proxy:
        return proxy
    with closing(get_db()) as conn:
        row = conn.execute("SELECT events FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row or not row["events"]:
        return {"events": []}
    return {"events": json.loads(row["events"])}


@app.get("/jobs/{job_id}/explanation", tags=["jobs"])
async def get_job_explanation(req: Request, job_id: str):
    proxy = await _maybe_proxy(req, job_id)
    if proxy:
        return proxy
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT id, status, stage, stage_detail, mode, brand, category, category_id, result_json, artifacts_json, events FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "Job not found")

    try:
        artifacts_raw = json.loads(row["artifacts_json"]) if row["artifacts_json"] else None
    except Exception:
        artifacts_raw = None
    artifacts = _normalize_job_artifacts(job_id, artifacts_raw)

    try:
        result_payload = json.loads(row["result_json"]) if row["result_json"] else None
    except Exception:
        result_payload = None

    try:
        events = json.loads(row["events"]) if row["events"] else []
        if not isinstance(events, list):
            events = []
    except Exception:
        events = []

    explanation = _build_job_explanation(job_id, row, artifacts, result_payload, events)
    return {"explanation": explanation}


@app.get("/jobs/{job_id}/stream", tags=["jobs"], response_model=None)
async def stream_job_events(req: Request, job_id: str):
    """SSE stream with low-latency job updates backed by DB polling."""
    if not req.query_params.get("internal"):
        target = None
        for node in cluster.nodes:
            if job_id.startswith(f"{node}-"):
                target = node
                break
        if target and target != cluster.self_name:
            target_url = cluster.get_node_url(target)
            if target_url:
                return RedirectResponse(url=f"{target_url}/jobs/{job_id}/stream", status_code=307)

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        last_updated = None

        while True:
            if await req.is_disconnected():
                logger.debug("job_stream_disconnected: job_id=%s", job_id)
                break

            with closing(get_db()) as conn:
                row = conn.execute(
                    """
                    SELECT status, stage, stage_detail, progress, error, updated_at, events
                    FROM jobs
                    WHERE id = ?
                    """,
                    (job_id,),
                ).fetchone()

            if not row:
                yield {"event": "error", "data": json.dumps({"detail": "Job not found"})}
                break

            status = row["status"] or "unknown"
            current_updated = row["updated_at"]
            is_terminal = status in {"completed", "failed"}

            if current_updated != last_updated:
                last_updated = current_updated
                try:
                    events = json.loads(row["events"]) if row["events"] else []
                    if not isinstance(events, list):
                        events = []
                except Exception:
                    events = []

                payload = {
                    "status": status,
                    "stage": row["stage"],
                    "stage_detail": row["stage_detail"],
                    "progress": row["progress"],
                    "error": row["error"],
                    "updated_at": current_updated,
                    "events": events,
                }
                yield {"event": "update", "data": json.dumps(payload)}

            if is_terminal:
                yield {"event": "complete", "data": json.dumps({"status": status, "updated_at": current_updated})}
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.get("/admin/logs/stream", tags=["admin"], response_model=None)
async def stream_admin_logs(req: Request):
    queue, unsubscribe = subscribe_log_stream(max_queue_size=1000)

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        try:
            recent_lines = get_recent_log_lines(limit=200)
            yield {
                "event": "bootstrap",
                "data": json.dumps({"lines": recent_lines}),
            }

            while True:
                if await req.is_disconnected():
                    break
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield {"event": "log", "data": json.dumps({"line": line})}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
        finally:
            unsubscribe()

    return EventSourceResponse(event_generator())


@app.get("/admin/logs/recent", tags=["admin"])
def admin_recent_logs(limit: int = 200):
    bounded_limit = max(1, min(limit, 1000))
    return {"lines": get_recent_log_lines(limit=bounded_limit)}


@app.post("/admin/logs/clear", tags=["admin"])
def admin_clear_logs():
    clear_recent_log_lines()
    return {"status": "cleared"}


@app.delete("/jobs/{job_id}", tags=["jobs"])
async def delete_job(req: Request, job_id: str):
    proxy = await _maybe_proxy(req, job_id)
    if proxy:
        return proxy
    with closing(get_db()) as conn:
        with conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    mark_job_aborted(job_id)
    logger.info("job_deleted: job_id=%s", job_id)
    return {"status": "deleted"}


@app.post("/jobs/bulk-delete", tags=["jobs"])
async def bulk_delete_jobs(body: BulkDeleteRequest):
    """Delete multiple jobs from local storage; skips IDs that don't exist."""
    if not body.job_ids:
        raise HTTPException(status_code=400, detail="No job IDs provided")
    if len(body.job_ids) > 500:
        raise HTTPException(status_code=400, detail="Too many IDs (max 500)")

    deleted = 0
    with closing(get_db()) as conn:
        with conn:
            for job_id in body.job_ids:
                cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                deleted += cursor.rowcount
                mark_job_aborted(job_id)

    logger.info("bulk_delete: requested=%d deleted=%d", len(body.job_ids), deleted)
    return {"status": "deleted", "requested": len(body.job_ids), "deleted": deleted}


# ── Admin aggregation ────────────────────────────────────────────────────────

@app.get("/admin/jobs", response_model=List[JobStatus], tags=["admin"])
def get_admin_jobs():
    """Per-node job list — called by cluster dashboard fan-out."""
    return _get_jobs_from_db()


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid_env_value name=%s value=%r fallback=%d", name, raw, default)
        return default


if __name__ == "__main__":  # pragma: no cover - manual launch path
    import uvicorn

    bind_host = (os.environ.get("BIND_HOST") or "0.0.0.0").strip() or "0.0.0.0"
    port = _env_int("PORT", 8000)
    uvicorn.run(app, host=bind_host, port=port)
