from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx

from video_service.app.models.job import JobMode, JobResponse, JobSettings, JobStatus
from video_service.core.logging_setup import job_context

logger = logging.getLogger(__name__)


class ScenalyzeServiceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class ScenalyzeService(Protocol):
    def submit_video_by_filepath(
        self,
        file_path: str,
        *,
        mode: JobMode = JobMode.pipeline,
        settings: JobSettings | None = None,
    ) -> dict[str, Any]: ...

    def submit_video_by_url(
        self,
        url: str,
        *,
        mode: JobMode = JobMode.pipeline,
        settings: JobSettings | None = None,
    ) -> dict[str, Any]: ...

    def get_job_status(self, job_id: str) -> dict[str, Any]: ...
    def list_recent_jobs(self, limit: int = 20) -> list[dict[str, Any]]: ...
    def get_job_result(self, job_id: str) -> dict[str, Any]: ...
    def get_job_artifacts(self, job_id: str) -> dict[str, Any]: ...
    def get_job_events(self, job_id: str) -> dict[str, Any]: ...
    def get_job_explanation(self, job_id: str) -> dict[str, Any]: ...
    def get_taxonomy_explorer(self) -> dict[str, Any]: ...
    def find_taxonomy_candidates(
        self,
        query: str,
        *,
        limit: int = 5,
        suggested_categories_text: str = "",
        predicted_brand: str = "",
        ocr_summary: str = "",
        reasoning_summary: str = "",
    ) -> dict[str, Any]: ...
    def get_cluster_nodes(self) -> dict[str, Any]: ...
    def get_device_diagnostics(self) -> dict[str, Any]: ...
    def get_system_profile(self) -> dict[str, Any]: ...
    def get_concurrency_diagnostics(self) -> dict[str, Any]: ...
    def list_provider_models(self, provider: str = "ollama") -> list[dict[str, Any]]: ...


def job_resource_uris(job_id: str) -> dict[str, str]:
    return {
        "status": f"scenalyze://jobs/{job_id}/status",
        "result": f"scenalyze://jobs/{job_id}/result",
        "artifacts": f"scenalyze://jobs/{job_id}/artifacts",
        "events": f"scenalyze://jobs/{job_id}/events",
        "explanation": f"scenalyze://jobs/{job_id}/explanation",
    }


def _get_db():
    from video_service.db.database import get_db

    return get_db()


def _validate_url(url: str) -> str:
    from video_service.core.security import validate_url

    return validate_url(url)


def _cluster():
    from video_service.core.cluster import cluster

    return cluster


def _category_mapper():
    from video_service.core.categories import category_mapper

    return category_mapper


def _get_category_explorer_payload() -> dict[str, Any]:
    from video_service.core.category_mapping import get_category_explorer_payload

    return get_category_explorer_payload()


def _get_device_diagnostics() -> dict[str, Any]:
    from video_service.core.device import get_diagnostics

    return get_diagnostics()


def _get_system_profile() -> dict[str, Any]:
    from video_service.core.hardware_profiler import get_system_profile

    return get_system_profile()


def _get_concurrency_diagnostics() -> dict[str, Any]:
    from video_service.core.concurrency import get_concurrency_diagnostics

    return get_concurrency_diagnostics()


def _coerce_mode(mode: JobMode | str | None) -> JobMode:
    if isinstance(mode, JobMode):
        return mode
    text = str(mode or JobMode.pipeline.value).strip() or JobMode.pipeline.value
    try:
        return JobMode(text)
    except ValueError as exc:
        raise ScenalyzeServiceError(f"Unsupported mode: {text}", status_code=400) from exc


def _coerce_settings(settings: JobSettings | dict[str, Any] | None) -> JobSettings:
    if settings is None:
        return JobSettings()
    if isinstance(settings, JobSettings):
        return settings
    if isinstance(settings, dict):
        return JobSettings.model_validate(settings)
    raise ScenalyzeServiceError("Invalid job settings payload", status_code=400)


def _job_not_found(job_id: str) -> ScenalyzeServiceError:
    return ScenalyzeServiceError(f"Job not found: {job_id}", status_code=404)


def _load_json(raw: str | None, *, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _row_value(row, key: str, default: Any = None) -> Any:
    return row[key] if key in row.keys() else default


def _normalize_result_row_payload(row: dict | None) -> dict:
    payload = row if isinstance(row, dict) else {}
    return {
        "brand": str(payload.get("brand") or payload.get("Brand") or ""),
        "confidence": payload.get("confidence", payload.get("Confidence")),
        "reasoning": payload.get("reasoning", payload.get("Reasoning")),
        "parent_category_id": str(payload.get("parent_category_id") or ""),
        "parent_category": str(payload.get("parent_category") or ""),
        "industry_id": str(payload.get("industry_id") or ""),
        "industry_name": str(payload.get("industry_name") or ""),
        "category_id": str(payload.get("category_id") or payload.get("Category ID") or ""),
        "category_name": str(
            payload.get("category_name")
            or payload.get("category")
            or payload.get("Category")
            or ""
        ),
    }


def _extract_result_summary(result_json: str | None) -> dict:
    payload = _load_json(result_json, default=None)
    if not isinstance(payload, list) or not payload:
        return {}
    first_row = payload[0] if isinstance(payload[0], dict) else {}
    return _normalize_result_row_payload(first_row)


def _extract_artifact_mapper_summary(artifacts_json: str | None) -> dict:
    payload = _load_json(artifacts_json, default=None)
    if not isinstance(payload, dict):
        return {}
    mapper_payload = payload.get("category_mapper")
    return mapper_payload if isinstance(mapper_payload, dict) else {}


def _default_job_artifacts(job_id: str) -> dict:
    return {
        "latest_frames": [],
        "llm_frames": [],
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
            "parent_category": "",
            "category_path_text": "",
            "method": "",
            "score": None,
            "confidence": None,
            "query_fragments": [],
            "top_matches": [],
            "vector_plot": None,
        },
        "processing_trace": None,
        "extras": {
            "events_url": f"/jobs/{job_id}/events",
        },
    }


def _normalize_job_artifacts(job_id: str, artifacts: dict | None) -> dict:
    payload = _default_job_artifacts(job_id)
    if not isinstance(artifacts, dict):
        return payload

    payload["latest_frames"] = artifacts.get("latest_frames") or []
    payload["llm_frames"] = artifacts.get("llm_frames") or []
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
        payload["category_mapper"]["category_id"] = str(mapper_payload.get("category_id") or "")
        payload["category_mapper"]["parent_category"] = mapper_payload.get("parent_category") or ""
        payload["category_mapper"]["category_path_text"] = mapper_payload.get("category_path_text") or ""
        payload["category_mapper"]["method"] = mapper_payload.get("method") or ""
        payload["category_mapper"]["score"] = mapper_payload.get("score")
        payload["category_mapper"]["confidence"] = mapper_payload.get("confidence")
        payload["category_mapper"]["query_fragments"] = mapper_payload.get("query_fragments") or []
        payload["category_mapper"]["top_matches"] = mapper_payload.get("top_matches") or []
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


def _build_job_explanation(
    job_id: str,
    job_row,
    artifacts: dict,
    result_payload: list | None,
    events: list[str],
) -> dict:
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
            "brand": first_result.get("brand") or first_result.get("Brand") or row_brand,
            "category": first_result.get("category_name") or first_result.get("category") or first_result.get("Category") or row_category,
            "category_id": first_result.get("category_id") or first_result.get("Category ID") or row_category_id,
            "industry_id": first_result.get("industry_id") or "",
            "industry_name": first_result.get("industry_name") or "",
            "confidence": first_result.get("confidence") or first_result.get("Confidence"),
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


def _build_job_status_payload(row) -> dict[str, Any]:
    result_summary = _extract_result_summary(_row_value(row, "result_json"))
    mapper_summary = _extract_artifact_mapper_summary(_row_value(row, "artifacts_json"))
    confidence_value = result_summary.get("confidence")
    if confidence_value is None:
        confidence_value = mapper_summary.get("confidence")
    if confidence_value is None:
        confidence_value = mapper_summary.get("score")
    payload = JobStatus(
        job_id=row["id"],
        status=row["status"],
        stage=_row_value(row, "stage"),
        stage_detail=_row_value(row, "stage_detail"),
        confidence=confidence_value,
        duration_seconds=_row_value(row, "duration_seconds"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        progress=row["progress"],
        error=_row_value(row, "error"),
        settings=JobSettings.model_validate_json(row["settings"]) if row["settings"] else None,
        mode=_row_value(row, "mode"),
        url=_row_value(row, "url"),
        brand=result_summary.get("brand") or _row_value(row, "brand"),
        category=result_summary.get("category_name") or _row_value(row, "category"),
        category_id=result_summary.get("category_id") or _row_value(row, "category_id"),
        category_name=result_summary.get("category_name") or _row_value(row, "category"),
        parent_category_id=result_summary.get("parent_category_id") or "",
        parent_category=result_summary.get("parent_category") or "",
        industry_id=result_summary.get("industry_id") or "",
        industry_name=result_summary.get("industry_name") or "",
    )
    return payload.model_dump(mode="json")


def _openai_compat_models_url() -> str:
    compat_url = os.environ.get(
        "OPENAI_COMPAT_URL",
        "http://localhost:1234/v1/chat/completions",
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


def _list_provider_models_sync(provider: str) -> list[dict[str, Any]]:
    provider_name = str(provider or "ollama").strip().lower()
    try:
        with httpx.Client(timeout=5.0) as client:
            if provider_name == "ollama":
                ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
                response = client.get(f"{ollama_host}/api/tags")
                if response.status_code == 200:
                    payload = response.json()
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
            if provider_name in {"llama-server", "llama server"}:
                response = client.get(_openai_compat_models_url())
                if response.status_code == 200:
                    payload = response.json()
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
    except Exception:
        return []
    return []


def _taxonomy_match_payload(
    query: str,
    *,
    limit: int,
    suggested_categories_text: str = "",
    predicted_brand: str = "",
    ocr_summary: str = "",
    reasoning_summary: str = "",
) -> dict[str, Any]:
    mapper = _category_mapper()
    bounded_limit = max(1, min(int(limit), 10))
    match = mapper.map_category(
        raw_category=query,
        suggested_categories_text=suggested_categories_text,
        predicted_brand=predicted_brand,
        ocr_summary=ocr_summary,
        reasoning_summary=reasoning_summary,
    )
    top_matches = list(match.get("top_matches") or [])[:bounded_limit]
    labels = []
    canonical = str(match.get("canonical_category") or "")
    if canonical:
        labels.append(canonical)
    labels.extend(str(item.get("label") or "") for item in top_matches)
    context_map = mapper.get_category_context_map([label for label in labels if label])

    enriched_top_matches = []
    for item in top_matches:
        label = str(item.get("label") or "")
        enriched = dict(item)
        enriched["context"] = context_map.get(label, "")
        enriched_top_matches.append(enriched)

    return {
        "query": str(query or ""),
        "canonical_category": canonical,
        "category_id": str(match.get("category_id") or ""),
        "industry_id": str(match.get("industry_id") or ""),
        "industry_name": str(match.get("industry_name") or ""),
        "parent_category": str(match.get("parent_category") or ""),
        "category_path_text": str(match.get("category_path_text") or ""),
        "category_context": context_map.get(canonical, ""),
        "category_match_method": str(match.get("category_match_method") or ""),
        "category_match_score": match.get("category_match_score"),
        "mapping_query_text": str(match.get("mapping_query_text") or ""),
        "mapping_query_fragments": list(match.get("mapping_query_fragments") or []),
        "top_matches": enriched_top_matches,
    }


class LocalScenalyzeService:
    def _create_job(
        self,
        mode: str,
        settings: JobSettings,
        url: str | None = None,
    ) -> str:
        cluster = _cluster()
        if not cluster.is_accepting_new_jobs(cluster.self_name):
            raise ScenalyzeServiceError(
                f"Node {cluster.self_name} is in maintenance mode and not accepting new jobs",
                status_code=503,
            )

        job_id = f"{cluster.self_name}-{uuid.uuid4()}"
        with closing(_get_db()) as conn:
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
                        "",
                        "",
                        "{}",
                    ),
                )
        with job_context(job_id):
            logger.info("mcp_job_created: job_id=%s mode=%s url=%s", job_id, mode, url)
        return job_id

    def submit_video_by_filepath(
        self,
        file_path: str,
        *,
        mode: JobMode = JobMode.pipeline,
        settings: JobSettings | None = None,
    ) -> dict[str, Any]:
        file_path = str(file_path or "").strip()
        if not file_path:
            raise ScenalyzeServiceError("Empty file path", status_code=400)
        resolved_settings = _coerce_settings(settings)
        job_id = self._create_job(_coerce_mode(mode).value, resolved_settings, url=file_path)
        response = JobResponse(job_id=job_id, status="queued").model_dump(mode="json")
        response["resources"] = job_resource_uris(job_id)
        return response

    def submit_video_by_url(
        self,
        url: str,
        *,
        mode: JobMode = JobMode.pipeline,
        settings: JobSettings | None = None,
    ) -> dict[str, Any]:
        safe_url = _validate_url(url)
        resolved_settings = _coerce_settings(settings)
        job_id = self._create_job(_coerce_mode(mode).value, resolved_settings, url=safe_url)
        response = JobResponse(job_id=job_id, status="queued").model_dump(mode="json")
        response["resources"] = job_resource_uris(job_id)
        return response

    def _get_job_row(self, job_id: str, select_sql: str = "*"):
        with closing(_get_db()) as conn:
            row = conn.execute(f"SELECT {select_sql} FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise _job_not_found(job_id)
        return row

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        row = self._get_job_row(job_id)
        return _build_job_status_payload(row)

    def list_recent_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 100))
        with closing(_get_db()) as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (bounded_limit,),
            ).fetchall()
        return [_build_job_status_payload(row) for row in rows]

    def get_job_result(self, job_id: str) -> dict[str, Any]:
        row = self._get_job_row(job_id, "result_json")
        if not row["result_json"]:
            return {"result": None}
        payload = _load_json(row["result_json"], default=None)
        if not isinstance(payload, list):
            return {"result": None}
        return {
            "result": [
                _normalize_result_row_payload(item if isinstance(item, dict) else {})
                for item in payload
            ]
        }

    def get_job_artifacts(self, job_id: str) -> dict[str, Any]:
        row = self._get_job_row(job_id, "artifacts_json")
        if not row["artifacts_json"]:
            payload = _default_job_artifacts(job_id)
        else:
            parsed = _load_json(row["artifacts_json"], default=None)
            payload = _normalize_job_artifacts(job_id, parsed)
        return {"artifacts": payload, **payload}

    def get_job_events(self, job_id: str) -> dict[str, Any]:
        row = self._get_job_row(job_id, "events")
        events = _load_json(row["events"], default=[])
        if not isinstance(events, list):
            events = []
        return {"events": [str(item) for item in events]}

    def get_job_explanation(self, job_id: str) -> dict[str, Any]:
        row = self._get_job_row(
            job_id,
            "id, status, stage, stage_detail, mode, brand, category, category_id, result_json, artifacts_json, events",
        )
        artifacts_raw = _load_json(row["artifacts_json"], default=None)
        artifacts = _normalize_job_artifacts(job_id, artifacts_raw)
        result_payload = _load_json(row["result_json"], default=None)
        if not isinstance(result_payload, list):
            result_payload = None
        events = _load_json(row["events"], default=[])
        if not isinstance(events, list):
            events = []
        explanation = _build_job_explanation(
            job_id,
            row,
            artifacts,
            result_payload,
            [str(item) for item in events],
        )
        return {"explanation": explanation}

    def get_taxonomy_explorer(self) -> dict[str, Any]:
        return _get_category_explorer_payload()

    def find_taxonomy_candidates(
        self,
        query: str,
        *,
        limit: int = 5,
        suggested_categories_text: str = "",
        predicted_brand: str = "",
        ocr_summary: str = "",
        reasoning_summary: str = "",
    ) -> dict[str, Any]:
        return _taxonomy_match_payload(
            query,
            limit=limit,
            suggested_categories_text=suggested_categories_text,
            predicted_brand=predicted_brand,
            ocr_summary=ocr_summary,
            reasoning_summary=reasoning_summary,
        )

    def get_cluster_nodes(self) -> dict[str, Any]:
        cluster = _cluster()
        return {
            "nodes": cluster.nodes,
            "status": cluster.node_status,
            "maintenance": cluster.node_maintenance,
            "accepting_new_jobs": {
                node: cluster.is_accepting_new_jobs(node)
                for node in cluster.nodes
            },
            "self": cluster.self_name,
        }

    def get_device_diagnostics(self) -> dict[str, Any]:
        return _get_device_diagnostics()

    def get_system_profile(self) -> dict[str, Any]:
        return _get_system_profile()

    def get_concurrency_diagnostics(self) -> dict[str, Any]:
        return _get_concurrency_diagnostics()

    def list_provider_models(self, provider: str = "ollama") -> list[dict[str, Any]]:
        return _list_provider_models_sync(provider)


class HttpScenalyzeService:
    def __init__(self, base_url: str, *, timeout: float = 30.0):
        self.base_url = str(base_url or "").rstrip("/")
        self.timeout = float(timeout)
        if not self.base_url:
            raise ScenalyzeServiceError("HTTP service mode requires a base URL", status_code=500)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, url, json=json_payload, params=params)
        except Exception as exc:
            raise ScenalyzeServiceError(f"Scenalyze API request failed: {exc}", status_code=503) from exc

        if response.status_code >= 400:
            detail = response.text or f"Scenalyze API returned HTTP {response.status_code}"
            raise ScenalyzeServiceError(detail, status_code=response.status_code)

        if not response.content:
            return None
        return response.json()

    def submit_video_by_filepath(
        self,
        file_path: str,
        *,
        mode: JobMode = JobMode.pipeline,
        settings: JobSettings | None = None,
    ) -> dict[str, Any]:
        payload = self._request(
            "POST",
            "/jobs/by-filepath",
            json_payload={
                "mode": _coerce_mode(mode).value,
                "file_path": str(file_path or ""),
                "settings": _coerce_settings(settings).model_dump(mode="json"),
            },
        )
        if not isinstance(payload, dict):
            raise ScenalyzeServiceError("Unexpected response for filepath submission", status_code=503)
        job_id = str(payload.get("job_id") or "")
        payload["resources"] = job_resource_uris(job_id)
        return payload

    def submit_video_by_url(
        self,
        url: str,
        *,
        mode: JobMode = JobMode.pipeline,
        settings: JobSettings | None = None,
    ) -> dict[str, Any]:
        payload = self._request(
            "POST",
            "/jobs/by-urls",
            json_payload={
                "mode": _coerce_mode(mode).value,
                "urls": [str(url or "")],
                "settings": _coerce_settings(settings).model_dump(mode="json"),
            },
        )
        if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
            raise ScenalyzeServiceError("Unexpected response for URL submission", status_code=503)
        response = payload[0]
        job_id = str(response.get("job_id") or "")
        response["resources"] = job_resource_uris(job_id)
        return response

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/jobs/{job_id}")
        if not isinstance(payload, dict):
            raise ScenalyzeServiceError("Unexpected job status payload", status_code=503)
        return payload

    def list_recent_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        payload = self._request("GET", "/jobs")
        if not isinstance(payload, list):
            raise ScenalyzeServiceError("Unexpected jobs payload", status_code=503)
        bounded_limit = max(1, min(int(limit), len(payload) or 1))
        return [item for item in payload[:bounded_limit] if isinstance(item, dict)]

    def get_job_result(self, job_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/jobs/{job_id}/result")
        if not isinstance(payload, dict):
            raise ScenalyzeServiceError("Unexpected job result payload", status_code=503)
        return payload

    def get_job_artifacts(self, job_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/jobs/{job_id}/artifacts")
        if not isinstance(payload, dict):
            raise ScenalyzeServiceError("Unexpected job artifacts payload", status_code=503)
        return payload

    def get_job_events(self, job_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/jobs/{job_id}/events")
        if not isinstance(payload, dict):
            raise ScenalyzeServiceError("Unexpected job events payload", status_code=503)
        return payload

    def get_job_explanation(self, job_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/jobs/{job_id}/explanation")
        if not isinstance(payload, dict):
            raise ScenalyzeServiceError("Unexpected job explanation payload", status_code=503)
        return payload

    def get_taxonomy_explorer(self) -> dict[str, Any]:
        payload = self._request("GET", "/api/taxonomy/explorer")
        if not isinstance(payload, dict):
            raise ScenalyzeServiceError("Unexpected taxonomy explorer payload", status_code=503)
        return payload

    def find_taxonomy_candidates(
        self,
        query: str,
        *,
        limit: int = 5,
        suggested_categories_text: str = "",
        predicted_brand: str = "",
        ocr_summary: str = "",
        reasoning_summary: str = "",
    ) -> dict[str, Any]:
        return _taxonomy_match_payload(
            query,
            limit=limit,
            suggested_categories_text=suggested_categories_text,
            predicted_brand=predicted_brand,
            ocr_summary=ocr_summary,
            reasoning_summary=reasoning_summary,
        )

    def get_cluster_nodes(self) -> dict[str, Any]:
        payload = self._request("GET", "/cluster/nodes")
        if not isinstance(payload, dict):
            raise ScenalyzeServiceError("Unexpected cluster payload", status_code=503)
        return payload

    def get_device_diagnostics(self) -> dict[str, Any]:
        payload = self._request("GET", "/diagnostics/device")
        if not isinstance(payload, dict):
            raise ScenalyzeServiceError("Unexpected diagnostics payload", status_code=503)
        return payload

    def get_system_profile(self) -> dict[str, Any]:
        payload = self._request("GET", "/api/system/profile")
        if not isinstance(payload, dict):
            raise ScenalyzeServiceError("Unexpected system profile payload", status_code=503)
        return payload

    def get_concurrency_diagnostics(self) -> dict[str, Any]:
        payload = self._request("GET", "/diagnostics/concurrency")
        if not isinstance(payload, dict):
            raise ScenalyzeServiceError("Unexpected concurrency diagnostics payload", status_code=503)
        return payload

    def list_provider_models(self, provider: str = "ollama") -> list[dict[str, Any]]:
        payload = self._request("GET", "/api/v1/models", params={"provider": provider})
        if not isinstance(payload, list):
            raise ScenalyzeServiceError("Unexpected provider models payload", status_code=503)
        return [item for item in payload if isinstance(item, dict)]
