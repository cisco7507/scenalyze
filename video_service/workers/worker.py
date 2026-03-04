"""
video_service/workers/worker.py
================================
Background worker — claims queued jobs from SQLite and processes them.

Structured logging: all log lines include job_id for correlation.
Never uses print() in hot paths; structured logger used throughout.
"""

import time
import os
import sys
import json
import re
import logging
import traceback
import multiprocessing
import threading
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from contextlib import closing

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from video_service.core.logging_setup import (
    configure_logging,
    reset_job_context,
    reset_stage_context,
    set_job_context,
    set_stage_context,
)

configure_logging()

from video_service.db.database import get_db, init_db
from video_service.core import run_pipeline_job, run_agent_job
from video_service.core.concurrency import (
    get_concurrency_diagnostics,
    get_pipeline_threads_per_job,
    get_worker_processes_config,
)
from video_service.core.device import get_diagnostics, DEVICE
from video_service.core.benchmarking import evaluate_benchmark_suite
from video_service.core.abort import clear_aborted_job

logger = logging.getLogger(__name__)
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/tmp/video_service_artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
WATCH_OUTPUT_DIR = (os.environ.get("WATCH_OUTPUT_DIR") or "").strip()
_WATCH_FOLDERS_RAW = (os.environ.get("WATCH_FOLDERS") or "").strip()
WATCH_ROOTS = [
    os.path.realpath(folder.strip())
    for folder in _WATCH_FOLDERS_RAW.split(",")
    if folder.strip()
]


def _short(text: str | None, max_len: int = 220) -> str:
    if not text:
        return ""
    compact = " ".join(str(text).split())
    return compact[:max_len]


def _sanitize_job_id(job_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", job_id or "")


def _resolve_enable_web_search(settings: dict) -> bool:
    if "enable_web_search" in settings:
        return bool(settings.get("enable_web_search"))
    if "enable_agentic_search" in settings:
        return bool(settings.get("enable_agentic_search"))
    return bool(settings.get("enable_search", False))


def _resolve_enable_vision_board(settings: dict) -> bool:
    if "enable_vision_board" in settings:
        return bool(settings.get("enable_vision_board"))
    if "enable_vision" in settings:
        return bool(settings.get("enable_vision"))
    return True


def _resolve_enable_llm_frame(settings: dict) -> bool:
    if "enable_llm_frame" in settings:
        return bool(settings.get("enable_llm_frame"))
    if "enable_vision" in settings:
        return bool(settings.get("enable_vision"))
    return True


def _build_default_artifacts(job_id: str) -> dict:
    return {
        "latest_frames": [],
        "ocr_text": {"text": "", "lines": [], "url": None},
        "per_frame_vision": [],
        "vision_board": {"image_url": None, "plot_url": None, "top_matches": [], "metadata": {}},
        "extras": {"events_url": f"/jobs/{job_id}/events"},
    }


def _extract_timestamp_seconds(label: str) -> float | None:
    if not label:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)s", str(label))
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _write_text_artifact(job_id: str, relative_name: str, text: str) -> str | None:
    if not text:
        return None
    safe_job = _sanitize_job_id(job_id)
    rel_path = Path(safe_job) / relative_name
    abs_path = ARTIFACTS_DIR / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(text, encoding="utf-8")
    return f"/artifacts/{rel_path.as_posix()}"


def _save_gallery_frames(job_id: str, gallery: list) -> list[dict]:
    frames: list[dict] = []
    safe_job = _sanitize_job_id(job_id)
    rel_dir = Path(safe_job) / "latest_frames"
    abs_dir = ARTIFACTS_DIR / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(gallery or []):
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        image_obj, label = item[0], str(item[1])
        if image_obj is None:
            continue
        filename = f"frame_{idx+1:03d}.jpg"
        rel_path = rel_dir / filename
        abs_path = ARTIFACTS_DIR / rel_path
        try:
            cv2.imwrite(str(abs_path), image_obj)
        except Exception:
            continue
        frames.append(
            {
                "timestamp": _extract_timestamp_seconds(label),
                "label": label,
                "url": f"/artifacts/{rel_path.as_posix()}",
            }
        )
    return frames


def _vision_board_from_scores(scores: dict) -> dict:
    from video_service.core.categories import category_mapper  # avoid circular at module level
    top_matches = []
    for label, score in (scores or {}).items():
        cat_id = category_mapper.cat_to_id.get(str(label))
        top_matches.append({"label": str(label), "score": float(score), "category_id": cat_id})
    return {
        "image_url": None,
        "plot_url": None,
        "top_matches": top_matches,
        "metadata": {"source": "pipeline_scores", "count": len(top_matches)},
    }



def _vision_board_from_nebula(job_id: str, nebula) -> dict:
    board = {
        "image_url": None,
        "plot_url": None,
        "top_matches": [],
        "metadata": {"source": "agent_nebula", "count": 0},
    }
    if nebula is None or not hasattr(nebula, "to_plotly_json"):
        return board
    try:
        payload = nebula.to_plotly_json()
        raw = json.dumps(payload)
        url = _write_text_artifact(job_id, "vision_board/plotly.json", raw)
        board["plot_url"] = url
        board["metadata"]["trace_count"] = len(payload.get("data", []))
    except Exception as exc:
        logger.debug("vision_board_export_failed: %s", exc)
    return board


def _extract_summary_fields(result_json: str | None) -> tuple[str, str, str]:
    if not result_json:
        return "", "", ""
    try:
        payload = json.loads(result_json)
    except Exception:
        return "", "", ""
    if not isinstance(payload, list) or not payload:
        return "", "", ""
    row = payload[0] if isinstance(payload[0], dict) else {}
    brand = str(row.get("Brand") or row.get("brand") or "")
    category = str(row.get("Category") or row.get("category") or "")
    category_id = str(row.get("Category ID") or row.get("category_id") or "")
    return brand, category, category_id


def _record_job_stats(
    job_id: str,
    status: str,
    source_url: str | None,
    mode: str | None,
    settings: dict,
    brand: str,
    category: str,
    category_id: str,
    duration_seconds: float | None,
) -> None:
    source = (source_url or "").strip()
    source_type = "url" if source.startswith(("http://", "https://")) else "local"
    try:
        with closing(get_db()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO job_stats
                    (id, status, mode, brand, category, category_id, duration_seconds,
                     scan_mode, provider, model_name, ocr_engine, source_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        status,
                        mode or "",
                        brand or "",
                        category or "",
                        category_id or "",
                        duration_seconds,
                        settings.get("scan_mode", ""),
                        settings.get("provider", ""),
                        settings.get("model_name", ""),
                        settings.get("ocr_engine", ""),
                        source_type,
                    ),
                )
    except Exception as exc:
        logger.warning("job_stats_write_failed: %s", exc)


def _extract_agent_ocr_text(events: list[str]) -> str:
    for evt in events:
        if not evt:
            continue
        if "Observation:" not in evt:
            continue
        idx = evt.find("Observation:")
        if idx >= 0:
            return evt[idx + len("Observation:") :].strip()
    return ""


def _is_path_within_roots(path_value: str, roots: list[str]) -> bool:
    if not roots:
        return False
    try:
        real_path = os.path.realpath(path_value)
    except (OSError, ValueError):
        return False
    for root in roots:
        try:
            if os.path.commonpath([real_path, root]) == root:
                return True
        except ValueError:
            continue
    return False


def _maybe_export_result_json(
    job_id: str,
    source_url: str | None,
    result_json: str | None,
    *,
    status: str,
    error_message: str | None = None,
) -> None:
    """Write JSON result for watch-folder ingested local files when configured."""
    if not WATCH_OUTPUT_DIR:
        return
    if not source_url:
        return
    if source_url.startswith(("http://", "https://")):
        return
    if not _is_path_within_roots(source_url, WATCH_ROOTS):
        return

    try:
        os.makedirs(WATCH_OUTPUT_DIR, exist_ok=True)
        source_name = os.path.splitext(os.path.basename(source_url))[0] or job_id
        safe_name = re.sub(r"[^A-Za-z0-9_.\-]", "_", source_name)
        output_path = os.path.join(WATCH_OUTPUT_DIR, f"{safe_name}.json")

        parsed_result = None
        if result_json:
            try:
                parsed_result = json.loads(result_json)
            except Exception:
                parsed_result = result_json

        payload = {
            "job_id": job_id,
            "source_file": source_url,
            "status": status,
            "error": _short(error_message, 1000) if error_message else None,
            "result": parsed_result,
        }

        tmp_path = f"{output_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, output_path)

        logger.info("result_exported: %s", output_path)
    except Exception as exc:
        logger.error("result_export_failed: job=%s error=%s", job_id, exc)


def _append_job_event(job_id: str, message: str) -> None:
    attempts = 0
    while attempts < 8:
        attempts += 1
        try:
            with closing(get_db()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("SELECT events FROM jobs WHERE id = ?", (job_id,)).fetchone()
                events = []
                if row and row["events"]:
                    try:
                        events = json.loads(row["events"])
                    except Exception:
                        events = []

                events.append(message)
                # Keep event payload bounded.
                events = events[-400:]
                conn.execute(
                    "UPDATE jobs SET events = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (json.dumps(events), job_id),
                )
                conn.commit()
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempts >= 8:
                logger.warning("event_append_failed: %s", exc)
                return
            time.sleep(0.05 * attempts)
        except Exception as exc:
            logger.warning("event_append_failed: %s", exc)
            return


def _execute_job_update_with_retry(sql: str, params: tuple, *, attempts: int = 10) -> None:
    for attempt in range(1, attempts + 1):
        try:
            with closing(get_db()) as conn:
                with conn:
                    conn.execute(sql, params)
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == attempts:
                raise
            time.sleep(0.05 * attempt)


def _start_job_heartbeat(
    job_id: str,
    stop_event: threading.Event,
    *,
    interval_seconds: float = 60.0,
) -> threading.Thread:
    """Refresh jobs.updated_at while a long-running job is executing."""

    def _heartbeat_loop() -> None:
        while not stop_event.wait(interval_seconds):
            try:
                _execute_job_update_with_retry(
                    "UPDATE jobs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job_id,),
                    attempts=5,
                )
                logger.debug("job_heartbeat: refreshed updated_at")
            except sqlite3.OperationalError as exc:
                logger.debug("job_heartbeat_db_locked: %s", exc)
            except Exception as exc:
                logger.warning("job_heartbeat_failed: %s", exc)

    thread = threading.Thread(
        target=_heartbeat_loop,
        name=f"job-heartbeat-{_sanitize_job_id(job_id)}",
        daemon=True,
    )
    thread.start()
    return thread


def _set_stage(
    job_id: str,
    stage: str,
    detail: str,
    *,
    status: str | None = None,
    error: str | None = None,
) -> None:
    stage_name = stage or "-"
    detail_msg = _short(detail)
    set_stage_context(stage_name, detail_msg)

    sql = "UPDATE jobs SET stage = ?, stage_detail = ?, updated_at = CURRENT_TIMESTAMP"
    params: list[str] = [stage_name, detail_msg]
    if status is not None:
        sql += ", status = ?"
        params.append(status)
    if error is not None:
        sql += ", error = ?"
        params.append(error[:4096])
    sql += " WHERE id = ?"
    params.append(job_id)

    _execute_job_update_with_retry(sql, tuple(params))

    logger.info("%s", detail_msg)
    _append_job_event(
        job_id,
        f"{datetime.now(timezone.utc).isoformat()} {stage_name}: {detail_msg}",
    )


def _stage_callback(job_id: str):
    def callback(stage: str, detail: str) -> None:
        _set_stage(job_id, stage, detail)
    return callback


def claim_and_process_job() -> bool:
    job_id = None
    job_token = None
    claim_conn = None
    heartbeat_stop_event: threading.Event | None = None
    heartbeat_thread: threading.Thread | None = None
    try:
        claim_conn = get_db()
        cur = claim_conn.cursor()
        claim_conn.execute("BEGIN IMMEDIATE")

        cur.execute(
            """
            SELECT *
            FROM jobs
            WHERE status IN ('queued', 're-queued')
            ORDER BY updated_at ASC
            LIMIT 1
            """
        )
        row = cur.fetchone()

        if not row:
            claim_conn.rollback()
            return False

        job_id = row["id"]
        cur.execute(
            "UPDATE jobs SET status = 'processing', stage = 'claim', stage_detail = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            ("worker claimed job", job_id),
        )
        claim_conn.commit()
        claim_conn.close()
        claim_conn = None
        processing_start = time.monotonic()
        heartbeat_stop_event = threading.Event()
        heartbeat_thread = _start_job_heartbeat(job_id, heartbeat_stop_event, interval_seconds=60.0)

        job_token = set_job_context(job_id)
        set_stage_context("claim", "worker claimed job")
        logger.info("worker claimed job (device=%s)", DEVICE)
        _append_job_event(
            job_id,
            f"{datetime.now(timezone.utc).isoformat()} claim: worker claimed job",
        )

        url      = row["url"]
        mode     = row["mode"]
        benchmark_suite_id = (row["benchmark_suite_id"] or "").strip() if "benchmark_suite_id" in row.keys() else ""
        settings = json.loads(row["settings"]) if row["settings"] else {}

        events: list[str] = []
        result_json: str | None = None
        artifacts_payload: dict = _build_default_artifacts(job_id)
        error_msg: str | None = None

        try:
            if mode == "pipeline":
                result_json, artifacts_payload = _run_pipeline(job_id, url, settings)
            elif mode == "agent":
                result_json, events, artifacts_payload = _run_agent(job_id, url, settings)
            elif mode == "benchmark":
                result_json, artifacts_payload = _run_pipeline(job_id, url, settings)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        except Exception as exc:
            logger.exception("job_error")
            error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        duration_seconds = round(time.monotonic() - processing_start, 2)

        # Persist result
        if error_msg:
            _set_stage(
                job_id,
                "failed",
                f"job failed: {_short(error_msg, 180)}",
                status="failed",
                error=error_msg,
            )
            _execute_job_update_with_retry(
                "UPDATE jobs SET duration_seconds = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (duration_seconds, job_id),
            )
            _maybe_export_result_json(
                job_id,
                url,
                result_json,
                status="failed",
                error_message=error_msg,
            )
            _record_job_stats(
                job_id=job_id,
                status="failed",
                source_url=url,
                mode=mode,
                settings=settings,
                brand="",
                category="",
                category_id="",
                duration_seconds=duration_seconds,
            )
            logger.error("job_failed: error=%.200s", error_msg)
        else:
            _set_stage(job_id, "persist", "persisting result payload")
            brand, category, category_id = _extract_summary_fields(result_json)
            _execute_job_update_with_retry(
                "UPDATE jobs SET status = 'completed', stage = 'completed', stage_detail = ?, progress = 100, result_json = ?, artifacts_json = ?, brand = ?, category = ?, category_id = ?, duration_seconds = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (
                    "result persisted",
                    result_json,
                    json.dumps(artifacts_payload),
                    brand,
                    category,
                    category_id,
                    duration_seconds,
                    job_id,
                ),
            )
            _append_job_event(
                job_id,
                f"{datetime.now(timezone.utc).isoformat()} completed: result persisted",
            )
            _maybe_export_result_json(
                job_id,
                url,
                result_json,
                status="completed",
            )
            _record_job_stats(
                job_id=job_id,
                status="completed",
                source_url=url,
                mode=mode,
                settings=settings,
                brand=brand,
                category=category,
                category_id=category_id,
                duration_seconds=duration_seconds,
            )
            set_stage_context("completed", "result persisted")
            logger.info("job_completed")

        if benchmark_suite_id:
            try:
                evaluation = evaluate_benchmark_suite(benchmark_suite_id)
                logger.info(
                    "benchmark_suite_eval: suite_id=%s status=%s completed=%s/%s",
                    benchmark_suite_id,
                    evaluation.get("status"),
                    evaluation.get("completed_jobs"),
                    evaluation.get("total_jobs"),
                )
            except Exception as exc:
                logger.warning(
                    "benchmark_suite_eval_failed: suite_id=%s error=%s",
                    benchmark_suite_id,
                    exc,
                )

        return True

    except Exception as exc:
        if claim_conn is not None:
            claim_conn.rollback()
        logger.error("worker_lock_error: %s", exc)
        return False
    finally:
        if job_id:
            clear_aborted_job(job_id)
        if heartbeat_stop_event is not None:
            heartbeat_stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)
        if claim_conn is not None:
            claim_conn.close()
        reset_stage_context()
        reset_job_context(job_token)


def _run_pipeline(job_id: str, url: str, settings: dict) -> tuple[str | None, dict]:
    stage_cb = _stage_callback(job_id)
    stage_cb("ingest", "validating input parameters")
    enable_web_search = _resolve_enable_web_search(settings)
    enable_vision_board = _resolve_enable_vision_board(settings)
    enable_llm_frame = _resolve_enable_llm_frame(settings)
    express_mode = bool(settings.get("express_mode", False))
    pipeline_threads = get_pipeline_threads_per_job()
    generator = run_pipeline_job(
        job_id=job_id,
        src="Web URLs",
        urls=url,
        fldr="",
        cats=settings.get("categories", ""),
        p=settings.get("provider", "Ollama"),
        m=settings.get("model_name", "qwen3-vl:8b-instruct"),
        oe=settings.get("ocr_engine", "EasyOCR"),
        om=settings.get("ocr_mode", "🚀 Fast"),
        override=settings.get("override", False),
        sm=settings.get("scan_mode", "Tail Only"),
        enable_search=enable_web_search,
        enable_vision_board=enable_vision_board,
        enable_llm_frame=enable_llm_frame,
        ctx=settings.get("context_size", 8192),
        workers=pipeline_threads,
        express_mode=express_mode,
        stage_callback=stage_cb,
    )
    final_df = None
    latest_scores: dict = {}
    latest_per_frame_vision: list[dict] = []
    latest_ocr_text = ""
    latest_gallery = []
    for content in generator:
        if len(content) == 6:
            latest_scores = content[0] if isinstance(content[0], dict) else {}
            latest_per_frame_vision = content[1] if isinstance(content[1], list) else []
            latest_ocr_text = content[2] if isinstance(content[2], str) else ""
            latest_gallery = content[4] if isinstance(content[4], list) else []
            final_df = content[5]
        elif len(content) == 5:
            # Backward compatibility for older payload shapes.
            latest_scores = content[0] if isinstance(content[0], dict) else {}
            latest_ocr_text = content[1] if isinstance(content[1], str) else ""
            latest_gallery = content[3] if isinstance(content[3], list) else []
            final_df = content[4]

    artifacts_payload = _build_default_artifacts(job_id)
    artifacts_payload["latest_frames"] = _save_gallery_frames(job_id, latest_gallery)
    artifacts_payload["ocr_text"]["text"] = latest_ocr_text
    artifacts_payload["ocr_text"]["lines"] = [
        line for line in (latest_ocr_text or "").splitlines() if line.strip()
    ]
    artifacts_payload["ocr_text"]["url"] = _write_text_artifact(
        job_id, "ocr/ocr_output.txt", latest_ocr_text
    )
    artifacts_payload["per_frame_vision"] = latest_per_frame_vision
    artifacts_payload["vision_board"] = _vision_board_from_scores(latest_scores)

    if final_df is not None and not final_df.empty:
        result = json.dumps(final_df.to_dict(orient="records"))
        logger.info("pipeline_done: rows=%d", len(final_df))
        return result, artifacts_payload

    logger.warning("pipeline_empty: no result rows")
    return None, artifacts_payload


def _run_agent(job_id: str, url: str, settings: dict) -> tuple[str | None, list[str], dict]:
    events: list[str] = []
    stage_cb = _stage_callback(job_id)
    stage_cb("ingest", "validating input parameters")
    enable_web_search = _resolve_enable_web_search(settings)
    enable_vision_board = _resolve_enable_vision_board(settings)
    enable_llm_frame = _resolve_enable_llm_frame(settings)
    generator = run_agent_job(
        job_id=job_id,
        src="Web URLs",
        urls=url,
        fldr="",
        cats=settings.get("categories", ""),
        p=settings.get("provider", "Ollama"),
        m=settings.get("model_name", "qwen3-vl:8b-instruct"),
        oe=settings.get("ocr_engine", "EasyOCR"),
        om=settings.get("ocr_mode", "🚀 Fast"),
        override=settings.get("override", False),
        sm=settings.get("scan_mode", "Tail Only"),
        enable_search=enable_web_search,
        enable_vision_board=enable_vision_board,
        enable_llm_frame=enable_llm_frame,
        ctx=settings.get("context_size", 8192),
        stage_callback=stage_cb,
    )
    final_df = None
    latest_gallery = []
    latest_nebula = None
    for content in generator:
        if len(content) == 4:
            log_str, gallery, df, nebula = content
            events.append(log_str)
            final_df = df
            latest_gallery = gallery if isinstance(gallery, list) else latest_gallery
            latest_nebula = nebula
            agent_log = log_str or ""
            if len(agent_log) > 12000:
                agent_log = f"{agent_log[:12000]}\n...[truncated]"
            _append_job_event(
                job_id,
                f"{datetime.now(timezone.utc).isoformat()} agent:\n{agent_log}",
            )
            logger.debug("agent_event: events=%d", len(events))

    artifacts_payload = _build_default_artifacts(job_id)
    artifacts_payload["latest_frames"] = _save_gallery_frames(job_id, latest_gallery)
    ocr_text = _extract_agent_ocr_text(events)
    artifacts_payload["ocr_text"]["text"] = ocr_text
    artifacts_payload["ocr_text"]["lines"] = [line for line in ocr_text.split(" | ") if line.strip()]
    artifacts_payload["ocr_text"]["url"] = _write_text_artifact(
        job_id, "ocr/ocr_output.txt", ocr_text
    )
    artifacts_payload["vision_board"] = _vision_board_from_nebula(job_id, latest_nebula)

    result_json = None
    if final_df is not None and not final_df.empty:
        result_json = json.dumps(final_df.to_dict(orient="records"))
        logger.info("agent_done: rows=%d events=%d", len(final_df), len(events))
    else:
        logger.warning("agent_empty: no result rows")

    return result_json, events, artifacts_payload


def run_worker() -> None:
    process_count = _get_worker_process_count()
    if process_count > 1:
        _run_worker_supervisor(process_count)
        return
    _run_single_worker()


def _run_single_worker() -> None:
    logger.info(
        "worker_start: diagnostics=%s concurrency=%s",
        json.dumps(get_diagnostics()),
        json.dumps(get_concurrency_diagnostics()),
    )
    init_db()
    while True:
        try:
            processed = claim_and_process_job()
        except Exception as exc:
            logger.error("worker_loop_error: %s", exc)
            processed = False
        if not processed:
            time.sleep(1)


def _get_worker_process_count() -> int:
    return get_worker_processes_config()


def _worker_child_main(index: int) -> None:
    logger.info("worker_child_start: index=%d", index)
    _run_single_worker()


def _spawn_worker_child(index: int) -> multiprocessing.Process:
    return multiprocessing.Process(
        target=_worker_child_main,
        kwargs={"index": index},
        name=f"worker-{index}",
        daemon=False,
    )


def _run_worker_supervisor(process_count: int) -> None:
    logger.info("worker_supervisor_start: processes=%d", process_count)
    children: list[multiprocessing.Process] = []
    for index in range(1, process_count + 1):
        proc = _spawn_worker_child(index)
        proc.start()
        logger.info("worker_child_spawned: index=%d pid=%s", index, proc.pid)
        children.append(proc)

    try:
        while True:
            time.sleep(1.0)
            for idx, proc in enumerate(children, start=1):
                if proc.is_alive():
                    continue
                logger.warning(
                    "worker_child_exited: index=%d pid=%s exit_code=%s; restarting",
                    idx,
                    proc.pid,
                    proc.exitcode,
                )
                replacement = _spawn_worker_child(idx)
                replacement.start()
                logger.info("worker_child_spawned: index=%d pid=%s", idx, replacement.pid)
                children[idx - 1] = replacement
    except KeyboardInterrupt:
        logger.info("worker_supervisor_shutdown: terminating children")
    finally:
        for proc in children:
            if proc.is_alive():
                proc.terminate()
        for proc in children:
            proc.join(timeout=5.0)
            if proc.is_alive():
                logger.warning("worker_child_force_kill: pid=%s", proc.pid)
                proc.kill()
                proc.join(timeout=2.0)


if __name__ == "__main__":
    run_worker()
