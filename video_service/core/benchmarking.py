import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from video_service.db.database import get_db


def normalize_scan_mode(scan_strategy: str) -> str:
    value = (scan_strategy or "").strip().lower()
    if value in {"full", "full video", "full_video"}:
        return "Full Video"
    return "Tail Only"


def normalize_ocr_engine(engine: str) -> str:
    value = (engine or "").strip().lower()
    if value in {"microsoft", "florence", "florence-2", "florence-2 (microsoft)"}:
        return "Florence-2 (Microsoft)"
    return "EasyOCR"


def normalize_ocr_mode(mode: str) -> str:
    value = (mode or "").strip().lower()
    if "detail" in value:
        return "Detailed"
    return "Fast"


def jaccard_similarity(actual_categories: list[str], expected_categories: list[str]) -> float:
    actual_set = {c.strip().lower() for c in actual_categories if c and c.strip()}
    expected_set = {c.strip().lower() for c in expected_categories if c and c.strip()}
    if not actual_set and not expected_set:
        return 1.0
    if not actual_set or not expected_set:
        return 0.0
    inter = actual_set & expected_set
    union = actual_set | expected_set
    if not union:
        return 0.0
    return float(len(inter) / len(union))


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr
    return prev[-1]


def levenshtein_similarity(actual_text: str, expected_text: str) -> float:
    lhs = (actual_text or "").strip().lower()
    rhs = (expected_text or "").strip().lower()
    if not lhs and not rhs:
        return 1.0
    max_len = max(len(lhs), len(rhs))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(lhs, rhs)
    return max(0.0, min(1.0, 1.0 - (distance / max_len)))


def extract_stage_duration_seconds(events: list[str], fallback_duration: float | None) -> float | None:
    frame_extract_ts = None
    persist_ts = None

    for entry in events or []:
        text = str(entry or "")
        if "frame_extract:" in text and frame_extract_ts is None:
            match = re.match(r"^(?P<ts>[0-9T:\-+.Z]+)\s+", text)
            if match:
                try:
                    frame_extract_ts = datetime.fromisoformat(
                        match.group("ts").replace("Z", "+00:00")
                    )
                except Exception:
                    frame_extract_ts = None
        if "persist:" in text or "completed:" in text:
            match = re.match(r"^(?P<ts>[0-9T:\-+.Z]+)\s+", text)
            if match:
                try:
                    persist_ts = datetime.fromisoformat(
                        match.group("ts").replace("Z", "+00:00")
                    )
                except Exception:
                    persist_ts = persist_ts

    if frame_extract_ts and persist_ts:
        delta = (persist_ts - frame_extract_ts).total_seconds()
        if delta >= 0:
            return round(float(delta), 3)
    return fallback_duration


def _extract_job_categories(result_json: str | None) -> list[str]:
    if not result_json:
        return []
    try:
        parsed = json.loads(result_json)
    except Exception:
        return []
    if not isinstance(parsed, list) or not parsed:
        return []
    row = parsed[0] if isinstance(parsed[0], dict) else {}
    category = str(row.get("Category") or row.get("category") or "").strip()
    return [category] if category else []


def _extract_job_ocr_text(artifacts_json: str | None) -> str:
    if not artifacts_json:
        return ""
    try:
        parsed = json.loads(artifacts_json)
    except Exception:
        return ""
    if not isinstance(parsed, dict):
        return ""
    ocr = parsed.get("ocr_text")
    if isinstance(ocr, dict):
        return str(ocr.get("text") or "")
    return ""


def _extract_job_events(events_json: str | None) -> list[str]:
    if not events_json:
        return []
    try:
        parsed = json.loads(events_json)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _extract_processing_trace(artifacts_json: str | None) -> dict[str, Any] | None:
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
        "entity_search_rescue": "Entity Search Rescue",
        "specificity_search_rescue": "Specificity Search Rescue",
    }
    key = str(value or "").strip().lower()
    return labels.get(key, value.replace("_", " ").title() if value else "Unknown")


def evaluate_benchmark_suite(suite_id: str) -> dict[str, Any]:
    with get_db() as conn:
        suite = conn.execute(
            "SELECT * FROM benchmark_suites WHERE id = ?",
            (suite_id,),
        ).fetchone()
        if not suite:
            return {"ok": False, "error": "suite_not_found"}

        truth = conn.execute(
            "SELECT * FROM benchmark_truth WHERE id = ?",
            (suite["truth_id"],),
        ).fetchone()
        if not truth:
            return {"ok": False, "error": "truth_not_found"}

        expected_categories = []
        expected_ocr = ""
        try:
            expected_categories = json.loads(truth["expected_categories_json"] or "[]")
            if not isinstance(expected_categories, list):
                expected_categories = []
        except Exception:
            expected_categories = []
        expected_ocr = str(truth["expected_ocr_text"] or "")

        jobs = conn.execute(
            """
            SELECT id, status, result_json, artifacts_json, events, duration_seconds, benchmark_params_json
            FROM jobs
            WHERE benchmark_suite_id = ?
            ORDER BY created_at ASC
            """,
            (suite_id,),
        ).fetchall()

        total_jobs = len(jobs)
        completed_jobs = 0
        failed_jobs = 0
        evaluated_rows = []
        accepted_path_counts: Counter[str] = Counter()
        transit_path_counts: Counter[str] = Counter()
        path_titles: dict[str, str] = {}
        jobs_with_trace = 0

        with conn:
            conn.execute("DELETE FROM benchmark_result WHERE suite_id = ?", (suite_id,))

            for row in jobs:
                status = str(row["status"] or "")
                if status == "completed":
                    completed_jobs += 1
                if status == "failed":
                    failed_jobs += 1

                trace = _extract_processing_trace(row["artifacts_json"])
                if trace:
                    jobs_with_trace += 1
                    summary = trace.get("summary")
                    if isinstance(summary, dict):
                        accepted_type = str(summary.get("accepted_attempt_type") or "").strip()
                        if accepted_type:
                            accepted_path_counts[accepted_type] += 1
                            path_titles.setdefault(
                                accepted_type,
                                _humanize_attempt_type(accepted_type),
                            )
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

                if status != "completed":
                    continue

                actual_categories = _extract_job_categories(row["result_json"])
                actual_ocr_text = _extract_job_ocr_text(row["artifacts_json"])
                events = _extract_job_events(row["events"])

                class_accuracy = jaccard_similarity(actual_categories, expected_categories)
                ocr_accuracy = levenshtein_similarity(actual_ocr_text, expected_ocr)
                composite = (class_accuracy + ocr_accuracy) / 2.0
                duration = extract_stage_duration_seconds(events, row["duration_seconds"])

                params = {}
                try:
                    params = json.loads(row["benchmark_params_json"] or "{}")
                    if not isinstance(params, dict):
                        params = {}
                except Exception:
                    params = {}

                conn.execute(
                    """
                    INSERT INTO benchmark_result (
                        id, suite_id, job_id, duration_seconds,
                        classification_accuracy, ocr_accuracy,
                        composite_accuracy, params_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{suite_id}:{row['id']}",
                        suite_id,
                        row["id"],
                        duration,
                        round(class_accuracy, 6),
                        round(ocr_accuracy, 6),
                        round(composite, 6),
                        json.dumps(params),
                    ),
                )
                evaluated_rows.append(
                    {
                        "job_id": row["id"],
                        "duration_seconds": duration,
                        "classification_accuracy": class_accuracy,
                        "ocr_accuracy": ocr_accuracy,
                        "composite_accuracy": composite,
                        "params": params,
                    }
                )

            is_finished = total_jobs > 0 and (completed_jobs + failed_jobs == total_jobs)
            suite_status = "completed" if is_finished else "running"
            conn.execute(
                """
                UPDATE benchmark_suites
                SET status = ?, completed_jobs = ?, failed_jobs = ?, total_jobs = ?,
                    evaluated_at = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    suite_status,
                    completed_jobs,
                    failed_jobs,
                    total_jobs,
                    datetime.now(timezone.utc).isoformat() if is_finished else None,
                    suite_id,
                ),
            )

    return {
        "ok": True,
        "suite_id": suite_id,
        "status": "completed" if (total_jobs > 0 and completed_jobs + failed_jobs == total_jobs) else "running",
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "results": evaluated_rows,
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
