import time
import os
import re
from collections.abc import Callable
import torch
import cv2
import numpy as np
import pandas as pd
import concurrent.futures
from contextvars import copy_context
from PIL import Image
from video_service.core.logging_setup import bind_current_log_context
from video_service.core.utils import logger, device, TORCH_DTYPE
from video_service.core.video_io import (
    extract_express_brand_frame,
    extract_frames_for_pipeline,
    extract_middle_frame,
    extract_tail_rescue_frames,
    get_pil_image,
    resolve_urls,
)
from video_service.core import categories as categories_runtime
from video_service.core.categories import category_mapper, normalize_feature_tensor
from video_service.core.category_mapping import (
    _looks_ambiguous_product_family_category,
    _looks_generic_freeform_category,
    build_product_cue_query_text,
)
from video_service.core.ocr import ocr_manager
from video_service.core.llm import llm_engine, create_provider

RESULT_COLUMNS = [
    "URL / Path",
    "Brand",
    "Category ID",
    "Category",
    "Confidence",
    "Reasoning",
    "category_match_method",
    "category_match_score",
]


def _normalize_ocr(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", (text or "").lower()).strip()


def _ocr_texts_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    if not a and not b:
        return True
    if not a or not b:
        return False
    words_a, words_b = set(a.split()), set(b.split())
    if not words_a and not words_b:
        return True
    union = words_a | words_b
    if not union:
        return True
    return (len(words_a & words_b) / len(union)) >= threshold


def _resolve_ocr_dedup_threshold() -> float:
    raw = os.environ.get("OCR_DEDUP_THRESHOLD", "0.85")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_dedup_threshold value=%r fallback=0.85", raw)
        return 0.85
    return max(0.0, min(1.0, value))


def _resolve_ocr_frame_similarity_threshold() -> float:
    raw = os.environ.get("OCR_FRAME_SIMILARITY_THRESHOLD", "0.985")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_frame_similarity_threshold value=%r fallback=0.985", raw)
        return 0.985
    return max(0.0, min(1.0, value))


def _resolve_ocr_prefilter_preserve_last_frames() -> int:
    raw = os.environ.get("OCR_PREFILTER_PRESERVE_LAST_FRAMES", "3")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_prefilter_preserve_last_frames value=%r fallback=3", raw)
        return 3


def _frame_hist_signature(frame_bgr: np.ndarray) -> np.ndarray:
    thumbnail = cv2.resize(frame_bgr, (96, 54), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    return cv2.normalize(hist, None).flatten()


def _frames_visually_similar(a_bgr: np.ndarray, b_bgr: np.ndarray, threshold: float) -> bool:
    score = float(
        cv2.compareHist(
            _frame_hist_signature(a_bgr),
            _frame_hist_signature(b_bgr),
            cv2.HISTCMP_CORREL,
        )
    )
    return score >= threshold


def _select_frames_for_ocr(frames: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    if len(frames) <= 2:
        return list(frames), 0

    similarity_threshold = _resolve_ocr_frame_similarity_threshold()
    preserve_recent = _resolve_ocr_prefilter_preserve_last_frames()
    tail_like_types = {"tail", "backward_ext", "tail_rescue"}
    preserve_recent_for_tail = any(
        str(frame.get("type", "") or "").lower() in tail_like_types for frame in frames
    )
    protected_indices: set[int] = set()
    if preserve_recent_for_tail and preserve_recent > 0:
        protected_start = max(0, len(frames) - preserve_recent)
        protected_indices = set(range(protected_start, len(frames)))

    selected: list[dict[str, object]] = [frames[0]]
    skipped = 0
    last_selected = frames[0]
    last_index = len(frames) - 1

    for idx, frame in enumerate(frames[1:], start=1):
        if idx in protected_indices:
            if frame is not selected[-1]:
                selected.append(frame)
                last_selected = frame
            continue
        is_last_frame = idx == last_index
        if is_last_frame:
            if frame is not selected[-1]:
                selected.append(frame)
            continue

        current_image = frame.get("ocr_image")
        last_image = last_selected.get("ocr_image")
        if not isinstance(current_image, np.ndarray) or not isinstance(last_image, np.ndarray):
            selected.append(frame)
            last_selected = frame
            continue

        if _frames_visually_similar(last_image, current_image, similarity_threshold):
            skipped += 1
            logger.debug(
                "ocr_frame_prefilter_skip: frame at %.1fs similar to previous threshold=%.3f",
                float(frame.get("time", 0.0)),
                similarity_threshold,
            )
            continue

        selected.append(frame)
        last_selected = frame

    if len(selected) == 2 and not protected_indices:
        first_image = selected[0].get("ocr_image")
        last_image = selected[1].get("ocr_image")
        if isinstance(first_image, np.ndarray) and isinstance(last_image, np.ndarray):
            if _frames_visually_similar(first_image, last_image, similarity_threshold):
                skipped += 1
                logger.debug(
                    "ocr_frame_prefilter_collapse: dropping earlier representative at %.1fs in favor of last frame %.1fs threshold=%.3f",
                    float(selected[0].get("time", 0.0)),
                    float(selected[1].get("time", 0.0)),
                    similarity_threshold,
                )
                selected = [selected[1]]

    logger.info(
        "ocr_frame_prefilter: selected=%d skipped=%d total_frames=%d threshold=%.3f",
        len(selected),
        skipped,
        len(frames),
        similarity_threshold,
    )
    return selected, skipped


def _ocr_roi_enabled(engine_name: str) -> bool:
    raw = os.environ.get("OCR_ROI_FIRST", "true").strip().lower()
    return engine_name == "EasyOCR" and raw not in {"0", "false", "no", "off"}


def _ocr_skip_no_roi_enabled(engine_name: str, scan_mode: str) -> bool:
    raw = os.environ.get("OCR_SKIP_NO_ROI_FRAMES", "true").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    mode = (scan_mode or "").strip().lower()
    return engine_name == "EasyOCR" and mode not in {"full video", "full scan"}


def _extract_ocr_focus_region(frame_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    height, width = frame_bgr.shape[:2]
    if height < 120 or width < 120:
        return frame_bgr, False

    scale = min(1.0, 640.0 / float(max(height, width)))
    if scale < 1.0:
        work = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = frame_bgr

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)
    blurred = cv2.GaussianBlur(grad_x, (3, 3), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(15, work.shape[1] // 12), max(3, work.shape[0] // 40)),
    )
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    connected = cv2.dilate(connected, None, iterations=1)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame_bgr, False

    min_area = float(work.shape[0] * work.shape[1]) * 0.002
    candidate_boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < min_area:
            continue
        if w / max(h, 1) < 1.2:
            continue
        candidate_boxes.append((x, y, x + w, y + h))

    if not candidate_boxes:
        return frame_bgr, False

    x1 = min(box[0] for box in candidate_boxes)
    y1 = min(box[1] for box in candidate_boxes)
    x2 = max(box[2] for box in candidate_boxes)
    y2 = max(box[3] for box in candidate_boxes)

    pad_x = max(8, int((x2 - x1) * 0.08))
    pad_y = max(8, int((y2 - y1) * 0.12))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(work.shape[1], x2 + pad_x)
    y2 = min(work.shape[0], y2 + pad_y)

    if scale < 1.0:
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    roi_area_ratio = ((x2 - x1) * (y2 - y1)) / float(width * height)
    if roi_area_ratio <= 0.03 or roi_area_ratio >= 0.9:
        return frame_bgr, False

    return frame_bgr[y1:y2, x1:x2], True


def _ocr_text_has_signal(text: str) -> bool:
    cleaned = re.sub(r"\[HUGE\]", " ", text or "", flags=re.IGNORECASE)
    tokens = re.findall(r"[a-z0-9.]{2,}", cleaned.lower())
    if not tokens:
        return False
    return len("".join(tokens)) >= 4


def _ocr_early_stop_enabled(scan_mode: str) -> bool:
    raw = os.environ.get("OCR_EARLY_STOP_ENABLED", "true").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    mode = (scan_mode or "").strip().lower()
    return mode not in {"full video", "full scan"}


def _resolve_ocr_early_stop_min_chars() -> int:
    raw = os.environ.get("OCR_EARLY_STOP_MIN_CHARS", "12")
    try:
        return max(4, int(raw))
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_early_stop_min_chars value=%r fallback=12", raw)
        return 12


def _ocr_edge_rescue_enabled(scan_mode: str, express_mode: bool) -> bool:
    raw = os.environ.get("OCR_EDGE_RESCUE_ENABLED", "true").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if express_mode:
        return False
    mode = (scan_mode or "").strip().lower()
    return mode not in {"full video", "full scan"}


def _ocr_context_rescue_enabled(scan_mode: str, express_mode: bool) -> bool:
    raw = os.environ.get("OCR_CONTEXT_RESCUE_ENABLED", "true").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if express_mode:
        return False
    mode = (scan_mode or "").strip().lower()
    return mode not in {"full video", "full scan"}


def _express_rescue_enabled() -> bool:
    raw = os.environ.get("EXPRESS_RESCUE_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _extended_tail_rescue_enabled() -> bool:
    raw = os.environ.get("OCR_EXTENDED_TAIL_RESCUE_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _full_video_rescue_enabled() -> bool:
    raw = os.environ.get("OCR_FULL_VIDEO_RESCUE_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _resolve_extended_tail_window_seconds() -> int:
    raw = os.environ.get("OCR_EXTENDED_TAIL_WINDOW_SECONDS", "30")
    try:
        return max(10, int(raw))
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_extended_tail_window_seconds value=%r fallback=30", raw)
        return 30


def _resolve_extended_tail_step_seconds() -> float:
    raw = os.environ.get("OCR_EXTENDED_TAIL_STEP_SECONDS", "2.0")
    try:
        return max(0.5, float(raw))
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_extended_tail_step_seconds value=%r fallback=2.0", raw)
        return 2.0


def _resolve_full_video_rescue_max_frames() -> int:
    raw = os.environ.get("OCR_FULL_VIDEO_RESCUE_MAX_FRAMES", "24")
    try:
        return max(4, int(raw))
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_full_video_rescue_max_frames value=%r fallback=24", raw)
        return 24


def _resolve_rescue_ocr_mode(engine_name: str, current_mode: str) -> str:
    if engine_name == "EasyOCR":
        return "Detailed"
    return current_mode


def _resolve_ocr_context_confidence_threshold() -> float:
    raw = os.environ.get("OCR_CONTEXT_CONFIDENCE_THRESHOLD", "0.80")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_context_confidence_threshold value=%r fallback=0.80", raw)
        return 0.80
    return max(0.0, min(1.0, value))


def _resolve_ocr_context_short_chars() -> int:
    raw = os.environ.get("OCR_CONTEXT_SHORT_TEXT_CHARS", "48")
    try:
        return max(16, int(raw))
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_context_short_text_chars value=%r fallback=48", raw)
        return 48


def _resolve_ocr_context_sparse_tokens() -> int:
    raw = os.environ.get("OCR_CONTEXT_SPARSE_TOKENS", "6")
    try:
        return max(2, int(raw))
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_context_sparse_tokens value=%r fallback=6", raw)
        return 6


def _resolve_ocr_context_max_lines() -> int:
    raw = os.environ.get("OCR_CONTEXT_MAX_LINES", "8")
    try:
        return max(3, int(raw))
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_context_max_lines value=%r fallback=8", raw)
        return 8


def _resolve_ocr_context_max_chars() -> int:
    raw = os.environ.get("OCR_CONTEXT_MAX_CHARS", "600")
    try:
        return max(160, int(raw))
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_context_max_chars value=%r fallback=600", raw)
        return 600


def _resolve_ocr_context_vision_score_threshold() -> float:
    raw = os.environ.get("OCR_CONTEXT_VISION_SCORE_THRESHOLD", "0.10")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_context_vision_score_threshold value=%r fallback=0.10", raw)
        return 0.10
    return max(0.0, min(1.0, value))


def _ocr_context_use_vision_assist() -> bool:
    raw = os.environ.get("OCR_CONTEXT_USE_VISION_ASSIST", "false").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _resolve_ocr_context_vision_margin_threshold() -> float:
    raw = os.environ.get("OCR_CONTEXT_VISION_MARGIN_THRESHOLD", "0.03")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_context_vision_margin_threshold value=%r fallback=0.03", raw)
        return 0.03
    return max(0.0, min(1.0, value))


def _resolve_ocr_context_mapper_score_threshold() -> float:
    raw = os.environ.get("OCR_CONTEXT_MAPPER_SCORE_THRESHOLD", "0.82")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_context_mapper_score_threshold value=%r fallback=0.82", raw)
        return 0.82
    return max(0.0, min(1.0, value))


def _resolve_ocr_support_score_threshold() -> float:
    raw = os.environ.get("OCR_SUPPORT_SCORE_THRESHOLD", "0.72")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_support_score_threshold value=%r fallback=0.72", raw)
        return 0.72
    return max(0.0, min(1.0, value))


def _specificity_search_rescue_enabled(enable_search: bool, express_mode: bool) -> bool:
    raw = os.environ.get("SPECIFICITY_SEARCH_RESCUE_ENABLED", "true").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if express_mode:
        return False
    return bool(enable_search)


def _resolve_specificity_search_mapper_threshold() -> float:
    raw = os.environ.get("SPECIFICITY_SEARCH_MAPPER_SCORE_THRESHOLD", "0.65")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid_specificity_search_mapper_score_threshold value=%r fallback=0.65", raw)
        return 0.65
    return max(0.0, min(1.0, value))


def _resolve_specificity_search_vision_threshold() -> float:
    raw = os.environ.get("SPECIFICITY_SEARCH_VISION_SCORE_THRESHOLD", "0.55")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid_specificity_search_vision_score_threshold value=%r fallback=0.55", raw)
        return 0.55
    return max(0.0, min(1.0, value))


def _specificity_search_broad_categories() -> set[str]:
    raw = os.environ.get(
        "SPECIFICITY_SEARCH_BROAD_CATEGORIES",
        "Financial Services,Retail,Technology,Educational,Banking,Insurance,Automotive,Telecommunications,Travel,Healthcare,Nonprofit Institutions",
    )
    values = {value.strip() for value in raw.split(",") if value.strip()}
    return values or {"Financial Services"}


def _specificity_search_generic_raw_categories() -> set[str]:
    raw = os.environ.get(
        "SPECIFICITY_SEARCH_GENERIC_RAW_CATEGORIES",
        "Movie,Film,Entertainment,Financial Services,Banking,Insurance,Retail,Technology,Educational,Healthcare,Travel,Automotive",
    )
    values = {value.strip() for value in raw.split(",") if value.strip()}
    return values or {"Movie"}


def _is_valid_search_domain(domain: str) -> bool:
    labels = [label for label in (domain or "").strip().lower().split(".") if label]
    if len(labels) < 2:
        return False
    if labels[0] in {"www", "m", "amp"} and len(labels) < 3:
        return False
    tld = labels[-1]
    if not re.fullmatch(r"[a-z]{2,24}", tld):
        return False
    if len(labels) == 2 and len(tld) > 10:
        return False
    return True


def _extract_ocr_domains(text: str) -> list[str]:
    matches = re.findall(
        r"\b((?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[a-z0-9/_-]+)?)\b",
        text or "",
        flags=re.IGNORECASE,
    )
    return list(
        dict.fromkeys(
            match for match in matches if match and _is_valid_search_domain(match.split("/", 1)[0])
        )
    )


def _top_visual_matches(sorted_vision: dict[str, float], limit: int = 3) -> list[tuple[str, float]]:
    return list(sorted_vision.items())[:limit]


def _looks_broad_family_taxonomy_label(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return False
    if normalized.endswith("- all else"):
        return True

    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    if not tokens:
        return False

    broad_tokens = {
        "cleaner",
        "cleaners",
        "product",
        "products",
        "provider",
        "providers",
        "service",
        "services",
        "store",
        "stores",
    }
    if tokens & broad_tokens:
        return True

    exact_broad_labels = {
        "general merchandise",
        "household cleaners",
        "household products",
        "media",
        "pharmaceuticals",
        "retail",
    }
    return normalized in exact_broad_labels


def _local_family_evidence_preference(
    *,
    current_canonical: str,
    primary_candidates: list[tuple[str, float]],
    evidence_neighbors: list[tuple[str, float]],
) -> str | None:
    if not _looks_broad_family_taxonomy_label(current_canonical):
        return None
    if not primary_candidates or not evidence_neighbors:
        return None

    evidence_top_label, evidence_top_score_raw = evidence_neighbors[0]
    evidence_top_label = str(evidence_top_label or "").strip()
    if not evidence_top_label or evidence_top_label == current_canonical:
        return None
    if _looks_broad_family_taxonomy_label(evidence_top_label):
        return None

    local_primary_labels = {
        str(label or "").strip()
        for label, _score in primary_candidates[:3]
        if str(label or "").strip()
    }
    if evidence_top_label not in local_primary_labels:
        return None

    try:
        evidence_top_score = float(evidence_top_score_raw or 0.0)
    except (TypeError, ValueError):
        evidence_top_score = 0.0
    if evidence_top_score < _resolve_category_rerank_evidence_score_threshold():
        return None

    return f"family_evidence_prefers={evidence_top_label!r}@{evidence_top_score:.4f}"


def _resolve_category_rerank_local_family_gap_threshold() -> float:
    raw = os.environ.get("CATEGORY_RERANK_LOCAL_FAMILY_GAP_THRESHOLD", "0.08")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        logger.warning("invalid_category_rerank_local_family_gap_threshold value=%r fallback=0.08", raw)
        return 0.08


def _local_family_primary_preference(
    *,
    current_canonical: str,
    raw_category: str,
    exact_taxonomy_match: bool,
    primary_candidates: list[tuple[str, float]],
) -> str | None:
    if exact_taxonomy_match:
        return None
    if not _looks_broad_family_taxonomy_label(current_canonical):
        return None
    if len(primary_candidates) < 2:
        return None

    try:
        top1_score = float(primary_candidates[0][1] or 0.0)
    except (TypeError, ValueError):
        top1_score = 0.0

    threshold = _resolve_category_rerank_evidence_score_threshold()
    gap_threshold = _resolve_category_rerank_local_family_gap_threshold()
    for label, score_raw in primary_candidates[1:4]:
        candidate_label = str(label or "").strip()
        if not candidate_label or candidate_label == current_canonical:
            continue
        if _looks_broad_family_taxonomy_label(candidate_label):
            continue
        try:
            candidate_score = float(score_raw or 0.0)
        except (TypeError, ValueError):
            candidate_score = 0.0
        if candidate_score < threshold:
            continue
        if (top1_score - candidate_score) > gap_threshold:
            continue
        return (
            f"family_primary_prefers={candidate_label!r}@{candidate_score:.4f}"
            f";raw={raw_category!r}"
        )
    return None


_CATEGORY_RERANK_OVERLAP_STOPWORDS = {
    "all",
    "and",
    "category",
    "consumer",
    "else",
    "for",
    "goods",
    "home",
    "industry",
    "item",
    "label",
    "manufacture",
    "medication",
    "of",
    "or",
    "other",
    "over",
    "product",
    "products",
    "sale",
    "service",
    "services",
    "store",
    "stores",
    "the",
}


def _normalize_category_overlap_token(token: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "", (token or "").lower())
    if len(clean) > 4 and clean.endswith("ies"):
        clean = f"{clean[:-3]}y"
    elif len(clean) > 4 and clean.endswith("s") and not clean.endswith("ss"):
        clean = clean[:-1]
    return clean


def _category_overlap_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in re.findall(r"[A-Za-zÀ-ÿ0-9]{3,}", str(text or "")):
        token = _normalize_category_overlap_token(raw_token)
        if len(token) < 3 or token in _CATEGORY_RERANK_OVERLAP_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _resolve_category_rerank_freeform_mismatch_score_threshold() -> float:
    raw = os.environ.get("CATEGORY_RERANK_FREEFORM_MISMATCH_SCORE_THRESHOLD", "0.78")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        logger.warning(
            "invalid_category_rerank_freeform_mismatch_score_threshold value=%r fallback=0.78",
            raw,
        )
        return 0.78


def _freeform_label_mismatch_reason(
    *,
    current_canonical: str,
    raw_category: str,
    exact_taxonomy_match: bool,
    primary_candidates: list[tuple[str, float]],
) -> str | None:
    if exact_taxonomy_match:
        return None
    if not current_canonical or not raw_category or not primary_candidates:
        return None
    if _looks_generic_freeform_category(raw_category):
        return None
    if _looks_ambiguous_product_family_category(raw_category):
        return None

    raw_tokens = _category_overlap_tokens(raw_category)
    if len(raw_tokens) < 2:
        return None
    if raw_tokens & _category_overlap_tokens(current_canonical):
        return None

    try:
        top1_score = float(primary_candidates[0][1] or 0.0)
    except (TypeError, ValueError):
        top1_score = 0.0
    if top1_score >= _resolve_category_rerank_freeform_mismatch_score_threshold():
        return None

    return (
        f"freeform_label_mismatch raw={raw_category!r};mapped={current_canonical!r};"
        f"top1_score={top1_score:.4f}"
    )


def _category_rerank_enabled() -> bool:
    raw = os.environ.get("CATEGORY_RERANK_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _resolve_category_rerank_top1_score_threshold() -> float:
    raw = os.environ.get("CATEGORY_RERANK_TOP1_SCORE_THRESHOLD", "0.62")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        logger.warning("invalid_category_rerank_top1_score_threshold value=%r fallback=0.62", raw)
        return 0.62


def _resolve_category_rerank_top2_gap_threshold() -> float:
    raw = os.environ.get("CATEGORY_RERANK_TOP2_GAP_THRESHOLD", "0.02")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        logger.warning("invalid_category_rerank_top2_gap_threshold value=%r fallback=0.02", raw)
        return 0.02


def _resolve_category_rerank_top3_gap_threshold() -> float:
    raw = os.environ.get("CATEGORY_RERANK_TOP3_GAP_THRESHOLD", "0.03")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        logger.warning("invalid_category_rerank_top3_gap_threshold value=%r fallback=0.03", raw)
        return 0.03


def _resolve_category_rerank_evidence_score_threshold() -> float:
    raw = os.environ.get("CATEGORY_RERANK_EVIDENCE_SCORE_THRESHOLD", "0.58")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        logger.warning("invalid_category_rerank_evidence_score_threshold value=%r fallback=0.58", raw)
        return 0.58


def _resolve_category_rerank_visual_score_threshold() -> float:
    raw = os.environ.get("CATEGORY_RERANK_VISUAL_SCORE_THRESHOLD", "0.55")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        logger.warning("invalid_category_rerank_visual_score_threshold value=%r fallback=0.55", raw)
        return 0.55


def _build_category_rerank_candidates(
    *,
    raw_category: str,
    current_match: dict[str, object],
    predicted_brand: str,
    ocr_text: str,
    reasoning: str = "",
    primary_limit: int = 5,
    evidence_limit: int = 3,
    combined_limit: int = 8,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[tuple[str, float]]]:
    if not hasattr(category_mapper, "get_mapper_neighbor_categories"):
        return [], [], []

    def _merge_ranked_candidates(
        *candidate_lists: list[tuple[str, float]],
        limit: int,
    ) -> list[tuple[str, float]]:
        merged: list[tuple[str, float]] = []
        seen: set[str] = set()
        for candidates in candidate_lists:
            for label, score in candidates:
                canonical_label = str(label or "").strip()
                if not canonical_label:
                    continue
                key = canonical_label.casefold()
                if key in seen:
                    continue
                seen.add(key)
                try:
                    numeric_score = float(score)
                except (TypeError, ValueError):
                    numeric_score = 0.0
                merged.append((canonical_label, numeric_score))
                if len(merged) >= limit:
                    return merged
        return merged

    try:
        primary_candidates = list(
            category_mapper.get_mapper_neighbor_categories(
                raw_category=raw_category,
                predicted_brand=predicted_brand,
                ocr_summary=ocr_text,
                reasoning_summary=reasoning,
                top_k=primary_limit,
            )
        )
    except Exception as exc:
        logger.warning("category_rerank_candidates_failed: %s", exc)
        return [], [], []

    canonical = str(current_match.get("canonical_category", "") or "").strip()
    if canonical and canonical not in {label for label, _score in primary_candidates}:
        score_raw = current_match.get("category_match_score", 0.0)
        try:
            score = float(score_raw or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        primary_candidates.append((canonical, score))

    evidence_neighbors: list[tuple[str, float]] = []
    evidence_query = _build_category_rerank_evidence_query(
        brand=predicted_brand,
        ocr_text=ocr_text,
        reasoning=reasoning,
        family_context=raw_category,
    )
    if evidence_query:
        try:
            evidence_neighbors = list(
                category_mapper.get_mapper_neighbor_categories(
                    raw_category=evidence_query,
                    predicted_brand="",
                    ocr_summary="",
                    top_k=evidence_limit,
                )
            )
        except Exception as exc:
            logger.warning("category_rerank_evidence_neighbors_failed: %s", exc)
            evidence_neighbors = []

    return (
        _merge_ranked_candidates(primary_candidates, evidence_neighbors, limit=combined_limit),
        evidence_neighbors,
        primary_candidates,
    )


def _build_category_rerank_evidence_query(
    *,
    brand: str,
    ocr_text: str,
    reasoning: str,
    family_context: str = "",
) -> str:
    compact_cues = build_product_cue_query_text(
        predicted_brand=brand,
        ocr_summary=ocr_text,
        reasoning_summary=reasoning,
        family_context=family_context,
        max_chars=260,
    )
    if compact_cues:
        return compact_cues

    parts: list[str] = []
    brand_text = str(brand or "").strip()
    if brand_text and brand_text.lower() not in {"unknown", "none", "n/a"}:
        parts.append(brand_text)

    family_text = str(family_context or "").strip()
    if family_text and family_text.lower() not in {"unknown", "none", "n/a"}:
        parts.append(family_text[:120])

    compact_ocr = " ".join((ocr_text or "").split())
    if compact_ocr:
        parts.append(compact_ocr[:260])

    compact_reasoning = " ".join((reasoning or "").split())
    if compact_reasoning:
        parts.append(compact_reasoning[:260])

    return "\n".join(parts).strip()


def _should_run_category_rerank(
    *,
    result_payload: dict[str, object] | None,
    category_match: dict[str, object],
    ocr_text: str,
    sorted_vision: dict[str, float],
) -> tuple[bool, str, list[str], list[tuple[str, float]]]:
    if not _category_rerank_enabled():
        return False, "disabled", [], []
    if not isinstance(result_payload, dict):
        return False, "invalid_result", [], []
    if str(category_match.get("category_match_method", "") or "") != "embeddings":
        return False, "mapping_not_embeddings", [], []

    raw_category = str(result_payload.get("category", "") or "").strip()
    canonical = str(category_match.get("canonical_category", "") or "").strip()
    brand = str(result_payload.get("brand", "") or "").strip()
    reasoning = str(result_payload.get("reasoning", "") or "").strip()
    if not raw_category or not canonical:
        return False, "missing_category", [], []

    mapper_candidates, evidence_neighbors, primary_candidates = _build_category_rerank_candidates(
        raw_category=raw_category,
        current_match=category_match,
        predicted_brand=brand,
        ocr_text=ocr_text,
        reasoning=reasoning,
    )
    candidate_categories = [label for label, _score in mapper_candidates]
    if len(candidate_categories) < 2:
        return False, "insufficient_candidates", candidate_categories, []

    uncertainty_reasons: list[str] = []
    top1_score = float(mapper_candidates[0][1])
    top2_score = float(mapper_candidates[1][1]) if len(mapper_candidates) > 1 else 0.0
    top3_score = float(mapper_candidates[2][1]) if len(mapper_candidates) > 2 else top2_score
    if top1_score < _resolve_category_rerank_top1_score_threshold():
        uncertainty_reasons.append(f"top1_score={top1_score:.4f}")
    if (top1_score - top2_score) < _resolve_category_rerank_top2_gap_threshold():
        uncertainty_reasons.append(f"top1_top2_gap={top1_score - top2_score:.4f}")
    if (top1_score - top3_score) < _resolve_category_rerank_top3_gap_threshold():
        uncertainty_reasons.append(f"top1_top3_gap={top1_score - top3_score:.4f}")

    family_preference_reason = _local_family_evidence_preference(
        current_canonical=canonical,
        primary_candidates=primary_candidates,
        evidence_neighbors=evidence_neighbors,
    )
    exact_taxonomy_match = hasattr(category_mapper, "categories") and raw_category in set(
        getattr(category_mapper, "categories", []) or []
    )
    family_primary_reason = _local_family_primary_preference(
        current_canonical=canonical,
        raw_category=raw_category,
        exact_taxonomy_match=exact_taxonomy_match,
        primary_candidates=primary_candidates,
    )
    freeform_mismatch_reason = _freeform_label_mismatch_reason(
        current_canonical=canonical,
        raw_category=raw_category,
        exact_taxonomy_match=exact_taxonomy_match,
        primary_candidates=primary_candidates,
    )
    if (
        not uncertainty_reasons
        and not family_preference_reason
        and not family_primary_reason
        and not freeform_mismatch_reason
    ):
        return False, "mapping_confident", candidate_categories, []

    contradiction_reasons: list[str] = []
    if not exact_taxonomy_match:
        contradiction_reasons.append("freeform_category_with_weak_mapping")

    if evidence_neighbors:
        evidence_top_label, evidence_top_score = evidence_neighbors[0]
        if (
            evidence_top_label != canonical
            and float(evidence_top_score) >= _resolve_category_rerank_evidence_score_threshold()
        ):
            contradiction_reasons.append(
                f"evidence_prefers={evidence_top_label!r}@{float(evidence_top_score):.4f}"
            )

    visual_matches = _top_visual_matches(sorted_vision, limit=3)
    if visual_matches:
        visual_top_label, visual_top_score = visual_matches[0]
        if (
            visual_top_label != canonical
            and float(visual_top_score) >= _resolve_category_rerank_visual_score_threshold()
        ):
            contradiction_reasons.append(
                f"vision_prefers={visual_top_label!r}@{float(visual_top_score):.4f}"
            )

    if not contradiction_reasons:
        if family_preference_reason:
            return True, family_preference_reason, candidate_categories, visual_matches
        if family_primary_reason:
            return True, family_primary_reason, candidate_categories, visual_matches
        if freeform_mismatch_reason:
            return True, freeform_mismatch_reason, candidate_categories, visual_matches
        return False, ";".join(uncertainty_reasons), candidate_categories, visual_matches

    reason_parts = [*uncertainty_reasons]
    if family_preference_reason:
        reason_parts.append(family_preference_reason)
    if family_primary_reason:
        reason_parts.append(family_primary_reason)
    if freeform_mismatch_reason:
        reason_parts.append(freeform_mismatch_reason)
    return (
        True,
        ";".join([*reason_parts, *contradiction_reasons]),
        candidate_categories,
        visual_matches,
    )


def _accept_category_rerank_result(
    current_match: dict[str, object],
    reranked_match: dict[str, object],
    candidate_categories: list[str],
) -> tuple[bool, str]:
    current_canonical = str(current_match.get("canonical_category", "") or "").strip()
    reranked_canonical = str(reranked_match.get("canonical_category", "") or "").strip()
    if not reranked_canonical:
        return False, "reranked_category_empty"
    if candidate_categories and reranked_canonical not in candidate_categories:
        return False, f"outside_candidate_set={reranked_canonical!r}"
    if reranked_canonical == current_canonical:
        return False, f"unchanged_category={reranked_canonical!r}"
    return True, f"reranked_to={reranked_canonical!r}"


def _exact_taxonomy_match_from_label(label: str) -> dict[str, object] | None:
    canonical = str(label or "").strip()
    categories = set(getattr(category_mapper, "categories", []) or [])
    if not canonical or canonical not in categories:
        return None
    cat_to_id = getattr(category_mapper, "cat_to_id", {}) or {}
    category_id = str(cat_to_id.get(canonical, "") or "")
    if not category_id:
        map_fn = getattr(category_mapper, "map_category", None)
        if callable(map_fn):
            try:
                exact_match = map_fn(
                    raw_category=canonical,
                    suggested_categories_text="",
                    predicted_brand="",
                    ocr_summary="",
                    reasoning_summary="",
                )
            except Exception:
                exact_match = None
            if isinstance(exact_match, dict):
                exact_canonical = str(exact_match.get("canonical_category", "") or "").strip()
                if exact_canonical == canonical:
                    category_id = str(exact_match.get("category_id", "") or "")
    return {
        "canonical_category": canonical,
        "category_id": category_id,
        "category_match_method": "exact_rerank_label",
        "category_match_score": 1.0,
        "mapping_query_text": canonical,
        "top_matches": [],
    }


def _build_specificity_search_candidates(
    *,
    raw_category: str,
    current_match: dict[str, object],
    predicted_brand: str,
    ocr_text: str,
    sorted_vision: dict[str, float],
    mapper_neighbor_limit: int = 6,
    visual_limit: int = 3,
) -> list[str]:
    candidates: list[str] = []

    def _append(label: str) -> None:
        clean = str(label or "").strip()
        if clean and clean not in candidates:
            candidates.append(clean)

    _append(str(current_match.get("canonical_category", "") or ""))

    if hasattr(category_mapper, "get_mapper_neighbor_categories"):
        try:
            for label, _score in category_mapper.get_mapper_neighbor_categories(
                raw_category=raw_category,
                predicted_brand=predicted_brand,
                ocr_summary=ocr_text,
                top_k=mapper_neighbor_limit,
            ):
                _append(label)
        except Exception as exc:
            logger.warning("specificity_search_mapper_neighbors_failed: %s", exc)

    for label, _score in _top_visual_matches(sorted_vision, limit=visual_limit):
        _append(label)

    return candidates


def _should_run_specificity_search_rescue(
    enable_search: bool,
    express_mode: bool,
    result_payload: dict[str, object] | None,
    category_match: dict[str, object],
    ocr_text: str,
    sorted_vision: dict[str, float],
) -> tuple[bool, str]:
    if not _specificity_search_rescue_enabled(enable_search, express_mode):
        return False, "disabled"
    if not isinstance(result_payload, dict):
        return False, "invalid_result"

    brand = str(result_payload.get("brand", "") or "").strip()
    raw_category = str(result_payload.get("category", "") or "").strip()
    canonical = str(category_match.get("canonical_category", "") or "").strip()
    if not brand or brand.lower() in {"unknown", "none", "n/a"}:
        return False, "brand_missing"

    broad_canonical = canonical in _specificity_search_broad_categories()
    generic_raw = raw_category in _specificity_search_generic_raw_categories()
    if not broad_canonical and not generic_raw:
        return False, f"category_not_generic raw={raw_category!r} canonical={canonical!r}"

    score_raw = category_match.get("category_match_score", 0.0)
    try:
        mapper_score = float(score_raw or 0.0)
    except (TypeError, ValueError):
        mapper_score = 0.0

    category_descriptor = f"raw={raw_category!r};canonical={canonical!r}"
    domain_hints = _extract_ocr_domains(ocr_text)
    mapper_threshold = _resolve_specificity_search_mapper_threshold()
    if mapper_score < mapper_threshold and domain_hints:
        return True, f"{category_descriptor};mapper_score={mapper_score:.4f};domain_hint={domain_hints[0]!r}"

    if generic_raw and mapper_score < mapper_threshold:
        return True, f"{category_descriptor};mapper_score={mapper_score:.4f};generic_raw_category"

    if domain_hints and broad_canonical:
        return True, f"{category_descriptor};domain_hint={domain_hints[0]!r}"

    visual_matches = _top_visual_matches(sorted_vision, limit=1)
    if visual_matches:
        top_label, top_score = visual_matches[0]
        if top_label != canonical and float(top_score) >= _resolve_specificity_search_vision_threshold():
            return True, f"{category_descriptor};vision_hint={top_label!r}@{float(top_score):.4f}"

    return False, f"{category_descriptor};no_specificity_signal"


def _accept_specificity_search_result(
    current_match: dict[str, object],
    refined_match: dict[str, object],
    sorted_vision: dict[str, float],
    candidate_categories: list[str],
) -> tuple[bool, str]:
    current_canonical = str(current_match.get("canonical_category", "") or "")
    refined_canonical = str(refined_match.get("canonical_category", "") or "")
    if not refined_canonical or refined_canonical == current_canonical:
        return False, f"unchanged_category={refined_canonical!r}"
    if candidate_categories and refined_canonical not in candidate_categories:
        return False, f"outside_candidate_set={refined_canonical!r}"

    if refined_canonical not in _specificity_search_broad_categories():
        return True, f"narrowed_to={refined_canonical!r}"

    visual_matches = _top_visual_matches(sorted_vision, limit=1)
    if visual_matches:
        top_label, top_score = visual_matches[0]
        if top_label == refined_canonical and float(top_score) >= _resolve_specificity_search_vision_threshold():
            return True, f"broad_refinement_confirmed_by_vision={refined_canonical!r}@{float(top_score):.4f}"

    return False, f"refined_category_still_broad={refined_canonical!r}"


def _ocr_text_has_commercial_context(text: str) -> tuple[bool, str]:
    normalized = _normalize_ocr(text or "")
    if not normalized:
        return False, "no_text"

    tokens = set(re.findall(r"[a-z0-9']{2,}", normalized))
    numeric_hint = bool(re.search(r"\d", normalized))
    commercial_terms = {
        "promo",
        "promotion",
        "offre",
        "offers",
        "offer",
        "achat",
        "purchase",
        "remise",
        "rabais",
        "discount",
        "deal",
        "sale",
        "soldes",
        "magasin",
        "magasins",
        "store",
        "stores",
        "shop",
        "shops",
        "cart",
        "carte",
        "card",
        "base",
        "reglable",
        "réglable",
        "mattress",
        "bed",
        "beds",
        "bedding",
        "linens",
        "massage",
        "bluetooth",
        "zero",
        "zéro",
        "gravite",
        "gravité",
        "tele",
        "télé",
        "service",
        "services",
        "insurance",
        "assurance",
        "mortgage",
        "loan",
        "credit",
        "banking",
        "banque",
        "expert",
        "experts",
        "online",
        "ligne",
    }
    matched = sorted(term for term in commercial_terms if term in tokens)
    if matched:
        return True, f"commercial_terms={','.join(matched[:4])}"
    if numeric_hint and any(term in tokens for term in {"promo", "offre", "achat", "store", "magasin", "card", "carte"}):
        return True, "numeric_offer_context"
    return False, "no_commercial_context"


def _ocr_context_needs_express_confirmation(
    result_payload: dict[str, object] | None,
    ocr_text: str,
    job_id: str | None,
) -> tuple[bool, str]:
    if not isinstance(result_payload, dict):
        return False, "invalid_result"

    has_context, context_reason = _ocr_text_has_commercial_context(ocr_text)
    if has_context:
        return False, context_reason

    raw_category = str(result_payload.get("category", "") or "").strip()
    if not raw_category:
        return False, "category_missing"

    mapped = category_mapper.map_category(
        raw_category=raw_category,
        job_id=job_id,
        suggested_categories_text="",
        predicted_brand=str(result_payload.get("brand", "") or ""),
        ocr_summary="",
    )
    canonical = str(mapped.get("canonical_category", "") or raw_category)
    score_raw = mapped.get("category_match_score", 0.0)
    try:
        score = float(score_raw or 0.0)
    except (TypeError, ValueError):
        score = 0.0

    generic_terms = {
        "services",
        "service",
        "financial",
        "education",
        "educational",
        "media",
        "programming",
        "organization",
        "institutions",
        "institution",
        "documentary",
        "historical",
        "nonprofit",
        "retail",
        "technology",
        "telecommunications",
        "insurance",
        "banking",
    }
    category_tokens = set(re.findall(r"[a-z0-9']{2,}", _normalize_ocr(canonical)))
    if category_tokens & generic_terms:
        return True, f"generic_category={canonical!r} without commercial OCR context"
    if len(category_tokens) <= 3 and score >= 0.99:
        return True, f"short_exact_category={canonical!r} score={score:.4f} without commercial OCR context"
    return False, f"specific_category={canonical!r} score={score:.4f}"


def _ocr_text_lacks_context(text: str) -> tuple[bool, str]:
    cleaned_lines: list[str] = []
    seen_lines: set[str] = set()
    for raw_line in (text or "").splitlines():
        line = re.sub(r"\[HUGE\]", " ", raw_line, flags=re.IGNORECASE)
        line = re.sub(r"\s+", " ", line).strip(" -•\t")
        if not line:
            continue
        normalized_line = _normalize_ocr(line)
        if normalized_line in seen_lines:
            continue
        cleaned_lines.append(line)
        seen_lines.add(normalized_line)

    tokens = re.findall(r"[a-z0-9.]{2,}", " ".join(cleaned_lines).lower())
    signal_tokens = [token for token in tokens if not token.isdigit()]
    if any("." in token and len(token) >= 6 for token in signal_tokens):
        return False, "domain_present"

    compact_len = len(" ".join(cleaned_lines))
    sparse_threshold = _resolve_ocr_context_sparse_tokens()
    if compact_len <= _resolve_ocr_context_short_chars():
        return True, f"short_text_chars={compact_len}"
    if len(cleaned_lines) <= 1 and len(signal_tokens) <= sparse_threshold:
        return True, f"single_line_sparse_tokens={len(signal_tokens)}"
    if len(set(signal_tokens)) <= sparse_threshold:
        return True, f"sparse_tokens={len(set(signal_tokens))}"
    return False, "rich_ocr"


def _ocr_context_visual_mismatch(
    sorted_vision: dict[str, float],
    result_payload: dict[str, object] | None,
    job_id: str | None,
    ocr_text: str,
) -> tuple[bool, str]:
    if not sorted_vision or not isinstance(result_payload, dict):
        return False, "vision_unavailable"

    ordered_scores = list(sorted_vision.items())
    top_category, top_score = ordered_scores[0]
    second_score = ordered_scores[1][1] if len(ordered_scores) > 1 else 0.0
    if float(top_score) < _resolve_ocr_context_vision_score_threshold():
        return False, f"vision_score_too_low={float(top_score):.4f}"
    if (float(top_score) - float(second_score)) < _resolve_ocr_context_vision_margin_threshold():
        return False, f"vision_margin_too_low={float(top_score) - float(second_score):.4f}"

    category_match = category_mapper.map_category(
        raw_category=str(result_payload.get("category", "") or ""),
        job_id=job_id,
        suggested_categories_text="",
        predicted_brand=str(result_payload.get("brand", "") or ""),
        ocr_summary=ocr_text,
    )
    canonical_category = str(category_match.get("canonical_category", "") or "")
    if not canonical_category:
        return False, "category_map_empty"
    if canonical_category == top_category:
        return False, f"vision_matches={canonical_category}"
    return True, f"vision_mismatch={canonical_category}->{top_category}"


def _ocr_context_mapper_is_weak(
    result_payload: dict[str, object] | None,
    job_id: str | None,
    ocr_text: str,
) -> tuple[bool, str]:
    if not isinstance(result_payload, dict):
        return False, "invalid_result"
    category_match = category_mapper.map_category(
        raw_category=str(result_payload.get("category", "") or ""),
        job_id=job_id,
        suggested_categories_text="",
        predicted_brand=str(result_payload.get("brand", "") or ""),
        ocr_summary=ocr_text,
    )
    score_raw = category_match.get("category_match_score", 0.0)
    try:
        score = float(score_raw or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    threshold = _resolve_ocr_context_mapper_score_threshold()
    if score < threshold:
        return True, f"mapper_score={score:.4f}<threshold={threshold:.2f}"
    return False, f"mapper_score={score:.4f}"


def _ocr_evidence_supports_result(
    result_payload: dict[str, object] | None,
    ocr_text: str,
    job_id: str | None,
) -> tuple[bool, str]:
    if not _ocr_text_has_signal(ocr_text):
        return True, "ocr_support_unavailable"
    if not isinstance(result_payload, dict):
        return False, "invalid_result"

    llm_match = category_mapper.map_category(
        raw_category=str(result_payload.get("category", "") or ""),
        job_id=job_id,
        suggested_categories_text="",
        predicted_brand="",
        ocr_summary="",
    )
    llm_canonical = str(llm_match.get("canonical_category", "") or "")

    ocr_match = category_mapper.map_category(
        raw_category="unknown",
        job_id=job_id,
        suggested_categories_text="",
        predicted_brand="",
        ocr_summary=ocr_text,
    )
    ocr_canonical = str(ocr_match.get("canonical_category", "") or "")
    ocr_score_raw = ocr_match.get("category_match_score", 0.0)
    try:
        ocr_score = float(ocr_score_raw or 0.0)
    except (TypeError, ValueError):
        ocr_score = 0.0

    threshold = _resolve_ocr_support_score_threshold()
    if ocr_score >= threshold and ocr_canonical and llm_canonical and ocr_canonical != llm_canonical:
        return False, f"ocr_support={ocr_canonical!r} score={ocr_score:.4f} llm={llm_canonical!r}"
    return True, f"ocr_support={ocr_canonical!r} score={ocr_score:.4f}"


def _should_run_ocr_context_rescue(
    scan_mode: str,
    express_mode: bool,
    ocr_text: str,
    res: dict[str, object] | None,
    source_frames: list[dict[str, object]],
    sorted_vision: dict[str, float],
    job_id: str | None,
) -> tuple[bool, str]:
    if not _ocr_context_rescue_enabled(scan_mode, express_mode):
        return False, "disabled"
    if len(source_frames) < 2:
        return False, "insufficient_frames"
    if not _ocr_text_has_signal(ocr_text):
        return False, "ocr_blank"
    blank_result, blank_reason = _llm_result_is_blank(res)
    if blank_result:
        return False, f"blank_result={blank_reason}"

    lacks_context, context_reason = _ocr_text_lacks_context(ocr_text)
    if not lacks_context:
        return False, context_reason

    confidence_raw = res.get("confidence", 0.0) if isinstance(res, dict) else 0.0
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < _resolve_ocr_context_confidence_threshold():
        return True, f"{context_reason};low_confidence={confidence:.2f}"

    mapper_weak, mapper_reason = _ocr_context_mapper_is_weak(res, job_id, ocr_text)
    if mapper_weak:
        return True, f"{context_reason};{mapper_reason}"

    if _ocr_context_use_vision_assist():
        visual_mismatch, visual_reason = _ocr_context_visual_mismatch(sorted_vision, res, job_id, ocr_text)
        if visual_mismatch:
            return True, f"{context_reason};{visual_reason}"
        return False, f"{context_reason};{mapper_reason};{visual_reason}"

    return False, f"{context_reason};{mapper_reason};vision_assist_disabled"


def _clean_ocr_context_line(line: str) -> str:
    cleaned = re.sub(r"\[HUGE\]", " ", line or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -•\t")
    return cleaned


def _build_ocr_context_pack(primary_text: str, expanded_text: str) -> str:
    lines: list[str] = []
    seen_lines: set[str] = set()
    max_chars = _resolve_ocr_context_max_chars()
    max_lines = _resolve_ocr_context_max_lines()

    for text in (primary_text or "", expanded_text or ""):
        for raw_line in text.splitlines():
            line = _clean_ocr_context_line(raw_line)
            if not _ocr_text_has_signal(line):
                continue
            normalized_line = _normalize_ocr(line)
            if normalized_line in seen_lines:
                continue
            seen_lines.add(normalized_line)
            lines.append(line)
            if len(lines) >= max_lines:
                break
        if len(lines) >= max_lines:
            break

    if not lines:
        return expanded_text or primary_text

    compact_lines: list[str] = []
    remaining = max_chars
    for line in lines:
        clipped = line[:remaining].rstrip()
        if not clipped:
            break
        compact_lines.append(clipped)
        remaining -= len(clipped) + 1
        if remaining <= 0:
            break
    return "\n".join(compact_lines)


def _limit_rescue_frames(frames: list[dict[str, object]], max_frames: int) -> list[dict[str, object]]:
    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
    unique_indices: list[int] = []
    seen: set[int] = set()
    for idx in indices.tolist():
        if idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)
    return [frames[idx] for idx in unique_indices]


def _llm_result_is_blank(res: dict[str, object] | None) -> tuple[bool, str]:
    if not isinstance(res, dict):
        return True, "non_dict_response"
    brand = str(res.get("brand", "") or "").strip().lower()
    category = str(res.get("category", "") or "").strip().lower()
    confidence = float(res.get("confidence", 0.0) or 0.0)
    blank_brand = brand in {"", "unknown", "none", "n/a", "err"}
    blank_category = category in {"", "unknown", "none", "n/a", "err"}
    if blank_brand and blank_category:
        return True, "blank_brand_and_category"
    if blank_brand:
        return True, "blank_brand"
    if blank_category:
        return True, "blank_category"
    if confidence <= 0.0:
        return True, "zero_confidence"
    return False, "ok"


def _should_run_ocr_edge_rescue(
    scan_mode: str,
    express_mode: bool,
    ocr_text: str,
    res: dict[str, object] | None,
) -> tuple[bool, str]:
    if not _ocr_edge_rescue_enabled(scan_mode, express_mode):
        return False, "disabled"
    has_signal = _ocr_text_has_signal(ocr_text)
    blank_result, blank_reason = _llm_result_is_blank(res)
    if has_signal or not blank_result:
        return False, "initial_result_usable"
    return True, blank_reason


def _ocr_text_is_strong_for_early_stop(text: str) -> bool:
    cleaned = re.sub(r"\[HUGE\]", " ", text or "", flags=re.IGNORECASE)
    tokens = re.findall(r"[a-z0-9.]{2,}", cleaned.lower())
    if not tokens:
        return False

    joined = "".join(tokens)
    min_chars = _resolve_ocr_early_stop_min_chars()
    has_domain_like_token = any("." in token and len(token) >= 6 for token in tokens)
    has_multi_token_signal = len(tokens) >= 2 and len(joined) >= min_chars
    has_single_long_token = any(len(token) >= min_chars for token in tokens)
    return has_domain_like_token or has_multi_token_signal or has_single_long_token


def _ocr_skip_high_confidence_enabled(
    scan_mode: str,
    provider: str,
    backend_model: str,
    enable_search: bool,
    enable_vision_board: bool,
    enable_llm_frame: bool,
    express_mode: bool,
    context_size: int,
) -> bool:
    raw = os.environ.get("OCR_SKIP_HIGH_CONFIDENCE", "true").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if express_mode or enable_search or not enable_vision_board or not enable_llm_frame:
        return False
    mode = (scan_mode or "").strip().lower()
    if mode in {"full video", "full scan"}:
        return False
    try:
        return bool(
            create_provider(provider, backend_model, context_size=int(context_size)).supports_vision
        )
    except Exception as exc:
        logger.debug("ocr_skip_disabled provider_init_failed=%s", exc)
        return False


def _resolve_ocr_skip_confidence_threshold() -> float:
    raw = os.environ.get("OCR_SKIP_CONFIDENCE_THRESHOLD", "0.90")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_skip_confidence_threshold value=%r fallback=0.90", raw)
        return 0.90
    return max(0.0, min(1.0, value))


def _resolve_ocr_skip_vision_score_threshold() -> float:
    raw = os.environ.get("OCR_SKIP_VISION_SCORE_THRESHOLD", "0.80")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid_ocr_skip_vision_score_threshold value=%r fallback=0.80", raw)
        return 0.80
    return max(0.0, min(1.0, value))


def _frame_quality_allows_ocr_skip(frame_bgr: np.ndarray) -> tuple[bool, str]:
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        return False, "invalid_frame"

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    min_brightness = float(os.environ.get("OCR_SKIP_MIN_BRIGHTNESS", "20"))
    min_sharpness = float(os.environ.get("OCR_SKIP_MIN_SHARPNESS", "40"))

    if brightness < min_brightness:
        return False, f"brightness={brightness:.1f}<min={min_brightness:.1f}"
    if sharpness < min_sharpness:
        return False, f"sharpness={sharpness:.1f}<min={min_sharpness:.1f}"
    return True, f"brightness={brightness:.1f} sharpness={sharpness:.1f}"


def _vision_allows_ocr_skip(
    sorted_vision: dict[str, float],
    per_frame_vision: list[dict[str, object]],
) -> tuple[bool, str, str]:
    if not sorted_vision or not per_frame_vision:
        return False, "", "vision_unavailable"

    top_category, top_score = next(iter(sorted_vision.items()))
    threshold = _resolve_ocr_skip_vision_score_threshold()
    if float(top_score) < threshold:
        return False, top_category, f"top_score={float(top_score):.4f}<threshold={threshold:.2f}"

    recent_frames = per_frame_vision[-2:] if len(per_frame_vision) >= 2 else per_frame_vision[-1:]
    for frame_info in recent_frames:
        frame_category = str(frame_info.get("top_category", ""))
        frame_score = float(frame_info.get("top_score", 0.0))
        if frame_category != top_category:
            return False, top_category, "recent_frame_category_mismatch"
        if frame_score < threshold:
            return False, top_category, f"recent_frame_score={frame_score:.4f}<threshold={threshold:.2f}"

    return True, top_category, "ready"


def _llm_result_allows_ocr_skip(
    llm_result: dict[str, object],
    expected_category: str,
    job_id: str | None,
) -> tuple[bool, str]:
    if not isinstance(llm_result, dict):
        return False, "invalid_llm_result"

    brand = str(llm_result.get("brand", "") or "").strip()
    category = str(llm_result.get("category", "") or "").strip()
    if brand.lower() in {"", "unknown", "none", "n/a"}:
        return False, "brand_missing"
    if not category:
        return False, "category_missing"

    confidence_raw = llm_result.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0

    threshold = _resolve_ocr_skip_confidence_threshold()
    if confidence < threshold:
        return False, f"confidence={confidence:.2f}<threshold={threshold:.2f}"

    category_match = category_mapper.map_category(
        raw_category=category,
        job_id=job_id,
        suggested_categories_text="",
        predicted_brand=brand,
        ocr_summary="",
    )
    canonical_category = str(category_match.get("canonical_category", "") or "")
    if canonical_category != expected_category:
        return False, f"vision_category={expected_category!r} llm_category={canonical_category!r}"

    return True, f"confidence={confidence:.2f} canonical_category={canonical_category!r}"


def _resolve_llm_recent_frame_count() -> int:
    raw = os.environ.get("LLM_RECENT_FRAME_COUNT", "4")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        logger.warning("invalid_llm_recent_frame_count value=%r fallback=4", raw)
        return 4


def _frame_visual_richness_metrics(frame: dict[str, object]) -> dict[str, float | bool]:
    frame_bgr = frame.get("ocr_image")
    if frame_bgr is None or not hasattr(frame_bgr, "shape"):
        return {
            "dark_ratio": 0.0,
            "edge_density": 0.0,
            "saturation_mean": 0.0,
            "richness": 0.0,
            "logo_like": False,
        }

    try:
        resized = cv2.resize(frame_bgr, (96, 96), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    except Exception:
        return {
            "dark_ratio": 0.0,
            "edge_density": 0.0,
            "saturation_mean": 0.0,
            "richness": 0.0,
            "logo_like": False,
        }

    dark_ratio = float(np.mean(gray < 28))
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(np.mean(edges > 0))
    saturation_mean = float(np.mean(hsv[..., 1]) / 255.0)
    richness = (edge_density * 3.0) + (saturation_mean * 0.7) + ((1.0 - dark_ratio) * 0.25)
    logo_like = dark_ratio >= 0.7 and edge_density <= 0.035 and saturation_mean <= 0.18
    return {
        "dark_ratio": dark_ratio,
        "edge_density": edge_density,
        "saturation_mean": saturation_mean,
        "richness": richness,
        "logo_like": logo_like,
    }


def _select_llm_evidence_frames(source_frames: list[dict[str, object]], frame_limit: int) -> list[dict[str, object]]:
    if frame_limit <= 0 or not source_frames:
        return []

    tail_like_types = {"tail", "backward_ext", "tail_rescue"}
    if any(str(frame.get("type", "") or "").lower() in tail_like_types for frame in source_frames):
        candidate_window = min(len(source_frames), max(frame_limit + 2, frame_limit))
        candidate_frames = list(source_frames[-candidate_window:])
    else:
        candidate_frames = list(source_frames[-1:])
    if len(candidate_frames) <= 2:
        return candidate_frames

    metrics_by_index = {
        idx: _frame_visual_richness_metrics(frame)
        for idx, frame in enumerate(candidate_frames)
    }

    selected_indices: set[int] = {len(candidate_frames) - 1}
    latest_index = len(candidate_frames) - 1

    non_logo_candidates = [
        idx for idx, metrics in metrics_by_index.items()
        if idx != latest_index and not bool(metrics.get("logo_like", False))
    ]
    if non_logo_candidates:
        richest_product_idx = max(
            non_logo_candidates,
            key=lambda idx: (
                float(metrics_by_index[idx].get("richness", 0.0)),
                -abs(idx - latest_index),
            ),
        )
        selected_indices.add(richest_product_idx)

    max_logo_like_frames = 1
    logo_like_selected = sum(
        1 for idx in selected_indices if bool(metrics_by_index[idx].get("logo_like", False))
    )

    ranked_remaining = sorted(
        (idx for idx in range(len(candidate_frames)) if idx not in selected_indices),
        key=lambda idx: (
            bool(metrics_by_index[idx].get("logo_like", False)),
            -float(metrics_by_index[idx].get("richness", 0.0)),
            -idx,
        ),
    )

    for idx in ranked_remaining:
        is_logo_like = bool(metrics_by_index[idx].get("logo_like", False))
        if is_logo_like and logo_like_selected >= max_logo_like_frames:
            continue
        selected_indices.add(idx)
        if is_logo_like:
            logo_like_selected += 1
        if len(selected_indices) >= frame_limit:
            break

    return [candidate_frames[idx] for idx in sorted(selected_indices)]


def process_single_video(
    url,
    categories,
    p,
    m,
    oe,
    om,
    override,
    sm,
    enable_search,
    enable_vision_board=None,
    enable_llm_frame=None,
    product_focus_guidance_enabled=True,
    ctx=8192,
    category_embedding_model=None,
    express_mode=False,
    job_id=None,
    stage_callback=None,
    enable_vision=None,  # Deprecated alias
):
    processing_trace: dict[str, object] = {
        "mode": "pipeline",
        "attempts": [],
        "summary": {},
    }
    pipeline_started_at = time.perf_counter()
    if enable_vision is not None:
        if enable_vision_board is None:
            enable_vision_board = bool(enable_vision)
        if enable_llm_frame is None:
            enable_llm_frame = bool(enable_vision)
    if enable_vision_board is None:
        enable_vision_board = True
    if enable_llm_frame is None:
        enable_llm_frame = True

    try:
        if stage_callback:
            stage_callback("ingest", "validating and preparing input")
        logger.info(f"[{url}] === STARTING PIPELINE WORKER ===")
        express_mode = bool(express_mode)
        send_llm_frame = bool(enable_llm_frame or express_mode)
        selected_category_embedding_model = str(
            category_embedding_model or "BAAI/bge-large-en-v1.5"
        ).strip() or "BAAI/bge-large-en-v1.5"
        if hasattr(category_mapper, "configure_embedding_model"):
            selected_category_embedding_model = category_mapper.configure_embedding_model(
                category_embedding_model
            )
        processing_trace.update(
            {
                "provider": p,
                "model": m,
                "category_embedding_model": selected_category_embedding_model,
                "ocr_engine": oe,
                "ocr_mode": om,
                "scan_mode": "Express" if express_mode else sm,
            }
        )
        visual_debug: dict[str, object] | None = None
        initial_frames: list[dict[str, object]] = []

        def _frame_times_snapshot(source_frames: list[dict[str, object]]) -> list[float]:
            times: list[float] = []
            for frame in source_frames or []:
                try:
                    times.append(round(float(frame.get("time", 0.0)), 1))
                except Exception:
                    continue
            return times

        def _ocr_excerpt(text: str, limit: int = 220) -> str:
            compact = " ".join((text or "").split())
            if len(compact) <= limit:
                return compact
            return f"{compact[:limit].rstrip()}..."

        def _llm_evidence_images(source_frames: list[dict[str, object]]) -> list[Image.Image]:
            frame_limit = _resolve_llm_recent_frame_count()
            candidate_frames = _select_llm_evidence_frames(source_frames, frame_limit)
            images: list[Image.Image] = []
            for frame in candidate_frames:
                try:
                    images.append(get_pil_image(frame))
                except Exception:
                    continue
            return images

        def _result_snapshot(payload: dict[str, object] | None) -> dict[str, object]:
            if not isinstance(payload, dict):
                return {"brand": "", "category": "", "confidence": 0.0}
            try:
                confidence = float(payload.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            return {
                "brand": str(payload.get("brand", "") or ""),
                "category": str(payload.get("category", "") or ""),
                "confidence": confidence,
                "brand_ambiguity_flag": bool(payload.get("brand_ambiguity_flag", False)),
                "brand_ambiguity_reason": str(payload.get("brand_ambiguity_reason", "") or ""),
                "brand_ambiguity_resolved": bool(payload.get("brand_ambiguity_resolved", False)),
                "brand_disambiguation_reason": str(payload.get("brand_disambiguation_reason", "") or ""),
                "brand_evidence_strength": str(payload.get("brand_evidence_strength", "") or ""),
            }

        def _append_trace_attempt(
            *,
            attempt_type: str,
            title: str,
            status: str,
            source_frames: list[dict[str, object]] | None,
            detail: str = "",
            trigger_reason: str = "",
            ocr_text_value: str = "",
            result_payload: dict[str, object] | None = None,
            ocr_mode_used: str = "",
            llm_mode: str = "standard",
            evidence_note: str = "",
            elapsed_ms: float | None = None,
        ) -> None:
            attempts = processing_trace.setdefault("attempts", [])
            if not isinstance(attempts, list):
                return
            attempts.append(
                {
                    "attempt_type": attempt_type,
                    "title": title,
                    "status": status,
                    "detail": detail,
                    "trigger_reason": trigger_reason,
                    "frame_count": len(source_frames or []),
                    "frame_times": _frame_times_snapshot(source_frames or []),
                    "ocr_excerpt": _ocr_excerpt(ocr_text_value),
                    "ocr_signal": _ocr_text_has_signal(ocr_text_value),
                    "ocr_mode": ocr_mode_used,
                    "llm_mode": llm_mode,
                    "evidence_note": evidence_note,
                    "elapsed_ms": round(float(elapsed_ms), 1) if elapsed_ms is not None else None,
                    "result": _result_snapshot(result_payload),
                }
            )

        def _finalize_processing_trace() -> None:
            attempts = processing_trace.get("attempts")
            if not isinstance(attempts, list):
                attempts = []
            accepted = next(
                (
                    attempt
                    for attempt in reversed(attempts)
                    if isinstance(attempt, dict) and attempt.get("status") == "accepted"
                ),
                None,
            )
            trigger_reason = next(
                (
                    str(attempt.get("trigger_reason") or "")
                    for attempt in attempts
                    if isinstance(attempt, dict) and attempt.get("trigger_reason")
                ),
                "",
            )
            if isinstance(accepted, dict):
                accepted_title = str(accepted.get("title") or "Processing Path")
                if accepted.get("attempt_type") == "initial":
                    headline = f"Completed on {accepted_title.lower()}."
                else:
                    headline = f"Recovered via {accepted_title.lower()}."
            else:
                headline = "No successful processing path was recorded."
            processing_trace["summary"] = {
                "headline": headline,
                "attempt_count": len(attempts),
                "retry_count": max(0, len(attempts) - 1),
                "accepted_attempt_type": accepted.get("attempt_type") if isinstance(accepted, dict) else "",
                "trigger_reason": trigger_reason,
            }
        if express_mode:
            if stage_callback:
                stage_callback("frame_extract", "express mode enabled; extracting static tail frame")
            logger.info("[%s] express mode enabled: bypassing OCR extraction", url)
            express_frame = extract_express_brand_frame(url, job_id=job_id)
            frame_type = "express_tail"
            if express_frame is None:
                logger.warning("[%s] express frame extraction failed; falling back to middle frame", url)
                express_frame = extract_middle_frame(url, job_id=job_id)
                frame_type = "middle_fallback"
            if express_frame is None:
                logger.warning("[%s] express mode failed to extract any frame", url)
                return {}, [], "Err", "No frames", [], [url, "Err", "", "Err", 0, "Empty", "none", None]
            express_bgr = cv2.cvtColor(np.array(express_frame), cv2.COLOR_RGB2BGR)
            frames = [
                {
                    "image": express_frame,
                    "_pil_cache": express_frame,
                    "ocr_image": express_bgr,
                    "time": 0.0,
                    "type": frame_type,
                }
            ]
        else:
            frames, cap = extract_frames_for_pipeline(url, scan_mode=sm, job_id=job_id)
            if cap and cap.isOpened():
                cap.release()
        initial_frames = list(frames)
        
        if not frames: 
            logger.warning(f"[{url}] Extraction yielded no frames.")
            return {}, [], "Err", "No frames", [], [url, "Err", "", "Err", 0, "Empty", "none", None]

        if stage_callback and not express_mode:
            extracted_mode = "full video" if frames[0].get("type") == "scene" else "tail only"
            stage_callback("frame_extract", f"extracted {len(frames)} frames ({extracted_mode})")
        
        def _do_vision() -> tuple[dict[str, float], list[dict[str, object]]]:
            nonlocal visual_debug
            logger.debug("[%s] parallel_task_start: vision", url)
            try:
                ready, reason = category_mapper.ensure_vision_text_features()
                siglip_model = categories_runtime.siglip_model
                siglip_processor = categories_runtime.siglip_processor
                if not ready or siglip_model is None or siglip_processor is None:
                    if stage_callback:
                        stage_callback("vision", f"vision skipped; {reason}")
                    logger.info("[%s] vision skipped: %s", url, reason)
                    return {}, []

                if stage_callback:
                    stage_callback("vision", "vision enabled; computing visual category scores")
                start_time = time.time()
                with torch.no_grad():
                    pil_images = [get_pil_image(f) for f in frames]
                    image_inputs = siglip_processor(images=pil_images, return_tensors="pt").to(device)
                    if TORCH_DTYPE != torch.float32:
                        image_inputs = {
                            k: v.to(dtype=TORCH_DTYPE) if torch.is_floating_point(v) else v
                            for k, v in image_inputs.items()
                        }
                    image_features = siglip_model.get_image_features(**image_inputs)
                    image_features = normalize_feature_tensor(
                        image_features,
                        source="SigLIP.get_image_features",
                    )

                    logit_scale = siglip_model.logit_scale.exp()
                    logit_bias = siglip_model.logit_bias
                    logits_per_image = (image_features @ category_mapper.vision_text_features.t()) * logit_scale + logit_bias
                    probs = torch.sigmoid(logits_per_image)

                per_frame_vision_local: list[dict[str, object]] = []
                for frame_idx in range(probs.shape[0]):
                    frame_probs = probs[frame_idx]
                    top_score, top_idx = torch.max(frame_probs, dim=0)
                    per_frame_vision_local.append(
                        {
                            "frame_index": int(frame_idx),
                            "top_category": category_mapper.categories[int(top_idx.item())],
                            "top_score": round(float(top_score.item()), 4),
                        }
                    )

                scores = probs.mean(dim=0).cpu().numpy()
                visual_debug = {
                    "image_feature": image_features.mean(dim=0).detach().cpu(),
                    "score_vector": probs.mean(dim=0).detach().cpu(),
                    "backend": getattr(categories_runtime, "SIGLIP_ID", "SigLIP"),
                    "query_label": (
                        f"Frame @ {frames[0]['time']:.1f}s"
                        if len(frames) == 1
                        else f"Mean of {len(frames)} sampled frames"
                    ),
                }
                sorted_vision_local = dict(
                    sorted(
                        {
                            category_mapper.categories[i]: float(scores[i])
                            for i in range(len(category_mapper.categories))
                        }.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:5]
                )
                logger.debug("[%s] vision_task_done in %.2fs", url, time.time() - start_time)
                return sorted_vision_local, per_frame_vision_local
            except Exception as exc:
                logger.error("[%s] vision task failed: %s", url, exc, exc_info=True)
                raise

        def _do_ocr(
            prepared_frames: list[dict[str, object]] | None = None,
            visually_skipped_count: int | None = None,
            rescue_profile: bool = False,
            ocr_mode_override: str | None = None,
        ) -> str:
            logger.debug("[%s] parallel_task_start: ocr", url)
            try:
                if stage_callback:
                    detail = f"ocr engine={oe.lower()}"
                    if rescue_profile:
                        detail += " rescue profile"
                    stage_callback("ocr", detail)
                effective_ocr_mode = ocr_mode_override or om
                dedup_threshold = _resolve_ocr_dedup_threshold()
                if prepared_frames is None or visually_skipped_count is None:
                    ocr_frames, visually_skipped_count = _select_frames_for_ocr(frames)
                else:
                    ocr_frames = prepared_frames
                    visually_skipped_count = int(visually_skipped_count)
                if rescue_profile:
                    logger.info(
                        "[%s] ocr_rescue_profile: frames=%d mode=%s",
                        url,
                        len(ocr_frames),
                        effective_ocr_mode,
                    )
                ocr_lines: list[str] = []
                prev_normalized: str | None = None
                skipped_count = 0
                roi_hits = 0
                roi_fallbacks = 0
                early_stop_skipped = 0
                no_roi_skipped = 0
                ocr_call_count = 0
                ocr_elapsed_seconds = 0.0
                early_stop_active = _ocr_early_stop_enabled(sm) and not rescue_profile
                last_index = len(ocr_frames) - 1
                idx = 0

                def _run_ocr(target_image: np.ndarray) -> str:
                    nonlocal ocr_call_count, ocr_elapsed_seconds
                    started_at = time.perf_counter()
                    try:
                        return ocr_manager.extract_text(oe, target_image, effective_ocr_mode)
                    finally:
                        ocr_call_count += 1
                        ocr_elapsed_seconds += time.perf_counter() - started_at

                while idx < len(ocr_frames):
                    frame = ocr_frames[idx]
                    ocr_image = frame["ocr_image"]
                    raw_text = ""
                    is_last_frame = idx == last_index
                    roi_image = ocr_image
                    roi_used = False
                    roi_detection_applicable = (
                        isinstance(ocr_image, np.ndarray)
                        and ocr_image.shape[0] >= 120
                        and ocr_image.shape[1] >= 120
                    )
                    roi_capable = roi_detection_applicable and (
                        _ocr_roi_enabled(oe) or (_ocr_skip_no_roi_enabled(oe, sm) and not rescue_profile)
                    )
                    if roi_capable:
                        roi_image, roi_used = _extract_ocr_focus_region(ocr_image)
                    if (
                        _ocr_skip_no_roi_enabled(oe, sm)
                        and not rescue_profile
                        and not is_last_frame
                        and roi_capable
                        and not roi_used
                    ):
                        no_roi_skipped += 1
                        logger.debug(
                            "ocr_no_roi_skip: frame at %.1fs no plausible text region detected",
                            frame["time"],
                        )
                        idx += 1
                        continue
                    if _ocr_roi_enabled(oe) and isinstance(ocr_image, np.ndarray):
                        if roi_used:
                            roi_hits += 1
                            logger.debug(
                                "ocr_roi_first: frame at %.1fs roi_shape=%s full_shape=%s",
                                frame["time"],
                                tuple(roi_image.shape[:2]),
                                tuple(ocr_image.shape[:2]),
                            )
                            raw_text = _run_ocr(roi_image)
                            if not _ocr_text_has_signal(raw_text):
                                roi_fallbacks += 1
                                logger.debug(
                                    "ocr_roi_fallback: frame at %.1fs weak roi text, retrying full frame",
                                    frame["time"],
                                )
                                raw_text = _run_ocr(ocr_image)
                        else:
                            raw_text = _run_ocr(ocr_image)
                    else:
                        raw_text = _run_ocr(ocr_image)
                    normalized = _normalize_ocr(raw_text)
                    if (
                        prev_normalized is not None
                        and not is_last_frame
                        and _ocr_texts_similar(normalized, prev_normalized, dedup_threshold)
                    ):
                        skipped_count += 1
                        logger.debug(
                            "ocr_dedup_skip: frame at %.1fs similar to previous threshold=%.2f",
                            frame["time"],
                            dedup_threshold,
                        )
                        idx += 1
                        continue
                    # Do not inject frame timestamps into OCR text; they pollute LLM/search input.
                    ocr_lines.append(raw_text)
                    prev_normalized = normalized
                    if (
                        early_stop_active
                        and not is_last_frame
                        and idx < last_index - 1
                        and _ocr_text_is_strong_for_early_stop(raw_text)
                    ):
                        early_stop_skipped += last_index - idx - 1
                        logger.debug(
                            "ocr_early_stop: frame at %.1fs strong_signal=True skipped_intermediate=%d",
                            frame["time"],
                            early_stop_skipped,
                        )
                        idx = last_index
                        continue
                    idx += 1
                logger.info(
                    "ocr_dedup: processed=%d text_skipped=%d visual_skipped=%d no_roi_skipped=%d early_stop_skipped=%d roi_hits=%d roi_fallbacks=%d ocr_calls=%d ocr_elapsed_ms=%.1f avg_ocr_ms=%.1f ocr_frames=%d total_frames=%d threshold=%.2f",
                    len(ocr_lines),
                    skipped_count,
                    visually_skipped_count,
                    no_roi_skipped,
                    early_stop_skipped,
                    roi_hits,
                    roi_fallbacks,
                    ocr_call_count,
                    ocr_elapsed_seconds * 1000.0,
                    (ocr_elapsed_seconds * 1000.0 / ocr_call_count) if ocr_call_count else 0.0,
                    len(ocr_frames),
                    len(frames),
                    dedup_threshold,
                )
                logger.debug("[%s] parallel_task_done: ocr", url)
                return "\n".join(ocr_lines)
            except Exception as exc:
                logger.error("[%s] ocr task failed: %s", url, exc, exc_info=True)
                raise

        sorted_vision: dict[str, float] = {}
        per_frame_vision: list[dict[str, object]] = []
        res: dict[str, object] | None = None
        if express_mode:
            if stage_callback:
                stage_callback("ocr", "express mode enabled; OCR bypassed")
            ocr_text = ""
            if enable_vision_board:
                sorted_vision, per_frame_vision = _do_vision()
        else:
            skip_gate_active = _ocr_skip_high_confidence_enabled(
                scan_mode=sm,
                provider=p,
                backend_model=m,
                enable_search=bool(enable_search),
                enable_vision_board=bool(enable_vision_board),
                enable_llm_frame=send_llm_frame,
                express_mode=express_mode,
                context_size=ctx,
            )
            skip_candidate_frames: list[dict[str, object]] | None = None
            skip_candidate_visual_skipped = 0
            if skip_gate_active:
                skip_candidate_frames, skip_candidate_visual_skipped = _select_frames_for_ocr(frames)
                if len(skip_candidate_frames) != 1:
                    logger.debug(
                        "[%s] ocr_skip_disabled reason=selected_frames_%d",
                        url,
                        len(skip_candidate_frames),
                    )
                    skip_gate_active = False

            if skip_gate_active and enable_vision_board:
                sorted_vision, per_frame_vision = _do_vision()
                frame_ok, frame_reason = _frame_quality_allows_ocr_skip(
                    skip_candidate_frames[-1]["ocr_image"] if skip_candidate_frames else frames[-1]["ocr_image"]
                )
                vision_ok, expected_category, vision_reason = _vision_allows_ocr_skip(
                    sorted_vision,
                    per_frame_vision,
                )
                if frame_ok and vision_ok:
                    if stage_callback:
                        stage_callback("llm", f"evaluating provider={p.lower()} model={m} for OCR skip")
                    evidence_images = _llm_evidence_images(skip_candidate_frames or frames) if send_llm_frame else []
                    tail_image = evidence_images[-1] if evidence_images else None
                    prelim_res = llm_engine.query_pipeline(
                        p,
                        m,
                        "",
                        tail_image,
                        override,
                        False,
                        send_llm_frame,
                        ctx,
                        express_mode=False,
                        evidence_images=evidence_images,
                        product_focus_guidance_enabled=product_focus_guidance_enabled,
                    )
                    llm_ok, llm_reason = _llm_result_allows_ocr_skip(
                        prelim_res,
                        expected_category,
                        job_id,
                    )
                    if llm_ok:
                        if stage_callback:
                            stage_callback("ocr", "skipped; high-confidence multimodal result")
                        logger.info(
                            "[%s] ocr_skip_accepted expected_category=%s %s",
                            url,
                            expected_category,
                            llm_reason,
                        )
                        ocr_text = ""
                        res = prelim_res
                    else:
                        logger.info(
                            "[%s] ocr_skip_rejected llm=%s frame=%s vision=%s",
                            url,
                            llm_reason,
                            frame_reason,
                            vision_reason,
                        )
                else:
                    logger.info(
                        "[%s] ocr_skip_rejected frame=%s vision=%s",
                        url,
                        frame_reason,
                        vision_reason,
                    )

            if res is None and enable_vision_board and not skip_gate_active:
                parallel_t0 = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    vision_future = pool.submit(bind_current_log_context(_do_vision))
                    ocr_future = pool.submit(bind_current_log_context(_do_ocr))
                    ocr_text = ocr_future.result()
                    sorted_vision, per_frame_vision = vision_future.result()
                logger.info("parallel_ocr_vision: completed in %.2fs", time.time() - parallel_t0)
            elif res is None and enable_vision_board:
                ocr_text = _do_ocr(skip_candidate_frames, skip_candidate_visual_skipped)
            elif res is None:
                ocr_text = _do_ocr()
        if stage_callback:
            stage_callback("llm", f"calling provider={p.lower()} model={m}")
        if res is None:
            evidence_images = _llm_evidence_images(frames) if send_llm_frame else []
            tail_image = evidence_images[-1] if evidence_images else None
            res = llm_engine.query_pipeline(
                p,
                m,
                ocr_text,
                tail_image,
                override,
                enable_search,
                send_llm_frame,
                ctx,
                express_mode=express_mode,
                evidence_images=evidence_images,
                product_focus_guidance_enabled=product_focus_guidance_enabled,
            )

        def _apply_fallback_result(
            fallback_type: str,
            fallback_frames: list[dict[str, object]],
            fallback_ocr_text: str,
            fallback_res: dict[str, object],
            detail: str = "",
            reason: str = "",
            trace_ocr_text: str | None = None,
            ocr_mode_used: str = "",
        ) -> None:
            nonlocal frames, ocr_text, res, sorted_vision, per_frame_vision
            frames = fallback_frames
            ocr_text = fallback_ocr_text
            res = fallback_res
            if enable_vision_board:
                sorted_vision, per_frame_vision = _do_vision()
            logger.info("[%s] fallback_accepted type=%s", url, fallback_type)
            _append_trace_attempt(
                attempt_type=fallback_type,
                title=fallback_type.replace("_", " ").title(),
                status="accepted",
                source_frames=fallback_frames,
                detail=detail,
                trigger_reason=reason,
                ocr_text_value=trace_ocr_text if trace_ocr_text is not None else fallback_ocr_text,
                result_payload=fallback_res,
                ocr_mode_used=ocr_mode_used or ("image-only" if fallback_type == "express_rescue" else ""),
                llm_mode="express" if fallback_type == "express_rescue" else "standard",
            )

        def _run_text_fallback(
            fallback_type: str,
            fallback_frames: list[dict[str, object]],
            detail: str,
            reason: str,
            ocr_mode_override: str | None = None,
            llm_text_builder: Callable[[str], str] | None = None,
        ) -> tuple[bool, str]:
            attempt_started_at = time.perf_counter()
            if not fallback_frames:
                logger.info(
                    "[%s] fallback_rejected type=%s reason=no_frames trigger=%s",
                    url,
                    fallback_type,
                    reason,
                )
                _append_trace_attempt(
                    attempt_type=fallback_type,
                    title=fallback_type.replace("_", " ").title(),
                    status="rejected",
                    source_frames=[],
                    detail=detail,
                    trigger_reason=reason,
                    ocr_mode_used=ocr_mode_override or "",
                    evidence_note="No candidate frames were available.",
                    elapsed_ms=(time.perf_counter() - attempt_started_at) * 1000.0,
                )
                return False, ""
            logger.info(
                "[%s] fallback_triggered type=%s reason=%s frames=%d detail=%s",
                url,
                fallback_type,
                reason,
                len(fallback_frames),
                detail,
            )
            if stage_callback:
                stage_callback("frame_extract", f"{fallback_type} extracted {len(fallback_frames)} frames ({detail})")
            fallback_ocr_text = _do_ocr(
                prepared_frames=fallback_frames,
                visually_skipped_count=0,
                rescue_profile=True,
                ocr_mode_override=ocr_mode_override,
            )
            if not _ocr_text_has_signal(fallback_ocr_text):
                logger.info(
                    "[%s] fallback_rejected type=%s reason=weak_ocr trigger=%s",
                    url,
                    fallback_type,
                    reason,
                )
                _append_trace_attempt(
                    attempt_type=fallback_type,
                    title=fallback_type.replace("_", " ").title(),
                    status="rejected",
                    source_frames=fallback_frames,
                    detail=detail,
                    trigger_reason=reason,
                    ocr_text_value=fallback_ocr_text,
                    ocr_mode_used=ocr_mode_override or "",
                    evidence_note="OCR remained too weak to retry the classifier.",
                    elapsed_ms=(time.perf_counter() - attempt_started_at) * 1000.0,
                )
                return False, fallback_ocr_text
            fallback_llm_text = llm_text_builder(fallback_ocr_text) if llm_text_builder else fallback_ocr_text
            fallback_images = _llm_evidence_images(fallback_frames) if send_llm_frame else []
            fallback_tail_image = fallback_images[-1] if fallback_images else None
            if stage_callback:
                stage_callback("llm", f"retrying {fallback_type} provider={p.lower()} model={m}")
            fallback_res = llm_engine.query_pipeline(
                p,
                m,
                fallback_llm_text,
                fallback_tail_image,
                override,
                enable_search,
                send_llm_frame,
                ctx,
                express_mode=False,
                evidence_images=fallback_images,
                product_focus_guidance_enabled=product_focus_guidance_enabled,
            )
            fallback_blank, fallback_blank_reason = _llm_result_is_blank(fallback_res)
            if fallback_blank:
                logger.info(
                    "[%s] fallback_rejected type=%s reason=%s trigger=%s",
                    url,
                    fallback_type,
                    fallback_blank_reason,
                    reason,
                )
                _append_trace_attempt(
                    attempt_type=fallback_type,
                    title=fallback_type.replace("_", " ").title(),
                    status="rejected",
                    source_frames=fallback_frames,
                    detail=detail,
                    trigger_reason=reason,
                    ocr_text_value=fallback_llm_text,
                    result_payload=fallback_res,
                    ocr_mode_used=ocr_mode_override or "",
                    evidence_note=f"Classifier response was rejected: {fallback_blank_reason}.",
                    elapsed_ms=(time.perf_counter() - attempt_started_at) * 1000.0,
                )
                return False, fallback_ocr_text
            ocr_supports_result, ocr_support_reason = _ocr_evidence_supports_result(
                fallback_res,
                fallback_llm_text,
                job_id,
            )
            if not ocr_supports_result:
                logger.info(
                    "[%s] fallback_rejected type=%s reason=%s trigger=%s",
                    url,
                    fallback_type,
                    ocr_support_reason,
                    reason,
                )
                _append_trace_attempt(
                    attempt_type=fallback_type,
                    title=fallback_type.replace("_", " ").title(),
                    status="rejected",
                    source_frames=fallback_frames,
                    detail=detail,
                    trigger_reason=reason,
                    ocr_text_value=fallback_llm_text,
                    result_payload=fallback_res,
                    ocr_mode_used=ocr_mode_override or "",
                    evidence_note=f"Classifier response was rejected because OCR evidence supported a different category: {ocr_support_reason}.",
                    elapsed_ms=(time.perf_counter() - attempt_started_at) * 1000.0,
                )
                return False, fallback_ocr_text
            challenge_express, challenge_reason = (False, "")
            if fallback_type == "ocr_context_rescue":
                challenge_express, challenge_reason = _ocr_context_needs_express_confirmation(
                    fallback_res,
                    fallback_llm_text,
                    job_id,
                )
            if challenge_express:
                logger.info(
                    "[%s] fallback_provisional type=%s reason=%s trigger=%s",
                    url,
                    fallback_type,
                    challenge_reason,
                    reason,
                )
                express_accepted = _run_express_fallback(
                    fallback_type="express_rescue",
                    reason=f"{reason}; {challenge_reason}",
                    fallback_ocr_text=fallback_ocr_text,
                )
                if express_accepted:
                    attempts = processing_trace.get("attempts")
                    rejected_attempt = {
                        "attempt_type": fallback_type,
                        "title": fallback_type.replace("_", " ").title(),
                        "status": "rejected",
                        "detail": detail,
                        "trigger_reason": reason,
                        "frame_count": len(fallback_frames or []),
                        "frame_times": _frame_times_snapshot(fallback_frames or []),
                        "ocr_excerpt": _ocr_excerpt(fallback_llm_text),
                        "ocr_signal": _ocr_text_has_signal(fallback_llm_text),
                        "ocr_mode": ocr_mode_override or "",
                        "llm_mode": "standard",
                        "evidence_note": f"Text rescue was superseded by express rescue because {challenge_reason}.",
                        "elapsed_ms": round((time.perf_counter() - attempt_started_at) * 1000.0, 1),
                        "result": _result_snapshot(fallback_res),
                    }
                    if isinstance(attempts, list) and attempts:
                        attempts.insert(len(attempts) - 1, rejected_attempt)
                    else:
                        _append_trace_attempt(
                            attempt_type=fallback_type,
                            title=fallback_type.replace("_", " ").title(),
                            status="rejected",
                            source_frames=fallback_frames,
                            detail=detail,
                            trigger_reason=reason,
                            ocr_text_value=fallback_llm_text,
                            result_payload=fallback_res,
                            ocr_mode_used=ocr_mode_override or "",
                            evidence_note=f"Text rescue was superseded by express rescue because {challenge_reason}.",
                            elapsed_ms=(time.perf_counter() - attempt_started_at) * 1000.0,
                        )
                    return True, fallback_ocr_text
            _apply_fallback_result(
                fallback_type,
                fallback_frames,
                fallback_ocr_text,
                fallback_res,
                detail=detail,
                reason=reason,
                trace_ocr_text=fallback_llm_text if llm_text_builder else None,
                ocr_mode_used=ocr_mode_override or "",
            )
            attempts = processing_trace.get("attempts")
            if isinstance(attempts, list) and attempts:
                attempts[-1]["elapsed_ms"] = round((time.perf_counter() - attempt_started_at) * 1000.0, 1)
            return True, fallback_ocr_text

        def _run_express_fallback(
            fallback_type: str,
            reason: str,
            fallback_ocr_text: str = "",
        ) -> bool:
            attempt_started_at = time.perf_counter()
            if not _express_rescue_enabled():
                logger.info(
                    "[%s] fallback_rejected type=%s reason=disabled trigger=%s",
                    url,
                    fallback_type,
                    reason,
                )
                _append_trace_attempt(
                    attempt_type=fallback_type,
                    title=fallback_type.replace("_", " ").title(),
                    status="rejected",
                    source_frames=[],
                    trigger_reason=reason,
                    llm_mode="express",
                    evidence_note="Express rescue is disabled.",
                    elapsed_ms=(time.perf_counter() - attempt_started_at) * 1000.0,
                )
                return False
            logger.info("[%s] fallback_triggered type=%s reason=%s", url, fallback_type, reason)
            if stage_callback:
                stage_callback("frame_extract", f"{fallback_type} extracting static frame")
            express_frame = extract_express_brand_frame(url, job_id=job_id)
            frame_type = "express_tail"
            if express_frame is None:
                logger.warning("[%s] %s failed to extract express frame; falling back to middle frame", url, fallback_type)
                express_frame = extract_middle_frame(url, job_id=job_id)
                frame_type = "middle_fallback"
            if express_frame is None:
                logger.info(
                    "[%s] fallback_rejected type=%s reason=no_frames trigger=%s",
                    url,
                    fallback_type,
                    reason,
                )
                _append_trace_attempt(
                    attempt_type=fallback_type,
                    title=fallback_type.replace("_", " ").title(),
                    status="rejected",
                    source_frames=[],
                    trigger_reason=reason,
                    llm_mode="express",
                    evidence_note="Express rescue could not extract a representative frame.",
                    elapsed_ms=(time.perf_counter() - attempt_started_at) * 1000.0,
                )
                return False
            express_bgr = cv2.cvtColor(np.array(express_frame), cv2.COLOR_RGB2BGR)
            express_frames = [
                {
                    "image": express_frame,
                    "_pil_cache": express_frame,
                    "ocr_image": express_bgr,
                    "time": 0.0,
                    "type": frame_type,
                }
            ]
            if stage_callback:
                stage_callback("llm", f"retrying {fallback_type} provider={p.lower()} model={m}")
            express_res = llm_engine.query_pipeline(
                p,
                m,
                "",
                express_frame if send_llm_frame else None,
                override,
                False,
                send_llm_frame,
                ctx,
                express_mode=True,
                product_focus_guidance_enabled=product_focus_guidance_enabled,
            )
            express_blank, express_blank_reason = _llm_result_is_blank(express_res)
            if express_blank:
                logger.info(
                    "[%s] fallback_rejected type=%s reason=%s trigger=%s",
                    url,
                    fallback_type,
                    express_blank_reason,
                    reason,
                )
                _append_trace_attempt(
                    attempt_type=fallback_type,
                    title=fallback_type.replace("_", " ").title(),
                    status="rejected",
                    source_frames=express_frames,
                    trigger_reason=reason,
                    result_payload=express_res,
                    llm_mode="express",
                    evidence_note=f"Express classifier response was rejected: {express_blank_reason}.",
                    elapsed_ms=(time.perf_counter() - attempt_started_at) * 1000.0,
                )
                return False
            _apply_fallback_result(
                fallback_type,
                express_frames,
                fallback_ocr_text if _ocr_text_has_signal(fallback_ocr_text) else "",
                express_res,
                detail="static tail frame",
                reason=reason,
            )
            attempts = processing_trace.get("attempts")
            if isinstance(attempts, list) and attempts:
                attempts[-1]["elapsed_ms"] = round((time.perf_counter() - attempt_started_at) * 1000.0, 1)
            return True

        def _extract_full_video_rescue_frames() -> list[dict[str, object]]:
            rescue_frames, rescue_cap = extract_frames_for_pipeline(
                url,
                scan_mode="Full Video",
                job_id=job_id,
            )
            if rescue_cap and rescue_cap.isOpened():
                rescue_cap.release()
            return _limit_rescue_frames(rescue_frames, _resolve_full_video_rescue_max_frames())

        initial_blank, initial_blank_reason = _llm_result_is_blank(res)
        initial_detail = "express image-only path" if express_mode else (
            "ocr skipped after high-confidence multimodal result" if not ocr_text and res is not None else "tail scan"
        )
        _append_trace_attempt(
            attempt_type="initial",
            title="Initial Express Pass" if express_mode else "Initial Tail Pass",
            status="accepted" if not initial_blank else "rejected",
            source_frames=initial_frames,
            detail=initial_detail,
            trigger_reason="" if not initial_blank else initial_blank_reason,
            ocr_text_value=ocr_text,
            result_payload=res,
            ocr_mode_used="" if express_mode else om,
            llm_mode="express" if express_mode else "standard",
            elapsed_ms=(time.perf_counter() - pipeline_started_at) * 1000.0,
        )
        context_rescue_needed, context_rescue_reason = _should_run_ocr_context_rescue(
            scan_mode=sm,
            express_mode=express_mode,
            ocr_text=ocr_text,
            res=res,
            source_frames=initial_frames,
            sorted_vision=sorted_vision,
            job_id=job_id,
        )
        context_rescue_failed = False
        if context_rescue_needed:
            context_rescue_mode = _resolve_rescue_ocr_mode(oe, om)
            context_rescue_failed = not _run_text_fallback(
                fallback_type="ocr_context_rescue",
                fallback_frames=initial_frames,
                detail="aggregated tail context",
                reason=context_rescue_reason,
                ocr_mode_override=context_rescue_mode,
                llm_text_builder=lambda expanded_text: _build_ocr_context_pack(ocr_text, expanded_text),
            )[0]
        rescue_needed, rescue_reason = _should_run_ocr_edge_rescue(
            scan_mode=sm,
            express_mode=express_mode,
            ocr_text=ocr_text,
            res=res,
        )
        if context_rescue_failed and not rescue_needed:
            rescue_needed = True
            rescue_reason = context_rescue_reason
        if rescue_needed:
            if stage_callback:
                stage_callback("ocr", f"rescue triggered; reason={rescue_reason}")
            rescue_mode = _resolve_rescue_ocr_mode(oe, om)
            rescue_frames, rescue_cap = extract_tail_rescue_frames(url, job_id=job_id)
            if rescue_cap and rescue_cap.isOpened():
                rescue_cap.release()

            rescue_text = ""
            fallback_accepted, rescue_text = _run_text_fallback(
                fallback_type="ocr_rescue",
                fallback_frames=rescue_frames,
                detail="tail rescue",
                reason=rescue_reason,
                ocr_mode_override=rescue_mode,
            )

            if not fallback_accepted:
                fallback_accepted = _run_express_fallback(
                    fallback_type="express_rescue",
                    reason=rescue_reason,
                    fallback_ocr_text=rescue_text,
                )

            if not fallback_accepted and _extended_tail_rescue_enabled():
                extended_frames, extended_cap = extract_tail_rescue_frames(
                    url,
                    lookback_seconds=_resolve_extended_tail_window_seconds(),
                    step_seconds=_resolve_extended_tail_step_seconds(),
                    job_id=job_id,
                )
                if extended_cap and extended_cap.isOpened():
                    extended_cap.release()
                fallback_accepted, _ = _run_text_fallback(
                    fallback_type="extended_tail",
                    fallback_frames=extended_frames,
                    detail="extended tail",
                    reason=rescue_reason,
                    ocr_mode_override=rescue_mode,
                )
            elif not fallback_accepted:
                logger.info(
                    "[%s] fallback_rejected type=extended_tail reason=disabled trigger=%s",
                    url,
                    rescue_reason,
                )

            if not fallback_accepted and _full_video_rescue_enabled():
                full_video_frames = _extract_full_video_rescue_frames()
                fallback_accepted, _ = _run_text_fallback(
                    fallback_type="full_video",
                    fallback_frames=full_video_frames,
                    detail="full video",
                    reason=rescue_reason,
                    ocr_mode_override=rescue_mode,
                )
            elif not fallback_accepted:
                logger.info(
                    "[%s] fallback_rejected type=full_video reason=disabled trigger=%s",
                    url,
                    rescue_reason,
                )

            if not fallback_accepted:
                logger.info("[%s] fallback_exhausted trigger=%s", url, rescue_reason)
        
        category_match = category_mapper.map_category(
            raw_category=res.get("category", "Unknown"),
            job_id=job_id,
            suggested_categories_text="",
            predicted_brand=res.get("brand", "Unknown"),
            ocr_summary=ocr_text,
            reasoning_summary=str(res.get("reasoning", "") or ""),
        )
        category_rerank_needed, category_rerank_reason, category_rerank_candidates, category_rerank_visual_matches = _should_run_category_rerank(
            result_payload=res,
            category_match=category_match,
            ocr_text=ocr_text,
            sorted_vision=sorted_vision,
        )
        if category_rerank_needed:
            if stage_callback:
                stage_callback("llm", f"category rerank provider={p.lower()} model={m}")
            logger.info(
                "[%s] fallback_triggered type=category_rerank reason=%s candidates=%s",
                url,
                category_rerank_reason,
                category_rerank_candidates,
            )
            rerank_fn = getattr(llm_engine, "query_category_rerank", None)
            if not callable(rerank_fn):
                reranked_res, rerank_status = None, "unsupported"
            else:
                reranked_res, rerank_status = rerank_fn(
                    p,
                    m,
                    brand=str(res.get("brand", "") or ""),
                    raw_category=str(res.get("category", "") or ""),
                    mapped_category=str(category_match.get("canonical_category", "") or ""),
                    ocr_text=ocr_text,
                    reasoning=str(res.get("reasoning", "") or ""),
                    candidate_categories=category_rerank_candidates,
                    visual_matches=category_rerank_visual_matches,
                    context_size=ctx,
                    product_focus_guidance_enabled=product_focus_guidance_enabled,
                )
            if isinstance(reranked_res, dict):
                reranked_exact_match = _exact_taxonomy_match_from_label(
                    str(reranked_res.get("category", "") or "")
                )
                if reranked_exact_match is not None:
                    reranked_exact_match["top_matches"] = list(
                        category_match.get("top_matches") or []
                    )
                    reranked_match = reranked_exact_match
                else:
                    reranked_match = category_mapper.map_category(
                        raw_category=reranked_res.get("category", "Unknown"),
                        job_id=job_id,
                        suggested_categories_text="",
                        predicted_brand=reranked_res.get("brand", "Unknown"),
                        ocr_summary=ocr_text,
                        reasoning_summary=str(reranked_res.get("reasoning", "") or ""),
                    )
                rerank_ok, rerank_accept_reason = _accept_category_rerank_result(
                    category_match,
                    reranked_match,
                    category_rerank_candidates,
                )
                if rerank_ok:
                    res = reranked_res
                    category_match = reranked_match
                    logger.info(
                        "[%s] fallback_accepted type=category_rerank reason=%s",
                        url,
                        rerank_accept_reason,
                    )
                    _append_trace_attempt(
                        attempt_type="category_rerank",
                        title="Category Rerank",
                        status="accepted",
                        source_frames=frames,
                        detail="candidate-only taxonomy rerank",
                        trigger_reason=category_rerank_reason,
                        ocr_text_value=ocr_text,
                        result_payload=reranked_res,
                        ocr_mode_used="" if express_mode else om,
                        llm_mode="standard",
                        evidence_note=(
                            f"Reranked within mapper candidates because {rerank_accept_reason}. "
                            f"Candidates: {', '.join(category_rerank_candidates)}."
                        ),
                    )
                else:
                    logger.info(
                        "[%s] fallback_rejected type=category_rerank reason=%s trigger=%s",
                        url,
                        rerank_accept_reason,
                        category_rerank_reason,
                    )
                    _append_trace_attempt(
                        attempt_type="category_rerank",
                        title="Category Rerank",
                        status="rejected",
                        source_frames=frames,
                        detail="candidate-only taxonomy rerank",
                        trigger_reason=category_rerank_reason,
                        ocr_text_value=ocr_text,
                        result_payload=reranked_res,
                        ocr_mode_used="" if express_mode else om,
                        llm_mode="standard",
                        evidence_note=(
                            f"Rerank was rejected because {rerank_accept_reason}. "
                            f"Candidates: {', '.join(category_rerank_candidates)}."
                        ),
                    )
            else:
                logger.info(
                    "[%s] fallback_rejected type=category_rerank reason=%s trigger=%s",
                    url,
                    rerank_status,
                    category_rerank_reason,
                )
                _append_trace_attempt(
                    attempt_type="category_rerank",
                    title="Category Rerank",
                    status="rejected",
                    source_frames=frames,
                    detail="candidate-only taxonomy rerank",
                    trigger_reason=category_rerank_reason,
                    ocr_text_value=ocr_text,
                    result_payload=None,
                    ocr_mode_used="" if express_mode else om,
                    llm_mode="standard",
                    evidence_note=(
                        f"Rerank did not run: {rerank_status}. "
                        f"Candidates: {', '.join(category_rerank_candidates)}."
                    ),
                )
        specificity_needed, specificity_reason = _should_run_specificity_search_rescue(
            enable_search=bool(enable_search),
            express_mode=express_mode,
            result_payload=res,
            category_match=category_match,
            ocr_text=ocr_text,
            sorted_vision=sorted_vision,
        )
        if specificity_needed:
            if stage_callback:
                stage_callback("llm", f"specificity search rescue provider={p.lower()} model={m}")
            logger.info("[%s] fallback_triggered type=specificity_search_rescue reason=%s", url, specificity_reason)
            visual_matches = _top_visual_matches(sorted_vision, limit=4)
            specificity_candidates = _build_specificity_search_candidates(
                raw_category=str(res.get("category", "") or ""),
                current_match=category_match,
                predicted_brand=str(res.get("brand", "") or ""),
                ocr_text=ocr_text,
                sorted_vision=sorted_vision,
            )
            refined_res, refined_reason = llm_engine.query_specificity_rescue(
                p,
                m,
                brand=str(res.get("brand", "") or ""),
                current_category=str(res.get("category", "") or category_match.get("canonical_category", "") or ""),
                ocr_text=ocr_text,
                candidate_categories=specificity_candidates,
                visual_matches=visual_matches,
                context_size=ctx,
            )
            if isinstance(refined_res, dict):
                refined_match = category_mapper.map_category(
                    raw_category=refined_res.get("category", "Unknown"),
                    job_id=job_id,
                    suggested_categories_text="",
                    predicted_brand=refined_res.get("brand", "Unknown"),
                    ocr_summary=ocr_text,
                )
                specificity_ok, specificity_accept_reason = _accept_specificity_search_result(
                    current_match=category_match,
                    refined_match=refined_match,
                    sorted_vision=sorted_vision,
                    candidate_categories=specificity_candidates,
                )
                if specificity_ok:
                    res = refined_res
                    category_match = refined_match
                    logger.info(
                        "[%s] fallback_accepted type=specificity_search_rescue reason=%s",
                        url,
                        specificity_accept_reason,
                    )
                    _append_trace_attempt(
                        attempt_type="specificity_search_rescue",
                        title="Specificity Search Rescue",
                        status="accepted",
                        source_frames=frames,
                        detail="web specificity refinement",
                        trigger_reason=specificity_reason,
                        ocr_text_value=ocr_text,
                        result_payload=refined_res,
                        ocr_mode_used="" if express_mode else om,
                        llm_mode="standard",
                        evidence_note=f"Search refinement narrowed the category because {specificity_accept_reason}. Candidates: {', '.join(specificity_candidates[:6])}.",
                    )
                else:
                    logger.info(
                        "[%s] fallback_rejected type=specificity_search_rescue reason=%s trigger=%s",
                        url,
                        specificity_accept_reason,
                        specificity_reason,
                    )
                    _append_trace_attempt(
                        attempt_type="specificity_search_rescue",
                        title="Specificity Search Rescue",
                        status="rejected",
                        source_frames=frames,
                        detail="web specificity refinement",
                        trigger_reason=specificity_reason,
                        ocr_text_value=ocr_text,
                        result_payload=refined_res,
                        ocr_mode_used="" if express_mode else om,
                        llm_mode="standard",
                        evidence_note=f"Search refinement was rejected because {specificity_accept_reason}. Candidates: {', '.join(specificity_candidates[:6])}.",
                    )
            else:
                logger.info(
                    "[%s] fallback_rejected type=specificity_search_rescue reason=%s trigger=%s",
                    url,
                    refined_reason,
                    specificity_reason,
                )
                _append_trace_attempt(
                    attempt_type="specificity_search_rescue",
                    title="Specificity Search Rescue",
                    status="rejected",
                    source_frames=frames,
                    detail="web specificity refinement",
                    trigger_reason=specificity_reason,
                    ocr_text_value=ocr_text,
                    result_payload=None,
                    ocr_mode_used="" if express_mode else om,
                    llm_mode="standard",
                    evidence_note=f"Search refinement did not run: {refined_reason}.",
                )
        cat_out = category_match["canonical_category"]
        cat_id_out = category_match["category_id"]
        signal_artifacts = {
            "mapper_plot": None,
            "mapper_top_matches": list(category_match.get("top_matches") or []),
            "mapper_query_fragments": list(category_match.get("mapping_query_fragments") or []),
            "visual_plot": None,
            "processing_trace": None,
            "llm_evidence_gallery": [],
        }
        if hasattr(category_mapper, "build_mapper_vector_plot"):
            try:
                signal_artifacts["mapper_plot"] = category_mapper.build_mapper_vector_plot(
                    raw_category=str(res.get("category", "Unknown") or "Unknown"),
                    selected_category=str(cat_out or ""),
                    predicted_brand=str(res.get("brand", "Unknown") or "Unknown"),
                    ocr_summary=ocr_text,
                    reasoning_summary=str(res.get("reasoning", "") or ""),
                )
            except Exception as exc:
                logger.debug("[%s] mapper_vector_plot_failed: %s", url, exc)
        if (
            visual_debug
            and hasattr(category_mapper, "build_visual_vector_plot")
            and enable_vision_board
        ):
            try:
                signal_artifacts["visual_plot"] = category_mapper.build_visual_vector_plot(
                    image_feature=visual_debug.get("image_feature"),
                    score_vector=visual_debug.get("score_vector"),
                    selected_category=str(cat_out or ""),
                    backend_name=str(visual_debug.get("backend") or "SigLIP"),
                    query_label=str(visual_debug.get("query_label") or "Sampled frame"),
                )
            except Exception as exc:
                logger.debug("[%s] visual_vector_plot_failed: %s", url, exc)
        _finalize_processing_trace()
        signal_artifacts["processing_trace"] = processing_trace
        if send_llm_frame:
            selected_llm_frames = _select_llm_evidence_frames(
                frames,
                _resolve_llm_recent_frame_count(),
            )
            signal_artifacts["llm_evidence_gallery"] = [
                (frame.get("ocr_image"), f"{frame.get('time')}s")
                for frame in selected_llm_frames
                if frame.get("ocr_image") is not None and frame.get("time") is not None
            ]
        row = [
            url,
            res.get("brand", "Unknown"),
            cat_id_out,
            cat_out,
            res.get("confidence", 0.0),
            res.get("reasoning", ""),
            category_match["category_match_method"],
            category_match["category_match_score"],
        ]
        
        return sorted_vision, per_frame_vision, ocr_text, f"Category: {cat_out}", [(f["ocr_image"], f"{f['time']}s") for f in frames], row, signal_artifacts
        
    except Exception as e: 
        logger.error(f"[{url}] Pipeline Worker Crash: {str(e)}", exc_info=True)
        processing_trace["summary"] = {
            "headline": "Processing crashed before a usable result was produced.",
            "attempt_count": len(processing_trace.get("attempts", [])) if isinstance(processing_trace.get("attempts"), list) else 0,
            "retry_count": max(0, len(processing_trace.get("attempts", [])) - 1) if isinstance(processing_trace.get("attempts"), list) else 0,
            "accepted_attempt_type": "",
            "trigger_reason": "pipeline_exception",
        }
        return {}, [], "Err", str(e), [], [url, "Err", "", "Err", 0, str(e), "none", None], {"mapper_plot": None, "visual_plot": None, "processing_trace": processing_trace, "llm_evidence_gallery": []}

def run_pipeline_job(
    src,
    urls,
    fldr,
    cats,
    p,
    m,
    oe,
    om,
    override,
    sm,
    enable_search,
    enable_vision_board=None,
    enable_llm_frame=None,
    product_focus_guidance_enabled=True,
    ctx=8192,
    category_embedding_model=None,
    workers=1,
    express_mode=False,
    job_id=None,
    stage_callback=None,
    enable_vision=None,  # Deprecated alias
):
    if enable_vision is not None:
        if enable_vision_board is None:
            enable_vision_board = bool(enable_vision)
        if enable_llm_frame is None:
            enable_llm_frame = bool(enable_vision)
    if enable_vision_board is None:
        enable_vision_board = True
    if enable_llm_frame is None:
        enable_llm_frame = True

    if stage_callback:
        stage_callback("ingest", "resolving input sources")
    urls_list = resolve_urls(src, urls, fldr)
    if stage_callback:
        stage_callback("ingest", f"resolved {len(urls_list)} input item(s)")
    cat_list = [c.strip() for c in cats.split(",") if c.strip()]
    master = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                copy_context().run,
                process_single_video,
                u,
                cat_list,
                p,
                m,
                oe,
                om,
                override,
                sm,
                enable_search,
                enable_vision_board,
                enable_llm_frame,
                product_focus_guidance_enabled,
                ctx,
                category_embedding_model,
                express_mode,
                job_id,
                stage_callback,
            ): u
            for u in urls_list
        }
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            if len(result) == 7:
                v, pfv, t, d, g, row, signal_artifacts = result
            else:
                v, pfv, t, d, g, row = result
                signal_artifacts = {}
            master.append(row)
            yield v, pfv, t, d, g, pd.DataFrame(master, columns=RESULT_COLUMNS), signal_artifacts
