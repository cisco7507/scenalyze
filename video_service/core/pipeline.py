import time
import os
import re
import torch
import cv2
import numpy as np
import pandas as pd
import concurrent.futures
from contextvars import copy_context
from video_service.core.utils import logger, device, TORCH_DTYPE
from video_service.core.video_io import (
    extract_express_brand_frame,
    extract_frames_for_pipeline,
    extract_middle_frame,
    get_pil_image,
    resolve_urls,
)
from video_service.core import categories as categories_runtime
from video_service.core.categories import category_mapper, normalize_feature_tensor
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
    selected: list[dict[str, object]] = [frames[0]]
    skipped = 0
    last_selected = frames[0]
    last_index = len(frames) - 1

    for idx, frame in enumerate(frames[1:], start=1):
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

    if len(selected) == 2:
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
    ctx=8192,
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

    try:
        if stage_callback:
            stage_callback("ingest", "validating and preparing input")
        logger.info(f"[{url}] === STARTING PIPELINE WORKER ===")
        express_mode = bool(express_mode)
        send_llm_frame = bool(enable_llm_frame or express_mode)
        visual_debug: dict[str, object] | None = None
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
        ) -> str:
            logger.debug("[%s] parallel_task_start: ocr", url)
            try:
                if stage_callback:
                    stage_callback("ocr", f"ocr engine={oe.lower()}")
                dedup_threshold = _resolve_ocr_dedup_threshold()
                if prepared_frames is None or visually_skipped_count is None:
                    ocr_frames, visually_skipped_count = _select_frames_for_ocr(frames)
                else:
                    ocr_frames = prepared_frames
                ocr_lines: list[str] = []
                prev_normalized: str | None = None
                skipped_count = 0
                roi_hits = 0
                roi_fallbacks = 0
                early_stop_skipped = 0
                no_roi_skipped = 0
                ocr_call_count = 0
                ocr_elapsed_seconds = 0.0
                early_stop_active = _ocr_early_stop_enabled(sm)
                last_index = len(ocr_frames) - 1
                idx = 0

                def _run_ocr(target_image: np.ndarray) -> str:
                    nonlocal ocr_call_count, ocr_elapsed_seconds
                    started_at = time.perf_counter()
                    try:
                        return ocr_manager.extract_text(oe, target_image, om)
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
                        _ocr_roi_enabled(oe) or _ocr_skip_no_roi_enabled(oe, sm)
                    )
                    if roi_capable:
                        roi_image, roi_used = _extract_ocr_focus_region(ocr_image)
                    if (
                        _ocr_skip_no_roi_enabled(oe, sm)
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
                    tail_image = get_pil_image(frames[-1]) if send_llm_frame else None
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
                    vision_future = pool.submit(_do_vision)
                    ocr_future = pool.submit(_do_ocr)
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
            tail_image = get_pil_image(frames[-1]) if send_llm_frame else None
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
            )
        
        category_match = category_mapper.map_category(
            raw_category=res.get("category", "Unknown"),
            job_id=job_id,
            suggested_categories_text="",
            predicted_brand=res.get("brand", "Unknown"),
            ocr_summary=ocr_text,
        )
        cat_out = category_match["canonical_category"]
        cat_id_out = category_match["category_id"]
        signal_artifacts = {
            "mapper_plot": None,
            "visual_plot": None,
        }
        if hasattr(category_mapper, "build_mapper_vector_plot"):
            try:
                signal_artifacts["mapper_plot"] = category_mapper.build_mapper_vector_plot(
                    raw_category=str(res.get("category", "Unknown") or "Unknown"),
                    selected_category=str(cat_out or ""),
                    predicted_brand=str(res.get("brand", "Unknown") or "Unknown"),
                    ocr_summary=ocr_text,
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
        return {}, [], "Err", str(e), [], [url, "Err", "", "Err", 0, str(e), "none", None], {"mapper_plot": None, "visual_plot": None}

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
    ctx=8192,
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
                ctx,
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
