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
from video_service.core.llm import llm_engine

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
    siglip_variant="v1",
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
            logger.debug("[%s] parallel_task_start: vision", url)
            try:
                ready, reason = category_mapper.ensure_vision_text_features()
                variant_result = categories_runtime._ensure_siglip_variant_loaded(siglip_variant)
                if variant_result is not None:
                    siglip_model, siglip_processor = variant_result
                else:
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

        def _do_ocr() -> str:
            logger.debug("[%s] parallel_task_start: ocr", url)
            try:
                if stage_callback:
                    stage_callback("ocr", f"ocr engine={oe.lower()}")
                dedup_threshold = _resolve_ocr_dedup_threshold()
                ocr_lines: list[str] = []
                prev_normalized: str | None = None
                skipped_count = 0
                last_index = len(frames) - 1
                for idx, frame in enumerate(frames):
                    raw_text = ocr_manager.extract_text(oe, frame["ocr_image"], om)
                    normalized = _normalize_ocr(raw_text)
                    is_last_frame = idx == last_index
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
                        continue
                    # Do not inject frame timestamps into OCR text; they pollute LLM/search input.
                    ocr_lines.append(raw_text)
                    prev_normalized = normalized
                logger.info(
                    "ocr_dedup: processed=%d skipped=%d total_frames=%d threshold=%.2f",
                    len(ocr_lines),
                    skipped_count,
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
        if express_mode:
            if stage_callback:
                stage_callback("ocr", "express mode enabled; OCR bypassed")
            ocr_text = ""
            if enable_vision_board:
                sorted_vision, per_frame_vision = _do_vision()
        else:
            if enable_vision_board:
                parallel_t0 = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    vision_future = pool.submit(_do_vision)
                    ocr_future = pool.submit(_do_ocr)
                    ocr_text = ocr_future.result()
                    sorted_vision, per_frame_vision = vision_future.result()
                logger.info("parallel_ocr_vision: completed in %.2fs", time.time() - parallel_t0)
            else:
                ocr_text = _do_ocr()
        if stage_callback:
            stage_callback("llm", f"calling provider={p.lower()} model={m}")
        send_llm_frame = bool(enable_llm_frame or express_mode)
        tail_image = get_pil_image(frames[-1]) if send_llm_frame else None
        # Fall back to category_mapper.categories (from categories.csv) when user
        # did not supply explicit categories.  This ensures both the LLM prompt and
        # the JSON Schema enum constraint receive the canonical category list.
        effective_categories = categories if categories else category_mapper.categories
        res = llm_engine.query_pipeline(
            p,
            m,
            ocr_text,
            effective_categories,
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
        
        return sorted_vision, per_frame_vision, ocr_text, f"Category: {cat_out}", [(f["ocr_image"], f"{f['time']}s") for f in frames], row
        
    except Exception as e: 
        logger.error(f"[{url}] Pipeline Worker Crash: {str(e)}", exc_info=True)
        return {}, [], "Err", str(e), [], [url, "Err", "", "Err", 0, str(e), "none", None]

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
    siglip_variant="v1",
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
                None,  # enable_vision deprecated alias
                siglip_variant,
            ): u
            for u in urls_list
        }
        for fut in concurrent.futures.as_completed(futures):
            v, pfv, t, d, g, row = fut.result()
            master.append(row)
            yield v, pfv, t, d, g, pd.DataFrame(master, columns=RESULT_COLUMNS)
