import time
import re
import torch
import pandas as pd
from video_service.core.utils import logger, device, TORCH_DTYPE
from video_service.core import categories as categories_runtime
from video_service.core.categories import category_mapper, normalize_feature_tensor
from video_service.core.video_io import extract_frames_for_agent, resolve_urls, get_pil_image
from video_service.core.ocr import ocr_manager
from video_service.core.llm import llm_engine, search_manager

# Backward-compatible aliases for tests that monkeypatch these symbols.
siglip_model = None
siglip_processor = None

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


def _get_siglip_handles():
    model = siglip_model if siglip_model is not None else categories_runtime.siglip_model
    processor = (
        siglip_processor if siglip_processor is not None else categories_runtime.siglip_processor
    )
    return model, processor


def _ensure_react_vision_ready() -> bool:
    siglip_model, siglip_processor = _get_siglip_handles()
    if hasattr(category_mapper, "ensure_vision_text_features"):
        ready, reason = category_mapper.ensure_vision_text_features()
        if not ready:
            logger.info("react_vision_unavailable: %s", reason)
            return False
        siglip_model, siglip_processor = _get_siglip_handles()
        return siglip_model is not None and siglip_processor is not None

    # Backward-compatible fallback for test doubles without helper method.
    if not getattr(category_mapper, "categories", None):
        return False
    if siglip_model is None or siglip_processor is None:
        return False
    if getattr(category_mapper, "vision_text_features", None) is not None:
        return True
    try:
        vision_prompts = [f"A video ad for {cat}" for cat in category_mapper.categories]
        text_inputs = siglip_processor(
            text=vision_prompts,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_features = siglip_model.get_text_features(**text_inputs)
            category_mapper.vision_text_features = normalize_feature_tensor(
                text_features,
                source="SigLIP.get_text_features",
            )
        return True
    except Exception:
        category_mapper.vision_text_features = None
        return False


class AdClassifierAgent:
    def __init__(self, max_iterations=4):
        self.max_iterations = max_iterations

    def run(
        self,
        frames_data,
        categories,
        provider,
        model,
        ocr_engine,
        ocr_mode,
        allow_override,
        enable_search,
        enable_vision_board=None,
        enable_llm_frame=None,
        context_size=8192,
        job_id=None,
        ocr_summary="",
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

        memory_log = "Initial State: I am investigating a chronological storyboard of scenes extracted from an ad.\n"
        pil_images = [get_pil_image(f) for f in frames_data]
        yield memory_log, "Unknown", "Unknown", "", "N/A", "Agent is thinking...", "pending", None
        
        for step in range(self.max_iterations):
            tools_list = ["- [TOOL: OCR] (Use first to extract all visible text from the video frames)"]
            examples_list = ["[TOOL: OCR]"]
            protocol_steps = ["1. You MUST always start by using [TOOL: OCR]."]
            step_num = 2
            
            if enable_search:
                tools_list.append('- [TOOL: SEARCH | query="search term"] (Use to web search company names, slogans, or partial URLs found in OCR)')
                examples_list.append('[TOOL: SEARCH | query="Nike slogan"]')
                protocol_steps.append(f"{step_num}. You MUST use [TOOL: SEARCH] at least once to fact-check the brand name or slogan found in the OCR before you are allowed to finish.")
                step_num += 1
                
            vision_tool_available = enable_vision_board and _ensure_react_vision_ready()
            if vision_tool_available:
                tools_list.append('- [TOOL: VISION] (Use to check the visual probability against our official industry categories)')
                examples_list.append('[TOOL: VISION]')
                protocol_steps.append(f"{step_num}. (Optional) Use [TOOL: VISION] if you are still unsure about the product context.")
                step_num += 1
            
            tools_list.append('- [TOOL: FINAL | brand="Brand", category="Category", reason="Logic"] (Use only when you have confidently identified the brand and category)')
            examples_list.append('[TOOL: FINAL | brand="Apple", category="Tech", reason="Apple logo and website found in OCR"]')
            
            tools_str = "\n".join(tools_list)
            examples_str = "\n".join(examples_list)
            protocol_str = "\n".join(protocol_steps)

            system_prompt = f"""You are a Senior Marketing Analyst and Global Brand Expert.
Your goal is to categorize video advertisements by combining extracted text (OCR) with your vast internal knowledge of companies, slogans, and industries.
Rely on Internal Brand Knowledge: You know every major brand, their parent companies, and their marketing styles. Use this knowledge as a strong prior, but direct OCR brand text, domains, explicit market cues, and on-frame evidence override memory when they conflict.
Treat OCR as Noisy Hints: The extracted OCR text is machine-generated and may contain typos, missing letters, and random artifacts. DO NOT blindly trust or copy the OCR text. Use your knowledge to autocorrect obvious errors.
IMPORTANT — Bilingual Content: The ads you analyze may be in English OR French (or a mix of both). French words and phrases are NOT OCR errors — they are legitimate content. Use them to identify brands, products, and categories just as you would English text.
(e.g., if OCR says 'Strbcks' or 'Star bucks co', you know the true brand is 'Starbucks'. But if OCR says 'Économisez avec Desjardins' or 'Assurance auto', those are valid French — do NOT treat them as typos).
Slogan-only brand matches are low-trust unless corroborated by an exact brand token, branded domain, country/market cue, or explicit web confirmation.
IGNORE TIMESTAMPS: The OCR and Scene data text will be prefixed with bracketed timestamps like '[71.7s]' or '[12.5s]'. THESE ARE NOT PART OF THE AD. Do NOT use these numbers to identify brands or products (e.g. do not guess 'Boeing 717' just because you see '[71.7s]'). Ignore them completely.
Determine the most appropriate product or service category. If Override Allowed is True, you may generate a professional category when the ad does not fit neatly into a standard industry label.

CRITICAL PROTOCOL - YOU MUST FOLLOW THESE STEPS IN ORDER:
{protocol_str}

CRITICAL INSTRUCTION: You MUST output exactly ONE tool command per turn. 
You must use the EXACT bracket syntax below. DO NOT output any conversational text. DO NOT output markdown blocks.

Tools available:
{tools_str}

Valid Examples:
{examples_str}

Current Memory:
{memory_log}"""
            
            if stage_callback:
                stage_callback("llm", f"calling provider={provider.lower()} model={model}")
            response = llm_engine.query_agent(
                provider,
                model,
                system_prompt,
                images=pil_images,
                force_multimodal=enable_llm_frame,
                context_size=context_size,
            )
            
            if not response:
                response = "[TOOL: ERROR | reason=\"LLM returned absolute empty string. Check backend.\"]"

            thought = response.split('[TOOL:')[0].strip() if '[TOOL:' in response else response
            thought_delta = f"Thought: {thought}\n" if thought else "Thought:\n"
            yield thought_delta, "Unknown", "Unknown", "", "N/A", "Executing Tool...", "pending", None
            
            tool_match = re.search(r"\[TOOL:\s*(.*?)(?:\|\s*(.*?))?\]", response)
            observation = ""
            
            if tool_match:
                tool_name = tool_match.group(1).strip()
                kwargs = dict(re.findall(r'(\w+)="(.*?)"', tool_match.group(2) or ""))

                if tool_name == "FINAL":
                    brand = kwargs.get("brand", "Unknown")
                    raw_cat = kwargs.get("category", "Unknown")
                    category_match = category_mapper.map_category(
                        raw_category=raw_cat,
                        job_id=job_id,
                        suggested_categories_text="",
                        predicted_brand=brand,
                        ocr_summary=ocr_summary,
                    )
                    official_cat = category_match["canonical_category"]
                    cat_id = category_match["category_id"]
                    reason = kwargs.get("reason", "No reason provided")
                    if raw_cat != official_cat:
                        reason += f" [Mapped from '{raw_cat}']"

                    final_delta = (
                        f"--- Step {step + 1} ---\n"
                        f"Action: {response}\n"
                        "Result: Final tool accepted.\n"
                        "FINAL CONCLUSION REACHED.\n"
                    )
                    memory_log += f"\n{final_delta}"
                    yield final_delta, brand, official_cat, cat_id, "N/A", reason, category_match["category_match_method"], category_match["category_match_score"]
                    return
                    
                elif tool_name == "OCR":
                    all_findings = []
                    for f in frames_data:
                        text = ocr_manager.extract_text(ocr_engine, f["ocr_image"], mode=ocr_mode)
                        if text:
                            all_findings.append(text)
                    observation = "Observation: " + (" | ".join(all_findings) if all_findings else "No text found.")
                    
                elif tool_name == "VISION":
                    if not enable_vision_board:
                        observation = "Observation: Formatting ERROR. The VISION tool is disabled by user settings. Proceed without it."
                    elif _ensure_react_vision_ready():
                        siglip_model, siglip_processor = _get_siglip_handles()
                        with torch.no_grad():
                            image_inputs = siglip_processor(images=pil_images, return_tensors="pt").to(device)
                            if TORCH_DTYPE != torch.float32:
                                image_inputs = {k: v.to(dtype=TORCH_DTYPE) if torch.is_floating_point(v) else v for k, v in image_inputs.items()}
                            image_features = siglip_model.get_image_features(**image_inputs)
                            image_features = normalize_feature_tensor(
                                image_features,
                                source="SigLIP.get_image_features",
                            )
                            
                            logit_scale = siglip_model.logit_scale.exp()
                            logit_bias = siglip_model.logit_bias
                            logits_per_image = (image_features @ category_mapper.vision_text_features.t()) * logit_scale + logit_bias
                            probs = torch.sigmoid(logits_per_image)
                            
                        scores = probs.mean(dim=0).cpu().numpy()
                        top_cats = dict(sorted({category_mapper.categories[i]: float(scores[i]) for i in range(len(category_mapper.categories))}.items(), key=lambda item: item[1], reverse=True)[:5])
                        observation = f"Observation: Vision Model's Top 5 matches from the official CSV taxonomy: {top_cats}"
                    else:
                        observation = "Observation: Vision Model unavailable or text embeddings failed to cache."
                
                elif tool_name == "SEARCH":
                    if not enable_search:
                        observation = "Observation: Formatting ERROR. Web Search is disabled by user settings. Proceed without searching."
                    else:
                        observation = f"Observation from Web: {search_manager.search(kwargs.get('query', ''))}"
            else:
                observation = "Observation: Formatting ERROR. Missing [TOOL: ] syntax. Remember to ONLY output the tool command."

            step_delta = f"--- Step {step + 1} ---\nAction: {response}\nResult: {observation}\n"
            memory_log += f"\n{step_delta}"
            yield step_delta, "Unknown", "Unknown", "", "N/A", "Step completed", "pending", None

        category_match = category_mapper.map_category(
            raw_category="Unknown",
            job_id=job_id,
            suggested_categories_text="",
            predicted_brand="Unknown",
            ocr_summary=ocr_summary,
        )
        timeout_delta = "Agent Timeout: Max iterations reached.\n"
        memory_log += f"\n{timeout_delta}"
        yield (
            timeout_delta,
            "Unknown",
            category_match["canonical_category"],
            category_match["category_id"],
            "N/A",
            "Agent Timeout: Max iterations reached.",
            category_match["category_match_method"],
            category_match["category_match_score"],
        )

def run_agent_job(
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
    category_embedding_model=None,
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
    if hasattr(category_mapper, "configure_embedding_model"):
        category_mapper.configure_embedding_model(category_embedding_model)

    if stage_callback:
        stage_callback("ingest", "resolving input sources")
    urls_list = resolve_urls(src, urls, fldr)
    if stage_callback:
        stage_callback("ingest", f"resolved {len(urls_list)} input item(s)")
    cat_list = [c.strip() for c in cats.split(",") if c.strip()]
    master = []
    agent = AdClassifierAgent()
    
    for url in urls_list:
        yield f"Processing {url}...", [], pd.DataFrame(master, columns=RESULT_COLUMNS), category_mapper.get_nebula_plot()
        try:
            if stage_callback:
                stage_callback("frame_extract", "extracting frames for agent mode")
            frames, cap = extract_frames_for_agent(url, job_id=job_id)
            if cap and cap.isOpened():
                cap.release()
            gallery = [(f["ocr_image"], f"{f['time']}s") for f in frames]
            ocr_chunks = []
            if stage_callback:
                stage_callback("ocr", f"ocr engine={oe.lower()}")
            for f in frames:
                txt = ocr_manager.extract_text(oe, f["ocr_image"], mode=om)
                if txt:
                    ocr_chunks.append(txt)
            ocr_summary = " ".join(ocr_chunks)[:600]
            if stage_callback and enable_vision_board:
                stage_callback("vision", "vision enabled; evaluating category cues")
            
            for agent_output in agent.run(
                frames,
                cat_list,
                p,
                m,
                oe,
                om,
                override,
                enable_search,
                enable_vision_board,
                enable_llm_frame,
                ctx,
                job_id=job_id,
                ocr_summary=ocr_summary,
                stage_callback=stage_callback,
            ):
                if len(agent_output) == 8:
                    log, b, c, cid, conf, r, match_method, match_score = agent_output
                elif len(agent_output) == 6:
                    log, b, c, cid, conf, r = agent_output
                    match_method, match_score = "pending", None
                else:
                    raise ValueError(f"Unexpected agent output length: {len(agent_output)}")

                brand, cat, cat_id, reason = b, c, cid, r
                category_match_method, category_match_score = match_method, match_score
                yield log, gallery, pd.DataFrame(master, columns=RESULT_COLUMNS), category_mapper.get_nebula_plot(cat)
            
            master.append([url, brand, cat_id, cat, "N/A", reason, category_match_method, category_match_score])
            yield "Result row appended.\n", gallery, pd.DataFrame(master, columns=RESULT_COLUMNS), category_mapper.get_nebula_plot(cat)
            time.sleep(4)
        except Exception as e:
            master.append([url, "Error", "", "Error", "N/A", str(e), "none", None])
