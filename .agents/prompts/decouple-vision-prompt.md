# Decouple Vision Board and LLM Frame Input

## Objective

Split the current monolithic "Enable Vision Tool" setting into two independent configurations across both the API and the UI:

1. **Enable Vision Board (SigLIP/OpenCLIP):** Controls whether we extract dense text/image features and score them against the taxonomy to create the "Vision" panel results.
2. **Send Frame to LLM:** Controls whether the final extracted frame actually gets attached and sent to the multimodal LLM provider alongside the OCR text.

## API Changes

1. Deprecate the `enable_vision: bool` field on the `Job` model and the API parameters.
2. Replace it with two explicit fields:
   - `enable_vision_board: bool = True`
   - `enable_llm_frame: bool = True`
3. Update `video_service/app/models/job.py`, `video_service/app/main.py`, and `video_service/workers/worker.py` to correctly map these new parsed JSON/Form payloads into the execution queues.

## Backend Core Changes

1. **Pipeline Mode (`video_service/core/pipeline.py`):**
   - The parallel `_do_vision()` block must now be gated solely by `enable_vision_board`.
   - The LLM inference call `llm_engine.query_pipeline(...)` should check `enable_llm_frame` to determine whether `tail_image` is populated with PIL data or set to `None`.
2. **Agent Mode (`video_service/core/agent.py`):**
   - The system prompt injection of `[TOOL: VISION]` must now be entirely gated by `enable_vision_board`.
   - The `force_multimodal` parameter passed to `llm_engine.query_agent` must now receive `enable_llm_frame` so the agent is only sent an image if explicitly permitted.

## UI Dashboard Changes

1. In the frontend submission modal (`SettingsPanel` / Dashboard UI):
   - Replace the single "👁️ Vision Tool: Enable Image Analysis" toggle.
   - Inject two distinct switches:
     - "📸 Generate Vision Board (SigLIP/OpenCLIP)"
     - "🧠 Send Keyframe to LLM"
2. Update the frontend API payload constructor to POST `enable_vision_board` and `enable_llm_frame` accurately instead of passing the legacy `enable_vision` flag.
