# Implementing "Express Mode" (Vision Only / OCR Bypass)

## The Goal

The current `ClassificationPipeline.classify` method performs a full OCR extraction across the entire video before invoking the LLM. This is incredibly safe, but slow.

We want to introduce a blazing fast **"Express Mode"** for the pipeline.

When this mode is enabled, the pipeline should **completely bypass the OCR module**. Instead of reading the whole video, it will use OpenCV to instantly scrub to the trailing end of the video file, find the single most static frame (which almost always contains the Call To Action, Brand Logo, and URL), and send only that single image payload to the VLM (Vision Language Model).

This drastically speeds up inference for 2B, 4B, and 8B VLMs by removing massive OCR text walls, allowing the model's vision geometry to natively pair the visual brand identity with a highly compressed classification prompt.

## The Instructions

Please implement the "Express Mode" pipeline fallback.

### 1. The OpenCV Extraction Logic (`video_service/core/video.py` or equivalent utility file)

Create a new generic utility function (e.g., `extract_express_brand_frame(video_path: str) -> Optional[Image.Image]`) that uses `cv2` to extract the best static end-frame:

- **Seek Backwards:** Open the `video_path` via `cv2.VideoCapture`. Query the total duration (`CAP_PROP_FRAME_COUNT` / `CAP_PROP_FPS`). Start exactly at the end (`duration - 0.1` seconds) and iterate backwards in 1.0 second leaps (e.g., `-0.1`, `-1.1`, `-2.1`, `-3.1`, `-4.1`).
- **Skip Black Frames:** Read each frame, convert to grayscale, and instantly check its average brightness (`cv2.mean(gray_frame)[0]`). If the mean brightness is `< 5.0` (pure black/fade-out padding), ignore it completely and jump back another second. Do this until you hit the first visible, illuminated frame.
- **Find the Static Plate:** Once you find the first illuminated frame (meaning the video content has technically ended), grab the next 2 visible frames directly preceding it. Compute their absolute difference (`cv2.absdiff`). Choose the frame pair with the lowest variance (closest to 0.0 diff score). That is your static CTA frame.
- **Convert and Return:** Convert that single optimal OpenCV `Mat` back to a PIL `Image.Image` (remembering to convert BGR to RGB) and return it.

### 2. The Pipeline Bypass (`video_service/core/llm.py`)

Modify the main classification driver (e.g., `ClassificationPipeline.classify`).

- Add an `express_mode: bool = False` argument to the method signature.
- If `express_mode` is `True`:
  1. **Skip OCR:** Do not call the OCR processing module or any transcription utilities at all. Set the `text` variable to empty.
  2. **Fetch Frame:** Call your newly created `extract_express_brand_frame(video_path)` utility. If it fails, fallback to grabbing the standard middle frame.
  3. **Ultra-Concise Prompting:** For `express_mode`, use a stripped-down, highly rigid system prompt: `"You are an Ad Classifier. Examine this final frame of a video commercial. Return a JSON object with 'brand' and 'category'. Do not output anything else."` Do not include complex conditional logic or examples from the standard prompt, to optimize for 2B parameter attention spans.
  4. **Call LLM:** Pass the single extracted base64 frame and the concise prompt directly to the LLM backend (e.g., `HybridLLM.query_pipeline`).
  5. Return the result object immediately.

### Constraints

- This feature is an optional toggle. The default complete OCR pipeline must remain intact as the primary fallback.
- The `fast` frame extraction must not re-read the entire video from the beginning. It must use explicit seek points (`CAP_PROP_POS_MSEC` or `CAP_PROP_POS_FRAMES`) to jump instantly to the 90% mark.
