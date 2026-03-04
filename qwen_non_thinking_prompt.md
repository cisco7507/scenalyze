# Implementing Qwen3.5-4B Non-Thinking Pipeline

We discovered that the 4B Instruct model mathematically forces a 30-40 second `<think>` reasoning loop regardless of standard settings like `format: json`, `temperature: 0`, or system instruction overrides.

However, we can force the model into "Non-Thinking" mode for the high-speed pipeline by bypassing the `/api/chat` wrapper entirely and using the raw `/api/generate` endpoint. By manually constructing the prompt template and pre-filling the beginning of the assistant's reply with a closed thinking block (`<think>\n</think>\n{`), the model instantly skips reasoning and streams the JSON fields.

## Required Changes to `video_service/core/llm.py`

Please update `OllamaQwenProvider.generate_json` to implement this raw prompt injection bypass.

### 1. Change Endpoint & Payload Formatting

Stop using `/api/chat` and `messages` for `generate_json`. Instead, use `/api/generate` with `raw: True`.

Construct the prompt entirely by hand using the Qwen `ChatML` format:

```python
# If images exist, encode them and inject the <|vision_start|>...<|vision_end|> tokens if needed,
# or simply stick to Ollama's `images` array if supported by generate endpoint in raw mode.
# Note: For Ollama /api/generate with raw=True, the images array still works perfectly.

raw_prompt = f\"\"\"<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
<think>
</think>
{{
  \"brand\": \"\"\"
```

**(Notice we leave the prompt hanging open right after `"brand": ` so the model instantly completes it).**

### 2. Maintain Qwen-Recommended Sampling Parameters

In the `options` dictionary for the payload, inject these specific parameters (as recommended by the Qwen documentation for non-reasoning instruct models):

- `"temperature": 1.0`
- `"top_p": 1.0`
- `"top_k": 40`
- `"presence_penalty": 2.0`
- `"num_ctx": self.context_size`

_Note: Since we are forcefully extracting JSON by manually seeding the first curly brace, we actually CAN use the high presence_penalty/temperature combo recommended by HuggingFace without triggering a hallucination loop, because the model is no longer fighting the `<think>` suppression mechanics._

### 3. Handle the Response String

Because we manually pre-filled the opening brace `{\n  "brand": `, the raw text response from the model will _only_ contain the rest of the string (e.g., `Toyota",\n  "category": ...}`).

Before returning the output, you must re-prepend the missing `{\n  "brand": ` to the `response` string so that `_clean_and_parse_json` receives a valid, complete JSON object.

### 4. Leave `generate_text` (Agent Mode) Unchanged

DO NOT modify `OllamaQwenProvider.generate_text`. The agent mode still heavily relies on the Chat API and the 30-second deep thinking capabilities of the model. This hack is _only_ for the high-speed `generate_json` pipeline endpoint.
