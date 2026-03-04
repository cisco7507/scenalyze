# Implementing LM Studio Non-Thinking Pipeline

Based on our curl tests, we confirmed that LM Studio (which runs `llama.cpp` and `MLX` under the hood) supports an undocumented but highly effective feature: **Assistant Pre-fill in the Chat API**.

By injecting an `"assistant"` message at the very end of the `messages` array containing our `</think>\n{\n  "brand": "` hack, LM Studio skips the internal reasoning engine completely and instantly streams the rest of the JSON string.

## Required Changes to `video_service/core/llm.py`

Please update `OpenAICompatibleProvider.generate_json` to implement this Assistant Pre-fill bypass.

### 1. Update the `messages` Array

Before building the payload, define the `json_prefix` and append it as an `assistant` message to the end of the `msgs` list.

```python
        json_prefix = '{\n  \"brand\": \"'
        msgs = [
            {\"role\": \"system\", \"content\": system_prompt},
            {\"role\": \"user\", \"content\": user_prompt},
            {\"role\": \"assistant\", \"content\": f\"</think>\\n{json_prefix}\"},
        ]
```

### 2. Guard the Image Injection Index

Currently, the image injection logic assumes the user message is always at index `1`. Keep it that way since we just added the assistant message at index `2`.

```python
        if images:
            msgs[1][\"content\"] = [
                {\"type\": \"text\", \"text\": user_prompt},
                {\"type\": \"image_url\", \"image_url\": {\"url\": f\"data:image/jpeg;base64,{images[0]}\"}},
            ]
```

### 3. Apply Qwen-Recommended Sampling Parameters

Update the `options`/body payload to include the higher temperature and presence penalty recommended for non-reasoning instruct models, to match our Ollama implementation.

```python
        payload = {
            \"model\": self.backend_model,
            \"messages\": msgs,
            \"temperature\": 0.0,        # Increased from 0.1
            \"top_p\": 1.0,
            \"presence_penalty\": 2.0    # Added vendor recommendation
        }
```

### 4. Reconstruct the JSON Response

Since we pre-filled `{\n  "brand": "`, LM Studio will return the text _immediately following_ that string (e.g., `Toyota",\n  "category": "Automotive"...}`). You must catch the raw response and prepend the `json_prefix` back onto it before passing it to `_clean_and_parse_json`.

```python
        try:
            resp = requests.post(OPENAI_COMPAT_URL, json=payload, timeout=LLM_TIMEOUT_SECONDS)
            resp.raise_for_status()

            # Safely extract the content string
            content_json = resp.json()
            raw_content = \"\"
            if \"choices\" in content_json and len(content_json[\"choices\"]) > 0:
                raw_content = content_json[\"choices\"][0].get(\"message\", {}).get(\"content\", \"\")

            # Reconstruct the valid JSON string
            content = raw_content if \"{\" in raw_content else f\"{json_prefix}{raw_content}\"

            return _clean_and_parse_json(content)
```

_(Leave `generate_text` alone, as it still needs the 30-second thinking block for Agent loops!)_
