# Prompt for Qwen3.5-4B Integration (Plugin Architecture)

Please modify `video_service/core/llm.py` to optimally support `Qwen3.5-4B` (Instruct) for the video ad classifier, utilizing the newly implemented plugin architecture.

Based on Qwen's official Hugging Face documentation, implement the following changes:

1. **Create an `OllamaQwenProvider` Plugin**:
   Add a new class `OllamaQwenProvider` that inherits from `OllamaProvider`. This provider must override both `generate_json` and `generate_text` to inject Qwen's highly specific recommended sampling parameters for its two distinct modes:
   - In `generate_json` (used by the Standard Pipeline for strict output without reasoning), use the **Instruct (or non-thinking) mode for reasoning tasks**:
     - `temperature: 1.0`
     - `top_p: 1.0`
     - `top_k: 40`
     - `presence_penalty: 2.0`
     - `repetition_penalty: 1.0`
   - In `generate_text` (used by the Agent for open-ended, multi-step ReAct tasks), use the **Thinking mode for general tasks**:
     - `temperature: 1.0`
     - `top_p: 0.95`
     - `top_k: 20`
     - `presence_penalty: 1.5`
     - `repetition_penalty: 1.0`

2. **Migrate Agent to Chat API**:
   Inside `OllamaQwenProvider.generate_text`, ensure the payload targets the `/api/chat` endpoint with a `messages` array (system and user roles) instead of the legacy `/api/generate` endpoint used by the base `OllamaProvider`. Qwen3.5 responds much more reliably to a structured conversation format.

3. **Register the New Provider**:
   Update the `create_provider` factory function. If `provider == "ollama"` AND the `backend_model` contains `"qwen"`, it should instantiate and return the newly created `OllamaQwenProvider`. Otherwise, it should safely fall back to the generic `OllamaProvider`.

4. **Vision Handling**:
   Qwen3.5-4B natively includes a vision encoder. Verify that `OllamaQwenProvider` correctly formats and passes the `images` list within its `/api/chat` payload just like the base provider does, as it fully supports multimodal inputs.
