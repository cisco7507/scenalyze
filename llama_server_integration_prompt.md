# Building Native Full-Stack Llama-Server Integration

## The Goal

Currently, the user is running `llama-server` locally on port 1234, processing vision requests using packed GGUF models. However, the system relies on hacking the "LM Studio" provider to interact with `llama-server`.

We want to elevate `llama-server` to a natively supported, primary LLM provider across the entire stack.

This integration must:

1. Provide a dedicated `"llama-server"` option in the UI backend.
2. Automatically fetch the running GGUF model via the OpenAI-compatible `/v1/models` endpoint for the frontend dropdown.
3. Automatically lock inference responses to valid JSON via the OpenAI-compatible `response_format` JSON parameter safely handled in the provider layer.

## The Instructions

Please implement the following full-stack changes to seamlessly tie `llama-server` into the system.

### 1. Backend Provider Updates (`video_service/core/llm.py`)

Update the `OpenAICompatibleProvider` to optionally enforce strict JSON GBNF grammars, and update the router.

- **`__init__`**: Modify the `OpenAICompatibleProvider` constructor to accept `force_json_mode: bool = False`, storing it on `self`.
- **`generate_json`**: In the `OpenAICompatibleProvider.generate_json` method, update the `payload` dictionary. **Only** if `self.force_json_mode` is True, inject `"response_format": {"type": "json_object"}` into the JSON payload out to the server. Do not touch `generate_text`.
- **`create_provider` Factory**: Inside the `create_provider(provider, backend_model, context_size)` factory function at the bottom:
  - Add a dedicated `elif provider_lower == "llama-server":` block.
  - Return an instance of `OpenAICompatibleProvider`, pointing it at the same `backend_model` and `context_size`, but passing `force_json_mode=True`.
  - _(Ensure the `lm studio` block also passes `force_json_mode=True` to retroactively give LM Studio the same JSON protection)._

### 2. Backend API Route Integration (`video_service/api/routes.py` or equivalent)

Find the FastAPI endpoint endpoint that serves available models to the frontend (often under a router like `@router.get("/models")` matching `/api/v1/models`).

- Add logic to scrape `http://localhost:1234/v1/models` (the `OPENAI_COMPAT_URL` base) when queried for `llama-server` models.
- Issue an HTTP GET request. `llama-server` will return a standard OpenAI `/v1/models` response format (`{"data": [{"id": "unsloth/Qwen3.5-4B-GGUF", ...}]}`). Extract the `id` field from the data array and append it to your returned model list to auto-populate the frontend.

### 3. Frontend Integration (`frontend/src/components/` or equivalent)

Find the component handling the job submission form where providers and models are selected (look for standard dropdown lists or arrays configuring the providers).

- Add `"Llama Server"` to the list of available LLM Providers.
- Ensure the frontend fetches and populates the models dropdown from your updated backend `/api/v1/models` endpoint when "Llama Server" is selected, just as it does for Ollama.

### Constraints

- Do not create a completely new provider class. Reuse `OpenAICompatibleProvider`.
- Do not make breaking GUI changes. Just add the new options to the existing dropdown state array.
- All LLM interactions must continue handling timeouts gracefully.
