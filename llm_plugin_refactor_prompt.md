# Prompt for LLM Plugin Refactoring

Before integrating specific model tweaks like Qwen3.5-4B's "Instruct" vs "Thinking" modes, refactor the monolithic `HybridLLM` in `video_service/core/llm.py` into a modular, plugin-based architecture. Currently, `HybridLLM` mixes HTTP request parsing, JSON cleanup, search/validation logic, and provider-specific quirks into massive `if/else` blobs.

Implement the following architectural changes:

1. **Define Core Interfaces:**
   Create an abstract base class or Protocol (e.g., `LLMProvider`) that all models must implement. It should define basic capabilities:
   - `@property def supports_vision(self) -> bool`
   - `def generate_json(self, system_prompt: str, user_prompt: str, images: list[str] = None, **kwargs) -> dict` (For standard pipeline processing)
   - `def generate_text(self, prompt: str, images: list[str] = None, **kwargs) -> str` (For agentic, open-ended reasoning)

2. **Implement Provider Plugins:**
   Extract the logic from the current `if/else` blocks into separate adapter classes implementing `LLMProvider`:
   - `GeminiCLIProvider`: Wraps `subprocess`.
   - `OllamaProvider`: Hits `/api/generate` and `/api/chat`.
   - `OpenAICompatibleProvider` (or `LMStudioProvider`): Hits `/v1/chat/completions`.
     _This structure will allow us to immediately add a custom `OllamaQwenProvider` later that automatically injects the Qwen-specific sampling parameters._

3. **Separate Business Logic from Model execution:**
   The fallback validation logic (the "AGENTIC RECOVERY" and "VALIDATION MODE" via DuckDuckGo) should not be inside the model execution method itself. Extract this orchestration into a higher-level class (e.g., `ClassificationPipeline`) that internally coordinates between the `SearchManager` and the chosen `LLMProvider`.

4. **Model Registry/Factory:**
   Create a factory method to instantiate the correct provider based on the `provider` and `backend_model` strings passed in (e.g., returning `OllamaProvider` when `provider == "ollama"`).

Make these structural changes without altering the underlying capabilities (e.g., keep the `DDGS` search, validation threshold logic, and `LLM_TIMEOUT_SECONDS` handling intact). The goal is purely architectural cleanup to support future plug-and-play models.
