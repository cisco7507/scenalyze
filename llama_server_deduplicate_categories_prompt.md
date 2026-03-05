# Prompt: Eliminate Duplicate Category Transmission for llama-server and LM Studio

## Problem

When the provider is **llama-server** or **LM Studio**, the full 1,100-category list is sent to the LLM **twice** in every request:

1. **In the user prompt text** — `query_pipeline()` builds `f"Categories: {categories}\nOverride: {override}"` and passes it as the user message content.
2. **In the JSON schema `enum`** — `generate_json()` constructs a `response_format` with `"json_schema"` containing `"enum": [all 1100 categories]` when `force_json_mode=True` and `suggested_categories` is non-empty.

Both paths receive the same `effective_categories` (all 1,100 from `category_mapper.categories`). This means every request carries **~120 KB** of duplicated category data, needlessly inflating the prompt and wasting context window tokens.

**Evidence**: TCP capture (`llama-Server-duplicate categories.txt`) shows the full list appears twice in a single HTTP POST to `/v1/chat/completions`.

**Ollama confirmation**: Ollama is NOT affected. Its `generate_json()` in `OllamaProvider` / `OllamaQwenProvider` does not accept `suggested_categories`, does not construct a `response_format`, and sends categories only once in the user prompt text. No changes needed for Ollama.

**Current `create_provider` state**: Both llama-server AND LM Studio use `OpenAICompatibleProvider` with `force_json_mode=True`. This means both providers get the JSON schema enum AND the text prompt categories — double transmission.

## Scope

This prompt addresses **llama-server** and **LM Studio**. Ollama is not affected and requires no changes.

## Desired Behavior

### llama-server (`force_json_mode=True`)

- The JSON schema `enum` constraint **already forces** the model to pick from the allowed category list — there is no need to also list them in the free-text user prompt.
- **Remove categories from the user prompt text** when the provider is llama-server.
- **Keep the JSON schema enum as-is** — it is the authoritative constraint and llama-server's grammar engine handles it efficiently.

### LM Studio (`force_json_mode=True`)

- LM Studio also uses `OpenAICompatibleProvider` with `force_json_mode=True`, so it also gets categories twice.
- **Same fix**: remove categories from the user prompt text.
- **Keep the JSON schema enum** — LM Studio's JSON grammar can use it as the constraint.
- **Additional benefit**: This directly fixes the "context length exceeded" 400 error seen with LM Studio, since the user prompt will be dramatically shorter.

### What the LLM receives after the fix

For both llama-server and LM Studio, the LLM should rely on:

- Its internal brand knowledge (from the system prompt)
- OCR text and/or the tail frame image (from the user prompt)
- The JSON schema enum (as the structural constraint on the `category` output field)

It does **not** need the categories repeated as free text.

## Files to Modify

### `video_service/core/llm.py`

#### `HybridLLM.query_pipeline()`

Currently (both express and non-express modes):

```python
# Express mode:
user_prompt = f"Categories: {categories}\nOverride: {override}"

# Non-express mode:
user_prompt = f'Categories: {categories}\nOverride: {override}\nOCR Text: "{text}"'
```

Change: When the provider uses `force_json_mode=True` (llama-server or LM Studio), omit the `Categories:` line from `user_prompt`. The simplest approach:

- Add a parameter like `skip_prompt_categories: bool = False` to `query_pipeline()`.
- When `skip_prompt_categories` is `True`, build the user prompt **without** the categories line:
  - Express mode: `f"Override: {override}"`
  - Non-express mode: `f'Override: {override}\nOCR Text: "{text}"'`
- The `suggested_categories` kwarg passed to `classify()` → `generate_json()` still carries the full list for the JSON schema enum. This is unchanged.

#### `create_provider()`

No changes needed — it already sets `force_json_mode=True` for both llama-server and LM Studio.

### `video_service/core/pipeline.py`

#### `process_single_video()`

Currently:

```python
res = llm_engine.query_pipeline(
    p, m, ocr_text, effective_categories,
    tail_image, override, enable_search,
    send_llm_frame, ctx, express_mode=express_mode,
)
```

Change: Determine whether to skip prompt categories based on the provider name. When `p.lower()` matches `"llama server"`, `"llama-server"`, `"lm studio"`, `"openai compatible"`, or `"openai-compatible"`, pass `skip_prompt_categories=True`.

## What NOT to Change

- **Ollama**: Not affected — categories appear only once (in user prompt text). No JSON schema enum is sent. No changes.
- **JSON schema enum logic in `generate_json()`**: Keep as-is. The enum is the correct mechanism for both llama-server and LM Studio.
- **`category_mapper`**: No changes to post-LLM category matching.
- **System prompt**: No changes — it already instructs "Pick from 'Suggested Categories'..." which still applies because the JSON schema enum acts as the suggestion constraint.

## Verification

1. **llama-server**: Run a classification job. Confirm via logs or TCP capture that the request to `/v1/chat/completions` contains the categories **only** in `response_format.json_schema.schema.properties.category.enum` — NOT in the user message text. Confirm classification still works correctly (e.g., Heineken → Beer/Cider/Lager).

2. **LM Studio**: Run a classification job. Confirm the request no longer triggers the "context length exceeded" 400 error. Confirm categories are only in the JSON schema enum. Confirm classification works correctly.

3. **Ollama**: Smoke test — confirm categories still appear in the user prompt text. Confirm no JSON schema is sent. Confirm classification works.

## Expected Impact

- **~50% reduction** in payload size for llama-server and LM Studio requests (categories sent once instead of twice).
- **Fixes LM Studio 400 error** — the massively shorter user prompt fits within default context windows.
- **Faster prompt processing** — fewer tokens to parse.
- **No behavioral change** — the JSON schema enum is the binding constraint; the free-text listing was redundant.
