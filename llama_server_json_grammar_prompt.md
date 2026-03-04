# Adding Robust JSON Grammar Support for Llama-Server

## The Goal

We are currently using the `OpenAICompatibleProvider` plugin in `video_service/core/llm.py` to route requests to a local `llama-server` instance or LM Studio.

One of the greatest strengths of `llama.cpp` (which powers `llama-server`) and LM Studio is their ability to enforce strict GBNF grammars. This structural enforcement guarantees that the model output is 100% valid JSON and physically prevents any conversational text or formatting hallucinations outside the target JSON block.

We want to trivially enable this feature for our API calls to these backends, without breaking generic OpenAI-compatible backends that might not support it.

## The Implementation

`llama-server` (and LM Studio) both possess a brilliant quality-of-life feature: they natively support the standard OpenAI `response_format` parameter.

When they intercept `"response_format": {"type": "json_object"}` in the HTTP request body, they automatically generate and apply a highly strict JSON GBNF grammar to the inference stream under the hood.

However, some smaller or older generic "OpenAI-compatible" backends might crash if they receive an unknown payload key like `response_format`.

**Your Instructions:**
Please update the `OpenAICompatibleProvider` class in `video_service/core/llm.py` to safely support this capability via an initialization toggle:

1. **Update `__init__`**: Modify the constructor of `OpenAICompatibleProvider` to accept an optional `force_json_mode: bool = False` argument and store it as `self.force_json_mode = force_json_mode`.
2. **Update `generate_json`**: Modify the `payload` dictionary construction. **Only** inject `"response_format": {"type": "json_object"}` into the JSON payload if `self.force_json_mode` is True.
3. **Update the Router**: In the `create_provider` factory function at the bottom of `video_service/core/llm.py`, when instantiating the `OpenAICompatibleProvider` for the `"lm studio"` provider (which also serves `llama-server`), aggressively pass `force_json_mode=True`.

### Constraints:

1. **Apply to JSON only**: Only inject `response_format` in the `generate_json` method. Do not apply it to `generate_text`, as the Agent mode relies on outputting freeform conversational reasoning text.
2. **Preserve Defaults**: Other generic backends using this provider class should default to `False` to prevent crashes.
