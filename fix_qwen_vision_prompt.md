# Fixing Qwen-VL Blindness in Raw Mode

## The Issue

In `video_service/core/llm.py`, the `OllamaQwenProvider.generate_json` method currently uses the `/api/generate` endpoint with `"raw": True` and manually constructs the ChatML prompt template.

**The hallucination you are seeing ("Desjardins / Financial Services") is caused by visual blindness.**

Because `presence_penalty` is correctly set to `0.0`, sampling randomness is off the table. The issue is that while the image base64 arrays are passed in `payload["images"]`, Qwen-VL models (including non-thinking models like `qwen3-vl:8b-instruct`) natively **require explicit ChatML vision spacer tokens embedded inside the literal text prompt** to bind the image payload to the text encoder.

Without `<|vision_start|><|image_pad|><|vision_end|>` manually injected into the `raw_prompt`, the model literally cannot see the frames. Because it is completely blind, it reads the noisy OCR text (`Fièrement d'ici depuis 1922`), correlates it with the example given in the System Prompt (`"if OCR says... 'Économisez avec Desjardins'"`), and falsely guesses "Desjardins".

## The Instructions

Please modify `video_service/core/llm.py` specifically targeting the `OllamaQwenProvider.generate_json` method.

You must update the construction of `raw_prompt` to dynamically inject the vision tags _before_ the `user_prompt` string **only if** the `images` argument is populated.

Your `raw_prompt` constructor should look exactly like this:

```python
        vision_tags = ""
        if images:
            # Qwen-VL requires one <|image_pad|> token per included image.
            # We reconstruct the vision spacer explicitly so `raw: True` inference can bind the payload.
            tags = "".join(["<|image_pad|>" for _ in images])
            vision_tags = f"<|vision_start|>{tags}<|vision_end|>\n"

        raw_prompt = (
            f"<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{vision_tags}{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<think>\n"
            f"</think>\n"
            f"{json_prefix}"
        )
```

### Constraints:

1. **Maintain all sampling parameters** (`presence_penalty: 0.0`, `temperature: 0.0`, etc.). The issue is strictly optical blindness, not the sampling config.
2. **Do not modify `OllamaQwenProvider.generate_text`**. The agent mode uses the `/api/chat` wrapper which automatically handles vision padding behind the scenes.
3. Keep the JSON prefix injection `{\n  "brand": "` exactly as it is. Even for models without built-in thinking (like `qwen3-vl:8b-instruct`), the hanging `<think>\n</think>\n` block guarantees the suppression of any unvanted verbal reasoning before the JSON block begins.
