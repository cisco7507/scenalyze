import base64
import json

payload = {
    'model': 'qwen3.5:4b',
    'messages': [
        {'role': 'system', 'content': '</think>\nOutput STRICT JSON: {\"brand\": \"...\", \"category\": \"...\", \"confidence\": 0.0, \"reasoning\": \"...\"}'},
        {'role': 'user', 'content': 'Categories: [\"Automotive\", \"Finance\"]\nOverride: False\nOCR Text: \"Buy a new car today!\"'}
    ],
    'stream': False,
    'format': 'json',
    'options': {
        'temperature': 0.0,
        'top_p': 1.0,
        'top_k': 40,
        'presence_penalty': 0.0,
        'repeat_penalty': 1.0,
        'num_ctx': 8192
    }
}

print(f"curl http://localhost:11434/api/chat -d '{json.dumps(payload)}'")
