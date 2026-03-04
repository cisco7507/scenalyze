import requests
import time

url_chat = "http://localhost:1234/v1/chat/completions"
url_completion = "http://localhost:1234/v1/completions"

print("--- TEST 1: Standard Chat API ---")
payload_chat = {
    "model": "qwen3.5-4b-mlx",
    "messages": [
        {"role": "system", "content": "Output STRICT JSON."},
        {"role": "user", "content": "Categories: [Automotive, Finance]. OCR: Buy a new car today!"}
    ],
    "temperature": 0.0
}
t0 = time.time()
try:
    res = requests.post(url_chat, json=payload_chat).json()
    print(f"Time: {time.time()-t0:.2f}s")
    print(res["choices"][0]["message"]["content"][:200])
except Exception as e:
    print(e)

print("\n--- TEST 2: Assistant Prefill in Chat API ---")
payload_chat["messages"].append({"role": "assistant", "content": "</think>\n{"})
t0 = time.time()
try:
    res = requests.post(url_chat, json=payload_chat).json()
    print(f"Time: {time.time()-t0:.2f}s")
    print(res["choices"][0]["message"]["content"][:200])
except Exception as e:
    print(e)
    if "error" in res: print(res)
