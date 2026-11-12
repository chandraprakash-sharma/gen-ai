import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "llama3.2",
    "prompt": "Explain what a Large Language Model is in simple words.",
    "stream": False
}

response = requests.post(url, json=payload)
response.raise_for_status()

result = response.json()

print("\n--- LLM RESPONSE ---")
print(result["response"])

print("\n--- GENERATION METRICS ---")
print("Model:", result["model"])
print("Input tokens:", result["prompt_eval_count"])
print("Output tokens:", result["eval_count"])
print("Total duration (seconds):", result["total_duration"] / 1_000_000_000)
print("Generation speed (tokens/sec):", round(
    result["eval_count"] / (result["eval_duration"] / 1_000_000_000), 2
))
