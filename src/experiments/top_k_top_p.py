
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

PROMPT = "Write a short story about a developer discovering an AI assistant."


def generate_response(top_k, top_p):
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_k": top_k,
            "top_p": top_p,
            "num_predict": 150
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    result = generate_response(top_k=40, top_p=1.0)

    print("\n--- GENERATED RESPONSE ---")
    print(result["response"])

    print("\n--- METRICS ---")
    print("Output tokens:", result["eval_count"])
    print("Duration (seconds):", round(
        result["total_duration"] / 1_000_000_000, 2
    ))
