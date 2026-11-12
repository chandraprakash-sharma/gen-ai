
import requests
import csv
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

PROMPT = """
Write a short story about a software developer
who discovers an AI assistant.
"""

TEMPERATURE = 0.7
RUNS_PER_CONFIG = 5

CONFIGURATIONS = [
    {"name": "A", "top_k": 40, "top_p": 0.3},
    {"name": "B", "top_k": 40, "top_p": 0.5},
    {"name": "C", "top_k": 40, "top_p": 0.9},
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "top_k_top_p_results.csv"


def generate_response(top_k, top_p):
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
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


def run_experiment():
    results = []

    for config in CONFIGURATIONS:
        for run_number in range(1, RUNS_PER_CONFIG + 1):

            print("\n" + "=" * 50)
            print(
                f"Config: {config['name']} | "
                f"Top-K: {config['top_k']} | "
                f"Top-P: {config['top_p']} | "
                f"Run: {run_number}"
            )
            print("=" * 50)

            result = generate_response(
                config["top_k"],
                config["top_p"]
            )

            results.append({
                "configuration": config["name"],
                "top_k": config["top_k"],
                "top_p": config["top_p"],
                "temperature": TEMPERATURE,
                "run_number": run_number,
                "response": result["response"],
                "input_tokens": result["prompt_eval_count"],
                "output_tokens": result["eval_count"],
                "duration_seconds": round(
                    result["total_duration"] / 1_000_000_000, 2
                )
            })

            print(result["response"])
            print("Output tokens:", result["eval_count"])

    return results


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with RESULTS_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to:", RESULTS_FILE)


if __name__ == "__main__":
    experiment_results = run_experiment()
    save_results(experiment_results)

    print("\nTotal generations:", len(experiment_results))
    print("Experiment completed!")
