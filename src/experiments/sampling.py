
import requests
import csv
import os

OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = """
Write a short story about a software developer
who discovers an AI assistant.
"""

TEMPERATURES = [0.0, 0.5, 1.0]
RUNS_PER_TEMPERATURE = 5

RESULTS_FILE = "results\sampling_results.csv"


def generate_response(temperature):

    payload = {
        "model": "llama3.2",
        "prompt": PROMPT,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()


def run_experiment():

    results = []

    for temperature in TEMPERATURES:

        for run_number in range(1, RUNS_PER_TEMPERATURE + 1):

            print("\n" + "=" * 50)
            print(f"Temperature: {temperature} | Run: {run_number}")
            print("=" * 50)

            result = generate_response(temperature)

            print(result["response"])

            output_tokens = result["eval_count"]
            duration = result["total_duration"] / 1_000_000_000

            results.append({
                "temperature": temperature,
                "run_number": run_number,
                "response": result["response"],
                "input_tokens": result["prompt_eval_count"],
                "output_tokens": output_tokens,
                "duration_seconds": round(duration, 2)
            })

            print("\nOutput tokens:", output_tokens)
            print("Duration:", round(duration, 2), "seconds")

    return results


def save_results(results):

    os.makedirs("results", exist_ok=True)

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as file:

        fieldnames = [
            "temperature",
            "run_number",
            "response",
            "input_tokens",
            "output_tokens",
            "duration_seconds"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    print("\nExperiment results saved to:", RESULTS_FILE)


if __name__ == "__main__":

    experiment_results = run_experiment()

    save_results(experiment_results)

    print("\nTotal generations:", len(experiment_results))
    print("Experiment completed!")
