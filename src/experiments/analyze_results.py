from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = PROJECT_ROOT / "results" / "sampling_results.csv"

print("Reading file:", CSV_FILE)

df = pd.read_csv(CSV_FILE)

# Display first five records
print("\n--- FIRST FIVE RESULTS ---")
print(df.head())

# Calculate summary metrics
summary = df.groupby("temperature").agg(
    total_runs=("run_number", "count"),
    unique_responses=("response", "nunique"),
    average_output_tokens=("output_tokens", "mean"),
    average_duration=("duration_seconds", "mean")
).reset_index()

print("\n--- EXPERIMENT SUMMARY ---")
print(summary.round(2).to_string(index=False))
