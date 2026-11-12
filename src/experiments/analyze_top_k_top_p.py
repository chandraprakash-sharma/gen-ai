
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = PROJECT_ROOT / "results" / "top_k_top_p_results.csv"

df = pd.read_csv(CSV_FILE)

print("\n--- FIRST FIVE RESULTS ---")
print(df.head().to_string(index=False))

summary = df.groupby(
    ["configuration", "top_k", "top_p"]
).agg(
    total_runs=("run_number", "count"),
    unique_responses=("response", "nunique"),
    average_output_tokens=("output_tokens", "mean"),
    average_duration=("duration_seconds", "mean")
).reset_index()

print("\n--- TOP-K AND TOP-P EXPERIMENT SUMMARY ---")
print(summary.round(2).to_string(index=False))
