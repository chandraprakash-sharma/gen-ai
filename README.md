# Week 1: LLM Sampling Lab

A hands-on lab exploring sampling parameters (`temperature`, `top_k`, and `top_p`) in Large Language Models using local Ollama inference (`llama3.2`).

## Project Structure

```text
├── results/
│   ├── sampling_results.csv        # Temperature experiment results
│   └── top_k_top_p_results.csv     # Top-k & Top-p experiment results
├── src/
│   ├── experiments/
│   │   ├── analyze_results.py      # Summary metrics and pandas analysis
│   │   ├── sampling.py             # Temperature sampling experiment
│   │   ├── top_k_top_p.py          # Single-run top-k and top-p test
│   │   └── top_k_top_p_experiment.py # Multi-configuration top-k/top-p experiment
│   └── main.py                     # Baseline Ollama API test & performance metrics
├── .gitignore
├── requirements.txt
└── README.md
```

## Prerequisites

1. **Python 3.10+**
2. **Ollama**: Installed and running locally.
3. **Model**: Pull the `llama3.2` model:
   ```bash
   ollama pull llama3.2
   ```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- **Run baseline test**:
  ```bash
  python src/main.py
  ```

- **Run sampling experiments**:
  ```bash
  python src/experiments/sampling.py
  python src/experiments/top_k_top_p_experiment.py
  ```

- **Analyze results**:
  ```bash
  python src/experiments/analyze_results.py
  ```

