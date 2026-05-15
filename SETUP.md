# Setup Guide

## Prerequisites
- Python 3.9+
- pip

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/search-trust-safety.git
cd search-trust-safety

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Pipeline

### Without LLM (no API key needed)
```bash
python run_pipeline.py --skip-llm
```

### With LLM policy evaluation
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key

export ANTHROPIC_API_KEY=your_key_here
python run_pipeline.py
```

### Launch Dashboard
```bash
streamlit run dashboard/app.py
```

### Run Tests
```bash
pytest tests/ -v
```

> **Note:** A `conftest.py` and `pytest.ini` at the project root handle `PYTHONPATH`
> setup automatically, so you do not need to set `PYTHONPATH` manually.

## Output Files

After running the pipeline, the following files are written to `models/`:

| File | Description |
|------|-------------|
| `abuse_detector.pkl` | Trained XGBoost model |
| `scaler.pkl` | Feature scaler (required for prediction) |
| `training_results.json` | ROC-AUC, SHAP importances, CV scores |
| `trust_metrics_report.json` | KPI report + drift analysis |
| `alerts.json` | Active alerts from the alert engine |
| `llm_eval_results.json` | LLM classifier accuracy (if Step 6 ran) |

> All `models/*.json` and `models/*.pkl` files are git-ignored. New contributors
> must run `python run_pipeline.py` to regenerate them locally. The dashboard
> gracefully handles missing files and shows prompts to generate them.

## Project Structure
```
search-trust-safety/
├── conftest.py         # Pytest path setup (run tests without PYTHONPATH)
├── pytest.ini          # Pytest config
├── data/               # Dataset generation
├── src/
│   ├── classifiers/    # ML abuse detector (XGBoost + SHAP)
│   ├── llm_eval/       # LLM policy classifier (Claude API)
│   └── metrics/        # KPI framework + alert engine
├── dashboard/          # Streamlit dashboard
├── models/             # Saved model artifacts (git-ignored)
├── tests/              # Pytest suite
└── run_pipeline.py     # End-to-end runner
```
