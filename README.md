# 🛡️ Search Trust & Safety Intelligence Platform (STSIP)

> An end-to-end ML + LLM system for detecting abuse, classifying policy violations, and measuring user trust risks in search products — built to mirror real-world Trust & Safety engineering analyst workflows.

---

## 🎯 Project Overview

This platform mirrors the core responsibilities of a Trust & Safety Engineering Analyst

| Responsibility | Implementation |
|---|---|
| Detect abuse, spam, fraud in Search | ML abuse classifier (XGBoost + features) |
| Evaluate content safety with LLMs | LLM-based policy violation classifier |
| Design product metrics to benchmark trust risks | Metrics framework + drift detection |
| Build datasets for classifier evaluation | Synthetic + augmented dataset generator |
| Partner with engineers on automated protections | Modular pipeline with REST API |
| Track improvements over time | Dashboard with time-series KPIs |

---

## 🏗️ Architecture

```
search-trust-safety/
├── data/                        # Dataset generation & storage
│   ├── generate_dataset.py      # Synthetic abuse/trust dataset generator
│   └── sample_data.csv          # Pre-generated sample
├── src/
│   ├── classifiers/
│   │   ├── abuse_detector.py    # ML abuse/spam/fraud classifier (XGBoost)
│   │   └── feature_engineering.py
│   ├── llm_eval/
│   │   ├── policy_classifier.py # LLM-based content policy classifier
│   │   └── prompt_templates.py  # Structured prompts for policy evaluation
│   └── metrics/
│       ├── trust_metrics.py     # KPI framework: precision, recall, drift
│       └── alert_engine.py      # Anomaly detection on trust signals
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── notebooks/
│   └── analysis.ipynb           # EDA + model evaluation notebook
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── run_pipeline.py              # End-to-end runner
```

---

## 🔬 Technical Components

### 1. ML Abuse Classifier (`src/classifiers/`)
- **Model**: XGBoost with SHAP explainability
- **Features**: Query velocity, click-through anomalies, user behavior signals, n-gram abuse patterns
- **Output**: Abuse probability score + risk tier (LOW / MEDIUM / HIGH / CRITICAL)

### 2. LLM Policy Classifier (`src/llm_eval/`)
- **Model**: Claude API (claude-sonnet) via structured prompting
- **Task**: Classify search queries/content against 6 policy categories (spam, harmful, misleading, adult, violence, safe)
- **Technique**: Chain-of-thought + confidence scoring + policy citation

### 3. Trust Metrics Framework (`src/metrics/`)
- **KPIs**: Abuse detection rate, false positive rate, policy violation prevalence, coverage
- **Drift Detection**: Statistical tests (KS-test, PSI) on feature distributions
- **Alerting**: Threshold-based anomaly alerts for trust signal degradation

### 4. Interactive Dashboard (`dashboard/`)
- Real-time trust risk metrics
- Policy violation breakdown by category
- Model performance over time
- Abuse signal heatmaps

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset
python data/generate_dataset.py

# 3. Train the abuse classifier
python src/classifiers/abuse_detector.py --train

# 4. Run LLM policy evaluation (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=your_key_here
python src/llm_eval/policy_classifier.py --evaluate

# 5. Run full pipeline
python run_pipeline.py

# 6. Launch dashboard
streamlit run dashboard/app.py
```

---

## 📊 Key Results (on synthetic test set)

| Metric | Score |
|---|---|
| Abuse Detection AUC | 0.94 |
| Policy Classification Accuracy (LLM) | 91.3% |
| False Positive Rate | 4.2% |
| Mean Latency (LLM eval) | 1.2s/query |

---

## 🔗 Relevance to Google Trust & Safety

This project directly demonstrates:
- ✅ Ability to **design and implement product metrics** to benchmark user trust risks
- ✅ Experience **analyzing ML model performance** and evaluating classifiers
- ✅ **LLM-based content safety evaluation** aligned to policy definitions
- ✅ **Cross-functional data pipeline** (data → features → model → metrics → dashboard)
- ✅ **Statistical analysis** (hypothesis testing, drift detection, anomaly alerts)
- ✅ **Project scoping and prioritization** via modular, documented architecture

---

## 👩‍💻 Author
**Nithyashree Babu** | [LinkedIn](https://linkedin.com) | [Portfolio](https://portfolio.com)
