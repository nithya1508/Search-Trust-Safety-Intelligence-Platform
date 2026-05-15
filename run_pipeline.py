"""
run_pipeline.py
---------------
End-to-end Trust & Safety pipeline runner.

Steps:
1. Generate synthetic dataset (if not exists)
2. Train ML abuse classifier
3. Run predictions + risk scoring
4. Compute trust metrics + drift analysis
5. Run alert engine
6. (Optional) LLM policy evaluation if API key set

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-llm    # skip LLM step (no API key needed)
    python run_pipeline.py --samples 2000
"""

import argparse
import os
import sys
import json

# ── Path setup (done once at module level) ────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

print("🛡️  Search Trust & Safety Intelligence Platform")
print("=" * 55)

# ── Step 1: Dataset ───────────────────────────────────────────────────────────
def step_generate_data(n_samples: int = 5000):
    print("\n📦 Step 1: Generating dataset...")
    from data.generate_dataset import generate_dataset
    df = generate_dataset(n_samples=n_samples)
    return df


# ── Step 2: Train ML model ────────────────────────────────────────────────────
def step_train_model():
    print("\n🏋️  Step 2: Training ML abuse classifier...")
    from classifiers.abuse_detector import train
    results = train("data/search_trust_dataset.csv")
    print(f"   ✅ AUC: {results['roc_auc']} | Avg Precision: {results['average_precision']}")
    return results


# ── Step 3: Predict on full dataset ──────────────────────────────────────────
def step_predict(df):
    print("\n🔮 Step 3: Scoring full dataset for risk tiers...")
    from classifiers.abuse_detector import predict
    try:
        scored_df = predict(df)
        print(f"   ✅ Risk tier distribution:")
        print(scored_df["risk_tier"].value_counts().to_string())
        scored_df.to_csv("data/scored_dataset.csv", index=False)
        return scored_df
    except FileNotFoundError:
        print("   ⚠️  Model not found, skipping prediction step")
        return df


# ── Step 4: Trust metrics ─────────────────────────────────────────────────────
def step_metrics(df):
    print("\n📊 Step 4: Computing trust metrics & drift analysis...")
    from metrics.trust_metrics import generate_full_report
    report = generate_full_report(df)
    weekly = report["weekly_summary"]
    print(f"   Abuse rate this week: {weekly['this_week']['abuse_rate']:.1%}")
    print(f"   Week-over-week: {weekly['trend_direction']}")
    return report


# ── Step 5: Alert engine ──────────────────────────────────────────────────────
def step_alerts(df, drift_report):
    print("\n🚨 Step 5: Running alert engine...")
    from metrics.alert_engine import AlertEngine
    engine = AlertEngine()
    alerts = engine.run_all_checks(df, drift_report=drift_report.get("feature_drift"))
    engine.export_alerts()
    return alerts


# ── Step 6: LLM evaluation ────────────────────────────────────────────────────
def step_llm_eval(limit: int = 30):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n⏭️  Step 6: Skipping LLM evaluation (ANTHROPIC_API_KEY not set)")
        print("   To enable: export ANTHROPIC_API_KEY=your_key && python run_pipeline.py")
        return None

    print(f"\n🤖 Step 6: LLM policy evaluation ({limit} samples)...")
    from llm_eval.policy_classifier import evaluate_dataset
    results = evaluate_dataset(limit=limit)
    print(f"   ✅ LLM accuracy: {results['overall_accuracy']:.1%}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM evaluation step")
    parser.add_argument("--samples", type=int, default=5000, help="Dataset size")
    parser.add_argument("--llm-limit", type=int, default=30, help="LLM eval sample count")
    args = parser.parse_args()

    df = step_generate_data(n_samples=args.samples)
    model_results = step_train_model()
    scored_df = step_predict(df)
    metrics_report = step_metrics(scored_df)
    alerts = step_alerts(scored_df, metrics_report)

    if not args.skip_llm:
        step_llm_eval(limit=args.llm_limit)

    print("\n" + "=" * 55)
    print("✅ Pipeline complete!")
    print("\n📁 Output files:")
    output_files = [
        "data/search_trust_dataset.csv",
        "data/scored_dataset.csv",
        "models/abuse_detector.pkl",
        "models/training_results.json",
        "models/trust_metrics_report.json",
        "models/alerts.json",
    ]
    for f in output_files:
        exists = "✅" if os.path.exists(f) else "⚠️ "
        print(f"   {exists} {f}")

    print("\n🖥️  Launch dashboard:")
    print("   streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
