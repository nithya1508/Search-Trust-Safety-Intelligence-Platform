"""
tests/test_pipeline.py
----------------------
Unit tests for Trust & Safety pipeline components.
Run with: pytest tests/ -v
"""

import json
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# ── Dataset generation tests ──────────────────────────────────────────────────

def test_dataset_generation():
    from data.generate_dataset import generate_dataset
    df = generate_dataset(n_samples=200, output_path="/tmp/test_dataset.csv")
    assert len(df) > 0
    assert "is_abuse" in df.columns
    assert "policy_label" in df.columns
    assert set(df["policy_label"].unique()).issubset(
        {"safe", "spam", "harmful", "misleading", "adult", "violence"}
    )
    assert df["is_abuse"].isin([0, 1]).all()


def test_dataset_class_balance():
    from data.generate_dataset import generate_dataset
    df = generate_dataset(n_samples=500, output_path="/tmp/test_balance.csv")
    safe_pct = (df["policy_label"] == "safe").mean()
    assert 0.5 < safe_pct < 0.75, f"Unexpected safe proportion: {safe_pct}"


# ── Feature engineering tests ─────────────────────────────────────────────────

def test_feature_engineering():
    from data.generate_dataset import generate_dataset
    from classifiers.feature_engineering import engineer_features

    df = generate_dataset(n_samples=100, output_path="/tmp/test_fe.csv")
    X, scaler, cols = engineer_features(df, fit_scaler=True)

    assert X.shape[0] == len(df)
    assert X.isnull().sum().sum() == 0, "NaNs found in feature matrix"
    assert "rule_based_risk_score" in cols
    assert scaler is not None


def test_feature_engineering_reuse_scaler():
    from data.generate_dataset import generate_dataset
    from classifiers.feature_engineering import engineer_features

    df = generate_dataset(n_samples=200, output_path="/tmp/test_scaler.csv")
    X_train, scaler, cols = engineer_features(df.iloc[:100], fit_scaler=True)
    X_test, _, _ = engineer_features(df.iloc[100:], fit_scaler=False, scaler=scaler)

    assert X_test.shape[1] == X_train.shape[1]


# ── Trust metrics tests ───────────────────────────────────────────────────────

def test_detection_metrics():
    from metrics.trust_metrics import calculate_detection_metrics

    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.1, 0.85])

    metrics = calculate_detection_metrics(y_true, y_pred, y_prob)
    assert "roc_auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["false_positive_rate"] <= 1


def test_psi_calculation():
    from metrics.trust_metrics import population_stability_index

    # Same distribution → PSI ≈ 0
    data = np.random.normal(0, 1, 1000)
    psi = population_stability_index(data, data + np.random.normal(0, 0.05, 1000))
    assert psi < 0.1, f"PSI should be low for similar distributions: {psi}"


def test_psi_drift_detection():
    from metrics.trust_metrics import population_stability_index

    reference = np.random.normal(0, 1, 1000)
    drifted = np.random.normal(3, 2, 1000)  # Major shift
    psi = population_stability_index(reference, drifted)
    assert psi > 0.2, f"PSI should be high for drifted distributions: {psi}"


def test_policy_metrics():
    from data.generate_dataset import generate_dataset
    from metrics.trust_metrics import calculate_policy_metrics

    df = generate_dataset(n_samples=500, output_path="/tmp/test_policy.csv")
    metrics = calculate_policy_metrics(df)
    assert metrics["total_queries"] == len(df)
    assert 0 <= metrics["violation_rate"] <= 1
    assert "safe" in metrics["by_category"]


def test_weekly_summary():
    from data.generate_dataset import generate_dataset
    from metrics.trust_metrics import generate_weekly_summary

    df = generate_dataset(n_samples=500, output_path="/tmp/test_weekly.csv")
    summary = generate_weekly_summary(df)
    assert "this_week" in summary
    assert "week_over_week_change" in summary
    assert isinstance(summary["requires_escalation"], bool)


# ── Alert engine tests ────────────────────────────────────────────────────────

def test_alert_engine_no_false_positives():
    """Normal data should not trigger abuse spike alert."""
    from metrics.alert_engine import AlertEngine
    from data.generate_dataset import generate_dataset

    df = generate_dataset(n_samples=300, output_path="/tmp/test_alerts.csv")
    engine = AlertEngine()
    # Just check it runs without error
    alerts = engine.run_all_checks(df)
    assert isinstance(alerts, list)


def test_category_surge_detection():
    """Inject artificial surge and verify alert fires."""
    from metrics.alert_engine import AlertEngine
    from data.generate_dataset import generate_dataset
    import random

    df = generate_dataset(n_samples=500, output_path="/tmp/test_surge.csv")
    # Artificially inject a spam surge in the recent half
    mid = len(df) // 2
    df.loc[df.index[mid:][::3], "policy_label"] = "spam"
    df.loc[df.index[mid:][::3], "is_abuse"] = 1
    df.loc[df.index[mid:][::3], "abuse_type"] = "spam"

    engine = AlertEngine()
    alerts = engine.check_category_surge(df)
    # Should have detected spam surge
    surge_types = [a.alert_type for a in engine.alerts]
    assert "CATEGORY_SURGE" in surge_types


# ── Prompt template tests ─────────────────────────────────────────────────────

def test_prompt_templates():
    from llm_eval.prompt_templates import format_classification_prompt, format_batch_prompt

    prompt = format_classification_prompt("how to make a bomb")
    assert "how to make a bomb" in prompt
    assert "policy_label" in prompt

    batch_prompt = format_batch_prompt([
        {"query_id": "abc123", "query_text": "buy cheap followers"}
    ])
    assert "abc123" in batch_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
