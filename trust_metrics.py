"""
trust_metrics.py
----------------
KPI framework for Search Trust & Safety.

Calculates and tracks:
- Abuse detection rate, precision, recall, F1
- Policy violation prevalence by category
- Feature distribution drift (PSI, KS-test)
- Time-series trust signal trends
- Weekly comparison summaries for leadership reporting

This mirrors the "design and implement product metrics to benchmark user trust
risks and track improvements over time" responsibility.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score
)


# ── Population Stability Index ────────────────────────────────────────────────

def population_stability_index(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    PSI measures how much a distribution has shifted.
    PSI < 0.1:  No significant change
    PSI 0.1-0.2: Moderate change, monitor
    PSI > 0.2:   Major shift, investigate
    """
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_percents = np.histogram(expected, bins=np.percentile(expected, breakpoints))[0] / len(expected)
    actual_percents = np.histogram(actual, bins=np.percentile(expected, breakpoints))[0] / len(actual)

    # Avoid log(0)
    expected_percents = np.where(expected_percents == 0, 1e-4, expected_percents)
    actual_percents = np.where(actual_percents == 0, 1e-4, actual_percents)

    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return round(float(psi), 4)


def ks_drift_test(reference: np.ndarray, current: np.ndarray) -> dict:
    """Kolmogorov-Smirnov test for distribution shift."""
    stat, p_value = stats.ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "drift_detected": p_value < 0.05,
    }


# ── Core KPI calculations ─────────────────────────────────────────────────────

def calculate_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Core classification metrics for the abuse detector."""
    return {
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "average_precision": round(average_precision_score(y_true, y_prob), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "false_positive_rate": round(
            ((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1), 4
        ),
        "false_negative_rate": round(
            ((y_pred == 0) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1), 4
        ),
        "abuse_prevalence": round(float(y_true.mean()), 4),
        "flagged_rate": round(float(y_pred.mean()), 4),
    }


def calculate_policy_metrics(df: pd.DataFrame, label_col: str = "policy_label") -> dict:
    """Policy violation prevalence and breakdown."""
    total = len(df)
    counts = df[label_col].value_counts().to_dict()

    return {
        "total_queries": total,
        "violation_rate": round(1 - counts.get("safe", 0) / total, 4),
        "by_category": {
            label: {
                "count": count,
                "prevalence": round(count / total, 4),
            }
            for label, count in counts.items()
        },
    }


def calculate_risk_tier_distribution(df: pd.DataFrame) -> dict:
    """Distribution of queries across risk tiers."""
    if "risk_tier" not in df.columns:
        return {}
    counts = df["risk_tier"].value_counts().to_dict()
    total = len(df)
    return {tier: {"count": counts.get(tier, 0), "pct": round(counts.get(tier, 0) / total, 4)}
            for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]}


# ── Drift detection ───────────────────────────────────────────────────────────

DRIFT_FEATURES = [
    "query_velocity_per_hour", "ctr", "session_duration_sec",
    "ip_subnet_concentration", "account_age_days", "prior_abuse_flags",
]


def detect_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """
    Compare feature distributions between a reference window and current window.
    Returns PSI + KS results per feature.
    """
    results = {}
    for feature in DRIFT_FEATURES:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue

        ref = reference_df[feature].dropna().values
        cur = current_df[feature].dropna().values

        if len(ref) < 10 or len(cur) < 10:
            continue

        psi = population_stability_index(ref, cur)
        ks = ks_drift_test(ref, cur)

        results[feature] = {
            "psi": psi,
            "psi_status": "stable" if psi < 0.1 else ("monitor" if psi < 0.2 else "alert"),
            **ks,
        }

    # Summary
    alert_features = [f for f, v in results.items() if v["psi_status"] == "alert"]
    results["_summary"] = {
        "features_checked": len(results) - 1,
        "alert_count": len(alert_features),
        "alert_features": alert_features,
        "drift_detected": len(alert_features) > 0,
    }

    return results


# ── Time-series trending ──────────────────────────────────────────────────────

def compute_daily_metrics(df: pd.DataFrame, n_days: int = 14) -> pd.DataFrame:
    """Compute daily abuse rate and violation counts for time-series dashboard."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    cutoff = df["date"].max() - timedelta(days=n_days)
    df = df[df["date"] >= cutoff]

    daily = (
        df.groupby("date")
        .agg(
            total_queries=("query_id", "count"),
            abuse_count=("is_abuse", "sum"),
            spam_count=("abuse_type", lambda x: (x == "spam").sum()),
            harmful_count=("abuse_type", lambda x: (x == "harmful").sum()),
            misleading_count=("abuse_type", lambda x: (x == "misleading").sum()),
        )
        .reset_index()
    )
    daily["abuse_rate"] = (daily["abuse_count"] / daily["total_queries"]).round(4)
    return daily


def generate_weekly_summary(df: pd.DataFrame) -> dict:
    """
    Compare this week vs last week — suitable for leadership reporting.
    Handles the edge case where there are zero abuse rows in a week.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    max_date = df["date"].max()

    this_week = df[df["date"] > (max_date - timedelta(days=7))]
    last_week = df[(df["date"] <= (max_date - timedelta(days=7))) &
                   (df["date"] > (max_date - timedelta(days=14)))]

    def abuse_rate(d): return d["is_abuse"].mean() if len(d) else 0

    rate_change = abuse_rate(this_week) - abuse_rate(last_week)
    direction = "↑ Increased" if rate_change > 0.01 else ("↓ Decreased" if rate_change < -0.01 else "→ Stable")

    # Guard against empty abuse subset (mode() returns empty Series → IndexError)
    abuse_rows = this_week[this_week["is_abuse"] == 1]
    top_violation = (
        abuse_rows["policy_label"].mode().iloc[0]
        if len(abuse_rows) > 0 else "none"
    )

    return {
        "this_week": {
            "total_queries": len(this_week),
            "abuse_rate": round(abuse_rate(this_week), 4),
            "top_violation": top_violation,
        },
        "last_week": {
            "total_queries": len(last_week),
            "abuse_rate": round(abuse_rate(last_week), 4),
        },
        "week_over_week_change": round(rate_change, 4),
        "trend_direction": direction,
        "requires_escalation": rate_change > 0.05,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def generate_full_report(df: pd.DataFrame, output_path: str = "models/trust_metrics_report.json") -> dict:
    """
    End-to-end trust metrics report.
    """
    policy_metrics = calculate_policy_metrics(df)
    weekly = generate_weekly_summary(df)
    daily = compute_daily_metrics(df)

    # Drift: first half vs second half as proxy
    mid = len(df) // 2
    drift = detect_feature_drift(df.iloc[:mid], df.iloc[mid:])

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "policy_metrics": policy_metrics,
        "weekly_summary": weekly,
        "feature_drift": drift,
        "daily_trend_sample": daily.tail(7).to_dict(orient="records"),
    }

    os.makedirs("models", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"✅ Trust metrics report saved to {output_path}")
    return report
