"""
feature_engineering.py
-----------------------
Transforms raw search trust dataset into model-ready features.
Includes:
- Behavioural signal normalisation
- Risk flag creation
- Feature importance grouping for SHAP analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = [
    "query_velocity_per_hour",
    "ctr",
    "session_duration_sec",
    "ip_subnet_concentration",
    "account_age_days",
    "unique_devices",
    "bounce_rate",
    "prior_abuse_flags",
    "query_contains_url",
    "query_length_tokens",
    "query_hour",
    "exact_query_repeats",
]

# Feature groups for interpretability reporting.
# Includes both raw numeric features and derived binary flags.
FEATURE_GROUPS = {
    "velocity_signals": ["query_velocity_per_hour", "exact_query_repeats", "is_high_velocity", "high_repeat_queries"],
    "engagement_signals": ["ctr", "session_duration_sec", "bounce_rate", "abnormal_ctr"],
    "identity_signals": ["account_age_days", "unique_devices", "ip_subnet_concentration",
                         "is_new_account", "high_ip_concentration"],
    "content_signals": ["query_contains_url", "query_length_tokens"],
    "risk_signals": ["prior_abuse_flags", "has_prior_flags", "rule_based_risk_score"],
    "temporal_signals": ["query_hour", "is_night_query"],
}


def engineer_features(df: pd.DataFrame, fit_scaler: bool = True, scaler: StandardScaler = None) -> tuple:
    """
    Full feature engineering pipeline.

    Returns:
        X (pd.DataFrame): Engineered feature matrix
        scaler (StandardScaler): Fitted scaler (for reuse on test data)
        feature_cols (list): Column names in X
    """
    df = df.copy()

    # ── Derived risk flags ────────────────────────────────────────────────────
    df["is_night_query"] = ((df["query_hour"] >= 0) & (df["query_hour"] <= 5)).astype(int)
    df["is_high_velocity"] = (df["query_velocity_per_hour"] > 50).astype(int)
    df["is_new_account"] = (df["account_age_days"] < 7).astype(int)
    df["has_prior_flags"] = (df["prior_abuse_flags"] > 0).astype(int)
    df["abnormal_ctr"] = ((df["ctr"] < 0.01) | (df["ctr"] > 0.9)).astype(int)
    df["high_ip_concentration"] = (df["ip_subnet_concentration"] > 0.6).astype(int)
    df["high_repeat_queries"] = (df["exact_query_repeats"] > 5).astype(int)

    # ── Composite risk score (rule-based, pre-ML signal) ─────────────────────
    df["rule_based_risk_score"] = (
        df["is_high_velocity"] * 3
        + df["has_prior_flags"] * 2
        + df["is_new_account"] * 1.5
        + df["high_ip_concentration"] * 2
        + df["abnormal_ctr"] * 1.5
        + df["high_repeat_queries"] * 2
        + df["query_contains_url"] * 1
        + df["is_night_query"] * 0.5
    )

    feature_cols = NUMERIC_FEATURES + [
        "is_night_query", "is_high_velocity", "is_new_account",
        "has_prior_flags", "abnormal_ctr", "high_ip_concentration",
        "high_repeat_queries", "rule_based_risk_score",
    ]

    X = df[feature_cols].copy()

    # ── Scale continuous features ─────────────────────────────────────────────
    continuous = [
        "query_velocity_per_hour", "session_duration_sec", "account_age_days",
        "query_length_tokens", "exact_query_repeats", "rule_based_risk_score"
    ]

    if fit_scaler:
        scaler = StandardScaler()
        X[continuous] = scaler.fit_transform(X[continuous])
    else:
        X[continuous] = scaler.transform(X[continuous])

    return X, scaler, feature_cols


def get_feature_group_importances(importances: pd.Series) -> pd.DataFrame:
    """Aggregate feature importances by group for reporting."""
    rows = []
    for group, feats in FEATURE_GROUPS.items():
        available = [f for f in feats if f in importances.index]
        if available:
            rows.append({
                "group": group,
                "total_importance": importances[available].sum(),
                "features": available,
            })
    return pd.DataFrame(rows).sort_values("total_importance", ascending=False)
