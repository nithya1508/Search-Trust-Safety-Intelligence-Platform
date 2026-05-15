"""
abuse_detector.py
-----------------
XGBoost-based binary abuse classifier for Search Trust & Safety.

Features:
- Trains on behavioural signals to detect abuse vs. safe traffic
- Outputs probability scores + tiered risk labels
- SHAP explainability for each prediction
- Persists model for reuse in pipeline

Usage:
    python src/classifiers/abuse_detector.py --train
    python src/classifiers/abuse_detector.py --predict --input data/search_trust_dataset.csv
"""

import argparse
import os
import pickle
import json

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifiers.feature_engineering import engineer_features, get_feature_group_importances

MODEL_PATH = "models/abuse_detector.pkl"
SCALER_PATH = "models/scaler.pkl"
RESULTS_PATH = "models/training_results.json"

RISK_TIERS = {
    (0.0, 0.25): "LOW",
    (0.25, 0.55): "MEDIUM",
    (0.55, 0.80): "HIGH",
    (0.80, 1.01): "CRITICAL",
}


def get_risk_tier(prob: float) -> str:
    for (low, high), label in RISK_TIERS.items():
        if low <= prob < high:
            return label
    return "CRITICAL"


def train(data_path: str = "data/search_trust_dataset.csv") -> dict:
    print("🔄 Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"   {len(df)} rows | {df['is_abuse'].mean():.1%} abuse rate")

    X, scaler, feat_cols = engineer_features(df, fit_scaler=True)
    y = df["is_abuse"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Model training ────────────────────────────────────────────────────────
    # Note: use_label_encoder was removed in XGBoost 2.0; omit it entirely.
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # handle class imbalance
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )

    print("🏋️  Training XGBoost classifier...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    # ── SHAP feature importances ──────────────────────────────────────────────
    print("🔍 Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    feature_importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feat_cols
    ).sort_values(ascending=False)

    group_importance = get_feature_group_importances(feature_importance)

    results = {
        "roc_auc": round(auc, 4),
        "average_precision": round(ap, 4),
        "cv_auc_mean": round(cv_scores.mean(), 4),
        "cv_auc_std": round(cv_scores.std(), 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "top_features": feature_importance.head(10).to_dict(),
        "feature_group_importance": group_importance[["group", "total_importance"]].to_dict(orient="records"),
    }

    print(f"\n📊 Results:")
    print(f"   ROC-AUC:  {auc:.4f}")
    print(f"   Avg Prec: {ap:.4f}")
    print(f"   CV AUC:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n   Top 5 features (SHAP):")
    for feat, val in feature_importance.head(5).items():
        print(f"     {feat:35s} {val:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Model saved to {MODEL_PATH}")
    return results


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Load saved model and predict on new data."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    X, _, _ = engineer_features(df, fit_scaler=False, scaler=scaler)
    probs = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["abuse_probability"] = probs
    df["risk_tier"] = [get_risk_tier(p) for p in probs]
    df["predicted_abuse"] = (probs >= 0.5).astype(int)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--input", default="data/search_trust_dataset.csv")
    args = parser.parse_args()

    if args.train:
        train(args.input)
    elif args.predict:
        df = pd.read_csv(args.input)
        results = predict(df)
        print(results[["query_text", "abuse_probability", "risk_tier"]].head(20))
    else:
        print("Use --train or --predict")
