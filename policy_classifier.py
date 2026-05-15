"""
policy_classifier.py
--------------------
LLM-based content policy classifier using Claude (Anthropic API).

Classifies search queries against 6 policy categories:
  safe | spam | harmful | misleading | adult | violence

Features:
- Single query classification with chain-of-thought
- Batch classification for dataset evaluation
- LLM-based trend analysis for drift reports
- Evaluates accuracy against ground-truth labels

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python src/llm_eval/policy_classifier.py --evaluate --limit 50
    python src/llm_eval/policy_classifier.py --query "how to hack a website"
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Optional

import anthropic
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_eval.prompt_templates import (
    SYSTEM_PROMPT,
    format_classification_prompt,
    format_batch_prompt,
    format_drift_prompt,
)

MODEL = "claude-sonnet-4-20250514"
RESULTS_PATH = "models/llm_eval_results.json"


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Export it before running:\n"
            "  export ANTHROPIC_API_KEY=your_key_here"
        )
    return anthropic.Anthropic(api_key=api_key)


def _strip_code_fences(text: str) -> str:
    """Safely strip markdown code fences from LLM output without crashing on malformed responses."""
    # Remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def classify_query(query: str, client: Optional[anthropic.Anthropic] = None) -> dict:
    """
    Classify a single search query using Claude.

    Returns:
        dict with keys: policy_label, confidence, reasoning,
                        policy_cited, risk_signals, requires_human_review
    """
    if client is None:
        client = get_client()

    prompt = format_classification_prompt(query)

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = _strip_code_fences(response.content[0].text)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "policy_label": "safe",
            "confidence": 0.0,
            "reasoning": "Parse error",
            "policy_cited": "None",
            "risk_signals": [],
            "requires_human_review": True,
            "parse_error": raw[:200],
        }

    result["query"] = query
    return result


def evaluate_dataset(
    data_path: str = "data/search_trust_dataset.csv",
    limit: int = 100,
    output_path: str = RESULTS_PATH,
) -> dict:
    """
    Evaluate LLM classifier on a sample of the dataset.
    Compares LLM predictions to ground-truth policy_label.
    """
    client = get_client()
    df = pd.read_csv(data_path)

    # Stratified sample
    sample = (
        df.groupby("policy_label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), max(1, limit // 6)), random_state=42))
        .reset_index(drop=True)
    )
    sample = sample.head(limit)

    print(f"🤖 Evaluating {len(sample)} queries with Claude ({MODEL})...")

    results = []
    correct = 0

    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        try:
            pred = classify_query(row["query_text"], client)
            pred["ground_truth"] = row["policy_label"]
            pred["correct"] = int(pred["policy_label"] == row["policy_label"])
            correct += pred["correct"]
            results.append(pred)
            time.sleep(0.3)  # rate limit courtesy
        except Exception as e:
            results.append({
                "query": row["query_text"],
                "ground_truth": row["policy_label"],
                "policy_label": "error",
                "confidence": 0.0,
                "correct": 0,
                "error": str(e),
            })

    accuracy = correct / len(results) if results else 0.0
    results_df = pd.DataFrame(results)

    # Per-class accuracy
    per_class = {}
    for label in results_df["ground_truth"].unique():
        subset = results_df[results_df["ground_truth"] == label]
        per_class[label] = round(subset["correct"].mean(), 3)

    summary = {
        "total_evaluated": len(results),
        "overall_accuracy": round(accuracy, 4),
        "per_class_accuracy": per_class,
        "requires_human_review_rate": round(results_df.get("requires_human_review", pd.Series([False])).mean(), 3),
        "avg_confidence": round(results_df["confidence"].mean(), 3) if "confidence" in results_df else None,
        "sample_results": results[:5],
    }

    os.makedirs("models", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n📊 LLM Evaluation Results:")
    print(f"   Overall Accuracy: {accuracy:.1%}")
    print(f"   Per-class accuracy:")
    for label, acc in per_class.items():
        print(f"     {label:12s}: {acc:.1%}")
    print(f"\n✅ Results saved to {output_path}")
    return summary


def analyse_trends(df: pd.DataFrame, client: Optional[anthropic.Anthropic] = None) -> dict:
    """
    Use LLM to analyse abuse trend data and generate leadership summary.
    """
    if client is None:
        client = get_client()

    # Build trend data string
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily_violations = (
        df[df["is_abuse"] == 1]
        .groupby(["date", "policy_label"])
        .size()
        .unstack(fill_value=0)
        .tail(7)
        .to_string()
    )

    top_patterns = (
        df[df["is_abuse"] == 1]["query_text"]
        .value_counts()
        .head(5)
        .to_string()
    )

    prompt = format_drift_prompt(daily_violations, top_patterns)

    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = _strip_code_fences(response.content[0].text)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Parse failed", "raw": raw[:300]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Policy Classifier")
    parser.add_argument("--evaluate", action="store_true", help="Run dataset evaluation")
    parser.add_argument("--query", type=str, help="Classify a single query")
    parser.add_argument("--limit", type=int, default=60, help="Number of samples to evaluate")
    parser.add_argument("--input", default="data/search_trust_dataset.csv")
    args = parser.parse_args()

    if args.query:
        result = classify_query(args.query)
        print(json.dumps(result, indent=2))
    elif args.evaluate:
        evaluate_dataset(data_path=args.input, limit=args.limit)
    else:
        print("Use --evaluate or --query 'your query here'")
