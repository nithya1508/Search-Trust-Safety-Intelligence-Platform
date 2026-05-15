"""
generate_dataset.py
-------------------
Generates a synthetic Search Trust & Safety dataset simulating:
- Organic (safe) search queries with normal behavioural signals
- Abusive queries: spam, fraud, scraping bots, cloaking, policy violations

Outputs: data/search_trust_dataset.csv
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker()
random.seed(42)
np.random.seed(42)

# ── Policy violation categories ──────────────────────────────────────────────
POLICY_LABELS = ["safe", "spam", "harmful", "misleading", "adult", "violence"]

SAFE_QUERIES = [
    "best python tutorials 2024", "weather in london today", "how to make sourdough",
    "nba scores last night", "iphone 15 review", "flight to new york", "french toast recipe",
    "machine learning roadmap", "yoga for beginners", "top rated restaurants near me",
    "how to learn piano", "history of the roman empire", "best laptops under 1000",
    "how does photosynthesis work", "numpy tutorial", "pandas dataframe merge",
]

SPAM_QUERIES = [
    "buy cheap followers instagram", "free robux generator no survey", "click here win prize",
    "make 1000 a day from home", "get rich quick scheme", "100% guaranteed weight loss",
    "unlimited coins hack", "cheap replica watches", "buy fake reviews",
]

HARMFUL_QUERIES = [
    "how to make dangerous chemicals at home", "illegal weapons purchase guide",
    "drug synthesis steps", "evade law enforcement tactics",
]

MISLEADING_QUERIES = [
    "vaccines cause autism proof", "moon landing was faked evidence", "flat earth proof",
    "covid cure suppressed by government", "miracle cancer cure doctors dont want you to know",
]

ADULT_QUERIES = [
    "adult content site", "explicit videos free", "18+ dating no verification",
]

VIOLENCE_QUERIES = [
    "how to hurt someone", "instructions for assault", "making a weapon guide",
]

QUERY_POOLS = {
    "safe": SAFE_QUERIES,
    "spam": SPAM_QUERIES,
    "harmful": HARMFUL_QUERIES,
    "misleading": MISLEADING_QUERIES,
    "adult": ADULT_QUERIES,
    "violence": VIOLENCE_QUERIES,
}

# ── Feature generators ────────────────────────────────────────────────────────

def generate_behavioral_features(label: str) -> dict:
    """Simulate user-level behavioral signals associated with abuse patterns."""
    is_abuse = label != "safe"

    return {
        # Query velocity: abusers send many queries quickly
        "query_velocity_per_hour": np.random.poisson(80 if is_abuse else 8),
        # Click-through rate: spam/bot traffic has abnormal CTR
        "ctr": round(np.random.beta(1, 9) if is_abuse else np.random.beta(3, 5), 4),
        # Session duration (seconds): bots have very short or scripted sessions
        "session_duration_sec": int(np.random.exponential(15 if is_abuse else 180)),
        # % queries from same IP subnet
        "ip_subnet_concentration": round(np.random.uniform(0.7, 1.0) if is_abuse else np.random.uniform(0.0, 0.3), 3),
        # User account age in days
        "account_age_days": int(np.random.exponential(5 if is_abuse else 500)),
        # Number of unique devices
        "unique_devices": int(np.random.poisson(1.2 if is_abuse else 2.5)),
        # Bounce rate
        "bounce_rate": round(np.random.uniform(0.6, 1.0) if is_abuse else np.random.uniform(0.1, 0.5), 3),
        # Prior abuse flags on account
        "prior_abuse_flags": int(np.random.poisson(2.5 if is_abuse else 0.05)),
        # Query contains URL (common in spam)
        "query_contains_url": int(random.random() < (0.6 if label == "spam" else 0.05)),
        # Query length (tokens)
        "query_length_tokens": int(np.random.normal(12 if is_abuse else 5, 3)),
        # Hour of day (night-time = higher bot activity)
        "query_hour": random.choice(range(0, 6)) if is_abuse else random.choice(range(7, 22)),
        # Repeated exact same query
        "exact_query_repeats": int(np.random.poisson(8 if is_abuse else 0.2)),
    }


def generate_row(label: str) -> dict:
    query_pool = QUERY_POOLS.get(label, SAFE_QUERIES)
    query = random.choice(query_pool)
    features = generate_behavioral_features(label)

    return {
        "query_id": fake.uuid4(),
        "query_text": query,
        "policy_label": label,
        "is_abuse": int(label != "safe"),
        "abuse_type": label if label != "safe" else "none",
        "timestamp": fake.date_time_between(start_date="-90d", end_date="now").isoformat(),
        **features,
    }


def generate_dataset(n_samples: int = 5000, output_path: str = "data/search_trust_dataset.csv") -> pd.DataFrame:
    """
    Generate balanced synthetic dataset.
    Distribution: 60% safe, 40% abuse (distributed across abuse types).

    Uses round() instead of int() to avoid silent row count shortfall from
    floating-point truncation (e.g. int(5000 * 0.08) = 400 but 6 × rounding
    error can drop ~50 rows total).
    """
    label_distribution = {
        "safe":       round(n_samples * 0.60),
        "spam":       round(n_samples * 0.15),
        "harmful":    round(n_samples * 0.08),
        "misleading": round(n_samples * 0.08),
        "adult":      round(n_samples * 0.05),
        "violence":   round(n_samples * 0.04),
    }

    rows = []
    for label, count in label_distribution.items():
        for _ in range(count):
            rows.append(generate_row(label))

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset generated: {len(df)} rows → {output_path}")
    print(f"   Label distribution:\n{df['policy_label'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    generate_dataset(n_samples=5000)
