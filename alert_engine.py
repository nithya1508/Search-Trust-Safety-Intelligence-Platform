"""
alert_engine.py
---------------
Anomaly detection and alerting on trust signals.

Raises alerts when:
- Abuse rate spikes above threshold vs. rolling average
- A specific abuse category surges
- Feature distributions drift (PSI alert)
- High-confidence CRITICAL risk queries are detected

Designed to integrate with the on-call rotation workflow.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import json


@dataclass
class TrustAlert:
    alert_id: str
    severity: str          # P0, P1, P2
    alert_type: str        # ABUSE_SPIKE, CATEGORY_SURGE, DRIFT, CRITICAL_QUERY
    title: str
    description: str
    metric_value: float
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    recommended_action: str = ""
    requires_escalation: bool = False


class AlertEngine:
    """
    Rule-based alert engine for trust signal monitoring.
    Used during on-call rotations to surface urgent issues.
    """

    # Thresholds
    ABUSE_RATE_SPIKE_THRESHOLD = 0.10      # >10% above rolling avg → alert
    CATEGORY_SURGE_THRESHOLD = 2.0         # 2× expected prevalence → alert
    PSI_ALERT_THRESHOLD = 0.2              # PSI > 0.2 → major drift
    CRITICAL_QUERY_THRESHOLD = 0.80        # abuse_probability > 0.80
    MIN_SAMPLES = 20                       # Minimum samples to trigger alert

    def __init__(self):
        self.alerts: list[TrustAlert] = []

    def check_abuse_rate_spike(self, df: pd.DataFrame) -> Optional[TrustAlert]:
        """Alert if today's abuse rate significantly exceeds the 7-day rolling average."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily = df.groupby("date")["is_abuse"].mean()

        if len(daily) < 3:
            return None

        rolling_avg = daily.iloc[:-1].mean()
        today_rate = daily.iloc[-1]
        delta = today_rate - rolling_avg

        if delta > self.ABUSE_RATE_SPIKE_THRESHOLD and len(df[df["date"] == daily.index[-1]]) >= self.MIN_SAMPLES:
            alert = TrustAlert(
                alert_id=f"ABUSE_SPIKE_{daily.index[-1]}",
                severity="P1",
                alert_type="ABUSE_SPIKE",
                title=f"Abuse Rate Spike Detected ({daily.index[-1]})",
                description=(
                    f"Today's abuse rate is {today_rate:.1%}, vs 7-day average of {rolling_avg:.1%}. "
                    f"Delta: +{delta:.1%}. Potential coordinated abuse campaign."
                ),
                metric_value=round(today_rate, 4),
                threshold=round(rolling_avg + self.ABUSE_RATE_SPIKE_THRESHOLD, 4),
                recommended_action="Review top flagged queries, check IP clustering, escalate if >20% spike.",
                requires_escalation=delta > 0.20,
            )
            self.alerts.append(alert)
            return alert
        return None

    def check_category_surge(self, df: pd.DataFrame) -> list[TrustAlert]:
        """Alert if a specific policy category surges vs. its expected baseline.

        Uses the chronological midpoint (median date) to split baseline vs.
        recent traffic, avoiding pandas quantile() inconsistencies on date columns.
        """
        category_alerts = []
        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        # Sort and split on median date for a stable, version-safe split
        sorted_dates = sorted(df["date"].unique())
        if len(sorted_dates) < 2:
            return []
        midpoint_date = sorted_dates[len(sorted_dates) // 2]

        baseline_df = df[df["date"] <= midpoint_date]
        recent_df = df[df["date"] > midpoint_date]

        if len(recent_df) < self.MIN_SAMPLES:
            return []

        for category in ["spam", "harmful", "misleading", "adult", "violence"]:
            baseline_rate = (baseline_df["policy_label"] == category).mean()
            recent_rate = (recent_df["policy_label"] == category).mean()

            if baseline_rate > 0 and recent_rate / baseline_rate > self.CATEGORY_SURGE_THRESHOLD:
                ratio = recent_rate / baseline_rate
                severity = "P0" if category in ("harmful", "violence") else "P1"
                alert = TrustAlert(
                    alert_id=f"CATEGORY_SURGE_{category.upper()}",
                    severity=severity,
                    alert_type="CATEGORY_SURGE",
                    title=f"{category.title()} Category Surge ({ratio:.1f}× baseline)",
                    description=(
                        f"'{category}' violations at {recent_rate:.1%} vs baseline {baseline_rate:.1%}. "
                        f"Ratio: {ratio:.1f}×."
                    ),
                    metric_value=round(recent_rate, 4),
                    threshold=round(baseline_rate * self.CATEGORY_SURGE_THRESHOLD, 4),
                    recommended_action=f"Investigate '{category}' query patterns. Update classifier thresholds if needed.",
                    requires_escalation=severity == "P0",
                )
                self.alerts.append(alert)
                category_alerts.append(alert)

        return category_alerts

    def check_high_risk_queries(self, df: pd.DataFrame) -> list[TrustAlert]:
        """Alert on any CRITICAL tier queries detected in the current window."""
        if "risk_tier" not in df.columns:
            return []

        critical = df[df["risk_tier"] == "CRITICAL"]
        if len(critical) == 0:
            return []

        alert = TrustAlert(
            alert_id="CRITICAL_QUERIES_DETECTED",
            severity="P0",
            alert_type="CRITICAL_QUERY",
            title=f"{len(critical)} CRITICAL Risk Queries Detected",
            description=(
                f"{len(critical)} queries scored above {self.CRITICAL_QUERY_THRESHOLD} abuse probability. "
                f"Sample: {critical['query_text'].iloc[0][:80]}"
            ),
            metric_value=len(critical),
            threshold=0,
            recommended_action="Immediate manual review. Block if policy violation confirmed. Log for classifier retraining.",
            requires_escalation=True,
        )
        self.alerts.append(alert)
        return [alert]

    def check_drift_alerts(self, drift_report: dict) -> Optional[TrustAlert]:
        """Raise alert if PSI drift analysis detected major feature shift."""
        summary = drift_report.get("_summary", {})
        if not summary.get("drift_detected"):
            return None

        alert_features = summary.get("alert_features", [])
        alert = TrustAlert(
            alert_id="FEATURE_DRIFT_DETECTED",
            severity="P1",
            alert_type="DRIFT",
            title=f"Feature Distribution Drift Detected ({len(alert_features)} features)",
            description=(
                f"PSI > 0.2 detected in: {', '.join(alert_features)}. "
                "This may indicate a shift in abuse patterns or data pipeline issues."
            ),
            metric_value=len(alert_features),
            threshold=1,
            recommended_action="Review data pipeline. Retrain model if behavioural shift confirmed. Notify data engineering.",
            requires_escalation=len(alert_features) > 2,
        )
        self.alerts.append(alert)
        return alert

    def run_all_checks(self, df: pd.DataFrame, drift_report: Optional[dict] = None) -> list[dict]:
        """Run all alert checks and return summary."""
        self.alerts = []
        self.check_abuse_rate_spike(df)
        self.check_category_surge(df)
        self.check_high_risk_queries(df)
        if drift_report:
            self.check_drift_alerts(drift_report)

        alerts_dict = [asdict(a) for a in self.alerts]
        p0_count = sum(1 for a in self.alerts if a.severity == "P0")
        p1_count = sum(1 for a in self.alerts if a.severity == "P1")

        print(f"🚨 Alert Engine: {len(alerts_dict)} alerts raised (P0: {p0_count}, P1: {p1_count})")
        for alert in self.alerts:
            icon = "🔴" if alert.severity == "P0" else "🟠"
            print(f"   {icon} [{alert.severity}] {alert.title}")

        return alerts_dict

    def export_alerts(self, path: str = "models/alerts.json"):
        import os
        os.makedirs("models", exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(a) for a in self.alerts], f, indent=2)
        print(f"✅ Alerts exported to {path}")
