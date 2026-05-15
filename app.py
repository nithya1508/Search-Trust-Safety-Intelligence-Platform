"""
dashboard/app.py
----------------
Streamlit Trust & Safety Intelligence Dashboard.

Displays:
- Real-time trust risk KPIs
- Policy violation breakdown
- Abuse rate time series
- Risk tier distribution
- ML model performance
- Active alerts

Run with: streamlit run dashboard/app.py
"""

import json
import os
import sys
from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Search Trust & Safety Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #7c3aed;
    }
    .alert-p0 { border-left: 4px solid #ef4444; background: #1f1015; border-radius: 8px; padding: 12px; margin: 6px 0; }
    .alert-p1 { border-left: 4px solid #f97316; background: #1a1208; border-radius: 8px; padding: 12px; margin: 6px 0; }
    .stMetric label { font-size: 0.85rem; color: #a0aec0; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    path = "data/search_trust_dataset.csv"
    if not os.path.exists(path):
        st.error("Dataset not found. Run: python data/generate_dataset.py")
        st.stop()
    return pd.read_csv(path)


@st.cache_data
def load_training_results():
    path = "models/training_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_alerts():
    path = "models/alerts.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


@st.cache_data
def load_llm_results():
    path = "models/llm_eval_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── Main dashboard ────────────────────────────────────────────────────────────
def main():
    df = load_dataset()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Keep date as Timestamp (not Python date) so pd.Timedelta arithmetic works correctly
    df["date"] = df["timestamp"].dt.normalize()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("# 🛡️ Search Trust & Safety Intelligence Platform")
    st.markdown("*Real-time abuse detection, policy violation tracking, and trust risk metrics*")
    st.divider()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔧 Filters")
        date_range = st.slider(
            "Days to display",
            min_value=7, max_value=90, value=30,
        )
        categories = st.multiselect(
            "Policy Categories",
            options=df["policy_label"].unique().tolist(),
            default=df["policy_label"].unique().tolist(),
        )

    # Use Timestamp comparison throughout — avoids TypeError with pd.Timedelta
    max_date = df["date"].max()
    df_filtered = df[
        (df["policy_label"].isin(categories)) &
        (df["date"] >= max_date - pd.Timedelta(days=date_range))
    ]

    # ── KPI Row ───────────────────────────────────────────────────────────────
    abuse_rate = df_filtered["is_abuse"].mean()
    total = len(df_filtered)
    abuse_count = df_filtered["is_abuse"].sum()
    categories_flagged = df_filtered[df_filtered["is_abuse"] == 1]["policy_label"].nunique()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Queries", f"{total:,}")
    col2.metric("Abuse Rate", f"{abuse_rate:.1%}", delta=f"{abuse_rate - 0.38:.1%}", delta_color="inverse")
    col3.metric("Abuse Detected", f"{int(abuse_count):,}")
    col4.metric("Categories Flagged", categories_flagged)
    col5.metric("Safe Queries", f"{int(total - abuse_count):,}")

    st.divider()

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### 📈 Abuse Rate Over Time")
        daily = (
            df_filtered.groupby("date")
            .agg(total=("query_id", "count"), abuse=("is_abuse", "sum"))
            .reset_index()
        )
        daily["abuse_rate"] = daily["abuse"] / daily["total"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["abuse_rate"],
            mode="lines+markers",
            line=dict(color="#7c3aed", width=2),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.1)",
            name="Abuse Rate",
        ))
        fig.add_hline(y=daily["abuse_rate"].mean(), line_dash="dash",
                      line_color="#f59e0b", annotation_text="Mean")
        fig.update_layout(
            template="plotly_dark", height=280,
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 🗂️ Policy Violation Breakdown")
        violation_counts = (
            df_filtered[df_filtered["is_abuse"] == 1]["policy_label"]
            .value_counts()
            .reset_index()
        )
        violation_counts.columns = ["policy_label", "count"]

        COLORS = {
            "spam": "#f59e0b", "harmful": "#ef4444", "misleading": "#8b5cf6",
            "adult": "#ec4899", "violence": "#dc2626"
        }
        fig2 = px.pie(
            violation_counts, values="count", names="policy_label",
            color="policy_label",
            color_discrete_map=COLORS,
            hole=0.4,
        )
        fig2.update_layout(
            template="plotly_dark", height=280,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(font=dict(size=11)),
            showlegend=True,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── ML Model Performance ──────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🤖 ML Model Performance")

    training_results = load_training_results()
    llm_results = load_llm_results()

    col_ml1, col_ml2, col_ml3, col_ml4 = st.columns(4)

    if training_results:
        col_ml1.metric("XGBoost ROC-AUC", f"{training_results.get('roc_auc', 'N/A')}")
        col_ml2.metric("Avg Precision", f"{training_results.get('average_precision', 'N/A')}")
        cv = training_results.get("cv_auc_mean")
        cv_std = training_results.get("cv_auc_std")
        col_ml3.metric("CV AUC (5-fold)", f"{cv} ± {cv_std}" if cv else "N/A")
    else:
        col_ml1.metric("XGBoost ROC-AUC", "Run --train first")

    if llm_results:
        col_ml4.metric("LLM Policy Accuracy", f"{llm_results.get('overall_accuracy', 0):.1%}")

    if training_results and "top_features" in training_results:
        st.markdown("**Top SHAP Features**")
        feat_df = pd.DataFrame(
            list(training_results["top_features"].items()),
            columns=["feature", "importance"]
        ).sort_values("importance", ascending=True)

        fig3 = px.bar(feat_df, x="importance", y="feature", orientation="h",
                      color="importance", color_continuous_scale="Purples")
        fig3.update_layout(template="plotly_dark", height=280,
                           margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Alerts Panel ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🚨 Active Alerts")

    alerts = load_alerts()
    if not alerts:
        st.info("No active alerts. Run `python run_pipeline.py` to generate alert data.")
    else:
        for alert in alerts[:8]:
            severity = alert.get("severity", "P2")
            css_class = "alert-p0" if severity == "P0" else "alert-p1"
            icon = "🔴" if severity == "P0" else "🟠"
            st.markdown(
                f'<div class="{css_class}">'
                f'<strong>{icon} [{severity}] {alert["title"]}</strong><br>'
                f'<small>{alert["description"]}</small><br>'
                f'<em>Action: {alert.get("recommended_action", "")}</em>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Raw Data Explorer ─────────────────────────────────────────────────────
    st.divider()
    with st.expander("🔍 Raw Query Explorer"):
        show_abuse_only = st.toggle("Show abuse only", value=False)
        display_df = df_filtered[df_filtered["is_abuse"] == 1] if show_abuse_only else df_filtered
        st.dataframe(
            display_df[["query_text", "policy_label", "query_velocity_per_hour",
                        "ctr", "prior_abuse_flags", "account_age_days"]].head(200),
            use_container_width=True,
        )

    st.caption("🛡️ Search Trust & Safety Intelligence Platform | Built by Nithyashree Babu")


if __name__ == "__main__":
    main()
