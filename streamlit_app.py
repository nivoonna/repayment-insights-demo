import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import altair as alt
import matplotlib.pyplot as plt

# ------------------ Page config ------------------
st.set_page_config(
    page_title="Repayment Insights Layer",
    layout="wide"
)

# ------------------ Data loading ------------------
@st.cache_data
def load_data(default_path: str = "repayment_insights_synthetic.csv", uploaded=None) -> pd.DataFrame:
    """Load repayment CSV and do basic cleaning + feature setup."""
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv(default_path)

    # Dates
    for col in ["cycle_start_date", "due_date", "statement_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Basic ratios
    df["pay_ratio"] = (df["amount_paid"] / df["amount_due"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["fee_to_paid"] = df.apply(
        lambda r: r["fee_assessed"] / max(r["amount_paid"], 1e-6),
        axis=1
    )
    df = df.sort_values(["borrower_id", "cycle_start_date"])
    return df


def label_partials(df, min_ratio: float, max_ratio: float):
    """Flag partial payments that show intent but are not full."""
    return ((df["pay_ratio"] >= min_ratio) & (df["pay_ratio"] < max_ratio)).astype(int)


def label_spirals(df, min_fee_cycles: int, fee_to_paid_threshold: float):
    """Flag potential penalty spirals."""
    def _per_borrower(g):
        g = g.sort_values("cycle_start_date").copy()
        # how many of last 3 cycles had fees
        g["fees_rolling3"] = g["fee_assessed"].rolling(3, min_periods=1).apply(
            lambda x: (x > 0).sum(),
            raw=True
        )
        g["fee_to_paid_prev"] = g["fee_to_paid"].shift(1).fillna(0)
        g["spiral_flag"] = (
            (g["fees_rolling3"] >= min_fee_cycles) |
            ((g["fee_to_paid"] > fee_to_paid_threshold) & (g["fee_to_paid_prev"] > fee_to_paid_threshold))
        ).astype(int)
        return g

    return df.groupby("borrower_id", group_keys=False).apply(_per_borrower)


def explain_row(row, is_partial: bool) -> str:
    """Plain-English repayment explanation."""
    text = str(row.get("statement_text", "")).lower()

    if "autopay failed" in text or "insufficient" in text:
        return "Autopay failure likely caused this delay or fee."
    if "hardship" in text:
        return "Hardship plan active – reduced payment accepted."
    if "dispute" in text or "posted late" in text or "duplicated" in text:
        return "Fee is disputed – needs manual review."
    if is_partial and row["days_late"] == 0:
        return "Paid part of the bill on time – shows intent but not full coverage."
    if row["days_late"] > 0 and row["fee_assessed"] > 0:
        return "Late with fee – could snowball if it keeps happening."
    if row["pay_ratio"] >= 0.95:
        return "On-time full or near-full payment."
    return "Mixed pattern – keep an eye on next cycle."


# ------------------ Sidebar (controls) ------------------
st.sidebar.title("Settings")
st.sidebar.caption("Tune how generous or strict you want the definitions to be.")

partial_min = st.sidebar.slider(
    "Minimum % of bill to count as 'trying to pay'",
    min_value=0.0,
    max_value=1.0,
    value=0.40,
    step=0.05,
    help="If a person pays at least this share of the bill, we treat it as a partial payment with intent."
)
partial_max = st.sidebar.slider(
    "Above this % counts as 'full payment'",
    min_value=0.5,
    max_value=1.1,
    value=0.95,
    step=0.05,
    help="Payments above this share are basically full / on-time."
)

min_fee_cycles = st.sidebar.slider(
    "Flag if fees in the last _ cycles",
    min_value=1,
    max_value=3,
    value=2,
    step=1,
    help="How many recent fee-bearing cycles should trigger a 'penalty spiral' risk flag?"
)

fee_to_paid_threshold = st.sidebar.slider(
    "Flag if fees > _% of payment twice in a row",
    min_value=0.05,
    max_value=0.5,
    value=0.20,
    step=0.05,
    help="If fees are bigger than this share of the payment in back-to-back months, we mark it as risky."
)

uploaded = st.sidebar.file_uploader(
    "Upload repayment CSV (optional)",
    type=["csv"],
    help="Use your own data with the same columns as the demo file."
)

# ------------------ Data & features ------------------
df = load_data(uploaded=uploaded)

df["is_partial_payment"] = label_partials(df, partial_min, partial_max)
df = label_spirals(df, min_fee_cycles, fee_to_paid_threshold)

# Borrower-level aggregates
agg = df.groupby("borrower_id").agg(
    cycles=("borrower_id", "size"),
    mean_pay_ratio=("pay_ratio", "mean"),
    partial_rate=("is_partial_payment", "mean"),
    late_rate=("days_late", lambda x: np.mean(x > 0)),
    mean_days_late=("days_late", "mean"),
    fee_rate=("fee_assessed", lambda x: np.mean(x > 0)),
    mean_fee=("fee_assessed", "mean"),
    spiral_rate=("spiral_flag", "mean"),
    hardship_any=("hardship_flag", "max"),
    apr=("apr", "mean")
).reset_index().fillna(0)

# Simple behavioral segments for humans
def segment_borrower(row):
    if row["spiral_rate"] > 0.3:
        return "Spiral risk"
    if row["late_rate"] == 0 and row["fee_rate"] == 0 and row["mean_pay_ratio"] >= 0.95:
        return "On-time & clean"
    if row["partial_rate"] > 0.3 and row["late_rate"] < 0.5:
        return "Partial but trying"
    return "Irregular"

agg["segment"] = agg.apply(segment_borrower, axis=1)

# Clustering for internal structure (still used, but hidden)
features = agg[["mean_pay_ratio", "partial_rate", "late_rate", "fee_rate", "mean_fee", "spiral_rate"]].values
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
agg["cluster"] = kmeans.fit_predict(features)

# Anomaly detection at event level
anomaly_features = df[["pay_ratio", "days_late", "fee_assessed", "fee_to_paid"]].copy()
iso = IsolationForest(n_estimators=200, contamination=0.08, random_state=42)
df["anomaly_score"] = iso.fit_predict(anomaly_features)  # -1 anomaly, 1 normal
df["is_anomaly"] = (df["anomaly_score"] == -1).astype(int)

# Explanations
df["explanation"] = [
    explain_row(r, p) for r, p in zip(df.to_dict("records"), df["is_partial_payment"])
]

# ------------------ Main layout ------------------
st.title("Repayment Insights Layer — Demo")

borrowers_n = agg["borrower_id"].nunique()
events_n = len(df)
any_partial = int((agg["partial_rate"] > 0).sum())
any_spiral = int((agg["spiral_rate"] > 0).sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Borrowers", f"{borrowers_n:,}")
col2.metric("Events (bills)", f"{events_n:,}")
col3.metric("Borrowers showing intent (partials)", f"{any_partial:,}")
col4.metric("Borrowers with spiral risk", f"{any_spiral:,}")

# Quick narrative line
st.info(
    f"Out of {borrowers_n} borrowers, {any_partial} show some intent to pay via partial payments, "
    f"and {any_spiral} show early signs of small fees snowballing into penalty spirals."
)

st.markdown("### 1. Repayment behaviour groups")

scatter_df = agg.copy()
scatter_df["segment"] = scatter_df["segment"].astype("category")

chart = (
    alt.Chart(scatter_df)
    .mark_circle(size=80)
    .encode(
        x=alt.X("mean_pay_ratio:Q", title="Average share of bill paid"),
        y=alt.Y("late_rate:Q", title="Share of months paid late"),
        color=alt.Color("segment:N", title="Repayment pattern"),
        tooltip=[
            "borrower_id",
            "segment",
            alt.Tooltip("mean_pay_ratio:Q", title="Average % paid", format=".2f"),
            alt.Tooltip("late_rate:Q", title="% months late", format=".2f"),
            alt.Tooltip("fee_rate:Q", title="% months with fees", format=".2f"),
            alt.Tooltip("spiral_rate:Q", title="Spiral risk score", format=".2f"),
        ],
    )
    .interactive()
)

st.altair_chart(chart, use_container_width=True)

st.caption("Each dot is a borrower. Hover to see how often they pay, pay late, and get hit with fees.")

# Segment summary
segment_summary = (
    agg.groupby("segment")
    .agg(
        borrowers=("borrower_id", "nunique"),
        avg_pay_share=("mean_pay_ratio", "mean"),
        months_late_share=("late_rate", "mean"),
        months_with_fees=("fee_rate", "mean"),
        spiral_score=("spiral_rate", "mean"),
    )
    .round(3)
    .reset_index()
)

st.markdown("**Segment overview**")
st.dataframe(segment_summary, use_container_width=True)

# ------------------ Risk & fee story ------------------
st.markdown("### 2. People who might be in trouble")

# Filter for anomalies or spiral flags
watch = df[(df["is_anomaly"] == 1) | (df["spiral_flag"] == 1)].copy()
watch["fee_paid_pct"] = (watch["fee_to_paid"] * 100).replace([np.inf, -np.inf], np.nan).round(1)

min_days_late = st.slider(
    "Show people who are at least this many days late",
    min_value=0,
    max_value=int(df["days_late"].max()),
    value=2,
    step=1
)
min_fee_pct = st.slider(
    "And/or have fees above this % of their payment",
    min_value=0.0,
    max_value=float(np.nanmax(watch["fee_paid_pct"].fillna(0))) if len(watch) else 1.0,
    value=20.0,
    step=5.0
)

watch_filtered = watch[
    (watch["days_late"] >= min_days_late) | (watch["fee_paid_pct"] >= min_fee_pct)
].copy()

watch_view = watch_filtered[[
    "borrower_id",
    "cycle_start_date",
    "amount_due",
    "amount_paid",
    "days_late",
    "fee_assessed",
    "fee_paid_pct",
    "explanation",
]].sort_values(["fee_paid_pct", "days_late"], ascending=False)

st.dataframe(
    watch_view.head(30),
    use_container_width=True,
    hide_index=True
)

st.caption("This list shows people where small fees and delays may already be turning into a problem.")

# ------------------ Single-borrower story ------------------
st.markdown("### 3. Zoom in on one borrower")

b_list = sorted(df["borrower_id"].unique())
sel_b = st.selectbox("Pick a borrower to see their story", b_list)

timeline = df[df["borrower_id"] == sel_b].sort_values("cycle_start_date")

st.write(
    f"Borrower **{sel_b}** · APR ~ {timeline['apr'].mean():.1%} · "
    f"{int(timeline['is_partial_payment'].sum())} partial payments · "
    f"{int(timeline['spiral_flag'].sum())} cycles with spiral risk."
)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(timeline["cycle_start_date"], timeline["amount_due"], label="Bill amount")
ax.plot(timeline["cycle_start_date"], timeline["amount_paid"], label="Paid")
ax.scatter(timeline["cycle_start_date"], timeline["fee_assessed"], marker="x", label="Fees")
ax.set_xlabel("Cycle")
ax.set_ylabel("Amount")
ax.set_title(f"How {sel_b}'s bills, payments and fees move over time")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

st.markdown("**Recent cycles for this borrower**")
st.dataframe(
    timeline[[
        "cycle_start_date",
        "amount_due",
        "amount_paid",
        "days_late",
        "fee_assessed",
        "pay_ratio",
        "explanation",
    ]].tail(8),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")
st.caption(
    "Idea: instead of only telling people they are 'late', we surface intent, small penalties, "
    "and a simple story of what is going on with their repayments."
)
