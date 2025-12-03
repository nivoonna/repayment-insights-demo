import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


st.set_page_config(
    page_title="Repayment Insights Layer — Portfolio Dashboard",
    layout="wide"
)

# Data loading & feature engineering
@st.cache_data
def load_data(path: str = "repayment_insights_synthetic.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Parse dates
    for col in ["cycle_start_date", "due_date", "statement_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Ratios
    df["pay_ratio"] = (df["amount_paid"] / df["amount_due"]).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    df["fee_to_paid"] = df.apply(
        lambda r: r["fee_assessed"] / max(r["amount_paid"], 1e-6),
        axis=1,
    )

    df = df.sort_values(["borrower_id", "cycle_start_date"])
    return df


def label_partials(df: pd.DataFrame, min_ratio: float, max_ratio: float) -> pd.Series:
    """Flag partial payments that show intent but are not full."""
    return ((df["pay_ratio"] >= min_ratio) & (df["pay_ratio"] < max_ratio)).astype(int)


def label_spirals(df: pd.DataFrame, min_fee_cycles: int, fee_to_paid_threshold: float) -> pd.DataFrame:
    """Flag potential penalty spirals per borrower."""
    def _per_borrower(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("cycle_start_date").copy()
        # count how many of last 3 cycles had any fee
        g["fees_rolling3"] = g["fee_assessed"].rolling(
            3, min_periods=1
        ).apply(lambda x: (x > 0).sum(), raw=True)
        g["fee_to_paid_prev"] = g["fee_to_paid"].shift(1).fillna(0)
        g["spiral_flag"] = (
            (g["fees_rolling3"] >= min_fee_cycles) |
            ((g["fee_to_paid"] > fee_to_paid_threshold) &
             (g["fee_to_paid_prev"] > fee_to_paid_threshold))
        ).astype(int)
        return g

    return df.groupby("borrower_id", group_keys=False).apply(_per_borrower)


def explain_row(row, is_partial: bool) -> str:
    """Plain-language explanation for a single cycle."""
    text = str(row.get("statement_text", "")).lower()

    if "autopay failed" in text or "insufficient" in text:
        return "Autopay failure likely caused this delay or fee."
    if "hardship" in text:
        return "Hardship plan active – reduced payment accepted."
    if "dispute" in text or "posted late" in text or "duplicated" in text:
        return "Fee is disputed – needs manual review."
    if is_partial and row["days_late"] == 0:
        return "Paid part of the bill on time – intent to pay but not in full."
    if row["days_late"] > 0 and row["fee_assessed"] > 0:
        return "Late with fee – could snowball if it continues."
    if row["pay_ratio"] >= 0.95:
        return "On-time full or near-full payment."
    return "Mixed behaviour in this cycle."


def segment_borrower(row) -> str:
    """High-level behavioural segment per borrower."""
    if row["spiral_rate"] > 0.3:
        return "Spiral risk"
    if row["late_rate"] == 0 and row["fee_rate"] == 0 and row["mean_pay_ratio"] >= 0.95:
        return "On-time & clean"
    if row["partial_rate"] > 0.3 and row["late_rate"] < 0.5:
        return "Partial but trying"
    return "Irregular"


# -------------------------------------------------------
# Sidebar: model thresholds
# -------------------------------------------------------
st.sidebar.title("Model thresholds")
st.sidebar.caption("Adjust how the layer interprets repayment behaviour across your portfolio.")

st.sidebar.subheader("Partial payments (intent band)")
partial_min_pct, partial_max_pct = st.sidebar.slider(
    "Treat a payment as 'partial but intentful' when it covers between…",
    min_value=10,
    max_value=110,
    value=(40, 95),   # 40%–95% by default
    step=5,
    help="Below the lower bound we treat it as weak intent; above the upper bound it behaves like a full payment.",
)
partial_min = partial_min_pct / 100.0
partial_max = partial_max_pct / 100.0

st.sidebar.subheader("Penalty spiral detection")
min_fee_cycles = st.sidebar.slider(
    "Minimum fee-charged cycles (out of the last 3)",
    min_value=1,
    max_value=3,
    value=2,
    step=1,
    help="How many of the last three cycles must have a fee before we flag spiral risk.",
)

fee_to_paid_threshold_pct = st.sidebar.slider(
    "Minimum fee share of payment in 2 consecutive cycles (%)",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    help="Example: 20% means a $10 fee on a $50 payment, repeated across two cycles.",
)
fee_to_paid_threshold = fee_to_paid_threshold_pct / 100.0

uploaded = st.sidebar.file_uploader(
    "Upload repayment CSV (optional)",
    type=["csv"],
    help="Override demo data with your own portfolio extract.",
)

# -------------------------------------------------------
# Data + features
# -------------------------------------------------------
if uploaded is not None:
    df = load_data(path=uploaded)
else:
    df = load_data()

# Label partials & spirals
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
    apr=("apr", "mean"),
).reset_index().fillna(0)

agg["segment"] = agg.apply(segment_borrower, axis=1)

# Internal clustering (for structure, optional but useful)
features = agg[["mean_pay_ratio", "partial_rate", "late_rate", "fee_rate", "mean_fee", "spiral_rate"]].values
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
agg["cluster"] = kmeans.fit_predict(features)

# Event-level anomaly detection
anomaly_features = df[["pay_ratio", "days_late", "fee_assessed", "fee_to_paid"]].copy()
iso = IsolationForest(n_estimators=200, contamination=0.08, random_state=42)
df["anomaly_score"] = iso.fit_predict(anomaly_features)  # -1 anomaly, 1 normal
df["is_anomaly"] = (df["anomaly_score"] == -1).astype(int)

# Explanations per event
df["explanation"] = [
    explain_row(r, p) for r, p in zip(df.to_dict("records"), df["is_partial_payment"])
]

# -------------------------------------------------------
# Header & high-level metrics
# -------------------------------------------------------
st.title("Repayment Insights Layer — Portfolio Dashboard")

borrowers_n = agg["borrower_id"].nunique()
events_n = len(df)
any_partial = int((agg["partial_rate"] > 0).sum())
any_spiral = int((agg["spiral_rate"] > 0).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Borrowers", f"{borrowers_n:,}")
c2.metric("Repayment events", f"{events_n:,}")
c3.metric("Borrowers with partial intent", f"{any_partial:,}")
c4.metric("Borrowers with spiral signals", f"{any_spiral:,}")

st.info(
    f"Out of {borrowers_n} borrowers, {any_partial} show some intent to pay via partial payments, "
    f"and {any_spiral} display early penalty spiral signals under the current thresholds."
)

# -------------------------------------------------------
# 1. Behaviour groups
# -------------------------------------------------------
st.markdown("### 1. Repayment behaviour groups")

scatter_df = agg.copy()
scatter_df["segment"] = scatter_df["segment"].astype("category")

chart = (
    alt.Chart(scatter_df)
    .mark_circle(size=80)
    .encode(
        x=alt.X("mean_pay_ratio:Q", title="Average share of bill paid"),
        y=alt.Y("late_rate:Q", title="Share of cycles paid late"),
        color=alt.Color("segment:N", title="Behaviour segment"),
        tooltip=[
            "borrower_id",
            "segment",
            alt.Tooltip("mean_pay_ratio:Q", title="Avg % paid", format=".2f"),
            alt.Tooltip("late_rate:Q", title="% cycles late", format=".2f"),
            alt.Tooltip("fee_rate:Q", title="% cycles with fees", format=".2f"),
            alt.Tooltip("spiral_rate:Q", title="Spiral rate", format=".2f"),
        ],
    )
    .interactive()
)

st.altair_chart(chart, use_container_width=True)
st.caption("Each dot is a borrower. Hover to see how often they pay, pay late, and get hit with fees under the current settings.")

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
st.dataframe(segment_summary, use_container_width=True, hide_index=True)

# -------------------------------------------------------
# 2. People who might be in trouble
# -------------------------------------------------------
st.markdown("### 2. People who might be in trouble")

watch = df[(df["is_anomaly"] == 1) | (df["spiral_flag"] == 1)].copy()
watch["fee_paid_pct"] = (watch["fee_to_paid"] * 100).replace(
    [np.inf, -np.inf], np.nan
).round(1)

min_days_late = st.slider(
    "Only show cycles that are at least this many days late",
    min_value=0,
    max_value=int(df["days_late"].max()),
    value=2,
    step=1,
)

max_fee_pct = float(np.nanmax(watch["fee_paid_pct"].fillna(0))) if len(watch) else 100.0
min_fee_pct = st.slider(
    "…and where fees are at least this % of the payment",
    min_value=0.0,
    max_value=max_fee_pct if max_fee_pct > 0 else 100.0,
    value=20.0,
    step=5.0,
)

watch_filtered = watch[
    (watch["days_late"] >= min_days_late) &
    (watch["fee_paid_pct"] >= min_fee_pct)
].copy()

# Composite risk score for ordering
if len(watch_filtered) > 0:
    watch_filtered["risk_score"] = (
        watch_filtered["days_late"].rank(pct=True) * 0.5 +
        watch_filtered["fee_paid_pct"].rank(pct=True) * 0.5
    )
else:
    watch_filtered["risk_score"] = []

cols = [
    "borrower_id",
    "cycle_start_date",
    "amount_due",
    "amount_paid",
    "days_late",
    "fee_assessed",
    "fee_paid_pct",
    "risk_score",
    "explanation",
]

st.dataframe(
    watch_filtered[cols]
        .sort_values("risk_score", ascending=False)
        .head(50),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "This table responds directly to the sliders above and surfaces cycles that are both late "
    "and fee-heavy, ordered from highest combined risk."
)

# -------------------------------------------------------
# 3. Zoom in on one borrower
# -------------------------------------------------------
st.markdown("### 3. Zoom in on one borrower")

b_list = sorted(df["borrower_id"].unique())
sel_b = st.selectbox("Pick a borrower to see their timeline", b_list)

timeline = df[df["borrower_id"] == sel_b].sort_values("cycle_start_date")

st.write(
    f"Borrower **{sel_b}** · APR ~ {timeline['apr'].mean():.1%} · "
    f"{int(timeline['is_partial_payment'].sum())} partial-payment cycles · "
    f"{int(timeline['spiral_flag'].sum())} cycles with spiral risk."
)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(
    timeline["cycle_start_date"],
    timeline["amount_due"],
    label="Bill amount",
)
ax.plot(
    timeline["cycle_start_date"],
    timeline["amount_paid"],
    label="Amount paid",
)
ax.scatter(
    timeline["cycle_start_date"],
    timeline["fee_assessed"],
    marker="x",
    s=60,
    label="Fee charged in that cycle",
)
ax.set_xlabel("Cycle")
ax.set_ylabel("Amount")
ax.set_title(f"How {sel_b}'s bills, payments and fees move over time")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

st.caption(
    "Lines show what was billed and what was paid in each cycle. "
    "The X markers indicate cycles where a fee was charged."
)

st.markdown("---")
st.caption(
    "Internal tool for risk and product teams: use thresholds on the left to explore how different "
    "definitions of 'intent to pay' and 'penalty spirals' reshape the portfolio view."
)
