import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="Repayment Insights Layer",
    layout="wide"
)

# -------------------------------------------------------
# Data loading & feature engineering
# -------------------------------------------------------
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


def detect_confusions(row) -> list:
    """Lightweight confusion detection based on text + behaviour."""
    confusions = []
    text = str(row.get("statement_text", "")).lower()

    if "minimum" in text and "statement" in text:
        confusions.append("Minimum vs statement balance confusion")
    if "autopay" in text and ("failed" in text or "insufficient" in text):
        confusions.append("Autopay failure / misconfiguration")
    if "posted late" in text or "posting delay" in text or "pending" in text:
        confusions.append("Payment posting delay confusion")
    if "dispute" in text or "fee" in text and "duplicate" in text:
        confusions.append("Fee dispute / duplicate fee suspicion")
    return confusions


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


def borrower_persona(user_df: pd.DataFrame) -> str:
    """Persona label for Borrower Story tab."""
    late_share = np.mean(user_df["days_late"] > 0)
    partial_share = np.mean(user_df["is_partial_payment"] == 1)
    spiral_share = np.mean(user_df["spiral_flag"] == 1)

    if spiral_share > 0.3:
        return "Penalty spiral risk"
    if late_share == 0 and spiral_share == 0 and partial_share < 0.2:
        return "Consistently on-time"
    if partial_share > 0.3 and late_share < 0.4 and spiral_share == 0:
        return "Intentful but constrained"
    if late_share > 0.4 and spiral_share == 0:
        return "Chronically late but fee-light"
    return "Mixed / unstable pattern"


def borrower_status(user_df: pd.DataFrame) -> tuple:
    """Return (emoji_label, color, blurb) for borrower health."""
    late_share = np.mean(user_df["days_late"] > 0)
    spiral_share = np.mean(user_df["spiral_flag"] == 1)
    partial_share = np.mean(user_df["is_partial_payment"] == 1)

    if spiral_share > 0.25 or (late_share > 0.4 and user_df["fee_assessed"].sum() > 0):
        return ("🔴 At risk of fee spirals",
                "#ff4b4b",
                "Frequent lateness and recurring fees suggest a growing penalty spiral.")
    if partial_share > 0.3 or late_share > 0:
        return ("🟡 Catching up",
                "#ffb000",
                "You’re paying and often show intent, but not always in full or on time.")
    return ("🟢 On track",
            "#00c851",
            "You’re paying on time and avoiding most fees. Behaviour looks healthy.")


def borrower_story(user_df: pd.DataFrame, confusions_all: list) -> str:
    """Generate a short textual repayment story for a borrower."""
    cycles = len(user_df)
    last_6 = user_df.tail(min(6, cycles))
    on_time_full = ((last_6["days_late"] == 0) & (last_6["pay_ratio"] >= 0.95)).sum()
    on_time_partial = ((last_6["days_late"] == 0) & (last_6["is_partial_payment"] == 1)).sum()
    late = (last_6["days_late"] > 0).sum()
    fees = (last_6["fee_assessed"] > 0).sum()

    lines = []
    lines.append(f"In the last {len(last_6)} cycles, this borrower paid {on_time_full} in full on time and {on_time_partial} as on-time partials.")
    if late > 0:
        lines.append(f"They were late in {late} cycles, and fees were charged in {fees} of those.")
    else:
        lines.append("They have not been late in the most recent cycles.")
    avg_ratio = last_6["pay_ratio"].mean()
    lines.append(f"On average, they cover about {avg_ratio*100:.0f}% of each bill.")
    if confusions_all:
        lines.append("There are signs of confusion around: " + ", ".join(sorted(set(confusions_all))) + ".")
    return " ".join(lines)


def borrower_recommendations(user_df: pd.DataFrame, confusions_all: list) -> list:
    """Generate 2–4 recommendations."""
    recs = []
    late_share = np.mean(user_df["days_late"] > 0)
    spiral_share = np.mean(user_df["spiral_flag"] == 1)
    partial_share = np.mean(user_df["is_partial_payment"] == 1)

    if spiral_share > 0 or user_df["fee_assessed"].sum() > 0:
        recs.append("Review recent fees and consider proactive outreach for customers with repeated small charges.")
    if partial_share > 0.3:
        recs.append("Offer flexible plans or payment date shifts to align with income cycles for partial payers.")
    if late_share > 0.3:
        recs.append("Test nudges before due dates for borrowers who are consistently 1–3 days late.")
    if any("Autopay failure" in c for c in confusions_all):
        recs.append("Prompt customers with failed autopay to re-confirm details and set a safety minimum.")
    if any("Minimum vs statement balance" in c for c in confusions_all):
        recs.append("Clarify minimum vs statement balance in-product for this segment.")
    if not recs:
        recs.append("Maintain current treatment strategy; behaviour appears stable and low-risk.")
    return recs[:4]


# -------------------------------------------------------
# Sidebar: model thresholds
# -------------------------------------------------------
st.sidebar.title("Model thresholds")
st.sidebar.caption("Tune how the layer interprets repayment behaviour across your portfolio.")

st.sidebar.subheader("Partial payments (intent band)")
partial_min_pct, partial_max_pct = st.sidebar.slider(
    "Treat a payment as 'partial but intentful' when it covers between…",
    min_value=10,
    max_value=110,
    value=(40, 95),   # 40%–95% by default
    step=5,
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
)

fee_to_paid_threshold_pct = st.sidebar.slider(
    "Minimum fee share of payment in 2 consecutive cycles (%)",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
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

# Confusion tags per event
df["confusions"] = df.apply(detect_confusions, axis=1)

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
# Layout: Tabs
# -------------------------------------------------------
tab1, tab2 = st.tabs(["Portfolio Intelligence", "Borrower Story Demo"])

# -------------------------------------------------------
# TAB 1: Portfolio Intelligence (Bank View)
# -------------------------------------------------------
with tab1:
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

    st.markdown("### 3. Zoom in on one borrower")

    b_list = sorted(df["borrower_id"].unique())
    sel_b = st.selectbox("Pick a borrower to see their timeline", b_list, key="bank_view_borrower")

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

# -------------------------------------------------------
# TAB 2: Borrower Story Demo (Fintech UX)
# -------------------------------------------------------
with tab2:
    st.title("Borrower Repayment Story (Demo)")

    # In a real app, this would be the logged-in user; here we pick a sample borrower.
    b_list_story = sorted(df["borrower_id"].unique())
    sel_story_borrower = st.selectbox("Choose a sample borrower profile", b_list_story, key="story_view_borrower")

    user_df = df[df["borrower_id"] == sel_story_borrower].sort_values("cycle_start_date")
    status_label, status_color, status_blurb = borrower_status(user_df)
    persona = borrower_persona(user_df)

    # Confusions across all events
    all_confusions = [c for sub in user_df["confusions"] for c in (sub if isinstance(sub, list) else [])]

    # Apple-style card section
    st.markdown("#### Repayment health at a glance")

    card = st.container()
    with card:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #ffffff 0%, #f5f7ff 40%, #f0f9f5 100%);
                border-radius: 18px;
                padding: 20px 24px;
                border: 1px solid #e6e9f2;
                ">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:18px; font-weight:600; margin-bottom:4px;">{status_label}</div>
                        <div style="font-size:13px; color:#4a4a4a; max-width:380px;">{status_blurb}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:#999;">Persona</div>
                        <div style="font-size:14px; font-weight:600; margin-bottom:8px;">{persona}</div>
                        <div style="width:120px; height:10px; border-radius:999px; background:#f1f1f1; overflow:hidden; margin-left:auto;">
                            <div style="width:100%; height:100%; background:{status_color}; opacity:0.85;"></div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Summary chips
    last_6 = user_df.tail(min(6, len(user_df)))
    on_time_full = ((last_6["days_late"] == 0) & (last_6["pay_ratio"] >= 0.95)).sum()
    on_time_partial = ((last_6["days_late"] == 0) & (last_6["is_partial_payment"] == 1)).sum()
    late_6 = (last_6["days_late"] > 0).sum()
    fee_6 = (last_6["fee_assessed"] > 0).sum()

    st.markdown("#### Last few months in numbers")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("On-time & full", f"{on_time_full}")
    c2.metric("On-time & partial", f"{on_time_partial}")
    c3.metric("Late cycles", f"{late_6}")
    c4.metric("Cycles with fees", f"{fee_6}")

    # Story text
    st.markdown("#### Repayment story")
    story_text = borrower_story(user_df, all_confusions)
    st.write(story_text)

    # Confusion indicators
    st.markdown("#### Possible confusion signals")
    if all_confusions:
        for c in sorted(set(all_confusions)):
            st.markdown(f"- {c}")
    else:
        st.markdown("_No clear confusion signals detected from notes._")

    # Recommendations
    st.markdown("#### Recommended next steps (for product / servicing strategy)")
    recs = borrower_recommendations(user_df, all_confusions)
    for r in recs:
        st.markdown(f"- {r}")

    # Simple chart
    st.markdown("#### Timeline of bills, payments and fees")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(
        user_df["cycle_start_date"],
        user_df["amount_due"],
        label="Bill amount",
    )
    ax2.plot(
        user_df["cycle_start_date"],
        user_df["amount_paid"],
        label="Amount paid",
    )
    ax2.scatter(
        user_df["cycle_start_date"],
        user_df["fee_assessed"],
        marker="x",
        s=60,
        label="Fee charged in that cycle",
    )
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("Amount")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)

    st.caption(
        "This view reuses the same repayment insights engine but presents a clear, narrative story for a single borrower."
    )
