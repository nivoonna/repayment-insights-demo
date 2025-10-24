
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import altair as alt
import matplotlib.pyplot as plt

st.set_page_config(page_title="Repayment Insights Layer — Demo", layout="wide")

@st.cache_data
def load_data(default_path=None, uploaded=None):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif default_path is not None:
        df = pd.read_csv(default_path)
    else:
        st.stop()
    # Basic hygiene
    for col in ["cycle_start_date","due_date","statement_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    # Features
    df["pay_ratio"] = (df["amount_paid"] / df["amount_due"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["fee_to_paid"] = df.apply(lambda r: r["fee_assessed"]/max(r["amount_paid"], 1e-6), axis=1)
    df.sort_values(["borrower_id","cycle_start_date"], inplace=True)
    return df

def label_partials(df, min_ratio=0.30, max_ratio=0.95):
    return ((df["pay_ratio"] >= min_ratio) & (df["pay_ratio"] < max_ratio)).astype(int)

def label_spirals(df, min_fees_last3=3, consec_fee_paid_threshold=0.2):
    def _per_borrower(g):
        g = g.sort_values("cycle_start_date").copy()
        g["fees_rolling3"] = g["fee_assessed"].rolling(3, min_periods=1).apply(lambda x: (x>0).sum(), raw=True)
        g["fee_to_paid_prev"] = g["fee_to_paid"].shift(1).fillna(0)
        g["spiral_flag"] = (
            (g["fees_rolling3"] >= min_fees_last3) |
            ((g["fee_to_paid"] > consec_fee_paid_threshold) & (g["fee_to_paid_prev"] > consec_fee_paid_threshold))
        ).astype(int)
        return g
    return df.groupby("borrower_id", group_keys=False).apply(_per_borrower)

def explain_row(row, partial_flag):
    text = str(row.get("statement_text","")).lower()
    if "autopay failed" in text or "insufficient" in text:
        return "Autopay failure likely caused the delay/fee; ensure funds & retry."
    if "hardship" in text:
        return "Hardship plan active; reduced payment accepted without fees."
    if "dispute" in text or "posted late" in text or "duplicated" in text:
        return "Fee dispute—manual review recommended."
    if partial_flag and row["days_late"] == 0:
        return "Partial on-time payment—signal of intent; consider nudge to full."
    if row["days_late"] > 0 and row["fee_assessed"] > 0:
        return "Late with fee—watch for penalty spiral if repeated."
    if row["pay_ratio"] >= 0.95:
        return "On-time full or near-full payment."
    return "Mixed behavior; monitor next cycle."

# ---------------- Sidebar Controls ----------------
st.sidebar.header("Settings")
st.sidebar.caption("Tune definitions to your taste and watch the insights update.")

partial_min = st.sidebar.slider("Partial payment minimum ratio", 0.0, 1.0, 0.30, 0.01)
partial_max = st.sidebar.slider("Partial payment maximum ratio", 0.0, 1.0, 0.95, 0.01)
min_fees_last3 = st.sidebar.slider("Spiral: min fee cycles in last 3", 1, 3, 3, 1)
consec_fee_paid_threshold = st.sidebar.slider("Spiral: fee/paid consecutive threshold", 0.0, 1.0, 0.20, 0.01)

uploaded = st.sidebar.file_uploader("Upload repayment CSV", type=["csv"])
default_path = "repayment_insights_synthetic.csv"

df = load_data(default_path=default_path, uploaded=uploaded)
df["is_partial_payment"] = label_partials(df, partial_min, partial_max)
df = label_spirals(df, min_fees_last3, consec_fee_paid_threshold)

# Borrower aggregates
agg = df.groupby("borrower_id").agg(
    cycles=("borrower_id","size"),
    mean_pay_ratio=("pay_ratio","mean"),
    partial_rate=("is_partial_payment","mean"),
    late_rate=("days_late", lambda x: np.mean(x > 0)),
    mean_days_late=("days_late","mean"),
    fee_rate=("fee_assessed", lambda x: np.mean(x > 0)),
    mean_fee=("fee_assessed","mean"),
    spiral_rate=("spiral_flag","mean"),
    hardship_any=("hardship_flag","max"),
    apr=("apr","mean"),
).reset_index().fillna(0)

# Clustering
features = agg[["mean_pay_ratio","partial_rate","late_rate","fee_rate","mean_fee","spiral_rate"]].values
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
agg["cluster"] = kmeans.fit_predict(features)

# Anomaly detection
anomaly_features = df[["pay_ratio","days_late","fee_assessed","fee_to_paid"]].copy()
iso = IsolationForest(n_estimators=200, contamination=0.08, random_state=42)
df["anomaly_score"] = iso.fit_predict(anomaly_features)  # -1 anomaly, 1 normal
df["is_anomaly"] = (df["anomaly_score"] == -1).astype(int)

# Explanations
df["explanation"] = [
    explain_row(r, partial_flag) for r, partial_flag in zip(df.to_dict("records"), df["is_partial_payment"])
]

# ---------------- Header ----------------
st.title("Repayment Insights Layer — Demo")
st.caption("Clustering partials, catching penalty spirals, translating statements into clear explanations.")

colA, colB, colC, colD = st.columns(4)
colA.metric("Borrowers", f"{agg['borrower_id'].nunique():,}")
colB.metric("Events", f"{len(df):,}")
colC.metric("Any partials", f"{int((agg['partial_rate']>0).sum()):,}")
colD.metric("Any spiral signals", f"{int((agg['spiral_rate']>0).sum()):,}")

tab1, tab2, tab3 = st.tabs(["Cohorts", "Alerts", "Explain"])

# ---------------- Tab 1: Cohorts ----------------
with tab1:
    st.subheader("Borrower Cohorts")
    scat_df = agg.copy()
    scat_df["cluster"] = scat_df["cluster"].astype(str)
    chart = alt.Chart(scat_df).mark_circle(size=80).encode(
        x=alt.X("mean_pay_ratio:Q", title="Mean pay ratio"),
        y=alt.Y("late_rate:Q", title="Late-rate (share of cycles late)"),
        color=alt.Color("cluster:N", title="Cluster"),
        tooltip=["borrower_id","mean_pay_ratio","partial_rate","late_rate","fee_rate","spiral_rate","cluster"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    st.write("**Cluster summaries**")
    cluster_summary = agg.groupby("cluster").agg(
        borrowers=("borrower_id","nunique"),
        mean_pay_ratio=("mean_pay_ratio","mean"),
        partial_rate=("partial_rate","mean"),
        late_rate=("late_rate","mean"),
        fee_rate=("fee_rate","mean"),
        spiral_rate=("spiral_rate","mean")
    ).round(3).reset_index()
    st.dataframe(cluster_summary)

    chosen_cluster = st.selectbox("Inspect a cluster", sorted(agg["cluster"].unique()))
    st.dataframe(agg[agg["cluster"]==chosen_cluster].sort_values("mean_pay_ratio"))

# ---------------- Tab 2: Alerts ----------------
with tab2:
    st.subheader("Penalty Spiral & Anomaly Alerts")
    # Controls
    min_days_late = st.slider("Minimum days late (filter)", 0, int(df["days_late"].max()), 2, 1)
    min_fee_to_paid = st.slider("Minimum fee-to-paid ratio (filter)", 0.0, float(df["fee_to_paid"].replace(np.inf, np.nan).max()), 0.2, 0.01)

    watch = df[(df["is_anomaly"]==1) | (df["spiral_flag"]==1)].copy()
    watch = watch[(watch["days_late"] >= min_days_late) | (watch["fee_to_paid"] >= min_fee_to_paid)]
    watch["fee_paid_pct"] = (watch["fee_to_paid"] * 100).round(1)
    cols = ["borrower_id","cycle_start_date","amount_due","amount_paid","days_late","fee_assessed","fee_paid_pct","explanation"]
    st.dataframe(watch[cols].sort_values(["fee_paid_pct","days_late"], ascending=False).head(100))

    # Borrower timeline
    b_list = sorted(df["borrower_id"].unique())
    sel_b = st.selectbox("View borrower timeline", b_list)
    t = df[df["borrower_id"]==sel_b].sort_values("cycle_start_date")
    st.write(f"APR ~ {t['apr'].mean():.2%} | Spiral flags in {int(t['spiral_flag'].sum())}/{len(t)} cycles | Partial cycles: {int(t['is_partial_payment'].sum())}")

    fig = plt.figure(figsize=(10,4))
    plt.plot(t["cycle_start_date"], t["amount_due"], label="Amount due")
    plt.plot(t["cycle_start_date"], t["amount_paid"], label="Amount paid")
    plt.scatter(t["cycle_start_date"], t["fee_assessed"], label="Fees", marker="x")
    plt.legend()
    plt.title(f"Borrower {sel_b} — amounts & fees over time")
    plt.xlabel("Cycle start")
    plt.ylabel("Amount")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.write("Recent explanations")
    st.write(t[["cycle_start_date","pay_ratio","days_late","fee_assessed","explanation"]].tail(10))

# ---------------- Tab 3: Explain ----------------
with tab3:
    st.subheader("Translate Statements → Plain English")
    q = st.text_input("Search statement text or explanations")
    view = df.copy()
    if q:
        ql = q.lower()
        view = view[view["statement_text"].str.lower().str.contains(ql) | view["explanation"].str.lower().str.contains(ql)]
    view_cols = ["borrower_id","cycle_start_date","statement_text","explanation","amount_due","amount_paid","days_late","fee_assessed"]
    st.dataframe(view[view_cols].sort_values("cycle_start_date").head(200))

    st.download_button("Download explanations CSV", data=view[view_cols].to_csv(index=False), file_name="repayment_explanations.csv", mime="text/csv")

st.markdown("---")
st.caption("Tip: Users like us shouldn't need a collections playbook to understand repayment. Clear cohorts, early spiral warnings, and plain-language explanations make it friendly.")
