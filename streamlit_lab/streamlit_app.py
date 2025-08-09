import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import hdbscan

from sklearn.preprocessing import StandardScaler

# ================== Page & Style ==================
st.set_page_config(page_title="TradeSense Analyst Lab", layout="wide")

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] .block-container { padding-top: 0.6rem; }
div.stMetric { background: #fafafa; border: 1px solid #eee; border-radius: 12px; padding: 8px 12px; }
hr { margin: 1rem 0 1.2rem 0; }
.dataframe th, .dataframe td { font-size: 0.92rem; }
.placeholder-min { min-height: 60px; } /* reserves space to reduce "jump" */
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ================== Paths ==================
MODEL_DIR = "models"
DATA_DIR = "data"

CLUSTER_BUNDLE_PATH   = os.path.join(MODEL_DIR, "customer_cluster_model.pkl")   # bundle: {"features","scaler","clusterer"}
SENTIMENT_MODEL_PATH  = os.path.join(MODEL_DIR, "sentiment_model.pkl")          # 3-class TF-IDF LR
SENTIMENT_META_PATH   = os.path.join(MODEL_DIR, "sentiment_model_meta.json")    # optional metadata

PATHS = {
    "customer_clusters": "data/processed/customer_clusters.csv",
    "campaign_summaries": "data/processed/campaign_summaries.csv",
    "prophet": "data/forecasts/prophet_forecast.csv",
    "neural": "data/forecasts/neuralprophet_forecast.csv",
    "ensemble": "data/forecasts/ensemble_forecast.csv",
}

# ================== Loaders ==================
@st.cache_resource
def load_cluster_bundle():
    if not os.path.exists(CLUSTER_BUNDLE_PATH):
        return None
    obj = joblib.load(CLUSTER_BUNDLE_PATH)
    if isinstance(obj, dict) and {"features", "scaler", "clusterer"}.issubset(obj.keys()):
        return obj
    # Legacy estimator only -> unsupported for prediction UI
    return None

@st.cache_resource
def load_sentiment_model():
    model = joblib.load(SENTIMENT_MODEL_PATH) if os.path.exists(SENTIMENT_MODEL_PATH) else None
    meta  = None
    if os.path.exists(SENTIMENT_META_PATH):
        with open(SENTIMENT_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return model, meta

@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

cluster_bundle = load_cluster_bundle()
sent_model, sent_meta = load_sentiment_model()
dataframes = {k: load_csv(v) for k, v in PATHS.items()}

# ================== Helpers ==================
def safe_to_datetime(df, col="date"):
    if df is None or col not in df.columns: return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out

def label_text(idx: int):
    return {0: "negative", 1: "neutral", 2: "positive"}.get(int(idx), "unknown")

def sentiment_to_segment(idx: int):
    return {0: "Bronze", 1: "Silver", 2: "Gold"}.get(int(idx), "Unknown")

def predict_sentiment(text: str):
    """Return ('negative'|'neutral'|'positive', idx 0/1/2) using the loaded 3-class model."""
    proba = sent_model.predict_proba([text])[0]
    idx = int(np.argmax(proba))
    return label_text(idx), idx

def download_df_button(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")

def build_cluster_profile(df_clusters: pd.DataFrame):
    """Slightly more informative profile: counts, share, avg monetary/frequency/recency."""
    base = df_clusters.copy()
    counts = base["cluster"].value_counts(dropna=False).rename("count").sort_index()
    total = counts.sum() if counts.size else 1

    prof = base.groupby("cluster", dropna=False).agg({
        "monetary": "mean",
        "frequency": "mean",
        "recency": "mean"
    }).rename(columns={
        "monetary": "avg_monetary",
        "frequency": "avg_frequency",
        "recency": "avg_recency"
    })

    out = prof.join(counts, how="outer")
    out["share_%"] = (out["count"] / total * 100).round(2)
    return out.reset_index()

def simulate_promo(df_clusters, target_cluster, delta_pct):
    """Return modified copy with monetary increased by delta_pct for selected cluster only."""
    out = df_clusters.copy()
    mask = out["cluster"] == target_cluster
    out.loc[mask, "monetary"] = out.loc[mask, "monetary"] * (1.0 + delta_pct / 100.0)
    return out

def style_cmp_table(cmp_df: pd.DataFrame, target_cluster: int):
    """Highlight only the selected cluster row, color Δ for that row (green/red)."""
    df = cmp_df.copy()
    if "delta" not in df.columns:
        df["delta"] = df["avg_monetary_after"] - df["avg_monetary_before"]

    def row_style(row):
        styles = ['' for _ in row.index]
        if row.name == target_cluster:
            styles = ['background-color: #f7fbff' for _ in row.index]
            try:
                j = list(row.index).index("delta")
                if row["delta"] > 0:
                    styles[j] += '; color: #1b5e20; font-weight:600'
                elif row["delta"] < 0:
                    styles[j] += '; color: #b71c1c; font-weight:600'
            except Exception:
                pass
        return styles

    return (
        df.round(2)
          .style
          .apply(row_style, axis=1)
          .format({"avg_monetary_before": "{:,.2f}", "avg_monetary_after": "{:,.2f}", "delta": "{:,.2f}"})
    )

def predict_cluster_from_inputs(inputs: dict):
    """Predict cluster using bundle -> (cluster_id, strength). Returns (-1, 0.0) if unassigned."""
    if not cluster_bundle:
        return None, None
    feats     = cluster_bundle["features"]
    scaler    = cluster_bundle["scaler"]
    clusterer = cluster_bundle["clusterer"]

    vec_df = pd.DataFrame([[float(inputs.get(f, 0.0)) for f in feats]], columns=feats)
    Xs = scaler.transform(vec_df)

    # module-level function (prevents attribute errors)
    labels, strengths = hdbscan.approximate_predict(clusterer, Xs)
    return int(labels[0]), float(strengths[0])

# ================== UI ==================
st.title("TradeSense Analyst Lab")

# --- Radio-based tab control (works on all Streamlit versions, no jump) ---
tab_names = [
    "1) Forecast & Clusters",
    "2) Review → Sentiment & Campaigns",
    "3) Model Edge-Case Testing",
]
sel = st.radio("Section", tab_names, index=0, key="tabs_control", horizontal=True)

# -------------------------------------------------
# TAB 1 — Forecast & Clusters (+ Promo Simulation)
# -------------------------------------------------
if sel == tab_names[0]:
    st.markdown('<div class="placeholder-min"></div>', unsafe_allow_html=True)
    st.subheader("📈 Forecast Explorer")

    # SKU selector (if present)
    skus = set()
    for key in ["prophet", "neural", "ensemble"]:
        dfk = dataframes[key]
        if dfk is not None and "sku" in dfk.columns:
            skus.update(dfk["sku"].dropna().unique().tolist())
    skus = sorted(list(skus))
    sel_sku = st.selectbox("Select SKU (if available)", ["(all)"] + skus, key="t1_sku") if skus else "(all)"

    chart_cols = st.columns(3)
    for (name, key), col in zip([("Prophet","prophet"),("NeuralProphet","neural"),("Ensemble","ensemble")], chart_cols):
        dfk = dataframes[key]
        with col:
            st.markdown(f"**{name}**")
            if dfk is None:
                st.info(f"Upload missing file: {PATHS[key]}")
            else:
                plot_df = safe_to_datetime(dfk, "date").copy()
                if sel_sku != "(all)" and "sku" in plot_df.columns:
                    plot_df = plot_df[plot_df["sku"] == sel_sku]
                y_col = None
                for cand in ["forecast_units","prophet_units","neural_units","ensemble_units"]:
                    if cand in plot_df.columns:
                        y_col = cand; break
                if not y_col:
                    st.warning("No forecast column found.")
                else:
                    plot_df = plot_df.dropna(subset=["date"]).sort_values("date")
                    st.line_chart(plot_df.set_index("date")[y_col], height=220, use_container_width=True)

    st.markdown("---")
    st.subheader("👥 Cluster Summary")

    clusters_df = dataframes["customer_clusters"]
    if clusters_df is None:
        st.info("Upload **data/processed/customer_clusters.csv** to enable cluster summaries and simulations.")
    else:
        # Key metrics up top
        noise_ratio = (clusters_df["cluster"] == -1).mean() if "cluster" in clusters_df.columns else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers", f"{len(clusters_df):,}")
        c2.metric("Unique clusters", f"{clusters_df['cluster'].nunique():,}")
        c3.metric("Noise (-1) share", f"{noise_ratio*100:.1f}%")

        # Slightly more informative table
        profile_df = build_cluster_profile(clusters_df)
        st.write("**Cluster Profile (count, share, average monetary/frequency/recency)**")
        st.dataframe(
            profile_df.style.format({
                "avg_monetary":"{:,.2f}",
                "avg_frequency":"{:,.2f}",
                "avg_recency":"{:,.2f}",
                "share_%":"{:.2f}"
            }),
            use_container_width=True
        )

        # ---------- Promo Simulation ----------
        st.markdown("---")
        st.subheader("Promo Simulation (What-if)")

        left, right = st.columns([1, 2])

        with left:
            cluster_list = sorted(clusters_df["cluster"].dropna().unique().tolist())
            target_cluster = st.selectbox("Target cluster", cluster_list, key="t1_sim_cluster")
            discount_pct  = st.slider("Assumed discount impact on spend (%)", -50, 50, 10, key="t1_sim_discount")

            # compute aggregates
            before = (clusters_df.groupby("cluster")[["monetary"]]
                      .mean(numeric_only=True)
                      .rename(columns={"monetary":"avg_monetary_before"}))
            after_df = simulate_promo(clusters_df, target_cluster, discount_pct)
            after = (after_df.groupby("cluster")[["monetary"]]
                     .mean(numeric_only=True)
                     .rename(columns={"monetary":"avg_monetary_after"}))
            cmp_df = before.join(after, how="outer")
            cmp_df["delta"] = cmp_df["avg_monetary_after"] - cmp_df["avg_monetary_before"]

            if target_cluster in cmp_df.index:
                b = float(cmp_df.loc[target_cluster, "avg_monetary_before"])
                a = float(cmp_df.loc[target_cluster, "avg_monetary_after"])
                delta_abs = a - b
                delta_pct_show = (delta_abs / b * 100) if b else 0.0

                m1, m2, m3 = st.columns(3)
                m1.metric("Avg monetary (before)", f"{b:,.2f}")
                m2.metric("Avg monetary (after)",  f"{a:,.2f}", f"{delta_pct_show:+.1f}%")
                m3.metric("Δ monetary (abs.)",     f"{delta_abs:+,.2f}")

        with right:
            st.markdown("##### Cluster Averages (all clusters)")
            styled = style_cmp_table(cmp_df, target_cluster)
            st.dataframe(styled, use_container_width=True)

        # Compact visual for selected cluster
        if target_cluster in cmp_df.index:
            b = float(cmp_df.loc[target_cluster, "avg_monetary_before"])
            a = float(cmp_df.loc[target_cluster, "avg_monetary_after"])
            st.markdown("##### Before vs After (selected cluster)")
            fig, ax = plt.subplots(figsize=(4.2, 2.6), dpi=150)
            bars = ax.bar(["Before","After"], [b,a], color=["#C0C0C0","#1f77b4"])
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.set_ylabel("Average monetary"); ax.set_title(f"Cluster {target_cluster}")
            for rect in bars:
                height = rect.get_height()
                ax.annotate(f"{height:,.0f}", xy=(rect.get_x()+rect.get_width()/2, height),
                            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
            st.pyplot(fig, use_container_width=False)

        st.markdown("##### Members in selected cluster")
        st.dataframe(
            clusters_df[clusters_df["cluster"] == target_cluster]
            .sort_values("monetary", ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )

# -------------------------------------------------
# TAB 2 — Review → Sentiment & Campaigns
# -------------------------------------------------
elif sel == tab_names[1]:
    st.markdown('<div class="placeholder-min"></div>', unsafe_allow_html=True)
    st.subheader("Review → Sentiment & Matching Campaigns")

    if sent_model is None:
        st.info("Train and save **models/sentiment_model.pkl** (3-class) to enable this section.")
    else:
        # Persistent state
        ss = st.session_state
        ss.setdefault("t2_pred_sent_text", None)
        ss.setdefault("t2_target_segment", None)
        ss.setdefault("t2_matches_df", None)
        ss.setdefault("t2_top_n", 10)

        # Stable containers
        info_box     = st.container()
        result_box   = st.container()
        table_box    = st.container()
        download_box = st.container()

        review = st.text_area(
            "Enter customer review (Arabic or English):",
            height=140,
            placeholder="اكتب المراجعة هنا / Paste review here…",
            key="t2_review_text",
        )
        top_n = st.slider("Show top-N matching campaign briefs", 5, 50, ss["t2_top_n"], step=5, key="t2_topn_slider")

        def on_analyze_click():
            text = st.session_state.get("t2_review_text", "")
            if not text.strip():
                ss["t2_pred_sent_text"] = None
                ss["t2_matches_df"] = None
                return
            sent_text, sent_idx = predict_sentiment(text)
            ss["t2_pred_sent_text"] = sent_text
            ss["t2_target_segment"] = sentiment_to_segment(sent_idx)
            ss["t2_top_n"] = st.session_state.get("t2_topn_slider", 10)

            camp_df = dataframes.get("campaign_summaries")
            if camp_df is not None and "segment" in camp_df.columns:
                tmp = camp_df.copy()
                tmp["segment_ci"] = tmp["segment"].astype(str).str.lower()
                matches = tmp[tmp["segment_ci"] == ss["t2_target_segment"].lower()].copy()
                if "campaign_brief" in matches.columns:
                    matches = matches.drop_duplicates(subset=["campaign_brief"])
                if "uplift_score" in matches.columns:
                    matches["uplift_score_num"] = pd.to_numeric(matches["uplift_score"], errors="coerce")
                    matches = matches.sort_values(["uplift_score_num"], ascending=[False])
                ss["t2_matches_df"] = matches
            else:
                ss["t2_matches_df"] = None

        st.button("Analyze Review", key="t2_analyze_btn", on_click=on_analyze_click, use_container_width=False)

        # Render from session_state
        if ss["t2_pred_sent_text"] is not None:
            result_box.write(f"**Predicted Sentiment:** {ss['t2_pred_sent_text']}")
            result_box.caption("Sentiment is mapped to segment as: Negative → Bronze, Neutral → Silver, Positive → Gold.")
            result_box.caption("Only the top-N matches are shown below. Adjust the slider or download all matches.")

            matches = ss["t2_matches_df"]
            if matches is None:
                info_box.info("Optional file missing: **data/processed/campaign_summaries.csv**")
            else:
                seg = ss["t2_target_segment"]
                total = len(matches)
                info_box.success(f"Found {total} matching campaign(s) for segment: {seg}")

                show_cols = [c for c in ["segment", "uplift_score", "campaign_brief"] if c in matches.columns]
                table_box.dataframe(matches[show_cols].head(ss["t2_top_n"]), use_container_width=True)

                csv = matches[show_cols].to_csv(index=False).encode("utf-8")
                download_box.download_button(
                    "Download all matches as CSV",
                    data=csv,
                    file_name=f"campaigns_{seg.lower()}.csv",
                    mime="text/csv",
                )

# -------------------------------------------------
# TAB 3 — Model Edge-Case Testing
# -------------------------------------------------
else:
    st.markdown('<div class="placeholder-min"></div>', unsafe_allow_html=True)
    st.subheader("Model Edge-Case Testing")

    if not cluster_bundle:
        st.info("Train & save **models/customer_cluster_model.pkl** as a bundle with keys: features, scaler, clusterer.")
    else:
        feats  = cluster_bundle["features"]
        scaler = cluster_bundle["scaler"]
        clus   = cluster_bundle["clusterer"]

        st.caption("Enter customer metrics; prediction updates live. "
                   "Note: HDBSCAN may assign **-1 (noise)** when the point doesn’t belong to any dense region.")

        # Preload medians when available (from customer_clusters, else zeros)
        clusters_df = dataframes["customer_clusters"]
        defaults = {}
        for f in feats:
            if clusters_df is not None and f in clusters_df.columns:
                defaults[f] = float(clusters_df[f].median())
            else:
                defaults[f] = 0.0

        # Inputs (no button -> continuous, minimal layout changes)
        cols = st.columns(len(feats))
        inputs = {}
        for i, f in enumerate(feats):
            inputs[f] = cols[i].number_input(f, value=defaults[f], key=f"t3_{f}")

        # Predict (stable; no click jump)
        cluster_id, strength = predict_cluster_from_inputs(inputs)
        if cluster_id is None:
            st.warning("Cluster bundle is not compatible. Retrain & save a full bundle.")
        else:
            st.metric("Predicted cluster id", f"{cluster_id}")
            st.metric("Membership strength", f"{strength:.2f}")

            # Show cluster medians for context if available
            if clusters_df is not None and "cluster" in clusters_df.columns:
                ctx_cols = [c for c in feats if c in clusters_df.columns]
                med = clusters_df.groupby("cluster")[ctx_cols].median(numeric_only=True).round(1)
                st.write("**Cluster medians (context)**")
                st.dataframe(med, use_container_width=True)
