import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import feedback_core as fc


def inject_app_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

          html, body, [class*="css"] {
            font-family: 'Outfit', system-ui, sans-serif;
          }

          /* ── Page shell ─────────────────────────────── */
          .block-container {
            padding-top: 0 !important;
            max-width: 1280px;
          }

          /* ── Hero banner ────────────────────────────── */
          .hero-wrap {
            background: #0a0a0f;
            border-bottom: 1px solid rgba(255,255,255,0.07);
            padding: 2rem 2.25rem 1.75rem;
            margin: -1rem -2rem 2rem;
            position: relative;
            overflow: hidden;
          }
          .hero-wrap::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
              radial-gradient(ellipse 60% 70% at 80% 50%, rgba(99,102,241,0.18) 0%, transparent 65%),
              radial-gradient(ellipse 40% 60% at 10% 80%, rgba(20,184,166,0.12) 0%, transparent 60%);
            pointer-events: none;
          }
          .hero-eyebrow {
            font-family: 'DM Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.18em;
            color: #6366f1;
            text-transform: uppercase;
            margin: 0 0 0.5rem;
          }
          .hero-title {
            font-family: 'Syne', sans-serif;
            font-size: 2.1rem;
            font-weight: 800;
            color: #f8fafc;
            margin: 0 0 0.55rem;
            line-height: 1.15;
            letter-spacing: -0.03em;
          }
          .hero-sub {
            font-size: 0.95rem;
            color: rgba(248,250,252,0.55);
            margin: 0;
            max-width: 540px;
            line-height: 1.6;
          }

          /* ── Sidebar ─────────────────────────────────── */
          [data-testid="stSidebar"] {
            background: #0d1117 !important;
            border-right: 1px solid rgba(255,255,255,0.06) !important;
          }
          [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
          [data-testid="stSidebar"] h3 {
            font-family: 'Syne', sans-serif !important;
            font-size: 0.75rem !important;
            letter-spacing: 0.14em !important;
            text-transform: uppercase !important;
            color: #6366f1 !important;
            margin-bottom: 0.5rem !important;
          }
          [data-testid="stSidebar"] .stButton > button {
            background: #6366f1 !important;
            color: #fff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            width: 100% !important;
          }
          [data-testid="stSidebar"] .stTextInput input,
          [data-testid="stSidebar"] .stNumberInput input {
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            color: #f1f5f9 !important;
            border-radius: 8px !important;
          }
          [data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.08) !important;
          }

          /* ── Metric cards ────────────────────────────── */
          div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            position: relative;
            overflow: hidden;
          }
          div[data-testid="stMetric"]::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #6366f1, #14b8a6);
          }
          div[data-testid="stMetric"] label {
            font-family: 'DM Mono', monospace !important;
            font-size: 0.72rem !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            color: #64748b !important;
          }
          div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-family: 'Syne', sans-serif !important;
            font-size: 1.7rem !important;
            font-weight: 700 !important;
            color: #0f172a !important;
          }

          /* ── Workflow status bar ─────────────────────── */
          .status-bar {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 1rem;
            background: #f1f5f9;
            border-radius: 10px;
            margin-bottom: 1.25rem;
            border: 1px solid #e2e8f0;
          }
          .step-node {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.82rem;
            font-weight: 500;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
          }
          .step-done { background: #dcfce7; color: #15803d; }
          .step-wait { background: #fef3c7; color: #92400e; }
          .step-idle { background: #e0e7ff; color: #3730a3; }
          .step-sep { color: #94a3b8; font-size: 0.9rem; }
          .dot { width: 7px; height: 7px; border-radius: 50%; }
          .dot-done { background: #16a34a; }
          .dot-wait { background: #d97706; }
          .dot-idle { background: #4f46e5; }

          /* ── Section headings ────────────────────────── */
          h1, h2, h3 {
            font-family: 'Syne', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            color: #0f172a !important;
          }
          h3 { font-size: 1rem !important; }

          /* ── Tabs ────────────────────────────────────── */
          .stTabs [data-baseweb="tab-list"] {
            border-bottom: 2px solid #e2e8f0 !important;
            gap: 0.25rem !important;
          }
          .stTabs [data-baseweb="tab"] {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            padding: 0.5rem 1.1rem !important;
            border-radius: 8px 8px 0 0 !important;
            color: #64748b !important;
            border: none !important;
          }
          .stTabs [aria-selected="true"] {
            color: #6366f1 !important;
            border-bottom: 2px solid #6366f1 !important;
            background: rgba(99,102,241,0.06) !important;
          }

          /* ── Buttons ─────────────────────────────────── */
          .stButton > button {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            border: 1.5px solid #e2e8f0 !important;
            transition: all 0.15s ease !important;
          }
          .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
            color: #fff !important;
            border: none !important;
            box-shadow: 0 2px 8px rgba(99,102,241,0.3) !important;
          }
          .stButton > button[kind="primary"]:hover {
            box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important;
            transform: translateY(-1px) !important;
          }

          /* ── File uploader ───────────────────────────── */
          [data-testid="stFileUploader"] {
            border: 2px dashed #c7d2fe !important;
            border-radius: 12px !important;
            background: #f8f9ff !important;
            padding: 0.5rem !important;
          }

          /* ── Expanders ───────────────────────────────── */
          [data-testid="stExpander"] {
            border: 1px solid #e2e8f0 !important;
            border-radius: 10px !important;
          }
          [data-testid="stExpander"] summary {
            font-weight: 500 !important;
            font-size: 0.9rem !important;
          }

          /* ── Dataframes ──────────────────────────────── */
          [data-testid="stDataFrame"] {
            border-radius: 10px !important;
            overflow: hidden !important;
            border: 1px solid #e2e8f0 !important;
          }

          /* ── Alerts ──────────────────────────────────── */
          [data-testid="stAlert"] {
            border-radius: 10px !important;
          }

          /* ── Section divider ─────────────────────────── */
          .section-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 1.75rem 0 1rem;
          }
          .section-header-line {
            flex: 1;
            height: 1px;
            background: #e2e8f0;
          }
          .section-header-label {
            font-family: 'DM Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: #94a3b8;
            white-space: nowrap;
          }

          /* ── Minority section ────────────────────────── */
          .minority-banner {
            background: linear-gradient(135deg, #fdf4ff 0%, #ede9fe 100%);
            border: 1px solid #ddd6fe;
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
          }
          .minority-banner h4 {
            font-family: 'Syne', sans-serif !important;
            font-size: 1rem !important;
            color: #4c1d95 !important;
            margin: 0 0 0.35rem !important;
          }
          .minority-banner p {
            font-size: 0.85rem;
            color: #6d28d9;
            margin: 0;
          }

          /* ── Download buttons ────────────────────────── */
          .stDownloadButton > button {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            border-radius: 8px !important;
            border: 1.5px solid #e2e8f0 !important;
            color: #334155 !important;
          }
          .stDownloadButton > button:hover {
            border-color: #6366f1 !important;
            color: #6366f1 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def workflow_pills(model_loaded: bool, has_csv: bool, has_preds: bool) -> None:
    def node(state: str, label: str) -> str:
        cls = {"done": "step-done", "wait": "step-wait", "idle": "step-idle"}[state]
        dot = {"done": "dot-done", "wait": "dot-wait", "idle": "dot-idle"}[state]
        return f'<span class="step-node {cls}"><span class="dot {dot}"></span>{label}</span>'

    m = node("done" if model_loaded else "wait", "Model")
    u = node("done" if has_csv else ("wait" if model_loaded else "idle"), "CSV uploaded")
    p = node("done" if has_preds else ("wait" if has_csv else "idle"), "Predictions ready")

    st.markdown(
        f'<div class="status-bar">{m}<span class="step-sep">›</span>{u}<span class="step-sep">›</span>{p}</div>',
        unsafe_allow_html=True,
    )


def section_divider(label: str) -> None:
    st.markdown(
        f"""<div class="section-header">
          <span class="section-header-line"></span>
          <span class="section-header-label">{label}</span>
          <span class="section-header-line"></span>
        </div>""",
        unsafe_allow_html=True,
    )


def reset_prediction_minority_state() -> None:
    for k in (
        "pred_df",
        "minority_df",
        "show_minority_table",
        "show_all_predictions",
        "detected_text_cols",
        "segment_col_auto",
    ):
        st.session_state.pop(k, None)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Feedback Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_app_styles()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-wrap">
      <p class="hero-eyebrow">◈ Student analytics</p>
      <h1 class="hero-title">Feedback Intelligence</h1>
      <p class="hero-sub">
        Classify free-text feedback, surface sentiment trends, and detect minority patterns
        that deserve a closer look — all in one pipeline.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session defaults ───────────────────────────────────────────────────────────
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "show_minority_table" not in st.session_state:
    st.session_state.show_minority_table = False
if "show_all_predictions" not in st.session_state:
    st.session_state.show_all_predictions = False
if "model_dir_input" not in st.session_state:
    st.session_state.model_dir_input = os.environ.get("FEEDBACK_MODEL_DIR", "final_feedback_classifier").strip() or "final_feedback_classifier"

# Auto-load when FEEDBACK_MODEL_DIR is set (e.g. Streamlit Cloud / Docker env).
if (
    not st.session_state.model_loaded
    and not st.session_state.get("_autoloaded_env_model")
    and os.environ.get("FEEDBACK_MODEL_DIR", "").strip()
):
    st.session_state._autoloaded_env_model = True
    try:
        auto_path = fc.resolve_classifier_dir(None)
        if auto_path.is_dir() and list(auto_path.glob("config.json")):
            tok, mdl, meta = fc.load_classifier(auto_path)
            st.session_state.tokenizer = tok
            st.session_state.model = mdl
            st.session_state.metadata = meta
            st.session_state.model_loaded = True
            st.session_state.model_dir_input = str(auto_path)
            reset_prediction_minority_state()
    except Exception:
        pass

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⬡ Model")
    model_dir = st.text_input(
        "Model folder",
        key="model_dir_input",
        help="Relative folder in the repo, or an absolute path. On deploy, set env FEEDBACK_MODEL_DIR or ship this folder in Git (see README).",
        label_visibility="collapsed",
        placeholder="Model folder path…",
    )
    if st.button("Load model", type="primary"):
        model_path = fc.resolve_classifier_dir(model_dir)
        if not model_path.is_dir():
            st.error("That folder path does not exist.")
            st.markdown(
                """
**Why this happens on Streamlit Cloud:** the trained weights live in
`final_feedback_classifier/` on your laptop, but that folder is usually **not**
in GitHub (large files / `.gitignore`), so the cloud app has nothing to load.

**Fix (pick one):**

1. **Commit the model** into the repo (use [Git LFS](https://git-lfs.github.com/) for `*.safetensors`), **or**
2. In Streamlit **App settings → Environment variables**, set  
   `FEEDBACK_MODEL_DIR` to an **absolute** path if your host mounts the model, **or**
3. Run locally / on a VPS where you copied `final_feedback_classifier/`.

See **`DEPLOY.md`** in the repository for Docker, Reflex, and Streamlit model options.
                """.strip(),
            )
        elif not list(model_path.glob("config.json")):
            st.error("No Transformer model found (missing config.json).")
        else:
            try:
                tok, mdl, meta = fc.load_classifier(model_path)
                st.session_state.tokenizer = tok
                st.session_state.model = mdl
                st.session_state.metadata = meta
                st.session_state.model_loaded = True
                reset_prediction_minority_state()
                st.success("✓ Model loaded successfully.")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.markdown("### ⬡ Minority detection")
    emb_default = st.session_state.get("metadata", {}).get("embedding_model", "all-MiniLM-L6-v2")
    cont_default = float(st.session_state.get("metadata", {}).get("contamination", 0.08))
    min_cl_default = int(st.session_state.get("metadata", {}).get("min_cluster_size", 10))

    embedding_model_sidebar = st.text_input(
        "Embedding model", emb_default, label_visibility="visible"
    )
    contamination_sidebar = st.slider(
        "IsolationForest contamination", 0.02, 0.25, min(cont_default, 0.25), 0.01
    )
    min_cluster_size_sidebar = st.number_input(
        "DBSCAN minority cluster max size",
        min_value=3,
        max_value=500,
        value=min(min_cl_default, 50),
    )

    st.divider()
    st.caption(
        "Deploy via [Streamlit Community Cloud](https://streamlit.io/cloud) "
        "with `streamlit run app.py`."
    )

# ── Gate: model not loaded ─────────────────────────────────────────────────────
if not st.session_state.model_loaded:
    workflow_pills(False, False, False)
    st.info(
        "**Get started:** set the model folder in the sidebar → click **Load model**. "
        "Run `studentfeedback_analysis.py` first if you need to create `final_feedback_classifier/`. "
        "On **Streamlit Cloud**, that folder must exist in the deployed repo or set **`FEEDBACK_MODEL_DIR`** "
        "(see `DEPLOY.md`). Then upload your CSV — text columns are detected automatically."
    )
    st.stop()

workflow_pills(True, False, "pred_df" in st.session_state)

# ── CSV upload ─────────────────────────────────────────────────────────────────
section_divider("Data source")

upload_col, info_col = st.columns([3, 1])
with upload_col:
    uploaded = st.file_uploader("Upload a CSV", type=["csv"], label_visibility="collapsed")
with info_col:
    st.markdown(
        """<div style="padding:0.75rem 0; color:#64748b; font-size:0.82rem; line-height:1.6;">
        Text columns are <strong>auto-detected</strong>.<br>
        Segment columns are inferred from column names.
        </div>""",
        unsafe_allow_html=True,
    )

if not uploaded:
    st.warning("Upload a CSV file to continue.")
    st.stop()

file_key = getattr(uploaded, "name", None) or str(id(uploaded))
if st.session_state.get("_upload_key") != file_key:
    st.session_state._upload_key = file_key
    reset_prediction_minority_state()

df = pd.read_csv(uploaded)
meta_text = st.session_state.metadata.get("text_cols") or []
selected_text_cols = fc.auto_detect_text_columns(df, meta_text)
segment_col = fc.auto_pick_segment_column(df, selected_text_cols)
st.session_state.detected_text_cols = selected_text_cols
st.session_state.segment_col_auto = segment_col

workflow_pills(True, True, "pred_df" in st.session_state)

# ── Dataset summary ────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Rows", f"{len(df):,}")
with m2:
    st.metric("Columns", len(df.columns))
with m3:
    st.metric("Text cols (auto)", len(selected_text_cols))
with m4:
    st.metric("Segment col", "✓ found" if segment_col else "none")

with st.expander("Detected column mapping", expanded=True):
    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f"""
            <div style="display:flex;flex-direction:column;gap:0.75rem;padding:0.5rem 0;">
              <div>
                <div style="font-size:0.72rem;font-family:'DM Mono',monospace;letter-spacing:0.1em;text-transform:uppercase;color:#94a3b8;margin-bottom:0.25rem;">Text columns</div>
                <div style="font-size:0.88rem;color:#1e293b;font-weight:500;">{", ".join(selected_text_cols) or "—  none found, check CSV"}</div>
              </div>
              <div>
                <div style="font-size:0.72rem;font-family:'DM Mono',monospace;letter-spacing:0.1em;text-transform:uppercase;color:#94a3b8;margin-bottom:0.25rem;">Segment column</div>
                <div style="font-size:0.88rem;color:#1e293b;font-weight:500;">{segment_col if segment_col else "—  using confidence split"}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.dataframe(df.head(8), use_container_width=True, height=240)

if not selected_text_cols:
    st.error(
        "Could not auto-detect feedback text columns. "
        "Please upload a CSV that contains at least one narrative/comment-style column."
    )
    st.stop()

combined_text = fc.combined_text_series(df, selected_text_cols)
id2label = {int(k): v for k, v in st.session_state.metadata["id2label"].items()}

# ── Run predictions ────────────────────────────────────────────────────────────
section_divider("Classification")

run_col, hint_col = st.columns([2, 3])
with run_col:
    run_predictions = st.button("▶  Run classifier predictions", type="primary", use_container_width=True)
with hint_col:
    st.markdown(
        "<p style='font-size:0.83rem;color:#64748b;padding:0.55rem 0;'>Runs the loaded transformer model across all rows. Results populate the charts below.</p>",
        unsafe_allow_html=True,
    )

if run_predictions:
    st.session_state.show_all_predictions = False
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model
    with st.spinner("Running classifier…"):
        output_df = fc.predict_dataframe(df, combined_text, tokenizer, model, id2label)
    st.session_state.pred_df = output_df
    st.success("✓ Predictions complete.")

if "pred_df" not in st.session_state:
    st.info("Click **▶ Run classifier predictions** to populate charts, segment views, and downloads.")
    st.stop()

pred_df = st.session_state.pred_df
workflow_pills(True, True, True)

# ── Results tabs ───────────────────────────────────────────────────────────────
section_divider("Results")

tab_overview, tab_segment, tab_compare = st.tabs(["Overview", "By segment", "Compare groups"])

with tab_overview:
    counts = pred_df["prediction"].value_counts()
    avg_conf = float(pred_df["confidence"].mean())

    ov_c1, ov_c2, ov_c3 = st.columns([3, 1, 1])
    with ov_c1:
        st.markdown("**Prediction distribution**")
        st.bar_chart(counts, use_container_width=True, height=240)
    with ov_c2:
        st.metric("Mean confidence", f"{avg_conf:.3f}")
        st.metric("Distinct labels", counts.size)
    with ov_c3:
        st.metric("Total rows", f"{len(pred_df):,}")
        st.metric("High confidence", f"{int((pred_df['confidence'] >= 0.8).sum()):,}")

    act1, act2 = st.columns(2)
    with act1:
        if st.button("Show sample predictions"):
            st.session_state.show_all_predictions = True
    with act2:
        st.download_button(
            "↓  Download predictions CSV",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

    if st.session_state.show_all_predictions:
        st.caption("Showing up to 80 rows — download for the full dataset.")
        st.dataframe(pred_df.head(80), use_container_width=True)

with tab_segment:
    if not segment_col:
        st.info("No suitable segment column was auto-detected. Use the **Compare** tab with confidence split.")
    else:
        st.markdown(f"#### Breakdown by `{segment_col}`")
        vc = pred_df[segment_col].astype(str).value_counts().head(25)
        st.bar_chart(vc, use_container_width=True, height=240)

        pivot = (
            pred_df.groupby([segment_col, "prediction"], dropna=False)
            .size()
            .unstack(fill_value=0)
        )
        st.caption("Row counts per segment × predicted label")
        st.dataframe(pivot.astype(int), use_container_width=True, height=min(420, 80 + 28 * len(pivot)))

with tab_compare:
    mode = st.radio(
        "Compare by",
        ["Segment column", "Model confidence split"],
        horizontal=True,
    )

    if mode == "Segment column":
        if not segment_col:
            st.warning("No segment column auto-detected — switch to confidence split.")
        else:
            vals = sorted(pred_df[segment_col].dropna().astype(str).unique().tolist())
            if len(vals) > 60:
                top = pred_df[segment_col].astype(str).value_counts().head(60).index.tolist()
                vals = sorted(top)
                st.caption("Showing the 60 most frequent segment values.")
            sel_col1, sel_col2 = st.columns(2)
            with sel_col1:
                a = st.selectbox("Group A", vals, index=0)
            with sel_col2:
                b = st.selectbox("Group B", vals, index=min(1, len(vals) - 1))
            da = pred_df[pred_df[segment_col].astype(str) == a]
            db = pred_df[pred_df[segment_col].astype(str) == b]
            ca, cb = fc.aligned_prediction_counts(da["prediction"].value_counts(), db["prediction"].value_counts())

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(
                    f"<div style='font-size:0.85rem;font-weight:600;color:#0f172a;margin-bottom:0.5rem;'>"
                    f"{segment_col}: <span style='color:#6366f1;'>{a}</span> &nbsp;·&nbsp; {len(da):,} rows</div>",
                    unsafe_allow_html=True,
                )
                st.metric("Mean confidence", f"{float(da['confidence'].mean()):.3f}")
                st.bar_chart(ca, use_container_width=True, height=200)
            with col_b:
                st.markdown(
                    f"<div style='font-size:0.85rem;font-weight:600;color:#0f172a;margin-bottom:0.5rem;'>"
                    f"{segment_col}: <span style='color:#14b8a6;'>{b}</span> &nbsp;·&nbsp; {len(db):,} rows</div>",
                    unsafe_allow_html=True,
                )
                st.metric("Mean confidence", f"{float(db['confidence'].mean()):.3f}")
                st.bar_chart(cb, use_container_width=True, height=200)
    else:
        med = float(pred_df["confidence"].median())
        hi = pred_df[pred_df["confidence"] >= med]
        lo = pred_df[pred_df["confidence"] < med]
        ca, cb = fc.aligned_prediction_counts(hi["prediction"].value_counts(), lo["prediction"].value_counts())

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"<div style='font-size:0.85rem;font-weight:600;color:#15803d;margin-bottom:0.5rem;'>"
                f"High confidence &nbsp;≥ {med:.3f} &nbsp;·&nbsp; {len(hi):,} rows</div>",
                unsafe_allow_html=True,
            )
            st.metric("Mean confidence", f"{float(hi['confidence'].mean()):.3f}")
            st.bar_chart(ca, use_container_width=True, height=200)
        with col_b:
            st.markdown(
                f"<div style='font-size:0.85rem;font-weight:600;color:#b45309;margin-bottom:0.5rem;'>"
                f"Low confidence &nbsp;< {med:.3f} &nbsp;·&nbsp; {len(lo):,} rows</div>",
                unsafe_allow_html=True,
            )
            st.metric("Mean confidence", f"{float(lo['confidence'].mean()):.3f}")
            st.bar_chart(cb, use_container_width=True, height=200)

# ── Minority detection ─────────────────────────────────────────────────────────
section_divider("Minority pattern detection")

st.markdown(
    """<div class="minority-banner">
      <h4>What is minority detection?</h4>
      <p>Uses <strong>IsolationForest</strong> (outlier scoring) and <strong>DBSCAN</strong> (small cluster membership)
      to surface feedback rows that stand apart from the mainstream — unusual phrasing, edge-case issues, or overlooked topics.</p>
    </div>""",
    unsafe_allow_html=True,
)

include_dbscan_noise = st.checkbox(
    "Also flag DBSCAN noise points (cluster −1)",
    value=False,
    help="Increases flagged rows significantly; off by default.",
)

det_col, _ = st.columns([2, 3])
with det_col:
    minority_run = st.button("◈  Detect minority patterns", use_container_width=True)

if minority_run:
    st.session_state.show_minority_table = False
    texts = combined_text.tolist()
    with st.spinner("Embedding and clustering — may take a minute on CPU…"):
        minority_df = fc.detect_minority_patterns(
            df,
            texts,
            embedding_model_sidebar,
            contamination_sidebar,
            int(min_cluster_size_sidebar),
            include_dbscan_noise,
            pred_df,
        )
        st.session_state.minority_df = minority_df

    total = int(minority_df["is_minority_pattern"].sum())
    pct = (total / max(1, len(df))) * 100.0
    st.success(f"◈  Flagged **{total} rows** as minority-related ({pct:.2f}% of dataset).")

if "minority_df" in st.session_state:
    minority_df = st.session_state.minority_df
    minority_only = minority_df[minority_df["is_minority_pattern"]].copy()

    if segment_col and segment_col in minority_df.columns:
        st.markdown("#### Minority rate by segment")
        g = minority_df.groupby(minority_df[segment_col].astype(str), dropna=False).agg(
            rows=("is_minority_pattern", "size"),
            minority=("is_minority_pattern", "sum"),
        )
        g["minority_pct"] = (g["minority"] / g["rows"].replace(0, np.nan) * 100).round(2)
        g = g.sort_values("minority_pct", ascending=False).head(30)

        chart_col, table_col = st.columns([3, 2])
        with chart_col:
            st.bar_chart(g["minority_pct"], use_container_width=True, height=260)
        with table_col:
            st.dataframe(g, use_container_width=True, height=260)

    show_col, dl_col1, dl_col2 = st.columns([2, 1, 1])
    with show_col:
        if st.button("Show minority feedback rows", key="btn_show_minority"):
            st.session_state.show_minority_table = True
    with dl_col1:
        st.download_button(
            "↓  Minority rows only",
            data=minority_only.to_csv(index=False).encode("utf-8"),
            file_name="minority_feedback_only.csv",
            mime="text/csv",
            key="dl_minority_only",
        )
    with dl_col2:
        st.download_button(
            "↓  Full sheet + flags",
            data=minority_df.to_csv(index=False).encode("utf-8"),
            file_name="minority_patterns_detected.csv",
            mime="text/csv",
            key="dl_minority_full",
        )

    if st.session_state.show_minority_table:
        st.subheader("Minority-pattern rows")
        display_cols = [c for c in minority_only.columns if c not in ("is_noise",)]
        st.dataframe(minority_only[display_cols], use_container_width=True, height=420)