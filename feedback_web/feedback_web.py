"""Reflex UI for feedback classification and minority detection."""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import reflex as rx

import feedback_core as fc

_TOK: Any = None
_MODEL: Any = None
_METADATA: dict | None = None


def _upload_base() -> Path:
    return Path(rx.get_upload_dir())


def paths_from_run_id(run_id: str) -> tuple[Path, Path, Path, Path]:
    base = _upload_base()
    rid = run_id or "norun"
    return (
        base / f"{rid}_data.csv",
        base / f"{rid}_predictions.csv",
        base / f"{rid}_minority_full.csv",
        base / f"{rid}_minority_only.csv",
    )


def resolve_model_dir(user_input: str) -> Path:
    """Resolve classifier folder (same rules as ``feedback_core.resolve_classifier_dir``)."""
    return fc.resolve_classifier_dir(user_input or None)


class State(rx.State):
    """App state."""

    model_dir: str = "final_feedback_classifier"
    banner: str = ""
    run_id: str = ""

    csv_label: str = ""
    n_rows: int = 0
    n_cols: int = 0
    cols: list[str] = []
    detected_cols: list[str] = []
    segment_auto: str = ""
    compare_choices: list[str] = []
    compare_a: str = ""
    compare_b: str = ""

    embedding_model: str = "all-MiniLM-L6-v2"
    contamination: float = 0.08
    min_cluster_sz: int = 10
    include_noise: bool = False

    pred_chart: list[dict[str, Any]] = []
    mean_conf: float = 0.0
    has_preds: bool = False
    pred_filename: str = ""

    cmp_a_chart: list[dict[str, Any]] = []
    cmp_b_chart: list[dict[str, Any]] = []
    cmp_a_title: str = ""
    cmp_b_title: str = ""

    minority_banner: str = ""
    minority_pct_chart: list[dict[str, Any]] = []
    minority_ready: bool = False
    minority_full_name: str = ""
    minority_only_name: str = ""

    def _paths(self) -> tuple[Path, Path, Path, Path]:
        return paths_from_run_id(self.run_id)

    @rx.event
    def load_model(self):
        global _TOK, _MODEL, _METADATA
        self.banner = ""
        try:
            path = resolve_model_dir(self.model_dir)
            _TOK, _MODEL, _METADATA = fc.load_classifier(path)
            emb = str(_METADATA.get("embedding_model", "all-MiniLM-L6-v2"))
            self.embedding_model = emb
            self.contamination = float(_METADATA.get("contamination", 0.08))
            self.min_cluster_sz = int(_METADATA.get("min_cluster_size", 10))
            self.banner = f"Model loaded from `{path}`."
        except Exception as e:
            _TOK, _MODEL, _METADATA = None, None, None
            self.banner = f"Load failed: {e}"

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            return
        f = files[0]
        data = await f.read()
        base = _upload_base()
        base.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:16]
        raw, *_ = self._paths()
        raw.write_bytes(data)
        df = pd.read_csv(raw)
        self.csv_label = f.name
        self.n_rows = int(len(df))
        self.n_cols = int(len(df.columns))
        self.cols = df.columns.tolist()
        meta_cols = list((_METADATA or {}).get("text_cols") or [])
        self.detected_cols = fc.auto_detect_text_columns(df, meta_cols)
        sc = fc.auto_pick_segment_column(df, self.detected_cols)
        self.segment_auto = sc or ""
        self.has_preds = False
        self.minority_ready = False
        self.pred_chart = []
        self.minority_pct_chart = []
        self.compare_choices = []
        self.compare_a = ""
        self.compare_b = ""
        self.cmp_a_chart = []
        self.cmp_b_chart = []
        self.cmp_a_title = ""
        self.cmp_b_title = ""
        self.banner = f"Loaded {self.n_rows:,} rows — text columns auto-detected."

    @rx.event(background=True)
    async def run_predict(self):
        global _TOK, _MODEL, _METADATA
        if _TOK is None or _MODEL is None or not _METADATA:
            async with self:
                self.banner = "Load a model first."
            return

        async with self:
            rid = self.run_id
            cols = list(self.detected_cols)
            seg = self.segment_auto
            if not rid:
                self.banner = "Upload a CSV first."
                return
            if not cols:
                self.banner = "Could not auto-detect text columns in this CSV."
                return
            self.banner = "Running classifier…"

        raw, pred_path, _, _ = paths_from_run_id(rid)

        if not raw.exists():
            async with self:
                self.banner = "Upload file missing — try uploading again."
            return

        def work() -> tuple[pd.DataFrame, list[dict[str, Any]], float]:
            df = pd.read_csv(raw)
            combined = fc.combined_text_series(df, cols)
            id2lab = {int(k): v for k, v in _METADATA["id2label"].items()}
            out = fc.predict_dataframe(df, combined, _TOK, _MODEL, id2lab)
            counts = out["prediction"].value_counts()
            chart = fc.counts_to_chart_rows(counts)
            mean_c = float(out["confidence"].mean())
            out.to_csv(pred_path, index=False)
            return out, chart, mean_c

        try:
            pred_df, chart, mean_c = await asyncio.to_thread(work)
        except Exception as e:
            async with self:
                self.banner = f"Prediction error: {e}"
            return

        def cmp_work() -> tuple[list[dict], list[dict], str, str, list[str], str, str]:
            if seg and seg in pred_df.columns:
                vals = sorted(pred_df[seg].dropna().astype(str).unique().tolist())
                if len(vals) > 60:
                    vals = pred_df[seg].astype(str).value_counts().head(60).index.tolist()
                    vals = sorted(vals)
                ca = vals[0] if vals else ""
                cb = vals[1] if len(vals) > 1 else ca
                da = pred_df[pred_df[seg].astype(str) == ca]
                db = pred_df[pred_df[seg].astype(str) == cb]
                a1, b1 = fc.aligned_prediction_counts(
                    da["prediction"].value_counts(), db["prediction"].value_counts()
                )
                return (
                    fc.counts_to_chart_rows(a1),
                    fc.counts_to_chart_rows(b1),
                    f"{seg}: {ca}",
                    f"{seg}: {cb}",
                    vals,
                    ca,
                    cb,
                )
            med = float(pred_df["confidence"].median())
            hi = pred_df[pred_df["confidence"] >= med]
            lo = pred_df[pred_df["confidence"] < med]
            a1, b1 = fc.aligned_prediction_counts(
                hi["prediction"].value_counts(), lo["prediction"].value_counts()
            )
            return (
                fc.counts_to_chart_rows(a1),
                fc.counts_to_chart_rows(b1),
                f"High confidence (≥ {med:.3f})",
                f"Low confidence (< {med:.3f})",
                [],
                "",
                "",
            )

        try:
            ac, bc, ta, tb, vlist, ca, cb = await asyncio.to_thread(cmp_work)
        except Exception as e:
            async with self:
                self.banner = f"Comparison error: {e}"
            ac, bc, ta, tb, vlist, ca, cb = [], [], "", "", [], "", ""

        async with self:
            self.pred_chart = chart
            self.mean_conf = mean_c
            self.has_preds = True
            self.pred_filename = pred_path.name
            self.cmp_a_chart = ac
            self.cmp_b_chart = bc
            self.cmp_a_title = ta
            self.cmp_b_title = tb
            self.compare_choices = vlist
            self.compare_a = ca
            self.compare_b = cb
            self.banner = "Predictions ready."

    @rx.event(background=True)
    async def run_minority(self):
        async with self:
            rid = self.run_id
            cols = list(self.detected_cols)
            seg = self.segment_auto
            emb = self.embedding_model
            contam = self.contamination
            min_cs = self.min_cluster_sz
            noise = self.include_noise
            if not rid:
                self.minority_banner = "Upload a CSV first."
                return
            if not cols:
                self.minority_banner = "Could not auto-detect text columns."
                return
            self.minority_banner = "Running minority detection (embeddings)…"

        raw, pred_path, mfull, monly = paths_from_run_id(rid)
        if not raw.exists():
            async with self:
                self.minority_banner = "Upload file missing — try uploading again."
            return

        pred_df = pd.read_csv(pred_path) if pred_path.exists() else None

        def work() -> tuple[list[dict[str, Any]], str]:
            df = pd.read_csv(raw)
            texts = fc.combined_text_series(df, cols).tolist()
            mdf = fc.detect_minority_patterns(
                df,
                texts,
                emb,
                contam,
                min_cs,
                noise,
                pred_df,
            )
            mdf.to_csv(mfull, index=False)
            only = mdf[mdf["is_minority_pattern"]].copy()
            only.to_csv(monly, index=False)
            chart: list[dict[str, Any]] = []
            if seg and seg in mdf.columns:
                g = mdf.groupby(mdf[seg].astype(str), dropna=False).agg(
                    rows=("is_minority_pattern", "size"),
                    minority=("is_minority_pattern", "sum"),
                )
                g["pct"] = (g["minority"] / g["rows"].replace(0, pd.NA) * 100).round(2)
                g = g.dropna().sort_values("pct", ascending=False).head(25)
                chart = [{"label": str(i), "pct": float(r["pct"])} for i, r in g.iterrows()]
            total = int(mdf["is_minority_pattern"].sum())
            pct = total / max(1, len(mdf)) * 100.0
            msg = f"Flagged {total} rows ({pct:.2f}% of data)."
            return chart, msg

        try:
            chart, msg = await asyncio.to_thread(work)
        except Exception as e:
            async with self:
                self.minority_banner = f"Error: {e}"
            return

        async with self:
            self.minority_pct_chart = chart
            self.minority_ready = True
            self.minority_full_name = mfull.name
            self.minority_only_name = monly.name
            self.minority_banner = msg

    @rx.event(background=True)
    async def refresh_compare(self):
        async with self:
            rid = self.run_id
            seg = self.segment_auto
            ca = self.compare_a
            cb = self.compare_b
            if not rid:
                self.banner = "Upload a CSV and run predictions first."
                return

        _, pred_path, _, _ = paths_from_run_id(rid)
        if not pred_path.exists():
            async with self:
                self.banner = "Run predictions first."
            return

        def work() -> tuple[list[dict], list[dict], str, str]:
            pred_df = pd.read_csv(pred_path)
            if seg and seg in pred_df.columns and ca and cb:
                da = pred_df[pred_df[seg].astype(str) == ca]
                db = pred_df[pred_df[seg].astype(str) == cb]
                a1, b1 = fc.aligned_prediction_counts(
                    da["prediction"].value_counts(), db["prediction"].value_counts()
                )
                return (
                    fc.counts_to_chart_rows(a1),
                    fc.counts_to_chart_rows(b1),
                    f"{seg}: {ca}",
                    f"{seg}: {cb}",
                )
            med = float(pred_df["confidence"].median())
            hi = pred_df[pred_df["confidence"] >= med]
            lo = pred_df[pred_df["confidence"] < med]
            a1, b1 = fc.aligned_prediction_counts(
                hi["prediction"].value_counts(), lo["prediction"].value_counts()
            )
            return (
                fc.counts_to_chart_rows(a1),
                fc.counts_to_chart_rows(b1),
                f"High confidence (≥ {med:.3f})",
                f"Low confidence (< {med:.3f})",
            )

        ac, bc, ta, tb = await asyncio.to_thread(work)
        async with self:
            self.cmp_a_chart = ac
            self.cmp_b_chart = bc
            self.cmp_a_title = ta
            self.cmp_b_title = tb
            self.banner = "Comparison updated."

    def set_model_dir(self, v: str):
        self.model_dir = v

    def set_embedding_model(self, v: str):
        self.embedding_model = v

    def set_compare_a(self, v: str):
        self.compare_a = v

    def set_compare_b(self, v: str):
        self.compare_b = v

    def set_contamination(self, v: list[float | int]):
        if v:
            self.contamination = float(v[0])

    @rx.event
    def bump_cluster_minus(self):
        self.min_cluster_sz = max(3, int(self.min_cluster_sz) - 1)

    @rx.event
    def bump_cluster_plus(self):
        self.min_cluster_sz = min(500, int(self.min_cluster_sz) + 1)

    def toggle_noise(self, checked: bool):
        self.include_noise = checked

    @rx.event
    def download_preds(self):
        _, pred, _, _ = self._paths()
        if not pred.exists():
            return rx.toast.error("No predictions yet.")
        return rx.download(data=pred.read_bytes(), filename="predictions.csv")

    @rx.event
    def download_minority_full(self):
        p = _upload_base() / self.minority_full_name
        if not p.exists():
            return rx.toast.error("Run minority detection first.")
        return rx.download(data=p.read_bytes(), filename="minority_full.csv")

    @rx.event
    def download_minority_only(self):
        p = _upload_base() / self.minority_only_name
        if not p.exists():
            return rx.toast.error("Run minority detection first.")
        return rx.download(data=p.read_bytes(), filename="minority_only.csv")


def _metric_card(label: str, val: rx.Var | str | int | float) -> rx.Component:
    return rx.box(
        rx.text(label, size="1", color="gray", weight="medium", margin_bottom="0.25rem"),
        rx.heading(val, size="6", weight="bold"),
        padding="1rem 1.25rem",
        border_radius="14px",
        border_width="1px",
        border_color="var(--gray-4)",
        background_color="var(--color-panel)",
        flex="1",
        min_width="120px",
    )


def _bar_chart(data: rx.Var, title: rx.Var | str, has_data: rx.Var[bool]) -> rx.Component:
    return rx.box(
        rx.text(title, weight="bold", size="4", margin_bottom="0.75rem"),
        rx.cond(
            has_data,
            rx.box(
                rx.recharts.bar_chart(
                    rx.recharts.cartesian_grid(stroke_dasharray="3 3", stroke="var(--gray-6)"),
                    rx.recharts.x_axis(data_key="label", tick_line=False),
                    rx.recharts.y_axis(tick_line=False, axis_line=False),
                    rx.recharts.graphing_tooltip(),
                    rx.recharts.bar(data_key="count", fill="#6366f1", radius=[8, 8, 0, 0]),
                    data=data,
                    width="100%",
                    height=280,
                ),
                padding="0.5rem",
                border_radius="12px",
                border_width="1px",
                border_color="var(--gray-4)",
                background_color="var(--gray-1)",
                width="100%",
            ),
            rx.text("No chart data yet.", color="gray", size="2"),
        ),
        width="100%",
    )


def _pct_chart(data: rx.Var, title: str) -> rx.Component:
    return rx.box(
        rx.text(title, weight="bold", size="4", margin_bottom="0.75rem"),
        rx.cond(
            data.length() > 0,
            rx.box(
                rx.recharts.bar_chart(
                    rx.recharts.cartesian_grid(stroke_dasharray="3 3", stroke="var(--gray-6)"),
                    rx.recharts.x_axis(data_key="label", tick_line=False),
                    rx.recharts.y_axis(tick_line=False, axis_line=False),
                    rx.recharts.graphing_tooltip(),
                    rx.recharts.bar(data_key="pct", fill="#0d9488", radius=[8, 8, 0, 0]),
                    data=data,
                    width="100%",
                    height=280,
                ),
                padding="0.5rem",
                border_radius="12px",
                border_width="1px",
                border_color="var(--gray-4)",
                background_color="var(--gray-1)",
                width="100%",
            ),
            rx.text(
                "Pick a segment column and run minority detection to populate this chart.",
                color="gray",
                size="2",
            ),
        ),
        width="100%",
    )


def _panel(title: str, *children: rx.Component) -> rx.Component:
    return rx.box(
        rx.text(title, size="5", weight="bold", margin_bottom="1rem", color="var(--gray-12)"),
        rx.vstack(*children, spacing="3", width="100%", align="start"),
        padding="1.5rem",
        border_radius="16px",
        border_width="1px",
        border_color="var(--gray-4)",
        background_color="var(--color-panel)",
        box_shadow="0 4px 24px -8px rgba(15, 23, 42, 0.12)",
        width="100%",
    )


def _hero() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.badge("Student feedback intelligence", variant="solid", color_scheme="indigo", size="2"),
                rx.spacer(),
                rx.hstack(
                    rx.text("Reflex · Radix · Recharts", size="1", color="white", opacity="0.85"),
                    rx.color_mode.button(),
                    spacing="3",
                    align="center",
                ),
                width="100%",
                align="center",
            ),
            rx.heading("Feedback lab", size="9", weight="bold", color="white", margin_top="0.5rem"),
            rx.text(
                "Classify open-ended feedback, compare groups, and surface minority patterns — "
                "same logic as your Streamlit prototype, with a sharper interface.",
                size="4",
                color="white",
                opacity="0.92",
                max_width="720px",
            ),
            spacing="2",
            align="start",
            width="100%",
        ),
        width="100%",
        padding="2rem 2.25rem",
        border_radius="20px",
        background="linear-gradient(125deg, #1e1b4b 0%, #4338ca 42%, #7c3aed 88%)",
        box_shadow="0 24px 48px -12px rgba(67, 56, 202, 0.45)",
    )


def index() -> rx.Component:
    return rx.box(
        rx.vstack(
            _hero(),
            rx.callout(
                "Public deploy: set API_URL to your backend URL (see DEPLOY.md). "
                "Optional FEEDBACK_MODEL_DIR points to your classifier folder on the server.",
                icon="cloud",
                color_scheme="blue",
                variant="soft",
                width="100%",
            ),
            rx.grid(
                _panel(
                    "1 · Load classifier",
                    rx.input(
                        value=State.model_dir,
                        on_change=State.set_model_dir,
                        placeholder="final_feedback_classifier",
                        width="100%",
                        size="3",
                    ),
                    rx.button("Load model", on_click=State.load_model, variant="solid", size="3", width="100%"),
                    rx.text(State.banner, size="2", color="gray"),
                ),
                _panel(
                    "2 · Upload CSV",
                    rx.upload(
                        rx.vstack(
                            rx.icon("cloud_upload", size=28, color="var(--gray-9)"),
                            rx.text("Drop a CSV here", weight="bold", size="3"),
                            rx.text("or click to choose a file", size="2", color="gray"),
                            padding="1.75rem",
                            align="center",
                            spacing="2",
                        ),
                        id="csv_up",
                        border="2px dashed var(--gray-7)",
                        border_radius="14px",
                        padding="0.25rem",
                        multiple=False,
                        width="100%",
                        cursor="pointer",
                    ),
                    rx.button(
                        "Upload & parse",
                        on_click=State.handle_upload(rx.upload_files(upload_id="csv_up")),
                        size="3",
                        width="100%",
                    ),
                    rx.hstack(
                        rx.badge(State.csv_label, variant="soft", color_scheme="gray"),
                        spacing="2",
                        flex_wrap="wrap",
                    ),
                    rx.hstack(
                        _metric_card("Rows", State.n_rows),
                        _metric_card("Columns", State.n_cols),
                        _metric_card("Text cols (auto)", State.detected_cols.length()),
                        spacing="3",
                        width="100%",
                    ),
                ),
                columns="2",
                spacing="4",
                width="100%",
            ),
            _panel(
                "3 · Auto-detected mapping",
                rx.text(
                    "Feedback text columns are chosen from the CSV using header names and average text length. "
                    "Segment (for group compare) is picked when a low-cardinality column is available.",
                    size="2",
                    color="gray",
                ),
                rx.text("Combined text columns", weight="bold", size="2", margin_top="0.5rem"),
                rx.flex(
                    rx.foreach(State.detected_cols, lambda c: rx.badge(c, variant="soft", color_scheme="indigo")),
                    flex_wrap="wrap",
                    gap="0.5rem",
                ),
                rx.text("Segment column", weight="bold", size="2", margin_top="0.75rem"),
                rx.cond(
                    State.segment_auto != "",
                    rx.badge(State.segment_auto, color_scheme="cyan", variant="soft"),
                    rx.text("None detected — Compare tab will use confidence split.", size="2", color="gray"),
                ),
            ),
            _panel(
                "4 · Minority detection",
                rx.input(
                    value=State.embedding_model,
                    on_change=State.set_embedding_model,
                    placeholder="Sentence-Transformers model id",
                    width="100%",
                    size="3",
                ),
                rx.hstack(
                    rx.text("IsolationForest contamination", size="2", color="gray"),
                    rx.spacer(),
                    rx.text(State.contamination, size="2", weight="bold"),
                    width="100%",
                ),
                rx.slider(
                    value=[State.contamination],
                    min=0.02,
                    max=0.25,
                    step=0.01,
                    on_change=State.set_contamination,
                    width="100%",
                    color_scheme="indigo",
                ),
                rx.hstack(
                    rx.text("DBSCAN small-cluster max size", size="2", weight="medium"),
                    rx.spacer(),
                    rx.hstack(
                        rx.button("−", variant="outline", size="2", on_click=State.bump_cluster_minus),
                        rx.badge(State.min_cluster_sz, size="2", variant="surface", color_scheme="indigo"),
                        rx.button("+", variant="outline", size="2", on_click=State.bump_cluster_plus),
                        spacing="2",
                        align="center",
                    ),
                    width="100%",
                    align="center",
                ),
                rx.checkbox(
                    "Also treat DBSCAN noise (-1) as minority",
                    checked=State.include_noise,
                    on_change=State.toggle_noise,
                    size="2",
                ),
            ),
            rx.hstack(
                rx.button(
                    "Run predictions",
                    on_click=State.run_predict,
                    size="3",
                    variant="solid",
                    color_scheme="indigo",
                ),
                rx.button("Update comparison", on_click=State.refresh_compare, size="3", variant="soft"),
                rx.button(
                    "Run minority detection",
                    on_click=State.run_minority,
                    size="3",
                    variant="outline",
                    color_scheme="teal",
                ),
                spacing="3",
                width="100%",
                flex_wrap="wrap",
            ),
            rx.box(
                rx.tabs.root(
                    rx.tabs.list(
                        rx.tabs.trigger("Overview", value="ov"),
                        rx.tabs.trigger("Compare", value="cmp"),
                        rx.tabs.trigger("Minority", value="min"),
                        width="100%",
                    ),
                    rx.tabs.content(
                        rx.vstack(
                            rx.hstack(
                                _metric_card("Mean confidence", State.mean_conf),
                                rx.button(
                                    "Download predictions",
                                    on_click=State.download_preds,
                                    variant="soft",
                                    size="3",
                                ),
                                spacing="4",
                                align="center",
                                flex_wrap="wrap",
                            ),
                            _bar_chart(State.pred_chart, "Prediction counts", State.has_preds),
                            spacing="4",
                            width="100%",
                        ),
                        value="ov",
                    ),
                    rx.tabs.content(
                        rx.vstack(
                            rx.text(
                                "Choose a segment column for A/B segments; otherwise charts split on median confidence.",
                                size="2",
                                color="gray",
                            ),
                            rx.cond(
                                State.segment_auto != "",
                                rx.hstack(
                                    rx.vstack(
                                        rx.text("Group A", weight="bold", size="2"),
                                        rx.select(
                                            State.compare_choices,
                                            value=State.compare_a,
                                            on_change=State.set_compare_a,
                                            width="100%",
                                        ),
                                        width="50%",
                                        spacing="2",
                                    ),
                                    rx.vstack(
                                        rx.text("Group B", weight="bold", size="2"),
                                        rx.select(
                                            State.compare_choices,
                                            value=State.compare_b,
                                            on_change=State.set_compare_b,
                                            width="100%",
                                        ),
                                        width="50%",
                                        spacing="2",
                                    ),
                                    width="100%",
                                    spacing="4",
                                ),
                                rx.callout(
                                    "No segment column — comparison uses high vs low confidence split.",
                                    color_scheme="gray",
                                ),
                            ),
                            rx.button("Apply comparison", on_click=State.refresh_compare, size="2", variant="surface"),
                            rx.hstack(
                                rx.box(
                                    _bar_chart(
                                        State.cmp_a_chart,
                                        State.cmp_a_title,
                                        State.cmp_a_chart.length() > 0,
                                    ),
                                    width=["100%", "50%", "50%"],
                                ),
                                rx.box(
                                    _bar_chart(
                                        State.cmp_b_chart,
                                        State.cmp_b_title,
                                        State.cmp_b_chart.length() > 0,
                                    ),
                                    width=["100%", "50%", "50%"],
                                ),
                                width="100%",
                                spacing="4",
                                flex_direction=["column", "row", "row"],
                            ),
                            spacing="4",
                            width="100%",
                        ),
                        value="cmp",
                    ),
                    rx.tabs.content(
                        rx.vstack(
                            rx.text(State.minority_banner, size="2"),
                            rx.hstack(
                                rx.button(
                                    "Download full + flags",
                                    on_click=State.download_minority_full,
                                    variant="soft",
                                ),
                                rx.button(
                                    "Download minority rows only",
                                    on_click=State.download_minority_only,
                                    variant="soft",
                                ),
                                spacing="3",
                                flex_wrap="wrap",
                            ),
                            _pct_chart(State.minority_pct_chart, "Minority rate by segment (%)"),
                            spacing="3",
                            width="100%",
                        ),
                        value="min",
                    ),
                    default_value="ov",
                    width="100%",
                ),
                padding="1.25rem",
                border_radius="16px",
                border_width="1px",
                border_color="var(--gray-4)",
                background_color="var(--color-panel)",
                width="100%",
            ),
            spacing="5",
            width="100%",
            max_width="1120px",
            padding_y="2rem",
            padding_x=["1rem", "1.5rem", "2rem"],
        ),
        width="100%",
        min_height="100vh",
        background="linear-gradient(180deg, var(--gray-2) 0%, var(--gray-3) 100%)",
    )


app = rx.App(
    theme=rx.theme(
        appearance="inherit",
        accent_color="indigo",
        gray_color="slate",
        radius="large",
        has_background=True,
        panel_background="translucent",
    ),
    style={"font_feature_settings": "'ss01' on, 'cv01' on"},
)
app.add_page(index, title="Feedback lab", route="/")
