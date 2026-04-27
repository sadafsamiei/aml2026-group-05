"""
dashboard/app.py
----------------
Streamlit dashboard for aml2026-group-05 — modern dark theme.

Three tabs:
  1. Headline Result      — interactive complexity vs F1 scatter
  2. Per-class Explorer   — model x feature breakdown with live confusion matrix
  3. Clip Explorer        — filter clips by agreement category, play audio inline

Usage
-----
    cd /home/renku/work/aml2026-group-05
    pip install streamlit plotly
    streamlit run dashboard/app.py

For best results, also create `.streamlit/config.toml` with:

    [theme]
    base = "dark"
    primaryColor = "#F59E0B"
    backgroundColor = "#0A0E1A"
    secondaryBackgroundColor = "#131825"
    textColor = "#E5E7EB"
    font = "sans serif"

The CSS injected below overrides most of this, but setting it natively
makes the loading flash less jarring.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

HERE = Path(__file__).resolve().parent           # dashboard/
sys.path.insert(0, str(HERE.parent / "src"))     # src/

import config as cfg

CLASS_NAMES = cfg.CLASS_NAMES
RESULTS_DIR = cfg.RES_DIR / "results"
WAV_DIR     = cfg.WAV_DIR
SPEC_DIR    = cfg.SPEC_DIR

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

# Surfaces
BG          = "#0A0E1A"              # main background
SURFACE     = "#131825"              # card background
SURFACE_2   = "#1A2030"              # elevated card / hover
BORDER      = "rgba(255,255,255,0.06)"
BORDER_HI   = "rgba(255,255,255,0.10)"

# Text
TEXT        = "#E5E7EB"              # primary
TEXT_MUTED  = "#9CA3AF"              # secondary
TEXT_DIM    = "#6B7280"              # tertiary / captions

# Accents
ACCENT      = "#F59E0B"              # warm amber (primary brand)
ACCENT_2    = "#22D3EE"              # bright cyan
GRID        = "rgba(255,255,255,0.06)"

# Model color palette (warmer, more pastel — better on dark)
COLOR_M1    = "#60A5FA"              # blue
COLOR_M2    = "#F59E0B"              # amber
COLOR_M3    = "#34D399"              # mint green

# Heatmap colorscale tuned to look good on dark surface
DARK_HEATMAP = [
    [0.00, "rgba(26, 32, 48, 1)"],       # near surface bg
    [0.20, "rgba(96, 165, 250, 0.25)"],
    [0.45, "rgba(96, 165, 250, 0.55)"],
    [0.70, "rgba(34, 211, 238, 0.85)"],
    [1.00, "rgba(34, 211, 238, 1)"],     # bright cyan
]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Child Vocalisation Context Classification — Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Default plotly template — every figure picks this up automatically
pio.templates["aml_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=12),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                   tickcolor=GRID, color=TEXT_MUTED),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                   tickcolor=GRID, color=TEXT_MUTED),
        colorway=[COLOR_M1, COLOR_M2, COLOR_M3, ACCENT_2, "#A78BFA", "#F472B6"],
        hoverlabel=dict(
            bgcolor=SURFACE_2, bordercolor=BORDER_HI,
            font=dict(family="Inter", size=12, color=TEXT),
        ),
        legend=dict(
            bgcolor="rgba(19,24,37,0.6)", bordercolor=BORDER,
            borderwidth=1, font=dict(color=TEXT, size=12),
        ),
    )
)
pio.templates.default = "aml_dark"

# ---------------------------------------------------------------------------
# Global CSS — dark theme, modern typography, refined components
# ---------------------------------------------------------------------------

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ---- Base & background ---- */
.stApp {{
    background: {BG};
    background-image:
        radial-gradient(at 0% 0%,    rgba(245,158,11,0.07) 0px, transparent 50%),
        radial-gradient(at 100% 0%,  rgba(96,165,250,0.06) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(52,211,153,0.04) 0px, transparent 60%);
    background-attachment: fixed;
}}

* {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
code, pre {{ font-family: 'JetBrains Mono', ui-monospace, monospace !important; }}

html, body, [class*="css"], .stMarkdown, .stMarkdown p, .stMarkdown li {{
    color: {TEXT};
}}

/* Hide default streamlit chrome */
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* Container */
.block-container {{
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    max-width: 1400px;
}}

/* ---- Title block ---- */
.main-title {{
    font-size: 38px;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 6px 0;
    background: linear-gradient(135deg, #F8FAFC 0%, {ACCENT} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.subtitle {{
    font-size: 14px;
    color: {TEXT_MUTED};
    margin: 0 0 32px 0;
    font-weight: 400;
}}
.subtitle-accent {{
    color: {ACCENT};
    font-weight: 500;
}}
.subtitle-dot {{
    color: {TEXT_DIM};
    margin: 0 8px;
}}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: {SURFACE};
    padding: 6px;
    border-radius: 12px;
    border: 1px solid {BORDER};
    margin-bottom: 24px;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 14px !important;
    font-weight: 500 !important;
    color: {TEXT_MUTED} !important;
    background: transparent !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    border: none !important;
    transition: all 0.18s ease !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT} !important;
    background: rgba(255,255,255,0.03) !important;
}}
.stTabs [aria-selected="true"] {{
    background: {SURFACE_2} !important;
    color: {TEXT} !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05),
                0 1px 2px rgba(0,0,0,0.2) !important;
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}

/* ---- Custom metric cards (HTML in tab1) ---- */
.metric-card {{
    background: {SURFACE};
    padding: 20px 22px;
    border: 1px solid {BORDER};
    border-radius: 14px;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
    height: 100%;
}}
.metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {ACCENT}, transparent 60%);
}}
.metric-card:hover {{
    border-color: rgba(245,158,11,0.25);
    transform: translateY(-1px);
}}
.metric-num {{
    font-size: 34px;
    font-weight: 700;
    color: {ACCENT};
    line-height: 1;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
    font-variant-numeric: tabular-nums;
}}
.metric-lbl {{
    font-size: 11px;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}}
.metric-sub {{
    font-size: 13px;
    color: {TEXT};
    margin-top: 10px;
    line-height: 1.45;
}}

/* ---- Headings inside tabs ---- */
.stTabs h3, .stTabs [data-testid="stHeading"] h3 {{
    color: {TEXT} !important;
    font-weight: 600 !important;
    font-size: 20px !important;
    margin: 0 0 8px 0 !important;
    letter-spacing: -0.01em;
}}
.stTabs h4 {{
    color: {TEXT} !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    margin-top: 0 !important;
}}

/* ---- Form widgets ---- */
.stRadio > label, .stCheckbox > label,
.stSelectbox > label, .stMultiSelect > label {{
    font-size: 11px !important;
    color: {TEXT_MUTED} !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
    margin-bottom: 6px !important;
}}

.stSelectbox > div > div, .stMultiSelect > div > div {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 8px !important;
}}
.stSelectbox > div > div:hover, .stMultiSelect > div > div:hover {{
    border-color: {BORDER_HI} !important;
}}

.stRadio > div {{
    background: {SURFACE};
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px solid {BORDER};
    gap: 4px;
}}
.stRadio label, .stRadio label p {{ color: {TEXT} !important; }}

.stCheckbox {{ color: {TEXT}; }}
.stCheckbox label p {{ color: {TEXT} !important; }}

/* Multiselect chips */
[data-baseweb="tag"] {{
    background: {SURFACE_2} !important;
    border: 1px solid {BORDER_HI} !important;
    color: {TEXT} !important;
}}

/* ---- Dataframe ---- */
[data-testid="stDataFrame"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
}}
[data-testid="stDataFrame"] [role="grid"] {{ background: transparent; }}

/* ---- Native st.metric ---- */
[data-testid="stMetric"] {{
    background: {SURFACE};
    padding: 16px 18px;
    border: 1px solid {BORDER};
    border-radius: 10px;
    transition: border-color 0.2s ease;
}}
[data-testid="stMetric"]:hover {{ border-color: {BORDER_HI}; }}
[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    font-variant-numeric: tabular-nums;
}}
[data-testid="stMetricDelta"] {{
    font-size: 12px !important;
    font-weight: 500 !important;
}}

/* ---- Alerts (error / warning / info / success) ---- */
.stAlert,
div[data-testid="stAlert"],
div[data-testid="stAlertContainer"],
div[data-baseweb="notification"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {TEXT} !important;
    padding: 14px 16px !important;
}}

/* Force themed text inside every alert variant */
.stAlert *, .stAlert p, .stAlert div, .stAlert span,
div[data-testid="stAlert"] *,
div[data-testid="stAlertContainer"] *,
[data-testid="stAlertContentError"] *,
[data-testid="stAlertContentWarning"] *,
[data-testid="stAlertContentInfo"] *,
[data-testid="stAlertContentSuccess"] * {{
    color: {TEXT} !important;
    fill: {TEXT_MUTED} !important;
}}

/* Color-coded left bar by alert kind */
div[data-testid="stAlert"][kind="error"],
div[data-baseweb="notification"][kind="error"]      {{ border-left: 3px solid #EF4444 !important; }}
div[data-testid="stAlert"][kind="warning"],
div[data-baseweb="notification"][kind="warning"]    {{ border-left: 3px solid {ACCENT} !important; }}
div[data-testid="stAlert"][kind="info"],
div[data-baseweb="notification"][kind="info"]       {{ border-left: 3px solid {COLOR_M1} !important; }}
div[data-testid="stAlert"][kind="success"],
div[data-baseweb="notification"][kind="success"]    {{ border-left: 3px solid {COLOR_M3} !important; }}

/* Fallback for older Streamlit DOM where kind is on a child */
.stAlert[data-baseweb="notification"] {{ border-left: 3px solid {ACCENT} !important; }}

/* ---- Expander ---- */
.streamlit-expanderHeader,
[data-testid="stExpander"] details summary {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {TEXT} !important;
    font-weight: 500 !important;
}}
[data-testid="stExpander"] {{
    border: none !important;
    background: transparent !important;
}}

/* ---- Audio player ---- */
audio {{
    width: 100%;
    border-radius: 10px;
    filter: invert(0.9) hue-rotate(180deg);
}}

/* ---- Markdown niceties ---- */
.stMarkdown p {{ color: {TEXT_MUTED}; line-height: 1.65; }}
.stMarkdown strong {{ color: {TEXT}; font-weight: 600; }}
.stMarkdown code {{
    background: {SURFACE_2};
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 0.88em;
    color: {ACCENT};
    border: 1px solid {BORDER};
}}

/* ---- Dividers ---- */
hr {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 28px 0;
}}

/* ---- Footer ---- */
.footer-caption {{
    color: {TEXT_DIM};
    font-size: 12px;
    text-align: center;
    margin-top: 48px;
    padding: 24px 0 0 0;
    border-top: 1px solid {BORDER};
    line-height: 1.7;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    '<p class="main-title">Child Vocalisation Context Classification</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">aml2026-group-05'
    '<span class="subtitle-dot">·</span>Interactive results dashboard'
    '<span class="subtitle-dot">·</span>'
    '<span class="subtitle-accent">The simplest model won</span></p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Canonical results — hardcoded from README to keep the dashboard portable
# ---------------------------------------------------------------------------

RESULTS = pd.DataFrame([
    {"model": "Model 1 (LR)",        "feature": "MFCC",     "params":   164,
     "val_f1": 0.240, "test_f1": 0.253, "std": 0.000, "config": "C=10, balanced"},
    {"model": "Model 1 (LR)",        "feature": "Mel",      "params":  1028,
     "val_f1": 0.232, "test_f1": 0.277, "std": 0.000, "config": "C=0.01, balanced"},
    {"model": "Model 1 (LR)",        "feature": "wav2vec2", "params":  6148,
     "val_f1": 0.253, "test_f1": 0.263, "std": 0.000, "config": "C=0.01, balanced"},
    {"model": "Model 2 (mean pool)", "feature": "MFCC",     "params": None,
     "val_f1": 0.258, "test_f1": 0.238, "std": 0.039, "config": "d=64, L=2"},
    {"model": "Model 2 (mean pool)", "feature": "Mel",      "params": 126_000,
     "val_f1": 0.270, "test_f1": 0.249, "std": 0.020, "config": "d=64, L=2, drop=0.5"},
    {"model": "Model 2 (mean pool)", "feature": "wav2vec2", "params": 167_000,
     "val_f1": 0.294, "test_f1": 0.219, "std": 0.018, "config": "d=64, L=2, drop=0.5"},
    {"model": "Model 3 (attention)", "feature": "MFCC",     "params": None,
     "val_f1": 0.257, "test_f1": 0.237, "std": 0.026, "config": "d=32, L=1"},
    {"model": "Model 3 (attention)", "feature": "Mel",      "params": 26_000,
     "val_f1": 0.252, "test_f1": 0.221, "std": 0.013, "config": "d=32, L=1, drop=0.3"},
    {"model": "Model 3 (attention)", "feature": "wav2vec2", "params": 648_000,
     "val_f1": 0.286, "test_f1": 0.246, "std": 0.016, "config": "d=192, L=1, drop=0.3"},
])

PERCLASS = {
    ("Model 1 (LR)",        "Mel"):      [0.220, 0.396, 0.263, 0.228],
    ("Model 1 (LR)",        "wav2vec2"): [0.228, 0.391, 0.188, 0.247],
    ("Model 1 (LR)",        "MFCC"):     [0.176, 0.388, 0.207, 0.241],
    ("Model 2 (mean pool)", "Mel"):      [0.185, 0.340, 0.220, 0.258],
    ("Model 2 (mean pool)", "wav2vec2"): [0.239, 0.255, 0.231, 0.153],
    ("Model 2 (mean pool)", "MFCC"):     [0.147, 0.428, 0.171, 0.169],
    ("Model 3 (attention)", "Mel"):      [0.130, 0.284, 0.106, 0.363],
    ("Model 3 (attention)", "wav2vec2"): [0.192, 0.395, 0.199, 0.198],
    ("Model 3 (attention)", "MFCC"):     [0.155, 0.395, 0.180, 0.220],
}

CM_M2_W2V = np.array([
    [0.34, 0.37, 0.18, 0.11],
    [0.41, 0.27, 0.24, 0.09],
    [0.31, 0.26, 0.28, 0.15],
    [0.30, 0.34, 0.26, 0.10],
])
CM_M3_W2V = np.array([
    [0.19, 0.62, 0.11, 0.08],
    [0.19, 0.59, 0.14, 0.08],
    [0.18, 0.49, 0.20, 0.13],
    [0.14, 0.57, 0.15, 0.14],
])

TEST_DIST = {"EXTEV": 50, "INTER": 79, "MOVE": 47, "PLTOY": 93}
N_TEST    = sum(TEST_DIST.values())
CHANCE    = 0.250

MODEL_COLORS = {
    "Model 1 (LR)":        COLOR_M1,
    "Model 2 (mean pool)": COLOR_M2,
    "Model 3 (attention)": COLOR_M3,
}

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_clips_cached(path_str: str, mtime: float) -> pd.DataFrame:
    """Inner cached loader — cache key is (path, mtime) so it auto-busts
    whenever the file on disk changes or a newer file appears."""
    with open(path_str) as f:
        data = json.load(f)

    clips = pd.DataFrame(data["per_clip"])

    def majority(row):
        preds = [row["m1_pred"], row["m2_pred"], row["m3_pred"]]
        for p in set(preds):
            if preds.count(p) >= 2:
                return p
        return "no_majority"

    def categorise(row):
        nw = int(row["n_wrong"])
        if nw == 0:
            return "all_agree_correct"
        if nw == 3:
            preds = [row["m1_pred"], row["m2_pred"], row["m3_pred"]]
            if len(set(preds)) == 1:
                return "all_agree_wrong"
            return "all_wrong_disagree"
        if not row["m1_wrong"] and row["m2_wrong"] and row["m3_wrong"]:
            return "lr_correct_transformers_wrong"
        if row["m1_wrong"] and not row["m2_wrong"] and not row["m3_wrong"]:
            return "transformers_correct_lr_wrong"
        return "partial_disagreement"

    clips["majority_pred"] = clips.apply(majority,   axis=1)
    clips["category"]      = clips.apply(categorise, axis=1)
    return clips


def load_clips() -> tuple[pd.DataFrame, list[Path]]:
    """Find the latest error_analysis JSON and load it.

    Searches several plausible directories so the dashboard works even if
    config.RES_DIR resolves to something unexpected on a particular machine.
    Returns (clips_df, list_of_paths_searched) so the UI can show debug info
    when nothing is found.
    """
    search_dirs = [
        RESULTS_DIR,                                  # config-derived
        Path("/home/renku/work/data/results"),        # absolute fallback 1
        cfg.RES_DIR / "results",                       # belt-and-braces
        HERE.parent / "data" / "results",              # repo-relative fallback
    ]
    # Deduplicate while preserving order
    seen = set()
    search_dirs = [d for d in search_dirs
                    if not (str(d) in seen or seen.add(str(d)))]

    for d in search_dirs:
        if not d.exists():
            continue
        matches = sorted(d.glob("error_analysis_*.json"))
        if matches:
            latest = matches[-1]
            return (_load_clips_cached(str(latest),
                                        latest.stat().st_mtime),
                    search_dirs)
    return (pd.DataFrame(), search_dirs)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_cm(cm: np.ndarray, title: str):
    """Confusion matrix with dark-friendly colorscale."""
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=CLASS_NAMES, y=CLASS_NAMES,
        text=[[f"{v:.2f}" for v in row] for row in cm],
        texttemplate="%{text}",
        textfont=dict(size=14, color="#F8FAFC", family="Inter"),
        colorscale=DARK_HEATMAP, zmin=0, zmax=0.7,
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>"
                      "Fraction: %{z:.3f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title=dict(text="Fraction", font=dict(color=TEXT_MUTED, size=11)),
            tickfont=dict(color=TEXT_MUTED, size=10),
            outlinewidth=0,
            thickness=10,
            len=0.8,
        ),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT)),
        xaxis_title="Predicted class",
        yaxis_title="True class",
        height=380,
        margin=dict(l=20, r=20, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Headline Result",
    "📊  Per-class Explorer",
    "🎧  Clip Explorer",
    "🌟  Presentation Picks",
])

# ===========================================================================
# TAB 1 — Headline Result
# ===========================================================================

with tab1:
    st.markdown("### Model complexity vs test F1")
    st.markdown(
        "Every dot is one (model, feature) pair from our 5-seed runs. "
        "The X axis is the parameter count on a log scale; the Y axis is test "
        "macro-F1. **The smallest models are the best.**"
    )

    col_l, col_r = st.columns([1, 4])
    with col_l:
        feat_filter = st.multiselect(
            "Show feature sources",
            ["MFCC", "Mel", "wav2vec2"],
            default=["MFCC", "Mel", "wav2vec2"],
        )
        show_chance = st.checkbox("Show chance line",         value=True)
        show_annot  = st.checkbox("Show point labels",        value=True)
        show_err    = st.checkbox("Show seed std error bars", value=True)

    with col_r:
        df_show = RESULTS[RESULTS["feature"].isin(feat_filter)]
        df_show = df_show[df_show["params"].notna()].copy()

        symbols = {"MFCC": "square", "Mel": "circle", "wav2vec2": "triangle-up"}

        fig = go.Figure()
        for (model, feature), sub in df_show.groupby(["model", "feature"]):
            for _, row in sub.iterrows():
                hover = (f"<b>{model} — {feature}</b><br>"
                         f"Params: {int(row['params']):,}<br>"
                         f"Val F1: {row['val_f1']:.3f}<br>"
                         f"Test F1: {row['test_f1']:.3f} ± {row['std']:.3f}<br>"
                         f"Config: {row['config']}")
                fig.add_trace(go.Scatter(
                    x=[row["params"]], y=[row["test_f1"]],
                    mode="markers+text" if show_annot else "markers",
                    marker=dict(
                        size=18, color=MODEL_COLORS[model],
                        symbol=symbols[feature],
                        line=dict(width=1.5, color=BG),
                        opacity=0.95,
                    ),
                    text=[f"  {feature}"] if show_annot else None,
                    textposition="middle right",
                    textfont=dict(size=11, color=TEXT_MUTED),
                    error_y=dict(
                        type="data", array=[row["std"]],
                        visible=show_err, color=MODEL_COLORS[model],
                        thickness=1.5, width=4,
                    ),
                    name=f"{model} — {feature}",
                    hovertemplate=hover + "<extra></extra>",
                    showlegend=False,
                ))

        # Manual legend (one per model)
        for model, color in MODEL_COLORS.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=14, color=color, line=dict(width=1.5, color=BG)),
                name=model, showlegend=True,
            ))

        if show_chance:
            fig.add_hline(
                y=CHANCE, line_dash="dash", line_color=TEXT_DIM,
                line_width=1,
                annotation_text=f"Chance = {CHANCE}",
                annotation_position="bottom right",
                annotation_font=dict(color=TEXT_MUTED, size=11),
            )

        fig.update_xaxes(
            type="log",
            title=dict(text="Model complexity [parameters, log scale]",
                       font=dict(color=TEXT_MUTED, size=12)),
        )
        fig.update_yaxes(
            title=dict(text="Test macro-F1", font=dict(color=TEXT_MUTED, size=12)),
            range=[0.15, 0.32],
        )
        fig.update_layout(
            height=480,
            margin=dict(l=20, r=20, t=20, b=40),
            legend=dict(orientation="h", y=1.05, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Best result callout + numbers
    best = df_show.loc[df_show["test_f1"].idxmax()] if len(df_show) else None
    c1, c2, c3 = st.columns(3)
    if best is not None:
        with c1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-num">{best["test_f1"]:.3f}</div>'
                f'<div class="metric-lbl">Best test F1 in current view</div>'
                f'<div class="metric-sub">{best["model"]} on {best["feature"]} '
                f'<br><span style="color:{TEXT_DIM};">'
                f'{int(best["params"]):,} params</span></div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-num">+{best["test_f1"] - CHANCE:.3f}</div>'
                f'<div class="metric-lbl">Above uniform chance baseline</div>'
                f'<div class="metric-sub">Theoretical chance = {CHANCE}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c3:
            n_above = (df_show["test_f1"] > CHANCE).sum()
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-num">{n_above} / {len(df_show)}</div>'
                f'<div class="metric-lbl">Configs above chance</div>'
                f'<div class="metric-sub">Some transformer configs are at or '
                f'below chance</div></div>',
                unsafe_allow_html=True,
            )

    with st.expander("📄 Full results table"):
        df_table = RESULTS.copy()
        df_table["params"] = df_table["params"].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else "—"
        )
        df_table["test_f1"] = df_table.apply(
            lambda r: f"{r['test_f1']:.3f} ± {r['std']:.3f}", axis=1
        )
        df_table["val_f1"]  = df_table["val_f1"].apply(lambda x: f"{x:.3f}")
        st.dataframe(
            df_table[["model", "feature", "params", "val_f1", "test_f1", "config"]],
            hide_index=True, use_container_width=True,
        )

# ===========================================================================
# TAB 2 — Per-class Explorer
# ===========================================================================

with tab2:
    st.markdown("### Per-class breakdown across models and features")
    st.markdown(
        "Pick a feature source and a model. The dashboard shows the per-class "
        "F1 alongside the test class distribution, plus the confusion matrix "
        "where available."
    )

    col_a, col_b = st.columns([1, 4])
    with col_a:
        feat = st.radio("Feature source", ["Mel", "wav2vec2", "MFCC"], index=0)
        model_options = ["Model 1 (LR)", "Model 2 (mean pool)",
                         "Model 3 (attention)", "All three (compare)"]
        model_sel = st.radio("Model", model_options, index=3)
        show_dist = st.checkbox("Overlay test class distribution", value=True)

    with col_b:
        if model_sel == "All three (compare)":
            models = ["Model 1 (LR)", "Model 2 (mean pool)", "Model 3 (attention)"]
        else:
            models = [model_sel]

        fig = go.Figure()
        for m in models:
            f1s = PERCLASS.get((m, feat), [0]*4)
            fig.add_trace(go.Bar(
                x=CLASS_NAMES, y=f1s, name=m,
                marker=dict(
                    color=MODEL_COLORS[m],
                    line=dict(width=0),
                ),
                text=[f"{v:.3f}" for v in f1s],
                textposition="outside",
                textfont=dict(size=11, color=TEXT_MUTED),
            ))

        if show_dist:
            dist = [TEST_DIST[c]/N_TEST for c in CLASS_NAMES]
            fig.add_trace(go.Scatter(
                x=CLASS_NAMES, y=dist, name="Test class rate",
                mode="lines+markers",
                line=dict(color=TEXT_MUTED, width=2, dash="dot"),
                marker=dict(size=8, symbol="diamond",
                            line=dict(color=BG, width=1)),
                yaxis="y2",
            ))

        fig.add_hline(
            y=CHANCE, line_dash="dash", line_color=TEXT_DIM, line_width=1,
            annotation_text=f"Chance = {CHANCE}",
            annotation_position="top left",
            annotation_font=dict(color=TEXT_MUTED, size=11),
        )

        layout = dict(
            barmode="group",
            height=420,
            margin=dict(l=20, r=20, t=30, b=40),
            yaxis=dict(title=dict(text="Per-class F1", font=dict(color=TEXT_MUTED)),
                       range=[0, 0.55]),
            xaxis=dict(color=TEXT_MUTED),
            legend=dict(orientation="h", y=1.13, x=0),
            bargap=0.25,
            bargroupgap=0.1,
        )
        if show_dist:
            layout["yaxis2"] = dict(
                title=dict(text="Test class rate", font=dict(color=TEXT_MUTED)),
                overlaying="y", side="right",
                range=[0, 0.55], showgrid=False,
                color=TEXT_MUTED,
            )
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices (only for wav2vec2)
    if feat == "wav2vec2" and model_sel in (
            "Model 2 (mean pool)", "Model 3 (attention)", "All three (compare)"):
        st.markdown("---")
        st.markdown("#### Confusion matrices on wav2vec2 (mean over 5 seeds)")

        if model_sel == "All three (compare)":
            colA, colB = st.columns(2)
            with colA:
                _plot_cm(CM_M2_W2V, "Model 2 — mean pool")
            with colB:
                _plot_cm(CM_M3_W2V, "Model 3 — attention")
        elif model_sel == "Model 2 (mean pool)":
            _plot_cm(CM_M2_W2V, "Model 2 — mean pool")
        elif model_sel == "Model 3 (attention)":
            _plot_cm(CM_M3_W2V, "Model 3 — attention")

    # INTER bias readout
    st.markdown("---")
    st.markdown("#### INTER prediction bias")

    inter_pred = {"Model 2 (mean pool)": 0.311, "Model 3 (attention)": 0.574}
    true_inter = TEST_DIST["INTER"] / N_TEST
    cards = st.columns(3)
    cards[0].metric(
        "True INTER rate", f"{true_inter:.1%}",
        help="Fraction of test clips whose true class is INTER",
    )
    cards[1].metric(
        "Model 2 INTER predictions", f"{inter_pred['Model 2 (mean pool)']:.1%}",
        delta=f"{inter_pred['Model 2 (mean pool)'] - true_inter:+.1%}",
    )
    cards[2].metric(
        "Model 3 INTER predictions", f"{inter_pred['Model 3 (attention)']:.1%}",
        delta=f"{inter_pred['Model 3 (attention)'] - true_inter:+.1%}",
    )
    st.caption(
        "Tagalog training prior is 46.9% INTER. Model 3 absorbs this prior; "
        "Model 2's mean pool dilutes it via averaging."
    )


# ===========================================================================
# TAB 3 — Clip Explorer
# ===========================================================================

with tab3:
    head_l, head_r = st.columns([6, 1])
    with head_l:
        st.markdown("### Explore individual test clips")
    with head_r:
        if st.button("↻ Reload", use_container_width=True,
                     help="Re-scan results dir and clear cached clips"):
            st.cache_data.clear()
            st.rerun()

    st.markdown(
        "Filter the 269 test clips by what each model predicted. "
        "Click a row, listen to the audio, see the spectrogram. "
        "Use this during Q&A when someone asks 'show me an example'."
    )

    clips, searched_dirs = load_clips()
    if len(clips) == 0:
        searched_html = "".join(
            f'<div style="margin:4px 0;"><code style="background:{SURFACE_2}; '
            f'padding:2px 6px; border-radius:4px; color:{TEXT_MUTED}; '
            f'font-family:JetBrains Mono,monospace; font-size:11px;">'
            f'{d}</code> <span style="color:{TEXT_DIM}; font-size:11px;">'
            f'({"exists" if d.exists() else "missing"})</span></div>'
            for d in searched_dirs
        )
        st.markdown(
            f'<div style="background:{SURFACE}; border:1px solid {BORDER}; '
            f'border-radius:14px; padding:32px; margin-top:16px;">'
            f'<div style="font-size:36px; margin-bottom:12px; opacity:0.6; '
            f'text-align:center;">📭</div>'
            f'<div style="font-size:16px; font-weight:600; color:{TEXT}; '
            f'margin-bottom:12px; text-align:center;">No clip data loaded</div>'
            f'<div style="font-size:13px; color:{TEXT_MUTED}; line-height:1.6; '
            f'text-align:center; margin-bottom:16px;">'
            f'Searched these directories for <code style="background:{SURFACE_2}; '
            f'padding:2px 6px; border-radius:4px; color:{ACCENT}; '
            f'font-family:JetBrains Mono,monospace;">error_analysis_*.json</code>:'
            f'</div>'
            f'<div style="background:{SURFACE_2}; padding:14px 18px; '
            f'border-radius:8px; border:1px solid {BORDER};">'
            f'{searched_html}</div>'
            f'<div style="font-size:13px; color:{TEXT_MUTED}; line-height:1.6; '
            f'text-align:center; margin-top:18px;">'
            f'Run <code style="background:{SURFACE_2}; padding:2px 6px; '
            f'border-radius:4px; color:{ACCENT}; '
            f'font-family:JetBrains Mono,monospace;">'
            f'python experiments/error_analysis.py</code> first, '
            f'then click ↻ Reload above.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    cat_labels = {
        "all_agree_correct":              "✅ All 3 correct (success cases)",
        "all_agree_wrong":                "❌ All 3 agree on the wrong class (data ceiling)",
        "all_wrong_disagree":             "❓ All 3 wrong, all disagree",
        "lr_correct_transformers_wrong":  "🟦 LR correct, both transformers wrong",
        "transformers_correct_lr_wrong":  "🟧 Both transformers correct, LR wrong",
        "partial_disagreement":           "⚪ Partial disagreement",
    }

    f1, f2, f3 = st.columns(3)
    with f1:
        cat = st.selectbox(
            "Agreement category",
            ["(all)"] + list(cat_labels.keys()),
            format_func=lambda x: cat_labels.get(x, x),
        )
    with f2:
        true_cls = st.selectbox(
            "True class",
            ["(all)"] + list(CLASS_NAMES),
        )
    with f3:
        majority = st.selectbox(
            "Majority predicted class",
            ["(all)"] + list(CLASS_NAMES) + ["no_majority"],
        )

    filtered = clips.copy()
    if cat != "(all)":
        filtered = filtered[filtered["category"] == cat]
    if true_cls != "(all)":
        filtered = filtered[filtered["true_label"] == true_cls]
    if majority != "(all)":
        filtered = filtered[filtered["majority_pred"] == majority]

    st.markdown(
        f'<div style="color:{TEXT_MUTED}; font-size:13px; margin:12px 0;">'
        f'Showing <strong style="color:{ACCENT};">{len(filtered)}</strong> of '
        f'{len(clips)} clips '
        f'({len(filtered)/len(clips):.1%})</div>',
        unsafe_allow_html=True,
    )

    if len(filtered) == 0:
        st.warning("No clips match these filters.")
        st.stop()

    show_cols = ["audio_id", "true_label", "m1_pred", "m2_pred", "m3_pred",
                 "majority_pred", "child", "lang"]
    display = filtered[show_cols].rename(columns={
        "audio_id":      "Clip ID",
        "true_label":    "True",
        "m1_pred":       "M1",
        "m2_pred":       "M2",
        "m3_pred":       "M3",
        "majority_pred": "Majority",
        "child":         "Child",
        "lang":          "Lang",
    }).reset_index(drop=True)

    selection = st.dataframe(
        display, hide_index=True, use_container_width=True, height=320,
        on_select="rerun", selection_mode="single-row",
    )

    selected_rows = selection.selection.rows
    if not selected_rows:
        st.info("👆 Click a row in the table to play the audio and view the spectrogram.")
        st.stop()

    row_idx = selected_rows[0]
    clip = filtered.iloc[row_idx]
    audio_id = clip["audio_id"]

    st.markdown("---")
    st.markdown(f"#### Clip: `{audio_id}`")

    info_cols = st.columns(5)
    info_cols[0].metric("True class", clip["true_label"])
    info_cols[1].metric(
        "Model 1", clip["m1_pred"],
        delta="✓" if clip["m1_pred"] == clip["true_label"] else "✗",
    )
    info_cols[2].metric(
        "Model 2", clip["m2_pred"],
        delta="✓" if clip["m2_pred"] == clip["true_label"] else "✗",
    )
    info_cols[3].metric(
        "Model 3", clip["m3_pred"],
        delta="✓" if clip["m3_pred"] == clip["true_label"] else "✗",
    )
    info_cols[4].metric("Child / Lang", f"{clip['child']} / {clip['lang']}")

    # Audio
    wav_path = WAV_DIR / f"{audio_id}.wav"
    if wav_path.exists():
        st.markdown("**🔊 Audio**")
        with open(wav_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")
    else:
        st.warning(f"WAV not found: {wav_path}")

    # Spectrogram
    spec_path = SPEC_DIR / f"{audio_id}.png"
    if spec_path.exists():
        st.markdown("**🎼 Mel spectrogram**")
        st.image(str(spec_path), use_container_width=True)
    else:
        st.caption(f"Spectrogram not found: {spec_path}")

# ===========================================================================
# TAB 4 — Presentation Picks
# ===========================================================================

with tab4:
    st.markdown("### Curated clips for the presentation")
    st.markdown(
        "Four hand-picked clips that tell the story of the project. "
        "Each one is a different category from the error analysis. "
        "Play them in sequence — the contrast between them is the project in 2 minutes."
    )

    # Each entry: (audio_id, true_label, m1, m2, m3, headline, story, takeaway)
    PRESENTATION_CLIPS = [
        {
            "id":       "audio_0445",
            "category": "all_agree_wrong",
            "title":    "All three models agree on the wrong class",
            "true":     "PLTOY",
            "m1":       "INTER", "m2": "INTER", "m3": "INTER",
            "tag":      "Data ceiling",
            "tag_col":  "#EF4444",
            "story": (
                "This clip is annotated as **PLTOY** (playing with toys). "
                "Every one of our three models — a 1,028-parameter linear "
                "classifier, a 126K mean-pool transformer, and a 648K "
                "attention transformer — predicted **INTER** (social interaction). "
                "When all three architectures agree on the wrong class, the "
                "limit is in the data, not the model."
            ),
            "interpretation": (
                "**40.9% of test clips fall in this category.** The Tagalog "
                "training prior (46.9% INTER) pulls every model toward INTER "
                "regardless of what the audio actually contains. No amount of "
                "extra capacity recovers these clips."
            ),
        },
        {
            "id":       "audio_0298",
            "category": "all_agree_wrong",
            "title":    "MOVE clip — every model says INTER",
            "true":     "MOVE",
            "m1":       "INTER", "m2": "INTER", "m3": "INTER",
            "tag":      "Tagalog INTER prior",
            "tag_col":  "#EF4444",
            "story": (
                "This clip is annotated as **MOVE** (movement activity). "
                "All three models predict **INTER** — the same systematic "
                "confusion we just saw with PLTOY."
            ),
            "interpretation": (
                "Confusion pairs in the *all-3-agree-wrong* set: "
                "**PLTOY → INTER (n=13), MOVE → INTER (n=6), EXTEV → INTER (n=5)**. "
                "The confusion direction is always *toward* INTER. This is the "
                "Tagalog training prior in its purest form — when models agree, "
                "they agree on INTER."
            ),
        },
        {
            "id":       "audio_0257",
            "category": "all_agree_correct",
            "title":    "All three models agree — correctly",
            "true":     "INTER",
            "m1":       "INTER", "m2": "INTER", "m3": "INTER",
            "tag":      "Success",
            "tag_col":  COLOR_M3,
            "story": (
                "This clip is annotated as **INTER**, and every model gets it "
                "right. Compare it acoustically to the previous two clips — "
                "the same predicted class, but here it actually *is* INTER."
            ),
            "interpretation": (
                "Only **7.1% of test clips** are correctly classified by all "
                "three models. Most of these are INTER clips (15 of 19) — "
                "consistent with INTER being the highest-F1 class on every "
                "model. The task is not impossible; it just has a high "
                "data-ceiling on the rest of the test set."
            ),
        },
        {
            "id":       "audio_0828",
            "category": "all_agree_wrong",
            "title":    "Even the dominant class can fool every model",
            "true":     "INTER",
            "m1":       "EXTEV", "m2": "EXTEV", "m3": "EXTEV",
            "tag":      "Reverse error",
            "tag_col":  ACCENT,
            "story": (
                "This clip is annotated as **INTER** — the class every model "
                "over-predicts. But here, all three predict **EXTEV**. The "
                "Tagalog prior is overridden by something specific in this "
                "clip's acoustics."
            ),
            "interpretation": (
                "This is the **only INTER clip** where all three models agree "
                "on a wrong class. It shows that the INTER bias is not absolute "
                "— there are clip-level features strong enough to override the "
                "training prior. Just not strong enough on the other 268 clips."
            ),
        },
    ]

    # ── Headline cards ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### The story in one row")
    headline_cols = st.columns(4)
    headline_data = [
        ("40.9%", "Wrong by ALL 3 models",   "Data-ceiling clips"),
        ("7.1%",  "Correct by ALL 3 models", "Genuinely classifiable"),
        ("11.2%", "All 3 agree, all wrong",  "Same prediction, wrong class"),
        ("29.7%", "All 3 wrong, disagree",   "No model has any signal"),
    ]
    for col, (num, label, sub) in zip(headline_cols, headline_data):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-num">{num}</div>'
                f'<div class="metric-lbl">{label}</div>'
                f'<div class="metric-sub" style="color:{TEXT_DIM};">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Clip cards ────────────────────────────────────────────────────
    for i, clip in enumerate(PRESENTATION_CLIPS, start=1):
        # Header row
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:12px; '
            f'margin: 28px 0 12px 0;">'
            f'<div style="background:{SURFACE_2}; color:{ACCENT}; '
            f'padding:4px 12px; border-radius:6px; font-size:11px; '
            f'font-weight:600; letter-spacing:0.06em; text-transform:uppercase; '
            f'border:1px solid {BORDER_HI}; '
            f'font-family:JetBrains Mono,monospace;">CLIP {i}</div>'
            f'<div style="background:{clip["tag_col"]}; color:#0A0E1A; '
            f'padding:4px 12px; border-radius:6px; font-size:11px; '
            f'font-weight:700; letter-spacing:0.04em;">'
            f'{clip["tag"].upper()}</div>'
            f'<div style="font-size:18px; font-weight:600; color:{TEXT};">'
            f'{clip["title"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Two-column layout: audio left, predictions right
        left, right = st.columns([3, 2])

        with left:
            # Audio
            wav_path = WAV_DIR / f"{clip['id']}.wav"
            if wav_path.exists():
                with open(wav_path, "rb") as f:
                    st.audio(f.read(), format="audio/wav")
            else:
                st.warning(f"Audio not found: {wav_path}")

            # Mel spectrogram
            spec_path = SPEC_DIR / f"{clip['id']}.png"
            if spec_path.exists():
                st.image(str(spec_path), use_container_width=True,
                         caption=f"Mel spectrogram — {clip['id']}")

        with right:
            # True/predicted predictions card
            def pred_row(label, value, true_val):
                ok = (value == true_val)
                color = COLOR_M3 if ok else "#EF4444"
                mark = "✓" if ok else "✗"
                return (
                    f'<div style="display:flex; justify-content:space-between; '
                    f'align-items:center; padding:8px 0; '
                    f'border-bottom:1px solid {BORDER};">'
                    f'<div style="color:{TEXT_MUTED}; font-size:12px; '
                    f'font-weight:500;">{label}</div>'
                    f'<div style="display:flex; align-items:center; gap:8px;">'
                    f'<span style="color:{TEXT}; font-weight:600; '
                    f'font-family:JetBrains Mono,monospace; font-size:13px;">'
                    f'{value}</span>'
                    f'<span style="color:{color}; font-weight:700; '
                    f'font-size:14px;">{mark}</span>'
                    f'</div></div>'
                )

            true_html = (
                f'<div style="display:flex; justify-content:space-between; '
                f'align-items:center; padding:8px 0; '
                f'border-bottom:1px solid {BORDER_HI};">'
                f'<div style="color:{ACCENT}; font-size:11px; font-weight:600; '
                f'text-transform:uppercase; letter-spacing:0.08em;">'
                f'True class</div>'
                f'<div style="color:{TEXT}; font-weight:700; '
                f'font-family:JetBrains Mono,monospace; font-size:14px;">'
                f'{clip["true"]}</div></div>'
            )
            st.markdown(
                f'<div style="background:{SURFACE}; border:1px solid {BORDER}; '
                f'border-radius:12px; padding:14px 18px;">'
                f'{true_html}'
                f'{pred_row("Model 1 — LR",          clip["m1"], clip["true"])}'
                f'{pred_row("Model 2 — mean pool",   clip["m2"], clip["true"])}'
                f'{pred_row("Model 3 — attention",   clip["m3"], clip["true"])}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div style="font-size:11px; color:{TEXT_DIM}; '
                f'margin-top:10px; font-family:JetBrains Mono,monospace;">'
                f'audio_id: <span style="color:{TEXT_MUTED};">{clip["id"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Story + interpretation (full width below)
        st.markdown(
            f'<div style="background:{SURFACE}; border:1px solid {BORDER}; '
            f'border-radius:12px; padding:18px 22px; margin-top:14px; '
            f'border-left:3px solid {clip["tag_col"]};">'
            f'<div style="font-size:11px; color:{TEXT_MUTED}; '
            f'text-transform:uppercase; letter-spacing:0.08em; '
            f'font-weight:600; margin-bottom:8px;">What we hear</div>'
            f'<div style="color:{TEXT}; line-height:1.65; font-size:14px; '
            f'margin-bottom:16px;">{clip["story"]}</div>'
            f'<div style="font-size:11px; color:{TEXT_MUTED}; '
            f'text-transform:uppercase; letter-spacing:0.08em; '
            f'font-weight:600; margin-bottom:8px;">What it tells us</div>'
            f'<div style="color:{TEXT_MUTED}; line-height:1.65; font-size:13px;">'
            f'{clip["interpretation"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Closing summary ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### What these four clips together say")
    st.markdown(
        f'<div style="background:linear-gradient(135deg, {SURFACE} 0%, '
        f'{SURFACE_2} 100%); border:1px solid {BORDER_HI}; '
        f'border-radius:14px; padding:24px 28px; margin-top:8px;">'
        f'<div style="color:{TEXT}; line-height:1.75; font-size:15px;">'
        f'These four clips compress the entire project into a single listening '
        f'experience. Two clips show <strong style="color:{ACCENT};">the data '
        f'ceiling</strong> — every architecture, from a 1,028-parameter linear '
        f'model to a 648K-parameter attention transformer, makes the same '
        f'mistake on the same clips. One clip shows a <strong style="color:'
        f'{COLOR_M3};">success case</strong> — proving the task is not '
        f'impossible. The fourth shows the <strong style="color:'
        f'{ACCENT_2};">reverse error direction</strong> — even the '
        f'over-predicted class has clips that fool every model.'
        f'</div>'
        f'<div style="color:{TEXT_MUTED}; line-height:1.65; font-size:13px; '
        f'margin-top:14px;">'
        f'The key finding is not which model wins. It is that the same 110 '
        f'clips defeat every architecture we tried. Adding parameters does '
        f'not break the ceiling. More <em>training children</em> would.'
        f'</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="footer-caption">'
    'aml2026-group-05 · Built with Streamlit · '
    'Numbers from canonical 5-seed multi-seed runs · '
    'Per-clip predictions from seed 42 (deterministic for LR)'
    '</div>',
    unsafe_allow_html=True,
)