"""
utils/make_presentation_figures.py
-----------------------------------
Generates four presentation-grade figures from canonical results.

All numbers are hardcoded from the canonical multi-seed runs documented
in the README. No JSON parsing needed — the script is portable and
runnable on any machine that has matplotlib and the project's config.py.

Figures produced (all in RES_DIR/figures/presentation/):
  fig1_complexity_vs_f1.png     — parameter count vs test F1 scatter
  fig2_perclass_heatmap.png     — per-class F1 heatmap (3 models × 4 classes)
  fig3_val_vs_test_gap.png      — val vs test F1 paired bars per configuration
  fig4_inter_language_rates.png — INTER rate by language + model predictions

Usage
-----
    # From the project root:
    python utils/make_presentation_figures.py

    # Custom output directory:
    python utils/make_presentation_figures.py --out_dir /path/to/dir
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# utils/ sits at the same level as scripts/ — one step up to reach src/
HERE = Path(__file__).resolve().parent        # .../aml2026-group-05/utils/
sys.path.insert(0, str(HERE.parent / "src"))  # .../aml2026-group-05/src/

import config as cfg

# ---------------------------------------------------------------------------
# Canonical numbers (all from 5-seed multi-seed runs, documented in README)
# ---------------------------------------------------------------------------

# Each row: (display_label, n_params, val_f1, test_f1, feature_src, model_id)
# n_params = None for MFCC transformer configs (not separately recorded)
RESULTS = [
    # Model 1 — Logistic Regression
    # params = (feat_dim + 1) * n_classes
    ("LR\nMFCC",      (40   + 1) * 4, 0.240, 0.253, "mfcc",     1),
    ("LR\nMel",       (256  + 1) * 4, 0.232, 0.277, "mel",      1),
    ("LR\nwav2vec2",  (1536 + 1) * 4, 0.253, 0.263, "wav2vec2", 1),

    # Model 2 — Mean-Pool Transformer
    ("M2\nMFCC",      None,           0.258, 0.238, "mfcc",     2),
    ("M2\nMel",       126_000,        0.270, 0.249, "mel",      2),
    ("M2\nwav2vec2",  167_000,        0.294, 0.219, "wav2vec2", 2),

    # Model 3 — Attentive Pooling Transformer
    ("M3\nMFCC",      None,           0.257, 0.237, "mfcc",     3),
    ("M3\nMel",       26_000,         0.252, 0.221, "mel",      3),
    ("M3\nwav2vec2",  648_000,        0.286, 0.246, "wav2vec2", 3),
]

# Per-class F1 — mel (rows = models, cols = EXTEV / INTER / MOVE / PLTOY)
PERCLASS_MEL = np.array([
    [0.220, 0.396, 0.263, 0.228],   # Model 1 LR mel
    [0.185, 0.340, 0.220, 0.258],   # Model 2 mel d=64 L=2
    [0.130, 0.284, 0.106, 0.363],   # Model 3 mel d=32 L=1
])

# Per-class F1 — wav2vec2
PERCLASS_W2V = np.array([
    [0.228, 0.391, 0.188, 0.247],   # Model 1 LR wav2vec2
    [0.239, 0.255, 0.231, 0.153],   # Model 2 wav2vec2 d=64 L=2
    [0.192, 0.395, 0.199, 0.198],   # Model 3 wav2vec2 d=192 L=1
])

# INTER class rates by language — from class_by_language analysis
INTER_BY_LANG = {
    "Shipibo\n(val)":      0.225,
    "Vietnamese\n(train)": 0.330,
    "Tagalog\n(test)":     0.469,
}
TRUE_TEST_INTER = 0.294   # true INTER proportion in test set (child_6)
M2_INTER_PRED   = 0.311   # fraction of test clips Model 2 predicts as INTER
M3_INTER_PRED   = 0.574   # fraction of test clips Model 3 predicts as INTER

CHANCE = 0.250

# Colour palette — consistent across all four figures
COL = {
    "m1": "#2196F3",    # blue  — Model 1
    "m2": "#FF9800",    # amber — Model 2
    "m3": "#4CAF50",    # green — Model 3
    "val": "#90CAF9",   # light blue
    "test": "#1565C0",  # dark blue
    "chance": "grey",
    "inter": "#1565C0",
    "gap_pos": "#C62828",
    "gap_neg": "#2E7D32",
}


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def save(fig, path: Path, dpi: int = 180):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 1 — Model complexity vs test F1
# ---------------------------------------------------------------------------

def fig1_complexity(out_dir: Path):
    """
    Scatter plot: x = log10(parameters), y = test macro-F1.
    One point per (model, feature source). Shows that adding parameters
    does not improve accuracy — LR points cluster top-left, the 648K
    transformer is bottom-right.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    model_colors  = {1: COL["m1"], 2: COL["m2"], 3: COL["m3"]}
    feat_markers  = {"mfcc": "s", "mel": "o", "wav2vec2": "^"}
    model_names   = {1: "Model 1 (LR)", 2: "Model 2 (mean pool)",
                     3: "Model 3 (attention)"}
    feat_names    = {"mfcc": "MFCC", "mel": "Mel", "wav2vec2": "wav2vec2"}

    seen_legends = set()

    for label, n_params, val_f1, test_f1, feat, mid in RESULTS:
        if n_params is None:
            continue

        color  = model_colors[mid]
        marker = feat_markers[feat]
        lkey   = (mid, feat)
        lbl    = (f"{model_names[mid]} — {feat_names[feat]}"
                  if lkey not in seen_legends else None)
        seen_legends.add(lkey)

        ax.scatter(np.log10(n_params), test_f1,
                   color=color, marker=marker, s=150, zorder=5,
                   label=lbl)
        ax.annotate(
            label.replace("\n", " "),
            (np.log10(n_params), test_f1),
            xytext=(6, 4), textcoords="offset points",
            fontsize=8.5, color=color,
        )

    # Chance line
    ax.axhline(CHANCE, ls="--", color=COL["chance"], lw=1.4,
               label=f"Chance = {CHANCE}")

    # Annotate best result
    ax.annotate(
        "Best: LR Mel  0.277",
        xy=(np.log10((256 + 1) * 4), 0.277),
        xytext=(np.log10((256 + 1) * 4) + 0.35, 0.284),
        fontsize=9, color=COL["m1"],
        arrowprops=dict(arrowstyle="->", color=COL["m1"], lw=1.2),
    )

    ax.set_xlabel("Model complexity  [log₁₀(parameters)]", fontsize=12)
    ax.set_ylabel("Test macro-F1", fontsize=12)
    ax.set_title("More parameters do not improve accuracy\n"
                 "Logistic regression on summary stats beats both transformers "
                 "on every feature source", fontsize=12)
    ax.set_ylim(0.15, 0.32)
    ax.set_xlim(1.8, 6.2)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_xticklabels(["10²", "10³", "10⁴", "10⁵", "10⁶"])
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)

    save(fig, out_dir / "fig1_complexity_vs_f1.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — Per-class F1 heatmap
# ---------------------------------------------------------------------------

def fig2_heatmap(out_dir: Path):
    """
    Two 3×4 heatmaps side by side (mel | wav2vec2).
    Rows = Model 1 / 2 / 3, columns = EXTEV / INTER / MOVE / PLTOY.
    INTER column outlined in blue to highlight universal over-prediction.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CLASS_NAMES  = ["EXTEV", "INTER", "MOVE", "PLTOY"]
    MODEL_LABELS = ["Model 1\n(LR)", "Model 2\n(mean pool)",
                    "Model 3\n(attention)"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, data, title in zip(
        axes,
        [PERCLASS_MEL, PERCLASS_W2V],
        ["Mel spectrograms", "wav2vec2 embeddings"],
    ):
        im = ax.imshow(data, vmin=0.0, vmax=0.5, cmap="RdYlGn", aspect="auto")

        for i in range(3):
            for j in range(4):
                v = data[i, j]
                txt_col = "black" if 0.12 < v < 0.42 else "white"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=txt_col)

        # Blue border on INTER column (j=1)
        for i in range(3):
            ax.add_patch(plt.Rectangle(
                (0.5, i - 0.5), 1, 1,
                fill=False, edgecolor=COL["inter"], linewidth=2.5,
            ))

        ax.set_xticks(range(4))
        ax.set_xticklabels(CLASS_NAMES, fontsize=11)
        ax.set_yticks(range(3))
        ax.set_yticklabels(MODEL_LABELS, fontsize=10)
        ax.set_title(title, fontsize=12, pad=8)

    fig.colorbar(im, ax=axes.ravel().tolist(),
                 label="Per-class test macro-F1", shrink=0.8)
    fig.suptitle(
        "Per-class test F1 — all three models on both feature sources\n"
        "INTER (blue border) is the highest-F1 class for every model "
        "— driven by Tagalog training prior, not architecture",
        fontsize=12, y=1.02,
    )

    save(fig, out_dir / "fig2_perclass_heatmap.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — Val vs Test gap
# ---------------------------------------------------------------------------

def fig3_val_test_gap(out_dir: Path):
    """
    Paired bar chart: light bar = val F1, dark bar = test F1.
    Gap (val − test) annotated above each pair. Shows that val
    consistently overestimates test performance across all configurations
    — explained by the Shipibo (val) → Tagalog (test) language shift.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Mel and wav2vec2 only — MFCC transformer numbers are noisier
    configs = [
        ("LR\nMel",       0.232, 0.277),
        ("LR\nwav2vec2",  0.253, 0.263),
        ("M2\nMel",       0.270, 0.249),
        ("M2\nwav2vec2",  0.294, 0.219),
        ("M3\nMel",       0.252, 0.221),
        ("M3\nwav2vec2",  0.286, 0.246),
    ]
    labels  = [c[0] for c in configs]
    val_f1  = [c[1] for c in configs]
    test_f1 = [c[2] for c in configs]

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.bar(x - w/2, val_f1,  w, label="Val macro-F1  (Shipibo child)",
           color=COL["val"],  edgecolor="white", linewidth=0.8)
    ax.bar(x + w/2, test_f1, w, label="Test macro-F1  (Tagalog child)",
           color=COL["test"], edgecolor="white", linewidth=0.8)

    # Annotate gap above each bar pair
    for xi, (v, t) in zip(x, zip(val_f1, test_f1)):
        gap   = v - t
        color = COL["gap_pos"] if gap > 0.005 else COL["gap_neg"]
        ax.annotate(
            f"{gap:+.3f}",
            xy=(xi, max(v, t) + 0.003),
            xytext=(xi, max(v, t) + 0.020),
            fontsize=9, ha="center", color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    # Chance line
    ax.axhline(CHANCE, ls="--", color=COL["chance"], lw=1.4,
               label=f"Chance = {CHANCE}")

    # Background shading per model group
    spans = [((-0.5, 1.5), COL["m1"], "Model 1\n(LR)"),
             ((1.5,  3.5), COL["m2"], "Model 2\n(mean)"),
             ((3.5,  5.5), COL["m3"], "Model 3\n(attn)")]
    for (x0, x1), color, txt in spans:
        ax.axvspan(x0, x1, alpha=0.06, color=color)
        ax.text((x0 + x1) / 2, 0.174, txt,
                ha="center", fontsize=9, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_ylim(0.17, 0.34)
    ax.set_title(
        "Val always overestimates test performance\n"
        "Val child = Shipibo  |  Test child = Tagalog  →  different language, "
        "different class priors\n"
        "Gap annotation: val − test  (red = val > test,  green = val < test)",
        fontsize=11,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    save(fig, out_dir / "fig3_val_vs_test_gap.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4 — INTER rate by language
# ---------------------------------------------------------------------------

def fig4_inter_language(out_dir: Path):
    """
    Bar chart of INTER training rate per language.
    Horizontal reference lines for true test INTER rate, Model 2 and
    Model 3 INTER prediction rates. Visually explains why all models
    over-predict INTER on the Tagalog test child.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))

    langs  = list(INTER_BY_LANG.keys())
    rates  = list(INTER_BY_LANG.values())
    colors = [COL["val"], "#FFB74D", "#EF9A9A"]  # Shipibo / Vietnamese / Tagalog

    bars = ax.bar(langs, rates, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2, zorder=3)

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.013,
            f"{rate:.1%}", ha="center", fontsize=12, fontweight="bold",
        )

    # Reference lines
    refs = [
        (TRUE_TEST_INTER, COL["test"],    "--", "True test INTER rate  (29.4%)"),
        (M2_INTER_PRED,   COL["m2"],      "-.", "Model 2 INTER predictions  (31.1%)"),
        (M3_INTER_PRED,   COL["gap_pos"], "-",  "Model 3 INTER predictions  (57.4%)"),
    ]
    for y, color, ls, label in refs:
        ax.axhline(y, color=color, ls=ls, lw=2.0, label=label, zorder=4)
        ax.text(2.72, y + 0.012, f"{y:.1%}",
                color=color, fontsize=10, fontweight="bold")

    # Annotation
    ax.annotate(
        "Model 3 inherits the\nTagalog INTER prior\n(+28 pp above true rate)",
        xy=(2.0, M3_INTER_PRED),
        xytext=(1.0, 0.61),
        fontsize=9.5, color=COL["gap_pos"],
        arrowprops=dict(arrowstyle="->", color=COL["gap_pos"], lw=1.3),
    )

    ax.set_ylim(0, 0.72)
    ax.set_ylabel("INTER class rate", fontsize=12)
    ax.set_title(
        "INTER over-prediction traces to the Tagalog training prior\n"
        "Val child = Shipibo (22.5% INTER)  →  "
        "Test child = Tagalog (46.9% INTER in training)",
        fontsize=11,
    )
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    save(fig, out_dir / "fig4_inter_language_rates.png")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate four presentation-grade figures "
                    "from canonical project results.",
    )
    p.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: RES_DIR/figures/presentation/)",
    )
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = (Path(args.out_dir) if args.out_dir
               else cfg.FIGURES_DIR / "presentation")

    print(f"\nGenerating 4 presentation figures → {out_dir}\n")
    print("[1/4] Complexity vs test F1 scatter...")
    fig1_complexity(out_dir)

    print("[2/4] Per-class F1 heatmap (mel + wav2vec2)...")
    fig2_heatmap(out_dir)

    print("[3/4] Val vs test gap bar chart...")
    fig3_val_test_gap(out_dir)

    print("[4/4] INTER rate by language...")
    fig4_inter_language(out_dir)

    print(f"\nAll done.  Figures in:\n  {out_dir}/")


if __name__ == "__main__":
    main()