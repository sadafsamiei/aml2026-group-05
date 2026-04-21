"""
scripts/cm_comparison.py
-------------------------
Side-by-side confusion matrix comparison: Model 2 vs Model 3 on wav2vec2.

Uses the canonical multi-seed JSONs so the figure is guaranteed to match
the numbers in the README. Both panels share the same colour scale so
the visual comparison is honest (the default per-panel autoscale
exaggerates whichever model has higher contrast).

Inputs
------
  /home/renku/work/data/results/audio_multiseed_20260421_125550.json
  /home/renku/work/data/results/attn_multiseed_20260421_000648.json

Output
------
  RES_DIR/figures/cm_comparison_wav2vec2_<timestamp>.png
      Single figure, two panels, shared 0–0.65 colourbar.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import config as cfg
from config import CLASS_NAMES, FIGURES_DIR, RESULTS_DIR


# ---------------------------------------------------------------------------
# Canonical multi-seed JSONs — update here if the canonical runs change
# ---------------------------------------------------------------------------

JSON_MODEL2 = "/home/renku/work/data/results/audio_multiseed_20260421_125550.json"
JSON_MODEL3 = "/home/renku/work/data/results/attn_multiseed_20260421_000648.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_per_seed_cm_from_multiseed(json_path):
    """
    Each per_seed_summary contains test_preds and test_labels in its
    fold_results[0]. Build a list of row-normalised per-seed CMs.
    """
    with open(json_path) as f:
        d = json.load(f)

    per_seed_cms = []
    per_seed_f1s = []
    for ps in d["per_seed_summaries"]:
        fold0 = ps["fold_results"][0]
        preds  = np.array(fold0["test_preds"])
        labels = np.array(fold0["test_labels"])
        cm = confusion_matrix(labels, preds,
                               labels=list(range(len(CLASS_NAMES))),
                               normalize="true")
        per_seed_cms.append(cm)
        per_seed_f1s.append(fold0["test_macro_f1"])

    mean_cm = np.stack(per_seed_cms, axis=0).mean(axis=0)
    std_cm  = np.stack(per_seed_cms, axis=0).std (axis=0)
    return {
        "per_seed_cms": per_seed_cms,
        "mean_cm": mean_cm,
        "std_cm":  std_cm,
        "per_seed_f1": per_seed_f1s,
        "test_f1_mean": d["test_macro_f1"]["mean"],
        "test_f1_std":  d["test_macro_f1"]["std"],
        "hparams": d["hparams"],
    }


def plot_cm_panel(ax, mean_cm, std_cm, title, vmin, vmax, show_cbar=False):
    """Render one CM panel with fixed colour scale."""
    im = ax.imshow(mean_cm, cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class",      fontsize=12)
    ax.set_title(title, fontsize=12, pad=10)

    # Annotations: mean in large, std in small. White text on dark cells.
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            val = mean_cm[i, j]
            std = std_cm[i, j]
            # Use white text when cell is dark
            text_color = "white" if val > (vmin + 0.55 * (vmax - vmin)) else "black"
            ax.text(j, i - 0.08, f"{val:.2f}",
                    ha="center", va="center",
                    color=text_color, fontsize=13, fontweight="bold")
            ax.text(j, i + 0.18, f"±{std:.2f}",
                    ha="center", va="center",
                    color=text_color, fontsize=9, alpha=0.85)

    return im


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    m2 = load_per_seed_cm_from_multiseed(JSON_MODEL2)
    m3 = load_per_seed_cm_from_multiseed(JSON_MODEL3)

    print(f"Model 2 (audio, d={m2['hparams']['d_model']}, L={m2['hparams']['num_layers']}): "
          f"test F1 = {m2['test_f1_mean']:.4f} ± {m2['test_f1_std']:.4f}")
    print(f"Model 3 (attn,  d={m3['hparams']['d_model']}, L={m3['hparams']['num_layers']}): "
          f"test F1 = {m3['test_f1_mean']:.4f} ± {m3['test_f1_std']:.4f}")

    # Shared colour scale. Take max of both means so neither panel gets
    # clipped; round up to a clean number.
    max_val = max(m2["mean_cm"].max(), m3["mean_cm"].max())
    vmax = float(np.ceil(max_val * 10) / 10)
    vmin = 0.0
    print(f"\nShared colour scale: [{vmin}, {vmax}]")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    im1 = plot_cm_panel(
        axes[0], m2["mean_cm"], m2["std_cm"],
        f"Model 2 — mean pooling  (d={m2['hparams']['d_model']}, L={m2['hparams']['num_layers']})\n"
        f"test macro-F1 = {m2['test_f1_mean']:.3f} ± {m2['test_f1_std']:.3f}",
        vmin=vmin, vmax=vmax, show_cbar=False,
    )
    im2 = plot_cm_panel(
        axes[1], m3["mean_cm"], m3["std_cm"],
        f"Model 3 — attentive pooling  (d={m3['hparams']['d_model']}, L={m3['hparams']['num_layers']})\n"
        f"test macro-F1 = {m3['test_f1_mean']:.3f} ± {m3['test_f1_std']:.3f}",
        vmin=vmin, vmax=vmax, show_cbar=False,
    )

    # Single shared colourbar on the right
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.70])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label("Fraction of true-class clips\n(row-normalised, mean over 5 seeds)",
                    fontsize=10)

    fig.suptitle(
        "Confusion matrices on held-out child_6 — wav2vec2 features, 5-seed means\n"
        "Shared colour scale makes the INTER-prediction bias of Model 3 directly visible",
        fontsize=13, y=1.02,
    )

    plt.tight_layout(rect=(0, 0, 0.90, 0.96))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = FIGURES_DIR / f"cm_comparison_wav2vec2_{timestamp}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure → {out_path}")

    # ── Sidebar: print the "INTER bias" metric ─────────────────────────
    # For each row, what fraction of predictions went to INTER?
    inter_col = CLASS_NAMES.index("INTER")
    print(f"\n{'='*62}")
    print("INTER-prediction bias  (row-wise weight on INTER column)")
    print(f"{'='*62}")
    print(f"  {'Class':<8} {'Model 2':>12} {'Model 3':>12} {'Δ':>10}")
    print("  " + "-"*46)
    for i, cname in enumerate(CLASS_NAMES):
        p2 = m2["mean_cm"][i, inter_col]
        p3 = m3["mean_cm"][i, inter_col]
        print(f"  {cname:<8} {p2:>12.3f} {p3:>12.3f} {p3 - p2:>+10.3f}")
    print(f"{'='*62}")
    non_inter_rows = [i for i, c in enumerate(CLASS_NAMES) if c != "INTER"]
    m2_bias = np.mean([m2["mean_cm"][i, inter_col] for i in non_inter_rows])
    m3_bias = np.mean([m3["mean_cm"][i, inter_col] for i in non_inter_rows])
    print(f"\n  Avg non-INTER-row INTER prediction rate:")
    print(f"    Model 2: {m2_bias:.3f}")
    print(f"    Model 3: {m3_bias:.3f}  (Δ = {m3_bias - m2_bias:+.3f})")
    print(f"  → Model 3 predicts INTER on {m3_bias:.0%} of clips whose true "
          f"class is NOT INTER.")


if __name__ == "__main__":
    main()