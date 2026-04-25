"""
experiments/error_analysis.py
------------------------------
Identifies which test clips are misclassified by all three models
(Model 1 LR mel, Model 2 transformer mel, Model 3 transformer mel)
and examines whether errors are shared or model-specific.

Why this experiment
-------------------
If all three models misclassify the same clips, errors are data-driven
(ambiguous clips, no discriminative acoustic signal) rather than
architecture-driven. If errors are disjoint, the architectures genuinely
make different trade-offs.

Fix notes vs v1
---------------
- Neural model JSONs store predictions as integers (0-3) or integer
  strings; LR JSON stores them as class-name strings. All predictions
  are now normalised to class-name strings via LABEL_DECODER before
  comparison.
- plot_error_by_class column name bug fixed (was looking for 'all3_wrong'
  column that doesn't exist; now uses n_wrong counts correctly).
- max_iter bumped to 5000 in LOCO to suppress convergence warnings.

Outputs
-------
RES_DIR/results/error_analysis_<timestamp>.json   — full per-clip table
RES_DIR/figures/error_analysis_<timestamp>_overlap.png  — error overlap
RES_DIR/figures/error_analysis_<timestamp>_by_class.png — error by class
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import config as cfg
from config import CLASS_NAMES, FIGURES_DIR, LABEL_DECODER, RESULTS_DIR
from data_utils import build_master_table, determine_eval_protocol, load_splits


# ---------------------------------------------------------------------------
# JSON auto-detection
# ---------------------------------------------------------------------------

def latest_json(pattern: str) -> Path:
    matches = sorted(RESULTS_DIR.glob(pattern),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' in {RESULTS_DIR}. "
            f"Run the corresponding training script first."
        )
    return matches[0]


def normalise_labels(arr) -> np.ndarray:
    """
    Convert an array of predictions/labels to CLASS_NAME strings.

    Neural model JSONs may store integer labels (0, 1, 2, 3) or
    integer-valued strings ('0', '1', '2', '3'). LR JSONs store
    class-name strings ('EXTEV', 'INTER', 'MOVE', 'PLTOY').
    This function normalises all three formats to class-name strings
    so that cross-model comparison is always apples-to-apples.
    """
    result = []
    for v in arr:
        if isinstance(v, str) and v in CLASS_NAMES:
            result.append(v)
        else:
            # Try integer or integer-string lookup via LABEL_DECODER
            try:
                result.append(LABEL_DECODER[int(v)])
            except (KeyError, ValueError):
                result.append(str(v))
    return np.array(result)


def load_seed0_preds(json_path: Path, seed_index: int = 0) -> tuple:
    """
    Return (preds, labels) as normalised class-name string arrays
    for the given seed index.
    """
    with open(json_path) as f:
        data = json.load(f)

    summaries = data.get("per_seed_summaries", [])
    if not summaries:
        raise ValueError(f"No per_seed_summaries found in {json_path}")

    seed_data = summaries[seed_index]
    fold0  = seed_data.get("fold_results", [{}])[0]

    raw_preds  = fold0.get("test_preds",  seed_data.get("test_preds",  []))
    raw_labels = fold0.get("test_labels", seed_data.get("test_labels", []))

    if not raw_preds or not raw_labels:
        raise ValueError(f"Empty predictions/labels in {json_path}")

    preds  = normalise_labels(raw_preds)
    labels = normalise_labels(raw_labels)
    return preds, labels


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overlap(sets: dict, n_total: int, save_path: Path):
    """Bar chart of error overlap categories."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        categories = [
            ("All 3 wrong\n(data ceiling)",      len(sets["all3"]),        "#C62828"),
            ("M1 + M2 wrong\n(not M3)",          len(sets["m1_m2"]),       "#EF9A9A"),
            ("M1 + M3 wrong\n(not M2)",          len(sets["m1_m3"]),       "#EF9A9A"),
            ("M2 + M3 wrong\n(not M1)",          len(sets["m2_m3"]),       "#EF9A9A"),
            ("M1 only wrong",                     len(sets["m1_only"]),     "#90CAF9"),
            ("M2 only wrong",                     len(sets["m2_only"]),     "#FFB74D"),
            ("M3 only wrong",                     len(sets["m3_only"]),     "#A5D6A7"),
            ("All 3 correct",                     len(sets["all_correct"]), "#C8E6C9"),
        ]
        labels = [c[0] for c in categories]
        vals   = [c[1] for c in categories]
        colors = [c[2] for c in categories]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        bars = ax.barh(labels, vals, color=colors,
                       edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_width() + 0.5,
                        bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", fontsize=10,
                        fontweight="bold")

        ax.set_xlabel(
            f"Number of test clips  "
            f"(total = {n_total}, wrong by ≥1 model = {len(sets['any_wrong'])})",
            fontsize=11,
        )
        ax.set_title(
            "Error overlap across three models — mel feature source\n"
            "Clips in 'All 3 wrong' are likely data-ceiling cases",
            fontsize=12,
        )
        ax.set_xlim(0, max(vals) * 1.18)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Plot → {save_path}")
    except Exception as e:
        print(f"Overlap plot failed: {e}")


def plot_error_by_class(error_df: pd.DataFrame, save_path: Path):
    """
    Grouped bar chart: for each true class, how many clips are wrong by
    3 / 2 / 1 / 0 models.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_wrong_groups = [
            ("All 3 wrong",  3, "#C62828"),
            ("2 models wrong", 2, "#EF9A9A"),
            ("1 model wrong",  1, "#90CAF9"),
            ("All correct",    0, "#C8E6C9"),
        ]

        x     = np.arange(len(CLASS_NAMES))
        width = 0.20
        fig, ax = plt.subplots(figsize=(9, 5.5))

        for i, (label, n_val, color) in enumerate(n_wrong_groups):
            counts = [
                int((error_df[error_df["true_label"] == cls]["n_wrong"] == n_val).sum())
                for cls in CLASS_NAMES
            ]
            ax.bar(x + i * width, counts, width,
                   label=label, color=color,
                   edgecolor="white", linewidth=0.8)

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(CLASS_NAMES, fontsize=11)
        ax.set_ylabel("Number of test clips", fontsize=11)
        ax.set_title(
            "Error distribution by true class — mel feature source\n"
            "How many clips per class are wrong by all 3 / 2 / 1 / 0 models",
            fontsize=12,
        )
        ax.legend(fontsize=9.5, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Plot → {save_path}")
    except Exception as e:
        print(f"Class error plot failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Error analysis across all three models on mel features.")
    p.add_argument("--m1_json", type=str, default=None,
                   help="Model 1 mel multiseed JSON (default: auto-detect latest)")
    p.add_argument("--m2_json", type=str, default=None,
                   help="Model 2 mel audio multiseed JSON (default: auto-detect latest)")
    p.add_argument("--m3_json", type=str, default=None,
                   help="Model 3 mel attn multiseed JSON (default: auto-detect latest)")
    p.add_argument("--seed_index", type=int, default=0,
                   help="Which seed to use (default: 0 = seed 42)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg.make_output_dirs()

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS — Model 1 / 2 / 3 on mel features")
    print("=" * 60)

    # ── Locate JSON files ─────────────────────────────────────────────
    m1_path = Path(args.m1_json) if args.m1_json else \
              latest_json("lr_baseline_mel_multiseed_*.json")

    # For M2 and M3 we need the MEL runs specifically.
    # auto-detect picks most-recent; user should override with --m2_json
    # if the most-recent audio/attn JSON is a wav2vec2 run.
    m2_path = Path(args.m2_json) if args.m2_json else \
              latest_json("audio_multiseed_*.json")
    m3_path = Path(args.m3_json) if args.m3_json else \
              latest_json("attn_multiseed_*.json")

    print(f"\n  Model 1 JSON : {m1_path.name}")
    print(f"  Model 2 JSON : {m2_path.name}")
    print(f"  Model 3 JSON : {m3_path.name}")
    print(f"  Seed index   : {args.seed_index}")

    # ── Load and normalise predictions ────────────────────────────────
    print("\n[1/3] Loading and normalising predictions...")
    m1_preds, labels = load_seed0_preds(m1_path, args.seed_index)
    m2_preds, _      = load_seed0_preds(m2_path, args.seed_index)
    m3_preds, _      = load_seed0_preds(m3_path, args.seed_index)

    n = len(labels)
    assert len(m1_preds) == n, f"M1 preds {len(m1_preds)} ≠ labels {n}"
    assert len(m2_preds) == n, f"M2 preds {len(m2_preds)} ≠ labels {n}"
    assert len(m3_preds) == n, f"M3 preds {len(m3_preds)} ≠ labels {n}"

    # Verify normalisation worked
    for name, arr in [("M1 preds", m1_preds), ("M2 preds", m2_preds),
                      ("M3 preds", m3_preds), ("labels",   labels)]:
        unknown = [v for v in arr if v not in CLASS_NAMES]
        if unknown:
            print(f"  WARN: {name} contains unrecognised values: "
                  f"{set(unknown)} — check JSON format")

    print(f"  {n} test clips loaded and normalised for all three models.")
    print(f"  Label sample: {labels[:5].tolist()}")
    print(f"  M1 pred sample: {m1_preds[:5].tolist()}")
    print(f"  M2 pred sample: {m2_preds[:5].tolist()}")
    print(f"  M3 pred sample: {m3_preds[:5].tolist()}")

    # ── Load test metadata ────────────────────────────────────────────
    df       = build_master_table(verbose=False)
    protocol = determine_eval_protocol(df)
    folds    = load_splits(protocol)
    _, _, test_df = folds[0]
    test_df = test_df.reset_index(drop=True)

    # ── Build per-clip table ──────────────────────────────────────────
    print("\n[2/3] Building error table...")

    m1_wrong = (m1_preds != labels)
    m2_wrong = (m2_preds != labels)
    m3_wrong = (m3_preds != labels)
    n_wrong  = (m1_wrong.astype(int)
                + m2_wrong.astype(int)
                + m3_wrong.astype(int))

    error_df = pd.DataFrame({
        "audio_id":   test_df["audio_ID"].values[:n],
        "child":      test_df["child"].values[:n],
        "lang":       test_df["lang"].values[:n],
        "true_label": labels,
        "m1_pred":    m1_preds,
        "m2_pred":    m2_preds,
        "m3_pred":    m3_preds,
        "m1_wrong":   m1_wrong,
        "m2_wrong":   m2_wrong,
        "m3_wrong":   m3_wrong,
        "n_wrong":    n_wrong,
    })

    # ── Error overlap sets ────────────────────────────────────────────
    idx_m1 = set(np.where(m1_wrong)[0])
    idx_m2 = set(np.where(m2_wrong)[0])
    idx_m3 = set(np.where(m3_wrong)[0])

    sets = {
        "all3":        idx_m1 & idx_m2 & idx_m3,
        "m1_m2":       (idx_m1 & idx_m2) - idx_m3,
        "m1_m3":       (idx_m1 & idx_m3) - idx_m2,
        "m2_m3":       (idx_m2 & idx_m3) - idx_m1,
        "m1_only":     idx_m1 - idx_m2 - idx_m3,
        "m2_only":     idx_m2 - idx_m1 - idx_m3,
        "m3_only":     idx_m3 - idx_m1 - idx_m2,
        "all_correct": set(range(n)) - idx_m1 - idx_m2 - idx_m3,
        "any_wrong":   idx_m1 | idx_m2 | idx_m3,
    }

    # ── Print summary ─────────────────────────────────────────────────
    all3_df            = error_df[error_df["n_wrong"] == 3]
    data_ceiling_pct   = len(sets["all3"]) / n * 100

    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS SUMMARY  (seed index {args.seed_index})")
    print(f"{'='*60}")
    print(f"  Total test clips      : {n}")
    print(f"  Wrong by ≥ 1 model    : {len(sets['any_wrong'])}  "
          f"({len(sets['any_wrong'])/n:.1%})")
    print(f"  All 3 wrong           : {len(sets['all3'])}  "
          f"({data_ceiling_pct:.1f}%) ← likely data ceiling")
    print(f"  M1 + M2 wrong (not M3): {len(sets['m1_m2'])}")
    print(f"  M1 + M3 wrong (not M2): {len(sets['m1_m3'])}")
    print(f"  M2 + M3 wrong (not M1): {len(sets['m2_m3'])}")
    print(f"  M1 only wrong         : {len(sets['m1_only'])}")
    print(f"  M2 only wrong         : {len(sets['m2_only'])}")
    print(f"  M3 only wrong         : {len(sets['m3_only'])}")
    print(f"  All 3 correct         : {len(sets['all_correct'])}  "
          f"({len(sets['all_correct'])/n:.1%})")

    print(f"\n  Error rates per model:")
    for name, wrong in [("M1 (LR mel)",        m1_wrong),
                         ("M2 (mean pool mel)", m2_wrong),
                         ("M3 (attn mel)",      m3_wrong)]:
        print(f"    {name:<22}: {wrong.sum():3d} wrong  ({wrong.sum()/n:.1%})")

    print(f"\n  'All 3 wrong' clips by true class:")
    for c in CLASS_NAMES:
        count = (all3_df["true_label"] == c).sum()
        total = (error_df["true_label"] == c).sum()
        print(f"    {c:<6}: {count:3d} / {total}  "
              f"({count/max(total,1):.0%} of {c} test clips all-wrong)")

    print(f"\n  Most common confusion pairs (all-3-wrong clips, M1 prediction):")
    if len(all3_df) > 0:
        pairs = (all3_df.groupby(["true_label", "m1_pred"])
                 .size().reset_index(name="count")
                 .sort_values("count", ascending=False).head(8))
        for _, row in pairs.iterrows():
            print(f"    {row['true_label']} → {row['m1_pred']}  "
                  f"(n={row['count']})")
    print(f"{'='*60}")

    # ── Save ──────────────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"error_analysis_{ts}"

    out = {
        "experiment":   "Error analysis — Model 1/2/3 mel",
        "seed_index":   args.seed_index,
        "json_paths":   {"m1": str(m1_path),
                          "m2": str(m2_path),
                          "m3": str(m3_path)},
        "n_test":       n,
        "error_overlap": {k: len(v) for k, v in sets.items()},
        "error_rates":  {"m1": int(m1_wrong.sum()),
                          "m2": int(m2_wrong.sum()),
                          "m3": int(m3_wrong.sum())},
        "all3_wrong_by_class": {
            c: int((all3_df["true_label"] == c).sum()) for c in CLASS_NAMES
        },
        "per_clip": error_df.to_dict(orient="records"),
    }
    result_path = RESULTS_DIR / f"{tag}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults → {result_path}")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n[3/3] Plotting...")
    plot_overlap(sets, n, FIGURES_DIR / f"{tag}_overlap.png")
    plot_error_by_class(error_df, FIGURES_DIR / f"{tag}_by_class.png")
    print("\nDone.")


if __name__ == "__main__":
    main()