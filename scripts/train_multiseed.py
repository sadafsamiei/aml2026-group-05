"""
train_multiseed.py
------------------
Multi-seed wrapper around the existing single-run trainers for Models 2 and 3.

Runs the same model+hparams across N seeds, then aggregates val/test
macro-F1 (mean ± std) and produces overlay plots that combine all seeds
into a single figure for each diagnostic.

Why
---
Single-seed reporting on n_test=269 is not a defensible estimate of
macro-F1. The Model 2 vs Model 3 delta from a single seed is well within
sampling noise. This wrapper produces the mean ± std reporting that the
final writeup needs, without changing models.py / data_utils.py /
train_utils.py / the per-model training scripts.

Usage
-----
    # Model 3 (attentive pooling)
    python scripts/train_multiseed.py --model attn  --seeds 42 123 456 \
        --use_existing_splits --baseline_test_f1 0.267

    # Model 2 (audio-only)
    python scripts/train_multiseed.py --model audio --seeds 42 123 456 \
        --use_existing_splits

Standard hyperparameter overrides (--d_model, --num_layers, ...) are
forwarded to the underlying trainer. The --grid_search flag is NOT
supported here — multi-seed wraps a *single* config.

Outputs (per run, tagged by run_tag = "{model}_multiseed_{timestamp}")
----------------------------------------------------------------------
  RES_DIR/results/{run_tag}.json
      Aggregated mean ± std across seeds, plus full per-seed fold results
      nested inside.

  RES_DIR/figures/{run_tag}_training_curves.png
      Two-panel overlay: train loss + val macro-F1, one line per seed.

  RES_DIR/figures/{run_tag}_cm_val.png
  RES_DIR/figures/{run_tag}_cm_test.png
      Mean of per-seed row-normalised confusion matrices.

  RES_DIR/figures/{run_tag}_attn_overlay_EXTEV.png   (Model 3 only)
      EXTEV-class attention weight profile from one held-out val example,
      one line per seed, with uniform reference dashed.

Per-seed checkpoints, per-seed PNG figures, and per-seed JSONs are still
written by the underlying trainers (with their own timestamps). They are
left in place but the wrapper does not depend on them.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Make src/ and scripts/ both importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from config import (
    LABEL_ENCODER, N_CLASSES,
    HPARAM_SMALL, HPARAM_FULL,
    RESULTS_DIR, FIGURES_DIR, RES_DIR,
    RANDOM_SEED,
)
from data_utils import (
    build_master_table, audit_spectrograms,
    determine_eval_protocol, build_splits, load_splits,
    SpectrogramDataset, MetadataSpectrogramDataset,
    check_metadata_viability, compute_meta_stats, load_png,
)

# Import run_single from BOTH per-model scripts. We rename them so we
# can dispatch by --model. The scripts have different signatures, which
# we handle in run_one_seed() below.
from train_transformer_audio import run_single as run_single_audio
from train_transformer_attn  import run_single as run_single_attn


# ---------------------------------------------------------------------------
# Reproducibility helper
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch (CPU + CUDA)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Per-seed dispatch
# ---------------------------------------------------------------------------

def run_one_seed(model_name: str, seed: int, folds, f_dim, max_len,
                 hparams, device, use_meta, meta_stats_per_fold,
                 baseline_test_f1=None) -> dict:
    """
    Run the underlying single-run trainer once, with `seed`.
    Returns the summary dict produced by the trainer.
    """
    set_seed(seed)
    print(f"\n{'#'*62}")
    print(f"#  SEED {seed}  ({model_name})")
    print(f"{'#'*62}")

    if model_name == "audio":
        return run_single_audio(
            folds, f_dim, max_len, hparams, device,
            run_tag=f"audio_seed{seed}",
        )
    elif model_name == "attn":
        return run_single_attn(
            folds, f_dim, max_len, hparams, device,
            use_meta, meta_stats_per_fold,
            run_tag=f"attn_seed{seed}",
            baseline_f1=baseline_test_f1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_seed_results(per_seed_summaries: list, seeds: list,
                           model_name: str, hparams: dict,
                           baseline_test_f1=None) -> dict:
    """
    Build the aggregated mean ± std summary across seeds.

    Per-seed summaries each contain a single fold (fold 0) under
    'fold_results' because we are using the holdout protocol with 9
    participants. We pull val/test macro-F1 and per-class F1 from
    that single fold for each seed.
    """
    val_f1s   = []
    test_f1s  = []
    per_class_val  = {c: [] for c in cfg.CLASS_NAMES}
    per_class_test = {c: [] for c in cfg.CLASS_NAMES}
    val_to_test_gaps = []
    stopped_epochs   = []
    elapsed_secs     = []

    for s in per_seed_summaries:
        fold0 = s["fold_results"][0]
        val_f1s .append(fold0["val_macro_f1"])
        test_f1s.append(fold0["test_macro_f1"])
        val_to_test_gaps.append(fold0["val_macro_f1"] - fold0["test_macro_f1"])
        stopped_epochs  .append(fold0["stopped_epoch"])
        elapsed_secs    .append(fold0["elapsed_s"])
        for c in cfg.CLASS_NAMES:
            per_class_val [c].append(fold0["val_per_class" ].get(c, 0.0))
            per_class_test[c].append(fold0["test_per_class"].get(c, 0.0))

    def m_s(xs):
        return {"mean": float(np.mean(xs)), "std": float(np.std(xs))}

    agg = {
        "model"             : per_seed_summaries[0]["model"],
        "model_name"        : model_name,
        "seeds"             : seeds,
        "n_seeds"           : len(seeds),
        "hparams"           : hparams,
        "n_present_classes" : per_seed_summaries[0]["n_present_classes"],
        "chance_level"      : per_seed_summaries[0]["chance_level"],
        "val_macro_f1"      : m_s(val_f1s),
        "test_macro_f1"     : m_s(test_f1s),
        "val_to_test_gap"   : m_s(val_to_test_gaps),
        "val_per_class_f1"  : {c: m_s(per_class_val [c]) for c in cfg.CLASS_NAMES},
        "test_per_class_f1" : {c: m_s(per_class_test[c]) for c in cfg.CLASS_NAMES},
        "stopped_epoch"     : m_s(stopped_epochs),
        "elapsed_s"         : m_s(elapsed_secs),
        "per_seed_val_f1"   : val_f1s,
        "per_seed_test_f1"  : test_f1s,
        "baseline_test_f1"  : baseline_test_f1,
        "per_seed_summaries": per_seed_summaries,
    }

    if baseline_test_f1 is not None:
        agg["delta_vs_baseline"] = {
            "mean": float(np.mean(test_f1s) - baseline_test_f1),
        }

    return agg


def print_aggregated_summary(agg: dict):
    print(f"\n{'='*62}")
    print(f"MULTI-SEED SUMMARY  —  {agg['model']}  ({agg['n_seeds']} seeds)")
    print(f"{'='*62}")
    print(f"  Seeds          : {agg['seeds']}")
    print(f"  Chance level   : {agg['chance_level']:.3f}")
    print(f"  Val  macro-F1  : {agg['val_macro_f1' ]['mean']:.4f} "
          f"± {agg['val_macro_f1' ]['std']:.4f}")
    print(f"  Test macro-F1  : {agg['test_macro_f1']['mean']:.4f} "
          f"± {agg['test_macro_f1']['std']:.4f}")
    print(f"  Val→Test gap   : {agg['val_to_test_gap']['mean']:+.4f} "
          f"± {agg['val_to_test_gap']['std']:.4f}")

    if agg.get("baseline_test_f1") is not None:
        delta = agg["delta_vs_baseline"]["mean"]
        sigma = agg["test_macro_f1"]["std"]
        ratio = abs(delta) / sigma if sigma > 0 else float("inf")
        flag  = "within seed noise" if ratio < 1.0 else (
                "marginal" if ratio < 2.0 else "outside seed noise")
        print(f"\n  Baseline (Model 2) test F1 : {agg['baseline_test_f1']:.4f}")
        print(f"  Δ test F1                  : {delta:+.4f}  "
              f"(|Δ| / σ = {ratio:.2f} → {flag})")

    print(f"\n  Per-seed test macro-F1:")
    for s, t in zip(agg["seeds"], agg["per_seed_test_f1"]):
        print(f"    seed {s:>5}: {t:.4f}")

    print(f"\n  Per-class TEST macro-F1 (mean ± std):")
    for c in cfg.CLASS_NAMES:
        m = agg["test_per_class_f1"][c]
        print(f"    {c:<6}: {m['mean']:.3f} ± {m['std']:.3f}")
    print(f"{'='*62}")


# ---------------------------------------------------------------------------
# Overlay plots
# ---------------------------------------------------------------------------

def plot_training_curves_overlay(per_seed_summaries: list, seeds: list,
                                 save_path: Path, n_present_classes: int,
                                 title: str = ""):
    """
    Two-panel figure: train loss (left) and val macro-F1 (right).
    One line per seed in each panel.
    """
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        chance = 1.0 / n_present_classes
        cmap   = plt.get_cmap("tab10")

        for i, (s, summary) in enumerate(zip(seeds, per_seed_summaries)):
            # `history` is added to fold_summary by the trainer patches
            # (see scripts/train_transformer_audio.py and ..._attn.py).
            # We still defensively check for None in case someone runs
            # this wrapper against an unpatched trainer.
            fold0   = summary["fold_results"][0]
            history = fold0.get("history")
            if history is None:
                print(f"  seed {s}: history missing — skipping in overlay. "
                      f"Re-run with the patched trainer.")
                continue
            color = cmap(i % 10)
            axes[0].plot(history["train_loss"], color=color, lw=1.4,
                         label=f"seed {s}")
            axes[1].plot(history["val_f1"],     color=color, lw=1.4,
                         label=f"seed {s}")

        axes[0].set_title("Train Loss"); axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-entropy"); axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8, loc="upper right")

        axes[1].axhline(chance, ls="--", color="grey", lw=1.2,
                        label=f"Chance ({chance:.2f})")
        axes[1].set_ylim(-0.02, 1.05)
        axes[1].set_title("Validation Macro-F1"); axes[1].set_xlabel("Epoch")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8, loc="lower right")

        if title: fig.suptitle(title, y=1.02, fontsize=11)
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Training curves overlay → {save_path}")
    except ImportError:
        print("matplotlib not available — skipping training curves overlay")
    except Exception as e:
        print(f"Training curves overlay failed: {e}")


def plot_mean_confusion_matrix(per_seed_summaries: list, split: str,
                               save_path: Path, title: str = ""):
    """
    Mean of per-seed row-normalised confusion matrices.

    Each per-seed summary now carries `{split}_preds` and `{split}_labels`
    (added by the trainer patches). For each seed we compute a row-
    normalised confusion matrix over the union of present classes, then
    average those matrices element-wise. The averaged matrix is the
    standard "what does the model do on each true class, on average
    across runs" diagnostic.

    Notes
    -----
    - Row normalisation = each row sums to 1 → cell (i,j) is "probability
      the model predicts j when the true class is i". This is the
      sensible thing to average across seeds (raw counts depend on
      n_test which is constant here, but normalised rates are more
      interpretable in the writeup).
    - We restrict the label set to the union of all classes that appear
      in either preds or labels across all seeds, so the matrix has a
      consistent shape if any seed missed a class.
    """
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        pred_key  = f"{split}_preds"
        label_key = f"{split}_labels"

        # Sanity-check that the patched fields exist on every seed.
        missing = [i for i, s in enumerate(per_seed_summaries)
                   if pred_key not in s["fold_results"][0]
                   or label_key not in s["fold_results"][0]]
        if missing:
            print(f"  Mean confusion matrix ({split}): skipped — seeds "
                  f"{missing} are missing {pred_key}/{label_key}. Re-run "
                  f"those seeds with the patched trainer.")
            return

        # Build the union label set across all seeds.
        all_label_ids = set()
        for s in per_seed_summaries:
            f0 = s["fold_results"][0]
            all_label_ids.update(f0[pred_key])
            all_label_ids.update(f0[label_key])
        ids = sorted(all_label_ids)
        from config import LABEL_DECODER
        names = [LABEL_DECODER.get(i, str(i)) for i in ids]

        # Compute one normalised CM per seed, then average.
        cms = []
        for s in per_seed_summaries:
            f0 = s["fold_results"][0]
            cm = confusion_matrix(
                f0[label_key], f0[pred_key],
                labels=ids, normalize="true",
            )
            cms.append(cm)
        mean_cm = np.mean(np.stack(cms, axis=0), axis=0)
        std_cm  = np.std (np.stack(cms, axis=0), axis=0)

        # Plot the mean CM with std overlaid in each cell as small text.
        n = len(ids)
        fig, ax = plt.subplots(figsize=(max(4.5, n * 2.2),
                                         max(3.5, n * 2.0)))
        disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm,
                                      display_labels=names)
        disp.plot(ax=ax, colorbar=True, values_format=".2f", cmap="Blues")
        # Add ± std under each cell value
        for i in range(n):
            for j in range(n):
                ax.text(j, i + 0.30, f"±{std_cm[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="grey")
        ax.set_title(title or f"{split.capitalize()} CM (mean over seeds)")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Mean confusion matrix ({split}) → {save_path}")
    except ImportError:
        print(f"matplotlib/sklearn not available — skipping mean CM ({split})")
    except Exception as e:
        print(f"Mean confusion matrix ({split}) failed: {e}")


def plot_attention_overlay(model_name: str, per_seed_summaries: list,
                           seeds: list, val_df, f_dim: int, max_len: int,
                           device, use_meta: bool, meta_stats_per_fold,
                           target_class: str, save_path: Path,
                           title: str = ""):
    """
    Single-axis attention overlay for one example of `target_class`.
    For each seed, we re-instantiate the model architecture, load that
    seed's checkpoint, run a forward pass on the same example, and plot
    the resulting attention weights.

    All seeds use the SAME val example so the comparison is meaningful.
    """
    if model_name != "attn":
        return

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from models import AttentivePoolingTransformer

        # Pick one example of target_class from val_df, deterministically.
        cls_df = val_df[val_df["class4"] == target_class]
        if len(cls_df) == 0:
            print(f"  No {target_class} examples in val — attention overlay skipped")
            return
        sample = cls_df.iloc[0]

        # Build a one-row dataset for that sample.
        meta_stats = meta_stats_per_fold[0]
        ds = MetadataSpectrogramDataset(
            cls_df.head(1), LABEL_ENCODER, max_len, f_dim,
            meta_stats=meta_stats, use_external_meta=use_meta)
        x, mask, _, meta_t = ds[0]

        # Real (non-padded) length so we can crop the x-axis sensibly.
        spec   = load_png(sample["png_path"], target_height=f_dim)
        T_real = min(spec.shape[1], max_len)

        x_b    = x.unsqueeze(0).to(device)
        mask_b = mask.unsqueeze(0).to(device)
        meta_b = meta_t.unsqueeze(0).to(device) if use_meta else None

        fig, ax = plt.subplots(figsize=(11, 4))
        cmap   = plt.get_cmap("tab10")

        # Architecture must match what the trainer used. Pull from the
        # first per-seed summary's hparams (all seeds share hparams here).
        hp = per_seed_summaries[0]["hparams"]

        for i, (s, summary) in enumerate(zip(seeds, per_seed_summaries)):
            ckpt_path = Path(summary["fold_results"][0]["checkpoint"])
            if not ckpt_path.exists():
                print(f"  Checkpoint missing for seed {s}: {ckpt_path}")
                continue

            model = AttentivePoolingTransformer(
                input_dim=f_dim, d_model=hp["d_model"],
                nhead=hp["nhead"], num_layers=hp["num_layers"],
                dropout=hp["dropout"], num_classes=N_CLASSES,
                max_len=max_len + 50,
                use_external_meta=use_meta, meta_dim=2,
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            with torch.no_grad():
                if use_meta:
                    _, w = model(x_b, padding_mask=mask_b, meta=meta_b)
                else:
                    _, w = model(x_b, padding_mask=mask_b)

            w = w.squeeze(0).cpu().numpy()[:T_real]
            ax.plot(range(T_real), w, color=cmap(i % 10), lw=1.4,
                    label=f"seed {s}")

        uniform = 1.0 / T_real
        ax.axhline(uniform, ls="--", color="grey", lw=1.0,
                   label=f"Uniform ({uniform:.4f})")
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Attention weight")
        ax.set_title(title or f"Attention profile — class {target_class} "
                              f"(same val example, {len(seeds)} seeds)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Attention overlay → {save_path}")
    except Exception as e:
        print(f"Attention overlay failed: {e}")
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# Save aggregated results
# ---------------------------------------------------------------------------

def save_aggregated(agg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, torch.Tensor): return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(agg, f, indent=2, default=_convert)
    print(f"Aggregated results → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-seed wrapper for Models 2 (audio) and 3 (attn)")
    p.add_argument("--model", choices=["audio", "attn"], required=True,
                   help="Which model to run multi-seed: 'audio' = Model 2, "
                        "'attn' = Model 3")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                   help="Random seeds to run (default: 42 123 456)")
    p.add_argument("--preset", choices=["small", "full"], default="full")
    p.add_argument("--d_model",    type=int,   default=None)
    p.add_argument("--nhead",      type=int,   default=None)
    p.add_argument("--num_layers", type=int,   default=None)
    p.add_argument("--dropout",    type=float, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--patience",   type=int,   default=None)
    p.add_argument("--use_existing_splits", action="store_true",
                   help="Load splits from RES_DIR/splits/ instead of rebuilding")
    p.add_argument("--baseline_test_f1", type=float, default=None,
                   help="Model 2 test F1 mean — for Δ vs baseline reporting "
                        "(only meaningful for --model attn)")
    p.add_argument("--attn_overlay_class", type=str, default="EXTEV",
                   choices=["EXTEV", "INTER", "MOVE", "PLTOY"],
                   help="Which class to use for the attention-weight overlay "
                        "plot (Model 3 only)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg.print_config()
    cfg.make_output_dirs()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"\nDevice: {device}")
    print(f"Model : {args.model}")
    print(f"Seeds : {args.seeds}")

    # Build hyperparameters from preset + CLI overrides
    base    = HPARAM_SMALL if args.preset == "small" else HPARAM_FULL
    hparams = dict(base)
    for key in ["d_model", "nhead", "num_layers", "dropout",
                "lr", "batch_size", "epochs", "patience"]:
        v = getattr(args, key)
        if v is not None:
            hparams[key] = v
    print(f"\nHyperparameters: {hparams}")

    # ── Data ────────────────────────────────────────────────────────────
    print("\n[1/4] Building master table & splits...")
    df = build_master_table(verbose=False)
    if df["png_exists"].sum() == 0:
        print(f"\n✗ No spectrograms in {cfg.SPEC_DIR}")
        sys.exit(1)
    df = df[df["png_exists"]].copy()

    audit_spectrograms(df["png_path"].tolist(), verbose=False)
    f_dim   = cfg.F_DIM
    max_len = cfg.MAX_LEN

    use_meta = False
    meta_stats_per_fold = None
    if args.model == "attn":
        meta_check = check_metadata_viability(df, verbose=True)
        use_meta   = meta_check["USE_EXTERNAL_META"]

    protocol = determine_eval_protocol(df)
    if args.use_existing_splits:
        folds = load_splits(protocol)
    else:
        folds = build_splits(df, protocol, save=True)

    if args.model == "attn":
        meta_stats_per_fold = [compute_meta_stats(tr) for tr, _, _ in folds]

    # We need val_df from fold 0 later for the attention overlay plot.
    val_df_fold0 = folds[0][1]

    # ── Per-seed runs ───────────────────────────────────────────────────
    print(f"\n[2/4] Running {len(args.seeds)} seeds...")
    per_seed_summaries = []
    for s in args.seeds:
        summary = run_one_seed(
            args.model, s, folds, f_dim, max_len, hparams, device,
            use_meta, meta_stats_per_fold,
            baseline_test_f1=args.baseline_test_f1,
        )
        per_seed_summaries.append(summary)
        # Free GPU memory between seeds to prevent hangs
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Aggregate ────────────────────────────────────────────────────────
    print("\n[3/4] Aggregating across seeds...")
    agg = aggregate_seed_results(
        per_seed_summaries, args.seeds, args.model, hparams,
        baseline_test_f1=args.baseline_test_f1,
    )
    print_aggregated_summary(agg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag   = f"{args.model}_multiseed_{timestamp}"
    save_aggregated(agg, RESULTS_DIR / f"{run_tag}.json")

    # ── Overlay plots ────────────────────────────────────────────────────
    print("\n[4/4] Building overlay plots...")
    title_suffix = (f"{args.model.upper()} — {len(args.seeds)} seeds, "
                    f"d={hparams['d_model']}, L={hparams['num_layers']}")

    plot_training_curves_overlay(
        per_seed_summaries, args.seeds,
        FIGURES_DIR / f"{run_tag}_training_curves.png",
        n_present_classes=agg["n_present_classes"],
        title=title_suffix,
    )

    plot_mean_confusion_matrix(
        per_seed_summaries, "val",
        FIGURES_DIR / f"{run_tag}_cm_val.png",
        title=f"Val confusion matrix (mean over seeds) — {title_suffix}",
    )
    plot_mean_confusion_matrix(
        per_seed_summaries, "test",
        FIGURES_DIR / f"{run_tag}_cm_test.png",
        title=f"Test confusion matrix (mean over seeds) — {title_suffix}",
    )

    if args.model == "attn":
        plot_attention_overlay(
            args.model, per_seed_summaries, args.seeds, val_df_fold0,
            f_dim, max_len, device, use_meta, meta_stats_per_fold,
            target_class=args.attn_overlay_class,
            save_path=FIGURES_DIR / f"{run_tag}_attn_overlay_{args.attn_overlay_class}.png",
            title=f"Attention overlay ({args.attn_overlay_class}) — {title_suffix}",
        )

    print(f"\nDone. Aggregated results in {RESULTS_DIR}/{run_tag}.json")
    print(f"     Overlay figures   in {FIGURES_DIR}/{run_tag}_*.png")


if __name__ == "__main__":
    main()