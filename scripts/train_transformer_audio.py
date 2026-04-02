"""
train_transformer_audio.py
--------------------------
Main training script for the audio-only Transformer baseline.

Usage (from repo root in RenkuLab):
    python scripts/train_transformer_audio.py

    # Use the larger full-size model (once 300+ examples available):
    python scripts/train_transformer_audio.py --preset full

    # Override individual hyperparameters:
    python scripts/train_transformer_audio.py --d_model 64 --num_layers 2 --dropout 0.3

    # Full hyperparameter grid search:
    python scripts/train_transformer_audio.py --grid_search

    # Skip split rebuild if splits already exist:
    python scripts/train_transformer_audio.py --use_existing_splits

All results are written under DATA_DIR (syncs to SwitchDrive):
    data/models/    ← model checkpoints (.pt)
    data/results/   ← metrics JSON files
    data/figures/   ← training curves + confusion matrices
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from config import (
    CLASS_NAMES, LABEL_ENCODER, LABEL_DECODER, N_CLASSES,
    HPARAM_DEFAULTS, HPARAM_SMALL, HPARAM_FULL, HPARAM_GRID,
    MODELS_DIR, RESULTS_DIR, FIGURES_DIR,
    RANDOM_SEED,
)
from data_utils import (
    build_master_table,
    audit_spectrograms,
    determine_eval_protocol,
    build_splits,
    load_splits,
    SpectrogramDataset,
)
from models import AudioTransformer
from train_utils import (
    train_model,
    evaluate,
    print_classification_report,
    save_results,
    sanity_check_forward,
    sanity_check_dataloader,
)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_history(history: dict, save_path: Path,
                          title: str = "", n_present_classes: int = None):
    """
    Plot train loss, validation macro-F1, and learning rate over epochs.

    Parameters
    ----------
    history            : dict with keys 'train_loss', 'val_f1', 'lr'
    save_path          : where to save the PNG
    title              : figure suptitle
    n_present_classes  : number of classes actually present in the split.
                         Used to draw the correct chance-level line.
                         If None, falls back to N_CLASSES (4-class full set).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Chance level = 1 / number of classes actually in this split
        n_cls    = n_present_classes if n_present_classes else N_CLASSES
        chance   = 1.0 / n_cls
        # Flag if the chance line differs from the full-4-class baseline
        chance_note = ""
        if n_cls != N_CLASSES:
            chance_note = (f" ({n_cls} classes present; "
                           f"full 4-class chance = {1/N_CLASSES:.2f})")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # -- Train loss --
        axes[0].plot(history["train_loss"], color="steelblue", linewidth=1.5)
        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-entropy loss")
        axes[0].grid(alpha=0.3)

        # -- Validation macro-F1 --
        axes[1].plot(history["val_f1"], color="darkorange", linewidth=1.5)
        axes[1].axhline(chance, linestyle="--", color="grey", linewidth=1.2,
                        label=f"Chance ({chance:.2f}){chance_note}")
        axes[1].set_ylim(-0.02, 1.05)
        axes[1].set_title("Validation Macro-F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1")
        axes[1].legend(fontsize=8, loc="lower right")
        axes[1].grid(alpha=0.3)

        # -- Learning rate --
        axes[2].plot(history["lr"], color="green", linewidth=1.5)
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_yscale("log")
        axes[2].grid(alpha=0.3)

        if title:
            fig.suptitle(title, y=1.02, fontsize=11)

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Training plot saved → {save_path}")

    except ImportError:
        print("matplotlib not available — skipping training plot")


def plot_confusion_matrix(preds, labels, save_path: Path, title: str = ""):
    """
    Plot a normalised confusion matrix showing only the classes present
    in this split. Handles partial class sets (e.g. only INTER + PLTOY)
    without crashing matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        present_ids   = sorted(set(labels) | set(preds))
        present_names = [LABEL_DECODER.get(i, str(i)) for i in present_ids]

        cm   = confusion_matrix(labels, preds,
                                labels=present_ids, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=present_names)

        n = len(present_ids)
        fig, ax = plt.subplots(figsize=(max(4, n * 2), max(3, n * 1.8)))
        disp.plot(ax=ax, colorbar=False, values_format=".2f")
        ax.set_title(title or "Confusion Matrix (row-normalised)")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved → {save_path}")

    except ImportError:
        print("matplotlib not available — skipping confusion matrix")


# ---------------------------------------------------------------------------
# Summary interpretation helper
# ---------------------------------------------------------------------------

def print_fold_interpretation(fold_results: list, n_present_classes: int):
    """
    Print a plain-language interpretation of CV results so the numbers
    can be read critically rather than taken at face value.
    """
    vals   = [r["val_macro_f1"]  for r in fold_results]
    tests  = [r["test_macro_f1"] for r in fold_results]
    chance = 1.0 / n_present_classes
    gaps   = [v - t for v, t in zip(vals, tests)]

    print(f"\n{'='*55}")
    print("RESULT INTERPRETATION")
    print(f"{'='*55}")
    print(f"  Active classes     : {n_present_classes}  "
          f"(of {N_CLASSES} total; missing classes have no spectrograms yet)")
    print(f"  Chance level       : {chance:.3f}  "
          f"(random classifier on {n_present_classes} balanced classes)")
    print(f"  Val  macro-F1      : {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    print(f"  Test macro-F1      : {np.mean(tests):.3f} ± {np.std(tests):.3f}")
    print(f"  Val→Test gap (mean): {np.mean(gaps):+.3f}  "
          f"{'⚠ overfitting signal' if np.mean(gaps) > 0.15 else '✓ acceptable'}")

    print(f"\n  Per-fold breakdown:")
    print(f"  {'Fold':<5} {'Val':>6} {'Test':>7} {'Gap':>7}  {'Note'}")
    print(f"  " + "-"*45)
    for r in fold_results:
        gap  = r["val_macro_f1"] - r["test_macro_f1"]
        note = ""
        if r["test_macro_f1"] < chance:
            note = "⚠ BELOW CHANCE"
        elif gap > 0.30:
            note = "⚠ large val→test gap"
        elif gap < 0.05:
            note = "✓ stable"
        print(f"  {r['fold']:<5} {r['val_macro_f1']:>6.3f} "
              f"{r['test_macro_f1']:>7.3f} {gap:>+7.3f}  {note}")

    # High-level verdict
    above_chance = sum(t > chance for t in tests)
    print(f"\n  Folds above chance : {above_chance}/{len(tests)}")

    if np.std(tests) > 0.15:
        print(f"\n  ⚠  High variance (std={np.std(tests):.3f}) — results are "
              f"unreliable at this dataset size. Do not over-interpret "
              f"individual fold scores.")
    if np.mean(tests) > chance + 0.10:
        print(f"  ✓  Mean test F1 ({np.mean(tests):.3f}) is {np.mean(tests)-chance:.3f} "
              f"above chance — some signal is present, but interpret cautiously.")
    elif np.mean(tests) <= chance:
        print(f"  ✗  Mean test F1 ({np.mean(tests):.3f}) is at or below chance "
              f"({chance:.3f}) — no reliable signal with current data size.")

    print(f"\n  NOTE: These results reflect WITHIN-PARTICIPANT generalisation "
          f"only (single participant, k-fold CV). Cross-participant evaluation "
          f"will be possible once more spectrograms are available.")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Single training run (all folds)
# ---------------------------------------------------------------------------

def run_single(
    folds: list,
    f_dim: int,
    max_len: int,
    hparams: dict,
    device: torch.device,
    run_tag: str = "default",
) -> dict:
    cfg.make_output_dirs()
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for fold_i, (train_df, val_df, test_df) in enumerate(folds):
        fold_tag = f"{run_tag}_fold{fold_i}_{timestamp}"

        # Detect how many classes are actually present in this fold
        present_classes = sorted(
            set(train_df["class4"]) | set(val_df["class4"]) | set(test_df["class4"])
        )
        n_present = len(present_classes)

        print(f"\n{'='*55}")
        print(f"FOLD {fold_i}  |  {len(train_df)} train, "
              f"{len(val_df)} val, {len(test_df)} test")
        print(f"  Classes present  : {present_classes}  "
              f"({'all' if n_present == N_CLASSES else f'{n_present} of {N_CLASSES}'})")
        print(f"  Chance level     : {1/n_present:.3f}  "
              f"(for {n_present} balanced classes)")
        print(f"  Hparams: d_model={hparams['d_model']}, "
              f"layers={hparams['num_layers']}, "
              f"dropout={hparams['dropout']}, "
              f"lr={hparams['lr']}, "
              f"batch={hparams['batch_size']}")
        print(f"{'='*55}")

        # --- Datasets ---
        train_ds = SpectrogramDataset(train_df, LABEL_ENCODER, max_len, f_dim)
        val_ds   = SpectrogramDataset(val_df,   LABEL_ENCODER, max_len, f_dim)
        test_ds  = SpectrogramDataset(test_df,  LABEL_ENCODER, max_len, f_dim)

        train_loader = DataLoader(
            train_ds, batch_size=hparams["batch_size"],
            shuffle=True, num_workers=0,
            pin_memory=(device.type == "cuda"))
        val_loader   = DataLoader(val_ds,  batch_size=32, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

        if fold_i == 0:
            sanity_check_dataloader(train_loader, f_dim, max_len)

        # --- Model ---
        model = AudioTransformer(
            input_dim  = f_dim,
            d_model    = hparams["d_model"],
            nhead      = hparams["nhead"],
            num_layers = hparams["num_layers"],
            dropout    = hparams["dropout"],
            num_classes= N_CLASSES,
            max_len    = max_len + 50,
        ).to(device)

        if fold_i == 0:
            model.print_architecture()
            sanity_check_forward(model, f_dim, max_len, N_CLASSES, device)

        # --- Loss with class weights ---
        class_weights = train_ds.class_weights().to(device)
        criterion     = torch.nn.CrossEntropyLoss(weight=class_weights)

        # --- Train ---
        save_path = MODELS_DIR / f"transformer_audio_{fold_tag}.pt"
        print(f"\nTraining...")
        train_result = train_model(
            model, train_loader, val_loader, criterion, device,
            lr           = hparams["lr"],
            weight_decay = hparams["weight_decay"],
            epochs       = hparams["epochs"],
            patience     = hparams["patience"],
            save_path    = save_path,
            verbose      = True,
        )

        # --- Training curves (with correct chance line) ---
        plot_training_history(
            train_result["history"],
            FIGURES_DIR / f"training_curves_{fold_tag}.png",
            title=(f"AudioTransformer — Fold {fold_i} "
                   f"(d={hparams['d_model']}, L={hparams['num_layers']}, "
                   f"{n_present} classes)"),
            n_present_classes=n_present,
        )

        # --- Evaluate best checkpoint ---
        model.load_state_dict(torch.load(save_path, map_location=device))
        val_result  = evaluate(model, val_loader,  device, return_preds=True)
        test_result = evaluate(model, test_loader, device, return_preds=True)

        print_classification_report(
            val_result["preds"], val_result["labels"],
            split_name=f"Validation (fold {fold_i})")
        print_classification_report(
            test_result["preds"], test_result["labels"],
            split_name=f"Test (fold {fold_i})")

        plot_confusion_matrix(
            val_result["preds"], val_result["labels"],
            FIGURES_DIR / f"cm_val_{fold_tag}.png",
            title=f"Val — Fold {fold_i} (normalised)",
        )
        plot_confusion_matrix(
            test_result["preds"], test_result["labels"],
            FIGURES_DIR / f"cm_test_{fold_tag}.png",
            title=f"Test — Fold {fold_i} (normalised)",
        )

        fold_summary = {
            "fold"             : fold_i,
            "hparams"          : hparams,
            "train_n"          : len(train_df),
            "val_n"            : len(val_df),
            "test_n"           : len(test_df),
            "n_present_classes": n_present,
            "present_classes"  : present_classes,
            "chance_level"     : round(1 / n_present, 4),
            "best_val_f1"      : train_result["best_val_f1"],
            "val_macro_f1"     : val_result["macro_f1"],
            "val_per_class"    : val_result["per_class_f1"],
            "test_macro_f1"    : test_result["macro_f1"],
            "test_per_class"   : test_result["per_class_f1"],
            "stopped_epoch"    : train_result["stopped_epoch"],
            "elapsed_s"        : train_result["elapsed_s"],
            "checkpoint"       : str(save_path),
        }
        all_results.append(fold_summary)
        save_results(fold_summary,
                     RESULTS_DIR / f"fold_results_{fold_tag}.json")

    # --- Aggregate ---
    val_f1s  = [r["val_macro_f1"]  for r in all_results]
    test_f1s = [r["test_macro_f1"] for r in all_results]

    # Use the present-class count from fold 0 for the summary
    # (all folds have the same classes with k-fold CV)
    n_present_summary = all_results[0]["n_present_classes"]

    summary = {
        "model"             : "AudioTransformer",
        "run_tag"           : run_tag,
        "hparams"           : hparams,
        "n_folds"           : len(folds),
        "n_present_classes" : n_present_summary,
        "chance_level"      : round(1 / n_present_summary, 4),
        "val_macro_f1_mean" : float(np.mean(val_f1s)),
        "val_macro_f1_std"  : float(np.std(val_f1s)),
        "test_macro_f1_mean": float(np.mean(test_f1s)),
        "test_macro_f1_std" : float(np.std(test_f1s)),
        "fold_results"      : all_results,
    }

    print_fold_interpretation(all_results, n_present_summary)
    save_results(summary,
                 RESULTS_DIR / f"transformer_audio_{run_tag}_{timestamp}.json")
    return summary


# ---------------------------------------------------------------------------
# Hyperparameter grid search
# ---------------------------------------------------------------------------

def run_grid_search(folds, f_dim, max_len, device) -> dict:
    print(f"\n{'='*55}")
    print(f"HYPERPARAMETER GRID SEARCH ({len(HPARAM_GRID)} configs)")
    print(f"Evaluated on fold 0 only; best config re-trained on all folds.")
    print(f"{'='*55}")

    grid_results = []
    for i, hp in enumerate(HPARAM_GRID):
        hp_full = {**HPARAM_DEFAULTS, **hp}
        print(f"\nConfig {i+1}/{len(HPARAM_GRID)}: {hp}")
        result = run_single(folds[:1], f_dim, max_len, hp_full, device,
                            run_tag=f"grid_{i}")
        grid_results.append({
            "config_id"   : i,
            "hparams"     : hp,
            "val_macro_f1": result["val_macro_f1_mean"],
        })

    grid_results.sort(key=lambda x: x["val_macro_f1"], reverse=True)

    print(f"\n{'='*55}")
    print("GRID SEARCH RESULTS (sorted by val macro-F1)")
    print(f"{'='*55}")
    for r in grid_results:
        print(f"  F1={r['val_macro_f1']:.4f}  {r['hparams']}")

    save_results(
        {"grid_results": grid_results},
        RESULTS_DIR / f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    best_hp = {**HPARAM_DEFAULTS, **grid_results[0]["hparams"]}
    print(f"\nBest config: {grid_results[0]['hparams']}")
    print("Re-training on all folds...")
    return run_single(folds, f_dim, max_len, best_hp, device, run_tag="best")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train audio-only Transformer")

    # Preset selector
    p.add_argument(
        "--preset", choices=["small", "full"], default="small",
        help=("Hyperparameter preset. "
              "'small' (default): d=32, L=1, batch=8 — use with < 100 examples. "
              "'full': d=64, L=2, batch=16 — use once 300+ examples are available.")
    )

    # Individual overrides (override the chosen preset)
    p.add_argument("--d_model",    type=int,   default=None)
    p.add_argument("--nhead",      type=int,   default=None)
    p.add_argument("--num_layers", type=int,   default=None)
    p.add_argument("--dropout",    type=float, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--patience",   type=int,   default=None)

    p.add_argument("--grid_search",        action="store_true")
    p.add_argument("--use_existing_splits",action="store_true")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg.print_config()
    cfg.make_output_dirs()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\nDevice: {device}")

    # --- Build hparams: start from preset, then apply any CLI overrides ---
    base = HPARAM_SMALL if args.preset == "small" else HPARAM_FULL
    hparams = dict(base)   # copy
    for key in ["d_model", "nhead", "num_layers", "dropout",
                "lr", "batch_size", "epochs", "patience"]:
        val = getattr(args, key)
        if val is not None:
            hparams[key] = val

    print(f"\nActive hyperparameters (preset='{args.preset}'):")
    for k, v in hparams.items():
        print(f"  {k:15s}: {v}")

    # --- Data ---
    print("\n[1/5] Building master table...")
    df = build_master_table(verbose=True)

    if df["png_exists"].sum() == 0:
        print(f"\n✗ ERROR: No PNGs found in {cfg.SPEC_DIR}")
        print("  Check that the Renku dataset is mounted correctly.")
        sys.exit(1)

    df = df[df["png_exists"]].copy()

    print("\n[2/5] Auditing spectrograms...")
    audit_stats = audit_spectrograms(df["png_path"].tolist(), verbose=True)
    f_dim   = audit_stats["f_dim"]
    max_len = audit_stats["max_len_p95"]

    if audit_stats["n_silent"] > 0:
        from PIL import Image as _PIL
        import numpy as _np
        valid = [p for p in df["png_path"]
                 if _np.array(_PIL.open(p), dtype=_np.float32).max() > 0]
        df = df[df["png_path"].isin(valid)].copy()
        print(f"  Removed {audit_stats['n_silent']} silent clip(s)")

    print("\n[3/5] Constructing splits...")
    if args.use_existing_splits:
        protocol = determine_eval_protocol(df)
        folds    = load_splits(protocol)
    else:
        protocol = determine_eval_protocol(df)
        folds    = build_splits(df, protocol, save=True)

    print(f"\n[4/5] Training{'  (grid search)' if args.grid_search else ''}...")
    if args.grid_search:
        result = run_grid_search(folds, f_dim, max_len, device)
    else:
        result = run_single(folds, f_dim, max_len, hparams, device,
                            run_tag="single")

    print(f"\n[5/5] Done.")
    print(f"  Val  macro-F1 : {result['val_macro_f1_mean']:.4f} "
          f"± {result['val_macro_f1_std']:.4f}")
    print(f"  Test macro-F1 : {result['test_macro_f1_mean']:.4f} "
          f"± {result['test_macro_f1_std']:.4f}")
    print(f"  Chance level  : {result['chance_level']:.4f}  "
          f"({result['n_present_classes']} classes)")
    print(f"\nAll outputs → {cfg.DATA_DIR}/")
    print("  models/   ← checkpoints")
    print("  results/  ← JSON metrics")
    print("  figures/  ← training curves + confusion matrices")


if __name__ == "__main__":
    main()