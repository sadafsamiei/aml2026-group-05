"""
train_transformer_audio.py
--------------------------
Main training script for the audio-only Transformer baseline.

Usage (from repo root in RenkuLab):
    python scripts/train_transformer_audio.py

    # With custom hyperparameters:
    python scripts/train_transformer_audio.py \
        --d_model 32 --num_layers 1 --dropout 0.5 --lr 5e-4

    # Run the full hyperparameter grid:
    python scripts/train_transformer_audio.py --grid_search

    # Skip data rebuild if splits already exist:
    python scripts/train_transformer_audio.py --use_existing_splits

Results are written to:
    data/results/transformer_audio_<timestamp>.json
    data/models/transformer_audio_best.pt
    data/figures/transformer_audio_training.png
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Make src/ importable when running from scripts/ or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from config import (
    CLASS_NAMES, LABEL_ENCODER, N_CLASSES,
    HPARAM_DEFAULTS, HPARAM_GRID,
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
# Plotting (optional — graceful fallback if matplotlib unavailable)
# ---------------------------------------------------------------------------

def plot_training_history(history: dict, save_path: Path, title: str = ""):
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for RenkuLab
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(history["train_loss"], color="steelblue")
        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-entropy loss")

        axes[1].plot(history["val_f1"], color="darkorange")
        axes[1].axhline(1 / N_CLASSES, linestyle="--", color="grey",
                        label=f"Chance ({1/N_CLASSES:.2f})")
        axes[1].set_title("Validation Macro-F1")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        axes[2].plot(history["lr"], color="green")
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_yscale("log")

        if title:
            fig.suptitle(title, y=1.02, fontsize=12)

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Training plot saved → {save_path}")

    except ImportError:
        print("matplotlib not available — skipping training plot")


def plot_confusion_matrix(preds, labels, save_path: Path, title: str = ""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            labels, preds,
            display_labels=CLASS_NAMES,
            ax=ax, colorbar=False,
            normalize="true",
            values_format=".2f",
        )
        ax.set_title(title or "Confusion Matrix (normalised)")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved → {save_path}")

    except ImportError:
        print("matplotlib not available — skipping confusion matrix plot")


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_single(
    folds: list,
    f_dim: int,
    max_len: int,
    hparams: dict,
    device: torch.device,
    run_tag: str = "default",
) -> dict:
    """
    Train and evaluate the AudioTransformer on all folds.

    Returns a summary dict with val and (if holdout) test macro-F1.
    """
    cfg.make_output_dirs()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for fold_i, (train_df, val_df, test_df) in enumerate(folds):
        fold_tag = f"{run_tag}_fold{fold_i}_{timestamp}"
        print(f"\n{'='*55}")
        print(f"FOLD {fold_i}  |  {len(train_df)} train, "
              f"{len(val_df)} val, {len(test_df)} test")
        print(f"  Hparams: d_model={hparams['d_model']}, "
              f"layers={hparams['num_layers']}, "
              f"dropout={hparams['dropout']}, "
              f"lr={hparams['lr']}")
        print(f"{'='*55}")

        # --- Datasets and loaders ---
        train_ds = SpectrogramDataset(
            train_df, LABEL_ENCODER, max_len, f_dim)
        val_ds   = SpectrogramDataset(
            val_df,   LABEL_ENCODER, max_len, f_dim)
        test_ds  = SpectrogramDataset(
            test_df,  LABEL_ENCODER, max_len, f_dim)

        train_loader = DataLoader(
            train_ds, batch_size=hparams["batch_size"],
            shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
        val_loader   = DataLoader(
            val_ds, batch_size=32, shuffle=False, num_workers=0)
        test_loader  = DataLoader(
            test_ds, batch_size=32, shuffle=False, num_workers=0)

        # --- Sanity checks (fold 0 only) ---
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
            max_len    = max_len + 50,   # small buffer above MAX_LEN
        ).to(device)

        if fold_i == 0:
            model.print_architecture()
            sanity_check_forward(model, f_dim, max_len, N_CLASSES, device)

        # --- Class-weighted loss ---
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

        # --- Plot training curves ---
        plot_training_history(
            train_result["history"],
            FIGURES_DIR / f"training_curves_{fold_tag}.png",
            title=f"AudioTransformer — Fold {fold_i} "
                  f"(d={hparams['d_model']}, L={hparams['num_layers']})",
        )

        # --- Load best checkpoint and evaluate ---
        model.load_state_dict(torch.load(save_path, map_location=device))

        val_result  = evaluate(model, val_loader,  device, return_preds=True)
        test_result = evaluate(model, test_loader, device, return_preds=True)

        print_classification_report(
            val_result["preds"], val_result["labels"],
            split_name=f"Validation (fold {fold_i})")
        print_classification_report(
            test_result["preds"], test_result["labels"],
            split_name=f"Test (fold {fold_i})")

        # --- Confusion matrices ---
        plot_confusion_matrix(
            val_result["preds"], val_result["labels"],
            FIGURES_DIR / f"cm_val_{fold_tag}.png",
            title=f"Val Confusion Matrix — Fold {fold_i}",
        )
        plot_confusion_matrix(
            test_result["preds"], test_result["labels"],
            FIGURES_DIR / f"cm_test_{fold_tag}.png",
            title=f"Test Confusion Matrix — Fold {fold_i}",
        )

        fold_summary = {
            "fold"          : fold_i,
            "hparams"       : hparams,
            "train_n"       : len(train_df),
            "val_n"         : len(val_df),
            "test_n"        : len(test_df),
            "best_val_f1"   : train_result["best_val_f1"],
            "val_macro_f1"  : val_result["macro_f1"],
            "val_per_class" : val_result["per_class_f1"],
            "test_macro_f1" : test_result["macro_f1"],
            "test_per_class": test_result["per_class_f1"],
            "stopped_epoch" : train_result["stopped_epoch"],
            "elapsed_s"     : train_result["elapsed_s"],
            "checkpoint"    : str(save_path),
        }
        all_results.append(fold_summary)

        # Save per-fold results immediately
        save_results(fold_summary,
                     RESULTS_DIR / f"fold_results_{fold_tag}.json")

    # --- Aggregate across folds ---
    val_f1s  = [r["val_macro_f1"]  for r in all_results]
    test_f1s = [r["test_macro_f1"] for r in all_results]

    summary = {
        "model"            : "AudioTransformer",
        "run_tag"          : run_tag,
        "hparams"          : hparams,
        "n_folds"          : len(folds),
        "val_macro_f1_mean": float(np.mean(val_f1s)),
        "val_macro_f1_std" : float(np.std(val_f1s)),
        "test_macro_f1_mean": float(np.mean(test_f1s)),
        "test_macro_f1_std" : float(np.std(test_f1s)),
        "fold_results"     : all_results,
    }

    print(f"\n{'='*55}")
    print(f"SUMMARY  ({len(folds)} fold(s))")
    print(f"  Val  macro-F1 : {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
    print(f"  Test macro-F1 : {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"{'='*55}")

    save_results(summary,
                 RESULTS_DIR / f"transformer_audio_{run_tag}_{timestamp}.json")
    return summary


# ---------------------------------------------------------------------------
# Hyperparameter grid search
# ---------------------------------------------------------------------------

def run_grid_search(
    folds: list,
    f_dim: int,
    max_len: int,
    device: torch.device,
) -> dict:
    """
    Run the full hyperparameter grid on fold 0 only (for efficiency).
    Then re-train the best config on all folds.
    """
    print(f"\n{'='*55}")
    print(f"HYPERPARAMETER GRID SEARCH ({len(HPARAM_GRID)} configs)")
    print(f"Grid is evaluated on fold 0 only.")
    print(f"{'='*55}")

    grid_results = []
    fold_0 = folds[:1]   # train/evaluate on fold 0 for selection

    for i, hp in enumerate(HPARAM_GRID):
        hp_full = {**HPARAM_DEFAULTS, **hp}
        print(f"\nConfig {i+1}/{len(HPARAM_GRID)}: {hp}")
        result = run_single(fold_0, f_dim, max_len, hp_full, device,
                            run_tag=f"grid_{i}")
        grid_results.append({
            "config_id"   : i,
            "hparams"     : hp,
            "val_macro_f1": result["val_macro_f1_mean"],
        })

    # Sort by val F1
    grid_results.sort(key=lambda x: x["val_macro_f1"], reverse=True)

    print(f"\n{'='*55}")
    print("GRID SEARCH RESULTS (sorted by val macro-F1)")
    print(f"{'='*55}")
    for r in grid_results:
        print(f"  F1={r['val_macro_f1']:.4f}  {r['hparams']}")

    # Save grid results
    save_results(
        {"grid_results": grid_results},
        RESULTS_DIR / f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # Re-train best config on all folds
    best_hp_raw = grid_results[0]["hparams"]
    best_hp     = {**HPARAM_DEFAULTS, **best_hp_raw}
    print(f"\nBest config: {best_hp_raw}")
    print("Re-training best config on all folds...")

    final_result = run_single(folds, f_dim, max_len, best_hp, device,
                              run_tag="best")
    return final_result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train audio-only Transformer baseline"
    )
    p.add_argument("--d_model",    type=int,   default=HPARAM_DEFAULTS["d_model"])
    p.add_argument("--nhead",      type=int,   default=HPARAM_DEFAULTS["nhead"])
    p.add_argument("--num_layers", type=int,   default=HPARAM_DEFAULTS["num_layers"])
    p.add_argument("--dropout",    type=float, default=HPARAM_DEFAULTS["dropout"])
    p.add_argument("--lr",         type=float, default=HPARAM_DEFAULTS["lr"])
    p.add_argument("--batch_size", type=int,   default=HPARAM_DEFAULTS["batch_size"])
    p.add_argument("--epochs",     type=int,   default=HPARAM_DEFAULTS["epochs"])
    p.add_argument("--patience",   type=int,   default=HPARAM_DEFAULTS["patience"])
    p.add_argument("--grid_search", action="store_true",
                   help="Run full hyperparameter grid search")
    p.add_argument("--use_existing_splits", action="store_true",
                   help="Load splits from data/splits/ instead of rebuilding")
    p.add_argument("--no_sanity_checks", action="store_true",
                   help="Skip forward-pass and dataloader sanity checks")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def main():
    args = parse_args()

    # --- Reproducibility ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Config ---
    cfg.print_config()
    cfg.make_output_dirs()

    # --- Device ---
    device = torch.device(
        "cuda"  if torch.cuda.is_available() else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\nDevice: {device}")

    # --- Master table ---
    print("\n[1/5] Building master table...")
    df = build_master_table(verbose=True)

    if df["png_exists"].sum() == 0:
        print("\n✗ ERROR: No spectrogram PNGs found.")
        print(f"  Expected location: {cfg.SPEC_DIR}")
        print("  Check that your 'data' dataset is correctly mounted in Renku.")
        sys.exit(1)

    df = df[df["png_exists"]].copy()

    # --- Spectrogram audit ---
    print("\n[2/5] Auditing spectrograms...")
    audit_stats = audit_spectrograms(df["png_path"].tolist(), verbose=True)
    f_dim   = audit_stats["f_dim"]
    max_len = audit_stats["max_len_p95"]

    if audit_stats["n_silent"] > 0:
        print(f"  ⚠ Removing {audit_stats['n_silent']} silent clip(s)")
        # Remove all-zero spectrograms
        from PIL import Image as _PIL
        valid = []
        for p in df["png_path"]:
            arr = np.array(_PIL.open(p), dtype=np.float32)
            if arr.max() > 0:
                valid.append(p)
        df = df[df["png_path"].isin(valid)].copy()

    # --- Splits ---
    print("\n[3/5] Constructing splits...")
    if args.use_existing_splits:
        protocol = determine_eval_protocol(df)
        folds = load_splits(protocol)
    else:
        protocol = determine_eval_protocol(df)
        folds = build_splits(df, protocol, save=True)

    # --- Train ---
    hparams = {
        "d_model"     : args.d_model,
        "nhead"       : args.nhead,
        "num_layers"  : args.num_layers,
        "dropout"     : args.dropout,
        "lr"          : args.lr,
        "weight_decay": HPARAM_DEFAULTS["weight_decay"],
        "batch_size"  : args.batch_size,
        "epochs"      : args.epochs,
        "patience"    : args.patience,
    }

    print(f"\n[4/5] Training{'  (grid search)' if args.grid_search else ''}...")
    if args.grid_search:
        result = run_grid_search(folds, f_dim, max_len, device)
    else:
        result = run_single(folds, f_dim, max_len, hparams, device,
                            run_tag="single")

    # --- Final summary ---
    print(f"\n[5/5] Done.")
    print(f"  Val  macro-F1 : {result['val_macro_f1_mean']:.4f} "
          f"± {result['val_macro_f1_std']:.4f}")
    print(f"  Test macro-F1 : {result['test_macro_f1_mean']:.4f} "
          f"± {result['test_macro_f1_std']:.4f}")
    print(f"\nAll outputs in: {cfg.DATA_DIR}/")
    print("  models/   ← checkpoints (.pt)")
    print("  results/  ← metrics JSON")
    print("  figures/  ← training curves + confusion matrices")


if __name__ == "__main__":
    main()
