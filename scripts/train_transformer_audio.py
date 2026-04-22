"""
train_transformer_audio.py
--------------------------
Model 2: Audio-only Transformer with mean pooling (neural baseline).

Usage:
    python scripts/train_transformer_audio.py               # default: full preset
    python scripts/train_transformer_audio.py --preset small
    python scripts/train_transformer_audio.py --d_model 128 --num_layers 3
    python scripts/train_transformer_audio.py --grid_search
    python scripts/train_transformer_audio.py --use_existing_splits

Inputs  : DATA_DIR/spectrograms/, DATA_DIR/contextsvocs_*.csv
Outputs : RES_DIR/models/, RES_DIR/results/, RES_DIR/figures/
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
    MODELS_DIR, RESULTS_DIR, FIGURES_DIR, RES_DIR,
    RANDOM_SEED,
)
from data_utils import (
    build_master_table, audit_spectrograms,
    determine_eval_protocol, build_splits, load_splits,
    SpectrogramDataset,
)
from models import AudioTransformer
from train_utils import (
    train_model, evaluate,
    print_classification_report, save_results,
    sanity_check_forward, sanity_check_dataloader,
)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_training_history(history, save_path, title="", n_present_classes=None):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_cls  = n_present_classes or N_CLASSES
        chance = 1.0 / n_cls
        note   = (f" ({n_cls} classes; 4-class={1/N_CLASSES:.2f})"
                  if n_cls != N_CLASSES else "")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(history["train_loss"], color="steelblue", lw=1.5)
        axes[0].set_title("Train Loss"); axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-entropy"); axes[0].grid(alpha=0.3)

        axes[1].plot(history["val_f1"], color="darkorange", lw=1.5)
        axes[1].axhline(chance, ls="--", color="grey", lw=1.2,
                        label=f"Chance ({chance:.2f}){note}")
        axes[1].set_ylim(-0.02, 1.05)
        axes[1].set_title("Validation Macro-F1"); axes[1].set_xlabel("Epoch")
        axes[1].legend(fontsize=8, loc="lower right"); axes[1].grid(alpha=0.3)

        axes[2].plot(history["lr"], color="green", lw=1.5)
        axes[2].set_title("Learning Rate"); axes[2].set_xlabel("Epoch")
        axes[2].set_yscale("log"); axes[2].grid(alpha=0.3)

        if title: fig.suptitle(title, y=1.02, fontsize=11)
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Training plot → {save_path}")
    except ImportError:
        print("matplotlib not available — skipping plot")


def plot_confusion_matrix(preds, labels, save_path, title=""):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        ids   = sorted(set(labels) | set(preds))
        names = [LABEL_DECODER.get(i, str(i)) for i in ids]
        cm    = confusion_matrix(labels, preds, labels=ids, normalize="true")
        disp  = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)

        n = len(ids)
        fig, ax = plt.subplots(figsize=(max(4, n*2), max(3, n*1.8)))
        disp.plot(ax=ax, colorbar=False, values_format=".2f")
        ax.set_title(title or "Confusion Matrix (normalised)")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix → {save_path}")
    except ImportError:
        print("matplotlib not available — skipping confusion matrix")


# ---------------------------------------------------------------------------
# Result interpretation
# ---------------------------------------------------------------------------

def print_fold_interpretation(fold_results, n_present_classes):
    vals   = [r["val_macro_f1"]  for r in fold_results]
    tests  = [r["test_macro_f1"] for r in fold_results]
    chance = 1.0 / n_present_classes
    gaps   = [v - t for v, t in zip(vals, tests)]

    print(f"\n{'='*55}")
    print("RESULT INTERPRETATION — AudioTransformer")
    print(f"{'='*55}")
    print(f"  Classes        : {n_present_classes} active (of {N_CLASSES})")
    print(f"  Chance level   : {chance:.3f}")
    print(f"  Val  macro-F1  : {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    print(f"  Test macro-F1  : {np.mean(tests):.3f} ± {np.std(tests):.3f}")
    print(f"  Val→Test gap   : {np.mean(gaps):+.3f}  "
          f"{'⚠ overfitting' if np.mean(gaps) > 0.15 else '✓ ok'}")
    print(f"\n  {'Fold':<5} {'Val':>6} {'Test':>7} {'Gap':>7}  Note")
    print(f"  " + "-"*40)
    for r in fold_results:
        gap  = r["val_macro_f1"] - r["test_macro_f1"]
        note = ("⚠ below chance" if r["test_macro_f1"] < chance
                else ("⚠ large gap" if gap > 0.30 else ""))
        fold_id = r.get("fold", "-")
        print(f"  {str(fold_id):<5} {r['val_macro_f1']:>6.3f} "
              f"{r['test_macro_f1']:>7.3f} {gap:>+7.3f}  {note}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Single run (all folds)
# ---------------------------------------------------------------------------

def run_single(folds, f_dim, max_len, hparams, device, run_tag="single"):
    cfg.make_output_dirs()
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for fold_i, (train_df, val_df, test_df) in enumerate(folds):
        present = sorted(
            set(train_df["class4"]) | set(val_df["class4"]) | set(test_df["class4"])
        )
        n_present = len(present)
        fold_tag  = f"{run_tag}_fold{fold_i}_{timestamp}"

        print(f"\n{'='*55}")
        print(f"FOLD {fold_i}  |  {len(train_df)} train / "
              f"{len(val_df)} val / {len(test_df)} test")
        print(f"  Classes  : {present}")
        print(f"  Chance   : {1/n_present:.3f}")
        print(f"  d={hparams['d_model']}, L={hparams['num_layers']}, "
              f"drop={hparams['dropout']}, lr={hparams['lr']}, "
              f"batch={hparams['batch_size']}, "
              f"wd={hparams.get('weight_decay', 1e-4)}, "
              f"ls={hparams.get('label_smoothing', 0.0)}")
        print(f"{'='*55}")

        train_ds = SpectrogramDataset(train_df, LABEL_ENCODER, max_len, f_dim,
                                      train=True)
        val_ds   = SpectrogramDataset(val_df,   LABEL_ENCODER, max_len, f_dim,
                                      train=False)
        test_ds  = SpectrogramDataset(test_df,  LABEL_ENCODER, max_len, f_dim,
                                      train=False)

        train_loader = DataLoader(train_ds, batch_size=hparams["batch_size"],
                                  shuffle=True, num_workers=0,
                                  pin_memory=(device.type == "cuda"))
        val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False,
                                  num_workers=0,
                                  pin_memory=(device.type == "cuda"))
        test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False,
                                  num_workers=0,
                                  pin_memory=(device.type == "cuda"))

        if fold_i == 0:
            sanity_check_dataloader(train_loader, f_dim, max_len)

        model = AudioTransformer(
            input_dim=f_dim, d_model=hparams["d_model"],
            nhead=hparams["nhead"], num_layers=hparams["num_layers"],
            dropout=hparams["dropout"], num_classes=N_CLASSES,
            max_len=max_len + 50,
        ).to(device)

        if fold_i == 0:
            model.print_architecture()
            sanity_check_forward(model, f_dim, max_len, N_CLASSES, device)

        # Label smoothing is pulled from hparams; defaults to 0.0 if absent,
        # making old hparam dicts bit-compatible with this script.
        criterion = torch.nn.CrossEntropyLoss(
            weight=train_ds.class_weights().to(device),
            label_smoothing=hparams.get("label_smoothing", 0.0),
        )

        save_path = MODELS_DIR / f"transformer_audio_{fold_tag}.pt"
        print(f"\nTraining...")
        train_result = train_model(
            model, train_loader, val_loader, criterion, device,
            lr=hparams["lr"], weight_decay=hparams["weight_decay"],
            epochs=hparams["epochs"], patience=hparams["patience"],
            save_path=save_path, verbose=True,
        )

        plot_training_history(
            train_result["history"],
            FIGURES_DIR / f"training_curves_{fold_tag}.png",
            title=(f"AudioTransformer — Fold {fold_i} "
                   f"(d={hparams['d_model']}, L={hparams['num_layers']}, "
                   f"{n_present} classes)"),
            n_present_classes=n_present,
        )

        model.load_state_dict(torch.load(save_path, map_location=device))
        val_result  = evaluate(model, val_loader,  device, return_preds=True)
        test_result = evaluate(model, test_loader, device, return_preds=True)

        print_classification_report(val_result["preds"],  val_result["labels"],
                                    f"Validation (fold {fold_i})")
        print_classification_report(test_result["preds"], test_result["labels"],
                                    f"Test (fold {fold_i})")

        plot_confusion_matrix(val_result["preds"],  val_result["labels"],
                              FIGURES_DIR / f"cm_val_{fold_tag}.png",
                              title=f"Val — Fold {fold_i}")
        plot_confusion_matrix(test_result["preds"], test_result["labels"],
                              FIGURES_DIR / f"cm_test_{fold_tag}.png",
                              title=f"Test — Fold {fold_i}")

        fold_summary = {
            "fold"             : fold_i,
            "hparams"          : hparams,
            "train_n"          : len(train_df),
            "val_n"            : len(val_df),
            "test_n"           : len(test_df),
            "n_present_classes": n_present,
            "present_classes"  : present,
            "chance_level"     : round(1 / n_present, 4),
            "best_val_f1"      : train_result["best_val_f1"],
            "val_macro_f1"     : val_result["macro_f1"],
            "val_per_class"    : val_result["per_class_f1"],
            "test_macro_f1"    : test_result["macro_f1"],
            "test_per_class"   : test_result["per_class_f1"],
            "stopped_epoch"    : train_result["stopped_epoch"],
            "elapsed_s"        : train_result["elapsed_s"],
            "checkpoint"       : str(save_path),
            "history"          : train_result["history"],
            "val_preds"        : val_result["preds"],
            "val_labels"       : val_result["labels"],
            "test_preds"       : test_result["preds"],
            "test_labels"      : test_result["labels"],
        }
        all_results.append(fold_summary)
        save_results(fold_summary, RESULTS_DIR / f"fold_results_{fold_tag}.json")

    val_f1s  = [r["val_macro_f1"]  for r in all_results]
    test_f1s = [r["test_macro_f1"] for r in all_results]
    n_pres   = all_results[0]["n_present_classes"]

    summary = {
        "model"             : "AudioTransformer",
        "run_tag"           : run_tag,
        "hparams"           : hparams,
        "n_folds"           : len(folds),
        "n_present_classes" : n_pres,
        "chance_level"      : round(1 / n_pres, 4),
        "val_macro_f1_mean" : float(np.mean(val_f1s)),
        "val_macro_f1_std"  : float(np.std(val_f1s)),
        "test_macro_f1_mean": float(np.mean(test_f1s)),
        "test_macro_f1_std" : float(np.std(test_f1s)),
        "fold_results"      : all_results,
    }
    print_fold_interpretation(all_results, n_pres)
    save_results(summary,
                 RESULTS_DIR / f"transformer_audio_{run_tag}_{timestamp}.json")
    return summary


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search(folds, f_dim, max_len, device):
    print(f"\n{'='*55}")
    print(f"GRID SEARCH — AudioTransformer ({len(HPARAM_GRID)} configs)")
    print(f"Evaluated on fold 0; best config re-run on all folds.")
    print(f"{'='*55}")

    grid_results = []
    for i, hp in enumerate(HPARAM_GRID):
        hp_full = {**HPARAM_DEFAULTS, **hp}
        print(f"\nConfig {i+1}/{len(HPARAM_GRID)}: {hp}")
        result = run_single(folds[:1], f_dim, max_len, hp_full, device,
                            run_tag=f"grid_{i}")
        # Capture best_epoch for overfitting diagnosis in the grid report
        fold0 = result["fold_results"][0]
        best_epoch = (fold0["history"]["val_f1"].index(
            max(fold0["history"]["val_f1"])) + 1) if fold0.get("history") else None
        final_train_loss = (fold0["history"]["train_loss"][-1]
                            if fold0.get("history") else None)
        grid_results.append({
            "config_id": i, "hparams": hp,
            "val_macro_f1"    : result["val_macro_f1_mean"],
            "test_macro_f1"   : result["test_macro_f1_mean"],
            "best_val_epoch"  : best_epoch,
            "final_train_loss": final_train_loss,
            "stopped_epoch"   : fold0["stopped_epoch"],
        })

    grid_results.sort(key=lambda x: x["val_macro_f1"], reverse=True)
    print(f"\n{'='*55}\nGrid results (sorted by val F1):")
    print(f"  {'val':>7} {'test':>7} {'best_ep':>7} {'stopped':>7} "
          f"{'final_tr_loss':>14}  hparams")
    for r in grid_results:
        print(f"  {r['val_macro_f1']:>7.4f} {r['test_macro_f1']:>7.4f} "
              f"{str(r['best_val_epoch']):>7} {str(r['stopped_epoch']):>7} "
              f"{r['final_train_loss']:>14.4f}  {r['hparams']}")
    print(f"\nOverfitting diagnosis (same hparams, different runs):")
    print(f"  - best_ep < 5 and final_tr_loss < 0.2 → still overfitting fast")
    print(f"  - best_ep > 15 and final_tr_loss > 0.3 → regularization working")
    print(f"  - sustained val > 0.30 across multiple configs → real signal")

    save_results({"grid_results": grid_results},
                 RESULTS_DIR / f"grid_search_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    best_hp = {**HPARAM_DEFAULTS, **grid_results[0]["hparams"]}
    print(f"\nBest: {grid_results[0]['hparams']} — re-training on all folds...")
    return run_single(folds, f_dim, max_len, best_hp, device, run_tag="best")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train audio-only Transformer (Model 2)")
    p.add_argument("--preset", choices=["small", "full"], default="full",
                   help="'full' (default, 1009 examples) or 'small' (< 100 examples)")
    p.add_argument("--d_model",      type=int,   default=None)
    p.add_argument("--nhead",        type=int,   default=None)
    p.add_argument("--num_layers",   type=int,   default=None)
    p.add_argument("--dropout",      type=float, default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--patience",     type=int,   default=None)
    p.add_argument("--weight_decay",    type=float, default=None)
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument("--grid_search",         action="store_true")
    p.add_argument("--use_existing_splits", action="store_true",
                   help="Load splits from RES_DIR/splits/ instead of rebuilding")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg.print_config()
    cfg.make_output_dirs()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"\nDevice: {device}")

    base    = HPARAM_SMALL if args.preset == "small" else HPARAM_FULL
    hparams = dict(base)
    for key in ["d_model", "nhead", "num_layers", "dropout",
                "lr", "batch_size", "epochs", "patience",
                "weight_decay", "label_smoothing"]:
        val = getattr(args, key, None)
        if val is not None:
            hparams[key] = val
    # Ensure label_smoothing has a default even if absent from the preset
    hparams.setdefault("label_smoothing", 0.0)

    print(f"\nHyperparameters (preset='{args.preset}'):")
    for k, v in hparams.items():
        print(f"  {k:17s}: {v}")

    print("\n[1/5] Building master table...")
    df = build_master_table(verbose=True)

    if df["png_exists"].sum() == 0:
        print(f"\n✗ No spectrograms found in {cfg.SPEC_DIR}")
        print("  Run first:  python scripts/generate_spectrograms.py")
        sys.exit(1)

    df = df[df["png_exists"]].copy()

    print("\n[2/5] Auditing spectrograms...")
    audit_spectrograms(df["png_path"].tolist(), verbose=True)
    f_dim   = cfg.F_DIM
    max_len = cfg.MAX_LEN

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
        result = run_single(folds, f_dim, max_len, hparams, device)

    print(f"\n[5/5] Done.")
    print(f"  Val  macro-F1 : {result['val_macro_f1_mean']:.4f} "
          f"± {result['val_macro_f1_std']:.4f}")
    print(f"  Test macro-F1 : {result['test_macro_f1_mean']:.4f} "
          f"± {result['test_macro_f1_std']:.4f}")
    print(f"  Chance level  : {result['chance_level']:.4f} "
          f"({result['n_present_classes']} classes)")
    print(f"\nOutputs saved to {RES_DIR}/")


if __name__ == "__main__":
    main()