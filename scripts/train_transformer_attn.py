"""
train_transformer_attn.py
--------------------------
Training script for the Attentive Pooling Transformer (Model 3).

This is the main model from PROBLEMSETTING.md — same encoder as the
audio-only baseline, but with query-based attentive pooling instead of
mean pooling.

Usage (from repo root):
    python scripts/train_transformer_attn.py

    # Full-size model (once 300+ examples available):
    python scripts/train_transformer_attn.py --preset full

    # Override hyperparameters:
    python scripts/train_transformer_attn.py --d_model 64 --num_layers 2

    # Grid search over hyperparameter configs:
    python scripts/train_transformer_attn.py --grid_search

    # Re-use existing splits (skip rebuild):
    python scripts/train_transformer_attn.py --use_existing_splits

Metadata / query mode
---------------------
The script checks metadata viability automatically at the start of each run.
  - If metadata is NOT viable (current state: age = identity, 1 participant
    with PNGs): runs with a LEARNED QUERY — no external metadata inputs.
    The model still uses attentive pooling; the query is a trained parameter.
  - If metadata IS viable (when full dataset arrives, >= 4 participants,
    age/gender not collinear with identity): runs with EXTERNAL METADATA —
    age + gender are projected to the query vector.

No code changes are needed to switch between modes. It happens automatically.

Outputs (all inside RES_DIR = /home/renku/work/data, synced to SwitchDrive):
    models/  transformer_attn_*.pt
    results/ fold_results_attn_*.json, transformer_attn_*.json
    figures/ training_curves_attn_*.png, cm_val_attn_*.png, etc.
    figures/ attn_weights_*.png   ← attention weight visualisations

NOTE: DATA_DIR (/home/renku/work/aml-project) is read-only input.
      All outputs are written to RES_DIR (/home/renku/work/data).
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
    MetadataSpectrogramDataset,
    check_metadata_viability,
    compute_meta_stats,
)
from models import AttentivePoolingTransformer
from train_utils import (
    train_model,
    evaluate,
    analyse_attention_weights,
    print_classification_report,
    save_results,
    sanity_check_forward,
    sanity_check_dataloader,
)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_history(history, save_path, title="", n_present_classes=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_cls  = n_present_classes or N_CLASSES
        chance = 1.0 / n_cls
        note   = (f" ({n_cls} classes; 4-class chance={1/N_CLASSES:.2f})"
                  if n_cls != N_CLASSES else "")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(history["train_loss"], color="steelblue", lw=1.5)
        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-entropy loss")
        axes[0].grid(alpha=0.3)

        axes[1].plot(history["val_f1"], color="darkorange", lw=1.5)
        axes[1].axhline(chance, ls="--", color="grey", lw=1.2,
                        label=f"Chance ({chance:.2f}){note}")
        axes[1].set_ylim(-0.02, 1.05)
        axes[1].set_title("Validation Macro-F1")
        axes[1].set_xlabel("Epoch")
        axes[1].legend(fontsize=8, loc="lower right")
        axes[1].grid(alpha=0.3)

        axes[2].plot(history["lr"], color="green", lw=1.5)
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


def plot_confusion_matrix(preds, labels, save_path, title=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        present_ids   = sorted(set(labels) | set(preds))
        present_names = [LABEL_DECODER.get(i, str(i)) for i in present_ids]
        cm   = confusion_matrix(labels, preds, labels=present_ids, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=present_names)

        n = len(present_ids)
        fig, ax = plt.subplots(figsize=(max(4, n*2), max(3, n*1.8)))
        disp.plot(ax=ax, colorbar=False, values_format=".2f")
        ax.set_title(title or "Confusion Matrix (row-normalised)")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved → {save_path}")
    except ImportError:
        print("matplotlib not available — skipping confusion matrix")


def plot_attention_weights(model, val_df, label_encoder, max_len, f_dim,
                           device, use_meta, meta_stats, save_path):
    """
    Visualise attention weight profiles for one example per class.
    This is the key qualitative check for the attentive pooling model —
    are different classes attended to at different time points?
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from data_utils import MetadataSpectrogramDataset, load_png

        present_classes = sorted(val_df["class4"].unique())
        n_cls = len(present_classes)

        fig, axes = plt.subplots(n_cls, 1, figsize=(14, 3 * n_cls),
                                 squeeze=False)
        model.eval()

        for row_i, cls_name in enumerate(present_classes):
            cls_df  = val_df[val_df["class4"] == cls_name]
            if len(cls_df) == 0:
                continue
            sample  = cls_df.iloc[0]
            spec    = load_png(sample["png_path"], target_height=f_dim) # (F, T)
            T_real  = min(spec.shape[1], max_len)

            # Build a single-item batch
            ds    = MetadataSpectrogramDataset(
                cls_df.head(1), label_encoder, max_len, f_dim,
                meta_stats=meta_stats, use_external_meta=use_meta)
            x, mask, _, meta_t = ds[0]
            x    = x.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)

            with torch.no_grad():
                if use_meta:
                    _, w = model(x, padding_mask=mask,
                                 meta=meta_t.unsqueeze(0).to(device))
                else:
                    _, w = model(x, padding_mask=mask)

            w = w.squeeze(0).cpu().numpy()[:T_real]

            ax = axes[row_i, 0]
            ax.plot(range(T_real), w, color="darkorange", lw=1.5)
            ax.axhline(1.0 / T_real, ls="--", color="grey", lw=1,
                       label=f"Uniform ({1/T_real:.4f})")
            ax.set_title(f"Class: {cls_name}  (example: {Path(sample['png_path']).stem})",
                         fontsize=10)
            ax.set_xlabel("Time frame")
            ax.set_ylabel("Attention weight")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.3)

        fig.suptitle("Attention Weight Profiles by Class\n"
                     "(higher = more attended; dashed = uniform baseline)",
                     y=1.01, fontsize=11)
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Attention plot saved → {save_path}")

    except Exception as e:
        print(f"Attention plot skipped: {e}")


# ---------------------------------------------------------------------------
# Result interpretation
# ---------------------------------------------------------------------------

def print_fold_interpretation(fold_results, n_present_classes,
                               baseline_test_f1=None):
    """Plain-language result summary with comparison to audio-only baseline."""
    vals   = [r["val_macro_f1"]  for r in fold_results]
    tests  = [r["test_macro_f1"] for r in fold_results]
    chance = 1.0 / n_present_classes
    gaps   = [v - t for v, t in zip(vals, tests)]

    print(f"\n{'='*55}")
    print("RESULT INTERPRETATION — Attentive Pooling Transformer")
    print(f"{'='*55}")
    print(f"  Active classes     : {n_present_classes}  (of {N_CLASSES})")
    print(f"  Chance level       : {chance:.3f}")
    print(f"  Val  macro-F1      : {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    print(f"  Test macro-F1      : {np.mean(tests):.3f} ± {np.std(tests):.3f}")
    print(f"  Val→Test gap (mean): {np.mean(gaps):+.3f}")

    if baseline_test_f1 is not None:
        delta = np.mean(tests) - baseline_test_f1
        flag  = ("✓ attentive pooling helps" if delta > 0.02
                 else ("✗ attentive pooling hurts" if delta < -0.02
                       else "≈ no meaningful difference"))
        print(f"\n  Audio-only baseline: {baseline_test_f1:.3f}")
        print(f"  Attentive pooling  : {np.mean(tests):.3f}")
        print(f"  Δ                  : {delta:+.3f}  {flag}")

    print(f"\n  Per-fold breakdown:")
    print(f"  {'Fold':<5} {'Val':>6} {'Test':>7} {'Gap':>7}  Note")
    print(f"  " + "-" * 40)
    for r in fold_results:
        gap  = r["val_macro_f1"] - r["test_macro_f1"]
        note = ("⚠ below chance" if r["test_macro_f1"] < chance
                else ("⚠ large gap" if gap > 0.30 else ""))
        print(f"  {r['fold']:<5} {r['val_macro_f1']:>6.3f} "
              f"{r['test_macro_f1']:>7.3f} {gap:>+7.3f}  {note}")

    print(f"\n  NOTE: Results reflect within-participant generalisation only "
          f"(k-fold CV, Scenario C). Interpretability is limited at this "
          f"dataset size. Cross-participant evaluation requires more data.")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_single(folds, f_dim, max_len, hparams, device,
               use_meta, meta_stats_per_fold, run_tag="default",
               baseline_test_f1=None):

    cfg.make_output_dirs()
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    query_mode  = "external_meta" if use_meta else "learned_query"

    for fold_i, (train_df, val_df, test_df) in enumerate(folds):
        fold_tag = f"attn_{run_tag}_fold{fold_i}_{timestamp}"

        present_classes = sorted(
            set(train_df["class4"]) | set(val_df["class4"]) | set(test_df["class4"])
        )
        n_present = len(present_classes)
        meta_stats = meta_stats_per_fold[fold_i]

        print(f"\n{'='*55}")
        print(f"FOLD {fold_i}  [AttentivePooling — {query_mode}]")
        print(f"  {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        print(f"  Classes  : {present_classes}")
        print(f"  Chance   : {1/n_present:.3f}")
        print(f"  Query    : {query_mode}")
        if use_meta:
            print(f"  Meta stats: age_mean={meta_stats['age_mean']:.2f}, "
                  f"gender_map={meta_stats['gender_map']}")
        print(f"{'='*55}")

        # --- Datasets ---
        train_ds = MetadataSpectrogramDataset(
            train_df, LABEL_ENCODER, max_len, f_dim,
            meta_stats=meta_stats, use_external_meta=use_meta)
        val_ds   = MetadataSpectrogramDataset(
            val_df,   LABEL_ENCODER, max_len, f_dim,
            meta_stats=meta_stats, use_external_meta=use_meta)
        test_ds  = MetadataSpectrogramDataset(
            test_df,  LABEL_ENCODER, max_len, f_dim,
            meta_stats=meta_stats, use_external_meta=use_meta)

        train_loader = DataLoader(train_ds, batch_size=hparams["batch_size"],
                                  shuffle=True, num_workers=0,
                                  pin_memory=(device.type == "cuda"))
        val_loader   = DataLoader(val_ds,   batch_size=32,
                                  shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=32,
                                  shuffle=False, num_workers=0)

        if fold_i == 0:
            sanity_check_dataloader(train_loader, f_dim, max_len, use_meta=use_meta)

        # --- Model ---
        model = AttentivePoolingTransformer(
            input_dim         = f_dim,
            d_model           = hparams["d_model"],
            nhead             = hparams["nhead"],
            num_layers        = hparams["num_layers"],
            dropout           = hparams["dropout"],
            num_classes       = N_CLASSES,
            max_len           = max_len + 50,
            use_external_meta = use_meta,
            meta_dim          = 2,
        ).to(device)

        if fold_i == 0:
            model.print_architecture()
            sanity_check_forward(model, f_dim, max_len, N_CLASSES, device,
                                 use_meta=use_meta)

        class_weights = train_ds.class_weights().to(device)
        criterion     = torch.nn.CrossEntropyLoss(weight=class_weights)

        save_path = MODELS_DIR / f"transformer_{fold_tag}.pt"
        print(f"\nTraining...")
        train_result = train_model(
            model, train_loader, val_loader, criterion, device,
            lr           = hparams["lr"],
            weight_decay = hparams["weight_decay"],
            epochs       = hparams["epochs"],
            patience     = hparams["patience"],
            save_path    = save_path,
            verbose      = True,
            use_meta     = use_meta,
        )

        plot_training_history(
            train_result["history"],
            FIGURES_DIR / f"training_curves_{fold_tag}.png",
            title=(f"AttentivePoolingTransformer — Fold {fold_i} "
                   f"(d={hparams['d_model']}, L={hparams['num_layers']}, "
                   f"{query_mode})"),
            n_present_classes=n_present,
        )

        # Load best checkpoint and evaluate
        model.load_state_dict(torch.load(save_path, map_location=device))

        val_result  = evaluate(model, val_loader,  device,
                               use_meta=use_meta, return_preds=True)
        test_result = evaluate(model, test_loader, device,
                               use_meta=use_meta, return_preds=True)

        print_classification_report(
            val_result["preds"], val_result["labels"],
            split_name=f"Validation (fold {fold_i})")
        print_classification_report(
            test_result["preds"], test_result["labels"],
            split_name=f"Test (fold {fold_i})")

        plot_confusion_matrix(
            val_result["preds"], val_result["labels"],
            FIGURES_DIR / f"cm_val_{fold_tag}.png",
            title=f"Val — Fold {fold_i} (Attentive Pooling)")
        plot_confusion_matrix(
            test_result["preds"], test_result["labels"],
            FIGURES_DIR / f"cm_test_{fold_tag}.png",
            title=f"Test — Fold {fold_i} (Attentive Pooling)")

        # Attention weight analysis
        if "attn_weights" in val_result:
            attn_stats = analyse_attention_weights(val_result["attn_weights"])
            print(f"\n  Attention weight stats (val):")
            print(f"    Mean std per example : {attn_stats['mean_weight_std']:.5f}")
            print(f"    Uniform proxy        : {attn_stats['uniform_std_proxy']:.5f}")
            print(f"    Non-uniform weights  : {attn_stats['is_non_uniform']}")
            if not attn_stats["is_non_uniform"]:
                print(f"    ⚠ Attention weights are nearly uniform — "
                      f"the query may not be discriminating frames effectively.")
        else:
            attn_stats = {}

        # Attention weight visualisation (fold 0 only to save time)
        if fold_i == 0:
            plot_attention_weights(
                model, val_df, LABEL_ENCODER, max_len, f_dim,
                device, use_meta, meta_stats,
                FIGURES_DIR / f"attn_weights_{fold_tag}.png",
            )

        fold_summary = {
            "fold"             : fold_i,
            "query_mode"       : query_mode,
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
            "attn_weight_stats": attn_stats,
            "stopped_epoch"    : train_result["stopped_epoch"],
            "elapsed_s"        : train_result["elapsed_s"],
            "checkpoint"       : str(save_path),
        }
        all_results.append(fold_summary)
        save_results(fold_summary,
                     RESULTS_DIR / f"fold_results_{fold_tag}.json")

    # Aggregate
    val_f1s  = [r["val_macro_f1"]  for r in all_results]
    test_f1s = [r["test_macro_f1"] for r in all_results]
    n_present_summary = all_results[0]["n_present_classes"]

    summary = {
        "model"             : "AttentivePoolingTransformer",
        "query_mode"        : query_mode,
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

    print_fold_interpretation(all_results, n_present_summary, baseline_test_f1)
    save_results(summary,
                 RESULTS_DIR / f"transformer_attn_{run_tag}_{timestamp}.json")
    return summary


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search(folds, f_dim, max_len, device,
                    use_meta, meta_stats_per_fold) -> dict:
    print(f"\n{'='*55}")
    print(f"GRID SEARCH — AttentivePoolingTransformer ({len(HPARAM_GRID)} configs)")
    print(f"Evaluated on fold 0; best config re-trained on all folds.")
    print(f"{'='*55}")

    grid_results = []
    for i, hp in enumerate(HPARAM_GRID):
        hp_full = {**HPARAM_DEFAULTS, **hp}
        print(f"\nConfig {i+1}/{len(HPARAM_GRID)}: {hp}")
        result = run_single(folds[:1], f_dim, max_len, hp_full, device,
                            use_meta, meta_stats_per_fold[:1],
                            run_tag=f"grid_{i}")
        grid_results.append({
            "config_id"   : i,
            "hparams"     : hp,
            "val_macro_f1": result["val_macro_f1_mean"],
        })

    grid_results.sort(key=lambda x: x["val_macro_f1"], reverse=True)
    print(f"\nGrid search results:")
    for r in grid_results:
        print(f"  F1={r['val_macro_f1']:.4f}  {r['hparams']}")

    save_results(
        {"grid_results": grid_results},
        RESULTS_DIR / f"grid_search_attn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    best_hp = {**HPARAM_DEFAULTS, **grid_results[0]["hparams"]}
    print(f"\nBest: {grid_results[0]['hparams']} — re-training on all folds...")
    return run_single(folds, f_dim, max_len, best_hp, device,
                      use_meta, meta_stats_per_fold, run_tag="best")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Attentive Pooling Transformer (Model 3)"
    )
    p.add_argument("--preset", choices=["small", "full"], default="small",
                   help="'small': d=32, L=1 (current data). "
                        "'full': d=64, L=2 (when 300+ examples available).")
    p.add_argument("--d_model",    type=int,   default=None)
    p.add_argument("--nhead",      type=int,   default=None)
    p.add_argument("--num_layers", type=int,   default=None)
    p.add_argument("--dropout",    type=float, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--patience",   type=int,   default=None)
    p.add_argument("--baseline_test_f1", type=float, default=None,
                   help="Test macro-F1 from the audio-only Transformer "
                        "(used to compute the attentive pooling Δ in the summary).")
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

    # Hparams
    base    = HPARAM_SMALL if args.preset == "small" else HPARAM_FULL
    hparams = dict(base)
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
        sys.exit(1)
    df = df[df["png_exists"]].copy()

    print("\n[2/5] Auditing spectrograms...")
    audit_stats = audit_spectrograms(df["png_path"].tolist(), verbose=True)
    f_dim   = audit_stats["f_dim"]
    max_len = audit_stats["max_len_p95"]

    # --- Metadata viability check ---
    print("\n[2b] Checking metadata viability...")
    meta_check = check_metadata_viability(df, verbose=True)
    use_meta   = meta_check["USE_EXTERNAL_META"]

    # --- Splits ---
    print("\n[3/5] Constructing splits...")
    if args.use_existing_splits:
        protocol = determine_eval_protocol(df)
        folds    = load_splits(protocol)
    else:
        protocol = determine_eval_protocol(df)
        folds    = build_splits(df, protocol, save=True)

    # Compute meta stats per fold (from each fold's training split)
    print("\n  Computing metadata normalisation stats per fold...")
    meta_stats_per_fold = []
    for train_df, val_df, test_df in folds:
        meta_stats_per_fold.append(compute_meta_stats(train_df))

    # --- Train ---
    print(f"\n[4/5] Training{'  (grid search)' if args.grid_search else ''}...")
    baseline_f1 = args.baseline_test_f1
    if baseline_f1 is None:
        print("  ℹ  Pass --baseline_test_f1 <value> to compare against the "
              "audio-only Transformer (Model 2).")

    if args.grid_search:
        result = run_grid_search(folds, f_dim, max_len, device,
                                 use_meta, meta_stats_per_fold)
    else:
        result = run_single(folds, f_dim, max_len, hparams, device,
                            use_meta, meta_stats_per_fold,
                            run_tag="single",
                            baseline_test_f1=baseline_f1)

    print(f"\n[5/5] Done.")
    print(f"  Query mode    : {result['query_mode']}")
    print(f"  Val  macro-F1 : {result['val_macro_f1_mean']:.4f} "
          f"± {result['val_macro_f1_std']:.4f}")
    print(f"  Test macro-F1 : {result['test_macro_f1_mean']:.4f} "
          f"± {result['test_macro_f1_std']:.4f}")
    print(f"  Chance level  : {result['chance_level']:.4f}  "
          f"({result['n_present_classes']} classes)")
    print(f"\nAll outputs → {cfg.RES_DIR}/")


if __name__ == "__main__":
    main()
