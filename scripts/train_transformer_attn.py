"""
train_transformer_attn.py
--------------------------
Model 3: Attentive Pooling Transformer (main model).

Same encoder as Model 2 but uses query-based attentive pooling instead
of mean pooling. The query source is determined automatically:

  - 9 participants, age has 5 unique values, gender has 2 → metadata IS viable
  → USE_EXTERNAL_META = True (age + gender projected to query vector)

Usage:
    python scripts/train_transformer_attn.py
    python scripts/train_transformer_attn.py --baseline_test_f1 0.XX
    python scripts/train_transformer_attn.py --grid_search
    python scripts/train_transformer_attn.py --use_existing_splits

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
    MetadataSpectrogramDataset,
    check_metadata_viability, compute_meta_stats,
)
from models import AttentivePoolingTransformer
from train_utils import (
    train_model, evaluate, analyse_attention_weights,
    print_classification_report, save_results,
    sanity_check_forward, sanity_check_dataloader,
)


# ---------------------------------------------------------------------------
# Plotting
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
        pass


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
        pass


def plot_attention_weights(model, val_df, label_encoder, max_len, f_dim,
                           device, use_meta, meta_stats, save_path):
    """Visualise per-class attention weight profiles for one example each."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from data_utils import load_png

        present_classes = sorted(val_df["class4"].unique())
        n_cls = len(present_classes)
        fig, axes = plt.subplots(n_cls, 1, figsize=(14, 3 * n_cls), squeeze=False)
        model.eval()

        for row_i, cls_name in enumerate(present_classes):
            cls_df = val_df[val_df["class4"] == cls_name]
            if len(cls_df) == 0: continue
            sample = cls_df.iloc[0]
            spec   = load_png(sample["png_path"], target_height=f_dim)
            T_real = min(spec.shape[1], max_len)

            ds = MetadataSpectrogramDataset(
                cls_df.head(1), label_encoder, max_len, f_dim,
                meta_stats=meta_stats, use_external_meta=use_meta)
            x, mask, _, meta_t = ds[0]
            x = x.unsqueeze(0).to(device)
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
            ax.set_title(f"Class: {cls_name}", fontsize=10)
            ax.set_xlabel("Time frame"); ax.set_ylabel("Attention weight")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

        fig.suptitle("Attention Weight Profiles by Class", y=1.01, fontsize=11)
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Attention plot → {save_path}")
    except Exception as e:
        print(f"Attention plot skipped: {e}")


# ---------------------------------------------------------------------------
# Result interpretation
# ---------------------------------------------------------------------------

def print_fold_interpretation(fold_results, n_present_classes, baseline_f1=None):
    vals   = [r["val_macro_f1"]  for r in fold_results]
    tests  = [r["test_macro_f1"] for r in fold_results]
    chance = 1.0 / n_present_classes
    gaps   = [v - t for v, t in zip(vals, tests)]

    print(f"\n{'='*55}")
    print("RESULT INTERPRETATION — Attentive Pooling Transformer")
    print(f"{'='*55}")
    print(f"  Classes        : {n_present_classes} active (of {N_CLASSES})")
    print(f"  Chance level   : {chance:.3f}")
    print(f"  Val  macro-F1  : {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    print(f"  Test macro-F1  : {np.mean(tests):.3f} ± {np.std(tests):.3f}")
    print(f"  Val→Test gap   : {np.mean(gaps):+.3f}")

    if baseline_f1 is not None:
        delta = np.mean(tests) - baseline_f1
        flag  = ("✓ helps" if delta > 0.02 else
                 ("✗ hurts" if delta < -0.02 else "≈ no difference"))
        print(f"\n  Audio-only baseline : {baseline_f1:.3f}")
        print(f"  Attentive pooling   : {np.mean(tests):.3f}  Δ={delta:+.3f}  {flag}")

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

def run_single(folds, f_dim, max_len, hparams, device,
               use_meta, meta_stats_per_fold,
               run_tag="single", baseline_f1=None):
    cfg.make_output_dirs()
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    query_mode  = "external_meta" if use_meta else "learned_query"

    for fold_i, (train_df, val_df, test_df) in enumerate(folds):
        present   = sorted(
            set(train_df["class4"]) | set(val_df["class4"]) | set(test_df["class4"])
        )
        n_present  = len(present)
        meta_stats = meta_stats_per_fold[fold_i]
        fold_tag   = f"attn_{run_tag}_fold{fold_i}_{timestamp}"

        print(f"\n{'='*55}")
        print(f"FOLD {fold_i}  [AttentivePooling — {query_mode}]")
        print(f"  {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")
        print(f"  Classes : {present}  |  Chance : {1/n_present:.3f}")
        print(f"  d={hparams['d_model']}, L={hparams['num_layers']}, "
              f"drop={hparams['dropout']}, lr={hparams['lr']}, "
              f"batch={hparams['batch_size']}")
        print(f"{'='*55}")

        train_ds = MetadataSpectrogramDataset(
            train_df, LABEL_ENCODER, max_len, f_dim,
            meta_stats=meta_stats, use_external_meta=use_meta, train=True)
        val_ds   = MetadataSpectrogramDataset(
            val_df,   LABEL_ENCODER, max_len, f_dim,
            meta_stats=meta_stats, use_external_meta=use_meta, train=False)
        test_ds  = MetadataSpectrogramDataset(
            test_df,  LABEL_ENCODER, max_len, f_dim,
            meta_stats=meta_stats, use_external_meta=use_meta, train=False)

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
            sanity_check_dataloader(train_loader, f_dim, max_len, use_meta=use_meta)

        model = AttentivePoolingTransformer(
            input_dim=f_dim, d_model=hparams["d_model"],
            nhead=hparams["nhead"], num_layers=hparams["num_layers"],
            dropout=hparams["dropout"], num_classes=N_CLASSES,
            max_len=max_len + 50,
            use_external_meta=use_meta, meta_dim=2,
        ).to(device)

        if fold_i == 0:
            model.print_architecture()
            sanity_check_forward(model, f_dim, max_len, N_CLASSES, device,
                                 use_meta=use_meta)

        criterion = torch.nn.CrossEntropyLoss(
            weight=train_ds.class_weights().to(device))

        save_path = MODELS_DIR / f"transformer_{fold_tag}.pt"
        print(f"\nTraining...")
        train_result = train_model(
            model, train_loader, val_loader, criterion, device,
            lr=hparams["lr"], weight_decay=hparams["weight_decay"],
            epochs=hparams["epochs"], patience=hparams["patience"],
            save_path=save_path, verbose=True, use_meta=use_meta,
        )

        plot_training_history(
            train_result["history"],
            FIGURES_DIR / f"training_curves_{fold_tag}.png",
            title=(f"AttentivePooling — Fold {fold_i} "
                   f"({query_mode}, d={hparams['d_model']}, "
                   f"L={hparams['num_layers']}, {n_present} classes)"),
            n_present_classes=n_present,
        )

        model.load_state_dict(torch.load(save_path, map_location=device))
        val_result  = evaluate(model, val_loader,  device,
                               use_meta=use_meta, return_preds=True)
        test_result = evaluate(model, test_loader, device,
                               use_meta=use_meta, return_preds=True)

        print_classification_report(val_result["preds"],  val_result["labels"],
                                    f"Validation (fold {fold_i})")
        print_classification_report(test_result["preds"], test_result["labels"],
                                    f"Test (fold {fold_i})")

        plot_confusion_matrix(val_result["preds"],  val_result["labels"],
                              FIGURES_DIR / f"cm_val_{fold_tag}.png",
                              title=f"Val — Fold {fold_i} (Attentive)")
        plot_confusion_matrix(test_result["preds"], test_result["labels"],
                              FIGURES_DIR / f"cm_test_{fold_tag}.png",
                              title=f"Test — Fold {fold_i} (Attentive)")

        # Attention weight analysis
        if "attn_weights" in val_result:
            attn_stats = analyse_attention_weights(val_result["attn_weights"])
            print(f"\n  Attention stats (val):")
            print(f"    Mean weight std : {attn_stats['mean_weight_std']:.5f}")
            print(f"    Uniform proxy   : {attn_stats['uniform_std_proxy']:.5f}")
            print(f"    Non-uniform     : {attn_stats['is_non_uniform']}")
        else:
            attn_stats = {}

        # Attention visualisation (fold 0 only)
        if fold_i == 0:
            plot_attention_weights(
                model, val_df, LABEL_ENCODER, max_len, f_dim,
                device, use_meta, meta_stats,
                FIGURES_DIR / f"attn_weights_{fold_tag}.png",
            )

        fold_summary = {
            "fold"              : fold_i,
            "query_mode"        : query_mode,
            "hparams"           : hparams,
            "train_n"           : len(train_df),
            "val_n"             : len(val_df),
            "test_n"            : len(test_df),
            "n_present_classes" : n_present,
            "present_classes"   : present,
            "chance_level"      : round(1 / n_present, 4),
            "best_val_f1"       : train_result["best_val_f1"],
            "val_macro_f1"      : val_result["macro_f1"],
            "val_per_class"     : val_result["per_class_f1"],
            "test_macro_f1"     : test_result["macro_f1"],
            "test_per_class"    : test_result["per_class_f1"],
            "attn_weight_stats" : attn_stats,
            "stopped_epoch"     : train_result["stopped_epoch"],
            "elapsed_s"         : train_result["elapsed_s"],
            "checkpoint"        : str(save_path),
            # ── Added for multi-seed wrapper ─────────────────────────────
            # The wrapper (scripts/train_multiseed.py) needs these fields
            # to build training-curve overlays and averaged confusion
            # matrices across seeds. Adding them is a pure data widening:
            # nothing about the per-fold training, evaluation, or saved
            # JSON shape changes for the existing single-run workflow,
            # because save_results() simply serialises whatever dict we
            # hand it (lists of ints/floats are JSON-safe).
            "history"           : train_result["history"],
            "val_preds"         : val_result["preds"],
            "val_labels"        : val_result["labels"],
            "test_preds"        : test_result["preds"],
            "test_labels"       : test_result["labels"],
        }
        all_results.append(fold_summary)
        save_results(fold_summary, RESULTS_DIR / f"fold_results_{fold_tag}.json")

    val_f1s = [r["val_macro_f1"]  for r in all_results]
    test_f1s= [r["test_macro_f1"] for r in all_results]
    n_pres  = all_results[0]["n_present_classes"]

    summary = {
        "model"              : "AttentivePoolingTransformer",
        "query_mode"         : query_mode,
        "run_tag"            : run_tag,
        "hparams"            : hparams,
        "n_folds"            : len(folds),
        "n_present_classes"  : n_pres,
        "chance_level"       : round(1 / n_pres, 4),
        "val_macro_f1_mean"  : float(np.mean(val_f1s)),
        "val_macro_f1_std"   : float(np.std(val_f1s)),
        "test_macro_f1_mean" : float(np.mean(test_f1s)),
        "test_macro_f1_std"  : float(np.std(test_f1s)),
        "fold_results"       : all_results,
    }
    print_fold_interpretation(all_results, n_pres, baseline_f1)
    save_results(summary,
                 RESULTS_DIR / f"transformer_attn_{run_tag}_{timestamp}.json")
    return summary


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search(folds, f_dim, max_len, device,
                    use_meta, meta_stats_per_fold):
    print(f"\n{'='*55}")
    print(f"GRID SEARCH — AttentivePooling ({len(HPARAM_GRID)} configs)")
    print(f"{'='*55}")

    grid_results = []
    for i, hp in enumerate(HPARAM_GRID):
        hp_full = {**HPARAM_DEFAULTS, **hp}
        print(f"\nConfig {i+1}/{len(HPARAM_GRID)}: {hp}")
        result = run_single(folds[:1], f_dim, max_len, hp_full, device,
                            use_meta, meta_stats_per_fold[:1],
                            run_tag=f"grid_{i}")
        grid_results.append({
            "config_id": i, "hparams": hp,
            "val_macro_f1": result["val_macro_f1_mean"],
        })

    grid_results.sort(key=lambda x: x["val_macro_f1"], reverse=True)
    print(f"\nGrid results (sorted):")
    for r in grid_results:
        print(f"  F1={r['val_macro_f1']:.4f}  {r['hparams']}")

    save_results({"grid_results": grid_results},
                 RESULTS_DIR / f"grid_search_attn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    best_hp = {**HPARAM_DEFAULTS, **grid_results[0]["hparams"]}
    print(f"\nBest: {grid_results[0]['hparams']} — re-training on all folds...")
    return run_single(folds, f_dim, max_len, best_hp, device,
                      use_meta, meta_stats_per_fold, run_tag="best")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Attentive Pooling Transformer (Model 3)")
    p.add_argument("--preset", choices=["small", "full"], default="full")
    p.add_argument("--d_model",    type=int,   default=None)
    p.add_argument("--nhead",      type=int,   default=None)
    p.add_argument("--num_layers", type=int,   default=None)
    p.add_argument("--dropout",    type=float, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--patience",   type=int,   default=None)
    p.add_argument("--baseline_test_f1", type=float, default=None,
                   help="Test F1 from audio-only Transformer for comparison")
    p.add_argument("--grid_search",         action="store_true")
    p.add_argument("--use_existing_splits", action="store_true")
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
                "lr", "batch_size", "epochs", "patience"]:
        val = getattr(args, key)
        if val is not None:
            hparams[key] = val

    print(f"\nHyperparameters (preset='{args.preset}'):")
    for k, v in hparams.items():
        print(f"  {k:15s}: {v}")

    # Data
    print("\n[1/5] Building master table...")
    df = build_master_table(verbose=True)

    if df["png_exists"].sum() == 0:
        print(f"\n✗ No spectrograms in {cfg.SPEC_DIR}")
        print("  Run first:  python scripts/generate_spectrograms.py")
        sys.exit(1)
    df = df[df["png_exists"]].copy()

    print("\n[2/5] Auditing spectrograms...")
    audit_spectrograms(df["png_path"].tolist(), verbose=True)
    # Always use config values — the model works at mel resolution (128×200),
    # NOT the PNG pixel resolution (400×1000). The resize happens inside
    # SpectrogramDataset.__getitem__ using these values.
    f_dim   = cfg.F_DIM    # 128
    max_len = cfg.MAX_LEN  # 200

    if False:  # silent clip check disabled — audit uses sample
        from PIL import Image as _I
        import numpy as _np
        df = df[df["png_path"].map(
            lambda p: _np.array(_I.open(p), dtype=_np.float32).max() > 0
        )].copy()
        print(f"  Removed {stats['n_silent']} silent clip(s)")

    print("\n[2b] Checking metadata viability...")
    meta_check = check_metadata_viability(df, verbose=True)
    use_meta   = meta_check["USE_EXTERNAL_META"]

    print("\n[3/5] Constructing splits...")
    if args.use_existing_splits:
        protocol = determine_eval_protocol(df)
        folds    = load_splits(protocol)
    else:
        protocol = determine_eval_protocol(df)
        folds    = build_splits(df, protocol, save=True)

    print("\n  Computing metadata stats per fold (from train split only)...")
    meta_stats_per_fold = [compute_meta_stats(tr) for tr, _, _ in folds]

    print(f"\n[4/5] Training{'  (grid search)' if args.grid_search else ''}...")
    if not args.baseline_test_f1:
        print("  ℹ  Run with --baseline_test_f1 <value> to compare vs Model 2.")

    if args.grid_search:
        result = run_grid_search(folds, f_dim, max_len, device,
                                 use_meta, meta_stats_per_fold)
    else:
        result = run_single(folds, f_dim, max_len, hparams, device,
                            use_meta, meta_stats_per_fold,
                            baseline_f1=args.baseline_test_f1)

    print(f"\n[5/5] Done.")
    print(f"  Query mode    : {result['query_mode']}")
    print(f"  Val  macro-F1 : {result['val_macro_f1_mean']:.4f} "
          f"± {result['val_macro_f1_std']:.4f}")
    print(f"  Test macro-F1 : {result['test_macro_f1_mean']:.4f} "
          f"± {result['test_macro_f1_std']:.4f}")
    print(f"  Chance level  : {result['chance_level']:.4f} "
          f"({result['n_present_classes']} classes)")
    print(f"\nOutputs saved to {RES_DIR}/")


if __name__ == "__main__":
    main()