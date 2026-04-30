"""
experiments/optuna_search.py
----------------------------
Bayesian hyperparameter optimisation using Optuna for Models 2 and 3.

Replaces the discrete grid search with a 40-trial TPE study that searches
CONTINUOUSLY over lr, dropout_encoder, dropout_head, dropout_input and
CATEGORICALLY over d_model, num_layers.

IMPORTANT — Feature type is controlled by config.py, NOT by --features.
Before running:
  mel non-pretrained  → USE_PRETRAINED=False, USE_MFCC=False  in config.py
  wav2vec2 pretrained → USE_PRETRAINED=True                   in config.py
  MFCC                → USE_MFCC=True                         in config.py

The --features flag is only used for labelling the output file.
Mismatching config.py and --features will produce incorrect output filenames
but will not break the run.

Usage
-----
    # Mel non-pretrained (make sure USE_PRETRAINED=False, USE_MFCC=False in config.py):
    python experiments/optuna_search.py --model audio --features mel --n_trials 40

    # Model 3 attentive, mel:
    python experiments/optuna_search.py --model attn --features mel --n_trials 40

    # Increase trials (fast GPU — recommend 60):
    python experiments/optuna_search.py --model audio --features mel --n_trials 60

    # Suppress Optuna progress bar:
    python experiments/optuna_search.py --model audio --features mel --quiet

Outputs (same format as grid_search_*.json)
-------------------------------------------
    RES_DIR/results/optuna_search_{model}_{features}_{timestamp}.json

Requirements
------------
    pip install optuna
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
    N_CLASSES, LABEL_ENCODER,
    MODELS_DIR, RESULTS_DIR,
    RANDOM_SEED, F_DIM, MAX_LEN,
    USE_PRETRAINED, USE_MFCC,
)
from data_utils import (
    build_master_table, audit_spectrograms,
    determine_eval_protocol, build_splits, load_splits,
    SpectrogramDataset, MetadataSpectrogramDataset, compute_meta_stats,
)
from models import AudioTransformer, AttentivePoolingTransformer
from train_utils import train_model, evaluate, save_results

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: optuna is not installed.  Run:  pip install optuna")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Fixed training settings (same as canonical runs)
# ---------------------------------------------------------------------------

FIXED = {
    "nhead"          : 4,
    "epochs"         : 100,
    "patience"       : 15,
    "weight_decay"   : 1e-4,
    "label_smoothing": 0.0,
}

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "d_model"        : [32, 64, 128],     # removed 192 — too slow per trial
    "num_layers"     : (1, 2),             # capped at 2 — 3 layers x100 epochs hangs
    "lr"             : (1e-4, 1e-2),       # log-uniform
    "dropout_encoder": (0.1, 0.7),         # uniform
    "dropout_head"   : (0.0, 0.5),         # uniform
    "dropout_input"  : (0.0, 0.4),         # uniform  (NEW location)
    "dropout_meta"   : (0.0, 0.4),         # uniform  (Model 3 only)
}


# ---------------------------------------------------------------------------
# Single trial — train on fold 0 and return val macro-F1
# ---------------------------------------------------------------------------

def _trial_run(trial, model_kind: str,
               folds, f_dim: int, max_len: int, device):

    # ── Sample hyperparameters ──────────────────────────────────────────
    d_model         = trial.suggest_categorical(
                          "d_model", SEARCH_SPACE["d_model"])
    num_layers      = trial.suggest_int(
                          "num_layers", *SEARCH_SPACE["num_layers"])
    lr              = trial.suggest_float(
                          "lr", *SEARCH_SPACE["lr"], log=True)
    dropout_encoder = trial.suggest_float(
                          "dropout_encoder", *SEARCH_SPACE["dropout_encoder"])
    dropout_head    = trial.suggest_float(
                          "dropout_head", *SEARCH_SPACE["dropout_head"])
    dropout_input   = trial.suggest_float(
                          "dropout_input", *SEARCH_SPACE["dropout_input"])

    nhead = FIXED["nhead"]
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned()

    # ── Datasets (fold 0 only for speed) ─────────────────────────────────
    train_df, val_df, test_df = folds[0]

    use_meta = (model_kind == "attn")

    if use_meta:
        # Same as train_transformer_attn.py: compute stats from train split
        meta_stats = compute_meta_stats(train_df)
        train_ds = MetadataSpectrogramDataset(
            train_df, LABEL_ENCODER, max_len, f_dim,
            meta_stats=meta_stats, use_external_meta=True, train=True)
        val_ds   = MetadataSpectrogramDataset(
            val_df,   LABEL_ENCODER, max_len, f_dim,
            meta_stats=meta_stats, use_external_meta=True, train=False)
        test_ds  = MetadataSpectrogramDataset(
            test_df,  LABEL_ENCODER, max_len, f_dim,
            meta_stats=meta_stats, use_external_meta=True, train=False)
    else:
        train_ds = SpectrogramDataset(train_df, LABEL_ENCODER, max_len, f_dim,
                                      train=True)
        val_ds   = SpectrogramDataset(val_df,   LABEL_ENCODER, max_len, f_dim,
                                      train=False)
        test_ds  = SpectrogramDataset(test_df,  LABEL_ENCODER, max_len, f_dim,
                                      train=False)

    bs = 32
    train_loader = DataLoader(train_ds, batch_size=bs,  shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ── Build model ─────────────────────────────────────────────────────
    if model_kind == "audio":
        model = AudioTransformer(
            input_dim       = f_dim,
            d_model         = d_model,
            nhead           = nhead,
            num_layers      = num_layers,
            dropout_encoder = dropout_encoder,
            dropout_input   = dropout_input,
            dropout_head    = dropout_head,
            num_classes     = N_CLASSES,
            max_len         = max_len + 50,
        ).to(device)
    else:
        dropout_meta = trial.suggest_float(
                           "dropout_meta", *SEARCH_SPACE["dropout_meta"])
        model = AttentivePoolingTransformer(
            input_dim         = f_dim,
            d_model           = d_model,
            nhead             = nhead,
            num_layers        = num_layers,
            dropout_encoder   = dropout_encoder,
            dropout_input     = dropout_input,
            dropout_head      = dropout_head,
            dropout_meta      = dropout_meta,
            use_external_meta = True,   # real metadata, not learned query
            meta_dim          = 2,
            num_classes       = N_CLASSES,
            max_len           = max_len + 50,
        ).to(device)

    # ── Loss ────────────────────────────────────────────────────────────
    criterion = torch.nn.CrossEntropyLoss(
        weight=train_ds.class_weights().to(device),
        label_smoothing=FIXED["label_smoothing"],
    )

    # ── Train ───────────────────────────────────────────────────────────
    save_path = MODELS_DIR / f"optuna_trial_{trial.number}.pt"
    train_result = train_model(
        model, train_loader, val_loader, criterion, device,
        lr           = lr,
        weight_decay = FIXED["weight_decay"],
        epochs       = FIXED["epochs"],
        patience     = FIXED["patience"],
        save_path    = save_path,
        verbose      = False,
        use_meta     = use_meta,   # ← tells train_utils to unpack meta from batch
    )

    model.load_state_dict(
        torch.load(save_path, map_location=device))
    val_result  = evaluate(model, val_loader,  device, use_meta=use_meta)
    test_result = evaluate(model, test_loader, device, use_meta=use_meta)

    val_f1 = val_result["macro_f1"]

    # Prune trials that never beat chance (avoids wasting budget on dead configs)
    if val_f1 < 0.15:
        if save_path.exists(): save_path.unlink()
        raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("test_macro_f1",  test_result["macro_f1"])
    trial.set_user_attr("best_val_epoch", train_result.get("stopped_epoch", -1))
    trial.set_user_attr("stopped_epoch",  train_result.get("stopped_epoch", -1))

    if save_path.exists(): save_path.unlink()
    return val_f1


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def run_optuna_search(model_kind: str, features: str, n_trials: int,
                      folds, f_dim: int, max_len: int,
                      device, quiet: bool = False):

    feat_label = ("wav2vec2" if USE_PRETRAINED else
                  "mfcc"     if USE_MFCC       else "mel")
    if feat_label != features:
        print(f"\n  WARNING: config.py says features={feat_label!r} "
              f"but --features={features!r}. "
              f"Output will be labelled '{features}' but trained on '{feat_label}'.")

    print(f"\n{'='*60}")
    print(f"OPTUNA SEARCH — {model_kind.upper()} / {features}")
    print(f"  Sampler  : TPE (Tree-structured Parzen Estimator)")
    print(f"  Trials   : {n_trials}  (trained on fold 0 only)")
    print(f"  Device   : {device}")
    print(f"  f_dim    : {f_dim}   max_len : {max_len}")
    print(f"  New dropout locations: dropout_input, dropout_head, dropout_meta")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction = "maximize",
        sampler   = optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner    = optuna.pruners.MedianPruner(n_startup_trials=5,
                                                n_warmup_steps=0),
    )

    completed = [0]

    def objective(trial):
        val_f1 = _trial_run(trial, model_kind, folds, f_dim, max_len, device)
        completed[0] += 1
        test_f1 = trial.user_attrs.get("test_macro_f1", 0)
        stopped = trial.user_attrs.get("stopped_epoch", -1)
        print(f"  Trial {trial.number:3d}/{n_trials}"
              f"  val={val_f1:.4f}  test={test_f1:.4f}"
              f"  stopped_ep={stopped}"
              f"  params={trial.params}")
        return val_f1

    study.optimize(
        objective,
        n_trials          = n_trials,
        show_progress_bar = False,
    )

    # ── Report ─────────────────────────────────────────────────────────
    best = study.best_trial

    print(f"\n{'='*60}")
    print(f"BEST TRIAL  (trial #{best.number})")
    print(f"  val  macro-F1 : {best.value:.4f}")
    test_f1 = best.user_attrs.get('test_macro_f1')
    print(f"  test macro-F1 : {test_f1:.4f}" if test_f1 else
          f"  test macro-F1 : n/a")
    print(f"  Hyperparameters:")
    for k, v in best.params.items():
        vstr = f"{v:.5f}" if isinstance(v, float) else str(v)
        print(f"    {k:20s}: {vstr}")
    print(f"{'='*60}")

    # Build all_trials table
    all_trials = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            all_trials.append({
                "trial_id"      : t.number,
                "params"        : t.params,
                "val_macro_f1"  : t.value,
                "test_macro_f1" : t.user_attrs.get("test_macro_f1"),
                "best_val_epoch": t.user_attrs.get("best_val_epoch"),
                "stopped_epoch" : t.user_attrs.get("stopped_epoch"),
            })
    all_trials.sort(key=lambda x: x["val_macro_f1"], reverse=True)

    result = {
        "model_kind"    : model_kind,
        "features"      : features,
        "n_trials"      : n_trials,
        "n_complete"    : len(all_trials),
        "best_params"   : best.params,
        "best_val_f1"   : best.value,
        "best_test_f1"  : best.user_attrs.get("test_macro_f1"),
        "search_space"  : {k: list(v) if isinstance(v, list) else v
                           for k, v in SEARCH_SPACE.items()},
        "fixed_hparams" : FIXED,
        "all_trials"    : all_trials,
    }

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"optuna_search_{model_kind}_{features}_{ts}.json"
    save_results(result, path)
    print(f"\nResults saved → {path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for Model 2 or Model 3")
    p.add_argument("--model", choices=["audio", "attn"], default="audio",
                   help="'audio'=Model 2 (mean pool), 'attn'=Model 3 (attention)")
    p.add_argument("--features", choices=["mel", "mfcc", "wav2vec2"],
                   default="mel",
                   help="Label for the output file. Must match config.py feature flags.")
    p.add_argument("--n_trials",  type=int, default=40)
    p.add_argument("--use_existing_splits", action="store_true")
    p.add_argument("--seed",  type=int, default=RANDOM_SEED)
    p.add_argument("--quiet", action="store_true")
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

    # ── Data ────────────────────────────────────────────────────────────
    print("\n[1/3] Building master table...")
    df = build_master_table(verbose=True)
    df = df[df["png_exists"]].copy()

    print("\n[2/3] Auditing spectrograms...")
    audit_spectrograms(df["png_path"].tolist(), verbose=False)
    f_dim   = cfg.F_DIM
    max_len = cfg.MAX_LEN

    print("\n[3/3] Loading existing splits (never rebuilt by this script)...")
    protocol = determine_eval_protocol(df)
    folds = load_splits(protocol)

    # ── Search ──────────────────────────────────────────────────────────
    result = run_optuna_search(
        model_kind = args.model,
        features   = args.features,
        n_trials   = args.n_trials,
        folds      = folds,
        f_dim      = f_dim,
        max_len    = max_len,
        device     = device,
        quiet      = args.quiet,
    )

    print(f"\nSummary:")
    print(f"  Best val  F1 : {result['best_val_f1']:.4f}")
    print(f"  Best test F1 : {result['best_test_f1']:.4f}"
          if result['best_test_f1'] else "  Best test F1 : n/a")
    print(f"  Completed    : {result['n_complete']} / {result['n_trials']} trials")
    print(f"\nNext step: re-train the winner with 5 seeds and compare to")
    print(f"  canonical M2 mel: 0.249 ± 0.020")
    print(f"  canonical M3 mel: 0.221 ± 0.013")


if __name__ == "__main__":
    main()