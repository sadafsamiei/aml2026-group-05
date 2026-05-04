"""
Line search for LR regularisation strength C using Optuna.
Replaces the 9-point logspace grid with a continuous log-uniform search.
Uses the same inner LOPO CV as the original train_lr_baseline.py.

Feature path resolution
-----------------------
- mel       → row["png_path"]                       (always present in master table)
- mfcc      → MFCC_DIR / f"{row['audio_ID']}.npy"   (built from audio_ID)
- wav2vec2  → PRETRAINED_DIR / f"{row['audio_ID']}.npy"  (built from audio_ID)

The mfcc and wav2vec2 paths are NOT columns in the master table — they are
constructed from `audio_ID` + the directory constants in config.py.
"""
import sys, json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score

import config as cfg
from config import (
    LABEL_ENCODER, RESULTS_DIR, RANDOM_SEED,
    MFCC_DIR, PRETRAINED_DIR,
)
from data_utils import build_master_table, determine_eval_protocol, load_splits


# ── Feature extraction ──────────────────────────────────────────────────

def extract_features(df, feature="mel"):
    """
    Extract mean+std summary features per clip.

    For mel: read PNG from row["png_path"], compute mean/std along time axis.
    For mfcc: load MFCC_DIR/{audio_ID}.npy with shape (20, T), pool over time.
    For wav2vec2: load PRETRAINED_DIR/{audio_ID}.npy with shape (768, T), pool over time.

    Both .npy files are stored as (F, T) — pool along axis=1 (time).
    """
    from PIL import Image

    X, y, groups = [], [], []
    n_skipped_missing = 0
    n_skipped_error   = 0

    for _, row in df.iterrows():
        try:
            if feature == "mel":
                img = np.array(Image.open(row["png_path"]).convert("L"),
                               dtype=np.float32) / 255.0   # (F, T)
                feat = np.concatenate([img.mean(axis=1), img.std(axis=1)])

            elif feature == "mfcc":
                npy_path = MFCC_DIR / f"{row['audio_ID']}.npy"
                if not npy_path.exists():
                    n_skipped_missing += 1
                    continue
                arr = np.load(npy_path)                       # (20, T)
                feat = np.concatenate([arr.mean(axis=1), arr.std(axis=1)])

            elif feature == "wav2vec2":
                npy_path = PRETRAINED_DIR / f"{row['audio_ID']}.npy"
                if not npy_path.exists():
                    n_skipped_missing += 1
                    continue
                arr = np.load(npy_path)                       # (768, T)
                feat = np.concatenate([arr.mean(axis=1), arr.std(axis=1)])

            else:
                raise ValueError(f"Unknown feature: {feature}")

            X.append(feat)
            y.append(LABEL_ENCODER[row["class4"]])
            groups.append(row["child"])

        except Exception as e:
            n_skipped_error += 1
            if n_skipped_error <= 3:
                print(f"  ! skipped {row['audio_ID']}: {type(e).__name__}: {e}")

    if n_skipped_missing > 0:
        print(f"  ! {n_skipped_missing} clips had missing {feature} files")
    if n_skipped_error > 0:
        print(f"  ! {n_skipped_error} clips had load errors (first 3 shown above)")

    return np.array(X), np.array(y), np.array(groups)


def inner_lopo_cv(X, y, groups, C):
    """Inner LOPO CV — same protocol as the original training script."""
    logo = LeaveOneGroupOut()
    scores = []
    for train_idx, val_idx in logo.split(X, y, groups):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=C, max_iter=5000, solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ))
        ])
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], preds, average="macro",
                               zero_division=0))
    return float(np.mean(scores))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--features", choices=["mel", "mfcc", "wav2vec2"],
                   default="mel")
    p.add_argument("--n_trials", type=int, default=80)
    args = p.parse_args()

    cfg.print_config()
    cfg.make_output_dirs()

    print("\n[1/3] Building master table...")
    df = build_master_table(verbose=False)
    df = df[df["png_exists"]].copy()

    print("[2/3] Loading existing splits...")
    protocol = determine_eval_protocol(df)
    folds    = load_splits(protocol)
    train_df, val_df, test_df = folds[0]

    print(f"[3/3] Extracting {args.features} features...")
    X_train, y_train, groups = extract_features(train_df, args.features)
    X_val,   y_val,   _      = extract_features(val_df,   args.features)
    X_test,  y_test,  _      = extract_features(test_df,  args.features)
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    if X_train.size == 0:
        raise SystemExit(
            f"\nERROR: zero {args.features} features extracted. "
            f"Check that the .npy files exist:\n"
            f"  MFCC_DIR       = {MFCC_DIR}\n"
            f"  PRETRAINED_DIR = {PRETRAINED_DIR}\n"
            f"and that {args.features} files for the audio_IDs in your splits "
            f"are present there.")

    print(f"\nOptuna line search — C log-uniform over [1e-6, 1e+6]")
    print(f"Trials: {args.n_trials}  (inner LOPO CV per trial)\n")

    def objective(trial):
        C = trial.suggest_float("C", 1e-6, 1e+6, log=True)
        score = inner_lopo_cv(X_train, y_train, groups, C)
        # Fit on full train and evaluate val+test for logging
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=C, max_iter=5000, solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ))
        ])
        pipe.fit(X_train, y_train)
        val_f1  = f1_score(y_val,  pipe.predict(X_val),  average="macro",
                           zero_division=0)
        test_f1 = f1_score(y_test, pipe.predict(X_test), average="macro",
                           zero_division=0)
        trial.set_user_attr("val_f1",  val_f1)
        trial.set_user_attr("test_f1", test_f1)
        print(f"  Trial {trial.number:3d}/{args.n_trials}"
              f"  C={C:.5f}  inner_cv={score:.4f}"
              f"  val={val_f1:.4f}  test={test_f1:.4f}")
        return score   # optimise inner CV score (same as original)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    # Seed with your 9 canonical grid points so Optuna starts informed
    for C_init in np.logspace(-4, 4, 9):
        study.enqueue_trial({"C": float(C_init)})

    study.optimize(objective, n_trials=args.n_trials,
                   show_progress_bar=False)

    best = study.best_trial
    best_C      = best.params["C"]
    best_inner  = best.value
    best_val    = best.user_attrs["val_f1"]
    best_test   = best.user_attrs["test_f1"]

    print(f"\n{'='*55}")
    print(f"BEST — C = {best_C:.6f}")
    print(f"  inner CV F1 : {best_inner:.4f}")
    print(f"  val  F1     : {best_val:.4f}")
    print(f"  test F1     : {best_test:.4f}")
    canonical_test = {"mel": 0.277, "mfcc": 0.253, "wav2vec2": 0.263}[args.features]
    print(f"  original 9-pt grid best ({args.features}): test={canonical_test:.4f}")
    print(f"{'='*55}")

    all_trials = [
        {"trial": t.number, "C": t.params["C"],
         "inner_cv": t.value,
         "val_f1":  t.user_attrs.get("val_f1"),
         "test_f1": t.user_attrs.get("test_f1")}
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    all_trials.sort(key=lambda x: x["inner_cv"], reverse=True)

    result = {
        "model": "LogisticRegression",
        "features": args.features,
        "n_trials": args.n_trials,
        "search_space": {"C": "log-uniform [1e-6, 1e+6]"},
        "best_C": best_C,
        "best_inner_cv_f1": best_inner,
        "best_val_f1": best_val,
        "best_test_f1": best_test,
        "original_grid_test_f1": canonical_test,
        "all_trials": all_trials[:20],   # top 20
    }
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"optuna_lr_{args.features}_{ts}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved → {path}")


if __name__ == "__main__":
    main()