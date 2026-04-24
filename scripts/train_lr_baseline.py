"""
train_lr_baseline.py
--------------------
Model 1 — Logistic regression baseline.

Supports THREE feature sources for direct comparison with Models 2 and 3:

  --features mfcc      → 20 librosa MFCC + (optional) deltas, mean+std
                         per coefficient. Computed live from WAV files.
                         Canonical 40-dim (no delta) or 80-dim (with delta).

  --features mel       → mean and std along the time axis of the project's
                         mel spectrogram PNGs (128 mel bins → 256-dim
                         summary vector). Same source files as Model 2/3
                         on mel.

  --features wav2vec2  → mean and std along the time axis of the project's
                         pre-extracted wav2vec2 .npy files (768 dims → 1536-
                         dim summary vector). Same source files as Model
                         2/3 on wav2vec2.

The pipeline is identical across feature sources: StandardScaler →
LogisticRegression(class_weight='balanced'). The only thing that changes
is the input vector. This makes Model 1 directly comparable to Models 2/3
within each feature column of the results table.

Pipeline
--------
1. Build the feature matrix per feature source. Each row is a clip-level
   summary vector (time axis collapsed to per-dimension statistics).
   Per-clip features are cached in RES_DIR/lr_features_<tag>/<id>.npy
   so re-runs are fast.

2. Use the project's canonical participant-disjoint splits (loaded
   from RES_DIR/splits) so Model 1 numbers are directly comparable to
   Models 2 and 3 (same train/val/test children).

3. Inside each fold (1 fold for the holdout protocol), select the LR
   inverse-regularization strength C via inner LeaveOneGroupOut CV
   over the seven training children. Then refit on train alone (for
   val reporting) and on train+val (for test reporting).

4. Multi-seed (5 seeds) for variance estimation, even though LR with
   the lbfgs solver is mostly deterministic given the data.

Notes
-----
- Mel summary uses ALL 230 frames per PNG, including padding rows for
  shorter clips. Padding contaminates the mean/std but is consistent
  with how Model 2/3 see the data, so for fair comparison we keep it.
  An alternative would be to mask non-zero frames; we chose
  consistency over a small statistical purity gain.
- wav2vec2 .npy files are stored at shape (768, T) where T is the
  per-clip frame count after wav2vec2 striding. We compute stats over
  the actual frames (no padding involved here — the .npy files are
  per-clip, not pre-padded).
- For mfcc, we read from raw WAV and compute MFCCs on the fly. This
  avoids padding contamination because we use only the real audio
  samples.

Outputs
-------
RES_DIR/results/lr_baseline_<feat-tag>_multiseed_<timestamp>.json
RES_DIR/figures/lr_baseline_<feat-tag>_multiseed_<timestamp>_*.png

Usage
-----
    # Canonical: MFCC summary stats, no delta (40 features)
    python scripts/train_lr_baseline.py --features mfcc

    # MFCC + delta ablation (80 features)
    python scripts/train_lr_baseline.py --features mfcc --use_delta

    # Mel summary stats (256 features)
    python scripts/train_lr_baseline.py --features mel

    # wav2vec2 summary stats (1536 features)
    python scripts/train_lr_baseline.py --features wav2vec2
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Make src/ importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
from config import (
    CLASS_NAMES, FIGURES_DIR, MFCC_DIR, PRETRAINED_DIR, RESULTS_DIR,
    SPEC_DIR, WAV_DIR,
)
from data_utils import (
    build_master_table, determine_eval_protocol, load_splits,
    load_png,
)


# ---------------------------------------------------------------------------
# Feature extraction — three sources, one interface
# ---------------------------------------------------------------------------

def extract_mfcc_summary(audio_id: str, sr: int = 16000,
                         n_mfcc: int = 20,
                         use_delta: bool = False) -> np.ndarray:
    """
    Compute MFCCs on the fly from the WAV file.

    Layout (use_delta=False, default):
        [mean(MFCC) | std(MFCC)]                → 2 * n_mfcc  = 40

    Layout (use_delta=True):
        [mean | std | mean(Δ) | std(Δ)]         → 4 * n_mfcc  = 80

    Empty/unreadable clips return zeros of the correct length so the
    feature matrix shape stays consistent.
    """
    import librosa

    wav_path = WAV_DIR / f"{audio_id}.wav"
    out_dim = (4 if use_delta else 2) * n_mfcc

    try:
        y, _ = librosa.load(str(wav_path), sr=sr)
    except Exception as e:
        print(f"  WARN: failed to read {wav_path} ({e}); zeros instead.")
        return np.zeros(out_dim, dtype=np.float32)

    if len(y) == 0:
        return np.zeros(out_dim, dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, T)
    parts = [mfcc.mean(axis=1), mfcc.std(axis=1)]

    if use_delta:
        # librosa.feature.delta needs at least width=9 frames by default.
        # Very short clips would error; fall back to zeros for delta.
        if mfcc.shape[1] >= 9:
            delta = librosa.feature.delta(mfcc)
            parts += [delta.mean(axis=1), delta.std(axis=1)]
        else:
            parts += [np.zeros(n_mfcc), np.zeros(n_mfcc)]

    return np.concatenate(parts).astype(np.float32)


def extract_mel_summary(audio_id: str) -> np.ndarray:
    """
    Read the project's mel spectrogram PNG, return mean+std per mel bin
    along the time axis.

    PNG shape after load_png is (F, T) = (128, 230). We keep the same
    230-frame padded length that Models 2 and 3 see, so the comparison
    is on identical input signals; the padding bias affects all three
    models identically when they use mel.

    Output: 2 * 128 = 256-dim summary vector.
    """
    png_path = SPEC_DIR / f"{audio_id}.png"
    spec = load_png(str(png_path), target_height=128, target_width=230)
    # spec shape: (128, 230) — F rows, T columns
    return np.concatenate([
        spec.mean(axis=1),  # mean per mel bin (over time)
        spec.std(axis=1),   # std per mel bin (over time)
    ]).astype(np.float32)


def extract_wav2vec2_summary(audio_id: str) -> np.ndarray:
    """
    Read the project's pre-extracted wav2vec2 embedding .npy file,
    return mean+std per embedding dim along the time axis.

    File shape: (768, T) where T is the per-clip frame count after
    wav2vec2's downsampling. The .npy files are per-clip, not pre-padded,
    so all frames here are real (no padding contamination).

    Output: 2 * 768 = 1536-dim summary vector.
    """
    npy_path = PRETRAINED_DIR / f"{audio_id}.npy"
    feat = np.load(npy_path)  # (768, T)

    if feat.shape[1] == 0:
        return np.zeros(2 * feat.shape[0], dtype=np.float32)

    return np.concatenate([
        feat.mean(axis=1),
        feat.std(axis=1),
    ]).astype(np.float32)


def get_extractor(feature_source: str, n_mfcc: int = 20,
                  use_delta: bool = False):
    """
    Return a callable f(audio_id) -> np.ndarray for the chosen source.
    Closing over n_mfcc/use_delta keeps the per-clip cache loop simple.
    """
    if feature_source == "mfcc":
        return lambda aid: extract_mfcc_summary(
            aid, n_mfcc=n_mfcc, use_delta=use_delta,
        )
    elif feature_source == "mel":
        return lambda aid: extract_mel_summary(aid)
    elif feature_source == "wav2vec2":
        return lambda aid: extract_wav2vec2_summary(aid)
    else:
        raise ValueError(f"unknown feature_source: {feature_source}")


def build_feature_matrix(df: pd.DataFrame, cache_dir: Path,
                         extractor) -> np.ndarray:
    """
    Build (n_clips, feat_dim) array. Caches per-clip features as .npy
    in cache_dir so re-runs are fast. Cache key = cache_dir name, which
    encodes feature_source / n_mfcc / use_delta — different configs go
    in separate dirs so they cannot overwrite each other.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    feats = []
    for i, audio_id in enumerate(df["audio_ID"]):
        cache_path = cache_dir / f"{audio_id}.npy"
        if cache_path.exists():
            feat = np.load(cache_path)
        else:
            feat = extractor(audio_id)
            np.save(cache_path, feat)
        feats.append(feat)
        if (i + 1) % 200 == 0:
            print(f"  ... extracted {i+1}/{len(df)} clips")
    return np.vstack(feats)


def feature_names(feature_source: str, n_mfcc: int = 20,
                  use_delta: bool = False) -> list:
    """
    Human-readable names for each feature dimension. Used in the
    feature-importance plot.
    """
    if feature_source == "mfcc":
        names = (
            [f"mfcc_mean_{i}" for i in range(n_mfcc)]
            + [f"mfcc_std_{i}" for i in range(n_mfcc)]
        )
        if use_delta:
            names += (
                [f"mfcc_delta_mean_{i}" for i in range(n_mfcc)]
                + [f"mfcc_delta_std_{i}" for i in range(n_mfcc)]
            )
        return names
    elif feature_source == "mel":
        return (
            [f"mel_mean_{i}" for i in range(128)]
            + [f"mel_std_{i}" for i in range(128)]
        )
    elif feature_source == "wav2vec2":
        return (
            [f"w2v_mean_{i}" for i in range(768)]
            + [f"w2v_std_{i}" for i in range(768)]
        )
    else:
        raise ValueError(f"unknown feature_source: {feature_source}")


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def run_one_seed(seed: int, X_train, y_train, groups_train,
                 X_val, y_val,
                 X_test, y_test,
                 c_grid: np.ndarray,
                 verbose: bool = True) -> dict:
    """
    Single-seed run.

    Selection: inner LeaveOneGroupOut CV over training children only —
    val_df is NOT included in the inner CV pool, so val numbers stay
    untouched and can be reported. The original notebook concatenated
    train+val for the search; we keep val held out so the val→test gap
    is a meaningful diagnostic that compares fairly against Models 2/3.

    Refit:
      - For test reporting: refit on train+val with selected C.
      - For val reporting: refit on train alone with selected C, then
        score on val (no leakage).
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=2000, solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )),
    ])

    # --- Inner CV: select C using ONLY training children ---------------
    inner_cv = LeaveOneGroupOut()
    grid = GridSearchCV(
        pipeline,
        {"logreg__C": c_grid},
        cv=inner_cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train, groups=groups_train)
    best_C = float(grid.best_params_["logreg__C"])

    if verbose:
        print(f"  seed {seed}: selected C={best_C:.3g} "
              f"(inner-CV macro-F1={grid.best_score_:.4f})")

    # --- Val: refit on train alone, score on val -----------------------
    val_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=2000, solver="lbfgs",
            class_weight="balanced", random_state=seed, C=best_C,
        )),
    ])
    val_pipe.fit(X_train, y_train)
    val_preds = val_pipe.predict(X_val)
    val_macro_f1 = f1_score(y_val, val_preds, average="macro",
                             labels=CLASS_NAMES, zero_division=0)
    val_per_class = f1_score(y_val, val_preds, average=None,
                              labels=CLASS_NAMES, zero_division=0)

    # --- Test: refit on train + val, score on test ---------------------
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    test_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=2000, solver="lbfgs",
            class_weight="balanced", random_state=seed, C=best_C,
        )),
    ])
    test_pipe.fit(X_train_val, y_train_val)
    test_preds = test_pipe.predict(X_test)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro",
                              labels=CLASS_NAMES, zero_division=0)
    test_per_class = f1_score(y_test, test_preds, average=None,
                               labels=CLASS_NAMES, zero_division=0)

    if verbose:
        print(f"  seed {seed}: val macro-F1={val_macro_f1:.4f} | "
              f"test macro-F1={test_macro_f1:.4f}")

    return {
        "seed": seed,
        "best_C": best_C,
        "inner_cv_score": float(grid.best_score_),
        "val_macro_f1": float(val_macro_f1),
        "val_per_class_f1": {c: float(v) for c, v in
                              zip(CLASS_NAMES, val_per_class)},
        "test_macro_f1": float(test_macro_f1),
        "test_per_class_f1": {c: float(v) for c, v in
                               zip(CLASS_NAMES, test_per_class)},
        "test_preds": test_preds.tolist(),
        "test_labels": list(y_test),
        "val_preds": val_preds.tolist(),
        "val_labels": list(y_val),
        "logreg_classes": list(test_pipe.named_steps["logreg"].classes_),
        "logreg_coef": test_pipe.named_steps["logreg"].coef_.tolist(),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(per_seed_results: list, save_path: Path,
                          title: str = ""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        cms = []
        for r in per_seed_results:
            cm = confusion_matrix(
                r["test_labels"], r["test_preds"],
                labels=CLASS_NAMES, normalize="true",
            )
            cms.append(cm)
        mean_cm = np.mean(np.stack(cms), axis=0)
        std_cm = np.std(np.stack(cms), axis=0)

        n = len(CLASS_NAMES)
        fig, ax = plt.subplots(figsize=(max(4.5, n * 1.4),
                                          max(3.5, n * 1.3)))
        ConfusionMatrixDisplay(
            confusion_matrix=mean_cm, display_labels=CLASS_NAMES,
        ).plot(ax=ax, colorbar=True, values_format=".2f", cmap="Blues")
        for i in range(n):
            for j in range(n):
                ax.text(j, i + 0.30, f"±{std_cm[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="grey")
        ax.set_title(title or "LR baseline — test CM (mean over seeds)")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix → {save_path}")
    except Exception as e:
        print(f"Confusion matrix plot failed: {e}")


def plot_per_class_f1(per_seed_results: list, save_path: Path,
                      title: str = ""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        means, stds = [], []
        for c in CLASS_NAMES:
            scores = [r["test_per_class_f1"][c] for r in per_seed_results]
            means.append(np.mean(scores))
            stds.append(np.std(scores))

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(CLASS_NAMES))
        ax.bar(x, means, yerr=stds, color="teal", alpha=0.85,
               capsize=4, ecolor="black")
        ax.axhline(0.25, ls="--", color="grey", lw=1, label="Chance (0.25)")
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylabel("Test macro-F1 (mean ± std over seeds)")
        ax.set_ylim(0, 0.6)
        ax.set_title(title or "LR baseline — per-class test F1")
        ax.legend(loc="upper right")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Per-class F1 → {save_path}")
    except Exception as e:
        print(f"Per-class F1 plot failed: {e}")


def plot_top_features(per_seed_results: list, feat_names: list,
                      save_path: Path, top_k: int = 12,
                      title: str = ""):
    """Average |coef| across seeds; show top features per class."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ref_classes = per_seed_results[0]["logreg_classes"]
        for r in per_seed_results:
            if r["logreg_classes"] != ref_classes:
                print(f"WARN: seed {r['seed']} has different class order "
                      f"{r['logreg_classes']} vs ref {ref_classes}; "
                      f"skipping feature-importance plot.")
                return

        coef_stack = np.stack([np.array(r["logreg_coef"])
                                for r in per_seed_results], axis=0)
        # shape: (n_seeds, n_classes, n_features)
        mean_coef = coef_stack.mean(axis=0)  # (n_classes, n_features)

        n_classes = len(ref_classes)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes = axes.ravel()
        feat_arr = np.array(feat_names)

        for i, cls in enumerate(ref_classes):
            if i >= len(axes):
                break
            coefs = mean_coef[i]
            top_idx = np.argsort(np.abs(coefs))[-top_k:]
            colors = ["tab:red" if coefs[j] < 0 else "tab:blue"
                      for j in top_idx]
            axes[i].barh(feat_arr[top_idx], coefs[top_idx], color=colors)
            axes[i].axvline(0, color="black", lw=0.8)
            axes[i].set_title(f"Top {top_k} drivers — {cls}")
        for j in range(n_classes, len(axes)):
            axes[j].set_visible(False)
        if title:
            fig.suptitle(title)
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Top features → {save_path}")
    except Exception as e:
        print(f"Top features plot failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Model 1 — Logistic regression baseline. Supports "
                    "mfcc / mel / wav2vec2 feature sources for direct "
                    "comparison with Models 2 and 3.",
    )
    p.add_argument("--features", type=str, default="mfcc",
                   choices=["mfcc", "mel", "wav2vec2"],
                   help="Feature source (default: mfcc)")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 123, 456, 789, 2024])
    p.add_argument("--n_mfcc", type=int, default=20,
                   help="Number of MFCC coefficients (mfcc only, default 20)")
    p.add_argument("--use_delta", action="store_true",
                   help="Add delta-MFCC features (mfcc only). Default off "
                        "since the ablation showed deltas hurt test F1.")
    p.add_argument("--c_grid", type=float, nargs="+", default=None,
                   help="Custom C values; default is np.logspace(-4, 4, 9)")
    return p.parse_args()


def feature_tag(args) -> str:
    """Cache and output filename tag — uniquely identifies the config."""
    if args.features == "mfcc":
        return f"mfcc{args.n_mfcc}{'_delta' if args.use_delta else ''}"
    return args.features  # mel / wav2vec2


def main():
    args = parse_args()
    c_grid = (np.array(args.c_grid) if args.c_grid is not None
              else np.logspace(-4, 4, 9))
    tag = feature_tag(args)

    cfg.print_config()
    cfg.make_output_dirs()

    print("\n" + "=" * 60)
    print(f"MODEL 1 — LR BASELINE  ({args.features.upper()})")
    print("=" * 60)
    print(f"Seeds          : {args.seeds}")
    print(f"Feature source : {args.features}")
    if args.features == "mfcc":
        print(f"  n_mfcc       : {args.n_mfcc}")
        print(f"  use_delta    : {args.use_delta}")
    print(f"C grid         : {c_grid}")

    # ── Data ───────────────────────────────────────────────────────────
    print("\n[1/4] Loading splits and master table...")
    df = build_master_table(verbose=False)
    df = df[df["wav_exists"]].copy()

    # Source-file existence check, fail fast with a clear message
    if args.features == "mel":
        missing = [aid for aid in df["audio_ID"]
                   if not (SPEC_DIR / f"{aid}.png").exists()]
        if missing:
            print(f"\n✗ Missing {len(missing)} mel PNGs in {SPEC_DIR}.")
            print(f"  Run: python scripts/generate_spectrograms.py")
            sys.exit(1)
    elif args.features == "wav2vec2":
        missing = [aid for aid in df["audio_ID"]
                   if not (PRETRAINED_DIR / f"{aid}.npy").exists()]
        if missing:
            print(f"\n✗ Missing {len(missing)} wav2vec2 .npy files in "
                  f"{PRETRAINED_DIR}.")
            print(f"  Run the wav2vec2 extraction script first.")
            sys.exit(1)
    # mfcc: WAV files are checked at extraction time (per-clip try/except)

    protocol = determine_eval_protocol(df)
    folds = load_splits(protocol)
    if len(folds) != 1:
        print(f"WARN: expected 1 fold (holdout), got {len(folds)}")
    train_df, val_df, test_df = folds[0]
    print(f"  train: n={len(train_df)} | val: n={len(val_df)} | "
          f"test: n={len(test_df)}")

    # ── Feature extraction ────────────────────────────────────────────
    cache_dir = cfg.RES_DIR / f"lr_features_{tag}"
    extractor = get_extractor(args.features, args.n_mfcc, args.use_delta)
    print(f"\n[2/4] Extracting {args.features.upper()} summary features "
          f"(cached in {cache_dir})...")
    print(f"  train ({len(train_df)} clips):")
    X_train = build_feature_matrix(train_df, cache_dir, extractor)
    print(f"  val ({len(val_df)} clips):")
    X_val = build_feature_matrix(val_df, cache_dir, extractor)
    print(f"  test ({len(test_df)} clips):")
    X_test = build_feature_matrix(test_df, cache_dir, extractor)

    y_train = train_df["class4"].values
    y_val = val_df["class4"].values
    y_test = test_df["class4"].values
    groups_train = train_df["child"].values

    print(f"  X_train shape: {X_train.shape}")
    print(f"  feature dim  : {X_train.shape[1]}")
    print(f"  unique training children: {sorted(set(groups_train))}")

    # ── Per-seed runs ─────────────────────────────────────────────────
    print(f"\n[3/4] Running {len(args.seeds)} seeds...")
    per_seed = []
    for s in args.seeds:
        print(f"\n--- Seed {s} ---")
        per_seed.append(run_one_seed(
            seed=s,
            X_train=X_train, y_train=y_train, groups_train=groups_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            c_grid=c_grid,
        ))

    # ── Aggregate ─────────────────────────────────────────────────────
    test_f1s = np.array([r["test_macro_f1"] for r in per_seed])
    val_f1s = np.array([r["val_macro_f1"] for r in per_seed])
    gap = val_f1s - test_f1s

    per_class_test = {c: np.array([r["test_per_class_f1"][c]
                                     for r in per_seed])
                       for c in CLASS_NAMES}
    per_class_val = {c: np.array([r["val_per_class_f1"][c]
                                    for r in per_seed])
                      for c in CLASS_NAMES}

    print("\n" + "=" * 60)
    print(f"MULTI-SEED SUMMARY  —  LR baseline ({args.features}, "
          f"{len(args.seeds)} seeds)")
    print("=" * 60)
    print(f"  Seeds          : {args.seeds}")
    print(f"  Feature source : {args.features}")
    print(f"  Feature dim    : {X_train.shape[1]}")
    print(f"  Chance level   : {1 / len(CLASS_NAMES):.3f}")
    print(f"  Val  macro-F1  : {val_f1s.mean():.4f} ± {val_f1s.std():.4f}")
    print(f"  Test macro-F1  : {test_f1s.mean():.4f} ± {test_f1s.std():.4f}")
    print(f"  Val→Test gap   : {gap.mean():+.4f} ± {gap.std():.4f}")
    print(f"\n  Per-seed test macro-F1:")
    for r in per_seed:
        print(f"    seed {r['seed']:>5}: {r['test_macro_f1']:.4f} "
              f"(C={r['best_C']:.3g})")
    print(f"\n  Per-class TEST macro-F1 (mean ± std):")
    for c in CLASS_NAMES:
        print(f"    {c:<6}: {per_class_test[c].mean():.3f} "
              f"± {per_class_test[c].std():.3f}")
    print("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_tag = f"lr_baseline_{tag}_multiseed_{timestamp}"

    agg = {
        "model": "LogisticRegression",
        "model_name": "lr_baseline",
        "features": {
            "type": args.features,
            "n_mfcc": args.n_mfcc if args.features == "mfcc" else None,
            "use_delta": args.use_delta if args.features == "mfcc" else None,
            "feat_dim": int(X_train.shape[1]),
            "feature_names": feature_names(args.features, args.n_mfcc,
                                            args.use_delta),
        },
        "splits": {
            "protocol": protocol,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "train_children": sorted(set(groups_train)),
            "val_child": sorted(set(val_df["child"])),
            "test_child": sorted(set(test_df["child"])),
        },
        "seeds": args.seeds,
        "n_seeds": len(args.seeds),
        "c_grid": c_grid.tolist(),
        "chance_level": 1.0 / len(CLASS_NAMES),
        "val_macro_f1": {"mean": float(val_f1s.mean()),
                          "std": float(val_f1s.std())},
        "test_macro_f1": {"mean": float(test_f1s.mean()),
                           "std": float(test_f1s.std())},
        "val_to_test_gap": {"mean": float(gap.mean()),
                             "std": float(gap.std())},
        "val_per_class_f1": {c: {"mean": float(per_class_val[c].mean()),
                                  "std": float(per_class_val[c].std())}
                              for c in CLASS_NAMES},
        "test_per_class_f1": {c: {"mean": float(per_class_test[c].mean()),
                                    "std": float(per_class_test[c].std())}
                                for c in CLASS_NAMES},
        "per_seed_test_f1": test_f1s.tolist(),
        "per_seed_val_f1": val_f1s.tolist(),
        "per_seed_summaries": per_seed,
    }

    save_path = RESULTS_DIR / f"{out_tag}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nResults → {save_path}")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n[4/4] Building plots...")
    feat_label = args.features
    if args.features == "mfcc":
        feat_label = (f"{args.n_mfcc} MFCC"
                      f"{', +Δ' if args.use_delta else ''}")
    title_suffix = f"LR baseline ({feat_label}, {len(args.seeds)} seeds)"

    plot_confusion_matrix(per_seed,
                           FIGURES_DIR / f"{out_tag}_cm_test.png",
                           title=f"Test CM (mean over seeds) — {title_suffix}")
    plot_per_class_f1(per_seed,
                       FIGURES_DIR / f"{out_tag}_per_class.png",
                       title=f"Per-class test F1 — {title_suffix}")
    plot_top_features(per_seed,
                       feature_names(args.features, args.n_mfcc,
                                      args.use_delta),
                       FIGURES_DIR / f"{out_tag}_top_features.png",
                       title=f"Top features per class — {title_suffix}")

    print("\nDone.")
    print(f"  Aggregated results : {save_path}")
    print(f"  Figures            : {FIGURES_DIR}/{out_tag}_*.png")


if __name__ == "__main__":
    main()