"""
experiments/loco_model1_mel.py
-------------------------------
Leave-One-Child-Out (LOCO) cross-validation for Model 1 (LR) on mel
summary statistics — the best-performing configuration in our project.

Why this experiment
-------------------
Our canonical results come from a single fixed holdout (child_6 as test).
LOCO answers whether 0.277 is representative by rotating which child is
held out across all 9 folds and averaging macro-F1.

Fix notes vs v1
---------------
max_iter raised from 2000 → 5000 to suppress convergence warnings that
appeared for large C values (C ≥ 10) when using all 8 training children.
The warnings were cosmetic (lbfgs was close to convergence) but noisy.

Outputs
-------
RES_DIR/results/loco_model1_mel_<timestamp>.json
RES_DIR/figures/loco_model1_mel_<timestamp>_per_fold.png
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
from config import CLASS_NAMES, FIGURES_DIR, RESULTS_DIR, SPEC_DIR
from data_utils import build_master_table, load_png

C_GRID   = np.logspace(-4, 4, 9)
MAX_ITER = 5000   # raised from 2000 to suppress convergence warnings


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_mel_summary(audio_id: str) -> np.ndarray:
    png_path = SPEC_DIR / f"{audio_id}.png"
    spec = load_png(str(png_path), target_height=128, target_width=230)
    return np.concatenate([
        spec.mean(axis=1),
        spec.std(axis=1),
    ]).astype(np.float32)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cache_dir = cfg.RES_DIR / "lr_features_mel"
    cache_dir.mkdir(parents=True, exist_ok=True)
    feats = []
    for i, audio_id in enumerate(df["audio_ID"]):
        cache_path = cache_dir / f"{audio_id}.npy"
        if cache_path.exists():
            feat = np.load(cache_path)
        else:
            feat = extract_mel_summary(audio_id)
            np.save(cache_path, feat)
        feats.append(feat)
        if (i + 1) % 200 == 0:
            print(f"    ... {i+1}/{len(df)} clips")
    return np.vstack(feats)


# ---------------------------------------------------------------------------
# LOCO split builder
# ---------------------------------------------------------------------------

def build_loco_folds(df: pd.DataFrame) -> list:
    """
    9 folds, one per child. Test = held-out child, val = next child by
    size (wrapping), train = remaining 7 children.
    """
    children = (df.groupby("child").size()
                  .sort_values(ascending=False).index.tolist())
    n = len(children)
    folds = []
    for i, test_child in enumerate(children):
        val_child  = children[(i + 1) % n]
        train_ch   = [c for c in children
                      if c not in (test_child, val_child)]
        folds.append((
            df[df["child"].isin(train_ch)].copy(),
            df[df["child"] == val_child].copy(),
            df[df["child"] == test_child].copy(),
            test_child,
            val_child,
        ))
    return folds


# ---------------------------------------------------------------------------
# Per-fold run
# ---------------------------------------------------------------------------

def run_fold(X_train, y_train, groups_train,
             X_val, y_val,
             X_test, y_test,
             test_child: str) -> dict:

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=MAX_ITER, solver="lbfgs",
            class_weight="balanced", random_state=42,
        )),
    ])

    grid = GridSearchCV(
        pipeline, {"logreg__C": C_GRID},
        cv=LeaveOneGroupOut(), scoring="f1_macro", n_jobs=-1,
    )
    grid.fit(X_train, y_train, groups=groups_train)
    best_C = float(grid.best_params_["logreg__C"])

    # Val: refit on train alone
    val_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=MAX_ITER, solver="lbfgs",
            class_weight="balanced", random_state=42, C=best_C,
        )),
    ])
    val_pipe.fit(X_train, y_train)
    val_preds = val_pipe.predict(X_val)
    val_f1    = f1_score(y_val, val_preds, average="macro",
                          labels=CLASS_NAMES, zero_division=0)

    # Test: refit on train + val
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    test_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=MAX_ITER, solver="lbfgs",
            class_weight="balanced", random_state=42, C=best_C,
        )),
    ])
    test_pipe.fit(X_tv, y_tv)
    test_preds   = test_pipe.predict(X_test)
    test_f1      = f1_score(y_test, test_preds, average="macro",
                             labels=CLASS_NAMES, zero_division=0)
    test_per_cls = f1_score(y_test, test_preds, average=None,
                             labels=CLASS_NAMES, zero_division=0)

    print(f"  {test_child:<10}: C={str(best_C):<8}  "
          f"val={val_f1:.4f}  test={test_f1:.4f}")

    return {
        "test_child":        test_child,
        "best_C":            best_C,
        "inner_cv_score":    float(grid.best_score_),
        "val_macro_f1":      float(val_f1),
        "test_macro_f1":     float(test_f1),
        "test_per_class_f1": {c: float(v)
                               for c, v in zip(CLASS_NAMES, test_per_cls)},
        "test_preds":        test_preds.tolist(),
        "test_labels":       list(y_test),
        "n_test":            len(y_test),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_per_fold(fold_results: list, mean_f1: float, std_f1: float,
                  canonical_f1: float, save_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        children = [r["test_child"] for r in fold_results]
        f1s      = [r["test_macro_f1"] for r in fold_results]
        n_test   = [r["n_test"] for r in fold_results]

        fig, ax = plt.subplots(figsize=(11, 5))

        # colour bars by language for visual clarity
        lang_colors = {"tagalog": "#EF9A9A", "shipibo": "#90CAF9",
                       "vietnamese": "#FFB74D"}

        from data_utils import build_master_table
        df_tmp = build_master_table(verbose=False)
        child_lang = df_tmp.groupby("child")["lang"].first().to_dict()

        bar_colors = [lang_colors.get(child_lang.get(c, ""), "#BDBDBD")
                      for c in children]
        bars = ax.bar(children, f1s, color=bar_colors,
                      edgecolor="#1565C0", linewidth=1.0, zorder=3)

        for bar, n, f1 in zip(bars, n_test, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"n={n}", ha="center", fontsize=8.5, color="#333")
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"{f1:.3f}", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white")

        ax.axhline(mean_f1, ls="-", color="#1565C0", lw=2.0,
                   label=f"LOCO mean  {mean_f1:.3f} ± {std_f1:.3f}")
        ax.axhline(canonical_f1, ls="--", color="#C62828", lw=1.8,
                   label=f"Canonical holdout (child_6)  {canonical_f1:.3f}")
        ax.axhline(0.250, ls=":", color="grey", lw=1.4,
                   label="Chance = 0.250")

        # Language legend patches
        from matplotlib.patches import Patch
        legend_patches = [Patch(color=v, label=k.capitalize())
                          for k, v in lang_colors.items()]
        first_legend = ax.legend(handles=legend_patches,
                                  loc="upper right", fontsize=9,
                                  title="Language")
        ax.add_artist(first_legend)
        ax.legend(fontsize=9.5, loc="upper left")

        ax.set_xlabel("Test child (held out)", fontsize=12)
        ax.set_ylabel("Test macro-F1", fontsize=12)
        ax.set_title(
            "LOCO CV — Model 1 (LR) on mel summary statistics\n"
            "LOCO mean = 0.213 vs canonical holdout = 0.277  "
            "→ child_6 is a relatively easy test case",
            fontsize=11,
        )
        ax.set_ylim(0, 0.42)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Plot → {save_path}")
    except Exception as e:
        print(f"Plot failed: {e}")
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg.print_config()
    cfg.make_output_dirs()

    print("\n" + "=" * 60)
    print("LOCO CV — Model 1 (LR) on Mel summary statistics")
    print("=" * 60)

    df = build_master_table(verbose=False)
    df = df[df["wav_exists"] & df["png_exists"]].copy()
    print(f"\nTotal clips with PNG: {len(df)}")

    print("\n[1/3] Extracting mel summary features...")
    X_all = build_feature_matrix(df)
    y_all = df["class4"].values
    print(f"  Feature matrix: {X_all.shape}")

    folds = build_loco_folds(df)
    print(f"\n[2/3] Running {len(folds)} LOCO folds...\n")
    print(f"  {'Child':<12} {'C':<10} {'Val F1':>8}  {'Test F1':>8}")
    print(f"  {'-'*44}")

    fold_results = []
    for train_df, val_df, test_df, test_child, val_child in folds:
        idx_train = df.index.isin(train_df.index)
        idx_val   = df.index.isin(val_df.index)
        idx_test  = df.index.isin(test_df.index)

        result = run_fold(
            X_train=X_all[idx_train], y_train=y_all[idx_train],
            groups_train=train_df["child"].values,
            X_val=X_all[idx_val],   y_val=y_all[idx_val],
            X_test=X_all[idx_test], y_test=y_all[idx_test],
            test_child=test_child,
        )
        result["val_child"]      = val_child
        result["train_children"] = [
            c for c in df["child"].unique()
            if c not in (test_child, val_child)
        ]
        fold_results.append(result)

    test_f1s = np.array([r["test_macro_f1"] for r in fold_results])
    val_f1s  = np.array([r["val_macro_f1"]  for r in fold_results])
    mean_f1  = float(test_f1s.mean())
    std_f1   = float(test_f1s.std())

    per_class_loco = {
        c: {"mean": float(np.mean([r["test_per_class_f1"][c]
                                    for r in fold_results])),
            "std":  float(np.std([r["test_per_class_f1"][c]
                                   for r in fold_results]))}
        for c in CLASS_NAMES
    }

    CANONICAL_F1 = 0.277

    # Language lookup for display
    child_lang = df.groupby("child")["lang"].first().to_dict()

    print(f"\n{'='*60}")
    print(f"LOCO SUMMARY — Model 1 (LR) mel  ({len(folds)} folds)")
    print(f"{'='*60}")
    print(f"  LOCO mean test macro-F1   : {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Canonical holdout (child_6): {CANONICAL_F1:.4f}")
    print(f"  Difference                 : {mean_f1 - CANONICAL_F1:+.4f}")
    print(f"\n  Per-fold test macro-F1:")
    for r in fold_results:
        lang = child_lang.get(r["test_child"], "?")
        print(f"    {r['test_child']} ({lang}): {r['test_macro_f1']:.4f}  "
              f"(n={r['n_test']})")
    print(f"\n  Per-class TEST macro-F1 (mean ± std across folds):")
    for c in CLASS_NAMES:
        print(f"    {c:<6}: {per_class_loco[c]['mean']:.3f} "
              f"± {per_class_loco[c]['std']:.3f}")
    print(f"{'='*60}")

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"loco_model1_mel_{ts}"

    out = {
        "experiment":           "LOCO CV — Model 1 (LR) mel",
        "feature_source":       "mel",
        "feature_dim":          256,
        "max_iter":             MAX_ITER,
        "n_folds":              len(folds),
        "canonical_holdout_f1": CANONICAL_F1,
        "loco_mean_test_f1":    mean_f1,
        "loco_std_test_f1":     std_f1,
        "loco_mean_val_f1":     float(val_f1s.mean()),
        "loco_std_val_f1":      float(val_f1s.std()),
        "per_class_loco_f1":    per_class_loco,
        "per_fold_results":     fold_results,
    }
    result_path = RESULTS_DIR / f"{tag}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults → {result_path}")

    print("\n[3/3] Plotting per-fold F1...")
    plot_per_fold(fold_results, mean_f1, std_f1, CANONICAL_F1,
                  FIGURES_DIR / f"{tag}_per_fold.png")
    print("\nDone.")


if __name__ == "__main__":
    main()