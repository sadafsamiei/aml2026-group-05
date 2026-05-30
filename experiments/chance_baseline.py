"""
experiments/chance_baseline.py
--------------------------------
Formal chance-level baselines using sklearn's DummyClassifier on the
project's canonical holdout split.

"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

import config as cfg
from config import CLASS_NAMES, RESULTS_DIR
from data_utils import build_master_table, determine_eval_protocol, load_splits

N_SEEDS = 1000   


def run_dummy(strategy: str, y_train: np.ndarray,
              y_test: np.ndarray, n_seeds: int = N_SEEDS) -> dict:
    deterministic = strategy in ("most_frequent", "prior")
    seeds = [0] if deterministic else range(n_seeds)

    f1s = []
    per_class_runs = {c: [] for c in CLASS_NAMES}

    for seed in seeds:
        clf = DummyClassifier(strategy=strategy, random_state=seed)
        clf.fit(
            np.zeros((len(y_train), 1)), y_train  
        )
        preds = clf.predict(np.zeros((len(y_test), 1)))

        macro = f1_score(y_test, preds, average="macro",
                          labels=CLASS_NAMES, zero_division=0)
        per   = f1_score(y_test, preds, average=None,
                          labels=CLASS_NAMES, zero_division=0)
        f1s.append(macro)
        for c, v in zip(CLASS_NAMES, per):
            per_class_runs[c].append(v)

    return {
        "strategy":         strategy,
        "macro_f1_mean":    float(np.mean(f1s)),
        "macro_f1_std":     float(np.std(f1s)),
        "n_seeds":          len(seeds),
        "per_class_f1": {
            c: {"mean": float(np.mean(per_class_runs[c])),
                "std":  float(np.std(per_class_runs[c]))}
            for c in CLASS_NAMES
        },
    }


def main():
    cfg.print_config()
    cfg.make_output_dirs()

    print("\n" + "=" * 60)
    print("FORMAL CHANCE-LEVEL BASELINE  (DummyClassifier)")
    print("=" * 60)

    df       = build_master_table(verbose=False)
    protocol = determine_eval_protocol(df)
    folds    = load_splits(protocol)
    train_df, val_df, test_df = folds[0]

    y_train = train_df["class4"].values
    y_val   = val_df["class4"].values
    y_test  = test_df["class4"].values

    print(f"\n  Train : n={len(y_train)}")
    print(f"  Val   : n={len(y_val)}")
    print(f"  Test  : n={len(y_test)}")
    print(f"\n  Test class distribution:")
    for c in CLASS_NAMES:
        n = (y_test == c).sum()
        print(f"    {c:<6}: {n:3d}  ({n/len(y_test):.1%})")

    print(f"\n  Running DummyClassifier (stratified/uniform: {N_SEEDS} seeds)...")

    strategies = ["most_frequent", "stratified", "uniform"]
    results    = []
    for s in strategies:
        r = run_dummy(s, y_train, y_test)
        results.append(r)

    print(f"\n{'='*60}")
    print(f"CHANCE BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Strategy':<14} Macro-F1         Per-class F1")
    print(f"  {'-'*56}")
    for r in results:
        m   = r["macro_f1_mean"]
        s   = r["macro_f1_std"]
        row = f"  {r['strategy']:<14} {m:.4f} ± {s:.4f}   "
        row += "  ".join(
            f"{c}:{r['per_class_f1'][c]['mean']:.3f}"
            for c in CLASS_NAMES
        )
        print(row)

    print(f"\n  Theoretical chance (uniform, 4 classes) : 0.2500")
    print(f"\n  Our best result (LR mel, fixed holdout) : 0.2770")
    print(f"  Δ vs uniform chance                     : +0.027")
    print(f"{'='*60}")

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"chance_baseline_{ts}.json"
    out  = {
        "experiment": "Formal chance-level baseline",
        "protocol":   protocol,
        "n_train":    int(len(y_train)),
        "n_val":      int(len(y_val)),
        "n_test":     int(len(y_test)),
        "n_seeds_stochastic": N_SEEDS,
        "test_class_counts": {
            c: int((y_test == c).sum()) for c in CLASS_NAMES
        },
        "strategies": results,
        "our_best_f1": 0.277,
        "delta_vs_uniform": 0.277 - next(
            r["macro_f1_mean"] for r in results
            if r["strategy"] == "uniform"
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults → {path}")
    print("Done.")


if __name__ == "__main__":
    main()
