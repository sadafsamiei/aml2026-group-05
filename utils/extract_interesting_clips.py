"""
utils/extract_interesting_clips.py
------------------------------------
Extracts and categorises every test clip based on whether all three
models agree or disagree, and whether that agreement is correct.

Six agreement categories
-------------------------
  1. ALL-AGREE-CORRECT   (n_wrong=0, all 3 same correct prediction)
     All three models predict the right class. Best case.

  2. ALL-AGREE-WRONG     (n_wrong=3, all 3 same wrong prediction)
     All three models agree on the wrong class. Strongest data-ceiling
     signal — even consensus is wrong.

  3. ALL-WRONG-DISAGREE  (n_wrong=3, but predictions differ)
     All three models are wrong but disagree on which wrong class.
     Even harder — no model has any signal.

  4. LR-CORRECT-TRANSFORMERS-WRONG  (m1 right, m2+m3 wrong)
     Linear baseline has the advantage.

  5. TRANSFORMERS-CORRECT-LR-WRONG  (m2+m3 right, m1 wrong)
     Temporal modeling has the advantage.

  6. PARTIAL-DISAGREEMENT
     Everything else (1 wrong, 2 wrong in various combinations).

Within each category the clips are sorted by true class then by
confusion pair (true_label → consensus_pred or majority_pred) so you
can immediately see which confusions are systematic.

Outputs
-------
  Printed to terminal — full clip-by-clip breakdown per category
  RES_DIR/results/clip_agreement_<timestamp>.csv — full table
  RES_DIR/results/clip_agreement_<timestamp>_summary.txt — summary

Usage
-----
    python utils/extract_interesting_clips.py
    python utils/extract_interesting_clips.py --json /path/to/error_analysis.json
    python utils/extract_interesting_clips.py --n_per_class 5
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import config as cfg
from config import CLASS_NAMES, RESULTS_DIR, WAV_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_json(pattern: str) -> Path:
    matches = sorted(RESULTS_DIR.glob(pattern),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' in {RESULTS_DIR}")
    return matches[0]


def majority_pred(row) -> str:
    """Return the prediction agreed upon by ≥2 models, or 'no_majority'."""
    preds = [row["m1_pred"], row["m2_pred"], row["m3_pred"]]
    for p in set(preds):
        if preds.count(p) >= 2:
            return p
    return "no_majority"


def categorise(row) -> str:
    nw = int(row["n_wrong"])
    if nw == 0:
        return "all_agree_correct"
    if nw == 3:
        preds = [row["m1_pred"], row["m2_pred"], row["m3_pred"]]
        if len(set(preds)) == 1:
            return "all_agree_wrong"
        else:
            return "all_wrong_disagree"
    if not row["m1_wrong"] and row["m2_wrong"] and row["m3_wrong"]:
        return "lr_correct_transformers_wrong"
    if row["m1_wrong"] and not row["m2_wrong"] and not row["m3_wrong"]:
        return "transformers_correct_lr_wrong"
    return "partial_disagreement"


def print_clips(df: pd.DataFrame, n_per_class: int,
                show_preds: bool = True):
    """Print up to n_per_class clips per true label."""
    for cls in CLASS_NAMES:
        sub = df[df["true_label"] == cls]
        if len(sub) == 0:
            continue
        print(f"\n  True class: {cls}  ({len(sub)} clips)")

        # Show most common confusion pair first
        if show_preds and "majority_pred" in df.columns:
            pairs = (sub.groupby(["true_label", "majority_pred"])
                     .size().reset_index(name="n")
                     .sort_values("n", ascending=False))
            for _, pr in pairs.iterrows():
                if pr["majority_pred"] != "no_majority":
                    print(f"    Most common: {cls} → {pr['majority_pred']}  "
                          f"(n={pr['n']})")

        for _, row in sub.head(n_per_class).iterrows():
            print(f"    {row['audio_id']}.wav  "
                  f"child={row['child']}  lang={row['lang']}")
            if show_preds:
                print(f"      M1→{row['m1_pred']}  "
                      f"M2→{row['m2_pred']}  "
                      f"M3→{row['m3_pred']}")
            print(f"      {row['wav_path']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract and categorise clips by model agreement.")
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--n_per_class", type=int, default=3,
                   help="Clips to print per true class (default 3)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg.make_output_dirs()

    json_path = (Path(args.json) if args.json
                 else latest_json("error_analysis_*.json"))
    print(f"\nReading: {json_path.name}")

    with open(json_path) as f:
        data = json.load(f)

    clips = pd.DataFrame(data["per_clip"])
    clips["wav_path"]      = clips["audio_id"].apply(
        lambda aid: str(WAV_DIR / f"{aid}.wav"))
    clips["majority_pred"] = clips.apply(majority_pred, axis=1)
    clips["category"]      = clips.apply(categorise, axis=1)

    n = len(clips)

    # ── Category definitions ─────────────────────────────────────────
    cats = [
        ("all_agree_correct",            "ALL AGREE — CORRECT",
         "All 3 models predict the right class"),
        ("all_agree_wrong",              "ALL AGREE — WRONG",
         "All 3 models agree on the same wrong class (strongest data ceiling)"),
        ("all_wrong_disagree",           "ALL WRONG — DISAGREE",
         "All 3 models wrong and disagree on which class"),
        ("lr_correct_transformers_wrong","LR CORRECT — TRANSFORMERS WRONG",
         "Linear baseline wins; both transformers wrong"),
        ("transformers_correct_lr_wrong","TRANSFORMERS CORRECT — LR WRONG",
         "Both transformers win; linear baseline wrong"),
        ("partial_disagreement",         "PARTIAL DISAGREEMENT",
         "Mixed results (1 or 2 models wrong, various combinations)"),
    ]

    # ── Print each category ───────────────────────────────────────────
    for cat_key, cat_title, cat_desc in cats:
        sub = clips[clips["category"] == cat_key].copy()
        pct = len(sub) / n * 100
        print(f"\n{'='*65}")
        print(f"{cat_title}  ({len(sub)} clips  {pct:.1f}%)")
        print(f"{cat_desc}")
        print("=" * 65)

        if len(sub) == 0:
            print("  (none)")
            continue

        show = cat_key != "all_agree_correct"  # correct clips need no pred
        print_clips(sub, args.n_per_class, show_preds=show)

    # ── Confusion pair summary ────────────────────────────────────────
    print(f"\n{'='*65}")
    print("CONFUSION PAIRS — ALL-AGREE-WRONG clips")
    print("(true class → agreed wrong prediction, sorted by frequency)")
    print("=" * 65)
    agree_wrong = clips[clips["category"] == "all_agree_wrong"]
    if len(agree_wrong) > 0:
        pairs = (agree_wrong
                 .groupby(["true_label", "majority_pred"])
                 .size().reset_index(name="n")
                 .sort_values("n", ascending=False))
        for _, row in pairs.iterrows():
            print(f"  {row['true_label']:<6} → {row['majority_pred']:<6}  "
                  f"n={row['n']}")
    else:
        print("  (no all-agree-wrong clips)")

    # ── Global summary ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY")
    print("=" * 65)
    print(f"  Total test clips : {n}")
    print()
    for cat_key, cat_title, _ in cats:
        cnt = (clips["category"] == cat_key).sum()
        print(f"  {cat_title:<40}: {cnt:3d}  ({cnt/n:.1%})")

    print(f"\n  Error rates per model:")
    for name, col in [("M1 (LR mel)", "m1_wrong"),
                       ("M2 (mean pool)", "m2_wrong"),
                       ("M3 (attention)", "m3_wrong")]:
        w = clips[col].sum()
        print(f"    {name:<22}: {w:3d} wrong  ({w/n:.1%})")

    print(f"\n  Presentation picks:")
    print(f"    All-agree-wrong PLTOY : audio_0445.wav  "
          f"(true=PLTOY, all predict INTER)")
    print(f"    All-agree-correct INTER: audio_0257.wav  "
          f"(all correct)")
    print(f"    LR-wins EXTEV         : audio_0793.wav  "
          f"(M1=EXTEV correct, M2/M3=INTER)")
    print(f"    Transformers-win INTER: audio_0266.wav  "
          f"(M1=MOVE wrong, M2/M3=INTER correct)")

    # ── Save CSV ──────────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_cols = ["audio_id", "wav_path", "true_label", "majority_pred",
                "m1_pred", "m2_pred", "m3_pred",
                "n_wrong", "category", "child", "lang"]
    csv_path = RESULTS_DIR / f"clip_agreement_{ts}.csv"
    clips[out_cols].sort_values(
        ["category", "true_label", "audio_id"]
    ).to_csv(csv_path, index=False)
    print(f"\nFull table → {csv_path}")

    # ── Save text summary ─────────────────────────────────────────────
    txt_path = RESULTS_DIR / f"clip_agreement_{ts}_summary.txt"
    lines = [f"Clip agreement analysis — {json_path.name}\n",
             f"Total test clips: {n}\n\n"]
    for cat_key, cat_title, cat_desc in cats:
        cnt = (clips["category"] == cat_key).sum()
        lines.append(f"{cat_title}  ({cnt} clips  {cnt/n:.1%})\n")
        lines.append(f"  {cat_desc}\n\n")
    with open(txt_path, "w") as f:
        f.writelines(lines)
    print(f"Text summary → {txt_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()