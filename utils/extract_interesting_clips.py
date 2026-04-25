"""
utils/extract_interesting_clips.py
------------------------------------
Extracts audio clip filenames from the error analysis results for
presentation purposes — so you can play specific clips to the audience.

Three categories of clips are extracted:

  1. ALL-3-WRONG  — clips all three models get wrong (data ceiling cases)
                    Sorted by true class so you can pick one per class.

  2. ALL-3-CORRECT — clips all three models get right (the 7.1% the
                     models can actually classify). These are your
                     "success stories".

  3. DISAGREEMENT — clips where Model 1 (LR) is right but both
                    transformers are wrong. Shows that the linear
                    baseline sometimes has an advantage.

Output: printed tables + a CSV file at RES_DIR/results/interesting_clips_<timestamp>.csv
The CSV has columns: audio_id, wav_path, true_label, m1_pred, m2_pred,
m3_pred, n_wrong, category, child, lang

Usage
-----
    # Auto-detect most recent error_analysis JSON:
    python utils/extract_interesting_clips.py

    # Specify JSON explicitly:
    python utils/extract_interesting_clips.py \
        --json /home/renku/work/data/results/error_analysis_20260425_164221.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import config as cfg
from config import CLASS_NAMES, RESULTS_DIR, WAV_DIR


def latest_json(pattern: str) -> Path:
    matches = sorted(RESULTS_DIR.glob(pattern),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' in {RESULTS_DIR}")
    return matches[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=str, default=None,
                   help="Path to error_analysis JSON "
                        "(default: auto-detect latest)")
    p.add_argument("--n_per_class", type=int, default=3,
                   help="Number of clips to show per class (default: 3)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg.make_output_dirs()

    json_path = (Path(args.json) if args.json
                 else latest_json("error_analysis_*.json"))
    print(f"\nReading: {json_path.name}\n")

    with open(json_path) as f:
        data = json.load(f)

    clips = pd.DataFrame(data["per_clip"])

    # Add wav path
    clips["wav_path"] = clips["audio_id"].apply(
        lambda aid: str(WAV_DIR / f"{aid}.wav")
    )

    # Categorise each clip
    def categorise(row):
        nw = row["n_wrong"]
        if nw == 3:
            return "all_3_wrong"
        elif nw == 0:
            return "all_3_correct"
        elif not row["m1_wrong"] and row["m2_wrong"] and row["m3_wrong"]:
            return "lr_correct_transformers_wrong"
        elif row["m1_wrong"] and not row["m2_wrong"] and not row["m3_wrong"]:
            return "transformers_correct_lr_wrong"
        else:
            return "partial_disagreement"

    clips["category"] = clips.apply(categorise, axis=1)

    # ── Category 1: All 3 wrong ────────────────────────────────────────
    all3_wrong = clips[clips["category"] == "all_3_wrong"].copy()
    print("=" * 65)
    print(f"ALL-3-WRONG CLIPS  ({len(all3_wrong)} total)")
    print("Data ceiling — no model gets these right")
    print("=" * 65)
    for cls in CLASS_NAMES:
        sub = all3_wrong[all3_wrong["true_label"] == cls].head(args.n_per_class)
        if len(sub) == 0:
            continue
        print(f"\n  True class: {cls}  ({len(all3_wrong[all3_wrong['true_label']==cls])} total all-wrong)")
        for _, row in sub.iterrows():
            print(f"    {row['audio_id']}.wav")
            print(f"      child={row['child']}  lang={row['lang']}")
            print(f"      M1→{row['m1_pred']}  M2→{row['m2_pred']}  M3→{row['m3_pred']}")
            print(f"      path: {row['wav_path']}")

    # ── Category 2: All 3 correct ──────────────────────────────────────
    all3_correct = clips[clips["category"] == "all_3_correct"].copy()
    print(f"\n{'='*65}")
    print(f"ALL-3-CORRECT CLIPS  ({len(all3_correct)} total)")
    print("Success cases — all three models classify these correctly")
    print("=" * 65)
    for cls in CLASS_NAMES:
        sub = all3_correct[all3_correct["true_label"] == cls].head(args.n_per_class)
        if len(sub) == 0:
            continue
        print(f"\n  True class: {cls}  ({len(all3_correct[all3_correct['true_label']==cls])} total correct)")
        for _, row in sub.iterrows():
            print(f"    {row['audio_id']}.wav")
            print(f"      child={row['child']}  lang={row['lang']}")
            print(f"      path: {row['wav_path']}")

    # ── Category 3: LR right, transformers wrong ───────────────────────
    lr_wins = clips[clips["category"] == "lr_correct_transformers_wrong"].copy()
    print(f"\n{'='*65}")
    print(f"LR CORRECT, BOTH TRANSFORMERS WRONG  ({len(lr_wins)} clips)")
    print("Clips where the linear baseline has an advantage")
    print("=" * 65)
    for cls in CLASS_NAMES:
        sub = lr_wins[lr_wins["true_label"] == cls].head(args.n_per_class)
        if len(sub) == 0:
            continue
        print(f"\n  True class: {cls}")
        for _, row in sub.iterrows():
            print(f"    {row['audio_id']}.wav")
            print(f"      child={row['child']}  lang={row['lang']}")
            print(f"      M1→{row['m1_pred']}  M2→{row['m2_pred']}  M3→{row['m3_pred']}")
            print(f"      path: {row['wav_path']}")

    # ── Category 4: Transformers right, LR wrong ──────────────────────
    trans_wins = clips[clips["category"] == "transformers_correct_lr_wrong"].copy()
    print(f"\n{'='*65}")
    print(f"TRANSFORMERS CORRECT, LR WRONG  ({len(trans_wins)} clips)")
    print("Clips where temporal modeling has an advantage")
    print("=" * 65)
    for cls in CLASS_NAMES:
        sub = trans_wins[trans_wins["true_label"] == cls].head(args.n_per_class)
        if len(sub) == 0:
            continue
        print(f"\n  True class: {cls}")
        for _, row in sub.iterrows():
            print(f"    {row['audio_id']}.wav")
            print(f"      child={row['child']}  lang={row['lang']}")
            print(f"      M1→{row['m1_pred']}  M2→{row['m2_pred']}  M3→{row['m3_pred']}")
            print(f"      path: {row['wav_path']}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("CATEGORY SUMMARY")
    print("=" * 65)
    for cat, label in [
        ("all_3_wrong",                  "All 3 wrong (data ceiling)"),
        ("all_3_correct",                "All 3 correct (success)"),
        ("lr_correct_transformers_wrong","LR correct, transformers wrong"),
        ("transformers_correct_lr_wrong","Transformers correct, LR wrong"),
        ("partial_disagreement",         "Partial disagreement"),
    ]:
        n = (clips["category"] == cat).sum()
        print(f"  {label:<40}: {n:3d}  ({n/len(clips):.1%})")
    print(f"  {'Total':<40}: {len(clips)}")

    # ── Save CSV ──────────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"interesting_clips_{ts}.csv"
    out_cols = ["audio_id", "wav_path", "true_label",
                "m1_pred", "m2_pred", "m3_pred",
                "n_wrong", "category", "child", "lang"]
    clips[out_cols].sort_values(
        ["category", "true_label"]
    ).to_csv(csv_path, index=False)
    print(f"\nFull clip table → {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()