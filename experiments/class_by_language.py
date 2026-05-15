"""
scripts/class_by_language.py
----------------------------
Visualise the class distribution across the three languages in the
training split. Makes visible the language-class confound that underlies
our reported Model 3 INTER-prediction bias.

Key numbers (from the current train.csv, 569 clips, 7 children):
  - Tagalog:    children 4, 5   (145 clips)  INTER = 46.9%  PLTOY = 9.7%
  - Shipibo:    children 1, 3   (142 clips)  INTER = 22.5%  PLTOY = 28.2%
  - Vietnamese: children 7, 8, 9 (282 clips) INTER = 33.0%  PLTOY = 16.7%

"""

import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CLASS_NAMES, FIGURES_DIR, SPLITS_DIR


def main():
    train = pd.read_csv(SPLITS_DIR / "train.csv")
    val   = pd.read_csv(SPLITS_DIR / "val.csv")
    test  = pd.read_csv(SPLITS_DIR / "test.csv")

    print("Split summary:")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name}: n={len(df)}, children={df['child'].nunique()}, "
              f"langs={sorted(df['lang'].unique())}")

    ct_col = pd.crosstab(train["class4"], train["lang"], normalize="columns")
    ct_col = ct_col.reindex(CLASS_NAMES)
    lang_order = ["shipibo", "tagalog", "vietnamese"]
    ct_col = ct_col.reindex(columns=lang_order)
    print("\nTrain column-normalised (within each language):")
    print((ct_col * 100).round(1))

    ct_raw = pd.crosstab(train["class4"], train["lang"])
    ct_raw = ct_raw.reindex(CLASS_NAMES).reindex(columns=lang_order)

    lang_totals = {l: int(train[train["lang"] == l].shape[0]) for l in lang_order}

    test_lang = test["lang"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(CLASS_NAMES))
    n_lang = len(lang_order)
    width = 0.25
    colors = {"shipibo": "#4C72B0", "tagalog": "#DD8452", "vietnamese": "#55A868"}

    for i, lang in enumerate(lang_order):
        offset = (i - (n_lang - 1) / 2) * width
        values = ct_col[lang].values * 100
        bars = ax.bar(x + offset, values, width,
                       label=f"{lang} (n={lang_totals[lang]})",
                       color=colors[lang], edgecolor="black", linewidth=0.6)
        for j, (bar, count) in enumerate(zip(bars, ct_raw[lang].values)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f"{values[j]:.0f}%\n(n={count})",
                    ha="center", va="bottom", fontsize=8.5,
                    color="black")

    ax.axhline(25.0, ls="--", color="grey", lw=1.2, alpha=0.7,
               label="Uniform 4-class (25%)")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylabel("Percentage of each language's training clips", fontsize=11)
    ax.set_title(
        "Training-data class distribution by language\n"
        f"Each child is monolingual → language is confounded with child identity. "
        f"Test child speaks {test_lang}.",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 55)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = FIGURES_DIR / f"class_by_language_{timestamp}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure → {out_path}")

    print(f"\n{'='*62}")
    print("KEY CONFOUND NUMBERS (train set)")
    print(f"{'='*62}")
    for lang in lang_order:
        inter_pct = ct_col.loc["INTER", lang] * 100
        pltoy_pct = ct_col.loc["PLTOY", lang] * 100
        print(f"  {lang:<12}: INTER={inter_pct:.1f}%, PLTOY={pltoy_pct:.1f}%")
    tag_inter = ct_col.loc["INTER", "tagalog"] * 100
    shi_inter = ct_col.loc["INTER", "shipibo"] * 100
    print(f"\n  Tagalog INTER / Shipibo INTER ratio: {tag_inter / shi_inter:.2f}x")
    print(f"  Test set child_6 is Tagalog — expect an INTER prior push.")


if __name__ == "__main__":
    main()