"""
data_utils.py  [FIXED v2]
-------------
Master table construction, spectrogram audit, split construction,
and PyTorch Dataset class.

Key fix over v1 and v2
-----------------------
The spectrogram PNGs are named spectrogram1.png, spectrogram2.png, ...
— sequential integers with no connection to the audio filenames in the CSV.
The mapping is purely positional:

    spectrogram1.png  →  row 1 of the language-filtered CSV
    spectrogram2.png  →  row 2 of the language-filtered CSV
    ...

build_png_lookup() now:
  1. Deduplicates the PNG list (fixes the double-glob bug that reported
     102 files instead of the real 51 unique files)
  2. Sorts PNGs numerically by the integer in the filename
  3. Assigns each PNG to the corresponding CSV row by position
"""

import collections
import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold

from config import (
    CONTEXTS_CSV, SPEC_DIR, LANG_FILTER,
    MOVE_LABELS, INTER_LABELS, EXTEV_LABELS, PLTOY_LABELS,
    CLASS_NAMES, LABEL_ENCODER, N_CLASSES,
    SPLITS_DIR, RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def map_label(lbl: str):
    if lbl in PLTOY_LABELS:  return "PLTOY"
    if lbl in MOVE_LABELS:   return "MOVE"
    if lbl in INTER_LABELS:  return "INTER"
    if lbl in EXTEV_LABELS:  return "EXTEV"
    return None


# ---------------------------------------------------------------------------
# PNG lookup — positional mapping (the real fix)
# ---------------------------------------------------------------------------

def _get_unique_pngs_sorted(spec_dir: Path) -> list:
    """
    Return a deduplicated, numerically sorted list of PNG Paths from spec_dir.

    Uses rglob instead of glob+glob to avoid duplicates.
    Sorts by the integer embedded in the filename (spectrogram1, spectrogram10, ...)
    so that spectrogram2 comes before spectrogram10 (not after, as in string sort).
    """
    # rglob finds all PNGs recursively without duplicating top-level files
    all_pngs = list(spec_dir.rglob("*.png"))

    # Deduplicate by resolved absolute path (handles any remaining duplicates)
    seen  = set()
    unique = []
    for p in all_pngs:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    # Sort numerically: extract the integer from names like "spectrogram42.png"
    def _sort_key(p):
        nums = re.findall(r'\d+', p.stem)
        return int(nums[0]) if nums else 0

    unique.sort(key=_sort_key)
    return unique


def build_png_lookup(spec_dir: Path, lang_filtered_df: pd.DataFrame,
                     verbose: bool = True) -> pd.Series:
    """
    Build a Series mapping audio_file → png_path using positional assignment.

    The spectrograms are named spectrogram1.png, spectrogram2.png, ...
    and correspond to the rows of lang_filtered_df in their original order.

    Parameters
    ----------
    spec_dir         : Path to the spectrograms directory
    lang_filtered_df : DataFrame already filtered to the target language,
                       in the same row order the spectrograms were generated from.
                       (Do NOT pass the label-filtered df here — the PNG indices
                       must align with the pre-label-filter row order.)
    verbose          : print mapping summary

    Returns
    -------
    pd.Series indexed by audio_file, values are png_path strings (or None)
    """
    if not spec_dir.exists():
        raise FileNotFoundError(
            f"Spectrograms directory not found: {spec_dir}\n"
            f"Check DATA_DIR in config.py."
        )

    pngs = _get_unique_pngs_sorted(spec_dir)
    n_png = len(pngs)
    n_csv = len(lang_filtered_df)

    if verbose:
        print(f"\n--- PNG positional mapping ---")
        print(f"  Unique PNGs (numerically sorted): {n_png}")
        print(f"  CSV rows (after language filter) : {n_csv}")
        if n_png > 0:
            print(f"  First PNG : {pngs[0].name}")
            print(f"  Last PNG  : {pngs[-1].name}")

    if n_png == 0:
        raise FileNotFoundError(
            f"No PNG files found in {spec_dir}. "
            f"Check that your spectrograms are saved there."
        )

    if n_png > n_csv:
        print(f"  ⚠  More PNGs ({n_png}) than CSV rows ({n_csv}). "
              f"Using first {n_csv} PNGs.")
        pngs = pngs[:n_csv]

    if n_png < n_csv:
        print(f"  ⚠  Fewer PNGs ({n_png}) than CSV rows ({n_csv}). "
              f"Only the first {n_png} rows will have a spectrogram.")

    # Build the mapping
    audio_files = lang_filtered_df["audio_file"].tolist()
    mapping = {}
    for i, audio_file in enumerate(audio_files):
        if i < len(pngs):
            mapping[audio_file] = str(pngs[i])
        else:
            mapping[audio_file] = None   # no PNG for this row

    if verbose:
        n_mapped = sum(v is not None for v in mapping.values())
        print(f"  Mapped     : {n_mapped}/{n_csv} rows have a PNG")
        # Show first 3 mappings for verification
        print(f"\n  Mapping preview (first 3):")
        for af, pp in list(mapping.items())[:3]:
            png_name = Path(pp).name if pp else "NONE"
            print(f"    {png_name:20s}  ←  {Path(af).name}")

    return mapping


# ---------------------------------------------------------------------------
# Master table
# ---------------------------------------------------------------------------

def build_master_table(
    contexts_csv: Path = CONTEXTS_CSV,
    spec_dir: Path = SPEC_DIR,
    lang_filter: str = LANG_FILTER,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the contexts CSV, apply language and label filters, assign PNG paths
    by position, and return a clean DataFrame.

    The positional assignment happens on the LANGUAGE-filtered rows
    (before label filtering), because the spectrograms were generated
    from all language-filtered rows in their original CSV order.
    """
    df = pd.read_csv(contexts_csv)

    # ── Step 1: Language filter (positional mapping is relative to this) ──
    if lang_filter:
        before = len(df)
        df = df[df["lang"] == lang_filter].copy()
        if verbose:
            print(f"Language filter '{lang_filter}': {before} → {len(df)} rows")
    else:
        df = df.copy()

    # ── Step 2: PNG mapping (must happen before label filter) ─────────────
    png_mapping = build_png_lookup(spec_dir, df, verbose=verbose)
    df["png_path"]  = df["audio_file"].map(png_mapping)
    df["png_exists"] = df["png_path"].notna()

    # ── Step 3: Label mapping ─────────────────────────────────────────────
    df["class4"] = df["context_label"].apply(map_label)

    excluded = df[df["class4"].isna()]["context_label"].value_counts()
    if verbose and len(excluded):
        print(f"\nExcluded labels (outside 4-class scheme):")
        for lbl, cnt in excluded.items():
            print(f"  {lbl:12s}: {cnt}")

    df = df[df["class4"].notna()].reset_index(drop=True)

    # ── Step 4: Duration ──────────────────────────────────────────────────
    df["dur_ms"] = df["end_ms"] - df["start_ms"]

    # ── Step 5: Summary ───────────────────────────────────────────────────
    if verbose:
        n_found   = df["png_exists"].sum()
        n_missing = (~df["png_exists"]).sum()
        print(f"\n{'='*50}")
        print(f"Usable examples (after label filter) : {len(df)}")
        print(f"  → with PNG                         : {n_found}")
        print(f"  → without PNG                      : {n_missing}")
        if n_missing > 0:
            print(f"\n  ⚠  {n_missing} usable rows have no spectrogram yet.")
            print(f"     They will be excluded from training.")
            print(f"     (Expected: more PNGs arrive with the 1500+ expansion)")
        print(f"\nClass distribution (all usable):")
        print(df["class4"].value_counts().to_string())
        print(f"\nClass distribution (with PNG only):")
        print(df[df["png_exists"]]["class4"].value_counts().to_string())
        print(f"\nParticipants : {df['child'].nunique()}")
        print(f"Per child:\n{df.groupby('child').size().to_string()}")
        print(f"\nClass × Child:\n{pd.crosstab(df['child'], df['class4']).to_string()}")
        print(f"{'='*50}")

    return df


# ---------------------------------------------------------------------------
# Spectrogram audit
# ---------------------------------------------------------------------------

def audit_spectrograms(png_paths: list, verbose: bool = True) -> dict:
    """
    Scan PNGs, report height (F), width (T) distribution, mode, pixel range.
    Returns a dict with f_dim and max_len_p95.
    """
    heights, widths, modes, max_vals = [], [], [], []

    for p in png_paths:
        img = Image.open(p)
        modes.append(img.mode)
        w, h = img.size
        heights.append(h)
        widths.append(w)
        max_vals.append(np.array(img, dtype=np.float32).max())

    height_counts = collections.Counter(heights)
    f_dim         = height_counts.most_common(1)[0][0]
    p95_width     = int(np.percentile(widths, 95))

    stats = {
        "f_dim"      : f_dim,
        "max_len_p95": p95_width,
        "mode"       : collections.Counter(modes).most_common(1)[0][0],
        "width_min"  : min(widths),
        "width_p50"  : int(np.median(widths)),
        "width_p95"  : p95_width,
        "width_max"  : max(widths),
        "n_silent"   : sum(v == 0 for v in max_vals),
    }

    if verbose:
        print("\n--- Spectrogram Audit ---")
        print(f"Mode distribution   : {dict(collections.Counter(modes))}")
        print(f"Height F (freq bins): {dict(height_counts)}")
        print(f"Width  T (frames)   : "
              f"min={stats['width_min']}, "
              f"p50={stats['width_p50']}, "
              f"p95={stats['width_p95']}, "
              f"max={stats['width_max']}")
        print(f"Pixel max range     : {min(max_vals):.1f} – {max(max_vals):.1f}")
        print(f"Silent clips (max=0): {stats['n_silent']}")
        print(f"\n>>> F_DIM   = {f_dim}")
        print(f">>> MAX_LEN = {p95_width}  (95th-percentile width)")

    return stats


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

def determine_eval_protocol(df: pd.DataFrame) -> str:
    n = df["child"].nunique()
    if n >= 5:
        protocol = "holdout"
    elif n >= 3:
        protocol = "lopo"
    else:
        protocol = "kfold"
    print(f"Participants: {n}  →  Evaluation protocol: '{protocol}'")
    return protocol


def build_splits(
    df: pd.DataFrame,
    protocol: str,
    k_folds: int = 5,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    save: bool = True,
) -> list:
    """Build train/val/test splits. Returns list of (train, val, test) tuples."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    folds = []

    # ── SCENARIO A: Holdout ───────────────────────────────────────────────
    if protocol == "holdout":
        random.seed(RANDOM_SEED)
        children = list(df["child"].unique())
        random.shuffle(children)
        n        = len(children)
        n_test   = max(1, round(n * test_frac))
        n_val    = max(1, round(n * val_frac))
        test_ch  = children[:n_test]
        val_ch   = children[n_test:n_test + n_val]
        train_ch = children[n_test + n_val:]
        train = df[df["child"].isin(train_ch)].copy()
        val   = df[df["child"].isin(val_ch)].copy()
        test  = df[df["child"].isin(test_ch)].copy()
        folds.append((train, val, test))
        print(f"Holdout: train={len(train)} {train_ch}, "
              f"val={len(val)} {val_ch}, test={len(test)} {test_ch}")
        if save:
            train.to_csv(SPLITS_DIR / "train.csv", index=False)
            val.to_csv(SPLITS_DIR   / "val.csv",   index=False)
            test.to_csv(SPLITS_DIR  / "test.csv",  index=False)

    # ── SCENARIO B: LOPO ─────────────────────────────────────────────────
    elif protocol == "lopo":
        children = list(df["child"].unique())
        for i, test_child in enumerate(children):
            remaining = [c for c in children if c != test_child]
            val_child  = remaining[i % len(remaining)]
            train_ch   = [c for c in remaining if c != val_child]
            train = df[df["child"].isin(train_ch)].copy()
            val   = df[df["child"] == val_child].copy()
            test  = df[df["child"] == test_child].copy()
            folds.append((train, val, test))
            print(f"LOPO fold {i}: train={train_ch}, "
                  f"val=[{val_child}], test=[{test_child}]")
            if save:
                fold_dir = SPLITS_DIR / f"fold_{i}"
                fold_dir.mkdir(exist_ok=True)
                train.to_csv(fold_dir / "train.csv", index=False)
                val.to_csv(fold_dir   / "val.csv",   index=False)
                test.to_csv(fold_dir  / "test.csv",  index=False)

    # ── SCENARIO C: Stratified k-fold ────────────────────────────────────
    else:
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True,
                              random_state=RANDOM_SEED)
        for i, (trval_idx, test_idx) in enumerate(
                skf.split(df, df["class4"])):
            trval = df.iloc[trval_idx]
            test  = df.iloc[test_idx]
            train, val = train_test_split(
                trval, test_size=val_frac,
                stratify=trval["class4"], random_state=RANDOM_SEED)
            folds.append((train.copy(), val.copy(), test.copy()))
            print(f"Fold {i}: train={len(train)}, val={len(val)}, "
                  f"test={len(test)} | "
                  f"classes={test['class4'].value_counts().to_dict()}")
            if save:
                fold_dir = SPLITS_DIR / f"fold_{i}"
                fold_dir.mkdir(exist_ok=True)
                train.to_csv(fold_dir / "train.csv", index=False)
                val.to_csv(fold_dir   / "val.csv",   index=False)
                test.to_csv(fold_dir  / "test.csv",  index=False)

    _verify_splits(folds, protocol)
    return folds


def _verify_splits(folds, protocol):
    print("\n--- Split verification ---")
    for i, (train, val, test) in enumerate(folds):
        assert len(set(train["audio_file"]) & set(test["audio_file"])) == 0, \
            f"Fold {i}: train/test file overlap!"
        assert len(set(val["audio_file"]) & set(test["audio_file"])) == 0, \
            f"Fold {i}: val/test file overlap!"
        if protocol in ("holdout", "lopo"):
            assert len(set(train["child"]) & set(test["child"])) == 0, \
                f"Fold {i}: child leakage!"
        for name, split in [("train", train), ("val", val), ("test", test)]:
            missing = [c for c in CLASS_NAMES
                       if c not in split["class4"].values]
            if missing:
                print(f"  ⚠ Fold {i} {name}: missing classes {missing}")
    print("  Leakage checks passed ✓")


def load_splits(protocol: str, k_folds: int = 5) -> list:
    folds = []
    if protocol == "holdout":
        folds.append((
            pd.read_csv(SPLITS_DIR / "train.csv"),
            pd.read_csv(SPLITS_DIR / "val.csv"),
            pd.read_csv(SPLITS_DIR / "test.csv"),
        ))
    else:
        i = 0
        while True:
            fold_dir = SPLITS_DIR / f"fold_{i}"
            if not fold_dir.exists():
                break
            folds.append((
                pd.read_csv(fold_dir / "train.csv"),
                pd.read_csv(fold_dir / "val.csv"),
                pd.read_csv(fold_dir / "test.csv"),
            ))
            i += 1
            if protocol == "kfold" and i >= k_folds:
                break
    print(f"Loaded {len(folds)} fold(s) from {SPLITS_DIR}")
    return folds


# ---------------------------------------------------------------------------
# PNG loader
# ---------------------------------------------------------------------------

def load_png(png_path: str, target_height: int = None) -> np.ndarray:
    """
    Load a spectrogram PNG → float32 (F, T) in [0, 1].
    Handles RGBA by converting to grayscale (luminance channel).
    """
    img = Image.open(png_path)

    # RGBA or RGB → grayscale
    if img.mode == "RGBA":
        # Convert via luminance to preserve intensity information
        img = img.convert("L")
    elif img.mode == "RGB":
        img = img.convert("L")
    # Mode 'L' (already grayscale) — no conversion needed

    if target_height is not None and img.size[1] != target_height:
        img = img.resize((img.size[0], target_height), Image.BILINEAR)

    return np.array(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SpectrogramDataset(Dataset):
    """
    Loads spectrogram PNGs as (max_len, f_dim) tensors.

    Returns per item:
        x        : (max_len, f_dim)  float32
        pad_mask : (max_len,)        bool — True = padded frame
        label    : ()                int64
    """

    def __init__(self, split_df, label_encoder, max_len, f_dim):
        self.records       = split_df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.max_len       = max_len
        self.f_dim         = f_dim

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row      = self.records.iloc[idx]
        spec     = load_png(row["png_path"], target_height=self.f_dim)  # (F, T)
        spec     = spec.T                                                  # (T, F)
        T        = spec.shape[0]
        pad_mask = torch.zeros(self.max_len, dtype=torch.bool)

        if T >= self.max_len:
            spec = spec[: self.max_len]
        else:
            pad  = np.zeros((self.max_len - T, self.f_dim), dtype=np.float32)
            spec = np.vstack([spec, pad])
            pad_mask[T:] = True

        label = self.label_encoder[row["class4"]]
        return (
            torch.from_numpy(spec),
            pad_mask,
            torch.tensor(label, dtype=torch.long),
        )

    def class_weights(self) -> torch.Tensor:
        y    = self.records["class4"].map(self.label_encoder).values
        cnts = np.bincount(y, minlength=len(self.label_encoder)).astype(float)
        w    = 1.0 / (cnts + 1e-6)
        return torch.FloatTensor(w / w.sum() * len(self.label_encoder))
