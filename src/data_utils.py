"""
data_utils.py
-------------
Master table construction, spectrogram audit, split construction,
and PyTorch Dataset class for the audio-only Transformer baseline.
"""

import os
import collections
import random
import pickle
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
    """Map a fine-grained context label to one of the 4 coarse classes.
    Returns None for labels outside the scheme (REST, TOYMO, FOOD*, etc.)."""
    if lbl in PLTOY_LABELS:  return "PLTOY"
    if lbl in MOVE_LABELS:   return "MOVE"
    if lbl in INTER_LABELS:  return "INTER"
    if lbl in EXTEV_LABELS:  return "EXTEV"
    return None


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
    Load the contexts CSV, apply language and label filters, add derived
    columns (duration, PNG path, existence flag), and return a clean DataFrame.

    Parameters
    ----------
    contexts_csv : path to contexts_vocalisation_combined_df.csv
    spec_dir     : directory containing spectrogram PNGs
    lang_filter  : language code to keep ('shp' for Shipibo-Conibo).
                   Pass None to keep all languages.
    verbose      : print summary statistics

    Returns
    -------
    pd.DataFrame with columns:
        audio_file, source_eaf, start_ms, end_ms, context_label, lang,
        child, gender, age, class4, dur_ms, png_path, png_exists
    """
    df = pd.read_csv(contexts_csv)

    # Language filter
    if lang_filter:
        before = len(df)
        df = df[df["lang"] == lang_filter].copy()
        if verbose:
            print(f"Language filter '{lang_filter}': {before} → {len(df)} rows")

    # Label mapping
    df["class4"] = df["context_label"].apply(map_label)

    # Report excluded labels before dropping
    excluded = df[df["class4"].isna()]["context_label"].value_counts()
    if verbose and len(excluded):
        print(f"\nExcluded labels (outside 4-class scheme):")
        for lbl, cnt in excluded.items():
            print(f"  {lbl:12s}: {cnt}")

    df = df[df["class4"].notna()].reset_index(drop=True)

    # Derived columns
    df["dur_ms"]     = df["end_ms"] - df["start_ms"]
    df["png_path"]   = df["audio_file"].apply(
        lambda f: str(spec_dir / f.replace(".wav", ".png"))
    )
    df["png_exists"] = df["png_path"].apply(os.path.exists)

    if verbose:
        print(f"\n{'='*45}")
        print(f"Usable examples  : {len(df)}")
        print(f"Class distribution:\n{df['class4'].value_counts().to_string()}")
        print(f"\nParticipants     : {df['child'].nunique()}")
        print(f"Per child:\n{df.groupby('child').size().to_string()}")
        print(f"\nClass × Child:\n{pd.crosstab(df['child'], df['class4']).to_string()}")
        print(f"\nDuration (ms) — mean: {df['dur_ms'].mean():.0f}, "
              f"std: {df['dur_ms'].std():.0f}, "
              f"min: {df['dur_ms'].min()}, max: {df['dur_ms'].max()}")
        missing = (~df["png_exists"]).sum()
        print(f"Missing PNGs     : {missing}")
        if missing:
            print("  ⚠ Some PNGs not found. Check SPEC_DIR in config.py")
        print(f"{'='*45}")

    return df


# ---------------------------------------------------------------------------
# Spectrogram audit
# ---------------------------------------------------------------------------

def audit_spectrograms(png_paths: list, verbose: bool = True) -> dict:
    """
    Scan all PNGs and report height (F), width (T) distribution, colour mode,
    and pixel value range. Returns a dict with key stats.

    Call this once before training and record F_DIM and MAX_LEN.
    """
    heights, widths, modes, max_vals = [], [], [], []

    for p in png_paths:
        img = Image.open(p)
        modes.append(img.mode)
        w, h = img.size          # PIL: (width, height)
        heights.append(h)
        widths.append(w)
        max_vals.append(np.array(img, dtype=np.float32).max())

    height_counts = collections.Counter(heights)
    p95_width     = int(np.percentile(widths, 95))
    f_dim         = height_counts.most_common(1)[0][0]

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
        print(f"Mode distribution  : {dict(collections.Counter(modes))}")
        print(f"Height F (freq bins): {dict(height_counts)}")
        print(f"Width  T (frames)   : "
              f"min={stats['width_min']}, "
              f"p50={stats['width_p50']}, "
              f"p95={stats['width_p95']}, "
              f"max={stats['width_max']}")
        print(f"Pixel max range     : {min(max_vals):.1f} – {max(max_vals):.1f}")
        print(f"Silent clips (max=0): {stats['n_silent']}")
        print(f"\n>>> Use F_DIM   = {f_dim}")
        print(f">>> Use MAX_LEN = {p95_width}  (95th-percentile width)")

    return stats


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

def determine_eval_protocol(df: pd.DataFrame) -> str:
    """
    Choose the evaluation protocol based on participant count.

    Returns one of: 'holdout', 'lopo', 'kfold'
    """
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
    """
    Construct train/val/test splits according to the evaluation protocol.

    Returns a list of (train_df, val_df, test_df) tuples.
    - For 'holdout': one tuple.
    - For 'lopo' and 'kfold': one tuple per fold.

    If save=True, CSVs are written to SPLITS_DIR.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    folds = []

    # ------------------------------------------------------------------
    # SCENARIO A: Fixed participant-disjoint holdout
    # ------------------------------------------------------------------
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
        print(f"Holdout — train: {len(train)} "
              f"({train_ch}), val: {len(val)} ({val_ch}), "
              f"test: {len(test)} ({test_ch})")

        if save:
            train.to_csv(SPLITS_DIR / "train.csv", index=False)
            val.to_csv(SPLITS_DIR   / "val.csv",   index=False)
            test.to_csv(SPLITS_DIR  / "test.csv",  index=False)

    # ------------------------------------------------------------------
    # SCENARIO B: Leave-one-participant-out CV
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # SCENARIO C: Stratified k-fold CV (within-participant)
    # ------------------------------------------------------------------
    else:
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True,
                              random_state=RANDOM_SEED)
        for fold_i, (trval_idx, test_idx) in enumerate(
                skf.split(df, df["class4"])):
            trval = df.iloc[trval_idx]
            test  = df.iloc[test_idx]
            train, val = train_test_split(
                trval, test_size=val_frac,
                stratify=trval["class4"], random_state=RANDOM_SEED)

            folds.append((train.copy(), val.copy(), test.copy()))
            print(f"Fold {fold_i}: train={len(train)}, "
                  f"val={len(val)}, test={len(test)} | "
                  f"test classes: {test['class4'].value_counts().to_dict()}")

            if save:
                fold_dir = SPLITS_DIR / f"fold_{fold_i}"
                fold_dir.mkdir(exist_ok=True)
                train.to_csv(fold_dir / "train.csv", index=False)
                val.to_csv(fold_dir   / "val.csv",   index=False)
                test.to_csv(fold_dir  / "test.csv",  index=False)

    _verify_splits(folds, protocol)
    return folds


def _verify_splits(folds, protocol):
    """Run leakage and class-coverage checks on all folds."""
    print("\n--- Split verification ---")
    for i, (train, val, test) in enumerate(folds):
        # No file appears in two splits
        assert len(set(train["audio_file"]) & set(test["audio_file"])) == 0, \
            f"Fold {i}: train/test file overlap!"
        assert len(set(val["audio_file"])   & set(test["audio_file"])) == 0, \
            f"Fold {i}: val/test file overlap!"

        # Participant disjointness for holdout and LOPO
        if protocol in ("holdout", "lopo"):
            assert len(set(train["child"]) & set(test["child"])) == 0, \
                f"Fold {i}: child leakage between train and test!"

        # Report class coverage (warn if a class is missing)
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            missing = [c for c in CLASS_NAMES if c not in split_df["class4"].values]
            if missing:
                print(f"  ⚠ Fold {i} {split_name}: missing classes {missing}")

    print("  All leakage checks passed ✓")


def load_splits(protocol: str, k_folds: int = 5) -> list:
    """Re-load previously saved splits from SPLITS_DIR."""
    folds = []
    if protocol == "holdout":
        train = pd.read_csv(SPLITS_DIR / "train.csv")
        val   = pd.read_csv(SPLITS_DIR / "val.csv")
        test  = pd.read_csv(SPLITS_DIR / "test.csv")
        folds.append((train, val, test))
    else:
        n = k_folds if protocol == "kfold" else None
        i = 0
        while True:
            fold_dir = SPLITS_DIR / f"fold_{i}"
            if not fold_dir.exists():
                break
            train = pd.read_csv(fold_dir / "train.csv")
            val   = pd.read_csv(fold_dir / "val.csv")
            test  = pd.read_csv(fold_dir / "test.csv")
            folds.append((train, val, test))
            i += 1
            if n and i >= n:
                break
    print(f"Loaded {len(folds)} fold(s) from {SPLITS_DIR}")
    return folds


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

def _load_png(png_path: str, target_height: int = None) -> np.ndarray:
    """Load a spectrogram PNG → float32 (F, T) in [0, 1]."""
    img = Image.open(png_path)
    if img.mode in ("RGB", "RGBA"):
        img = img.convert("L")   # colour → grayscale
    if target_height is not None and img.size[1] != target_height:
        img = img.resize((img.size[0], target_height), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


class SpectrogramDataset(Dataset):
    """
    PyTorch Dataset that loads spectrogram PNGs and returns:
        x          : (MAX_LEN, F_DIM) float tensor
        pad_mask   : (MAX_LEN,) bool tensor  — True = padded position
        label      : scalar int64 tensor

    Parameters
    ----------
    split_df      : DataFrame with at least 'png_path' and 'class4' columns
    label_encoder : dict mapping class name → int
    max_len       : maximum number of time frames (pad/truncate to this)
    f_dim         : number of frequency bins (height of each PNG)
    """

    def __init__(
        self,
        split_df: pd.DataFrame,
        label_encoder: dict,
        max_len: int,
        f_dim: int,
    ):
        self.records       = split_df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.max_len       = max_len
        self.f_dim         = f_dim

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row  = self.records.iloc[idx]
        spec = _load_png(row["png_path"], target_height=self.f_dim)  # (F, T)
        spec = spec.T                                                  # (T, F)

        T          = spec.shape[0]
        pad_mask   = torch.zeros(self.max_len, dtype=torch.bool)

        if T >= self.max_len:
            spec = spec[: self.max_len]
        else:
            pad  = np.zeros((self.max_len - T, self.f_dim), dtype=np.float32)
            spec = np.vstack([spec, pad])
            pad_mask[T:] = True   # True marks padded (ignored) positions

        label = self.label_encoder[row["class4"]]
        return (
            torch.from_numpy(spec),                       # (MAX_LEN, F_DIM)
            pad_mask,                                     # (MAX_LEN,)
            torch.tensor(label, dtype=torch.long),        # scalar
        )

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for use with CrossEntropyLoss."""
        y    = self.records["class4"].map(self.label_encoder).values
        cnts = np.bincount(y, minlength=len(self.label_encoder)).astype(float)
        w    = 1.0 / (cnts + 1e-6)
        w    = w / w.sum() * len(self.label_encoder)
        return torch.FloatTensor(w)
