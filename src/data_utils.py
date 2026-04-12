"""
data_utils.py
-------------
Master table construction, spectrogram audit, split construction,
and PyTorch Dataset classes for both models.

Supports the full 2026 training dataset:
  - CSV: contextsvocs_groupproject_2026-04-07_133933.csv
  - Audio IDs: audio_0000 … audio_1008
  - PNG paths: training_data/spectrograms/audio_XXXX.png
  - 9 children, 3 languages, 4 classes, 1009 clips

Label mapping (new 2026 scheme):
  PLTOY ← PLAY1, PLAY+, TOYMO
  MOVE  ← WALK, CARRIED, ARRIVES, LEAVES, FALLS
  INTER ← CONV, OBSERV_INTER, SEEK_INTER, GROOMED, AGGR
  EXTEV ← FOOD, REST, SLE
"""

import collections
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold

from config import (
    CONTEXTS_CSV, SPEC_DIR, WAV_DIR,
    PLTOY_LABELS, MOVE_LABELS, INTER_LABELS, EXTEV_LABELS,
    CLASS_NAMES, LABEL_ENCODER, N_CLASSES,
    SPLITS_DIR, RANDOM_SEED,
    USE_PER_CLIP_NORM,
    USE_SPECAUGMENT, SPECAUG_TIME_MASKS, SPECAUG_TIME_WIDTH,
    SPECAUG_FREQ_MASKS, SPECAUG_FREQ_WIDTH,
)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def map_label(lbl: str):
    """
    Labels in contextsvocs_groupproject_2026-04-09_104847.csv are already
    in 4-class format (EXTEV, INTER, MOVE, PLTOY). No mapping needed.
    Returns None for any label not in CLASS_NAMES (safety check).
    """
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
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the 2026 contexts CSV, apply label mapping, resolve PNG paths,
    and return a clean DataFrame.

    The 2026 CSV format differs from the pilot CSV:
      - Column 'audio_ID'  (e.g. 'audio_0000') — no .wav suffix
      - Timestamps in seconds (start_sec, end_sec, duration_sec)
      - No language filter — all languages are part of the dataset
      - Direct filename matching: audio_ID + '.png' → spectrogram path

    Returns
    -------
    pd.DataFrame with columns:
        audio_ID, audio_file, start_sec, end_sec, duration_sec,
        child, lang, gender, age, context_label, class4, png_path, png_exists
    """
    df = pd.read_csv(contexts_csv)

    # Drop index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if verbose:
        print(f"Loaded {len(df)} rows from {contexts_csv.name}")

    # Label mapping
    df["class4"] = df["context_label"].apply(map_label)

    excluded = df[df["class4"].isna()]["context_label"].value_counts()
    if verbose and len(excluded):
        print(f"\nExcluded labels (outside 4-class scheme):")
        for lbl, cnt in excluded.items():
            print(f"  {lbl:15s}: {cnt}")

    df = df[df["class4"].notna()].reset_index(drop=True)

    # Add audio_file column (with .wav suffix) for compatibility
    df["audio_file"] = df["audio_ID"] + ".wav"

    # PNG path: direct filename match (no positional assignment needed)
    df["png_path"]   = df["audio_ID"].apply(
        lambda aid: str(spec_dir / f"{aid}.png")
    )
    df["wav_path"]   = df["audio_ID"].apply(
        lambda aid: str(WAV_DIR / f"{aid}.wav")
    )
    df["png_exists"] = df["png_path"].apply(os.path.exists)
    df["wav_exists"] = df["wav_path"].apply(os.path.exists)

    if verbose:
        n_png = df["png_exists"].sum()
        n_wav = df["wav_exists"].sum()
        print(f"\n{'='*55}")
        print(f"Total usable examples  : {len(df)}")
        print(f"Spectrograms found     : {n_png}/{len(df)}")
        print(f"WAV files found        : {n_wav}/{len(df)}")
        if n_png < len(df):
            print(f"\n  ⚠  {len(df)-n_png} spectrograms missing.")
            print(f"     Run:  python scripts/generate_spectrograms.py")
        print(f"\nClass distribution:")
        print(df["class4"].value_counts().to_string())
        print(f"\nParticipants           : {df['child'].nunique()}")
        print(f"Per child:")
        print(df.groupby("child").size().to_string())
        print(f"\nLanguages:")
        print(df["lang"].value_counts().to_string())
        print(f"\nClass × Child:")
        print(pd.crosstab(df["child"], df["class4"]).to_string())
        print(f"\nDuration (sec) — mean: {df['duration_sec'].mean():.2f}, "
              f"std: {df['duration_sec'].std():.2f}, "
              f"max: {df['duration_sec'].max():.2f}")
        print(f"{'='*55}")

    return df


# ---------------------------------------------------------------------------
# Spectrogram audit
# ---------------------------------------------------------------------------

def audit_spectrograms(png_paths: list, verbose: bool = True) -> dict:
    """
    Scan a small sample of PNGs to determine format, dimensions, and pixel range.
    Samples 20 files only — sufficient to confirm all images are consistent.
    """
    import random as _random

    n_total = len(png_paths)
    sample  = _random.sample(png_paths, min(20, n_total))

    heights, widths, modes, max_vals = [], [], [], []
    for p in sample:
        img = Image.open(p)
        modes.append(img.mode)
        w, h = img.size
        heights.append(h)
        widths.append(w)
        extrema = img.getextrema()
        pmax = max(b[1] for b in extrema) if isinstance(extrema[0], tuple) else extrema[1]
        max_vals.append(float(pmax))

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
        "n_silent"   : sum(v == 0 for v in max_vals),  # from 20-file sample
    }

    if verbose:
        print(f"\n--- Spectrogram Audit (sampled {len(sample)}/{n_total} files) ---")
        print(f"Mode distribution   : {dict(collections.Counter(modes))}")
        print(f"Height F (mel bins) : {dict(height_counts)}")
        print(f"Width  T (frames)   : "
              f"min={stats['width_min']}, "
              f"p50={stats['width_p50']}, "
              f"p95={stats['width_p95']}, "
              f"max={stats['width_max']}")
        print(f"Pixel max range     : {min(max_vals):.1f} – {max(max_vals):.1f}")
        print(f"Silent clips (max=0): {stats['n_silent']}  (from sample)")
        print(f"\n>>> F_DIM   = {f_dim}")
        print(f">>> MAX_LEN = {p95_width}  (95th-percentile T)")

    return stats


# ---------------------------------------------------------------------------
# Metadata viability check
# ---------------------------------------------------------------------------

def check_metadata_viability(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Determine whether age and gender can safely be used as metadata inputs.
    Returns dict with viability flags and recommended USE_EXTERNAL_META value.
    """
    n_children       = df["child"].nunique()
    n_unique_ages    = df.groupby("child")["age"].first().nunique()
    n_unique_genders = df.groupby("child")["gender"].first().nunique()

    age_is_identity    = (n_unique_ages    == n_children)
    gender_is_identity = (n_unique_genders == n_children)
    metadata_safe      = (n_children >= 4 and
                          not age_is_identity and
                          not gender_is_identity)

    result = {
        "n_children"         : n_children,
        "n_unique_ages"      : n_unique_ages,
        "n_unique_genders"   : n_unique_genders,
        "age_is_identity"    : age_is_identity,
        "gender_is_identity" : gender_is_identity,
        "USE_EXTERNAL_META"  : metadata_safe,
    }

    if verbose:
        print("\n--- Metadata viability check ---")
        print(f"  Participants   : {n_children}")
        print(f"  Unique ages    : {n_unique_ages}  "
              f"{'(= identity proxy)' if age_is_identity else '(safe — not unique per child)'}")
        print(f"  Unique genders : {n_unique_genders}  "
              f"{'(= identity proxy)' if gender_is_identity else '(safe — not unique per child)'}")
        print(f"  → USE_EXTERNAL_META = {metadata_safe}")
        if metadata_safe:
            print(f"    External metadata (age + gender) is safe to use as query input.")
        else:
            print(f"    Metadata collinear with identity → using learned query.")

    return result


def compute_meta_stats(train_df: pd.DataFrame) -> dict:
    """Compute age normalisation stats from the training split only."""
    age_mean   = float(train_df["age"].mean())
    age_std    = float(train_df["age"].std()) + 1e-8
    genders    = sorted(train_df["gender"].dropna().unique())
    gender_map = {g: float(i) / max(len(genders) - 1, 1)
                  for i, g in enumerate(genders)}
    print(f"  Meta stats: age_mean={age_mean:.2f}, age_std={age_std:.2f}, "
          f"gender_map={gender_map}")
    return {"age_mean": age_mean, "age_std": age_std, "gender_map": gender_map}


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


def build_splits(df: pd.DataFrame, protocol: str,
                 k_folds: int = 5, val_frac: float = 0.15,
                 test_frac: float = 0.15, save: bool = True) -> list:
    """
    Construct train/val/test splits. Returns list of (train, val, test) tuples.
    For 'holdout': one tuple. For 'lopo'/'kfold': one tuple per fold.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    folds = []

    # ── SCENARIO A: Participant-disjoint holdout (≥5 participants) ─────────
    if protocol == "holdout":
        random.seed(RANDOM_SEED)
        children = list(df["child"].unique())

        # Sort children by size so the two largest go to test and val
        # (they have the most balanced class distributions by sheer volume),
        # and remaining children form the training set.
        child_sizes = df.groupby("child").size().sort_values(ascending=False)
        children_by_size = child_sizes.index.tolist()

        n        = len(children_by_size)
        n_test   = max(1, round(n * test_frac))
        n_val    = max(1, round(n * val_frac))
        # Largest children → test, second largest → val, rest → train
        test_ch  = children_by_size[:n_test]
        val_ch   = children_by_size[n_test:n_test + n_val]
        train_ch = children_by_size[n_test + n_val:]

        train = df[df["child"].isin(train_ch)].copy()
        val   = df[df["child"].isin(val_ch)].copy()
        test  = df[df["child"].isin(test_ch)].copy()
        folds.append((train, val, test))

        print(f"Holdout split:")
        print(f"  Train : {train_ch} → n={len(train)}")
        print(f"  Val   : {val_ch}   → n={len(val)}")
        print(f"  Test  : {test_ch}  → n={len(test)}")
        for name, split in [("train", train), ("val", val), ("test", test)]:
            print(f"  {name} classes: {split['class4'].value_counts().to_dict()}")

        if save:
            train.to_csv(SPLITS_DIR / "train.csv", index=False)
            val.to_csv(SPLITS_DIR   / "val.csv",   index=False)
            test.to_csv(SPLITS_DIR  / "test.csv",  index=False)

    # ── SCENARIO B: LOPO (3–4 participants) ────────────────────────────────
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
            print(f"LOPO fold {i}: train={train_ch}, val=[{val_child}], test=[{test_child}]")
            if save:
                fold_dir = SPLITS_DIR / f"fold_{i}"
                fold_dir.mkdir(exist_ok=True)
                train.to_csv(fold_dir / "train.csv", index=False)
                val.to_csv(fold_dir   / "val.csv",   index=False)
                test.to_csv(fold_dir  / "test.csv",  index=False)

    # ── SCENARIO C: Stratified k-fold (< 3 participants, fallback) ─────────
    else:
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True,
                              random_state=RANDOM_SEED)
        for i, (trval_idx, test_idx) in enumerate(skf.split(df, df["class4"])):
            trval = df.iloc[trval_idx]
            test  = df.iloc[test_idx]
            train, val = train_test_split(
                trval, test_size=val_frac,
                stratify=trval["class4"], random_state=RANDOM_SEED)
            folds.append((train.copy(), val.copy(), test.copy()))
            print(f"Fold {i}: train={len(train)}, val={len(val)}, test={len(test)}")
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
        assert len(set(train["audio_file"]) & set(test["audio_file"])) == 0
        assert len(set(val["audio_file"])   & set(test["audio_file"])) == 0
        if protocol in ("holdout", "lopo"):
            overlap = set(train["child"]) & set(test["child"])
            assert len(overlap) == 0, f"Fold {i}: child leakage {overlap}!"
        for name, split in [("train", train), ("val", val), ("test", test)]:
            missing = [c for c in CLASS_NAMES if c not in split["class4"].values]
            if missing:
                print(f"  ⚠ Fold {i} {name}: missing classes {missing}")
    print("  Leakage checks passed ✓")


def load_splits(protocol: str, k_folds: int = 5) -> list:
    """Re-load previously saved splits from SPLITS_DIR."""
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

def load_png(png_path: str, target_height: int = None,
            target_width: int = None) -> np.ndarray:
    """
    Load a spectrogram PNG → float32 (F, T) in [0, 1].
    Converts RGB/RGBA to grayscale. Resizes if target dimensions are given.

    Uses a `with` block so the file handle is released immediately. Without
    this, PIL's lazy loader can leave file descriptors open long enough for
    FD exhaustion on larger datasets, which manifests as a silent hang on
    later reads.
    """
    with Image.open(png_path) as img:
        img.load()  # force decode so the file can be closed
        if img.mode in ("RGB", "RGBA"):
            img = img.convert("L")
        if target_height is not None or target_width is not None:
            h = target_height if target_height is not None else img.size[1]
            w = target_width  if target_width  is not None else img.size[0]
            if img.size != (w, h):
                img = img.resize((w, h), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class SpectrogramDataset(Dataset):
    """
    Audio-only dataset (Model 2).
    Returns: (x, pad_mask, label)
      x        : (max_len, f_dim)  float32
      pad_mask : (max_len,)        bool — True = padded frame
      label    : ()                int64

    Spectrograms are preloaded into RAM at __init__ time. With ~570 train
    clips × 230 × 128 × 4 bytes ≈ 67 MB the whole cache is comfortable on
    any GPU node, eliminates per-batch PNG decoding overhead, and sidesteps
    file-descriptor / mount / PIL-handle issues that can hang training.

    Preprocessing pipeline (controlled by config flags):

      1. PNG decode → float [0, 1] → (max_len, f_dim) tensor (cached)
      2. Per-clip mean/variance normalisation along the time axis
         (USE_PER_CLIP_NORM). Removes loudness/timbre drift across
         children, the dominant nuisance variable in participant-disjoint
         evaluation. Vectorised over the cache.
      3. SpecAugment time + frequency masking (USE_SPECAUGMENT, train only)
         applied stochastically inside __getitem__. Forces invariance to
         specific time/frequency regions.

    Parameters
    ----------
    train : if True, SpecAugment is applied in __getitem__ (when enabled
            via config). Set False for val/test so evaluation is
            deterministic. Defaults to False.
    """

    def __init__(self, split_df, label_encoder, max_len, f_dim,
                 train: bool = False):
        self.records       = split_df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.max_len       = max_len
        self.f_dim         = f_dim
        self.train         = train

        # Preload every spectrogram into a single contiguous tensor of
        # shape (N, max_len, f_dim). One-time cost: ~1–2s for 569 PNGs.
        n = len(self.records)
        self._cache = torch.empty((n, max_len, f_dim), dtype=torch.float32)
        for i in range(n):
            png_path = self.records.iloc[i]["png_path"]
            spec = load_png(png_path,
                            target_height=f_dim,
                            target_width=max_len)        # (f_dim, max_len)
            self._cache[i] = torch.from_numpy(spec.T)    # (max_len, f_dim)

        # ── Per-clip normalisation (vectorised over the whole cache) ────
        # Mean/std along the time axis (dim=1) per clip per feature, then
        # standardise. Done once at construction so the cost is amortised
        # across all epochs. After this, the per-clip mean is 0 and std is
        # 1 along time, which is also why SpecAugment uses 0 as the
        # mask-fill value below.
        if USE_PER_CLIP_NORM:
            mean = self._cache.mean(dim=1, keepdim=True)
            std  = self._cache.std (dim=1, keepdim=True).clamp(min=1e-6)
            self._cache = (self._cache - mean) / std

    def __len__(self):
        return len(self.records)

    def _spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment time + frequency masking to a single clip.
        Operates on a CLONE of the cached tensor — never the cache
        itself, which is shared across epochs and must stay immutable.
        Masked values are set to 0.0 (the per-clip mean after step 2).
        """
        x = x.clone()
        T, F = x.shape

        for _ in range(SPECAUG_TIME_MASKS):
            t_width = int(torch.randint(0, SPECAUG_TIME_WIDTH + 1, (1,)).item())
            if t_width == 0 or T - t_width <= 0:
                continue
            t_start = int(torch.randint(0, T - t_width, (1,)).item())
            x[t_start:t_start + t_width, :] = 0.0

        for _ in range(SPECAUG_FREQ_MASKS):
            f_width = int(torch.randint(0, SPECAUG_FREQ_WIDTH + 1, (1,)).item())
            if f_width == 0 or F - f_width <= 0:
                continue
            f_start = int(torch.randint(0, F - f_width, (1,)).item())
            x[:, f_start:f_start + f_width] = 0.0

        return x

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        # Pure RAM lookup — no file I/O during training.
        x        = self._cache[idx]
        if self.train and USE_SPECAUGMENT:
            x = self._spec_augment(x)
        pad_mask = torch.zeros(self.max_len, dtype=torch.bool)
        label    = self.label_encoder[row["class4"]]
        return (
            x,
            pad_mask,
            torch.tensor(label, dtype=torch.long),
        )

    def class_weights(self) -> torch.Tensor:
        y    = self.records["class4"].map(self.label_encoder).values
        cnts = np.bincount(y, minlength=len(self.label_encoder)).astype(float)
        w    = 1.0 / (cnts + 1e-6)
        return torch.FloatTensor(w / w.sum() * len(self.label_encoder))


class MetadataSpectrogramDataset(SpectrogramDataset):
    """
    Metadata-aware dataset (Model 3).
    Returns: (x, pad_mask, label, meta)
      meta : (2,) float32 — [age_normalised, gender_binary]
    """

    def __init__(self, split_df, label_encoder, max_len, f_dim,
                 meta_stats: dict = None, use_external_meta: bool = False,
                 train: bool = False):
        super().__init__(split_df, label_encoder, max_len, f_dim, train=train)
        self.use_external_meta = use_external_meta
        self.meta_stats        = meta_stats or {}

    def __getitem__(self, idx):
        x, pad_mask, label = super().__getitem__(idx)

        if self.use_external_meta and self.meta_stats:
            row        = self.records.iloc[idx]
            age_mean   = self.meta_stats["age_mean"]
            age_std    = self.meta_stats["age_std"]
            gender_map = self.meta_stats["gender_map"]
            age_norm   = (float(row["age"]) - age_mean) / age_std
            gender_val = gender_map.get(str(row.get("gender", "U")), 0.5)
            meta = torch.FloatTensor([age_norm, gender_val])
        else:
            meta = torch.zeros(2, dtype=torch.float32)

        return x, pad_mask, label, meta