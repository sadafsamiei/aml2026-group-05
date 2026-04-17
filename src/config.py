"""
config.py
---------
Single source of truth for all paths, constants, and hyperparameters.

Path layout on RenkuLab
------------------------
Input data (read-only SwitchDrive mount):
    DATA_DIR = /home/renku/work/aml-project/training_data/
        ├── anonymised_audios/   ← audio_0000.wav … audio_1008.wav
        ├── spectrograms/        ← audio_0000.png … (generated once)
        └── contextsvocs_groupproject_2026-04-07_133933.csv

Output / results (writable, syncs to SwitchDrive data folder):
    RES_DIR = /home/renku/work/data/
        ├── splits/              ← train/val/test CSVs
        ├── models/              ← model checkpoints (.pt)
        ├── results/             ← JSON metrics
        └── figures/             ← training curves + confusion matrices
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# ★ Adjust these two paths if your mount points differ ★
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/renku/work/aml-project/training_data")
RES_DIR  = Path("/home/renku/work/data")

# ---------------------------------------------------------------------------
# Input paths  (derived from DATA_DIR — READ ONLY)
# ---------------------------------------------------------------------------
WAV_DIR      = DATA_DIR / "denoised_audios"     # audio_0000.wav … audio_1008.wav
CONTEXTS_CSV = DATA_DIR / "contextsvocs_groupproject_2026-04-09_104847.csv"

# ---------------------------------------------------------------------------
# Output paths  (derived from RES_DIR — READ/WRITE, synced to SwitchDrive)
# ---------------------------------------------------------------------------
SPEC_DIR    = RES_DIR / "spectrograms"   # generated from WAV files — writable
MFCC_DIR    = RES_DIR / "mfccs"          # MFCC .npy files (from audio_to_mfccs.py)
SPLITS_DIR  = RES_DIR / "splits"
MODELS_DIR  = RES_DIR / "models"
RESULTS_DIR = RES_DIR / "results"
FIGURES_DIR = RES_DIR / "figures"

# ---------------------------------------------------------------------------
# Label mapping  (labels already in 4-class format — no mapping needed)
# context_label column contains: EXTEV, INTER, MOVE, PLTOY directly
# ---------------------------------------------------------------------------
PLTOY_LABELS = {"PLTOY"}
MOVE_LABELS  = {"MOVE"}
INTER_LABELS = {"INTER"}
EXTEV_LABELS = {"EXTEV"}

CLASS_NAMES   = ["EXTEV", "INTER", "MOVE", "PLTOY"]
LABEL_ENCODER = {c: i for i, c in enumerate(CLASS_NAMES)}
LABEL_DECODER = {i: c for c, i in LABEL_ENCODER.items()}
N_CLASSES     = len(CLASS_NAMES)

# No language filter — the dataset spans Shipibo, Tagalog, Vietnamese
# intentionally. All three languages are used together.
LANG_FILTER = None

# ---------------------------------------------------------------------------
# Spectrogram generation parameters
# Used by generate_spectrograms.py and determines F_DIM
# ---------------------------------------------------------------------------
SPEC_SR    = 16000   # resample rate (Hz)
SPEC_N_MELS = 128    # mel frequency bins  →  F_DIM = 128
SPEC_HOP   = 256     # hop length in samples
SPEC_N_FFT = 1024    # FFT window size
SPEC_FMIN  = 20      # min frequency (Hz)
SPEC_FMAX  = 8000    # max frequency (Hz)

# ---------------------------------------------------------------------------
# Sequence constants (set here; verified at runtime by audit_spectrograms)
# ---------------------------------------------------------------------------
F_DIM   = 128    # mel frequency bins (actual audio resolution, not image size)
MAX_LEN = 230    # time frame cap (p95 of actual frame counts before resizing)
                 # The 1000×400 viridis PNGs are resized to MAX_LEN×F_DIM inside load_png()

# ---------------------------------------------------------------------------
# Hyperparameter presets
#
# HPARAM_SMALL → < 100 training examples  (pilot fallback)
# HPARAM_FULL  → 300+ training examples   (current: 1009 clips)
# ---------------------------------------------------------------------------
HPARAM_SMALL = {
    "d_model": 32, "nhead": 4, "num_layers": 1,
    "dropout": 0.4, "lr": 1e-3, "weight_decay": 1e-4,
    "batch_size": 8, "epochs": 150, "patience": 20,
}

HPARAM_FULL = {
    "d_model": 64, "nhead": 4, "num_layers": 2,
    "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4,
    "batch_size": 64, "epochs": 100, "patience": 15,
}

# Active default — FULL because we have 1009 examples
HPARAM_DEFAULTS = HPARAM_FULL

# ---------------------------------------------------------------------------
# Preprocessing / augmentation flags
# ---------------------------------------------------------------------------
# Input feature selection. When False (default), the pipeline loads mel
# spectrogram PNGs from SPEC_DIR — this is the spec-compliant configuration.
# When True, it loads MFCC .npy files from MFCC_DIR instead. Both produce
# (MAX_LEN, F_DIM) tensors so no other config changes are needed.
USE_MFCC = True

# Per-clip mean/variance normalisation of the cached spectrogram tensor.
# Subtracts the per-clip mean and divides by the per-clip std along the
# time axis. Removes loudness/timbre drift across children, which is the
# dominant nuisance variable in participant-disjoint evaluation.
# Vectorised, applied once at dataset construction.
USE_PER_CLIP_NORM = True

# SpecAugment (time + frequency masking) applied train-only inside
# __getitem__. Default values are conservative for short clips.
USE_SPECAUGMENT     = True
SPECAUG_TIME_MASKS  = 2     # number of time bands to mask
SPECAUG_TIME_WIDTH  = 25    # max width (frames) of each time band
SPECAUG_FREQ_MASKS  = 2     # number of frequency bands to mask
SPECAUG_FREQ_WIDTH  = 12    # max width (mel bins) of each freq band

# Hyperparameter grid (used with --grid_search flag)
HPARAM_GRID = [
    {"d_model": 32,  "num_layers": 1, "dropout": 0.3, "lr": 1e-3,  "batch_size": 16},
    {"d_model": 32,  "num_layers": 1, "dropout": 0.5, "lr": 1e-3,  "batch_size": 16},
    {"d_model": 64,  "num_layers": 2, "dropout": 0.3, "lr": 1e-3,  "batch_size": 32},
    {"d_model": 64,  "num_layers": 2, "dropout": 0.5, "lr": 1e-3,  "batch_size": 32},
    {"d_model": 64,  "num_layers": 2, "dropout": 0.3, "lr": 5e-4,  "batch_size": 32},
    {"d_model": 128, "num_layers": 2, "dropout": 0.3, "lr": 5e-4,  "batch_size": 32},
    {"d_model": 128, "num_layers": 3, "dropout": 0.3, "lr": 5e-4,  "batch_size": 32},
    {"d_model": 128, "num_layers": 3, "dropout": 0.5, "lr": 5e-4,  "batch_size": 32},
]

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_output_dirs():
    """Create all output directories. Safe to call multiple times."""
    for d in [SPEC_DIR, MFCC_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def print_config():
    preset = ("FULL (d=64, L=2)"  if HPARAM_DEFAULTS is HPARAM_FULL
              else "SMALL (d=32, L=1)")
    print("=" * 62)
    print("PROJECT CONFIGURATION")
    print("=" * 62)
    print(f"DATA_DIR      : {DATA_DIR}")
    print(f"  WAV dir     : {WAV_DIR}  (exists={WAV_DIR.exists()})")
    print(f"  Contexts CSV: {CONTEXTS_CSV.name}  (exists={CONTEXTS_CSV.exists()})")
    print(f"RES_DIR       : {RES_DIR}  (read/write)")
    print(f"  Spec dir    : {SPEC_DIR}  (exists={SPEC_DIR.exists()})")
    print(f"  Splits      : {SPLITS_DIR}")
    print(f"  Models      : {MODELS_DIR}")
    print(f"  Results     : {RESULTS_DIR}")
    print(f"  Figures     : {FIGURES_DIR}")
    print(f"Classes       : {CLASS_NAMES}")
    print(f"Lang filter   : {LANG_FILTER!r}  (None = all languages)")
    print(f"Active preset : {preset}")
    feat_src = "MFCC .npy" if USE_MFCC else "mel spectrogram PNG"
    print(f"F_DIM / MAX_LEN : {F_DIM} / {MAX_LEN}  (input: {feat_src})")
    print(f"Features        : {'MFCC (40+Δ+ΔΔ → 120, padded to 128)' if USE_MFCC else 'mel spectrogram (128 bins)'}")
    print("=" * 62)