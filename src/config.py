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
WAV_DIR      = DATA_DIR / "denoised_audios"
CONTEXTS_CSV = DATA_DIR / "contextsvocs_groupproject_2026-04-09_104847.csv"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
SPEC_DIR    = RES_DIR / "spectrograms"
MFCC_DIR    = RES_DIR / "mfccs"
PRETRAINED_DIR = RES_DIR / "pretrained_embeddings"
SPLITS_DIR  = RES_DIR / "splits"
MODELS_DIR  = RES_DIR / "models"
RESULTS_DIR = RES_DIR / "results"
FIGURES_DIR = RES_DIR / "figures"

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
PLTOY_LABELS = {"PLTOY"}
MOVE_LABELS  = {"MOVE"}
INTER_LABELS = {"INTER"}
EXTEV_LABELS = {"EXTEV"}

CLASS_NAMES   = ["EXTEV", "INTER", "MOVE", "PLTOY"]
LABEL_ENCODER = {c: i for i, c in enumerate(CLASS_NAMES)}
LABEL_DECODER = {i: c for c, i in LABEL_ENCODER.items()}
N_CLASSES     = len(CLASS_NAMES)

LANG_FILTER = None

# ---------------------------------------------------------------------------
# Spectrogram generation parameters
# ---------------------------------------------------------------------------
SPEC_SR    = 16000
SPEC_N_MELS = 128
SPEC_HOP   = 256
SPEC_N_FFT = 1024
SPEC_FMIN  = 20
SPEC_FMAX  = 8000

# ---------------------------------------------------------------------------
# Sequence constants
# ---------------------------------------------------------------------------
F_DIM   = 128
MAX_LEN = 230

# ---------------------------------------------------------------------------
# Hyperparameter presets
# ---------------------------------------------------------------------------
HPARAM_SMALL = {
    "d_model": 32, "nhead": 4, "num_layers": 1,
    "dropout": 0.4, "lr": 1e-3, "weight_decay": 1e-4,
    "batch_size": 8, "epochs": 150, "patience": 20,
    "label_smoothing": 0.0,
}

HPARAM_FULL = {
    "d_model": 64, "nhead": 4, "num_layers": 2,
    "dropout": 0.5, "lr": 1e-3, "weight_decay": 1e-4,
    "batch_size": 32, "epochs": 100, "patience": 15,
    "label_smoothing": 0.1,
}


"""

HPARAM_FULL = {
    "d_model": 192, "nhead": 4, "num_layers": 1,
    "dropout": 0.3, "lr": 5e-4, "weight_decay": 1e-4,
    "batch_size": 16, "epochs": 100, "patience": 15,
    "label_smoothing": 0.0,   # ← match canonical wav2vec2
}
"""

HPARAM_DEFAULTS = HPARAM_FULL

# ---------------------------------------------------------------------------
# Feature-type switches
# ---------------------------------------------------------------------------
USE_MFCC = False

# Pretrained audio embeddings (wav2vec2-base). When True, loads .npy files
# from PRETRAINED_DIR and overrides F_DIM to 768 (wav2vec2 embedding dim).
USE_PRETRAINED = False  # ← flip to True when doing wav2vec2 runs

assert not (USE_MFCC and USE_PRETRAINED), \
    "Cannot use both USE_MFCC and USE_PRETRAINED — pick one."

# Override F_DIM for pretrained embeddings (768-dim wav2vec2 output)
if USE_PRETRAINED:
    F_DIM = 768

# ---------------------------------------------------------------------------
# Preprocessing flags — derive from USE_PRETRAINED so you don't have to
# remember to flip them manually.
# ---------------------------------------------------------------------------
USE_PER_CLIP_NORM = True
# SpecAugment is for spectrogram-like inputs only. Masking wav2vec2
# embeddings would zero out learned phonetic content — not useful.
USE_SPECAUGMENT = not USE_PRETRAINED

# SpecAugment hyperparameters (only used when USE_SPECAUGMENT=True)
SPECAUG_TIME_MASKS  = 2
SPECAUG_TIME_WIDTH  = 25
SPECAUG_FREQ_MASKS  = 2
SPECAUG_FREQ_WIDTH  = 12

# ---------------------------------------------------------------------------
# Metadata ablation (Model 3 only — ignored when use_external_meta=False)
# ---------------------------------------------------------------------------
# Which metadata fields to pass to the attention query. Used to isolate
# each field's contribution to test F1 without editing model or training
# code. See MetadataSpectrogramDataset in data_utils.py.
#
# Allowed values:
#   "age_gender"  : use both (canonical — matches what the README reports)
#   "age_only"    : gender is zeroed out at the dataset boundary
#   "gender_only" : age is zeroed out at the dataset boundary
#   "none"        : both zeroed (same as use_external_meta=False)
#
# Flip this while running the ablation, then flip back to "age_gender"
# for your canonical runs. The flag is read at dataset-construction time.
# ---------------------------------------------------------------------------
METADATA_MODE = "age_gender"
#METADATA_MODE = "age_only" 
#METADATA_MODE = "gender_only" 



# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

if USE_PRETRAINED:
    # wav2vec2 grid — capacity-focused
    HPARAM_GRID = [
        {"d_model": 64,  "num_layers": 1, "dropout": 0.3, "lr": 1e-3, "batch_size": 16},
        {"d_model": 128, "num_layers": 1, "dropout": 0.3, "lr": 1e-3, "batch_size": 16},
        {"d_model": 128, "num_layers": 1, "dropout": 0.5, "lr": 1e-3, "batch_size": 16},
        {"d_model": 128, "num_layers": 2, "dropout": 0.5, "lr": 5e-4, "batch_size": 16},
        {"d_model": 192, "num_layers": 1, "dropout": 0.3, "lr": 5e-4, "batch_size": 16},
        {"d_model": 192, "num_layers": 1, "dropout": 0.5, "lr": 5e-4, "batch_size": 16},
        {"d_model": 192, "num_layers": 2, "dropout": 0.5, "lr": 5e-4, "batch_size": 16},
        {"d_model": 256, "num_layers": 1, "dropout": 0.5, "lr": 5e-4, "batch_size": 16},
    ]
else:
    # Mel grid — regularization-focused mini-grid for hyperparameter tuning
    HPARAM_GRID = [
        {"d_model": 64, "num_layers": 2, "dropout": 0.5,
         "lr": 1e-3, "batch_size": 32,
         "weight_decay": 1e-4, "label_smoothing": 0.0},
        {"d_model": 64, "num_layers": 2, "dropout": 0.5,
         "lr": 1e-3, "batch_size": 32,
         "weight_decay": 1e-4, "label_smoothing": 0.1},
        {"d_model": 64, "num_layers": 2, "dropout": 0.5,
         "lr": 1e-3, "batch_size": 32,
         "weight_decay": 5e-4, "label_smoothing": 0.1},
        {"d_model": 64, "num_layers": 2, "dropout": 0.6,
         "lr": 1e-3, "batch_size": 32,
         "weight_decay": 1e-4, "label_smoothing": 0.1},
        {"d_model": 64, "num_layers": 2, "dropout": 0.7,
         "lr": 1e-3, "batch_size": 32,
         "weight_decay": 5e-4, "label_smoothing": 0.1},
        {"d_model": 32, "num_layers": 1, "dropout": 0.5,
         "lr": 1e-3, "batch_size": 32,
         "weight_decay": 1e-4, "label_smoothing": 0.1},
    ]

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_output_dirs():
    for d in [SPEC_DIR, MFCC_DIR, PRETRAINED_DIR, SPLITS_DIR,
              MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
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
    feat_src = ("wav2vec2 pretrained .npy" if USE_PRETRAINED else
                "MFCC .npy" if USE_MFCC else "mel spectrogram PNG")
    feat_desc = ("wav2vec2-base (768-dim, last layer)" if USE_PRETRAINED else
                 "MFCC (40+Δ+ΔΔ → 120, padded to 128)" if USE_MFCC else
                 "mel spectrogram (128 bins)")
    print(f"F_DIM / MAX_LEN : {F_DIM} / {MAX_LEN}  (input: {feat_src})")
    print(f"Features        : {feat_desc}")
    print(f"Preprocessing   : norm={USE_PER_CLIP_NORM}, "
          f"SpecAugment={USE_SPECAUGMENT}")
    print(f"Metadata mode   : {METADATA_MODE}")
    print(f"Grid size       : {len(HPARAM_GRID)} configs")
    print("=" * 62)