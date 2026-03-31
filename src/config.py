"""
config.py
---------
Single source of truth for all paths, constants, and hyperparameters.

RenkuLab path layout
---------------------
Renku mounts datasets relative to the repository root.
Your 'data' dataset (linked to SwitchDrive) is mounted at:
    <repo_root>/data/

Expected structure after mounting:
    data/
    ├── spectrograms/          ← PNG spectrograms  (6.7 MB)
    ├── vocalisations/         ← .wav files        (11.5 MB)
    ├── contexts_vocalisation_combined_df.csv
    └── mfccs_batch1.csv       ← broken, not used

Results will be written to:
    data/results/              ← created automatically
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root detection
# ---------------------------------------------------------------------------
# Works whether you run from the repo root or from scripts/
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR   = _THIS_FILE.parent
REPO_ROOT = Path("/home/renku/work")

# ---------------------------------------------------------------------------
# Data paths  (all relative to repo root so they work in RenkuLab)
# ---------------------------------------------------------------------------
DATA_DIR = REPO_ROOT / "aml-project"
RES_DIR = REPO_ROOT / "data"
SPEC_DIR        = DATA_DIR  / "spectrograms"
WAV_DIR         = DATA_DIR  / "vocalisations"
CONTEXTS_CSV    = DATA_DIR  / "contexts_vocalisation_combined_df.csv"

# Output directories (created on first run)
RESULTS_DIR     = RES_DIR  / "results"
SPLITS_DIR      = RES_DIR  / "splits"
MODELS_DIR      = RES_DIR  / "models"
FIGURES_DIR     = RES_DIR  / "figures"

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
MOVE_LABELS  = {"MOVE2", "MOVE3", "MOVE5"}
INTER_LABELS = {"INTER1", "INTER2", "INTER4", "INTER5", "INTER6", "INTER7"}
EXTEV_LABELS = {"EXTEV1", "EXTEV2", "EXTEV3", "EXTEV6"}
PLTOY_LABELS = {"PLTOY"}

# Sorted so the integer encoding is deterministic across runs
CLASS_NAMES   = ["EXTEV", "INTER", "MOVE", "PLTOY"]   # 4-class
LABEL_ENCODER = {c: i for i, c in enumerate(CLASS_NAMES)}
LABEL_DECODER = {i: c for c, i in LABEL_ENCODER.items()}
N_CLASSES     = len(CLASS_NAMES)

# Language filter
LANG_FILTER = None   # Shipibo-Conibo + other communities

# ---------------------------------------------------------------------------
# Spectrogram / feature constants
# These are filled in at runtime by audit_spectrograms() in data_utils.py.
# Hard-coded fallbacks are provided for scripts that skip the audit step.
# ---------------------------------------------------------------------------
# F_DIM:   number of frequency bins (height of each PNG)
#          — set after running audit_spectrograms()
# MAX_LEN: number of time frames to keep (width cap at 95th percentile)
#          — p95 of clip widths based on current 122-example set ≈ 150 frames
#            (exact value depends on your spectrogram hop_length)
F_DIM   = None    # set dynamically; fallback below used only if None
MAX_LEN = 150     # safe fallback; override with value from audit

# ---------------------------------------------------------------------------
# Training hyperparameters  (defaults for the functionality test)
# ---------------------------------------------------------------------------
HPARAM_DEFAULTS = {
    "d_model"    : 64,
    "nhead"      : 4,
    "num_layers" : 2,
    "dropout"    : 0.3,
    "lr"         : 1e-3,
    "weight_decay": 1e-4,
    "batch_size" : 16,
    "epochs"     : 100,
    "patience"   : 15,   # early stopping patience
}

# Hyperparameter grid for the full search (Week 5)
HPARAM_GRID = [
    {"d_model": 32,  "num_layers": 1, "dropout": 0.3, "lr": 1e-3},
    {"d_model": 32,  "num_layers": 1, "dropout": 0.5, "lr": 1e-3},
    {"d_model": 64,  "num_layers": 2, "dropout": 0.3, "lr": 1e-3},
    {"d_model": 64,  "num_layers": 2, "dropout": 0.5, "lr": 1e-3},
    {"d_model": 64,  "num_layers": 2, "dropout": 0.3, "lr": 5e-4},
    {"d_model": 64,  "num_layers": 2, "dropout": 0.5, "lr": 5e-4},
    {"d_model": 128, "num_layers": 2, "dropout": 0.5, "lr": 5e-4},
    {"d_model": 128, "num_layers": 3, "dropout": 0.5, "lr": 5e-4},
]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Helper: create all output directories
# ---------------------------------------------------------------------------
def make_output_dirs():
    for d in [RESULTS_DIR, SPLITS_DIR, MODELS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def print_config():
    print("=" * 55)
    print("PROJECT CONFIGURATION")
    print("=" * 55)
    print(f"Repo root    : {REPO_ROOT}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Spec dir     : {SPEC_DIR}")
    print(f"Contexts CSV : {CONTEXTS_CSV}")
    print(f"Results dir  : {RESULTS_DIR}")
    print(f"Classes      : {CLASS_NAMES}")
    print(f"Lang filter  : {LANG_FILTER}")
    print(f"MAX_LEN      : {MAX_LEN}  (override after audit)")
    print("=" * 55)
