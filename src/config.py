"""
config.py
---------
Single source of truth for all paths, constants, and hyperparameters.

RenkuLab path layout
---------------------
    /home/renku/work/
    └── aml-project/          ← DATA_DIR
        ├── spectrograms/
        ├── vocalisations/
        ├── contexts_vocalisation_combined_df.csv
        ├── splits/            ← created automatically
        ├── models/            ← created automatically
        ├── results/           ← created automatically
        └── figures/           ← created automatically
"""

from pathlib import Path


DATA_DIR = Path("/home/renku/work/aml-project")
RES_DIR = Path("/home/renku/work/data")
# ---------------------------------------------------------------------------
# Input paths  (derived from DATA_DIR)
# ---------------------------------------------------------------------------
SPEC_DIR     = DATA_DIR / "spectrograms"
WAV_DIR      = DATA_DIR / "vocalisations"
CONTEXTS_CSV = DATA_DIR / "contexts_vocalisation_combined_df.csv"

# ---------------------------------------------------------------------------
# Output paths  — all inside DATA_DIR so they sync back to SwitchDrive
# ---------------------------------------------------------------------------
SPLITS_DIR  = RES_DIR / "splits"
MODELS_DIR  = RES_DIR / "models"
RESULTS_DIR = RES_DIR / "results"
FIGURES_DIR = RES_DIR / "figures"

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
MOVE_LABELS  = {"MOVE2", "MOVE3", "MOVE5"}
INTER_LABELS = {"INTER1", "INTER2", "INTER4", "INTER5", "INTER6", "INTER7"}
EXTEV_LABELS = {"EXTEV1", "EXTEV2", "EXTEV3", "EXTEV6"}
PLTOY_LABELS = {"PLTOY"}

CLASS_NAMES   = ["EXTEV", "INTER", "MOVE", "PLTOY"]
LABEL_ENCODER = {c: i for i, c in enumerate(CLASS_NAMES)}
LABEL_DECODER = {i: c for c, i in LABEL_ENCODER.items()}
N_CLASSES     = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Language filter
# ---------------------------------------------------------------------------
LANG_FILTER = None   # 'shp' = Shipibo-Conibo only; None = all languages

# ---------------------------------------------------------------------------
# Spectrogram constants (set dynamically after audit)
# ---------------------------------------------------------------------------
F_DIM   = None
MAX_LEN = 150    # safe fallback; overridden by audit_spectrograms()

# ---------------------------------------------------------------------------
# Hyperparameter presets
#
# HPARAM_DEFAULTS  — used for a single training run (python train_...py)
#
# Two named presets:
#
#   HPARAM_SMALL  (current default)
#     Appropriate for < 100 training examples.
#     ~26k parameters — stable training, low overfitting risk.
#     Switch to HPARAM_FULL once the 1500+ spectrograms arrive.
#
#   HPARAM_FULL
#     Appropriate for 300+ training examples.
#     ~193k parameters — more expressive but needs more data to regularise.
#     To use: python scripts/train_transformer_audio.py --preset full
#             or: python scripts/train_transformer_audio.py --d_model 64 --num_layers 2
# ---------------------------------------------------------------------------

HPARAM_SMALL = {
    "d_model"     : 32,    # << reduced from 64
    "nhead"       : 4,
    "num_layers"  : 1,     # << reduced from 2
    "dropout"     : 0.4,   # << slightly higher dropout for small N
    "lr"          : 1e-3,
    "weight_decay": 1e-4,
    "batch_size"  : 8,     # << smaller batch so optimizer sees more updates
    "epochs"      : 150,   # << more epochs to compensate for smaller model
    "patience"    : 20,    # << more patience for noisy val signal
}

HPARAM_FULL = {
    "d_model"     : 64,
    "nhead"       : 4,
    "num_layers"  : 2,
    "dropout"     : 0.3,
    "lr"          : 1e-3,
    "weight_decay": 1e-4,
    "batch_size"  : 16,
    "epochs"      : 100,
    "patience"    : 15,
}

# Active default — uses SMALL until expansion data arrives
HPARAM_DEFAULTS = HPARAM_SMALL

# ---------------------------------------------------------------------------
# Hyperparameter grid for grid search (--grid_search flag)
# Covers both small and full model sizes so the search works regardless
# of which stage you are at.
# ---------------------------------------------------------------------------
HPARAM_GRID = [
    # Small configs — appropriate now
    {"d_model": 32, "num_layers": 1, "dropout": 0.3, "lr": 1e-3,  "batch_size": 8},
    {"d_model": 32, "num_layers": 1, "dropout": 0.5, "lr": 1e-3,  "batch_size": 8},
    {"d_model": 32, "num_layers": 1, "dropout": 0.4, "lr": 5e-4,  "batch_size": 8},
    {"d_model": 32, "num_layers": 2, "dropout": 0.5, "lr": 5e-4,  "batch_size": 8},
    # Full configs — use once expansion data arrives
    {"d_model": 64, "num_layers": 2, "dropout": 0.3, "lr": 1e-3,  "batch_size": 16},
    {"d_model": 64, "num_layers": 2, "dropout": 0.5, "lr": 1e-3,  "batch_size": 16},
    {"d_model": 64, "num_layers": 2, "dropout": 0.3, "lr": 5e-4,  "batch_size": 16},
    {"d_model": 128,"num_layers": 2, "dropout": 0.5, "lr": 5e-4,  "batch_size": 16},
]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_output_dirs():
    for d in [SPLITS_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def print_config():
    print("=" * 55)
    print("PROJECT CONFIGURATION")
    print("=" * 55)
    print(f"Data dir     : {DATA_DIR}  (exists={DATA_DIR.exists()})")
    print(f"Spec dir     : {SPEC_DIR}  (exists={SPEC_DIR.exists()})")
    print(f"Contexts CSV : {CONTEXTS_CSV}  (exists={CONTEXTS_CSV.exists()})")
    print(f"Results dir  : {RESULTS_DIR}")
    print(f"Models dir   : {MODELS_DIR}")
    print(f"Figures dir  : {FIGURES_DIR}")
    print(f"Classes      : {CLASS_NAMES}")
    print(f"Lang filter  : {LANG_FILTER!r}")
    print(f"Active preset: {'SMALL (d=32, L=1)' if HPARAM_DEFAULTS is HPARAM_SMALL else 'FULL (d=64, L=2)'}")
    print("=" * 55)