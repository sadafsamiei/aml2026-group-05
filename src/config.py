"""
config.py
---------
Single source of truth for all paths, constants, and hyperparameters.

RenkuLab path layout
---------------------
The 'data' dataset (linked to SwitchDrive AML_Project) is mounted as a
subfolder of your repository. Based on your session output the layout is:

    /home/renku/work/
    └── aml-project/          ← DATA_DIR  (the mounted dataset folder)
        ├── spectrograms/
        ├── vocalisations/
        ├── contexts_vocalisation_combined_df.csv
        ├── mfccs_batch1.csv
        ├── splits/            ← created automatically
        ├── models/            ← created automatically
        ├── results/           ← created automatically
        └── figures/           ← created automatically

If your layout differs, change DATA_DIR below — everything else derives from it.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# ★ ONLY CHANGE THIS if your data folder is somewhere else ★
# ---------------------------------------------------------------------------
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
# 'shp'  → Shipibo-Conibo only  (default, recommended)
# None   → keep all languages in the CSV
# ---------------------------------------------------------------------------
LANG_FILTER = "shp"

# ---------------------------------------------------------------------------
# Spectrogram / sequence constants  (set dynamically after audit)
# ---------------------------------------------------------------------------
F_DIM   = None
MAX_LEN = 150    # safe fallback; overridden by audit_spectrograms()

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
HPARAM_DEFAULTS = {
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
    print(f"Classes      : {CLASS_NAMES}")
    print(f"Lang filter  : {LANG_FILTER!r}")
    print(f"MAX_LEN      : {MAX_LEN}  (override after audit)")
    print("=" * 55)
