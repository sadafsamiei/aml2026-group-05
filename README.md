# aml2026-group-05 — Child Vocalisation Context Classification

Results and model checkpoints are saved on SwitchDrive:
- **Project data & inputs**: https://drive.switch.ch/index.php/s/c3PtXCu4gbGut9y
- **Results archive**: https://drive.switch.ch/index.php/s/NRK9nN0yo6g2n3I

---

## Problem Setting

We formulate the task as a **4-class supervised classification** problem. Each
example is a short child vocalisation clip with an associated context label and
limited metadata. The goal is to learn a classifier:

    f(x_audio, x_meta) = y

where `x_audio` is an acoustic representation (mel spectrogram), `x_meta` is a
metadata vector (age, gender), and `y` is the target context label.

### Label Space

| Label | Meaning |
|---|---|
| **PLTOY** | Playing with toys |
| **MOVE** | Movement activity |
| **INTER** | Social interaction |
| **EXTEV** | External event |

Labels are provided directly in the dataset CSV in 4-class format — no mapping needed.

---

## Dataset

**Active CSV**: `contextsvocs_groupproject_2026-04-09_104847.csv`

| Property | Value |
|---|---|
| Total clips | 1009 |
| Class distribution | INTER=325, PLTOY=240, EXTEV=222, MOVE=222 |
| Children | 9 (child_1 … child_9) |
| Languages | Shipibo (313), Tagalog (414), Vietnamese (282) |
| Ages | 8–12 months, 5 unique values across 9 children |
| Gender | M/F, 2 unique values across 9 children |
| Mean clip duration | 1.33 s (std=0.71, max=8.01) |

**Metadata viability**: age has 5 unique values and gender has 2 unique values
across 9 children — neither is collinear with participant identity. External
metadata query is therefore safe to use in Model 3.

---

## Three Models

| # | Model | Script | Status |
|---|---|---|---|
| 1 | Logistic Regression (MFCC features) | `train_lr_baseline.py` | Not yet built |
| 2 | Audio-only Transformer (mean pooling) | `train_transformer_audio.py` | Training |
| 3 | Attentive Pooling Transformer | `train_transformer_attn.py` | Waiting for Model 2 |

### Model 1 — Logistic Regression
MFCC summary features: mean and std of each of the 20 MFCC coefficients
over time → 40-dimensional vector per clip. Input: `MFCC_Full_Raw_20_nico_.csv`
(1009 rows, shape (20, T) per clip). Regularisation strength C tuned on
validation macro-F1.

### Model 2 — Audio-only Transformer
Transformer encoder over spectrogram frames → masked mean pooling → classifier.
Architecture: `input_proj(128→64) + TransformerEncoder(d=64, heads=4, L=2) +
mean_pool + Linear(64→4)`. ~193k parameters.

### Model 3 — Attentive Pooling Transformer
Same encoder as Model 2 but replaces mean pooling with metadata-query attentive
pooling. Age and gender are projected to a query vector that attends over
frame-level representations. Query mode is determined automatically at runtime:
- **External metadata** (current, 9 participants): `USE_EXTERNAL_META = True`
- **Learned query** (fallback, fewer participants or collinear metadata)

---

## Evaluation Protocol

With 9 participants the pipeline uses **participant-disjoint holdout**:

| Split | Child | N |
|---|---|---|
| Test | child_6 | 269 |
| Val | child_2 | 171 |
| Train | child_1, 3, 4, 5, 7, 8, 9 | 569 |

Children are assigned by size so the two largest (most class-balanced) go to
test and val. Leakage checks verify no participant appears in multiple splits.

**Metric**: macro-F1 (unweighted mean of per-class F1 scores). Used instead of
accuracy because macro-F1 equally penalises poor recall on any class including
minorities. **Chance level = 0.25** (4 classes, uniform).

---

## Project Structure

```
aml2026-group-05/
├── src/
│   ├── config.py          ← all paths, constants, hyperparameters
│   ├── data_utils.py      ← master table, PNG loader, splits, datasets
│   ├── models.py          ← AudioTransformer + AttentivePoolingTransformer
│   └── train_utils.py     ← training loop, evaluation, sanity checks
└── scripts/
    ├── generate_spectrograms.py   ← generate viridis PNGs from WAV files
    ├── train_transformer_audio.py ← Model 2 entry point
    ├── train_transformer_attn.py  ← Model 3 entry point
    └── train_lr_baseline.py       ← Model 1 entry point (not yet built)
```

**RenkuLab path layout:**
```
DATA_DIR = /home/renku/work/aml-project/training_data/   ← READ ONLY
  ├── anonymized_audios/          ← audio_0000.wav … audio_1008.wav
  ├── contextsvocs_groupproject_2026-04-09_104847.csv  ← ACTIVE CSV
  ├── mfccs_070426.csv            ← MFCC features (for Model 1)
  └── MFCC_Full_Raw_20_nico_.csv  ← MFCC features, 20 coefficients (for Model 1)

RES_DIR = /home/renku/work/data/   ← READ/WRITE
  ├── spectrograms/   ← 1009 viridis PNGs, 1000x400 RGB (already generated)
  ├── splits/         ← train.csv, val.csv, test.csv (auto-created)
  ├── models/         ← checkpoints (.pt)
  ├── results/        ← JSON metrics
  └── figures/        ← training curves + confusion matrices
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Models

### Step 0 — Generate spectrograms (already done, skip if PNGs exist)

```bash
python scripts/generate_spectrograms.py
# Safe to re-run:
python scripts/generate_spectrograms.py --skip_existing
```

Produces 1000×400 px RGB viridis mel spectrograms with Hz/time/dB labels.
The model resizes these to **128×200 px** inside `SpectrogramDataset.__getitem__`
(F_DIM=128, MAX_LEN=200). Confirm with the DataLoader sanity line:
```
Batch: x=torch.Size([32, 200, 128])   ← correct
Batch: x=torch.Size([32, 1000, 400])  ← resize failed, check data_utils.py
```

### Step 1 — Train Model 2

```bash
# First run — builds splits automatically
python scripts/train_transformer_audio.py

# If re-running after a crash — reuse existing splits
python scripts/train_transformer_audio.py --use_existing_splits

# Hyperparameter overrides
python scripts/train_transformer_audio.py --d_model 128 --num_layers 3
python scripts/train_transformer_audio.py --grid_search
```

### Step 2 — Train Model 3

```bash
# Pass Model 2 test F1 for direct comparison output
python scripts/train_transformer_attn.py --baseline_test_f1 <model2_test_f1>

# Reuse splits from Model 2 run
python scripts/train_transformer_attn.py --baseline_test_f1 <value> --use_existing_splits
```

### Step 3 — Train Model 1 (not yet built)

```bash
python scripts/train_lr_baseline.py
```

Uses `MFCC_Full_Raw_20_nico_.csv`. Uses same holdout splits as Models 2 and 3.

---

## Hyperparameter Presets

| Preset | d_model | layers | batch | When to use |
|---|---|---|---|---|
| `small` | 32 | 1 | 8 | < 100 training examples |
| `full` *(default)* | 64 | 2 | 32 | 300+ training examples — current |

---

## Current Results

> Model 2 is currently training. All results below will be filled in once all
> three models complete. Protocol: participant-disjoint holdout.
> Chance level = 0.25 (4 classes).

| Model | Val macro-F1 | Test macro-F1 | Δ vs chance |
|---|---|---|---|
| 1 — Logistic Regression | TBD | TBD | TBD |
| 2 — Audio Transformer | TBD | TBD | TBD |
| 3 — Attentive Pooling Transformer | TBD | TBD | TBD |

---

## Key Design Decisions

**Participant-disjoint evaluation** — prevents the model exploiting
speaker-specific acoustic cues instead of context-specific ones. Children
are assigned to splits once, never moved.

**Macro-F1 metric** — rewards balanced performance across all four classes.
Accuracy would be misleading because predicting the majority class (INTER)
on every example would give 32% accuracy but near-zero utility.

**No child ID as feature** — direct identity creates leakage in
participant-disjoint evaluation and violates participant privacy. Age and
gender are used instead, verified non-collinear with identity.

**Resize 1000×400 PNGs to 128×200 for training** — the 1000×400 image
includes axes, tick labels, and the dB colorbar which encode no acoustic
content. Attention cost at T=1000 is 25× higher than at T=200. The resize
happens inside `SpectrogramDataset.__getitem__` to 128 mel bins × 200 time
frames, matching the actual acoustic resolution of the data. The full-size
PNGs are kept for reports and presentations.
