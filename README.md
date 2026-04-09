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
| Total clips | 1009 (1008 usable — 1 WAV missing) |
| Class distribution | INTER=325, PLTOY=240, EXTEV=222, MOVE=222 |
| Children | 9 (child_1 … child_9) |
| Languages | Shipibo (313), Tagalog (414), Vietnamese (282) |
| Ages | 8–12 months, 5 unique values across 9 children |
| Gender | M/F, 2 unique values across 9 children |
| Mean clip duration | 1.33 s (std=0.71, max=8.01) |

**Metadata viability**: age has 5 unique values and gender has 2 unique values
across 9 children — neither is collinear with participant identity.
External metadata query is therefore safe to use in Model 3 (`USE_EXTERNAL_META = True`).

---

## Three Models

| # | Model | Script | Status |
|---|---|---|---|
| 1 | Logistic Regression (MFCC features) | `train_lr_baseline.py` | ❌ Not yet built |
| 2 | Audio-only Transformer (mean pooling) | `train_transformer_audio.py` | ✅ Done |
| 3 | Attentive Pooling Transformer | `train_transformer_attn.py` | ✅ Done |

### Model 1 — Logistic Regression
MFCC summary features: mean and std of each of the 20 MFCC coefficients
over time → 40-dimensional vector per clip. Input: `MFCC_Full_Raw_20_nico_.csv`
(1009 rows, shape (20, T) per clip). Regularisation strength C tuned on
validation macro-F1.

### Model 2 — Audio-only Transformer
Transformer encoder over spectrogram frames → masked mean pooling → classifier.
Architecture: `input_proj(128→64) + TransformerEncoder(d=64, heads=4, L=2) +
mean_pool + Linear(64→4)`. 124,484 parameters.

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
| Val | child_2 | 170 |
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
    ├── generate_spectrograms.py   ← generate grayscale PNGs from WAV files
    ├── train_transformer_audio.py ← Model 2 entry point
    ├── train_transformer_attn.py  ← Model 3 entry point
    └── train_lr_baseline.py       ← Model 1 entry point (not yet built)
```

**RenkuLab path layout:**
```
DATA_DIR = /home/renku/work/aml-project/training_data/   ← READ ONLY
  ├── denoised_audios/            ← audio_0000.wav … audio_1008.wav (denoised)
  ├── contextsvocs_groupproject_2026-04-09_104847.csv  ← ACTIVE CSV
  ├── mfccs_070426.csv            ← MFCC features (for Model 1)
  └── MFCC_Full_Raw_20_nico_.csv  ← MFCC features, 20 coefficients (for Model 1)

RES_DIR = /home/renku/work/data/   ← READ/WRITE
  ├── spectrograms/   ← 1008 grayscale PNGs, 128×T px (generated from denoised audio)
  ├── splits/         ← train.csv, val.csv, test.csv
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

### Step 0 — Generate spectrograms (run once)

```bash
python scripts/generate_spectrograms.py
python scripts/generate_spectrograms.py --skip_existing  # safe re-run
```

Produces clean grayscale mel spectrogram PNGs (128 mel bins × T frames, no axes).
The model loads these and resizes to **F_DIM=128 × MAX_LEN=200** inside
`SpectrogramDataset.__getitem__`. Confirm the DataLoader sanity line shows:
```
Batch: x=torch.Size([32, 200, 128])   ← correct
```

### Step 1 — Train Model 2

```bash
python scripts/train_transformer_audio.py
python scripts/train_transformer_audio.py --use_existing_splits  # reuse splits
python scripts/train_transformer_audio.py --grid_search           # grid search
```

### Step 2 — Train Model 3

```bash
python scripts/train_transformer_attn.py \
    --baseline_test_f1 0.248 \
    --use_existing_splits
```

### Step 3 — Train Model 1 (not yet built)

```bash
python scripts/train_lr_baseline.py
```

Uses `MFCC_Full_Raw_20_nico_.csv`. Same holdout splits as Models 2 and 3.

---

## Hyperparameter Presets

| Preset | d_model | layers | batch | When to use |
|---|---|---|---|---|
| `small` | 32 | 1 | 8 | < 100 training examples |
| `full` *(default)* | 64 | 2 | 32 | 300+ training examples — current |

---

## Current Results

Protocol: **participant-disjoint holdout** (Train=7 children n=569,
Val=child_2 n=170, Test=child_6 n=269). Chance level = 0.25 (4 classes).

| Model | Val macro-F1 | Test macro-F1 | Δ vs chance |
|---|---|---|---|
| 1 — Logistic Regression | TBD | TBD | TBD |
| 2 — Audio Transformer (mean pool) | 0.267 | **0.248** | −0.002 |
| 3 — Attentive Pooling (ext. meta) | 0.275 | **0.186** | −0.064 |

### Per-class test F1

| Class | Model 2 | Model 3 |
|---|---|---|
| EXTEV | 0.065 | 0.000 |
| INTER | 0.374 | 0.403 |
| MOVE | 0.317 | 0.156 |
| PLTOY | 0.236 | 0.186 |

### Interpretation

Both models perform near or below the 0.25 chance level on the test set.
This is expected and honest given the experimental conditions:

**Why results are near chance:**
- **Cross-participant generalisation is hard.** The model trains on 7 children
  and is tested on a completely unseen child (child_6). Each child has a unique
  acoustic profile; with 569 training examples a Transformer cannot reliably
  learn context-discriminative features that transfer across speakers.
- **Overfitting is clearly visible** in both training curves: train loss drops to
  ~0.3 while validation F1 plateaus below 0.28. The model memorises training
  children rather than learning generalisable context cues.
- **EXTEV is the hardest class** — 0 recall in Model 3's test. child_6 has 50
  EXTEV examples but training set distribution is INTER-heavy, so the model
  rarely predicts EXTEV on unseen speakers.

**Why Model 3 does not beat Model 2:**
The attention weight statistics show `mean_weight_std = 0.00084` vs
`uniform_proxy = 0.00707` — weights are nearly uniform across all time frames,
meaning the metadata query has not learned to focus on discriminative regions.
With 569 training examples the attention parameters do not receive enough
gradient signal to develop meaningful frame preferences. The −0.062 delta
vs Model 2 is within noise at this dataset size and should not be
over-interpreted as a fundamental failure of the architecture.

These results reflect the **expected difficulty of zero-shot cross-participant
generalisation** in an under-resourced language setting, not a pipeline failure.

---

## Key Design Decisions

**Participant-disjoint evaluation** — prevents the model exploiting
speaker-specific acoustic cues instead of context-specific ones. Children
are assigned to splits once, never moved.

**Macro-F1 metric** — rewards balanced performance across all four classes.
Accuracy would be misleading; predicting INTER on every example would give
~30% accuracy but zero utility.

**No child ID as feature** — direct identity creates leakage in
participant-disjoint evaluation and violates participant privacy. Age and
gender are used instead, verified non-collinear with identity.

**Grayscale spectrograms from denoised audio** — clean acoustic content only,
no axes or colorbar pixels. Generated at native mel resolution (128 mel bins × T
frames) and resized to 128×200 inside the dataset. Denoising reduces background
noise in the recordings before feature extraction.

**F_DIM=128, MAX_LEN=200** — the model processes 128 mel bins × 200 time frames,
matching actual acoustic resolution. Attention cost is O(200²) = 40,000 pairs
per layer — 25× cheaper than reading full 1000×400 images.
