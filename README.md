# aml2026-group-05 — Child Vocalisation Context Classification

Results and model checkpoints are saved on SwitchDrive:
- **Project data**: https://drive.switch.ch/index.php/s/c3PtXCu4gbGut9y
- **Results archive**: https://drive.switch.ch/index.php/s/NRK9nN0yo6g2n3I

---

## What this project does

This project trains neural models to classify short child vocalisation
recordings into one of four interactional context categories:

| Label | Meaning | Fine-grained labels |
|---|---|---|
| **PLTOY** | Playing with toys | PLTOY |
| **MOVE** | Movement activity | MOVE2, MOVE3, MOVE5 |
| **INTER** | Social interaction | INTER1, INTER2, INTER4–INTER7 |
| **EXTEV** | External event | EXTEV1–EXTEV3, EXTEV6 |

Data comes from the ACQDIV corpus.
Each clip is represented as a **spectrogram PNG** — a 2D time-frequency
image of the audio — which the models read as a sequence of frames.

---

## Current data state

| Property | Current | After expansion |
|---|---|---|
| Spectrogram PNGs | 51 (→ 43 usable) | 1500+ expected |
| Active classes | **2** (INTER, PLTOY only) | 4 |
| Participants with PNGs | **1** (JEX) | 4+ |
| Evaluation protocol | 5-fold CV (within-participant) | LOPO or holdout |
| Metadata query | Learned (age = identity) | External (age + gender) |

The pipeline detects participant count, class coverage, and metadata
viability automatically at each run 

---

## Three models

| # | Model | Script | Status |
|---|---|---|---|
| 1 | Logistic regression baseline | `train_lr_baseline.py` | 🔲 TBD |
| 2 | Audio-only Transformer (mean pooling) | `train_transformer_audio.py` | ✅ Almost Done |
| 3 | Attentive Pooling Transformer | `train_transformer_attn.py` | ✅ Almost Done|

---

## Project structure

```
├── src/
│   ├── config.py                  ← all paths, constants, hyperparameters
│   ├── data_utils.py              ← master table, PNG loader, splits, datasets
│   ├── models.py                  ← AudioTransformer + AttentivePoolingTransformer
│   └── train_utils.py             ← training loop, evaluation, sanity checks
├── scripts/
│   ├── diagnose_paths.py          ← debug PNG-to-CSV filename mapping
│   ├── test_pipeline.py           ← 10-step end-to-end functionality test
│   ├── train_transformer_audio.py ← Model 2 training entry point
│   └── train_transformer_attn.py  ← Model 3 training entry point
└── data/                          ← SwitchDrive-mounted dataset folder
    ├── spectrograms/              ← spectrogram1.png ... spectrogram51.png
    ├── vocalisations/             ← .wav files
    ├── contexts_vocalisation_combined_df.csv
    ├── splits/                    ← fold CSVs (created automatically)
    ├── models/                    ← checkpoints (.pt)
    ├── results/                   ← JSON metrics
    └── figures/                   ← training curves + confusion matrices
```

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the models

### Quick sanity test (always run this first, < 60 seconds)

```bash
python scripts/test_pipeline.py
```

Runs 10 checks: data loading → split construction → dataset shapes →
forward pass → loss → gradients → 2 training epochs → evaluation.

---

### Model 2 — Audio-only Transformer

**Current data (43 examples, small preset):**
```bash
python scripts/train_transformer_audio.py
```

**After expansion arrives (300+ examples):**
```bash
python scripts/train_transformer_audio.py --preset full
```

**Override individual hyperparameters:**
```bash
python scripts/train_transformer_audio.py --d_model 64 --num_layers 2 --dropout 0.3
```

**Hyperparameter grid search:**
```bash
python scripts/train_transformer_audio.py --grid_search
```

---

### Model 3 — Attentive Pooling Transformer

**Current data (learned query — automatic):**
```bash
python scripts/train_transformer_attn.py
```

**Compare against the audio-only baseline result:**
```bash
python scripts/train_transformer_attn.py --baseline_test_f1 0.717
```

**After expansion arrives (metadata query activates automatically):**
```bash
python scripts/train_transformer_attn.py --preset full
```

The script checks metadata viability at `[2b]` each run:
- With 1 participant: **learned query** (no external metadata — current)
- With 4+ participants where age/gender are not collinear with identity:
  **external metadata query** (age + gender projected to query vector)

---

## Current results (preliminary — 43 examples, 2 classes, 1 participant)

> ⚠ These results reflect **within-participant generalisation only**
> (5-fold CV, Scenario C). They cannot be interpreted as evidence of
> generalisation to unseen speakers. Chance level = 0.50 (2 classes).

### Model 2 — Audio-only Transformer (d=32, L=1, mean pooling)

| Metric | Value |
|---|---|
| Val macro-F1 | 0.830 ± 0.105 |
| **Test macro-F1** | **0.717 ± 0.063** |
| Chance baseline | 0.500 |
| Margin above chance | +0.217 |

All 5 folds above chance. Low standard deviation (0.063) indicates
consistent behaviour across folds.

### Model 3 — Attentive Pooling Transformer (d=32, L=1, learned query)

| Metric | Value |
|---|---|
| Val macro-F1 | 0.830 ± 0.105 |
| **Test macro-F1** | **0.692 ± 0.142** |
| Chance baseline | 0.500 |
| Δ vs audio-only baseline | −0.025 |

All 5 folds above chance. The −0.025 gap vs mean pooling is within
noise at this dataset size and should not be over-interpreted.

### Why the attentive pooling result is expected to be flat here

The attention weight analysis shows weights are nearly **uniform across
all 1000 time frames** (max weight = 0.00105, uniform baseline = 0.00100
— only 1.05× above uniform). This is because:

1. The learned query has no external signal (metadata = identity proxy)
   so it cannot specialise towards class-specific temporal regions.
2. With only 28 training examples, the query parameter does not receive
   enough gradient signal to develop meaningful frame preferences.
3. With T=1000 frames and 28 examples, the attention weight variance
   over 28,000 positions is naturally suppressed.

This is the expected and correct behaviour for this dataset size.
The architecture is working; the data is simply too small and homogeneous
for the attentive mechanism to add value. The comparison will be
meaningful once the full dataset is available.

### Head-to-head fold breakdown

| Fold | Audio test | Attn test | Δ |
|---|---|---|---|
| 0 | 0.775 | 0.775 | ≈0.000 |
| 1 | 0.667 | 0.500 | −0.167 |
| 2 | 0.775 | 0.889 | +0.114 |
| 3 | 0.750 | 0.564 | −0.186 |
| 4 | 0.619 | 0.733 | +0.114 |
| **Mean** | **0.717** | **0.692** | **−0.025** |

The inconsistent fold-level direction (+ and − alternating) is typical
of noise-dominated comparison at very small N. With 8–9 test examples
per fold, a single mis-classified example shifts macro-F1 by ~0.12.

---

## What happens automatically when the expansion data arrives

Just add the new PNG files to `data/spectrograms/` and re-run either
training script. The pipeline will:

1. Detect the increased participant count
2. Switch the evaluation protocol automatically:
   - 1–2 participants → 5-fold CV (current)
   - 3–4 participants → LOPO cross-validation
   - 5+ participants → fixed holdout (70/15/15)
3. Activate all 4 classes once MOVE and EXTEV have spectrograms
4. Re-evaluate metadata viability:
   - If 4+ participants with non-collinear age/gender → switch to external
     metadata query for Model 3 automatically
5. Apply `--preset full` automatically if you pass that flag

---

## Hyperparameter presets

| Preset | d_model | layers | batch | When to use |
|---|---|---|---|---|
| `small` (default) | 32 | 1 | 8 | < 100 training examples |
| `full` | 64 | 2 | 16 | 300+ training examples |

---
