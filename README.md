# aml2026-group-05 — Child Vocalisation Context Classification

This project classifies short child vocalisation clips into four context classes:

- `EXTEV` — External event
- `INTER` — Social interaction
- `MOVE` — Movement activity
- `PLTOY` — Playing with toys

We compare three models:

1. **Model 1:** Logistic Regression on summary audio features
2. **Model 2:** Audio-only Transformer with mean pooling
3. **Model 3:** Transformer with metadata-query attentive pooling

The main result is that **Logistic Regression on mel summary statistics**, improved with Optuna, achieves the best test macro-F1.

---

## Project Structure

```text
configs/
└── model/
    ├── attn_mel.yaml
    ├── attn_wav2vec2.yaml
    ├── audio_mel.yaml
    └── default.yaml

dashboard/
├── app.py                  # Streamlit dashboard
└── methodology_tab.py      # Methodology and architecture diagrams

experiments/
├── chance_baseline.py
├── class_by_language.py
├── cm_comparison.py
├── ensemble_seeds.py
├── error_analysis.py
├── loco_model1_mel.py
├── optuna_lr.py
└── optuna_search.py

pretrained_embeddings/
└── extract_wav2vec2_embeddings.ipynb

scripts/
├── audio_to_mfccs.py
├── train_hydra.py
├── train_lr_baseline.py
├── train_multiseed.py
├── train_transformer_attn.py
└── train_transformer_audio.py

src/
├── config.py               # Paths and global config
├── data_utils.py           # Dataset, splits, preprocessing
├── models.py               # Transformer model definitions
└── train_utils.py          # Training and evaluation helpers

utils/
├── attention_analysis.py
├── attention_analysis_v2.py
├── extract_interesting_clips.py
├── generate_heatmaps.py
├── generate_spectrograms.py
└── make_presentation_figures.py

PROBLEMSETTING.md
README.md
requirements.txt
```

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure the data paths in `src/config.py` point to the correct local folders.

Expected data/results layout:

```text
DATA_DIR/
├── denoised_audios/
├── contextsvocs_groupproject_2026-04-09_104847.csv
└── MFCC / metadata files

RES_DIR/
├── spectrograms/
├── pretrained_embeddings/
├── splits/
├── models/
├── results/
└── figures/
```

---

## Generate Features

Generate mel spectrograms:

```bash
python utils/generate_spectrograms.py
```

Generate MFCC features:

```bash
python scripts/audio_to_mfccs.py
```

Generate wav2vec2 embeddings:

```bash
jupyter notebook pretrained_embeddings/extract_wav2vec2_embeddings.ipynb
```

---

## Train Models

### Model 1 — Logistic Regression

```bash
python scripts/train_lr_baseline.py
```

### Model 2 — Mean-Pool Transformer

Run grid search:

```bash
python scripts/train_transformer_audio.py --grid_search --use_existing_splits
```

Run multi-seed training:

```bash
python scripts/train_multiseed.py --model audio --seeds 42 123 456 789 2024 --use_existing_splits
```

### Model 3 — Attentive Transformer

Run grid search:

```bash
python scripts/train_transformer_attn.py --grid_search --use_existing_splits
```

Run multi-seed training:

```bash
python scripts/train_multiseed.py --model attn --seeds 42 123 456 789 2024 --use_existing_splits
```

---

## Optuna Hyperparameter Search

Run Optuna for Logistic Regression:

```bash
python experiments/optuna_lr.py
```

Run Optuna for transformer models:

```bash
python experiments/optuna_search.py
```

Optuna is the main hyperparameter-tuning extension. It improves LR mel from the canonical result to the best overall result.

---

## Analysis Scripts

Run diagnostic experiments:

```bash
python experiments/chance_baseline.py
python experiments/class_by_language.py
python experiments/cm_comparison.py
python experiments/error_analysis.py
python experiments/loco_model1_mel.py
```

Run attention analysis:

```bash
python utils/attention_analysis.py
python utils/attention_analysis_v2.py
```

---

## Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run dashboard/app.py 
```

The dashboard shows:

- headline model comparison
- per-class results
- confusion matrices
- Optuna results
- model architecture diagrams
- error-analysis views
- methodology overview

---

## Main Result

```text
Model 1 — Logistic Regression on mel summary statistics
Optuna C = 0.00492
Test macro-F1 = 0.311
```

The key conclusion is that, for this small participant-disjoint dataset, a simple linear model on robust summary statistics outperforms larger transformer models. The transformer results suggest that the main bottleneck is the dataset split, speaker/language confounding, and data ceiling rather than model capacity.
