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
    ├── attn_mel.yaml          # Hydra config: Model 3 on mel (d=32, L=1, dropout=0.3, lr=1e-3)
    ├── attn_wav2vec2.yaml     # Hydra config: Model 3 on wav2vec2 (d=192, L=1, dropout=0.3, lr=5e-4)
    ├── audio_mel.yaml         # Hydra config: Model 2 on mel (d=64, L=2, dropout=0.5, lr=1e-3)
    └── default.yaml           # Shared defaults (batch_size, epochs, patience, label_smoothing)

dashboard/
├── app.py                     # Streamlit dashboard entry point — 5 tabs (results, per-class, CMs, Optuna, methodology)
└── methodology_tab.py         # Methodology tab content: pipeline overview + architecture diagrams

experiments/
├── EDA/
│   └── EDA.ipynb                           # Frequency-domain EDA + band-pass denoising of raw audio.
│                                           #   Computes per-file FFT and aggregate Welch's PSD across the
│                                           #   1,009 clips, then applies a Butterworth band-pass filter
│                                           #   (HP=85 Hz, LP=6 kHz, order=5, zero-phase via sosfiltfilt).
│                                           #   Reads anonymized_audios/, writes denoised_audios/.
│                                           #   Must run once before any feature extraction.
├── pretrained_embeddings/
│   ├── extract_wav2vec2_embeddings.ipynb   # One-off notebook: extracts frozen wav2vec2-base embeddings, caches to disk
│   └── finetune_wav2vec2.py                # Controlled experiment: frozen vs top-2 / top-4 layer fine-tuning of
│                                           #   wav2vec2 (1 seed, 35 epochs). Train F1 rises to 0.636 but val-test
│                                           #   gap widens 10x (+0.006 to +0.059), confirming speaker-identity
│                                           #   overfitting. Supports the decision to use frozen feature extraction.
├── chance_baseline.py         # Computes uniform-random and stratified chance macro-F1 (4-class, 0.245 +/- 0.027)
├── class_by_language.py       # Computes class distributions per language (used for the INTER-by-language figure)
├── ensemble_seeds.py          # Ablation: averages logits across 5 seeds for M2 mel (result: -0.024 vs single-seed mean)
├── error_analysis.py          # Computes the per-clip agreement table (which clips each model gets right/wrong) — drives the 110/40.9% finding
├── loco_model1_mel.py         # Leave-One-Child-Out CV for Model 1 (LR mel) across all 9 children — sigma_speaker = 0.053
├── optuna_lr.py               # Optuna 80-trial continuous C line search for LR (extension; finds C=0.00492)
└── optuna_search.py           # Optuna 60-trial TPE search for transformers (M2/M3, d/L/lr/dropouts)

scripts/
├── audio_to_mfccs.py          # Extracts 20-coefficient MFCCs (n_fft=2048, hop=512) and caches to disk
├── train_hydra.py             # Hydra-based training runner — reads configs/model/*.yaml; supports flat overrides
├── train_lr_baseline.py       # Trains Model 1 with inner-LOPO grid search over 9-point logspace C
├── train_multiseed.py         # Wrapper that runs train_transformer_{audio,attn}.py for 5 seeds and aggregates results
├── train_transformer_attn.py  # Trains Model 3 (AttentivePoolingTransformer); --grid_search for the 8-config sweep
└── train_transformer_audio.py # Trains Model 2 (AudioTransformer mean-pool); --grid_search for the 8-config sweep

src/
├── config.py                  # Paths, HPARAM_FULL hyperparameters, USE_PRETRAINED, METADATA_MODE flags
├── data_utils.py              # Dataset classes (Spectrogram, MetadataSpectrogram), splits loader, per-clip norm, SpecAugment
├── models.py                  # AudioTransformer + AttentivePoolingTransformer (v2: 3 dropout locations for M2, 4 for M3)
└── train_utils.py             # Training loop, evaluate(), early stopping, sanity checks (forward pass, NaN guards)

utils/
├── attention_analysis.py      # v1: extracts attention weights from Model 3, computes attention concentration / mean focus span
├── attention_analysis_v2.py   # v2: same as above but loads multi-dropout v2 checkpoints (used after the Optuna extension)
├── extract_interesting_clips.py # Categorises test clips into 6 agreement buckets (all-correct, all-wrong, LR-wins, etc.) — CSV for presentation
├── generate_heatmaps.py       # Generates the per-class F1 heatmap (3 models x 4 classes x 3 feature sources)
├── generate_spectrograms.py   # Generates and caches 128-bin log-mel spectrograms as PNGs (n_fft=2048, hop=512, fmax=8kHz)
└── make_presentation_figures.py # Generates all 5 presentation figures (complexity scatter, heatmap, val-vs-test gap, INTER-by-lang, Optuna summary)

results/
├── canonical/          # 6 files  — canonical 5-seed multi-seed JSONs (Tables 1 and 2 in report)
├── optuna/             # 14 files — Optuna search JSONs and 5-seed reruns of Optuna-selected configs
└── supplementary/      # 2 files  — LOCO cross-validation JSON (sigma_speaker = 0.053) and
                        #            wav2vec2 fine-tuning experiment JSON (val-test gap analysis)

PROBLEMSETTING.md              # Problem setting adapted from the course brief to reflect the actual project methodology
README.md                      # This file
requirements.txt               # Python dependencies (torch, librosa, scikit-learn, optuna, streamlit, etc.)
project_report.pdf             # Final report
presetation_slides.key         # slides for the presentation in .key format
presentation_slides.pdf        # slides for the presentation in .pdf format
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
├── anonymized_audios/         # Raw input — source for the EDA notebook
├── denoised_audios/           # Produced by experiments/eda/EDA.ipynb; consumed by every feature extractor
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

The pipeline is: **denoise raw audio** → **extract three feature sources** → train.

### Step 0 — Denoise raw audio (run once)

```bash
jupyter notebook experiments/eda/EDA.ipynb
```

Produces `denoised_audios/`, which every subsequent feature extractor reads.
Applies a Butterworth band-pass filter (85 Hz - 6 kHz) chosen from the
aggregate frequency profile of the dataset.

### Step 1 — Generate mel spectrograms

```bash
python utils/generate_spectrograms.py
```

### Step 2 — Generate MFCC features

```bash
python scripts/audio_to_mfccs.py
```

### Step 3 — Generate wav2vec2 embeddings

```bash
jupyter notebook experiments/pretrained_embeddings/extract_wav2vec2_embeddings.ipynb
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

Run Optuna for transformer models (runs both Models 2 and 3 sequentially, ~6 hours total):

```bash
python experiments/optuna_search.py
```

Optuna is the main hyperparameter-tuning extension. It yields three genuine gains: LR mel (0.277 -> 0.311), LR wav2vec2 (0.263 -> 0.285), and M2 wav2vec2 (0.219 -> 0.251). All other combinations are within seed noise or degrade.

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

Run wav2vec2 fine-tuning experiment (writes JSON to results/supplementary/):

```bash
python experiments/pretrained_embeddings/finetune_wav2vec2.py
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
