# aml2026-group-05 — Child Vocalisation Context Classification

Results and model checkpoints: https://drive.switch.ch/index.php/s/NRK9nN0yo6g2n3I
Project data and inputs: https://drive.switch.ch/index.php/s/c3PtXCu4gbGut9y

---

## Problem Setting

4-class supervised classification of short child vocalisation clips:

| Label | Meaning |
|---|---|
| **PLTOY** | Playing with toys |
| **MOVE** | Movement activity |
| **INTER** | Social interaction |
| **EXTEV** | External event |

Classifier: `f(x_audio, x_meta) = y` where `x_audio` is an acoustic representation,
`x_meta = [age, gender]`, and `y ∈ {EXTEV, INTER, MOVE, PLTOY}`.

---

## Dataset

| Property | Value |
|---|---|
| Total clips | 1009 |
| Class distribution | INTER=325, PLTOY=240, EXTEV=222, MOVE=222 |
| Children | 9 participants |
| Languages | Shipibo (313), Tagalog (414), Vietnamese (282) |
| Ages | 8–12 months, 5 unique values across 9 children |
| Gender | M/F, 2 unique values |
| Mean clip duration | 1.33 s (std=0.71, max=8.01) |

**Metadata viability**: 5 unique ages and 2 unique genders across 9 children —
neither is collinear with participant identity, so external metadata query is
safe to use in Model 3 (`USE_EXTERNAL_META = True`).

---

## Three Models

| # | Model | Status |
|---|---|---|
| 1 | Logistic Regression (Spectrogram + MFCC + Δ + ΔΔ features) | In progress |
| 2 | Audio-only Transformer (mean pooling) | ✅ Done |
| 3 | Attentive Pooling Transformer (metadata query) | ✅ Done |

**Model 1** — clip-level MFCC summary features (mean + std + delta + delta-delta
of 20 coefficients → feature vector per clip). `LogisticRegression(class_weight='balanced')`,
C tuned on val macro-F1.

**Model 2** — `input_proj(F→d)` → `TransformerEncoder(d, heads=4, L)` →
masked mean pool → `Linear(d→4)`. Three input feature types evaluated:
mel spectrogram, MFCC, wav2vec2.

**Model 3** — same encoder as Model 2, replaces mean pooling with
metadata-query attentive pooling. Age and gender are projected to a query
vector that attends over frame embeddings via unscaled dot product with a
learned temperature scalar `τ = exp(log_tau)` (init 0 → τ=1) before softmax.
The `/sqrt(d)` scaling used in self-attention is intentionally omitted here —
with a single query over T frames it flattens the softmax to near-uniform,
making the mechanism equivalent to mean pooling.

---

## Evaluation Protocol

**Participant-disjoint holdout** with 9 children:

| Split | Child | N |
|---|---|---|
| Test | child_6 | 269 |
| Val | child_2 | 171 |
| Train | child_1, 3, 4, 5, 7, 8, 9 | 569 |

**Metric**: macro-F1 (chance = 0.25 for 4 classes).

**Multi-seed reporting**: all Transformer results are mean ± std across 5 seeds
(42, 123, 456, 789, 2024). Single-seed comparisons on n_test=269 are dominated
by sampling noise (σ_seed ≈ 0.02–0.03) and are not used to draw conclusions.

---

## Hyperparameter Selection

Tuned via grid search over 8 configurations on val macro-F1 (preprocessing ON:
`USE_PER_CLIP_NORM=True`, `USE_SPECAUGMENT=True`).

### Model 2 — best config: d=64, L=2, dropout=0.5, lr=1e-3

| d_model | layers | dropout | lr | Val F1 |
|---|---|---|---|---|
| **64** | **2** | **0.5** | **1e-3** | **0.306** ← selected |
| 64 | 2 | 0.3 | 5e-4 | 0.302 |
| 128 | 2 | 0.3 | 5e-4 | 0.300 |
| 64 | 2 | 0.3 | 1e-3 | 0.300 |
| 32 | 1 | 0.3 | 1e-3 | 0.280 |
| 128 | 3 | 0.3 | 5e-4 | 0.280 |
| 32 | 1 | 0.5 | 1e-3 | 0.279 |
| 128 | 3 | 0.5 | 5e-4 | 0.266 |

Top 4 configs are within 0.006 of each other — all d=64, L=2. Higher dropout
(0.5 vs 0.3) gives a marginal regularisation edge.

### Model 3 — best config: d=32, L=1, dropout=0.3, lr=1e-3

| d_model | layers | dropout | lr | Val F1 |
|---|---|---|---|---|
| **32** | **1** | **0.3** | **1e-3** | **0.286** ← selected |
| 128 | 3 | 0.3 | 5e-4 | 0.254 |
| 64 | 2 | 0.5 | 1e-3 | 0.250 |
| 64 | 2 | 0.3 | 1e-3 | 0.245 |
| 64 | 2 | 0.3 | 5e-4 | 0.229 |
| 32 | 1 | 0.5 | 1e-3 | 0.222 |
| 128 | 2 | 0.3 | 5e-4 | 0.219 |
| 128 | 3 | 0.5 | 5e-4 | 0.218 |

The smallest model (d=32, L=1, ~26K params) won by +0.032. With attentive
pooling the attention mechanism itself adds capacity, so the encoder should
be kept minimal to avoid overfitting.

---

## Running the Models

```bash
# Generate spectrograms (run once)
python scripts/generate_spectrograms.py

# Grid search — selects best config on val F1
python scripts/train_transformer_audio.py --grid_search --use_existing_splits
python scripts/train_transformer_attn.py  --grid_search --use_existing_splits

# Multi-seed final runs with grid-selected configs
python scripts/train_multiseed.py --model audio --seeds 42 123 456 789 2024 \
    --use_existing_splits \
    --d_model 64 --num_layers 2 --dropout 0.5 --lr 0.001 --batch_size 32

python scripts/train_multiseed.py --model attn --seeds 42 123 456 789 2024 \
    --use_existing_splits \
    --d_model 32 --num_layers 1 --dropout 0.3 --lr 0.001 --batch_size 16
```

---

## Results

**Protocol**: participant-disjoint holdout. All Transformer numbers are
mean ± std across 5 seeds. Chance = 0.25.

### Full comparison across all feature types

| Features | Model 2 Test F1 | Model 3 Test F1 |
|---|---|---|
| Mel spectrogram | 0.249 ± 0.020 | 0.221 ± 0.013 |
| MFCC + Δ + ΔΔ | 0.238 ± 0.039 | 0.237 ± 0.026 |
| **wav2vec2** | **0.279 ± 0.028** | 0.221 ± 0.018 |
| Chance | 0.250 | 0.250 |

### Final reported numbers (grid-selected hyperparameters, mel spectrogram input)

| Model | Config | Params | Val F1 | Test F1 |
|---|---|---|---|---|
| 1 — Logistic Regression | MFCC + Δ + ΔΔ | — | — | 0.238 ± 0.039 |
| **2 — Audio Transformer** | d=64, L=2, drop=0.5 | 126K | 0.270 ± 0.022 | 0.249 ± 0.020 |
| **3 — Attentive Pooling** | d=32, L=1, drop=0.3 | 26K | 0.252 ± 0.045 | 0.221 ± 0.013 |
| Chance | — | — | 0.250 | 0.250 |

### wav2vec2 per-seed breakdown (Model 2)

| Seed | Test F1 |
|---|---|
| 42 | 0.304 |
| 123 | 0.297 |
| 456 | 0.262 |
| 789 | 0.293 |
| 2024 | 0.238 |
| **Mean ± Std** | **0.279 ± 0.028** |

### wav2vec2 per-seed breakdown (Model 3)

| Seed | Test F1 |
|---|---|
| 42 | 0.207 |
| 123 | 0.228 |
| 456 | 0.198 |
| 789 | 0.232 |
| 2024 | 0.241 |
| **Mean ± Std** | **0.221 ± 0.018** |

### Per-class test F1 (mel spectrogram, grid-selected configs, 5 seeds)

| Class | Model 2 | Model 3 |
|---|---|---|
| EXTEV | 0.194 ± 0.072 | 0.147 ± 0.043 |
| INTER | 0.401 ± 0.021 | 0.318 ± 0.044 |
| MOVE | 0.220 ± 0.074 | 0.093 ± 0.045 |
| PLTOY | 0.183 ± 0.084 | 0.326 ± 0.036 |

INTER is the strongest class for both models. EXTEV and MOVE are the weakest.

### Preprocessing ablation (FULL preset: d=64, L=2, drop=0.3, mel spectrogram)

| Model | Preprocessing | Val F1 | Test F1 | Val→Test gap |
|---|---|---|---|---|
| 2 | raw | 0.224 ± 0.008 | 0.231 ± 0.027 | −0.003 |
| 2 | norm + SpecAugment | 0.288 ± 0.017 | 0.228 ± 0.007 | +0.060 |
| 3 | raw | 0.227 ± 0.022 | 0.224 ± 0.027 | +0.003 |
| 3 | norm + SpecAugment | 0.239 ± 0.028 | 0.213 ± 0.025 | +0.026 |

Preprocessing lifts val F1 but not test F1. The val→test gap widens from ≈0 to
+0.06 for Model 2 — a clear signal that inter-child variance, not feature
quality, is the dominant bottleneck.

---

## Interpretation

### The project narrative in four steps

**Step 1 — From-scratch features: both models at chance, indistinguishable.**
With mel spectrograms or MFCCs, Models 2 and 3 both land near 0.22–0.25. The
difference between them (|Δ| ≈ 0.013–0.028) is within the cross-seed noise
(σ ≈ 0.02–0.03). No from-scratch feature type allows either architecture to
reliably exceed the 0.25 chance level.

**Step 2 — Pretrained features partially solve the speaker-variance bottleneck.**
wav2vec2 embeddings lift Model 2 to 0.279 ± 0.028 — the best result in the
project and the only configuration that consistently exceeds chance across
most seeds (three of five seeds reach 0.29+, seed 42 reaches 0.304). The
+0.030 lift over mel spectrograms indicates that pretrained representations
encode more transferable acoustic patterns than in-domain training from scratch.

**Step 3 — Pretrained features expose a second bottleneck: Model 3 fails.**
Model 3 with wav2vec2 achieves only 0.221 ± 0.018 — identical to its mel
spectrogram result and well below Model 2. With wav2vec2 the compression ratio
through the input projection is 768→32 (24:1), compared to 128→32 (4:1) for
mel spectrograms. The 24:1 bottleneck destroys too much information before the
attentive pooling mechanism operates. Model 3 with wav2vec2 typically
early-stops at epoch 2–3 with val F1 of 0.16–0.19, often scoring 0.000 on
entire classes. The architecture that won the grid search at 128-dim input
does not scale to 768-dim input.

**Step 4 — Architecture matters only when features are good enough.**
The architectural question (does metadata-query attentive pooling help beyond
mean pooling?) was unresolvable with from-scratch features — both models were
indistinguishable. Pretrained features finally separate them: Model 2 > Model 3
by +0.058 with wav2vec2. This is a meaningful finding. The attentive pooling
mechanism cannot exploit strong input features when the encoder bottleneck
discards them first.

### Why results are near chance overall

The combination of 7 training children, 1 held-out test child, and ~570
training examples is too restrictive for reliable cross-participant
generalisation regardless of architecture or features. Each child has a unique
acoustic profile; the model cannot reliably distinguish context-specific from
speaker-specific patterns with so few training speakers. The classic overfitting
signature appears in every training curve: train loss drops to ~0.05–0.15 while
val F1 plateaus near chance.

### Why EXTEV and MOVE are the hardest classes

EXTEV covers reactions to diverse external stimuli with no consistent acoustic
signature from the child. MOVE clips often contain movement artefacts but weak
vocalisation. Neither class has a stable acoustic pattern that survives the
speaker shift from training to test.

---

## Project Structure

```
aml2026-group-05/
├── src/
│   ├── config.py          ← paths, hyperparameters, preprocessing flags
│   ├── data_utils.py      ← splits, datasets, per-clip norm, SpecAugment
│   ├── models.py          ← AudioTransformer + AttentivePoolingTransformer
│   └── train_utils.py     ← training loop, evaluation, sanity checks
└── scripts/
    ├── generate_spectrograms.py
    ├── train_transformer_audio.py  ← Model 2 + grid search
    ├── train_transformer_attn.py   ← Model 3 + grid search
    ├── train_multiseed.py          ← multi-seed wrapper
    └── train_lr_baseline.py        ← Model 1
```

**RenkuLab path layout:**
```
DATA_DIR = /home/renku/work/aml-project/training_data/   ← READ ONLY
  ├── denoised_audios/            ← 1009 denoised WAV files
  ├── contextsvocs_groupproject_2026-04-09_104847.csv
  ├── MFCC_Full_Raw_20_nico_.csv
  └── mfccs_070426.csv

RES_DIR = /home/renku/work/data/   ← READ/WRITE
  ├── spectrograms/   ← 1009 grayscale PNGs, 128 mel bins × T frames
  ├── splits/         ← train.csv (569), val.csv (171), test.csv (269)
  ├── models/         ← checkpoints (.pt)
  ├── results/        ← JSON metrics
  └── figures/        ← training curves, confusion matrices
```

---

## Key Design Decisions

**Participant-disjoint evaluation** — the two largest, most class-balanced
children (child_6 n=269, child_2 n=171) are assigned to test and val. This
yields a 56/17/27 clip ratio rather than 70/15/15, a necessary deviation given
9 unequal-sized participants. Any alternative assignment either misses classes
in held-out splits or places more children in test than the dataset can support.

**Macro-F1** — equally penalises poor recall on any class. Predicting INTER
for everything gives ~30% accuracy but near-zero macro-F1.

**No child ID as feature** — direct identity causes leakage and violates
participant privacy. Age and gender are used instead and verified non-collinear.

**Multi-seed reporting (n=5)** — required because σ_seed ≈ 0.02–0.03 on
n_test=269, meaning a single seed comparison can easily show the wrong model
winning by chance.

**Unscaled dot product in Model 3** — the `/sqrt(d)` convention from
multi-head self-attention is removed. With one query over T=230 frames it
divides scores by ~8, collapsing the softmax to near-uniform
(measured mean_weight_std ≈ 0.00086 vs uniform proxy 0.00659). Removing it
restores usable dynamic range. A learned temperature scalar τ=exp(log_tau)
(one parameter, init τ=1) provides end-to-end control over sharpness.

**Preprocessing as opt-in ablation** — `USE_PER_CLIP_NORM` and
`USE_SPECAUGMENT` are flags in `config.py`. The ablation shows preprocessing
helps val but not test, demonstrating that inter-child variance is the ceiling.
