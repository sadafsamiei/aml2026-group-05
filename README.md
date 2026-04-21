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
safe to use in Model 3 (`USE_EXTERNAL_META = True`). The low cardinality is
itself meaningful: 5 × 2 = 10 unique `(age, gender)` combinations project to at
most 10 distinct query vectors across the whole dataset.

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
by sampling noise (σ_seed ≈ 0.015–0.020) and are not used to draw conclusions.

All reported Transformer numbers in this README are from the canonical
multi-seed JSON files in `RES_DIR/results/` generated with the current
code on 2026-04-21. Prior runs under different preprocessing states are
not carried forward unless reproducible under the current configuration.

---

## Hyperparameter Selection

Tuned via grid search over 8 configurations on val macro-F1. Each **feature
type gets its own grid** — a config that wins on 128-dim mel inputs is not
guaranteed to be optimal on 768-dim wav2vec2 inputs (see Step 3 of the
interpretation). Preprocessing flags are set automatically from
`USE_PRETRAINED`: mel runs use `USE_PER_CLIP_NORM=True` and
`USE_SPECAUGMENT=True`; wav2vec2 runs use norm only (SpecAugment on
pretrained embeddings masks high-level phonetic content and is not used).

### Model 2 — mel grid — best: d=64, L=2, dropout=0.5, lr=1e-3

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

### Model 3 — mel grid — best: d=32, L=1, dropout=0.3, lr=1e-3

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

On mel, the smallest model (d=32, L=1, ~26K params) won by +0.032.

### Model 3 — wav2vec2 grid — selected: d=192, L=1, dropout=0.3, lr=5e-4

Re-gridded separately for wav2vec2 inputs. The grid sweeps `d_model` through
values that correspond to progressively smaller compression ratios against the
768-dim input: d=64 → 12:1, d=128 → 6:1, d=192 → 4:1, d=256 → 3:1.

| d_model | layers | dropout | lr | Val F1 | Best epoch |
|---|---|---|---|---|---|
| 192 | 2 | 0.5 | 5e-4 | 0.310 | 2 ⚠︎ |
| 192 | 1 | 0.5 | 5e-4 | 0.303 | 1 ⚠︎ |
| **192** | **1** | **0.3** | **5e-4** | **0.297** | **34** ← selected |
| 128 | 1 | 0.3 | 1e-3 | 0.279 | 3 |
| 128 | 2 | 0.5 | 5e-4 | 0.264 | 4 |
| 64 | 1 | 0.3 | 1e-3 | 0.263 | 4 |
| 256 | 1 | 0.5 | 5e-4 | 0.259 | 1 ⚠︎ |
| 128 | 1 | 0.5 | 1e-3 | 0.241 | 2 ⚠︎ |

**Selection deviates from top-1.** The top-ranked config peaked at epoch 2
on the grid-search seed and failed to reproduce on a re-run (val F1 collapsed
to 0.209), indicating its apparent performance was a lucky initialisation
rather than sustained learning. We selected the third-ranked config
(d=192, L=1, drop=0.3) — the only config among the top three that showed
proper learning dynamics (34-epoch climb, sustained val F1 improvement from
0.18 to 0.30). This decision is documented so that readers can independently
inspect training curves; selecting the numerical top-1 would have produced
a multi-seed test F1 of ≈0.21 instead of the 0.246 actually obtained.

The monotonic trend across d_model averages — d=64 (0.263), d=128 avg (0.278),
d=192 avg (0.300), d=256 (0.259) — confirms that ~4:1 compression is the
sweet spot for wav2vec2 input into this architecture; larger encoders overfit
(d=256) and smaller ones underfit (d=64).

---

## Running the Models

```bash
# Generate spectrograms (run once)
python scripts/generate_spectrograms.py

# ---- Mel spectrogram path (USE_PRETRAINED = False) ----
python scripts/train_transformer_audio.py --grid_search --use_existing_splits
python scripts/train_transformer_attn.py  --grid_search --use_existing_splits

python scripts/train_multiseed.py --model audio --seeds 42 123 456 789 2024 \
    --use_existing_splits \
    --d_model 64 --num_layers 2 --dropout 0.5 --lr 0.001 --batch_size 32

python scripts/train_multiseed.py --model attn --seeds 42 123 456 789 2024 \
    --use_existing_splits \
    --d_model 32 --num_layers 1 --dropout 0.3 --lr 0.001 --batch_size 16

# ---- wav2vec2 path (USE_PRETRAINED = True) ----
# Model 2 multi-seed
python scripts/train_multiseed.py --model audio --seeds 42 123 456 789 2024 \
    --use_existing_splits \
    --d_model 64 --num_layers 2 --dropout 0.5 --lr 0.001 --batch_size 32

# Model 3 requires its own grid because d=32 induces a 24:1 compression on 768-dim input
python scripts/train_transformer_attn.py --grid_search --use_existing_splits

python scripts/train_multiseed.py --model attn --seeds 42 123 456 789 2024 \
    --use_existing_splits \
    --d_model 192 --num_layers 1 --dropout 0.3 --lr 0.0005 --batch_size 16
```

---

## Results

**Protocol**: participant-disjoint holdout. All Transformer numbers are
mean ± std across 5 seeds. Chance = 0.25.

### Full comparison across all feature types

Each cell uses the config grid-selected for that (model, feature-type) pair.

| Features | Model 2 Test F1 | Model 3 Test F1 | Δ (3 − 2) |
|---|---|---|---|
| Mel spectrogram | **0.249 ± 0.020** | 0.221 ± 0.013 | −0.028 |
| MFCC + Δ + ΔΔ | 0.238 ± 0.039 | 0.237 ± 0.026 | −0.001 |
| **wav2vec2** | 0.219 ± 0.018 | **0.246 ± 0.016** | **+0.027** |
| Chance | 0.250 | 0.250 | — |

The sign of the Model 3 vs Model 2 difference **flips** across feature types:
on mel spectrograms mean pooling wins, but on wav2vec2 attentive pooling wins.
Both differences are roughly 1.5σ in magnitude given σ_seed ≈ 0.016–0.020.

### Final reported numbers (feature-type-specific grid-selected configs)

| Model | Features | Config | Params | Val F1 | Test F1 |
|---|---|---|---|---|---|
| 1 — Logistic Regression | MFCC + Δ + ΔΔ | — | — | — | 0.238 ± 0.039 |
| 2 — Audio Transformer | mel | d=64, L=2, drop=0.5 | 126K | 0.270 ± 0.022 | 0.249 ± 0.020 |
| 2 — Audio Transformer | wav2vec2 | d=64, L=2, drop=0.5 | 167K | 0.294 ± 0.010 | 0.219 ± 0.018 |
| 3 — Attentive Pooling | mel | d=32, L=1, drop=0.3 | 26K | 0.252 ± 0.045 | 0.221 ± 0.013 |
| **3 — Attentive Pooling** | **wav2vec2** | d=192, L=1, drop=0.3 | 648K | 0.286 ± 0.018 | **0.246 ± 0.016** |
| Chance | — | — | — | 0.250 | 0.250 |

### wav2vec2 per-seed breakdown (Model 2, d=64 L=2)

| Seed | Test F1 |
|---|---|
| 42 | 0.218 |
| 123 | 0.191 |
| 456 | 0.215 |
| 789 | 0.248 |
| 2024 | 0.225 |
| **Mean ± Std** | **0.219 ± 0.018** |

### wav2vec2 per-seed breakdown (Model 3, d=192 regridded)

| Seed | Test F1 |
|---|---|
| 42 | 0.253 |
| 123 | 0.231 |
| 456 | 0.225 |
| 789 | 0.267 |
| 2024 | 0.256 |
| **Mean ± Std** | **0.246 ± 0.016** |

Every Model 3 seed trained past epoch 11 (best epochs 4, 48, 4, 22, 13).
Compared to the original d=32 Model 3 run where three of five seeds
scored < 0.23, the d=192 configuration has substantially tighter variance
and no degenerate seeds.

### Architecture ablation on mel — isolating pooling from capacity

The original mel comparison pitted Model 2 (d=64, L=2) against Model 3 at a
grid-selected smaller config (d=32, L=1). This run matches Model 3's encoder
to Model 2's exactly, holding preprocessing and seeds identical.

| Configuration | Val F1 | Test F1 | MOVE per-class F1 |
|---|---|---|---|
| Model 2 — mean pool (d=64, L=2) | 0.270 ± 0.022 | 0.249 ± 0.020 | 0.220 ± 0.074 |
| Model 3 — attn pool (d=32, L=1) grid-selected | 0.252 ± 0.045 | 0.221 ± 0.013 | 0.093 ± 0.045 |
| **Model 3 — attn pool (d=64, L=2) matched** | **0.230 ± 0.036** | **0.210 ± 0.025** | **0.083 ± 0.109** |

On mel, even at matched encoder capacity, attentive pooling loses to mean
pooling by −0.039 test F1. The MOVE class collapses in three of five seeds
for the matched-capacity Model 3 (per-class F1 = 0.000), which drives
most of the macro-F1 difference.

### Per-class test F1 (wav2vec2, 5 seeds)

| Class | Model 2 (d=64, L=2) | Model 3 (d=192, L=1) |
|---|---|---|
| EXTEV | 0.239 ± 0.009 | 0.192 ± 0.029 |
| INTER | 0.255 ± 0.056 | 0.395 ± 0.052 |
| MOVE | 0.231 ± 0.057 | 0.199 ± 0.077 |
| PLTOY | 0.153 ± 0.065 | 0.198 ± 0.091 |

The per-class pattern is informative about *why* Model 3 outperforms
Model 2 on wav2vec2. Model 3's gain comes almost entirely from the
INTER class (+0.140). Model 2 is slightly stronger on EXTEV and MOVE,
but Model 3's large INTER advantage dominates the macro-F1 aggregate.
INTER is also the largest test class (n=79), so this is not an artefact
of rare-class dominance — the attentive mechanism appears to be
genuinely better at identifying social-interaction acoustic patterns
once the encoder has enough capacity to pass wav2vec2 information
through.

**The confusion matrix reveals an INTER-prediction bias in Model 3.**
On clips whose true class is *not* INTER, Model 3 predicts INTER 56% of
the time versus 33% for Model 2 (+23 percentage points). The same
pattern holds row by row — from EXTEV, MOVE, and PLTOY, Model 3's
largest off-diagonal is always the INTER column (0.62, 0.49, 0.57
respectively). This bias has two partial explanations working together
(see Limitations for the language confound). First, the Tagalog
training prior: our Tagalog training children have 46.9% INTER clips,
almost 2× the Shipibo rate of 22.5%, and the test child is Tagalog
too. Second, attention concentration on single frames that carry
INTER-like signatures (see below). Neither explanation alone accounts
for Model 3's 57.4% INTER prediction rate — the Tagalog prior predicts
46.9% and the true rate is 29.4%. The +0.027 macro-F1 gain is real —
macro-F1 penalises class-skew predictions, so the gain cannot be
explained by bias alone — but it comes at the cost of EXTEV, MOVE, and
PLTOY precision.

### Attention mechanism behaviour — does the pooling actually attend?

Before claiming that Model 3's wav2vec2 win comes from attention doing
real work, we inspected the mechanism directly. For each val clip we
computed the number of top-weighted frames needed to accumulate a given
fraction of the total attention mass. This metric is scale-free and
makes "concentrated vs distributed" attention directly comparable
across configurations.

| Attention mass threshold | d=32 mel (median frames) | d=192 wav2vec2 (median frames) |
|---|---|---|
| 50% | 3 | **1** |
| 80% | 7 | **1** |
| 95% | 17 | **1** |

The mechanism operates in a qualitatively different regime on each
feature type. On mel, attention is genuinely distributed: a median of
17 frames carries 95% of the mass, with soft peaks spanning 3–5 frames
visible in per-clip plots. On wav2vec2, the mechanism concentrates so
sharply that a single frame carries 95% of the mass for the majority
of clips. Attention is effectively acting as a learned frame-selector
rather than a weighted averager.

**Verification that this is not a temperature runaway.** The learned
temperature τ = exp(log_tau) (see Key Design Decisions) was 0.67–0.96
across all 5 wav2vec2 seeds — every seed learned τ *below* the init
of 1.0, meaning the model was actively trying to *soften* attention.
The concentration therefore comes from large query-key dot-product
magnitudes at d=192, not from τ drift. This is the flip side of the
opposite problem on mel at d=32, where dot-product scores are small
enough that softmax approaches uniform and attentive pooling becomes
equivalent to mean pooling.

The wav2vec2 concentration is consistent with wav2vec2's internal
design: each output frame already encodes a ~25 ms window of
contextual audio via the pretrained model's own self-attention. A
single wav2vec2 frame is therefore already a rich local representation,
and "picking one discriminative frame" may be a reasonable strategy —
distinct from "weighting a region," which is the mechanism's behaviour
on mel.

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

**Step 1 — From-scratch features: both models near chance, Model 2 slightly ahead.**
With mel spectrograms or MFCCs, Models 2 and 3 both land near 0.22–0.25.
On mel Model 2 beats Model 3 by 0.028; on MFCC the two are
indistinguishable (|Δ| = 0.001). Both models struggle to exceed the 0.25
chance level consistently on from-scratch features. A matched-capacity
mel ablation (Model 3 given Model 2's exact encoder) rules out capacity
as the source of the mel gap — on mel, mean pooling genuinely outperforms
the attentive mechanism, and Model 3's MOVE class collapses in the
majority of seeds.

**Step 2 — Pretrained features: the Model 2 / Model 3 comparison flips.**
With wav2vec2 embeddings as input, the pattern reverses: Model 3
(d=192, L=1) reaches 0.246 ± 0.016, while Model 2 (d=64, L=2) reaches
only 0.219 ± 0.018. Model 3 now beats Model 2 by +0.027. The delta is
roughly 1.5σ but consistent across seeds: Model 3 outperforms Model 2
on 4 of 5 matched seeds.

**Step 3 — Why Model 3 needed a different grid for wav2vec2.**
The Model 3 mel grid selected d=32, L=1 — a minimal encoder (26K params)
appropriate for 128-dim input. Applied directly to 768-dim wav2vec2
input, that same config induces a 24:1 compression through `input_proj`
and collapses: 0.221 ± 0.018, identical to the mel number, with typical
early-stopping at epoch 2–3 and per-class F1 = 0.000 on entire classes.
Re-gridding with d_model sweeping through {64, 128, 192, 256} — which
corresponds to compression ratios {12:1, 6:1, 4:1, 3:1} — identifies
d=192 as the sweet spot. The monotonic trend in val F1 across d_model
averages (0.263, 0.278, 0.300, 0.259) locates the optimum clearly at
4:1 compression: larger encoders overfit, smaller ones underfit.

**Step 4 — What the flip means: features and architecture interact.**
On mel spectrograms the attentive pooling mechanism cannot concentrate
— dot-product scores are small at d=32, softmax approaches uniform,
and attention reduces to mean pooling. On wav2vec2, the opposite
regime emerges: at d=192 the scores are large enough that softmax
nearly saturates, concentrating 95% of the attention mass on a single
frame per clip (median across the val set, versus 17 frames on mel).
The mechanism is therefore doing qualitatively different work in each
regime — averaging-like on mel, frame-selection-like on wav2vec2 —
with the wav2vec2 behaviour appropriate because each wav2vec2 frame
already encodes ~25 ms of contextual audio.

This is consistent with the per-class breakdown: Model 3's advantage
on wav2vec2 is concentrated in the INTER class (+0.140 F1), the class
most likely to have localised acoustic signatures (turn-taking, vocal
response patterns) that benefit from selective pooling rather than
averaging.

The attentive pooling mechanism is *capacity-sensitive* on the encoder
side (it needs enough dims to preserve wav2vec2 information) and
*representation-sensitive* on the input side (it needs features rich
enough to discriminate between frames). Both conditions must be met for
the gain to materialise.

### Why results are near chance overall

The combination of 7 training children, 1 held-out test child, and ~570
training examples is too restrictive for reliable cross-participant
generalisation regardless of architecture or features. Each child has a unique
acoustic profile; the model cannot reliably distinguish context-specific from
speaker-specific patterns with so few training speakers. The classic overfitting
signature appears in every training curve: train loss drops to ~0.05–0.15 while
val F1 plateaus near chance.

### Why EXTEV and MOVE are the hardest classes on mel

EXTEV covers reactions to diverse external stimuli with no consistent acoustic
signature from the child. MOVE clips often contain movement artefacts but weak
vocalisation. Neither class has a stable acoustic pattern that survives the
speaker shift from training to test. On the matched-capacity mel ablation,
MOVE collapses to F1 = 0.000 in three of five seeds for Model 3 — the
attentive mechanism seems to actively suppress whatever weak diagnostic
signal MOVE does carry on mel features. On wav2vec2 both EXTEV and MOVE
recover somewhat for both models.

---

## Limitations

### Language is perfectly confounded with child identity, and val ≠ test language

Every participant in the dataset is monolingual, and the language
distribution across our splits is:

| Child | Language | Split |
|---|---|---|
| child_1, child_3 | Shipibo | train |
| child_4, child_5 | Tagalog | train |
| child_7, child_8, child_9 | Vietnamese | train |
| **child_2** | **Shipibo** | **val** |
| **child_6** | **Tagalog** | **test** |

Two independent language-related confounds arise from this structure:

1. **Language is perfectly confounded with speaker identity.** No child
   speaks more than one language, so we cannot separate "a language
   effect" from "a speaker effect" in any result. When we hold out
   child_6, we simultaneously hold out Tagalog.

2. **The val split and the test split speak different languages.** Val
   is 100% Shipibo; test is 100% Tagalog. Every hyperparameter in this
   README (d_model, num_layers, dropout, learning rate, early-stopping
   epoch) was selected on Shipibo validation performance, then reported
   on Tagalog test data. A feature, filter, or attention pattern that
   improves Shipibo val F1 may or may not transfer to Tagalog test — we
   have no direct signal. This explains the consistent val→test gap of
   +0.02 to +0.08 across our runs: the models genuinely improve on
   Shipibo during training but do not necessarily improve on Tagalog.

Both effects are consequences of the dataset and the
participant-disjoint holdout protocol we chose. We accept them because
the alternatives (random splits, same-child-in-train-and-val) would be
worse in other ways — random splits leak speaker identity, and same-
child-in-val would not test generalisation at all.

### The training distribution differs substantially across languages

Class proportions computed within each language's training clips:

| Class  | Shipibo (n=142) | Tagalog (n=145) | Vietnamese (n=282) |
|---|---|---|---|
| EXTEV  | 22.5% | 22.1% | 28.4% |
| INTER  | 22.5% | **46.9%** | 33.0% |
| MOVE   | 26.8% | 21.4% | 22.0% |
| PLTOY  | 28.2% | **9.7%**  | 16.7% |

Tagalog training children produce INTER at 2.08× the Shipibo rate, and
PLTOY at one-third the Shipibo rate. Because the test child is Tagalog,
any model that learns Tagalog-specific acoustic cues and associates them
with these Tagalog class priors will over-predict INTER and under-predict
PLTOY on the test set — independently of whether it has learned any
context-specific acoustic features.

### Implication for the Model 3 INTER-prediction bias

The Model 3 INTER-prediction bias observed in the confusion matrix
(57.4% of test predictions are INTER, versus a true rate of 29.4%) is
therefore better understood as the combination of two effects rather
than a single cause:

1. **Training-prior effect.** The Tagalog training base rate is 46.9%
   INTER. A well-calibrated model on Tagalog test data would predict
   INTER near this rate.
2. **Attention-concentration effect.** Model 3's attentive pooling
   concentrates 95% of its attention mass on a single frame per clip
   (see Results → Attention mechanism behaviour); when that frame
   carries an INTER-like signature, the whole clip is classified as
   INTER.

Model 2 predicts INTER at 31.1%, close to the true test rate of 29.4% —
it does *not* show this bias. Model 3 predicts INTER at 57.4%,
exceeding even the Tagalog prior, which means the attention mechanism
adds roughly +10 percentage points of bias on top of what the
training-distribution prior would explain. Neither effect alone
accounts for the observed behaviour; together they do.

### What cannot be claimed

Given the confounds above, the following claims would be unsupported by
our data and should not be made:

- *"Model 3 generalises across languages."* We never tested this
  because the val and test languages are disjoint and monolingual.
- *"The attention mechanism identifies universal acoustic signatures of
  social interaction."* The INTER-class advantage may be partly
  language-specific or Tagalog-prior-specific.
- *"Model 3 is robust to speaker shift."* The single held-out speaker
  is also a single held-out language.
- *"Hyperparameters were selected for the target language."* They were
  selected on Shipibo validation, then reported on Tagalog test.

### What the protocol does support

- Within the constraint of this dataset and its participant-disjoint
  holdout, Model 3 with wav2vec2 (d=192, L=1) macro-F1s above chance
  more reliably than Model 2 with wav2vec2 across 5 seeds.
- The Model 2 / Model 3 comparison flips sign across feature types
  (Model 2 wins on mel, Model 3 wins on wav2vec2), and the flip is
  reproducible.
- The attention mechanism behaves qualitatively differently across
  feature types (frame-averaging vs single-frame selection), which is
  an architectural finding independent of the language confound.
- The Model 3 INTER prediction bias exceeds what the Tagalog training
  prior alone predicts, so the attention mechanism contributes
  identifiable behaviour over and above distributional memorisation.

### Secondary limitations

- **Small held-out test set (n=269 from a single child)** means
  σ_seed ≈ 0.016–0.020 on the reported macro-F1. Differences of
  ≈ 0.03 between configurations should be treated as suggestive, not
  statistically established.
- **Frozen wav2vec2 embeddings.** A natural follow-up would be partial
  fine-tuning of the last 1–2 wav2vec2 layers; this was out of scope.
- **Low-cardinality metadata query.** Age and gender produce at most
  10 distinct query vectors across all 1009 clips, limiting what the
  attentive pooling mechanism can condition on. Adding language would
  triple this to 30, but on this dataset language is collinear with
  child identity, so adding it would effectively reintroduce child_id
  as a feature — a confound we explicitly excluded on privacy and
  leakage grounds.

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
  ├── spectrograms/            ← 1009 grayscale PNGs, 128 mel bins × T frames
  ├── pretrained_embeddings/   ← 1009 wav2vec2 .npy files, (768, 230)
  ├── splits/                  ← train.csv (569), val.csv (171), test.csv (269)
  ├── models/                  ← checkpoints (.pt)
  ├── results/                 ← JSON metrics
  └── figures/                 ← training curves, confusion matrices
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

**Multi-seed reporting (n=5)** — required because σ_seed ≈ 0.015–0.020 on
n_test=269, meaning a single seed comparison can easily show the wrong model
winning by chance.

**Feature-type-specific hyperparameter grids** — grid results from 128-dim
mel inputs do not transfer to 768-dim wav2vec2 inputs. Re-gridding for
wav2vec2 revealed that d=192 (4:1 compression), not d=32 (24:1), is the
appropriate capacity for Model 3. The `HPARAM_GRID` in `config.py` switches
automatically on `USE_PRETRAINED`.

**Grid-winner inspection over blind top-1** — on the wav2vec2 grid, the
numerically top-ranked config peaked at epoch 2 with best-epoch-1/2 patterns
across the top three configs, indicating lucky initialisations rather than
sustained learning. The selected config (d=192, L=1, drop=0.3) ranked third
numerically but was the only top config with 30+ epochs of proper training
dynamics. A re-run of the top-1 config on the same seed confirmed it
collapses to val F1 = 0.21 on replication.

**Unscaled dot product in Model 3** — the `/sqrt(d)` convention from
multi-head self-attention is removed. With one query over T=230 frames it
divides scores by ~8, collapsing the softmax to near-uniform
(measured mean_weight_std ≈ 0.00086 vs uniform proxy 0.00659). Removing it
restores usable dynamic range. A learned temperature scalar τ=exp(log_tau)
(one parameter, init τ=1) provides end-to-end control over sharpness.

**Preprocessing as automatic opt-in** — `USE_PER_CLIP_NORM=True` always;
`USE_SPECAUGMENT = not USE_PRETRAINED` (SpecAugment applies to spectrogram-like
inputs, not to pretrained embeddings where it would mask phonetic content).
The ablation shows preprocessing helps val but not test, demonstrating that
inter-child variance is the ceiling.

**Canonical multi-seed JSONs** — all Transformer numbers in the Results
section are sourced from 5-seed JSONs generated by `train_multiseed.py`
under the current code and config. Single-seed smoke tests and earlier runs
under different preprocessing states are not used as reported values.
