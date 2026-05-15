# Multi-Class Classification of Child Vocalisation Contexts in ACQDIV

This project investigates whether short child-vocalisation segments from an
ACQDIV-based dataset can be classified into broad interactional context
categories. The dataset spans three typologically distinct languages
(Shipibo-Conibo, Tagalog, Vietnamese) and nine participants. Each segment is
associated with an annotated context label and limited metadata. The main
goal is to compare a simple statistical baseline with two small neural
models across three acoustic feature sources, and to examine whether
metadata-aware pooling improves classification beyond acoustic information
alone.

## Problem setting

We formulate the task as a **4-class supervised classification** problem.
Each example is

$$
(x_i^{\text{audio}},\ x_i^{\text{meta}},\ y_i)
$$

where $x_i^{\text{audio}}$ is an acoustic representation of the clip,
$x_i^{\text{meta}}$ is a small metadata vector (child age and gender), and
$y_i$ is the target context label.

The label space is

$$
y_i \in \{\texttt{PLTOY},\ \texttt{MOVE},\ \texttt{INTER},\ \texttt{EXTEV}\}
$$

defined as follows:

- **PLTOY** — playing with a toy
- **MOVE** = {MOVE2, MOVE3, MOVE5} — movement activity
- **INTER** = {INTER1, INTER2, INTER4, INTER5, INTER6, INTER7} — social interaction
- **EXTEV** = {EXTEV1, EXTEV2, EXTEV3, EXTEV6} — external event

The objective is to learn a classifier

$$
f_\theta(x_i^{\text{audio}},\ x_i^{\text{meta}}) = y_i
$$

that generalises to an unseen participant speaking a language not seen at
validation time. This cross-speaker, cross-linguistic generalisation
requirement distinguishes the problem from standard audio classification
benchmarks and is the central challenge of the project.

To reduce leakage and protect participant privacy, direct identity features
such as child ID are excluded from model inputs. Metadata is restricted to
age and gender; these are evaluated carefully because they may still
correlate with participant identity, and their contribution is ablated
explicitly.

## Dataset

The dataset consists of **1,009 short audio clips** extracted from the
ACQDIV corpus, each annotated with one of the four context labels above.

| Label | Full name          | Count       |
|-------|--------------------|-------------|
| INTER | Social interaction | 325 (32.2%) |
| PLTOY | Playing with toys  | 240 (23.8%) |
| EXTEV | External event     | 222 (22.0%) |
| MOVE  | Movement activity  | 222 (22.0%) |

Clips come from **9 children** spanning three typologically distinct
languages: Shipibo-Conibo (South America), Tagalog (Philippines), and
Vietnamese (Southeast Asia). Recording durations range from under 1 s to
8 s (mean 1.33 s ± 0.71 s). All recordings are loaded at 16 kHz and
denoised with a spectral noise-reduction procedure before feature
extraction.

### Feature sources

Three acoustic representations are evaluated, allowing the comparison to be
disentangled from feature-engineering choices:

1. **Mel spectrogram.** 128-bin log-mel spectrogram (n\_fft=2048,
   hop=512, f\_min=0 Hz, f\_max=8 kHz). For Model 1, collapsed to a
   256-dimensional summary vector (mean and std per bin). For Models 2–3,
   used as a padded 128 × 230 sequence.

2. **MFCC.** 20 Mel-frequency cepstral coefficients (n\_fft=2048,
   hop=512, 16 kHz). For Model 1, collapsed to a 40-dimensional summary
   vector. For Models 2–3, used as a padded 128 × 230 sequence (coefficients zero-padded to F_DIM=128).

3. **wav2vec2 embeddings.** Last-layer embeddings from a frozen,
   pretrained wav2vec2-base encoder (768 dimensions per frame), extracted
   once and cached. For Model 1, collapsed to a 1,536-dimensional summary
   vector. For Models 2–3, used as a padded 768 × 230 sequence. Using a
   frozen encoder means the model cannot overfit to speaker identity
   through fine-tuning, which is consistent with the privacy constraint.

## Models

We compare three models that differ in architectural complexity and their
use of metadata.

**Model 1 — Multinomial logistic regression (simple baseline).**
Receives a fixed-length summary feature vector (mean ∥ std collapse along
the time axis), applies StandardScaler (fit on training data only), and
fits an L2-regularised softmax classifier with
`class_weight='balanced'`. The regularisation strength $C$ is selected via
inner Leave-One-Participant-Out cross-validation over the 7 training
children. The lbfgs solver converges to a deterministic solution; seed
variance is exactly zero. Total parameters: 1,028 (mel), 164 (MFCC), or
6,148 (wav2vec2).

**Model 2 — Audio-only Transformer with mean pooling (neural baseline).**
A small Transformer encoder (pre-norm, multi-head attention, ReLU FFN)
applied to a sequence of spectrogram or embedding frames. Each input
sequence is per-clip normalised (z-score across all time–frequency bins)
before being fed to the model. SpecAugment data augmentation is applied
during training on mel and MFCC features only (2 frequency masks
$F_\text{max}=27$, 2 time masks $T_\text{max}=70$); it is not applied to
wav2vec2 embeddings, where masking would destroy learned phonetic content.
Frame representations are combined via masked mean pooling before
classification. The v2 architecture introduces three independent dropout
locations replacing the original single parameter: `dropout_encoder`
(inside each sub-layer), `dropout_input` (after input projection), and
`dropout_head` (after pooling, before classifier).

**Model 3 — Transformer with metadata-query attentive pooling (main
model).** Identical encoder to Model 2, but mean pooling is replaced by
metadata-query attentive pooling. The metadata vector (z-scored age ⊕
one-hot gender) is projected to a query vector $\mathbf{q} \in
\mathbb{R}^d$ via a learned linear layer followed by LayerNorm. Attention
weights and pooled representation are computed as

$$
s_t = \frac{\mathbf{q}^\top \mathbf{h}_t}{\sqrt{d}\cdot\tau}, \qquad
\alpha = \text{softmax}(\mathbf{s}), \qquad
\hat{\mathbf{h}} = \sum_{t=1}^{T_\text{real}} \alpha_t \mathbf{h}_t
$$

where $\tau$ is a learned temperature scalar (initialised to 1.0) and
padded frames receive $\alpha_t = 0$ via the attention mask. A fourth
dropout location `dropout_meta` is applied to the metadata query after
projection (v2 only, unique to Model 3).

This setup keeps the comparison focused: one simple linear baseline, one
strong audio-only neural baseline, and one metadata-conditioned model —
with a single architectural modification (attentive pooling vs. mean
pooling) isolating the contribution of metadata conditioning.

## Evaluation protocol

All models are evaluated using the same protocol, applied consistently
across all three model types and all three feature sources.

### Split strategy

With 9 participants the problem setting specifies a **participant-disjoint
fixed holdout**, with participants assigned to train, validation, and test
sets in a **56 / 17 / 27** ratio by clip count:

| Split      | Children     | Clips | Ratio | Language                            |
|------------|--------------|-------|-------|-------------------------------------|
| Train      | 7 children   | 569   | 56.4% | Shipibo-Conibo, Tagalog, Vietnamese |
| Validation | 1 (child\_2) | 171   | 16.9% | Shipibo-Conibo                      |
| Test       | 1 (child\_6) | 269   | 26.7% | Tagalog                             |

The ratio is not perfectly balanced by clip count because participants
rather than clips are the unit of assignment, and recording coverage varies
across participants. Splits are **fixed once and never rebuilt**. Model
selection is performed on the validation set using macro-F1. Final
performance is reported on the held-out test set once, after all
hyperparameter choices are finalised.

For neural models, final test scores are reported as **mean ± std across 5
random seeds** (42, 123, 456, 789, 2024). Seeds are used for variance
estimation only, not for model selection. The logistic regression solver
(lbfgs) is deterministic; its seed variance is exactly zero.

### Cross-linguistic confound

The validation and test children speak **different languages**: the
Shipibo-Conibo validation child (child\_2) belongs to a language group
with 22.5% INTER in training, while the Tagalog test child (child\_6)
belongs to a group with 46.9% INTER in training. This distributional
mismatch between validation and test is a structural confound that cannot
be eliminated given the dataset size. Its consequence is that validation
macro-F1 is an unreliable selection criterion: a model tuned to fit the
Shipibo-Conibo prior will encounter distributional shift at test time and
may generalise worse. This is documented as a known limitation rather than
an experimental error, and its effect on hyperparameter search is
quantified explicitly in the results.

### Supplementary Leave-One-Child-Out cross-validation

To separate speaker-identity variance from seed variance — two sources of
variability that are conflated in the single fixed holdout — a
supplementary Leave-One-Child-Out (LOCO) cross-validation is run for
Model 1 on mel features, holding out each of the 9 children in turn as
the test participant. LOCO mean and standard deviation across folds
provide an estimate of $\sigma_\text{speaker}$ that is not available from
the fixed holdout alone.

### Evaluation metric

The single reported metric is **macro-F1** (unweighted mean of per-class
F1 scores). Macro-F1 is preferred over accuracy because the class
distribution is imbalanced (INTER at 32.2% would allow a majority-class
classifier to achieve misleadingly high accuracy) and over weighted-F1
because the goal is to measure performance equally across all four
interactional contexts regardless of their frequency. All three models are
evaluated under the identical protocol and metric; no model-specific
adjustments are made.

## Hyperparameter tuning

Hyperparameter tuning proceeds in two phases to keep the project feasible
while enabling a rigorous extension.

### Phase 1 — Discrete grid search (canonical results)

For **logistic regression**, regularisation strength $C$ is selected from
a 9-point logspace grid $C \in \{10^{-4},\, 10^{-3},\, \ldots,\, 10^{4}\}$
via inner Leave-One-Participant-Out cross-validation over the 7 training
children. This is equivalent to nested cross-validation and avoids leaking
test information into model selection.

For the **Transformer models**, an 8-configuration grid per (model,
feature) combination tunes:

- $d_\text{model} \in \{32, 64\}$
- $L \in 1, 2\$ (number of encoder layers)
- dropout $\in \{0.3, 0.5\}$
- learning rate: fixed at $10^{-3}$ for mel/MFCC and $5 \times 10^{-4}$
  for wav2vec2

giving $2 \times 2 \times 2 = 8$ configurations. All model selection is
performed on the validation set using macro-F1. For Models 2 and 3 on
MFCC and wav2vec2 features, the mel-optimised configuration is reused
directly, as the feature dimensionality differences are handled by a
shared input projection layer.

### Phase 2 — Bayesian hyperparameter search (extension)

As an extension, the discrete grids are replaced by Bayesian optimisation
using **Optuna Tree-structured Parzen Estimator (TPE)**, applied across
all 9 (model, feature) combinations:

- **LR (all features):** 80-trial log-uniform line search over
  $C \in [10^{-6}, 10^{6}]$, seeded with the 9 canonical grid points.
  Objective: inner LOPO-CV macro-F1. LR is deterministic; no seed
  averaging is required.

- **M2 and M3 (all features):** 60-trial search over
  $d_\text{model} \in \{32, 64, 128\}$, $L \in [1, 2]$,
  lr log-uniform $[10^{-4}, 10^{-2}]$, and three continuous dropout
  parameters (plus `dropout_meta` for M3). Objective: validation
  macro-F1. Each Optuna winner is re-trained with 5 random seeds to
  produce a verified mean ± std.

The extension also introduces the v2 multi-dropout architecture described
above, which expands the search space from a single dropout parameter to
four independent locations.

## Expected contribution

This project provides a small, well-controlled benchmark for classifying
child-vocalisation contexts in a cross-linguistic, under-resourced setting.
The specific contributions are:

1. **A systematic three-model, three-feature comparison** under a
   participant-disjoint holdout with 5-seed variance estimation, covering
   all 9 (model, feature) combinations and reporting every result with a
   traceable JSON.

2. **Evidence that model simplicity beats architectural complexity** at
   this data scale: logistic regression on mel summary statistics achieves
   the best test macro-F1 across all feature sources and pooling
   strategies, a result that is robust to regularisation choices, random
   seeds, and LOCO cross-validation.

3. **A data-driven error ceiling analysis** showing that 40.9% of test
   clips fail all three models simultaneously, and that the uniformity of
   per-class errors across a 1,028-parameter LR and a 648K-parameter
   Transformer is strong evidence that the bottleneck is in the data, not
   the architecture.

4. **Quantification of the cross-linguistic confound** via the Optuna
   extension: across 60 TPE trials for both transformer-on-mel
   configurations, higher validation macro-F1 does not predict higher test
   macro-F1, providing direct evidence that the Shipibo validation prior
   and the Tagalog test distribution are structurally mismatched.

5. **Bayesian hyperparameter search yielding three genuine gains**: LR mel
   ($0.277 \to 0.311$, $+0.034$), LR wav2vec2 ($0.263 \to 0.285$, $+0.022$) and Model 2 wav2vec2
   ($0.219 \to 0.251$, $+0.032$), with the remaining six combinations
   within seed noise of canonical or degraded — a pattern that correlates
   with the canonical val-test gap and confirms that tuning adds signal
   only when the selector criterion is reliable.

6. **Speaker-identity dominance**: LOCO cross-validation yields
   $\sigma_\text{speaker} = 0.053$ vs. $\sigma_\text{seed} = 0.000$,
   demonstrating that the primary bottleneck is participant diversity, not
   model capacity or random initialisation — a finding with direct
   implications for data collection priorities in future work.
