# Multi-Class Classification of Child Vocalisation Contexts in ACQDIV

This project investigates whether short child-vocalisation segments from an ACQDIV-based dataset can be classified into broad interactional context categories. The initial experiments focus on the Shipibo-Conibo subset, while leaving open the possibility of extending the analysis to additional cultural or linguistic subsets if this is necessary to improve participant coverage. Each segment is associated with an annotated context label and limited metadata. The main goal is to compare a simple baseline with a small neural model and to examine whether metadata-aware pooling improves classification beyond acoustic information alone.

## Problem setting

We formulate the task as a 4-class supervised classification problem. Each example is

$$\[
(x_i^{audio}, x_i^{meta}, y_i)
\]$$

where $$\(x_i^{audio}\)$$ is an acoustic representation of the clip, $$\(x_i^{meta}\)$$ is a small metadata vector, and $$\(y_i\)$$ is the target context label.

The label space is

$$\[
y_i \in \{\mathrm{PLTOY}, \mathrm{MOVE}, \mathrm{INTER}, \mathrm{EXTEV}\}
\]$$

The four labels are defined as follows:

- **PLTOY**
- **MOVE** = {MOVE2, MOVE3, MOVE5}
- **INTER** = {INTER1, INTER2, INTER4, INTER5, INTER6, INTER7}
- **EXTEV** = {EXTEV1, EXTEV2, EXTEV3, EXTEV6}

The objective is to learn a classifier

$$\[
f_\theta(x_i^{audio}, x_i^{meta}) = y_i
\]$$

To reduce leakage and protect participant privacy, direct identity features such as child ID are not used as model inputs. Metadata is restricted to safer variables such as age and gender, and these are evaluated carefully because they may still correlate with participant identity.

## Models

We compare three models.

1. **Simple baseline: multinomial logistic regression**  
   This model uses clip-level MFCC summary features, such as the mean and standard deviation of each MFCC over time.

2. **Neural baseline: audio-only Transformer**  
   A small Transformer encoder is applied to a sequence of spectrogram frames. The resulting frame representations are pooled with standard mean pooling before classification.

3. **Main model: Transformer with metadata-query attentive pooling**  
   The same Transformer encoder is used, but the pooled representation is computed with metadata-query attentive pooling. In this model, the metadata is mapped to a query vector that attends over the frame-level audio representations, allowing the model to weight parts of the clip differently before classification.

This setup keeps the comparison focused: one simple baseline, one strong baseline without metadata conditioning, and one metadata-aware model.

## Evaluation protocol

All models are evaluated using the same protocol, applied consistently across all three model types.

### Split strategy

The preferred evaluation strategy is **participant-disjoint**, meaning no child
appears in both training and test data. The exact protocol depends on how many
unique participants are available after the full dataset is assembled:

- **Five or more participants:** a fixed holdout split is used, with participants
  assigned to train, validation, and test sets in approximately a 70 / 15 / 15
  ratio. Model selection is performed on the validation set. Final performance
  is reported on the held-out test set once, after all hyperparameter choices
  are finalised.

- **Three or four participants:** leave-one-participant-out (LOPO) cross-validation
  is used. In each fold, one participant is held out as the test set, one serves
  as the validation set for early stopping and model selection, and the remaining
  participants form the training set. Performance is reported as mean macro-F1
  across folds, together with the standard deviation.

- **Fewer than three participants (current data state):** a stratified k-fold
  cross-validation (k = 5) is used, stratified by the 4-class label. In this
  fallback, the same participant may appear in both training and test folds.
  This evaluates only within-participant generalisation and is explicitly
  acknowledged as a limitation. Results obtained under this protocol cannot
  be interpreted as evidence of generalisation to unseen speakers.

The applicable protocol will be stated explicitly in the results section once
the final dataset size is known.

### Why participant structure influences protocol choice

The current dataset contains vocalisations from two children (JEX, n = 43;
PED, n = 79). Several context classes are strongly confounded with participant
identity in the current data: MOVE and EXTEV examples come exclusively from PED,
while JEX contributes no MOVE or EXTEV examples at all. A fixed holdout under
these conditions risks either (a) placing an entire class in the test set with
no training signal, or (b) producing inflated estimates because the model can
exploit participant-specific acoustic cues rather than context-specific ones.
Cross-validation distributes this risk across folds and produces a more honest
estimate of out-of-sample performance. The choice of protocol is therefore
not arbitrary — it is a direct consequence of dataset size and participant
coverage, and is documented as such.

### Evaluation metric

The single reported metric is **macro-F1** (the unweighted mean of per-class
F1 scores). Macro-F1 is used rather than accuracy because the class distribution
is imbalanced, and a majority-class classifier would achieve misleadingly high
accuracy. Macro-F1 weights each class equally regardless of its frequency,
making it sensitive to performance on minority classes.

All three models — logistic regression, audio-only Transformer, and Transformer
with metadata-query attentive pooling — are evaluated under the identical
protocol and metric. No model-specific evaluation adjustments are made.

## Hyperparameter tuning

Hyperparameter tuning is kept limited to make the project feasible.

For logistic regression, we tune:
- regularization strength.

For the Transformer models, we tune:
- number of layers,
- hidden dimension,
- dropout,
- learning rate.

All model selection is done on the validation set using macro-F1.

## Expected contribution

This project will provide a small, well-controlled benchmark for classifying child-vocalisation contexts in an under-resourced language setting. It will show whether a small Transformer improves over a simple feature-based baseline, and whether metadata-query attentive pooling helps beyond an audio-only Transformer while avoiding direct use of participant identity.
