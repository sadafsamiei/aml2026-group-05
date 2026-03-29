# Multi-Class Classification of Child Vocalisation Contexts in Shipibo-Conibo

This project investigates whether short child-vocalisation segments from the Shipibo-Conibo subset of the ACQDIV corpus can be classified into broad interactional context categories. Each segment is paired with an annotated context label and limited metadata. The main goal is to compare a simple baseline with a small neural model and to test whether metadata-aware pooling improves classification beyond acoustics alone.

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

All models are evaluated with the same protocol.

To reduce overly optimistic results, the preferred split is **participant-disjoint**, so that the same child does not appear in both training and test data. If a fully participant-disjoint split is not feasible because of limited data, we use a grouped split by recording or session and state clearly that this evaluates only within-participant generalization.

The single evaluation metric is:

- **Macro-F1**

Macro-F1 is used because the class distribution is expected to be imbalanced.

The main comparison is between:
1. logistic regression,
2. the audio-only Transformer,
3. the Transformer with metadata-query attentive pooling.

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
