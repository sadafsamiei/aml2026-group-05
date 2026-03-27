# Multi-Class Classification of Child Vocalisation Contexts in Shipibo-Conibo

This project asks whether short child vocalisation segments from the ACQDIV corpus can be classified into broad interactional context labels using acoustic features alone. The data comes from the Shipibo-Conibo subset (`shp`) and contains annotated audio segments with timestamps, context labels and metadata such as child age, gender and language.

## Problem setting

This project studies whether and how a scontaining a child vocalisation can be classified into one of four interactional context classes in the Shipibo-Conibo subset of ACQDIV.

Each example is
$$\[
(x_i^{audio}, x_i^{meta}, y_i)\]$$
where $$\(x_i^{audio}\)$$ is the audio clip, $$\(x_i^{meta}\)$$ is a small metadata vector containing age and gender, and
$$\[
y_i \in \{\mathrm{PLTOY}, \mathrm{MOVE}, \mathrm{INTER}, \mathrm{EXTEV}\}\]$$
The four labels are defined as:
- **PLTOY**
- **MOVE** = {MOVE2, MOVE3, MOVE5}
- **INTER** = {INTER1, INTER2, INTER4, INTER5, INTER6, INTER7}
- **EXTEV** = {EXTEV1, EXTEV2, EXTEV3, EXTEV6}

The goal is to learn a classifier
$$\[
f_\theta(x_i^{audio}, x_i^{meta}) = y_i\]$$

that maps the audio segment and metadata to one of these four labels.


## Models

As a simple baseline, we use **multinomial logistic regression** on clip-level MFCC summary features, such as the mean and standard deviation of each MFCC over time.

Our main model is a **small Transformer** over time frames. The audio is represented as a mel-spectrogram or MFCC frame sequence, encoded by the Transformer, and then pooled with **metadata-query attentive pooling**. In this step, the metadata is turned into a query vector that decides which time frames are most important before classification.

This gives one simple baseline and one stronger model with a non-trivial metadata fusion method.

## Evaluation protocol

We split the data by **recording/session**, not by random clips, so that similar segments do not appear in both training and test sets. The same split is used for all models.

We use **macro-F1** as the single evaluation metric, because the classes are likely imbalanced.

We compare:
1. Logistic regression baseline
2. Audio-only Transformer
3. Transformer with metadata-query attentive pooling

This shows whether the Transformer improves over a simple baseline and whether metadata helps beyond acoustics alone.

## Expected Contribution

This project will provide a useful benchmark for classifying child vocalisation contexts in an under-resourced language setting. It will show whether simple acoustic features, small neural models and basic metadata such as age and gender can help distinguish between broad context groups in Shipibo-Conibo child vocalisations.
