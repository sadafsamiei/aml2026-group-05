# Multi-Class Classification of Child Vocalisation Contexts in Shipibo-Conibo


This project asks whether short child vocalisation segments from the ACQDIV corpus can be classified into broad interactional context labels using acoustic features alone. The data comes from the Shipibo-Conibo subset (`shp`) and contains about 1,000 annotated audio segments with timestamps and context labels. 

## Research Questions

1. **Can short child vocalisation segments be classified into a small set of context labels using only acoustic features?**
2. **Do neural models, especially a small Transformer, perform better than simple classical baselines on this task?**
3. **Which labels are most often confused with each other?**
4. **Which parts of the vocalisation seem most useful for prediction and does the Transformer help capture them through self-attention?**

## Problem Setting

We formulate the task as a **4-class supervised classification problem**. Each example is a short audio segment containing one child vocalisation, and the model must predict one of these labels:

- **PLTOY**
- **MOVE** = `MOVE2`, `MOVE3`, `MOVE5`
- **INTER** = `INTER1`, `INTER2`, `INTER4`, `INTER5`, `INTER6`, `INTER7`
- **EXTEV** = `EXTEV1`, `EXTEV2`, `EXTEV3`, `EXTEV6`

We exclude labels such as `REST`, `FOOD2`, `FOOD5`, `TOYMO`, and `SLE` because they are likely too small or presumably less balanced.

Formally, let

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}
$$

where $x_i$ is a vocalisation segment and

$$
y_i \in \{\mathrm{PLTOY}, \mathrm{MOVE}, \mathrm{INTER}, \mathrm{EXTEV}\}.
$$

The goal is to learn a classifier

$$
f_{\theta}: \mathcal{X} \rightarrow \mathcal{Y}
$$

that maps each input segment to one of these four labels.

## Approach

We first extract each vocalisation segment from the recordings using its start and end timestamps. The clips are then resampled and padded or truncated to a fixed length.

We use two input representations:

- **MFCC features** for simple baseline models
- **Mel-spectrograms** for neural models

As baselines, we use **logistic regression** and possibly an **SVM** on the MFCC features.

For the neural models, we compare a small **CNN** and a small **Transformer** on mel-spectrograms. The CNN learns local patterns in the spectrogram. The Transformer treats the spectrogram as a sequence over time and uses self-attention to focus on the most relevant parts of the vocalisation.

## Evaluation

To avoid data leakage, we split the data by **recording or session** rather than by random segment. This prevents very similar clips from appearing in both training and test sets.

We evaluate performance using:

- **Macro F1-score**
- **Accuracy**
- **Confusion matrix**

Macro F1 is the main metric because the classes may not be perfectly balanced.

## Expected Contribution

This project will provide a useful benchmark for classifying child vocalisation contexts in an under-resourced language setting. It will show whether simple acoustic features and small neural models can distinguish between broad context groups in Shipibo-Conibo child vocalisations.
