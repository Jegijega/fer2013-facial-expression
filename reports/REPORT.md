# Notes for my W&B report

These are the notes I use when writing up the report on W&B. Each bracketed line
is the panel I attach there.

## Problem
7-class emotion classification on 48x48 grayscale faces. I use the official
splits (Training / PublicTest / PrivateTest). The data is imbalanced: lots of
"Happy", almost no "Disgust".
*[panel: class distribution bar chart from the EDA cell]*

## Sanity checks
The first loss of every fresh model is about ln(7) = 1.946, and every model can
overfit a single batch to ~100%, so I know the training code is correct.
*[panel: the single-batch overfit loss curve]*

## Comparing the models
*[panel: val/acc vs epoch, all runs on one chart]*
*[panel: test_acc per run as a bar chart, with param count]*

- **00 linear** - test 0.364, gap ~0. Underfits, no spatial features.
- **01 tiny_cnn** - test 0.526, gap ~0.47. Conv helps but overfits.
- **02 deeper_cnn** - test 0.584, gap ~0.42. Overfits, memorizes training set.
- **03 regularized_cnn** - test 0.704, gap ~0.03. BN + dropout + aug fix the gap.
- **04 resnet18** - test 0.716, gap ~0.28. Transfer learning, highest accuracy.

## Underfitting vs overfitting
*[panel: generalization gap vs epoch for runs 00, 02 and 03 together]*
- Underfitting (00, 01): train and val are both low and close. High bias.
- Overfitting (02): train accuracy goes to ~99%, val stalls and val loss even
  goes back up. High variance.
- Good fit (03, 04): small gap, val loss follows train loss. Augmentation did the
  most to get here.

## Per-class results
*[panel: confusion matrices for runs 03 and 04]*
Happy and Surprise are the easiest. Disgust is the hardest (too few examples).
Fear, sad and neutral get mixed up with each other, which matches the known label
noise in this dataset.

## Hyperparameter choices
I tuned the hyperparameters per model by hand (visible in the configs and in
each run's W&B config): different learning rates (1e-3 vs 5e-4), optimizers
(adam vs adamw), dropout, weight decay and a cosine schedule for the deeper
models. The biggest levers were the learning rate and the regularization
(augmentation + dropout).

## Conclusion
Best model was ResNet18 with transfer learning (~68%), best from-scratch was the
regularized CNN (~66%). Capacity on its own didn't help; regularization and
augmentation did. If I kept going I'd try stronger augmentation, an ensemble, and
maybe focal loss for the rare classes.
