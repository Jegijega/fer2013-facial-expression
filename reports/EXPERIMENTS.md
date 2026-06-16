# Experiment log

This is where I write down what actually happened in each run and why I think it
happened. The numbers here are the real ones from my W&B runs (project
`fer2013-fer-challenge`). I keep the "expected vs what I saw" honest, including
the cases where the result surprised me.

For every run I look at three things:
- train accuracy vs val accuracy (the gap = how much it overfits)
- the val loss curve (if it goes back up, the model is memorizing)
- per-class accuracy / confusion matrix (the rare classes are the hard part)

---

## Experiment 0 - Linear baseline

**Why I ran it:** I wanted a floor. One `Linear(2304, 7)` layer can only learn a
global weighting of the pixels, it has no idea about spatial structure, so it
should be bad. If a real model can't beat this, something is wrong.

**What I got:** test acc **36.4%**, best val acc 36.6%. Train acc ended around
41%. Train and val stayed close to each other and both low.

**Why:** this is textbook **underfitting / high bias**. The model is too simple
for the task. The small gap tells me it's not overfitting at all, it just can't
represent the problem. Per-class it already shows the imbalance: Happy 61% (lots
of data) but Disgust only 9% (almost no data).

---

## Experiment 1 - TinyCNN (2 conv blocks)

**Why I ran it:** add convolutions so the model can actually look at local
patterns. I expected a clear jump over the linear baseline.

**What I got:** test acc **52.6%**, best val acc 52.2% (reached at epoch 6).
But the interesting part: train acc climbed all the way to **98%** while val acc
froze around 52%, and val loss went from 1.27 up to 4.34.

**Why this surprised me a bit:** even this "small" CNN (~0.6M params) **overfits**
FER2013 once it trains long enough. After about epoch 6 the validation accuracy
stops improving but the training accuracy keeps rising - the model is now
memorizing the training set instead of learning general features. You can see it
most clearly in the val loss, which keeps going *up* while train loss goes to
almost zero.

**What saved the score:** my training loop keeps the weights from the best
validation epoch, so the reported test accuracy uses the epoch-6 weights, not the
overfitted epoch-30 ones. This is basically early stopping by hand.

**Takeaway:** convolutions clearly help (36% -> 53%), but with no regularization
the gap is already huge (~0.47). That's exactly what pushes me to (a) try a
bigger model to see the overfitting get even worse, and (b) then add
regularization to fix it.

---

## Experiment 2 - DeeperCNN (no regularization)

**Why I ran it:** deliberately make the overfitting worse - 4 conv blocks, big FC
head (~5.3M params), no BatchNorm, no dropout, no augmentation, no weight decay. I
want to see the biggest possible gap.

**What I got:** test acc **58.4%**, best val acc 57.9% (reached very early, at
epoch 7). Train acc went to **99%** by around epoch 9 and stayed there; the gap
settled at **~0.42** and val loss climbed from 1.15 all the way to ~4.1.

**Why:** this is the overfitting case I wanted. With more capacity than
experiment 1 and still zero regularization, the network drives training error to
almost nothing while validation stops improving after ~7 epochs. Everything after
that is pure memorization - the val loss going up while train loss goes down is
the clearest signal.

**Two things I noticed:**
- The extra capacity actually got a *higher* test score than TinyCNN (58% vs
  53%), and it learned the rare classes better on the training set (Disgust
  per-class jumped to 53%). So capacity does help the model fit - the problem is
  purely that it doesn't *generalize*, the gap is the issue, not the raw ability.
- Almost all of the 40 epochs were wasted. The best validation model appears at
  epoch 7, so training 5x longer with no regularization bought me nothing except
  a bigger gap. This is exactly the motivation for experiment 3.

---

## Experiment 3 - RegularizedCNN (BN + dropout + augmentation)

**Why I ran it:** same depth idea as experiment 2, but now I add all the
regularizers: BatchNorm in every block, Dropout (0.25-0.4), data augmentation
(flips + small rotations/shifts), weight decay, label smoothing and a cosine
learning-rate schedule. The goal is to take the huge gap from experiment 2 and
close it.

**What I got:** test acc **70.4%**, best val acc 68.7%, and the gap stayed at
about **+0.025** - basically flat. Even better, for most of training the gap was
slightly *negative* (val accuracy was higher than train accuracy).

**Why this worked:**
- The negative gap looks weird at first but it makes sense: dropout and
  augmentation make the *training* task harder (the model sees distorted images
  and half its units switched off), while validation is on clean images. So the
  model is genuinely learning features instead of memorizing, and the val loss
  follows the train loss down instead of exploding like in experiments 1 and 2.
- BatchNorm let the network train smoothly for all 60 epochs and keep improving,
  instead of collapsing into memorization after 7 epochs.
- Augmentation is doing most of the heavy lifting - it effectively gives the model
  more varied data, which is the direct cure for the overfitting I saw earlier.

**Per-class:** much more even now. Happy 91%, Surprise 81%, Neutral 75%, and even
Disgust got to 56% despite being so rare. Fear is still the weakest (47%) and
mostly gets confused with sad/neutral, which is a known hard case in FER2013.

**Takeaway:** this is the payoff of the whole progression. Same rough capacity as
the model that overfit to a 0.42 gap, but with regularization it generalizes to
70% test accuracy with an almost-zero gap. This is my best from-scratch model.

---

## Experiment 4 - ResNet18 (transfer learning)

**Why I ran it:** see if a known residual architecture with ImageNet weights beats
my hand-built CNN. I changed the first conv to take 1 channel and removed the
early downsampling (3x3 stride-1 stem, no maxpool) because the faces are only
48x48 and I didn't want to throw away resolution immediately.

**What I got:** test acc **71.6%** - the highest of all my models - with best val
acc 70.6% at epoch 40. But the gap grew to **0.28** by the end, and the val loss
started climbing after about epoch 8 (1.04 -> 1.25) while train loss kept
dropping.

**Why it's the best but not the cleanest:**
- The ImageNet-pretrained features give it a strong head start; even epoch 1 is
  already at 58% val. That's the value of transfer learning.
- But ResNet18 has ~11M params, about 3x my regularized CNN, so even with
  augmentation it slips into overfitting in the second half (train acc reaches
  98.5%, val plateaus ~70%). The augmentation slows it down but doesn't stop it.
- Early stopping + keeping the best-val weights is what gives the clean 71.6%
  test number instead of the overfitted late-epoch weights.
- Best per-class result on Disgust (73%), the rare class - the pretrained
  features generalize better there than my from-scratch model did.

**Takeaway / trade-off:** ResNet18 wins on raw accuracy (71.6% vs 70.4%), but my
RegularizedCNN is the "healthier" model - almost the same accuracy with a tiny
0.03 gap versus ResNet's 0.28, and a third of the parameters. If I cared about a
clean, efficient model I'd pick the regularized CNN; if I only cared about the
score, ResNet18.

---

## Final summary

The whole point of the assignment shows up clearly across the five runs:

- **Underfitting** (linear): both train and val low, tiny gap. The model is too
  weak.
- **Overfitting** (tiny_cnn, deeper_cnn): train accuracy near 99%, val stuck,
  val loss going up. Big gap. Too much freedom, not enough constraints.
- **Good fit** (regularized_cnn): regularization + augmentation closes the gap to
  almost zero at 70% test.
- **Transfer learning** (resnet18): highest accuracy 71.6%, but a real reminder
  that big pretrained models still overfit a small dataset and need early
  stopping.

What made the biggest difference, in order: data augmentation, then BatchNorm,
then transfer learning. Capacity by itself never helped generalization - only
regularization and better features did.

The rare classes (Disgust, Fear) stayed the hardest the whole way through, and
Fear kept getting confused with Sad/Neutral. That's a known property of FER2013
(noisy labels, ambiguous faces), not something more training fixed.

---

## Summary so far

- **00 linear** - test acc 0.364, gap ~0.06. Underfits, too simple.
- **01 tiny_cnn** - test acc 0.526, gap ~0.47. Conv helps, but overfits without
  regularization.
- **02 deeper_cnn** - test acc 0.584, gap ~0.42. More capacity fits better but
  overfits hard, best val already at epoch 7.
- **03 regularized_cnn** - test acc 0.704, gap ~0.03. Regularization fixes the
  gap, best from-scratch model.
- **04 resnet18** - test acc 0.716, gap ~0.28. Highest accuracy via transfer
  learning, but overfits more.
