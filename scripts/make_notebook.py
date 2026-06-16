"""Generate notebooks/fer2013_colab.ipynb (run locally: python scripts/make_notebook.py)."""
import json
import os

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# FER2013 - Facial Expression Recognition

This is my notebook for the Kaggle "Facial Expression Recognition Challenge".
The task is to look at a 48x48 grayscale face and predict one of 7 emotions.

I run everything from here: it pulls my code from GitHub, downloads the data,
connects to W&B, and trains the models one by one. I start from a very simple
model and slowly make it bigger so I can see when it starts to underfit or
overfit, and then I try to fix that.

One thing before running anything: I need a GPU. Runtime -> Change runtime type
-> T4 GPU, otherwise training is painfully slow.
"""))

cells.append(md("""## 1. Setup

First I clone my repo and install the libraries it needs. The repo holds all the
real code (data loading, the models, the training loop); this notebook just
drives it."""))
cells.append(code("""REPO_URL = "https://github.com/Jegijega/fer2013-facial-expression.git"
REPO_DIR = "fer2013-facial-expression"

import os
if not os.path.exists(REPO_DIR):
    !git clone $REPO_URL
%cd $REPO_DIR
!pip install -q -r requirements.txt
"""))

cells.append(md("""## 2. Get the data from Kaggle

The dataset comes from Kaggle, so I upload my `kaggle.json` token and let the
Kaggle API download it. I also had to join the competition once on its page,
otherwise the download fails with a 403 error."""))
cells.append(code("""from google.colab import files
print("Upload kaggle.json:")
files.upload()

!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!bash scripts/download_data.sh data

import glob
print("CSV files found:", glob.glob("data/**/*.csv", recursive=True))
"""))

cells.append(md("""## 3. Connect to Weights & Biases

This is where all my experiments get logged. When I run the cell it asks for my
API key (from https://wandb.ai/authorize) and after that every run shows up on
my W&B project automatically."""))
cells.append(code("""import wandb
wandb.login()

PROJECT = "fer2013-fer-challenge"
ENTITY = None          # my wandb username, or None for the default
DATA_DIR = "data"
"""))

cells.append(md("""## 4. A quick look at the data

Before training I want to see what I'm working with. The classes are not
balanced at all: there are a lot of "Happy" faces and almost no "Disgust" ones.
I keep this in mind later, because it explains why the model is much worse on
the rare classes."""))
cells.append(code("""import matplotlib.pyplot as plt
from src.data import load_dataframe, find_csv, _pixels_to_array, EMOTIONS

df = load_dataframe(find_csv(DATA_DIR))
print(df["Usage"].value_counts())

train = df[df.Usage == "Training"]
counts = train["emotion"].value_counts().sort_index()
plt.figure(figsize=(8,3))
plt.bar([EMOTIONS[i] for i in counts.index], counts.values)
plt.title("Training class distribution"); plt.xticks(rotation=45); plt.show()

# Show one sample per class
imgs = _pixels_to_array(train["pixels"].iloc[:5000])
labels = train["emotion"].iloc[:5000].to_numpy()
fig, axes = plt.subplots(1, 7, figsize=(14, 2.4))
for c in range(7):
    idx = (labels == c).argmax()
    axes[c].imshow(imgs[idx], cmap="gray"); axes[c].set_title(EMOTIONS[c]); axes[c].axis("off")
plt.show()
"""))

cells.append(md("""## 5. Sanity checks before training

Something we talked about in the lectures: don't trust training curves until you
know the model and the training loop actually work. So for each model I do two
checks.

Forward: I push a random batch through and look at the first loss. For 7 classes
a fresh network should give a loss around ln(7) = 1.946, because at the start it
basically guesses uniformly. If it's way off, something is wrong with the logits
or the loss.

Backward: I run one backward pass and check every parameter gets a real
gradient, then I try to overfit a single small batch to ~100% accuracy. If a
model can't even memorize 16 images, it will never learn the whole dataset, so
that tells me the bug is in the code, not the data."""))
cells.append(code("""import torch
from src.models import MODEL_REGISTRY, build_model, count_parameters
from src.sanity import run_all_checks
from src.utils import get_device

device = get_device(); print("device:", device)

for name in ["linear", "tiny_cnn", "deeper_cnn", "regularized_cnn", "resnet18"]:
    print(f"\\n========== {name} ({count_parameters(build_model(name)):,} params) ==========")
    model = build_model(name).to(device)
    run_all_checks(model, device)
"""))

cells.append(code("""# plot the single-batch overfit to confirm the model can actually learn
from src.sanity import overfit_single_batch
m = build_model("regularized_cnn").to(device)
hist = overfit_single_batch(m, device, steps=200)
plt.plot(hist["losses"]); plt.title("Overfit a single batch (loss should drop to ~0)")
plt.xlabel("step"); plt.ylabel("loss"); plt.show()
print("final acc on the batch:", hist["final_acc"])
"""))

cells.append(md("""## 6. Training the models

Now the actual experiments. I go from the smallest model to the biggest one on
purpose, so I can watch the behaviour change: first underfitting, then
overfitting once the model gets too big with no regularization, and finally a
model that generalizes well. Each cell below is a separate W&B run so I can
compare them side by side. I explain the reasoning for each one in the README."""))
cells.append(code("""from src.engine import TrainConfig, train_model
from src.data import get_dataloaders

# reuse the same loaders for all the non-augmented runs
loaders_plain = get_dataloaders(DATA_DIR, batch_size=128, augment=False)
"""))

cells.append(code("""# model 0: linear baseline, this one should underfit
train_model(TrainConfig(
    model="linear", run_name="00_linear_baseline", group="baselines",
    tags=["baseline","underfit"], data_dir=DATA_DIR, epochs=25, lr=1e-3,
    project=PROJECT, entity=ENTITY), loaders=loaders_plain)
"""))
cells.append(code("""# model 1: small CNN, first real convolutional model
train_model(TrainConfig(
    model="tiny_cnn", run_name="01_tiny_cnn", group="from_scratch",
    tags=["cnn","small"], data_dir=DATA_DIR, epochs=30, lr=1e-3,
    project=PROJECT, entity=ENTITY), loaders=loaders_plain)
"""))
cells.append(code("""# model 2: deeper CNN with no regularization, left like this on purpose so it overfits
train_model(TrainConfig(
    model="deeper_cnn", run_name="02_deeper_cnn_overfit", group="from_scratch",
    tags=["cnn","deep","overfit"], data_dir=DATA_DIR, epochs=40, lr=1e-3,
    augment=False, weight_decay=0.0, project=PROJECT, entity=ENTITY),
    loaders=loaders_plain)
"""))
cells.append(code("""# model 3: same depth but add batchnorm, dropout, augmentation and weight decay
loaders_aug = get_dataloaders(DATA_DIR, batch_size=128, augment=True)
train_model(TrainConfig(
    model="regularized_cnn", run_name="03_regularized_cnn", group="from_scratch",
    tags=["cnn","deep","regularized","best_scratch"], model_kwargs={"dropout":0.4},
    data_dir=DATA_DIR, epochs=60, lr=1e-3, augment=True, weight_decay=1e-4,
    scheduler="cosine", label_smoothing=0.05, grad_clip=5.0,
    early_stop_patience=12, project=PROJECT, entity=ENTITY), loaders=loaders_aug)
"""))
cells.append(code("""# --- Exp 4: ResNet18 transfer learning ---
train_model(TrainConfig(
    model="resnet18", run_name="04_resnet18_pretrained", group="transfer_learning",
    tags=["resnet","transfer"], model_kwargs={"pretrained":True},
    data_dir=DATA_DIR, epochs=50, lr=5e-4, augment=True, optimizer="adamw",
    weight_decay=5e-4, scheduler="cosine", label_smoothing=0.05, grad_clip=5.0,
    early_stop_patience=10, project=PROJECT, entity=ENTITY), loaders=loaders_aug)
"""))

cells.append(md("""## 7. After training

At this point all my runs are in the W&B project, grouped into baselines,
from-scratch CNNs and transfer learning. From there I build a short W&B Report:
the validation accuracy of all runs on one chart, the train-vs-val gap to show
the overfitting, and the confusion matrices. My notes for that write-up are in
reports/REPORT.md.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "fer2013_colab.ipynb")
with open(os.path.abspath(out), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", os.path.abspath(out))
