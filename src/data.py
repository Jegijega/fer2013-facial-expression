from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 48

# grayscale mean/std of the training split
FER_MEAN = 0.5077
FER_STD = 0.2550


def find_csv(data_dir: str) -> str:
    candidates = ["fer2013.csv", "icml_face_data.csv", "fer2013/fer2013.csv"]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".csv"):
                path = os.path.join(root, f)
                try:
                    cols = [c.strip().lower() for c in pd.read_csv(path, nrows=1).columns]
                    if "pixels" in cols:
                        return path
                except Exception:
                    continue
    raise FileNotFoundError(f"No FER2013 csv found in {data_dir}.")


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _pixels_to_array(pixel_strings: pd.Series) -> np.ndarray:
    arr = np.array(
        [np.asarray(p.split(), dtype=np.uint8) for p in pixel_strings],
        dtype=np.uint8,
    )
    return arr.reshape(-1, IMG_SIZE, IMG_SIZE)


class FER2013Dataset(Dataset):
    def __init__(self, images: np.ndarray, labels: Optional[np.ndarray], transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        if self.labels is None:
            return img
        return img, int(self.labels[idx])


def build_transforms(augment: bool):
    norm = T.Normalize(mean=[FER_MEAN], std=[FER_STD])
    if augment:
        return T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([T.ToPILImage(), T.ToTensor(), norm])


def get_datasets(data_dir: str, augment: bool = False):
    df = load_dataframe(find_csv(data_dir))
    train_tf = build_transforms(augment)
    eval_tf = build_transforms(False)

    def subset(usage):
        rows = df[df["Usage"] == usage]
        return _pixels_to_array(rows["pixels"]), rows["emotion"].to_numpy()

    tr_x, tr_y = subset("Training")
    va_x, va_y = subset("PublicTest")
    te_x, te_y = subset("PrivateTest")
    return (
        FER2013Dataset(tr_x, tr_y, train_tf),
        FER2013Dataset(va_x, va_y, eval_tf),
        FER2013Dataset(te_x, te_y, eval_tf),
    )


def get_dataloaders(data_dir, batch_size=128, augment=False, num_workers=2):
    train_ds, val_ds, test_ds = get_datasets(data_dir, augment=augment)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def compute_class_weights(data_dir: str) -> torch.Tensor:
    df = load_dataframe(find_csv(data_dir))
    counts = df[df["Usage"] == "Training"]["emotion"].value_counts().sort_index()
    counts = counts.reindex(range(NUM_CLASSES), fill_value=0).to_numpy()
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)
