import os
import numpy as np
import torch
from torch.utils.data import Dataset

DATASET_PATH = "/content/UCI HAR Dataset"


def load_signals(split):
    signals_path = os.path.join(
        DATASET_PATH, split, "Inertial Signals"
    )

    signal_files = [
        f"body_acc_x_{split}.txt",
        f"body_acc_y_{split}.txt",
        f"body_acc_z_{split}.txt",
        f"body_gyro_x_{split}.txt",
        f"body_gyro_y_{split}.txt",
        f"body_gyro_z_{split}.txt",
        f"total_acc_x_{split}.txt",
        f"total_acc_y_{split}.txt",
        f"total_acc_z_{split}.txt",
    ]

    signals = []
    for file in signal_files:
        path = os.path.join(signals_path, file)
        signals.append(np.loadtxt(path))

    # (samples, channels, timesteps)
    return np.stack(signals, axis=1)


def load_labels(split):
    path = os.path.join(DATASET_PATH, split, f"y_{split}.txt")
    y = np.loadtxt(path).astype(int)
    return y - 1  # 1–6 → 0–5


class HARDataset(Dataset):
    def __init__(self, split="train"):
        X = load_signals(split)
        y = load_labels(split)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
