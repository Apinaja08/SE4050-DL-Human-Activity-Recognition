"""
Dataset Pipeline for UCI HAPT (Human Activities and Postural Transitions)
Handles:
1. Parsing raw inertial data (Accelerometer + Gyroscope) & labels.txt
2. Sliding window segmentation (T = 128 timesteps, 50% overlap = 64 stride)
3. Subject-based partitioning (Anti-leakage):
   - Train: Subjects 1 to 21 (~70%)
   - Val:   Subjects 22 to 25 (~15%)
   - Test:  Subjects 26 to 30 (~15%)
4. Z-score normalization (computed ONLY on train set)
5. PyTorch Dataset & DataLoader generation with class weights computation
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "HAPT_Data_Set"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
    7: "STAND_TO_SIT",
    8: "SIT_TO_STAND",
    9: "SIT_TO_LIE",
    10: "LIE_TO_SIT",
    11: "STAND_TO_LIE",
    12: "LIE_TO_STAND"
}

CLASS_NAMES = [ACTIVITY_NAMES[i] for i in range(1, 13)]


class HAPTDataset(Dataset):
    """PyTorch Dataset wrapper for HAPT windowed sensor sequences."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def locate_raw_data_dir():
    """Finds the directory containing RawData or raw files."""
    candidates = [
        RAW_DATA_DIR / "RawData",
        RAW_DATA_DIR,
        *RAW_DATA_DIR.glob("**/RawData")
    ]
    for c in candidates:
        if c.exists() and (c / "labels.txt").exists():
            return c
    labels = list(RAW_DATA_DIR.glob("**/labels.txt"))
    if labels:
        return labels[0].parent
    raise FileNotFoundError(f"Could not locate labels.txt in {RAW_DATA_DIR}. Please run download_data.py first.")


def create_windows_from_raw(raw_dir: Path, window_size: int = 128, stride: int = 64):
    """Reads continuous experiment files and extracts labeled sliding windows."""
    labels_file = raw_dir / "labels.txt"
    labels_df = pd.read_csv(
        labels_file,
        sep=r"\s+",
        header=None,
        names=["exp_id", "user_id", "act_id", "start_idx", "end_idx"]
    )

    windows = []
    labels = []
    subjects = []

    for (exp_id, user_id), group in labels_df.groupby(["exp_id", "user_id"]):
        acc_file = raw_dir / f"acc_exp{exp_id:02d}_user{user_id:02d}.txt"
        gyro_file = raw_dir / f"gyro_exp{exp_id:02d}_user{user_id:02d}.txt"

        if not acc_file.exists() or not gyro_file.exists():
            continue

        acc_data = np.loadtxt(acc_file)
        gyro_data = np.loadtxt(gyro_file)

        min_len = min(len(acc_data), len(gyro_data))
        sensor_data = np.hstack([acc_data[:min_len], gyro_data[:min_len]])

        for _, row in group.iterrows():
            act_id = int(row["act_id"])
            label_0indexed = act_id - 1
            
            start = int(row["start_idx"]) - 1
            end = int(row["end_idx"])

            duration = end - start
            if duration < window_size:
                if duration >= window_size // 2:
                    pad_width = window_size - duration
                    sub_seq = sensor_data[start:end]
                    padded = np.pad(sub_seq, ((0, pad_width), (0, 0)), mode="edge")
                    windows.append(padded)
                    labels.append(label_0indexed)
                    subjects.append(user_id)
                continue

            for w_start in range(start, end - window_size + 1, stride):
                w_end = w_start + window_size
                window = sensor_data[w_start:w_end]
                windows.append(window)
                labels.append(label_0indexed)
                subjects.append(user_id)

    X = np.array(windows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    subj = np.array(subjects, dtype=np.int32)
    return X, y, subj


def load_scaler():
    """Loads saved train mean and std normalization parameters."""
    mean_file = PROCESSED_DATA_DIR / "train_mean.npy"
    std_file = PROCESSED_DATA_DIR / "train_std.npy"
    if mean_file.exists() and std_file.exists():
        return np.load(mean_file), np.load(std_file)
    return None, None


def prepare_and_cache_dataset(window_size: int = 128, stride: int = 64, force_recompute: bool = False):
    """
    Prepares train, validation, and test datasets with subject-wise anti-leakage splitting.
    Caches processed numpy arrays and normalization scaler to data/processed/.
    Validates cache metadata against requested window_size and stride.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_files = [
        PROCESSED_DATA_DIR / "X_train.npy", PROCESSED_DATA_DIR / "y_train.npy",
        PROCESSED_DATA_DIR / "X_val.npy", PROCESSED_DATA_DIR / "y_val.npy",
        PROCESSED_DATA_DIR / "X_test.npy", PROCESSED_DATA_DIR / "y_test.npy",
        PROCESSED_DATA_DIR / "train_mean.npy", PROCESSED_DATA_DIR / "train_std.npy"
    ]
    meta_file = PROCESSED_DATA_DIR / "metadata.json"

    cache_valid = False
    if not force_recompute and all(f.exists() for f in cache_files) and meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            if meta.get("window_size") == window_size and meta.get("stride") == stride:
                cache_valid = True
        except Exception:
            cache_valid = False

    if cache_valid:
        print(f"[INFO] Loading validated cached data (window={window_size}, stride={stride}) from {PROCESSED_DATA_DIR} ...")
        X_train = np.load(PROCESSED_DATA_DIR / "X_train.npy")
        y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")
        X_val = np.load(PROCESSED_DATA_DIR / "X_val.npy")
        y_val = np.load(PROCESSED_DATA_DIR / "y_val.npy")
        X_test = np.load(PROCESSED_DATA_DIR / "X_test.npy")
        y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")
        return X_train, y_train, X_val, y_val, X_test, y_test

    print(f"[INFO] Processing dataset from raw signals (window_size={window_size}, stride={stride})...")
    raw_dir = locate_raw_data_dir()
    X, y, subj = create_windows_from_raw(raw_dir, window_size=window_size, stride=stride)

    train_mask = (subj >= 1) & (subj <= 21)
    val_mask = (subj >= 22) & (subj <= 25)
    test_mask = (subj >= 26) & (subj <= 30)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Z-Score Standardization (Fitted exclusively on X_train to prevent data leakage)
    train_mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    train_std = np.std(X_train, axis=(0, 1), keepdims=True)
    train_std[train_std == 0] = 1.0

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    np.save(PROCESSED_DATA_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DATA_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DATA_DIR / "X_val.npy", X_val)
    np.save(PROCESSED_DATA_DIR / "y_val.npy", y_val)
    np.save(PROCESSED_DATA_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DATA_DIR / "y_test.npy", y_test)
    np.save(PROCESSED_DATA_DIR / "train_mean.npy", train_mean)
    np.save(PROCESSED_DATA_DIR / "train_std.npy", train_std)

    with open(meta_file, "w") as f:
        json.dump({
            "window_size": window_size,
            "stride": stride,
            "num_classes": 12,
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "test_samples": int(len(y_test))
        }, f, indent=2)

    print(f"[SUCCESS] Preprocessed arrays and scaler saved to {PROCESSED_DATA_DIR}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_class_weights(y_train: np.ndarray, num_classes: int = 12) -> torch.Tensor:
    """Computes inverse frequency class weights to balance CrossEntropyLoss."""
    counts = np.bincount(y_train, minlength=num_classes)
    total_samples = len(y_train)
    weights = []
    for count in counts:
        if count > 0:
            w = total_samples / (num_classes * count)
        else:
            w = 1.0
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(batch_size: int = 64, window_size: int = 128, stride: int = 64):
    """Main entry point to get PyTorch DataLoaders and class weights."""
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_and_cache_dataset(
        window_size=window_size, stride=stride
    )
    
    train_dataset = HAPTDataset(X_train, y_train)
    val_dataset = HAPTDataset(X_val, y_val)
    test_dataset = HAPTDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    class_weights = compute_class_weights(y_train, num_classes=12)

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    t_loader, v_loader, te_loader, c_weights = get_dataloaders()
    for batch_x, batch_y in t_loader:
        print(f"[TEST] Batch X: {batch_x.shape}, Batch y: {batch_y.shape}")
        break
    print(f"[TEST] Class Weights: {c_weights}")
