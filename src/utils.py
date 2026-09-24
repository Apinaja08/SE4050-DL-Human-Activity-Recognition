"""
Utility functions for HAPT Deep Learning Project:
- Reproducibility (random seed initialization)
- Hardware device detection
- Plotting functions (Learning curves, Confusion matrix heatmap)
- Metrics export
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch


def set_seed(seed: int = 42):
    """Sets random seed across all libraries for deterministic reproducibility."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    print(f"[INFO] Random seed set to {seed} (Reproducibility guaranteed).")


def get_device() -> torch.device:
    """Detects and returns optimal hardware device (Apple Silicon MPS -> CUDA -> CPU)."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Hardware Acceleration: Apple Silicon GPU (MPS) detected!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Hardware Acceleration: NVIDIA GPU ({torch.cuda.get_device_name(0)}) detected!")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU for computation.")
    return device


def synchronize_device(device: torch.device):
    """Ensures GPU/MPS kernels finish execution for accurate latency benchmarking."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()



def plot_learning_curves(history: dict, save_path: str = None):
    """Plots training and validation loss and accuracy side-by-side."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2, markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r--s", label="Validation Loss", linewidth=2, markersize=4)
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Weighted Cross-Entropy Loss", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Accuracy / F1 curve
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", linewidth=2, markersize=4)
    axes[1].plot(epochs, history["val_acc"], "r--s", label="Val Accuracy", linewidth=2, markersize=4)
    if "val_macro_f1" in history:
        axes[1].plot(epochs, history["val_macro_f1"], "g-^", label="Val Macro F1", linewidth=2, markersize=4)
    axes[1].set_title("Accuracy & Macro F1 Evolution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Score (%)", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SUCCESS] Saved learning curves to: {save_path}")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str = None, normalize: bool = True):
    """Plots a high-resolution normalized confusion matrix heatmap with percentage values."""
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".1%"
    else:
        cm_display = cm
        fmt = "d"

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        annot_kws={"size": 9}
    )
    plt.title("Normalized 12-Class Confusion Matrix (HAPT)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, fontweight="semibold")
    plt.ylabel("Ground Truth Class", fontsize=12, fontweight="semibold")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SUCCESS] Saved confusion matrix to: {save_path}")
    plt.close()


def save_metrics(metrics: dict, save_path: str):
    """Saves dictionary of metrics as nicely formatted JSON."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[SUCCESS] Saved test metrics to: {save_path}")
