"""
Evaluation Script for ActivityLSTM on HAPT Dataset
Evaluates best_lstm_model.pth on unseen test subjects (Volunteers 26 to 30).

Computes:
- Overall Accuracy
- Macro-averaged & Weighted Precision, Recall, F1-score
- Normalized 12-Class Confusion Matrix Heatmap
- MCC (Matthews Correlation Coefficient)
- ROC-AUC (Macro One-vs-Rest)
- Inference Latency per sample (ms)
- Trainable Parameter Count & Model Memory Footprint
"""

import os
import time
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for p in [str(BASE_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
)

from dataset import get_dataloaders, CLASS_NAMES
from models import ActivityLSTM, count_parameters
from utils import get_device, plot_confusion_matrix, save_metrics, synchronize_device

MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"


def evaluate_lstm(batch_size: int = 64):
    """Evaluates the saved ActivityLSTM checkpoint on unseen test subjects."""
    device = get_device()
    checkpoint_path = MODELS_DIR / "best_lstm_model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Train model first via train.py.")

    print(f"\n=======================================================")
    print(f" Evaluating Model: ActivityLSTM on Unseen Test Subjects (Users 26-30)")
    print(f" Loading checkpoint: {checkpoint_path}")
    print(f"=======================================================\n")

    _, _, test_loader, _ = get_dataloaders(batch_size=batch_size)

    model = ActivityLSTM(input_dim=6, hidden_dim_1=128, hidden_dim_2=64, num_classes=12).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # GPU / MPS Warm-up to avoid driver initialization overhead contaminating latency measurement
    warmup_input = torch.randn(batch_size, 128, 6, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(warmup_input)
    synchronize_device(device)

    all_preds = []
    all_targets = []
    all_probs = []
    latencies = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)

            synchronize_device(device)
            t0 = time.perf_counter()
            logits = model(batch_x)
            synchronize_device(device)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0 / batch_x.size(0))

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro") * 100.0
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted") * 100.0
    macro_precision = precision_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    macro_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    mcc = matthews_corrcoef(all_targets, all_preds) * 100.0

    try:
        roc_auc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro") * 100.0
    except Exception:
        roc_auc = float("nan")

    avg_latency_ms = float(np.mean(latencies))
    param_count = count_parameters(model)
    model_size_mb = float(os.path.getsize(checkpoint_path) / (1024 * 1024))

    print("\n------------------- TEST RESULTS -------------------")
    print(f" Accuracy:           {accuracy:.2f}%")
    print(f" Macro F1-Score:     {macro_f1:.2f}%")
    print(f" Weighted F1-Score:  {weighted_f1:.2f}%")
    print(f" Macro Precision:    {macro_precision:.2f}%")
    print(f" Macro Recall:       {macro_recall:.2f}%")
    print(f" MCC (Matthews):     {mcc:.2f}%")
    print(f" ROC-AUC (Macro OvR): {roc_auc:.2f}%")
    print(f" Avg Latency/Sample: {avg_latency_ms:.3f} ms")
    print(f" Total Parameters:   {param_count:,}")
    print(f" Model File Size:    {model_size_mb:.2f} MB")
    print("----------------------------------------------------\n")

    report_dict = classification_report(
        all_targets, all_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    print("Detailed Classification Report:\n")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(all_targets, all_preds, labels=range(12))
    cm_path = FIGURES_DIR / "lstm_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names=CLASS_NAMES, save_path=str(cm_path), normalize=True)

    metrics_summary = {
        "model_name": "ActivityLSTM",
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "test_macro_precision": macro_precision,
        "test_macro_recall": macro_recall,
        "mcc": mcc,
        "roc_auc_ovr": roc_auc,
        "avg_latency_per_sample_ms": avg_latency_ms,
        "trainable_parameters": param_count,
        "model_size_mb": model_size_mb,
        "per_class_report": report_dict
    }
    metrics_path = METRICS_DIR / "lstm_test_metrics.json"
    save_metrics(metrics_summary, str(metrics_path))

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ActivityLSTM on unseen test subjects.")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64, help="Evaluation batch size")
    args = parser.parse_args()

    evaluate_lstm(batch_size=args.batch_size)

