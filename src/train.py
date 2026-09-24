"""
PyTorch Training Script for ActivityLSTM (Bidirectional LSTM) on HAPT Dataset

Features:
- Dedicated exclusively to ActivityLSTM
- Weighted Cross-Entropy Loss (resolves 12-class imbalance)
- Learning rate scheduler (ReduceLROnPlateau)
- Early stopping with best checkpoint restoration
- Training & validation metrics logging and learning curves export
"""

import os
import argparse
import time
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for p in [str(BASE_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score

from dataset import get_dataloaders, CLASS_NAMES
from models import ActivityLSTM, count_parameters
from utils import set_seed, get_device, plot_learning_curves

MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"


def train_one_epoch(model, loader, criterion, optimizer, device, class_weights):
    """Executes single training epoch with mathematically exact weighted loss aggregation."""
    model.train()
    running_weighted_loss = 0.0
    total_weight = 0.0
    all_preds = []
    all_targets = []

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # PyTorch CrossEntropyLoss with reduction='mean' normalizes by sum of target weights in batch
        batch_weight_sum = class_weights[batch_y].sum().item()
        running_weighted_loss += loss.item() * batch_weight_sum
        total_weight += batch_weight_sum

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(batch_y.detach().cpu().numpy())

    epoch_loss = running_weighted_loss / (total_weight if total_weight > 0 else 1.0)
    epoch_acc = accuracy_score(all_targets, all_preds) * 100.0
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, class_weights):
    """Evaluates on validation set and computes Loss, Accuracy, and Macro-F1."""
    model.eval()
    running_weighted_loss = 0.0
    total_weight = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            batch_weight_sum = class_weights[batch_y].sum().item()
            running_weighted_loss += loss.item() * batch_weight_sum
            total_weight += batch_weight_sum

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.detach().cpu().numpy())

    val_loss = running_weighted_loss / (total_weight if total_weight > 0 else 1.0)
    val_acc = accuracy_score(all_targets, all_preds) * 100.0
    val_macro_f1 = f1_score(all_targets, all_preds, average="macro") * 100.0
    return val_loss, val_acc, val_macro_f1


def train_lstm(
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 10,
    seed: int = 42
):
    """Full training pipeline for ActivityLSTM with early stopping and checkpointing."""
    set_seed(seed)
    device = get_device()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = MODELS_DIR / "best_lstm_model.pth"

    print(f"\n=======================================================")
    print(f" Starting Training: ActivityLSTM (Bidirectional LSTM)")
    print(f" Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    print(f" Target Checkpoint: {checkpoint_path}")
    print(f"=======================================================\n")

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(batch_size=batch_size)
    class_weights = class_weights.to(device)

    model = ActivityLSTM(input_dim=6, hidden_dim_1=128, hidden_dim_2=64, num_classes=12).to(device)
    print(f"[MODEL] Total Trainable Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_macro_f1": []
    }

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, class_weights)
        val_loss, val_acc, val_macro_f1 = validate(model, val_loader, criterion, device, class_weights)
        scheduler.step(val_macro_f1)

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Macro-F1: {val_macro_f1:.2f}%"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> [SAVED] Best checkpoint updated! (Val Macro-F1: {best_val_f1:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Validation Macro-F1 did not improve for {patience} epochs. Stopping.")
                break

    total_training_time = time.time() - start_time
    print(f"\n[COMPLETE] Training finished in {total_training_time:.1f} seconds.")
    print(f"[BEST EPOCH] Epoch {best_epoch} with Val Macro-F1: {best_val_f1:.2f}%")

    plot_path = FIGURES_DIR / "lstm_learning_curves.png"
    plot_learning_curves(history, save_path=str(plot_path))

    return history, str(checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ActivityLSTM on HAPT dataset.")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum training epochs")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", "--lr", dest="lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    train_lstm(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed
    )
