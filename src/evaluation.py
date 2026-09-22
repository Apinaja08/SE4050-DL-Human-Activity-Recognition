"""Evaluation metrics (accuracy, precision, recall, F1, macro/weighted F1),
classification report, confusion matrix, and evaluation plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Metrics below are computed by hand from a confusion matrix (not via
# sklearn.metrics) because sklearn.metrics pulls in a compiled extension
# (_pairwise_fast) that Windows Smart App Control blocks on this machine.
# The math is identical to sklearn's; this just avoids that blocked import.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
CONFUSION_DIR = PROJECT_ROOT / "results" / "confusion_matrices"

# Validated chart palette (see project's dataviz reference): sequential blue
# for the confusion matrix heatmap, categorical blue/orange (CVD-safe pair)
# for the two training-curve series, and standard ink/gridline chrome.
_BLUE_SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
_SERIES_TRAIN = "#2a78d6"
_SERIES_VAL = "#eb6834"
_INK_PRIMARY = "#0b0b0b"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"


def _confusion_matrix(y_true, y_pred, num_classes):
    indices = y_true * num_classes + y_pred
    counts = np.bincount(indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def _per_class_precision_recall_f1(cm):
    true_positives = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)  # actual count per class (row)
    predicted = cm.sum(axis=0).astype(float)  # predicted count per class (column)

    precision = np.divide(true_positives, predicted, out=np.zeros_like(true_positives), where=predicted > 0)
    recall = np.divide(true_positives, support, out=np.zeros_like(true_positives), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(true_positives), where=denom > 0)

    return precision, recall, f1, support


def _format_classification_report(precision, recall, f1, support, class_names, accuracy):
    lines = [f"{'':<20}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}", ""]
    for i, name in enumerate(class_names):
        lines.append(f"{name:<20}{precision[i]:>10.2f}{recall[i]:>10.2f}{f1[i]:>10.2f}{int(support[i]):>10d}")

    total_support = int(support.sum())
    macro_p, macro_r, macro_f1 = precision.mean(), recall.mean(), f1.mean()
    weighted_p = np.average(precision, weights=support)
    weighted_r = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)

    lines.append("")
    lines.append(f"{'accuracy':<20}{'':>10}{'':>10}{accuracy:>10.2f}{total_support:>10d}")
    lines.append(f"{'macro avg':<20}{macro_p:>10.2f}{macro_r:>10.2f}{macro_f1:>10.2f}{total_support:>10d}")
    lines.append(f"{'weighted avg':<20}{weighted_p:>10.2f}{weighted_r:>10.2f}{weighted_f1:>10.2f}{total_support:>10d}")
    return "\n".join(lines)


def evaluate_model(model, X_test, y_test, class_names=None):
    """Run the model on X_test and compute the full metric suite.

    Accuracy alone is not used to judge the model, since transition classes
    are a small fraction of windows (per STEP 5); macro F1 weighs every
    class equally regardless of frequency, which is what actually reflects
    transition-class performance.
    """
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    num_classes = len(class_names) if class_names is not None else int(max(y_test.max(), y_pred.max())) + 1

    cm = _confusion_matrix(y_test, y_pred, num_classes)
    precision, recall, f1, support = _per_class_precision_recall_f1(cm)
    accuracy = float(np.trace(cm) / cm.sum())

    metrics = {
        "accuracy": accuracy,
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "precision_weighted": float(np.average(precision, weights=support)),
        "recall_weighted": float(np.average(recall, weights=support)),
        "f1_weighted": float(np.average(f1, weights=support)),
    }

    names = class_names if class_names is not None else [str(i) for i in range(num_classes)]
    report = _format_classification_report(precision, recall, f1, support, names, accuracy)

    return {"metrics": metrics, "report": report, "confusion_matrix": cm, "y_pred": y_pred}


def save_metrics_csv(metrics, save_path=METRICS_DIR / "cnn_metrics.csv"):
    """Save the scalar metrics dict as a one-row CSV."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(save_path, index=False)
    return save_path


def plot_confusion_matrix(cm, class_names, save_path=CONFUSION_DIR / "cnn_confusion_matrix.png"):
    """Save the confusion matrix as a sequential-blue heatmap (magnitude encoding)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cmap = LinearSegmentedColormap.from_list("blue_sequential", _BLUE_SEQUENTIAL)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap=cmap)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", color=_INK_PRIMARY, fontsize=8)
    ax.set_yticklabels(class_names, color=_INK_PRIMARY, fontsize=8)
    ax.set_xlabel("Predicted", color=_INK_PRIMARY)
    ax.set_ylabel("True", color=_INK_PRIMARY)
    ax.set_title("CNN Confusion Matrix", color=_INK_PRIMARY)

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else _INK_PRIMARY
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=7)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor="white")
    plt.close(fig)
    return save_path


def plot_training_history(history, save_path=FIGURES_DIR / "cnn_training_history.png"):
    """Plot loss and accuracy curves side by side (never on a shared/dual axis).

    history: a Keras History.history dict (loss/accuracy per epoch).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, (train_key, val_key, title) in zip(
        axes, [("loss", "val_loss", "Loss"), ("accuracy", "val_accuracy", "Accuracy")]
    ):
        ax.plot(history[train_key], label="train", color=_SERIES_TRAIN, linewidth=2)
        ax.plot(history[val_key], label="val", color=_SERIES_VAL, linewidth=2)
        ax.set_title(title, color=_INK_PRIMARY)
        ax.set_xlabel("Epoch", color=_INK_PRIMARY)
        ax.tick_params(colors=_INK_MUTED)
        ax.grid(color=_GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(_GRIDLINE)
        ax.legend(frameon=False, labelcolor=_INK_PRIMARY)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor="white")
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    from tensorflow import keras

    try:
        from .data_loader import load_activity_labels
        from .preprocessing import PROCESSED_DATA_DIR, SCALER_PATH, apply_scaler, build_dataset, encode_activity_labels
        from .training import HISTORY_PATH, MODEL_PATH
    except ImportError:
        from data_loader import load_activity_labels
        from preprocessing import PROCESSED_DATA_DIR, SCALER_PATH, apply_scaler, build_dataset, encode_activity_labels
        from training import HISTORY_PATH, MODEL_PATH
    import json

    import joblib

    print("Loading saved model and scaler...")
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("Building test set...")
    dataset = build_dataset(save_dir=PROCESSED_DATA_DIR)
    X_test = apply_scaler(scaler, dataset["X_test"])
    y_test = encode_activity_labels(dataset["y_test"])

    activity_names = load_activity_labels()
    class_names = [activity_names[i] for i in range(1, 13)]

    result = evaluate_model(model, X_test, y_test, class_names=class_names)

    print("\nTest metrics:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value:.4f}")

    print("\nPer-class classification report:")
    print(result["report"])

    metrics_path = save_metrics_csv(result["metrics"])
    cm_path = plot_confusion_matrix(result["confusion_matrix"], class_names)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {cm_path}")

    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        history_plot_path = plot_training_history(history)
        print(f"Saved training history plot to {history_plot_path}")
    else:
        print(f"\nNo training history found at {HISTORY_PATH} -- re-run src/training.py to generate it.")
