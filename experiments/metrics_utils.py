"""Metric suite used by the CNN report scripts.

pandas is deliberately not imported: Windows Smart App Control blocks one of
its compiled modules (pandas._libs.join) on the development machine.
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

CLASS_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING",
    "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE", "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND",
]
NUM_CLASSES = len(CLASS_NAMES)
BASIC = np.arange(6)            # zero-based indices of the six basic activities
TRANSITIONS = np.arange(6, 12)  # zero-based indices of the six postural transitions


def _normalise(proba):
    proba = np.asarray(proba, dtype=np.float64)
    return proba / proba.sum(axis=1, keepdims=True)


def classification_metrics(y_true, proba):
    """Full metric suite from integer labels (0-11) and softmax probabilities."""
    y_true = np.asarray(y_true)
    proba = _normalise(proba)
    y_pred = proba.argmax(axis=1)
    labels = np.arange(NUM_CLASSES)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    one_hot = np.eye(NUM_CLASSES)[y_true]
    class_auc = [roc_auc_score(one_hot[:, k], proba[:, k]) for k in labels]
    class_ap = [average_precision_score(one_hot[:, k], proba[:, k]) for k in labels]
    basic = np.isin(y_true, BASIC)

    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision.mean(),
        "recall_macro": recall.mean(),
        "f1_macro": f1.mean(),
        "precision_weighted": np.average(precision, weights=support),
        "recall_weighted": np.average(recall, weights=support),
        "f1_weighted": np.average(f1, weights=support),
        "roc_auc_macro_ovr": roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels),
        "roc_auc_weighted_ovr": roc_auc_score(y_true, proba, multi_class="ovr", average="weighted", labels=labels),
        "pr_auc_macro": np.mean(class_ap),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "log_loss": log_loss(y_true, proba, labels=labels),
        "accuracy_basic": accuracy_score(y_true[basic], y_pred[basic]),
        "accuracy_transition": accuracy_score(y_true[~basic], y_pred[~basic]),
        "f1_macro_basic": f1[BASIC].mean(),
        "f1_macro_transition": f1[TRANSITIONS].mean(),
        "recall_macro_transition": recall[TRANSITIONS].mean(),
        "precision_macro_transition": precision[TRANSITIONS].mean(),
    }
    summary = {k: float(v) for k, v in summary.items()}
    summary["n"] = int(len(y_true))

    per_class = [
        {
            "class_id": k + 1,
            "activity": name,
            "type": "basic" if k in BASIC else "transition",
            "precision": float(precision[k]),
            "recall": float(recall[k]),
            "f1": float(f1[k]),
            "support": int(support[k]),
            "roc_auc": float(class_auc[k]),
            "pr_auc": float(class_ap[k]),
        }
        for k, name in enumerate(CLASS_NAMES)
    ]
    return {"summary": summary, "per_class": per_class, "confusion_matrix": cm.tolist(), "y_pred": y_pred}


def quick_metrics(y_true, proba):
    """Accuracy, macro-F1 and cross-entropy, for the train/val/test comparison."""
    proba = _normalise(proba)
    y_pred = proba.argmax(axis=1)
    labels = np.arange(NUM_CLASSES)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=labels)),
    }


def per_subject_metrics(y_true, y_pred, subjects):
    """Accuracy and macro-F1 for each held-out subject separately.

    Macro-F1 is taken over the classes that subject actually performed, so a
    subject is not penalised for a class it has no windows of.
    """
    rows = []
    for s in np.unique(subjects):
        mask = subjects == s
        present = np.unique(y_true[mask])
        transition = np.isin(y_true[mask], TRANSITIONS)
        rows.append({
            "subject": int(s),
            "windows": int(mask.sum()),
            "transition_windows": int(transition.sum()),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "f1_macro": float(f1_score(y_true[mask], y_pred[mask], labels=present, average="macro", zero_division=0)),
        })
    return rows


def expected_calibration_error(y_true, proba, n_bins=15):
    """ECE: support-weighted gap between confidence and accuracy per confidence bin."""
    proba = _normalise(proba)
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return {
        "ece_15_bins": float(ece),
        "mean_confidence_correct": float(confidence[correct].mean()),
        "mean_confidence_incorrect": float(confidence[~correct].mean()),
    }


def write_csv(path, rows):
    """Write a list of flat dicts to CSV (first row's keys are the header)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
    return path
