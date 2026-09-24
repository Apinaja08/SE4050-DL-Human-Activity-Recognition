"""Complete evaluation of the final CNN (models/cnn/cnn_model.keras).

Produces what src/evaluation.py does not: ROC-AUC / PR-AUC, per-class and
per-subject results, the train/val/test gap, top confusions, calibration, and
efficiency (parameters, FLOPs, receptive field, size, latency). The test set is
only scored here - no decision is taken from it. Existing results files
(cnn_metrics.csv, cnn_confusion_matrix.png, ...) are left untouched.

Run: C:/venvs/har-cnn/Scripts/python.exe experiments/cnn_evaluate_full.py
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style as ps  # noqa: E402
from metrics_utils import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    TRANSITIONS,
    classification_metrics,
    expected_calibration_error,
    per_subject_metrics,
    quick_metrics,
    write_csv,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cnn" / "cnn_model.keras"
SCALER_PATH = ROOT / "models" / "cnn" / "scaler.pkl"
PROCESSED_DIR = ROOT / "data" / "processed"
HISTORY_PATH = ROOT / "results" / "metrics" / "cnn_training_history.json"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
CONFUSION_DIR = ROOT / "results" / "confusion_matrices"

SAMPLING_HZ = 50
SPLITS = ["train", "val", "test"]


def load_split(split, scaler):
    X = np.load(PROCESSED_DIR / f"X_{split}.npy")
    y = np.load(PROCESSED_DIR / f"y_{split}.npy") - 1  # activity 1-12 -> class 0-11
    subjects = np.load(PROCESSED_DIR / f"subjects_{split}.npy")
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype("float32")
    return X, y, subjects


def architecture_table(model):
    """Per-layer shapes/params, multiply-accumulates, and the conv receptive field."""
    rows, macs, other_flops = [], 0, 0
    receptive_field, jump = 1, 1
    for layer in model.layers:
        out_shape = tuple(layer.output.shape[1:])
        layer_macs = 0
        if isinstance(layer, keras.layers.Conv1D):
            length, filters = out_shape
            k, in_ch = layer.kernel_size[0], layer.input.shape[-1]
            layer_macs = length * k * in_ch * filters
            other_flops += length * filters  # bias add
            receptive_field += (k - 1) * jump
            jump *= layer.strides[0]
        elif isinstance(layer, keras.layers.MaxPooling1D):
            receptive_field += (layer.pool_size[0] - 1) * jump
            jump *= layer.strides[0]
            other_flops += int(np.prod(out_shape)) * (layer.pool_size[0] - 1)
        elif isinstance(layer, keras.layers.Dense):
            layer_macs = layer.input.shape[-1] * layer.units
            other_flops += layer.units
        elif isinstance(layer, (keras.layers.BatchNormalization,)):
            other_flops += 2 * int(np.prod(out_shape))  # scale + shift at inference
        elif isinstance(layer, (keras.layers.Activation, keras.layers.GlobalAveragePooling1D)):
            other_flops += int(np.prod(layer.input.shape[1:]))
        macs += layer_macs
        rows.append({
            "layer": layer.name,
            "type": type(layer).__name__,
            "output_shape": "x".join(str(d) for d in out_shape),
            "params": int(layer.count_params()),
            "macs": int(layer_macs),
        })
    return rows, {
        "macs_conv_dense": int(macs),
        "flops_total_approx": int(2 * macs + other_flops),
        "receptive_field_samples": int(receptive_field),
        "receptive_field_seconds": receptive_field / SAMPLING_HZ,
    }


def measure_latency(model, X):
    """Single-window latency (eager and compiled) and batched throughput."""
    one = X[:1]

    for _ in range(20):
        model(one, training=False)
    eager = []
    for _ in range(300):
        t0 = time.perf_counter()
        model(one, training=False)
        eager.append(time.perf_counter() - t0)

    serve = tf.function(lambda x: model(x, training=False))
    for _ in range(20):
        serve(one)
    compiled = []
    for _ in range(1000):
        t0 = time.perf_counter()
        serve(one).numpy()
        compiled.append(time.perf_counter() - t0)

    model.predict(X, batch_size=256, verbose=0)
    batched = []
    for _ in range(5):
        t0 = time.perf_counter()
        model.predict(X, batch_size=256, verbose=0)
        batched.append((time.perf_counter() - t0) / len(X))

    ms = lambda values: float(np.median(values) * 1000)  # noqa: E731
    return {
        "single_window_eager_ms_median": ms(eager),
        "single_window_compiled_ms_median": ms(compiled),
        "single_window_compiled_ms_p95": float(np.percentile(compiled, 95) * 1000),
        "batched_ms_per_window_median": ms(batched),
        "batched_windows_per_second": float(1.0 / np.median(batched)),
        "batch_size_for_throughput": 256,
        "new_window_every_ms_at_50pct_overlap": 64 / SAMPLING_HZ * 1000,
    }


def history_summary(path):
    with open(path) as f:
        h = json.load(f)
    best = int(np.argmin(h["val_loss"]))
    return {
        "epochs_run": len(h["loss"]),
        "best_epoch": best + 1,
        "best_val_loss": float(h["val_loss"][best]),
        "val_accuracy_at_best": float(h["val_accuracy"][best]),
        "train_accuracy_at_best_with_dropout": float(h["accuracy"][best]),
        "train_loss_at_best_with_class_weights": float(h["loss"][best]),
        "max_val_accuracy": float(max(h["val_accuracy"])),
        "first_epoch": {k: float(v[0]) for k, v in h.items()},
        "last_epoch": {k: float(v[-1]) for k, v in h.items()},
    }


def subject_main_confusion(y_true, y_pred, subjects):
    """For each subject: error count and the single most frequent wrong prediction."""
    out = []
    for s in np.unique(subjects):
        mask = (subjects == s) & (y_true != y_pred)
        pairs, counts = np.unique(np.stack([y_true[mask], y_pred[mask]], axis=1), axis=0, return_counts=True)
        top = pairs[np.argmax(counts)] if len(counts) else None
        out.append({
            "subject": int(s),
            "errors": int(mask.sum()),
            "main_confusion": f"{CLASS_NAMES[top[0]]} -> {CLASS_NAMES[top[1]]}" if top is not None else "",
            "main_confusion_count": int(counts.max()) if len(counts) else 0,
        })
    return out


def top_confusions(cm, k=10):
    cm = np.asarray(cm)
    row_totals = cm.sum(axis=1)
    pairs = [
        (cm[i, j], i, j)
        for i in range(NUM_CLASSES)
        for j in range(NUM_CLASSES)
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(reverse=True)
    return [
        {
            "true": CLASS_NAMES[i],
            "predicted": CLASS_NAMES[j],
            "count": int(count),
            "pct_of_true_class": float(100 * count / row_totals[i]),
        }
        for count, i, j in pairs[:k]
    ]


# ----------------------------------------------------------------------------- figures


def plot_confusion_normalised(cm, path):
    cm = np.asarray(cm, dtype=float)
    pct = 100 * cm / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("seq_blue", [ps.SURFACE] + ps.SEQUENTIAL_BLUE)

    fig, ax = ps.plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("CNN confusion matrix, % of each true class (test subjects, n = 3,162)", loc="left")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j] == 0:
                continue
            text = "<1" if pct[i, j] < 1 else f"{pct[i, j]:.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5,
                    color="white" if pct[i, j] > 55 else ps.INK)
    ax.axhline(5.5, color=ps.INK_SECONDARY, linewidth=0.8)
    ax.axvline(5.5, color=ps.INK_SECONDARY, linewidth=0.8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("% of true class", color=ps.INK)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelcolor=ps.INK_SECONDARY)
    return ps.save(fig, path)


def plot_per_class(per_class, path):
    fig, ax = ps.plt.subplots(figsize=(8, 5.6))
    rows = np.arange(NUM_CLASSES)[::-1]  # class 1 at the top
    series = [("precision", "Precision", ps.BLUE, "o"), ("recall", "Recall", ps.ORANGE, "s"),
              ("f1", "F1-score", ps.AQUA, "D")]
    for r, pc in zip(rows, per_class):
        values = [pc[key] for key, *_ in series]
        ax.plot([min(values), max(values)], [r, r], color=ps.GRID, linewidth=2, zorder=1)
    for key, label, color, marker in series:
        ax.scatter([pc[key] for pc in per_class], rows, s=46, color=color, marker=marker,
                   edgecolors=ps.SURFACE, linewidths=1.5, zorder=3, label=label)
    ax.set_yticks(rows, [f"{pc['activity']}  (n={pc['support']})" for pc in per_class], fontsize=8)
    ax.axhline(5.5, color=ps.BASELINE, linewidth=0.8)
    ax.text(1.005, 8.5, "basic\nactivities", transform=ax.get_yaxis_transform(), fontsize=8,
            color=ps.MUTED, va="center")
    ax.text(1.005, 2.5, "postural\ntransitions", transform=ax.get_yaxis_transform(), fontsize=8,
            color=ps.MUTED, va="center")
    ax.set_xlim(0, 1.025)  # room for markers that sit exactly at 1.0
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Score on the test subjects")
    ps.style_axis(ax, grid_axis="x")
    ax.legend(loc="lower left", ncols=3, bbox_to_anchor=(0, 1.02, 1, 0.1), borderaxespad=0, handletextpad=0.3)
    ax.set_title("CNN per-class precision, recall and F1-score", loc="left", pad=26)
    return ps.save(fig, path)


def plot_curve_grid(y_true, proba, per_class, kind, path):
    """Small multiples, one class per panel: ROC or precision-recall."""
    fig, axes = ps.plt.subplots(3, 4, figsize=(10.5, 7.8), sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat):
        positive = (y_true == k).astype(int)
        if kind == "roc":
            fpr, tpr, _ = roc_curve(positive, proba[:, k])
            ax.plot([0, 1], [0, 1], color=ps.BASELINE, linewidth=0.8)
            ax.plot(fpr, tpr, color=ps.BLUE, linewidth=2)
            label = f"AUC {per_class[k]['roc_auc']:.3f}"
        else:
            precision, recall, _ = precision_recall_curve(positive, proba[:, k])
            ax.axhline(positive.mean(), color=ps.BASELINE, linewidth=0.8)
            ax.plot(recall, precision, color=ps.BLUE, linewidth=2)
            label = f"AP {per_class[k]['pr_auc']:.3f}"
        # ROC curves hug the top-left and PR curves the top, so the value goes where the plot is empty.
        ax.text(0.95 if kind == "roc" else 0.05, 0.06, label, transform=ax.transAxes,
                ha="right" if kind == "roc" else "left", fontsize=8.5, color=ps.INK)
        ax.set_title(CLASS_NAMES[k], fontsize=8.5, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ps.style_axis(ax, grid_axis="both")
    xlabel, ylabel = ("False positive rate", "True positive rate") if kind == "roc" else ("Recall", "Precision")
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    title = ("CNN one-vs-rest ROC curves per class (grey line = chance)" if kind == "roc"
             else "CNN precision-recall curves per class (grey line = class prevalence)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="semibold", color=ps.INK)
    fig.tight_layout()
    return ps.save(fig, path)


def plot_per_subject(val_rows, test_rows, overall_test_acc, path):
    groups = [("Validation subjects", val_rows), ("Test subjects (unseen)", test_rows)]
    labels, acc, f1, positions, y = [], [], [], [], 0
    group_label_y = []
    for name, rows in reversed(groups):
        for row in reversed(rows):
            labels.append(f"Subject {row['subject']}  (n={row['windows']})")
            acc.append(row["accuracy"])
            f1.append(row["f1_macro"])
            positions.append(y)
            y += 1
        group_label_y.append((name, y - 0.4))
        y += 1.2

    fig, ax = ps.plt.subplots(figsize=(8, 5.4))
    for p, a, f in zip(positions, acc, f1):
        ax.plot([min(a, f), max(a, f)], [p, p], color=ps.GRID, linewidth=2, zorder=1)
    ax.scatter(acc, positions, s=46, color=ps.BLUE, marker="o", edgecolors=ps.SURFACE, linewidths=1.5,
               zorder=3, label="Accuracy")
    ax.scatter(f1, positions, s=46, color=ps.ORANGE, marker="s", edgecolors=ps.SURFACE, linewidths=1.5,
               zorder=3, label="Macro-F1 (classes the subject performed)")
    ax.axvline(overall_test_acc, color=ps.MUTED, linewidth=0.8)
    ax.set_ylim(-1.6, positions[-1] + 1.6)
    ax.text(overall_test_acc, -1.1, f" overall test accuracy {overall_test_acc:.3f}",
            fontsize=8, color=ps.INK_SECONDARY, va="center")
    ax.set_yticks(positions, labels, fontsize=8)
    for name, gy in group_label_y:
        ax.text(0.0, gy + 0.2, name, transform=ax.get_yaxis_transform(), fontsize=8.5,
                color=ps.INK, fontweight="semibold", va="bottom")
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Score")
    ps.style_axis(ax, grid_axis="x")
    ax.legend(loc="lower left", ncols=2, bbox_to_anchor=(0, 1.02, 1, 0.1), borderaxespad=0)
    ax.set_title("CNN accuracy per person: validation vs unseen test subjects", loc="left", pad=26)
    return ps.save(fig, path)


def plot_split_comparison(split_rows, path):
    metrics = [("accuracy", "Accuracy"), ("f1_macro", "Macro-F1")]
    colors = {"train": ps.BLUE, "val": ps.ORANGE, "test": ps.AQUA}
    names = {"train": "Train (17 subjects)", "val": "Validation (4 subjects)", "test": "Test (9 unseen subjects)"}
    width = 0.11  # thin bars; the rest of each band stays empty
    fig, ax = ps.plt.subplots(figsize=(6.4, 4.2))
    for i, row in enumerate(split_rows):
        xs = np.arange(len(metrics)) + (i - 1) * (width + 0.02)
        values = [row[key] for key, _ in metrics]
        ax.bar(xs, values, width=width, color=colors[row["split"]], label=names[row["split"]])
        for x, v in zip(xs, values):
            ax.text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=ps.INK_SECONDARY)
    ax.set_xticks(np.arange(len(metrics)), [label for _, label in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (evaluated without dropout)")
    ps.style_axis(ax, grid_axis="y")
    ax.legend(loc="lower left", ncols=3, bbox_to_anchor=(0, 1.02, 1, 0.1), borderaxespad=0)
    ax.set_title("CNN generalisation: the same model on each split", loc="left", pad=26)
    return ps.save(fig, path)


# ----------------------------------------------------------------------------- main


def main():
    ps.setup()
    keras.utils.set_random_seed(0)
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    data = {split: load_split(split, scaler) for split in SPLITS}

    proba = {split: model.predict(data[split][0], batch_size=256, verbose=0) for split in SPLITS}
    X_test, y_test, subj_test = data["test"]
    _, y_val, subj_val = data["val"]

    test = classification_metrics(y_test, proba["test"])
    y_pred_test = test.pop("y_pred")
    y_pred_val = proba["val"].argmax(axis=1)

    split_rows = [{"split": s, **quick_metrics(data[s][1], proba[s])} for s in SPLITS]
    val_subjects = per_subject_metrics(y_val, y_pred_val, subj_val)
    test_subjects = per_subject_metrics(y_test, y_pred_test, subj_test)
    layers_table, compute = architecture_table(model)
    latency = measure_latency(model, X_test)

    params_total = int(model.count_params())
    params_trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    efficiency = {
        "params_total": params_total,
        "params_trainable": params_trainable,
        "params_non_trainable": params_total - params_trainable,
        "weights_float32_kb": params_total * 4 / 1024,
        "model_file_kb": os.path.getsize(MODEL_PATH) / 1024,
        **compute,
        **latency,
    }
    test_subject_acc = np.array([r["accuracy"] for r in test_subjects])
    test_subject_f1 = np.array([r["f1_macro"] for r in test_subjects])
    report = {
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "os": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "test_summary": test["summary"],
        "test_per_class": test["per_class"],
        "test_confusion_matrix": test["confusion_matrix"],
        "top_confusions": top_confusions(test["confusion_matrix"]),
        "calibration_test": expected_calibration_error(y_test, proba["test"]),
        "split_comparison": split_rows,
        "per_subject_validation": val_subjects,
        "per_subject_test": test_subjects,
        "per_subject_test_main_confusion": subject_main_confusion(y_test, y_pred_test, subj_test),
        "per_subject_test_spread": {
            "accuracy_mean": float(test_subject_acc.mean()), "accuracy_std": float(test_subject_acc.std(ddof=1)),
            "accuracy_min": float(test_subject_acc.min()), "accuracy_max": float(test_subject_acc.max()),
            "f1_macro_mean": float(test_subject_f1.mean()), "f1_macro_std": float(test_subject_f1.std(ddof=1)),
            "f1_macro_min": float(test_subject_f1.min()), "f1_macro_max": float(test_subject_f1.max()),
        },
        "training_history_final_model": history_summary(HISTORY_PATH),
        "architecture": layers_table,
        "efficiency": efficiency,
        "transition_classes": [CLASS_NAMES[k] for k in TRANSITIONS],
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "cnn_full_evaluation.json", "w") as f:
        json.dump(report, f, indent=2)
    write_csv(METRICS_DIR / "cnn_per_class_metrics.csv", test["per_class"])
    write_csv(METRICS_DIR / "cnn_per_subject_metrics.csv",
              [{"split": "val", **r} for r in val_subjects] + [{"split": "test", **r} for r in test_subjects])
    write_csv(METRICS_DIR / "cnn_split_comparison.csv", split_rows)
    write_csv(METRICS_DIR / "cnn_top_confusions.csv", report["top_confusions"])
    write_csv(METRICS_DIR / "cnn_architecture_layers.csv", layers_table)
    write_csv(METRICS_DIR / "cnn_test_summary_metrics.csv", [test["summary"]])
    write_csv(METRICS_DIR / "cnn_efficiency.csv", [efficiency])
    cm = np.asarray(test["confusion_matrix"])
    write_csv(CONFUSION_DIR / "cnn_confusion_matrix_counts.csv",
              [{"true\\predicted": CLASS_NAMES[i], **{CLASS_NAMES[j]: int(cm[i, j]) for j in range(NUM_CLASSES)}}
               for i in range(NUM_CLASSES)])
    np.savez_compressed(METRICS_DIR / "cnn_test_predictions.npz", y_true=y_test, proba=proba["test"],
                        subjects=subj_test)

    plot_confusion_normalised(test["confusion_matrix"], CONFUSION_DIR / "cnn_confusion_matrix_normalized.png")
    plot_per_class(test["per_class"], FIGURES_DIR / "cnn_per_class_metrics.png")
    plot_curve_grid(y_test, proba["test"], test["per_class"], "roc", FIGURES_DIR / "cnn_roc_curves.png")
    plot_curve_grid(y_test, proba["test"], test["per_class"], "pr", FIGURES_DIR / "cnn_pr_curves.png")
    plot_per_subject(val_subjects, test_subjects, test["summary"]["accuracy"],
                     FIGURES_DIR / "cnn_per_subject_accuracy.png")
    plot_split_comparison(split_rows, FIGURES_DIR / "cnn_split_comparison.png")

    s = test["summary"]
    print(f"test accuracy {s['accuracy']:.4f} | macro-F1 {s['f1_macro']:.4f} | "
          f"ROC-AUC {s['roc_auc_macro_ovr']:.4f} | PR-AUC {s['pr_auc_macro']:.4f} | MCC {s['mcc']:.4f}")
    for row in split_rows:
        print(f"{row['split']:>5}: acc {row['accuracy']:.4f} macro-F1 {row['f1_macro']:.4f} loss {row['log_loss']:.4f}")
    print(json.dumps(efficiency, indent=2))


if __name__ == "__main__":
    main()
