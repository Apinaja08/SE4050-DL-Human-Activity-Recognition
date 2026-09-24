"""Training stability and class-weighting study for the CNN.

Re-trains build_cnn() with the unchanged src/training.py routine (same
hyperparameters as the final model) under several random seeds, with and
without balanced class weights. Inside each run, model selection uses only the
validation set (early stopping + checkpoint on val_loss); the test set is
scored once per run afterwards, for reporting. models/cnn/cnn_model.keras and
the existing results files are not touched.

Run: C:/venvs/har-cnn/Scripts/python.exe experiments/cnn_seed_study.py
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from tensorflow import keras

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style as ps  # noqa: E402
from metrics_utils import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    classification_metrics,
    per_subject_metrics,
    quick_metrics,
    write_csv,
)
from models import build_cnn  # noqa: E402
from training import compute_class_weights, train_cnn  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
SCALER_PATH = ROOT / "models" / "cnn" / "scaler.pkl"
RUN_DIR = ROOT / "results" / "metrics" / "cnn_seed_study"
MODEL_DIR = ROOT / "models" / "cnn" / "seed_study"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

SEEDS = [0, 1, 2, 3, 4]
CONFIGS = {"weighted": True, "unweighted": False}

SUMMARY_KEYS = [
    "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
    "roc_auc_macro_ovr", "pr_auc_macro", "mcc", "cohen_kappa", "log_loss",
    "accuracy_basic", "accuracy_transition", "f1_macro_basic", "f1_macro_transition",
]
RUN_KEYS = ["epochs_run", "best_epoch", "train_time_s", "time_per_epoch_s", "val_accuracy", "val_f1_macro",
            "train_accuracy", "train_f1_macro"]


def load_data():
    scaler = joblib.load(SCALER_PATH)  # fitted on X_train only (src/preprocessing.py)
    data = {}
    for split in ["train", "val", "test"]:
        X = np.load(PROCESSED_DIR / f"X_{split}.npy")
        X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype("float32")
        y = np.load(PROCESSED_DIR / f"y_{split}.npy") - 1
        data[split] = (X, y, np.load(PROCESSED_DIR / f"subjects_{split}.npy"))
    return data


def run_one(seed, config, data):
    tag = f"seed{seed}_{config}"
    out_path = RUN_DIR / f"{tag}.json"
    if out_path.exists():
        print(f"[skip] {tag} already done")
        with open(out_path) as f:
            return json.load(f)

    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, s_te) = data["train"], data["val"], data["test"]
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = build_cnn()
    model_path = MODEL_DIR / f"cnn_{tag}.keras"
    print(f"\n=== {tag} ===", flush=True)
    history, elapsed = train_cnn(model, X_tr, y_tr, X_va, y_va, epochs=50, batch_size=64,
                                 use_class_weight=CONFIGS[config], model_path=model_path, patience=10)

    best = keras.models.load_model(model_path)  # checkpoint of the lowest val_loss epoch
    proba = {split: best.predict(data[split][0], batch_size=256, verbose=0) for split in data}
    test = classification_metrics(y_te, proba["test"])
    y_pred = test.pop("y_pred")
    h = {k: [float(v) for v in values] for k, values in history.history.items()}
    epochs_run = len(h["loss"])
    train_q, val_q = quick_metrics(y_tr, proba["train"]), quick_metrics(y_va, proba["val"])

    record = {
        "seed": seed,
        "config": config,
        "epochs_run": epochs_run,
        "best_epoch": int(np.argmin(h["val_loss"])) + 1,
        "train_time_s": float(elapsed),
        "time_per_epoch_s": float(elapsed / epochs_run),
        "train_accuracy": train_q["accuracy"],
        "train_f1_macro": train_q["f1_macro"],
        "val_accuracy": val_q["accuracy"],
        "val_f1_macro": val_q["f1_macro"],
        "test_summary": test["summary"],
        "test_per_class": test["per_class"],
        "test_confusion_matrix": test["confusion_matrix"],
        "per_subject_test": per_subject_metrics(y_te, y_pred, s_te),
        "history": h,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    s = test["summary"]
    print(f"[done] {tag}: {epochs_run} epochs ({elapsed:.1f}s), best epoch {record['best_epoch']}, "
          f"val acc {val_q['accuracy']:.4f} | test acc {s['accuracy']:.4f} macro-F1 {s['f1_macro']:.4f}",
          flush=True)
    return record


def aggregate(records):
    run_rows, summary_rows, per_class_rows = [], [], []
    for r in records:
        run_rows.append({"config": r["config"], "seed": r["seed"],
                         **{k: r[k] for k in RUN_KEYS}, **{k: r["test_summary"][k] for k in SUMMARY_KEYS}})

    for config in CONFIGS:
        group = [r for r in records if r["config"] == config]
        for key in RUN_KEYS + SUMMARY_KEYS:
            values = np.array([r[key] if key in RUN_KEYS else r["test_summary"][key] for r in group])
            summary_rows.append({
                "config": config, "metric": key, "runs": len(values),
                "mean": float(values.mean()), "std": float(values.std(ddof=1)),
                "min": float(values.min()), "max": float(values.max()),
            })
        for k, name in enumerate(CLASS_NAMES):
            row = {"config": config, "class_id": k + 1, "activity": name}
            for metric in ["precision", "recall", "f1"]:
                values = np.array([r["test_per_class"][k][metric] for r in group])
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
            per_class_rows.append(row)

    write_csv(METRICS_DIR / "cnn_seed_study_runs.csv", run_rows)
    write_csv(METRICS_DIR / "cnn_seed_study_summary.csv", summary_rows)
    write_csv(METRICS_DIR / "cnn_class_weight_ablation_per_class.csv", per_class_rows)
    return summary_rows, per_class_rows


# ----------------------------------------------------------------------------- figures


def plot_learning_curves(records, path):
    """Every weighted-config seed: train vs validation loss and accuracy per epoch."""
    group = [r for r in records if r["config"] == "weighted"]
    fig, axes = ps.plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [("loss", "val_loss", "Loss (train includes class weights)"),
              ("accuracy", "val_accuracy", "Accuracy (train measured with dropout on)")]
    for ax, (train_key, val_key, title) in zip(axes, panels):
        for i, r in enumerate(group):
            epochs = np.arange(1, len(r["history"][train_key]) + 1)
            ax.plot(epochs, r["history"][train_key], color=ps.BLUE, linewidth=1.5, alpha=0.6,
                    label="Train" if i == 0 else None)
            ax.plot(epochs, r["history"][val_key], color=ps.ORANGE, linewidth=1.5, alpha=0.6,
                    label="Validation" if i == 0 else None)
            b = r["best_epoch"]
            ax.scatter([b], [r["history"][val_key][b - 1]], s=40, color=ps.ORANGE, edgecolors=ps.SURFACE,
                       linewidths=1.5, zorder=3, label="Selected epoch (lowest val loss)" if i == 0 else None)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("Epoch")
        ps.style_axis(ax, grid_axis="y")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"CNN learning curves across {len(group)} random seeds (one line per seed)",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold", color=ps.INK)
    fig.tight_layout()
    return ps.save(fig, path)


def plot_seed_spread(records, path):
    """Per-seed test scores for both configs: spread = stability, shift = class-weight effect."""
    metrics = [("accuracy", "Accuracy"), ("f1_macro", "Macro-F1"), ("roc_auc_macro_ovr", "ROC-AUC (macro)"),
               ("accuracy_transition", "Transition accuracy"), ("f1_macro_transition", "Transition macro-F1")]
    configs = [("weighted", "With class weights (final setup)", ps.BLUE, "o", 0.12),
               ("unweighted", "Without class weights", ps.ORANGE, "s", -0.12)]
    fig, ax = ps.plt.subplots(figsize=(8, 4.4))
    rows = np.arange(len(metrics))[::-1]
    for config, label, color, marker, offset in configs:
        group = [r for r in records if r["config"] == config]
        for row, (key, _) in zip(rows, metrics):
            values = np.array([r["test_summary"][key] for r in group])
            ax.plot([values.min(), values.max()], [row + offset] * 2, color=ps.GRID, linewidth=2, zorder=1)
            ax.scatter(values, [row + offset] * len(values), s=34, color=color, marker=marker,
                       edgecolors=ps.SURFACE, linewidths=1.5, zorder=3,
                       label=label if key == metrics[0][0] else None)
            ax.plot([values.mean()] * 2, [row + offset - 0.09, row + offset + 0.09], color=ps.INK, linewidth=1.5,
                    zorder=4)
    ax.set_yticks(rows, [label for _, label in metrics])
    ax.set_xlabel("Test score (each dot = one seed; black tick = mean)")
    ps.style_axis(ax, grid_axis="x")
    ax.legend(loc="lower left", ncols=2, bbox_to_anchor=(0, 1.02, 1, 0.1), borderaxespad=0)
    ax.set_title("CNN run-to-run stability over 5 seeds, with and without class weights", loc="left", pad=26)
    return ps.save(fig, path)


def plot_ablation(per_class_rows, path):
    """Dumbbell per class: mean recall and precision without -> with class weights."""
    by = {(r["config"], r["class_id"]): r for r in per_class_rows}
    fig, axes = ps.plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    rows = np.arange(NUM_CLASSES)[::-1]
    for ax, metric, title in zip(axes, ["recall", "precision"], ["Recall", "Precision"]):
        for row, k in zip(rows, range(1, NUM_CLASSES + 1)):
            a, b = by[("unweighted", k)][f"{metric}_mean"], by[("weighted", k)][f"{metric}_mean"]
            ax.plot([a, b], [row, row], color=ps.GRID, linewidth=2, zorder=1)
        ax.scatter([by[("unweighted", k)][f"{metric}_mean"] for k in range(1, NUM_CLASSES + 1)], rows, s=46,
                   color=ps.ORANGE, marker="s", edgecolors=ps.SURFACE, linewidths=1.5, zorder=3,
                   label="Without class weights")
        ax.scatter([by[("weighted", k)][f"{metric}_mean"] for k in range(1, NUM_CLASSES + 1)], rows, s=46,
                   color=ps.BLUE, marker="o", edgecolors=ps.SURFACE, linewidths=1.5, zorder=3,
                   label="With class weights")
        ax.axhline(5.5, color=ps.BASELINE, linewidth=0.8)
        ax.set_xlim(0, 1.025)
        ax.set_title(f"{title} (mean of 5 seeds)", loc="left", fontsize=9.5)
        ps.style_axis(ax, grid_axis="x")
    axes[0].set_yticks(rows, CLASS_NAMES, fontsize=8)
    axes[0].legend(loc="lower left", ncols=2, bbox_to_anchor=(0, 1.08, 2, 0.1), borderaxespad=0)
    fig.suptitle("Effect of balanced class weights on each class (CNN, test subjects)",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold", color=ps.INK)
    fig.tight_layout()
    return ps.save(fig, path)


def main():
    parser = argparse.ArgumentParser(description="CNN seed / class-weight study")
    parser.add_argument("--plots-only", action="store_true", help="Re-draw figures from saved run files")
    args = parser.parse_args()

    ps.setup()
    data = None if args.plots_only else load_data()
    records = []
    for config in CONFIGS:
        for seed in SEEDS:
            if args.plots_only:
                with open(RUN_DIR / f"seed{seed}_{config}.json") as f:
                    records.append(json.load(f))
            else:
                records.append(run_one(seed, config, data))

    summary_rows, per_class_rows = aggregate(records)
    weights = compute_class_weights(np.load(PROCESSED_DIR / "y_train.npy") - 1)
    with open(METRICS_DIR / "cnn_class_weights.json", "w") as f:
        json.dump({CLASS_NAMES[k]: w for k, w in weights.items()}, f, indent=2)

    plot_learning_curves(records, FIGURES_DIR / "cnn_seed_learning_curves.png")
    plot_seed_spread(records, FIGURES_DIR / "cnn_seed_stability.png")
    plot_ablation(per_class_rows, FIGURES_DIR / "cnn_class_weight_ablation.png")

    for row in summary_rows:
        if row["metric"] in ("accuracy", "f1_macro", "roc_auc_macro_ovr", "f1_macro_transition",
                             "epochs_run", "train_time_s"):
            print(f"{row['config']:>10} {row['metric']:<22} {row['mean']:.4f} ± {row['std']:.4f} "
                  f"[{row['min']:.4f}, {row['max']:.4f}]")


if __name__ == "__main__":
    main()
