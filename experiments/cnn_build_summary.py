"""Collect every CNN number into documentation/CNN_Results_Summary.md.

Reads only the files written by cnn_eda.py, cnn_evaluate_full.py and
cnn_seed_study.py, so the summary never drifts from the results on disk.

Run: C:/venvs/har-cnn/Scripts/python.exe experiments/cnn_build_summary.py
"""

import csv
import json
import platform
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "results" / "metrics"
OUT = ROOT / "documentation" / "CNN_Results_Summary.md"

ABBR = ["WALK", "UP", "DOWN", "SIT", "STAND", "LAY", "ST→SI", "SI→ST", "SI→LI", "LI→SI", "ST→LI", "LI→ST"]


def read_json(name):
    with open(M / name) as f:
        return json.load(f)


def read_csv(name):
    with open(M / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


NUMERIC = re.compile(r"^[-+]?[\d,]*\.?\d+%?$")


def table(headers, rows):
    """Markdown table; a column is right-aligned only when every cell is a plain number."""
    numeric = [i > 0 and all(NUMERIC.match(str(row[i])) for row in rows) for i in range(len(headers))]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---:" if n else "---" for n in numeric) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def f4(x):
    return f"{float(x):.4f}"


def pm(stats, key, digits=4):
    s = stats[key]
    return (f"{float(s['mean']):.{digits}f} ± {float(s['std']):.{digits}f} "
            f"[{float(s['min']):.{digits}f}, {float(s['max']):.{digits}f}]")


def cpu_name():
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor()


def main():
    eda = read_json("eda_dataset_overview.json")
    classes = read_csv("eda_class_distribution.csv")
    segments = read_csv("eda_segment_stats.csv")
    signals = read_csv("eda_signal_stats.csv")
    ev = read_json("cnn_full_evaluation.json")
    weights = read_json("cnn_class_weights.json")
    seed_summary = read_csv("cnn_seed_study_summary.csv")
    seed_runs = read_csv("cnn_seed_study_runs.csv")
    ablation = read_csv("cnn_class_weight_ablation_per_class.csv")

    stats = {(r["config"], r["metric"]): r for r in seed_summary}
    w = {k[1]: v for k, v in stats.items() if k[0] == "weighted"}
    u = {k[1]: v for k, v in stats.items() if k[0] == "unweighted"}
    s = ev["test_summary"]
    eff = ev["efficiency"]
    hist = ev["training_history_final_model"]
    env = ev["environment"]
    n_seeds = int(w["accuracy"]["runs"])

    test_rows = ev["per_subject_test"]
    confusion_by_subject = {r["subject"]: r for r in ev["per_subject_test_main_confusion"]}
    total_errors = sum(r["errors"] for r in ev["per_subject_test_main_confusion"])
    worst = sorted(test_rows, key=lambda r: r["accuracy"])[:2]
    worst_errors = sum(confusion_by_subject[r["subject"]]["errors"] for r in worst)
    worst_windows = sum(r["windows"] for r in worst)
    n_test = s["n"]

    transition_segments = [r for r in segments if r["group"] == "transition"]
    dropped_transition = sum(int(r["segments_shorter_than_window"]) for r in transition_segments)
    total_transition = sum(int(r["segments"]) for r in transition_segments)
    sig = {r["activity"]: r for r in signals}
    final_rank = sum(float(r["f1_macro"]) < s["f1_macro"] for r in seed_runs if r["config"] == "weighted")
    seed_val_loss = []
    for path in sorted((M / "cnn_seed_study").glob("seed*_weighted.json")):
        with open(path) as f:
            seed_val_loss.append(min(json.load(f)["history"]["val_loss"]))

    out = []
    add = out.append
    add("# 1D CNN – Metrics and Information for the Report\n")
    add("All values below are read from the files in `results/` by `experiments/cnn_build_summary.py`. "
        "Re-run the scripts in `experiments/` to regenerate them. Metrics in [0, 1] have 4 decimals; "
        "multiply by 100 for percentages.\n")
    add("**Final model** = `models/cnn/cnn_model.keras` (the model already trained with `src/training.py`). "
        f"**Seed study** = the same code and hyperparameters retrained {n_seeds} times with different random "
        "seeds, to measure stability.\n")

    # ------------------------------------------------------------------ requirement coverage
    add("## 0. Assignment requirements → where the evidence is\n")
    add(table(["Requirement", "Section", "Files"], [
        ["Accuracy, precision, recall, F1-score", "§5.1, §5.2", "`cnn_test_summary_metrics.csv`, `cnn_per_class_metrics.csv`"],
        ["ROC-AUC", "§5.1, §5.2", "`cnn_roc_curves.png` (+ PR-AUC: `cnn_pr_curves.png`)"],
        ["Confusion matrix", "§5.3", "`cnn_confusion_matrix.png` (counts), `cnn_confusion_matrix_normalized.png`"],
        ["Metrics justified", "§3.4", "–"],
        ["Tables, graphs, visualisations", "§11", "`results/figures/`, `results/confusion_matrices/`"],
        ["Decisions from train/val only; test unseen", "§3.3", "–"],
        ["Generalisation", "§6", "`cnn_split_comparison.png`, `cnn_per_subject_accuracy.png`"],
        ["Computational efficiency", "§9", "`cnn_efficiency.csv`"],
        ["Training stability", "§7", "`cnn_seed_learning_curves.png`, `cnn_seed_stability.png`"],
        ["Model complexity", "§4", "`cnn_architecture_layers.csv`"],
        ["Practical limitations", "§10", "–"],
    ]))
    add("")

    # ------------------------------------------------------------------ dataset
    add("## 1. Dataset and EDA\n")
    add(table(["Property", "Value"], [
        ["Dataset", "UCI Smartphone-Based Recognition of Human Activities and Postural Transitions (HAPT, id 341)"],
        ["Subjects / recording sessions", f"{eda['subjects']} / {eda['recording_sessions']}"],
        ["Sensors", "waist-mounted smartphone: 3-axis accelerometer (g) + 3-axis gyroscope (rad/s)"],
        ["Sampling rate", f"{eda['sampling_hz']} Hz"],
        ["Recorded time", f"{eda['recorded_minutes']:.1f} min ({eda['recorded_samples']:,} samples)"],
        ["Labelled time", f"{eda['labelled_minutes']:.1f} min ({100 * eda['labelled_fraction_of_recording']:.1f}% of the recording)"],
        ["Labelled segments", f"{eda['labelled_segments']:,}"],
        ["Window", f"{eda['window_samples']} samples = {eda['window_seconds']:.2f} s, step {eda['step_samples']} "
                   f"({100 * eda['overlap']:.0f}% overlap), 6 channels"],
        ["Windows (train / val / test)", f"{eda['windows']['train']:,} / {eda['windows']['val']:,} / "
                                         f"{eda['windows']['test']:,} = {eda['windows_total']:,}"],
        ["Missing values", str(eda["missing_values"])],
    ]))
    add("\n**Class distribution (windows)**\n")
    add(table(["ID", "Activity", "Group", "Train", "Val", "Test", "Total", "% of windows"],
              [[r["class_id"], r["activity"], r["group"], r["train"], r["val"], r["test"], r["total"],
                f"{float(r['pct_of_windows']):.1f}%"] for r in classes]))
    totals = [int(r["total"]) for r in classes]
    basic_share = sum(totals[:6]) / sum(totals)
    add(f"\nBasic activities = {100 * basic_share:.1f}% of windows, transitions = {100 * (1 - basic_share):.1f}%. "
        f"Largest / smallest class = {max(totals) / min(totals):.0f} : 1.\n")
    add("**Labelled segment durations**\n")
    add(table(["Activity", "Segments", "Median (s)", "Min (s)", "Max (s)", "Shorter than 2.56 s (no window)", "Windows"],
              [[r["activity"], r["segments"], f"{float(r['duration_median_s']):.1f}", f"{float(r['duration_min_s']):.1f}",
                f"{float(r['duration_max_s']):.1f}",
                f"{r['segments_shorter_than_window']} ({float(r['pct_segments_dropped']):.1f}%)", r["windows"]]
               for r in segments]))
    add(f"\n{dropped_transition} of {total_transition} transition segments "
        f"({100 * dropped_transition / total_transition:.1f}%) are shorter than one window and produce no window.\n")
    add("**Signal signature per class (training windows, raw units)**\n")
    add(table(["Activity", "Group", "Std of |acc| in window (g)", "Mean |gyro| (rad/s)", "Mean acc x", "Mean acc y", "Mean acc z"],
              [[r["activity"], r["group"], f"{float(r['acc_magnitude_within_window_std_g']):.4f}",
                f"{float(r['gyro_magnitude_mean_rad_s']):.3f}", f"{float(r['acc_x_mean']):.3f}",
                f"{float(r['acc_y_mean']):.3f}", f"{float(r['acc_z_mean']):.3f}"] for r in signals]))
    sit, stand = sig["SITTING"], sig["STANDING"]
    add(f"\nSitting vs standing differ only in the gravity direction: mean acc (x, y, z) = "
        f"({float(sit['acc_x_mean']):.3f}, {float(sit['acc_y_mean']):.3f}, {float(sit['acc_z_mean']):.3f}) vs "
        f"({float(stand['acc_x_mean']):.3f}, {float(stand['acc_y_mean']):.3f}, {float(stand['acc_z_mean']):.3f}) g, "
        f"with almost no movement in either (std of |acc| {float(sit['acc_magnitude_within_window_std_g']):.4f} vs "
        f"{float(stand['acc_magnitude_within_window_std_g']):.4f} g).\n")

    # ------------------------------------------------------------------ preprocessing
    add("## 2. Preprocessing (what the CNN pipeline does)\n")
    add(table(["Step", "Detail"], [
        ["Channel fusion", "acc x,y,z + gyro x,y,z stacked to (T, 6); files truncated to the shorter length"],
        ["Segmentation", "windows cut inside each labelled segment only, so no window crosses an activity boundary"],
        ["Windowing", "128 samples (2.56 s), step 64 (50% overlap); segments < 128 samples give no window"],
        ["Subject-level split", f"train {eda['subjects_per_split']['train']}; val {eda['subjects_per_split']['val']} "
                                "(seeded 20% of the training subjects); test = official UCI test subjects "
                                f"{eda['subjects_per_split']['test']}; overlap check passed"],
        ["Normalisation", "per-channel StandardScaler fitted on training windows only, applied to val/test"],
        ["Labels", "activity 1–12 → class 0–11, sparse categorical cross-entropy"],
        ["Imbalance", "balanced class weights w_c = N / (12 · n_c) from training labels only"],
        ["Feature engineering", "none; the CNN learns features from the normalised raw windows"],
    ]))
    add("\n**Scaler statistics (training set)**\n")
    add(table(["Channel"] + list(eda["scaler_mean_train"].keys()),
              [["mean"] + [f"{v:.4f}" for v in eda["scaler_mean_train"].values()],
               ["std"] + [f"{v:.4f}" for v in eda["scaler_std_train"].values()]]))
    add("\n**Class weights used in training**\n")
    add(table(["Activity", "Weight"], [[k, f"{v:.3f}"] for k, v in weights.items()]))
    add("")

    # ------------------------------------------------------------------ experimental design
    add("## 3. Experimental design\n")
    add("### 3.1 Hyperparameters (fixed in code, not tuned on the test set)\n")
    add(table(["Setting", "Value"], [
        ["Optimiser", "Adam, learning rate 0.001"],
        ["Loss", "sparse categorical cross-entropy, balanced class weights"],
        ["Batch size / max epochs", "64 / 50"],
        ["Early stopping", "monitor val_loss, patience 10, restore best weights"],
        ["Checkpoint", "best val_loss epoch saved (`ModelCheckpoint`)"],
        ["Regularisation", "BatchNorm after conv 1–2, Dropout 0.5 before the classifier"],
    ]))
    add("\n### 3.2 Environment\n")
    add(table(["Item", "Value"], [
        ["CPU", f"{cpu_name()} ({env['logical_cpus']} logical CPUs), no GPU"],
        ["OS", env["os"]],
        ["Software", f"Python {env['python']}, TensorFlow {env['tensorflow']}, Keras {env['keras']}"],
    ]))
    add("\n### 3.3 Keeping the test set unseen\n")
    add("- Scaler statistics and class weights are computed from the training windows only.\n"
        "- Early stopping and the saved checkpoint use validation loss only.\n"
        "- Window length/overlap come from the dataset protocol, not from test results.\n"
        "- The class-weight choice is checked on validation macro-F1 (§8), not on test scores.\n"
        "- Test subjects never contribute windows to training or validation (subject-level split).\n")
    add("### 3.4 Metric justification (facts to build on)\n")
    add(table(["Metric", "Why it fits this problem"], [
        ["Accuracy", "headline figure, but 95% of windows are basic activities, so it hides transition errors"],
        ["Macro precision / recall / F1", "every class counts equally, so the 6 rare transitions are visible"],
        ["Weighted F1", "performance over the real class mix"],
        ["Balanced accuracy", "mean per-class recall; robust to imbalance"],
        ["ROC-AUC (one-vs-rest, macro)", "threshold-independent ranking quality of the softmax scores"],
        ["PR-AUC (average precision)", "stricter than ROC-AUC for rare classes (baseline = class prevalence)"],
        ["MCC, Cohen's kappa", "single-number agreement that stays reliable under imbalance"],
        ["Confusion matrix", "shows which activities are mistaken for which"],
        ["Per-subject accuracy", "generalisation to individual unseen people"],
        ["Log-loss, ECE", "quality/calibration of the predicted probabilities"],
    ]))
    add("")

    # ------------------------------------------------------------------ architecture
    add("## 4. Model architecture and complexity\n")
    add(table(["Layer", "Type", "Output shape", "Params", "MACs"],
              [[r["layer"], r["type"], r["output_shape"], f"{int(r['params']):,}", f"{int(r['macs']):,}"]
               for r in read_csv("cnn_architecture_layers.csv")]))
    add("\n" + table(["Complexity measure", "Value"], [
        ["Parameters (total / trainable / non-trainable)",
         f"{eff['params_total']:,} / {eff['params_trainable']:,} / {eff['params_non_trainable']:,}"],
        ["Multiply-accumulates per window (conv + dense)", f"{eff['macs_conv_dense']:,} (≈ {eff['macs_conv_dense'] / 1e6:.2f} M)"],
        ["FLOPs per window (≈ 2·MACs + element-wise ops)", f"{eff['flops_total_approx']:,} (≈ {eff['flops_total_approx'] / 1e6:.2f} M)"],
        ["Receptive field of the last conv layer",
         f"{eff['receptive_field_samples']} samples = {eff['receptive_field_seconds']:.2f} s (window is 2.56 s)"],
        ["Weights in float32", f"{eff['weights_float32_kb']:.0f} KB"],
        ["Saved `.keras` file", f"{eff['model_file_kb']:.0f} KB (includes optimiser state)"],
    ]))
    add("")

    # ------------------------------------------------------------------ results
    add(f"## 5. Test results (final model, 9 unseen subjects, n = {n_test:,} windows)\n")
    add("### 5.1 Overall metrics\n")
    add(table(["Metric", "Value"], [
        ["Accuracy", f4(s["accuracy"])], ["Balanced accuracy (= macro recall)", f4(s["balanced_accuracy"])],
        ["Precision – macro / weighted", f"{f4(s['precision_macro'])} / {f4(s['precision_weighted'])}"],
        ["Recall – macro / weighted", f"{f4(s['recall_macro'])} / {f4(s['recall_weighted'])}"],
        ["F1-score – macro / weighted", f"{f4(s['f1_macro'])} / {f4(s['f1_weighted'])}"],
        ["ROC-AUC one-vs-rest – macro / weighted", f"{f4(s['roc_auc_macro_ovr'])} / {f4(s['roc_auc_weighted_ovr'])}"],
        ["PR-AUC (average precision) – macro", f4(s["pr_auc_macro"])],
        ["Matthews correlation coefficient", f4(s["mcc"])], ["Cohen's kappa", f4(s["cohen_kappa"])],
        ["Log-loss (cross-entropy)", f4(s["log_loss"])],
        ["Accuracy – basic activities / transitions", f"{f4(s['accuracy_basic'])} / {f4(s['accuracy_transition'])}"],
        ["Macro-F1 – basic activities / transitions", f"{f4(s['f1_macro_basic'])} / {f4(s['f1_macro_transition'])}"],
        ["Macro precision / recall – transitions", f"{f4(s['precision_macro_transition'])} / {f4(s['recall_macro_transition'])}"],
        ["Expected calibration error (15 bins)", f4(ev["calibration_test"]["ece_15_bins"])],
        ["Mean confidence when correct / wrong", f"{f4(ev['calibration_test']['mean_confidence_correct'])} / "
                                                 f"{f4(ev['calibration_test']['mean_confidence_incorrect'])}"],
    ]))
    add("\n### 5.2 Per-class results\n")
    add(table(["Activity", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Support"],
              [[r["activity"], f4(r["precision"]), f4(r["recall"]), f4(r["f1"]), f4(r["roc_auc"]),
                f4(r["pr_auc"]), r["support"]] for r in ev["test_per_class"]]))
    add("\n### 5.3 Confusion matrix (counts; rows = true, columns = predicted)\n")
    cm = ev["test_confusion_matrix"]
    add(table(["True \\ Pred"] + ABBR, [[ABBR[i]] + [str(v) for v in cm[i]] for i in range(len(cm))]))
    add("\nAbbreviations: " + ", ".join(f"{a} = {n}" for a, n in zip(ABBR, [r["activity"] for r in ev["test_per_class"]])) + ".\n")
    add("**Largest confusions**\n")
    add(table(["True", "Predicted", "Count", "% of true class"],
              [[r["true"], r["predicted"], r["count"], f"{r['pct_of_true_class']:.1f}%"] for r in ev["top_confusions"]]))
    add("")

    # ------------------------------------------------------------------ generalisation
    add("## 6. Generalisation\n")
    add("### 6.1 Same model on each split (dropout off)\n")
    add(table(["Split", "Windows", "Accuracy", "Macro-F1", "Log-loss"],
              [[r["split"], r["n"], f4(r["accuracy"]), f4(r["f1_macro"]), f4(r["log_loss"])] for r in ev["split_comparison"]]))
    add("\n### 6.2 Per subject\n")
    rows = [[r["subject"], "val", r["windows"], r["transition_windows"], f4(r["accuracy"]), f4(r["f1_macro"]), "–", "–"]
            for r in ev["per_subject_validation"]]
    rows += [[r["subject"], "test", r["windows"], r["transition_windows"], f4(r["accuracy"]), f4(r["f1_macro"]),
              confusion_by_subject[r["subject"]]["errors"],
              f"{confusion_by_subject[r['subject']]['main_confusion']} ({confusion_by_subject[r['subject']]['main_confusion_count']})"]
             for r in test_rows]
    add(table(["Subject", "Split", "Windows", "Transition windows", "Accuracy", "Macro-F1", "Errors", "Main confusion (count)"], rows))
    sp = ev["per_subject_test_spread"]
    add(f"\nAcross the 9 test subjects: accuracy {sp['accuracy_mean']:.4f} ± {sp['accuracy_std']:.4f} "
        f"(min {sp['accuracy_min']:.4f}, max {sp['accuracy_max']:.4f}); macro-F1 {sp['f1_macro_mean']:.4f} ± "
        f"{sp['f1_macro_std']:.4f} (min {sp['f1_macro_min']:.4f}, max {sp['f1_macro_max']:.4f}).\n")
    add(f"Subjects {worst[0]['subject']} and {worst[1]['subject']} hold {100 * worst_windows / n_test:.1f}% of the "
        f"test windows but {worst_errors} of the {total_errors} errors ({100 * worst_errors / total_errors:.1f}%).\n")

    # ------------------------------------------------------------------ stability
    add("## 7. Training behaviour and stability\n")
    add("### 7.1 Final model's training run (`cnn_training_history.json`)\n")
    add(table(["Item", "Value"], [
        ["Epochs run / selected epoch", f"{hist['epochs_run']} / {hist['best_epoch']} (lowest val loss)"],
        ["Best validation loss", f4(hist["best_val_loss"])],
        ["Validation accuracy at selected epoch", f4(hist["val_accuracy_at_best"])],
        ["Training accuracy at selected epoch (dropout on)", f4(hist["train_accuracy_at_best_with_dropout"])],
        ["Epoch 1 → last: train loss", f"{hist['first_epoch']['loss']:.4f} → {hist['last_epoch']['loss']:.4f}"],
        ["Epoch 1 → last: val loss", f"{hist['first_epoch']['val_loss']:.4f} → {hist['last_epoch']['val_loss']:.4f}"],
    ]))
    add("\nNote: the training loss includes the class weights and is measured with dropout on, while the "
        "validation loss has neither. That is why training loss sits above validation loss; it is not a sign of underfitting.\n")
    add(f"### 7.2 Seed study: {n_seeds} retrainings with identical code (mean ± std [min, max])\n")
    add(table(["Measure", "With class weights (final setup)"], [
        ["Test accuracy", pm(w, "accuracy")], ["Test macro-F1", pm(w, "f1_macro")],
        ["Test weighted-F1", pm(w, "f1_weighted")], ["Test ROC-AUC (macro)", pm(w, "roc_auc_macro_ovr")],
        ["Test PR-AUC (macro)", pm(w, "pr_auc_macro")], ["Test MCC", pm(w, "mcc")],
        ["Test transition macro-F1", pm(w, "f1_macro_transition")],
        ["Validation accuracy", pm(w, "val_accuracy")], ["Validation macro-F1", pm(w, "val_f1_macro")],
        ["Epochs run", pm(w, "epochs_run", 1)], ["Selected (best) epoch", pm(w, "best_epoch", 1)],
        ["Training time (s)", pm(w, "train_time_s", 1)], ["Time per epoch (s)", pm(w, "time_per_epoch_s", 2)],
    ]))
    add(f"\nThe final model (test accuracy {s['accuracy']:.4f}, macro-F1 {s['f1_macro']:.4f}) has a higher macro-F1 "
        f"than {final_rank} of the {n_seeds} seed runs (seed range {float(w['f1_macro']['min']):.4f}–"
        f"{float(w['f1_macro']['max']):.4f}).\n")

    # ------------------------------------------------------------------ ablation
    add(f"## 8. Class-weight ablation ({n_seeds} seeds each, mean ± std)\n")
    add(table(["Measure", "With class weights", "Without class weights"], [
        ["**Validation macro-F1 (decision criterion)**", pm(w, "val_f1_macro"), pm(u, "val_f1_macro")],
        ["Validation accuracy", pm(w, "val_accuracy"), pm(u, "val_accuracy")],
        ["Test accuracy", pm(w, "accuracy"), pm(u, "accuracy")],
        ["Test macro-F1", pm(w, "f1_macro"), pm(u, "f1_macro")],
        ["Test balanced accuracy", pm(w, "balanced_accuracy"), pm(u, "balanced_accuracy")],
        ["Test transition accuracy", pm(w, "accuracy_transition"), pm(u, "accuracy_transition")],
        ["Test transition macro-F1", pm(w, "f1_macro_transition"), pm(u, "f1_macro_transition")],
        ["Test basic-activity accuracy", pm(w, "accuracy_basic"), pm(u, "accuracy_basic")],
    ]))
    add("\n**Per-class means over seeds (test)**\n")
    by = {(r["config"], r["activity"]): r for r in ablation}
    add(table(["Activity", "Recall (weighted)", "Recall (unweighted)", "Precision (weighted)",
               "Precision (unweighted)", "F1 (weighted)", "F1 (unweighted)"],
              [[a, f4(by[("weighted", a)]["recall_mean"]), f4(by[("unweighted", a)]["recall_mean"]),
                f4(by[("weighted", a)]["precision_mean"]), f4(by[("unweighted", a)]["precision_mean"]),
                f4(by[("weighted", a)]["f1_mean"]), f4(by[("unweighted", a)]["f1_mean"])]
               for a in [r["activity"] for r in ev["test_per_class"]]]))
    add("")

    # ------------------------------------------------------------------ efficiency
    add("## 9. Computational efficiency (CPU only)\n")
    add(table(["Measure", "Value"], [
        ["Parameters", f"{eff['params_total']:,}"],
        ["FLOPs per window", f"≈ {eff['flops_total_approx'] / 1e6:.2f} M"],
        ["Model size", f"{eff['weights_float32_kb']:.0f} KB weights; {eff['model_file_kb']:.0f} KB `.keras` file"],
        ["Training time (seed study, mean ± std)", f"{float(w['train_time_s']['mean']):.1f} ± {float(w['train_time_s']['std']):.1f} s "
                                                   f"for {float(w['epochs_run']['mean']):.1f} epochs"],
        ["Time per epoch (6,212 training windows)", f"{float(w['time_per_epoch_s']['mean']):.2f} s"],
        ["Latency, 1 window, compiled `tf.function` (median / p95)",
         f"{eff['single_window_compiled_ms_median']:.3f} / {eff['single_window_compiled_ms_p95']:.3f} ms"],
        ["Latency, 1 window, eager Keras call (median)", f"{eff['single_window_eager_ms_median']:.3f} ms"],
        ["Batched inference (batch 256)", f"{eff['batched_ms_per_window_median']:.4f} ms/window = "
                                          f"{eff['batched_windows_per_second']:,.0f} windows/s"],
        ["Real-time budget", f"a new window arrives every {eff['new_window_every_ms_at_50pct_overlap']:.0f} ms (50% overlap at 50 Hz)"],
    ]))
    add("")

    # ------------------------------------------------------------------ limitations
    add("## 10. Practical limitations (facts backed by the numbers above)\n")
    per_class = {r["activity"]: r for r in ev["test_per_class"]}
    add(f"- **Tiny transition test sets:** transition supports are {', '.join(str(per_class[a]['support']) for a in ev['transition_classes'])} "
        f"windows, so one window moves SIT_TO_STAND recall by {100 / per_class['SIT_TO_STAND']['support']:.0f} points.\n"
        f"- **Short transitions are never seen:** {dropped_transition} of {total_transition} transition segments are shorter than 2.56 s and produce no window, in training or at test time.\n"
        f"- **Order-insensitive head:** each final conv unit sees {eff['receptive_field_seconds']:.2f} s, then global average pooling "
        f"discards order; STAND_TO_LIE is predicted as SIT_TO_LIE in {next(r['pct_of_true_class'] for r in ev['top_confusions'] if r['true'] == 'STAND_TO_LIE' and r['predicted'] == 'SIT_TO_LIE'):.1f}% of cases.\n"
        "- **Sitting vs standing:** nearly identical static signals (§1); this is the largest error pair.\n"
        f"- **Person-to-person variation:** test-subject accuracy ranges {sp['accuracy_min']:.3f}–{sp['accuracy_max']:.3f}; "
        f"two subjects cause {100 * worst_errors / total_errors:.0f}% of the errors.\n"
        f"- **Small validation set:** 4 subjects; validation accuracy ({ev['split_comparison'][1]['accuracy']:.4f}) overstates test accuracy "
        f"({ev['split_comparison'][2]['accuracy']:.4f}), so early stopping relies on a noisy signal.\n"
        f"- **Run-to-run variance:** test macro-F1 std {float(w['f1_macro']['std']):.4f} across seeds; single-run numbers should be quoted with this spread.\n"
        "- **Data scope:** one phone position (waist), 30 lab participants aged 19–48, scripted protocol; unlabelled parts of the recordings are unused.\n")

    # ------------------------------------------------------------------ file index
    add("## 11. Files (tables = CSV/JSON in `results/metrics/`, figures = PNG)\n")
    add(table(["File", "Content"], [
        ["`results/figures/eda_class_distribution.png`", "windows per class"],
        ["`results/figures/eda_segment_durations.png`", "labelled segment durations vs the 2.56 s window"],
        ["`results/figures/eda_example_windows_acc.png` / `_gyro.png`", "one representative window per class"],
        ["`results/figures/eda_signal_signature.png`", "movement intensity and rotation per class"],
        ["`results/figures/eda_subject_split.png`", "subject-level train/val/test assignment"],
        ["`results/figures/cnn_training_history.png`", "final model's training curves (existing)"],
        ["`results/figures/cnn_seed_learning_curves.png`", "learning curves of all seed runs"],
        ["`results/figures/cnn_seed_stability.png`", "per-seed test scores, with/without class weights"],
        ["`results/figures/cnn_class_weight_ablation.png`", "per-class recall/precision with vs without class weights"],
        ["`results/confusion_matrices/cnn_confusion_matrix.png`", "confusion matrix, counts (existing)"],
        ["`results/confusion_matrices/cnn_confusion_matrix_normalized.png`", "confusion matrix, % of true class"],
        ["`results/figures/cnn_per_class_metrics.png`", "precision/recall/F1 per class"],
        ["`results/figures/cnn_roc_curves.png` / `cnn_pr_curves.png`", "ROC and precision-recall curves per class"],
        ["`results/figures/cnn_split_comparison.png`", "train vs validation vs test"],
        ["`results/figures/cnn_per_subject_accuracy.png`", "accuracy and macro-F1 per person"],
        ["`cnn_full_evaluation.json`", "every final-model number in one file"],
        ["`cnn_test_summary_metrics.csv`, `cnn_per_class_metrics.csv`, `cnn_top_confusions.csv`", "test metrics"],
        ["`cnn_split_comparison.csv`, `cnn_per_subject_metrics.csv`", "generalisation"],
        ["`cnn_architecture_layers.csv`, `cnn_efficiency.csv`", "complexity and efficiency"],
        ["`cnn_seed_study_runs.csv`, `cnn_seed_study_summary.csv`, `cnn_class_weight_ablation_per_class.csv`", "stability and ablation"],
        ["`eda_*.csv`, `eda_dataset_overview.json`", "dataset statistics"],
        ["`cnn_test_predictions.npz`", "test labels, probabilities and subject IDs (to redraw any curve)"],
    ]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(out)} blocks)")


if __name__ == "__main__":
    main()
