"""Dataset description and exploratory data analysis for the CNN pipeline.

Everything is computed from data/raw/RawData (labels + raw signals) and the
windows actually built by src/preprocessing.py (data/processed/*.npy), so the
numbers describe exactly the data the CNN was trained and tested on.

Run: C:/venvs/har-cnn/Scripts/python.exe experiments/cnn_eda.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style as ps  # noqa: E402
from metrics_utils import CLASS_NAMES, NUM_CLASSES, write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "RawData"
PROCESSED_DIR = ROOT / "data" / "processed"
SCALER_PATH = ROOT / "models" / "cnn" / "scaler.pkl"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

SAMPLING_HZ = 50
WINDOW, STEP = 128, 64
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
SPLITS = ["train", "val", "test"]
GROUP = {k: ("dynamic" if k < 3 else "static" if k < 6 else "transition") for k in range(NUM_CLASSES)}


def windows_from_length(length):
    return 0 if length < WINDOW else (length - WINDOW) // STEP + 1


def load_processed():
    data = {}
    for split in SPLITS:
        data[split] = {
            "X": np.load(PROCESSED_DIR / f"X_{split}.npy"),
            "y": np.load(PROCESSED_DIR / f"y_{split}.npy") - 1,
            "subjects": np.load(PROCESSED_DIR / f"subjects_{split}.npy"),
        }
    return data


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def dataset_overview(labels, data):
    """Headline numbers: subjects, recordings, labelled time, windows per split."""
    recorded = sum(count_lines(p) for p in sorted(RAW_DIR.glob("acc_exp*_user*.txt")))
    labelled = int((labels[:, 4] - labels[:, 3] + 1).sum())
    scaler = joblib.load(SCALER_PATH)
    return {
        "subjects": int(len(np.unique(labels[:, 1]))),
        "recording_sessions": int(len(np.unique(labels[:, 0]))),
        "labelled_segments": int(len(labels)),
        "sampling_hz": SAMPLING_HZ,
        "recorded_samples": int(recorded),
        "recorded_minutes": recorded / SAMPLING_HZ / 60,
        "labelled_samples": labelled,
        "labelled_minutes": labelled / SAMPLING_HZ / 60,
        "labelled_fraction_of_recording": labelled / recorded,
        "window_samples": WINDOW,
        "window_seconds": WINDOW / SAMPLING_HZ,
        "step_samples": STEP,
        "overlap": 1 - STEP / WINDOW,
        "windows": {split: int(len(data[split]["y"])) for split in SPLITS},
        "windows_total": int(sum(len(data[s]["y"]) for s in SPLITS)),
        "subjects_per_split": {split: sorted(int(s) for s in np.unique(data[split]["subjects"])) for split in SPLITS},
        "missing_values": int(sum(np.isnan(data[s]["X"]).sum() for s in SPLITS)),
        "scaler_mean_train": dict(zip(CHANNELS, scaler.mean_.round(5).tolist())),
        "scaler_std_train": dict(zip(CHANNELS, np.sqrt(scaler.var_).round(5).tolist())),
        "raw_value_range_all_splits": {
            ch: [float(min(data[s]["X"][..., i].min() for s in SPLITS)),
                 float(max(data[s]["X"][..., i].max() for s in SPLITS))]
            for i, ch in enumerate(CHANNELS)
        },
    }


def class_distribution(data):
    rows = []
    total_all = sum(len(data[s]["y"]) for s in SPLITS)
    for k, name in enumerate(CLASS_NAMES):
        counts = {s: int((data[s]["y"] == k).sum()) for s in SPLITS}
        total = sum(counts.values())
        rows.append({"class_id": k + 1, "activity": name, "group": GROUP[k], **counts,
                     "total": total, "pct_of_windows": 100 * total / total_all})
    return rows


def segment_stats(labels, class_rows):
    rows = []
    for k, name in enumerate(CLASS_NAMES):
        seg = labels[labels[:, 2] == k + 1]
        lengths = seg[:, 4] - seg[:, 3] + 1
        seconds = lengths / SAMPLING_HZ
        dropped = int((lengths < WINDOW).sum())
        windows = int(sum(windows_from_length(n) for n in lengths))
        if windows != class_rows[k]["total"]:
            print(f"[check] {name}: {windows} windows expected from labels.txt, "
                  f"{class_rows[k]['total']} found in data/processed")
        rows.append({
            "class_id": k + 1, "activity": name, "group": GROUP[k],
            "segments": int(len(seg)),
            "total_minutes": float(seconds.sum() / 60),
            "duration_mean_s": float(seconds.mean()),
            "duration_median_s": float(np.median(seconds)),
            "duration_min_s": float(seconds.min()),
            "duration_max_s": float(seconds.max()),
            "segments_shorter_than_window": dropped,
            "pct_segments_dropped": 100 * dropped / len(seg),
            "windows": windows,
            "windows_per_segment_mean": windows / len(seg),
        })
    return rows


def signal_stats(data):
    """Per-class signature on the raw (unscaled) training windows."""
    X, y = data["train"]["X"], data["train"]["y"]
    rows = []
    for k, name in enumerate(CLASS_NAMES):
        w = X[y == k]
        acc_mag = np.linalg.norm(w[..., :3], axis=-1)
        gyro_mag = np.linalg.norm(w[..., 3:], axis=-1)
        rows.append({
            "class_id": k + 1, "activity": name, "group": GROUP[k], "windows": int(len(w)),
            "acc_magnitude_mean_g": float(acc_mag.mean()),
            "acc_magnitude_within_window_std_g": float(acc_mag.std(axis=1).mean()),
            "gyro_magnitude_mean_rad_s": float(gyro_mag.mean()),
            **{f"{ch}_mean": float(w[..., i].mean()) for i, ch in enumerate(CHANNELS)},
        })
    return rows


def subject_windows(data):
    rows = []
    for split in SPLITS:
        for s in np.unique(data[split]["subjects"]):
            mask = data[split]["subjects"] == s
            transitions = int((data[split]["y"][mask] >= 6).sum())
            rows.append({"subject": int(s), "split": split, "windows": int(mask.sum()),
                         "basic_windows": int(mask.sum()) - transitions, "transition_windows": transitions})
    return sorted(rows, key=lambda r: r["subject"])


def representative_windows(data):
    """For each class, the training window closest to the class median signature."""
    X, y = data["train"]["X"], data["train"]["y"]
    picks = {}
    for k in range(NUM_CLASSES):
        w = X[y == k]
        feats = np.stack([np.linalg.norm(w[..., 3:], axis=-1).mean(1),
                          np.linalg.norm(w[..., :3], axis=-1).std(1)], axis=1)
        z = (feats - np.median(feats, axis=0)) / (feats.std(axis=0) + 1e-9)
        picks[k] = w[np.argmin((z ** 2).sum(axis=1))]
    return picks


# ----------------------------------------------------------------------------- figures

GROUP_STYLE = {
    "basic": ("Basic activity", ps.BLUE),
    "transition": ("Postural transition", ps.ORANGE),
}


def plot_class_distribution(class_rows, path):
    fig, ax = ps.plt.subplots(figsize=(8, 4.8))
    rows = np.arange(NUM_CLASSES)[::-1]
    totals = [r["total"] for r in class_rows]
    for row, r in zip(rows, class_rows):
        key = "basic" if r["group"] != "transition" else "transition"
        ax.barh(row, r["total"], height=0.5, color=GROUP_STYLE[key][1])
        ax.text(r["total"] + 20, row, f"{r['total']:,}", va="center", fontsize=8, color=ps.INK_SECONDARY)
    ax.set_yticks(rows, [r["activity"] for r in class_rows], fontsize=8)
    ax.set_xlim(0, max(totals) * 1.12)
    ax.set_xlabel("Windows (128 samples, 50% overlap), all 30 subjects")
    ps.style_axis(ax, grid_axis="x")
    ax.legend(handles=[Patch(color=color, label=label) for label, color in GROUP_STYLE.values()],
              loc="lower left", ncols=2, bbox_to_anchor=(0, 1.02, 1, 0.1), borderaxespad=0)
    ratio = max(totals) / min(totals)
    ax.set_title(f"Class distribution: largest class has {ratio:.0f}x the windows of the smallest",
                 loc="left", pad=26)
    return ps.save(fig, path)


def plot_segment_durations(labels, path):
    fig, ax = ps.plt.subplots(figsize=(8.5, 5))
    rng = np.random.default_rng(0)
    rows = np.arange(NUM_CLASSES)[::-1]
    for row, k in zip(rows, range(NUM_CLASSES)):
        seg = labels[labels[:, 2] == k + 1]
        seconds = (seg[:, 4] - seg[:, 3] + 1) / SAMPLING_HZ
        key = "basic" if GROUP[k] != "transition" else "transition"
        jitter = rng.uniform(-0.22, 0.22, len(seconds))
        ax.scatter(seconds, row + jitter, s=12, color=GROUP_STYLE[key][1], alpha=0.55, linewidths=0,
                   label=GROUP_STYLE[key][0] if k in (0, 6) else None)
        ax.plot([np.median(seconds)] * 2, [row - 0.32, row + 0.32], color=ps.INK, linewidth=1.5)
    ax.axvline(WINDOW / SAMPLING_HZ, color=ps.INK_SECONDARY, linewidth=0.8)
    ax.text(WINDOW / SAMPLING_HZ, NUM_CLASSES - 0.35, "  window length 2.56 s", fontsize=8,
            color=ps.INK_SECONDARY, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.set_yticks(rows, CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Labelled segment duration, seconds (log scale; black tick = median)")
    ps.style_axis(ax, grid_axis="x")
    ax.legend(loc="lower left", ncols=2, bbox_to_anchor=(0, 1.02, 1, 0.1), borderaxespad=0)
    ax.set_title("How long each labelled activity segment lasts", loc="left", pad=26)
    return ps.save(fig, path)


def plot_example_windows(picks, sensor, path):
    offset, unit = (0, "acceleration (g)") if sensor == "acc" else (3, "angular velocity (rad/s)")
    t = np.arange(WINDOW) / SAMPLING_HZ
    fig, axes = ps.plt.subplots(3, 4, figsize=(11, 7), sharex=True, sharey=True)
    colors = [ps.BLUE, ps.ORANGE, ps.AQUA]
    for k, ax in enumerate(axes.flat):
        for i, axis_name in enumerate("xyz"):
            ax.plot(t, picks[k][:, offset + i], color=colors[i], linewidth=1.5,
                    label=f"{sensor} {axis_name}" if k == 0 else None)
        ax.set_title(CLASS_NAMES[k], fontsize=8.5, loc="left")
        ps.style_axis(ax, grid_axis="y")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel(unit)
    fig.legend(loc="upper right", ncols=3, bbox_to_anchor=(0.99, 1.0))
    name = "Accelerometer" if sensor == "acc" else "Gyroscope"
    fig.suptitle(f"{name}: one representative 2.56 s window per class (raw units, training data)",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold", color=ps.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return ps.save(fig, path)


def plot_signal_signature(signal_rows, path):
    """Two measures, two panels (never a shared dual axis)."""
    groups = {"dynamic": ("Dynamic", ps.BLUE), "static": ("Static", ps.ORANGE),
              "transition": ("Transition", ps.AQUA)}
    fig, axes = ps.plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    rows = np.arange(NUM_CLASSES)[::-1]
    panels = [("acc_magnitude_within_window_std_g", "Movement intensity: std of |acc| within a window (g, log)"),
              ("gyro_magnitude_mean_rad_s", "Rotation: mean |gyro| (rad/s)")]
    for ax, (key, title) in zip(axes, panels):
        for row, r in zip(rows, signal_rows):
            ax.barh(row, r[key], height=0.5, color=groups[r["group"]][1])
            ax.text(r[key] * (1.08 if key.startswith("acc") else 1.0) + (0 if key.startswith("acc") else 0.015),
                    row, f"{r[key]:.3f}", va="center", fontsize=7.5, color=ps.INK_SECONDARY)
        if key.startswith("acc"):
            ax.set_xscale("log")
            ax.set_xlim(0.003, 1.2)
            ax.set_xticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
            ax.minorticks_off()
        else:
            ax.set_xlim(0, 1.15)
        ax.set_title(title, loc="left", fontsize=9)
        ps.style_axis(ax, grid_axis="x")
    axes[0].set_yticks(rows, [r["activity"] for r in signal_rows], fontsize=8)
    axes[0].legend(handles=[Patch(color=color, label=label) for label, color in groups.values()],
                   loc="lower left", ncols=3, bbox_to_anchor=(0, 1.1, 1, 0.1), borderaxespad=0)
    fig.suptitle("Signal signature per class (training windows): what separates the groups",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold", color=ps.INK)
    fig.tight_layout()
    return ps.save(fig, path)


def plot_subject_split(subject_rows, path):
    colors = {"train": (ps.BLUE, "Train (17 subjects)"), "val": (ps.ORANGE, "Validation (4)"),
              "test": (ps.AQUA, "Test (9, official UCI test subjects)")}
    fig, ax = ps.plt.subplots(figsize=(10, 3.8))
    for split, (color, label) in colors.items():
        rows = [r for r in subject_rows if r["split"] == split]
        ax.bar([r["subject"] for r in rows], [r["windows"] for r in rows], width=0.55, color=color, label=label)
    ax.set_xticks(range(1, 31), [str(s) for s in range(1, 31)], fontsize=8)
    ax.set_xlim(0.3, 30.7)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Windows")
    ps.style_axis(ax, grid_axis="y")
    ax.legend(loc="lower left", ncols=3, bbox_to_anchor=(0, 1.02, 1, 0.1), borderaxespad=0)
    ax.set_title("Subject-level split: every person's windows sit in exactly one split", loc="left", pad=26)
    return ps.save(fig, path)


def main():
    ps.setup()
    labels = np.loadtxt(RAW_DIR / "labels.txt", dtype=int)  # exp, user, activity, start, end
    data = load_processed()

    class_rows = class_distribution(data)
    seg_rows = segment_stats(labels, class_rows)
    sig_rows = signal_stats(data)
    subj_rows = subject_windows(data)
    overview = dataset_overview(labels, data)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "eda_dataset_overview.json", "w") as f:
        json.dump(overview, f, indent=2)
    write_csv(METRICS_DIR / "eda_class_distribution.csv", class_rows)
    write_csv(METRICS_DIR / "eda_segment_stats.csv", seg_rows)
    write_csv(METRICS_DIR / "eda_signal_stats.csv", sig_rows)
    write_csv(METRICS_DIR / "eda_subject_windows.csv", subj_rows)

    picks = representative_windows(data)
    plot_class_distribution(class_rows, FIGURES_DIR / "eda_class_distribution.png")
    plot_segment_durations(labels, FIGURES_DIR / "eda_segment_durations.png")
    plot_example_windows(picks, "acc", FIGURES_DIR / "eda_example_windows_acc.png")
    plot_example_windows(picks, "gyro", FIGURES_DIR / "eda_example_windows_gyro.png")
    plot_signal_signature(sig_rows, FIGURES_DIR / "eda_signal_signature.png")
    plot_subject_split(subj_rows, FIGURES_DIR / "eda_subject_split.png")

    print(json.dumps({k: v for k, v in overview.items() if not isinstance(v, dict)}, indent=1))
    for r in seg_rows:
        print(f"{r['activity']:<20} segs {r['segments']:>3}  median {r['duration_median_s']:5.1f}s  "
              f"min {r['duration_min_s']:4.1f}s  dropped {r['segments_shorter_than_window']:>2} "
              f"({r['pct_segments_dropped']:4.1f}%)  windows {r['windows']}")


if __name__ == "__main__":
    main()
