"""Activity segmentation and fixed-size (128-sample, 50% overlap) window
generation. Windows never cross activity boundaries.
"""

import numpy as np

try:
    from .data_loader import RAW_DATA_DIR, load_experiment_signals, load_labels
except ImportError:
    from data_loader import RAW_DATA_DIR, load_experiment_signals, load_labels

WINDOW_SIZE = 128
STEP = 64


def create_windows_for_segment(signal, start, end, activity_id, window_size=WINDOW_SIZE, step=STEP):
    """Slide fixed-size windows over one labelled segment only.

    start/end are 1-based, inclusive sample numbers (as given in labels.txt).
    """
    segment = signal[start - 1:end]

    windows = [
        segment[begin:begin + window_size]
        for begin in range(0, len(segment) - window_size + 1, step)
    ]

    if not windows:
        return np.empty((0, window_size, signal.shape[1])), np.empty((0,), dtype=int)

    X = np.stack(windows)
    y = np.full(len(windows), activity_id, dtype=int)
    return X, y


def create_windows_for_experiment(experiment, user, raw_data_dir=RAW_DATA_DIR, window_size=WINDOW_SIZE, step=STEP):
    """Build (X, y) for one experiment/user by windowing each labelled
    activity segment independently, so no window crosses an activity boundary.
    """
    signal = load_experiment_signals(raw_data_dir, experiment, user)
    labels = load_labels(raw_data_dir)
    segments = labels[(labels["experiment"] == experiment) & (labels["user"] == user)]

    X_parts, y_parts = [], []
    for _, row in segments.iterrows():
        X_seg, y_seg = create_windows_for_segment(
            signal,
            int(row["start"]),
            int(row["end"]),
            int(row["activity_id"]),
            window_size=window_size,
            step=step,
        )
        if len(X_seg):
            X_parts.append(X_seg)
            y_parts.append(y_seg)

    if not X_parts:
        return np.empty((0, window_size, signal.shape[1])), np.empty((0,), dtype=int)

    return np.concatenate(X_parts), np.concatenate(y_parts)


if __name__ == "__main__":
    X, y = create_windows_for_experiment(experiment=1, user=1)
    print(f"Experiment 1 / User 1 -> X shape: {X.shape}, y shape: {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"activity_id counts in this experiment/user: {dict(zip(unique.tolist(), counts.tolist()))}")
