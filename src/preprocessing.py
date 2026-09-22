"""Sensor preprocessing, subject-level train/val/test split, normalization
(fit on train only), and label preparation.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from .data_loader import RAW_DATA_DIR, get_experiment_user_pairs, load_activity_labels
    from .windowing import STEP, WINDOW_SIZE, create_windows_for_experiment
except ImportError:
    from data_loader import RAW_DATA_DIR, get_experiment_user_pairs, load_activity_labels
    from windowing import STEP, WINDOW_SIZE, create_windows_for_experiment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SCALER_PATH = PROJECT_ROOT / "models" / "cnn" / "scaler.pkl"

# Fixed subject-level split (from the project spec) so results are reproducible
# and comparable across the CNN/LSTM/GRU/CNN+LSTM team members.
TRAIN_SUBJECTS = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
TEST_SUBJECTS = [2, 4, 9, 10, 12, 13, 18, 20, 24]

NUM_CLASSES = 12


def split_train_val_subjects(train_subjects=TRAIN_SUBJECTS, val_fraction=0.2, random_state=42):
    """Split the training subjects into fit/validation subject lists.

    The split is done on subject IDs, not on individual windows, so a subject
    used for validation never contributes any windows to the fit set.
    """
    subjects = np.array(sorted(train_subjects))
    rng = np.random.RandomState(random_state)
    rng.shuffle(subjects)

    n_val = max(1, round(len(subjects) * val_fraction))
    val_subjects = sorted(subjects[:n_val].tolist())
    fit_subjects = sorted(subjects[n_val:].tolist())
    return fit_subjects, val_subjects


def encode_activity_labels(activity_ids):
    """Map activity_id (1-12) to zero-based class indices (0-11) for the softmax output."""
    return np.asarray(activity_ids) - 1


def decode_activity_labels(class_indices):
    """Inverse of encode_activity_labels: class indices (0-11) back to activity_id (1-12)."""
    return np.asarray(class_indices) + 1


def fit_scaler(X_train):
    """Fit a StandardScaler on training data only, per sensor channel.

    X_train: (N, window_size, channels). Flattened to (N*window_size,
    channels) so the scaler learns one mean/std per channel, not per
    timestep -- the standard approach for multivariate time series.
    """
    channels = X_train.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, channels))
    return scaler


def apply_scaler(scaler, X):
    """Apply an already-fitted scaler to X, preserving its original shape."""
    shape = X.shape
    return scaler.transform(X.reshape(-1, shape[-1])).reshape(shape)


def normalize_dataset(dataset, save_path=None):
    """Fit a per-channel StandardScaler on X_train only, then apply it to
    train/val/test. Fitting only on training data prevents val/test
    statistics from leaking into the normalization.
    """
    scaler = fit_scaler(dataset["X_train"])

    normalized = dict(dataset)
    for split in ["train", "val", "test"]:
        normalized[f"X_{split}"] = apply_scaler(scaler, dataset[f"X_{split}"])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)

    return normalized, scaler


def build_dataset(
    raw_data_dir=RAW_DATA_DIR,
    window_size=WINDOW_SIZE,
    step=STEP,
    val_fraction=0.2,
    random_state=42,
    save_dir=None,
):
    """Window every experiment and assign it to train/val/test by its subject,
    then concatenate into full (X, y) arrays per split.

    Subject-to-split assignment happens once, up front, so an experiment's
    windows land entirely in one split -- the same subject-level guarantee
    from split_train_val_subjects() carried through to the full dataset.
    """
    fit_subjects, val_subjects = split_train_val_subjects(TRAIN_SUBJECTS, val_fraction, random_state)
    subject_to_split = {}
    subject_to_split.update({s: "train" for s in fit_subjects})
    subject_to_split.update({s: "val" for s in val_subjects})
    subject_to_split.update({s: "test" for s in TEST_SUBJECTS})

    buckets = {"train": ([], [], []), "val": ([], [], []), "test": ([], [], [])}
    for experiment, user in get_experiment_user_pairs(raw_data_dir):
        split = subject_to_split.get(user)
        if split is None:
            continue

        X, y = create_windows_for_experiment(experiment, user, raw_data_dir, window_size, step)
        if len(X) == 0:
            continue

        buckets[split][0].append(X)
        buckets[split][1].append(y)
        buckets[split][2].append(np.full(len(X), user, dtype=int))

    dataset = {}
    for split, (X_list, y_list, subj_list) in buckets.items():
        dataset[f"X_{split}"] = np.concatenate(X_list) if X_list else np.empty((0, window_size, 6))
        dataset[f"y_{split}"] = np.concatenate(y_list) if y_list else np.empty((0,), dtype=int)
        dataset[f"subjects_{split}"] = np.concatenate(subj_list) if subj_list else np.empty((0,), dtype=int)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for key, array in dataset.items():
            np.save(save_dir / f"{key}.npy", array)

    return dataset


def verify_dataset_splits(dataset):
    """Verify subject-level integrity and report class distribution per split.

    Subjects are read back from dataset["subjects_<split>"] (recorded per
    window in build_dataset), not recomputed from split_train_val_subjects(),
    so this is an independent check on the data actually produced.
    """
    subjects_by_split = {
        split: set(np.unique(dataset[f"subjects_{split}"]).tolist()) for split in ["train", "val", "test"]
    }

    print("Subjects per split (read back from built windows):")
    for split, subjects in subjects_by_split.items():
        print(f"  {split}: {sorted(subjects)}")

    no_leakage = True
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = subjects_by_split[a] & subjects_by_split[b]
        if overlap:
            no_leakage = False
        print(f"  {a} vs {b} overlap: {sorted(overlap)} -> {'OK' if not overlap else 'LEAK DETECTED'}")
    print(f"Subject-level leakage check: {'PASSED' if no_leakage else 'FAILED'}")

    activity_names = load_activity_labels()
    print("\nClass distribution per split:")
    for split in ["train", "val", "test"]:
        y = dataset[f"y_{split}"]
        total = len(y)
        unique, counts = np.unique(y, return_counts=True)
        print(f"  {split} (n={total}):")
        for activity_id, count in zip(unique.tolist(), counts.tolist()):
            name = activity_names.get(activity_id, "?")
            print(f"    {activity_id:>2} {name:<18} {count:>5} ({100 * count / total:5.1f}%)")

    return no_leakage


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing smoke test / full dataset build")
    parser.add_argument("--build", action="store_true", help="Build and save the full windowed dataset")
    parser.add_argument("--verify", action="store_true", help="Build the dataset and verify split integrity")
    parser.add_argument("--normalize", action="store_true", help="Build the dataset and fit/apply normalization")
    args = parser.parse_args()

    overlap = set(TRAIN_SUBJECTS) & set(TEST_SUBJECTS)
    print(f"train subjects: {len(TRAIN_SUBJECTS)}, test subjects: {len(TEST_SUBJECTS)}")
    print(f"train/test subject overlap (must be empty set): {overlap}")

    fit_subjects, val_subjects = split_train_val_subjects()
    print(f"fit subjects ({len(fit_subjects)}): {fit_subjects}")
    print(f"val subjects ({len(val_subjects)}): {val_subjects}")
    print(f"fit/val overlap (must be empty set): {set(fit_subjects) & set(val_subjects)}")

    sample_ids = [1, 5, 12]
    encoded = encode_activity_labels(sample_ids)
    decoded = decode_activity_labels(encoded)
    print(f"encode/decode round trip: {sample_ids} -> {encoded.tolist()} -> {decoded.tolist()}")

    if args.build or args.verify or args.normalize:
        dataset = build_dataset(save_dir=PROCESSED_DATA_DIR)
        for split in ["train", "val", "test"]:
            X, y = dataset[f"X_{split}"], dataset[f"y_{split}"]
            print(f"{split}: X {X.shape}, y {y.shape}")
        print(f"Saved .npy files to {PROCESSED_DATA_DIR}")

    if args.verify:
        print()
        verify_dataset_splits(dataset)

    if args.normalize:
        print(f"\nPer-channel mean/std BEFORE normalization (train): "
              f"mean={dataset['X_train'].reshape(-1, 6).mean(axis=0).round(3)}, "
              f"std={dataset['X_train'].reshape(-1, 6).std(axis=0).round(3)}")

        normalized, scaler = normalize_dataset(dataset, save_path=SCALER_PATH)
        print(f"Saved fitted scaler to {SCALER_PATH}")

        print("Per-channel mean/std AFTER normalization:")
        for split in ["train", "val", "test"]:
            X = normalized[f"X_{split}"]
            mean = X.reshape(-1, 6).mean(axis=0).round(3)
            std = X.reshape(-1, 6).std(axis=0).round(3)
            print(f"  {split}: mean={mean}, std={std}")
