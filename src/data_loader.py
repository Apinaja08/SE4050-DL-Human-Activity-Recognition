"""Loading raw accelerometer/gyroscope data, labels.txt, and activity_labels.txt
for the UCI Smartphone HAR dataset (data/raw/RawData/).
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "RawData"

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/341/"
    "smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"
)


def download_dataset(raw_data_dir=RAW_DATA_DIR, force=False):
    """Download and extract the UCI Smartphone HAPT dataset into data/raw/RawData/.

    Skips the download if labels.txt is already present, unless force=True.
    """
    raw_data_dir = Path(raw_data_dir)
    if not force and (raw_data_dir / "labels.txt").exists():
        print(f"Dataset already present at {raw_data_dir}, skipping download.")
        return raw_data_dir

    data_root = raw_data_dir.parent
    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / "hapt_dataset.zip"

    print(f"Downloading dataset ({DATASET_URL}) ...")
    urllib.request.urlretrieve(DATASET_URL, zip_path)

    print("Extracting...")
    extract_dir = data_root / "_extracted"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    found = next(extract_dir.rglob("RawData"), None)
    if found is None:
        raise FileNotFoundError("Could not find a RawData/ folder inside the downloaded archive.")

    if raw_data_dir.exists():
        shutil.rmtree(raw_data_dir)
    shutil.move(str(found), str(raw_data_dir))

    # activity_labels.txt ships at the archive root, not inside RawData/,
    # but the rest of this module expects it alongside labels.txt.
    activity_labels_src = extract_dir / "activity_labels.txt"
    shutil.copy(activity_labels_src, raw_data_dir / "activity_labels.txt")

    shutil.rmtree(extract_dir)
    zip_path.unlink()

    print(f"Dataset ready at {raw_data_dir}")
    return raw_data_dir


def load_activity_labels(raw_data_dir=RAW_DATA_DIR):
    """Return {activity_id: activity_name} from activity_labels.txt."""
    path = Path(raw_data_dir) / "activity_labels.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["activity_id", "activity_name"])
    return dict(zip(df["activity_id"], df["activity_name"]))


def load_labels(raw_data_dir=RAW_DATA_DIR):
    """Return labels.txt as a DataFrame with columns:
    experiment, user, activity_id, start, end.
    """
    path = Path(raw_data_dir) / "labels.txt"
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["experiment", "user", "activity_id", "start", "end"],
    )


def load_signal(raw_data_dir, signal_type, experiment, user):
    """Load one acc_ or gyro_ file as an (N, 3) array.

    signal_type: "acc" or "gyro"
    """
    filename = f"{signal_type}_exp{experiment:02d}_user{user:02d}.txt"
    path = Path(raw_data_dir) / filename
    return pd.read_csv(path, sep=r"\s+", header=None).values


def load_experiment_signals(raw_data_dir, experiment, user):
    """Load and combine acc + gyro for one experiment/user into an (N, 6) array:
    columns [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].

    acc and gyro files for the same experiment/user can differ by a sample or
    two, so both are truncated to the shorter length before combining.
    """
    acc = load_signal(raw_data_dir, "acc", experiment, user)
    gyro = load_signal(raw_data_dir, "gyro", experiment, user)
    n = min(len(acc), len(gyro))
    return np.hstack([acc[:n], gyro[:n]])


def get_experiment_user_pairs(raw_data_dir=RAW_DATA_DIR):
    """Return a sorted list of unique (experiment, user) pairs from labels.txt."""
    labels = load_labels(raw_data_dir)
    pairs = labels[["experiment", "user"]].drop_duplicates()
    return sorted(pairs.itertuples(index=False, name=None))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data loader smoke test / dataset download")
    parser.add_argument("--download", action="store_true", help="Download and extract the dataset first")
    args = parser.parse_args()

    if args.download:
        download_dataset()

    activity_labels = load_activity_labels()
    print(f"activity_labels.txt: {len(activity_labels)} classes -> {activity_labels}")

    labels = load_labels()
    print(f"labels.txt: {len(labels)} rows")

    pairs = get_experiment_user_pairs()
    print(f"unique (experiment, user) pairs: {len(pairs)}")

    X = load_experiment_signals(RAW_DATA_DIR, experiment=1, user=1)
    print(f"Experiment 1 / User 1 combined signal shape: {X.shape}")
