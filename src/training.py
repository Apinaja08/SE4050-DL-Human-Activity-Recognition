"""CNN training utilities: callbacks, training-history handling, and timing."""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "cnn" / "cnn_model.keras"
HISTORY_PATH = PROJECT_ROOT / "results" / "metrics" / "cnn_training_history.json"


def save_history(history, save_path=HISTORY_PATH):
    """Save a Keras History.history dict as JSON so it can be plotted later
    without retraining.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(history, f)
    return save_path


def compute_class_weights(y_train):
    """Balanced class weights from y_train only.

    Transition classes make up well under 2% of windows each (verified in
    STEP 5). Without weighting, cross-entropy loss is dominated by the six
    basic activities and the model has little incentive to learn the rarer
    transitions at all -- it could reach high accuracy while nearly ignoring
    them, which is exactly the failure mode per-class metrics are meant to
    catch.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return dict(zip(classes.tolist(), weights.tolist()))


def get_callbacks(model_path=MODEL_PATH, patience=10):
    """Early stopping on val_loss, plus checkpointing the best model to disk."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True),
    ]


def train_cnn(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=50,
    batch_size=64,
    use_class_weight=True,
    model_path=MODEL_PATH,
    patience=10,
):
    """Train model with early stopping + checkpointing. Returns (history, elapsed_seconds)."""
    class_weight = compute_class_weights(y_train) if use_class_weight else None
    callbacks = get_callbacks(model_path=model_path, patience=patience)

    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - start

    return history, elapsed


if __name__ == "__main__":
    import argparse

    try:
        from .models import build_cnn
        from .preprocessing import PROCESSED_DATA_DIR, SCALER_PATH, build_dataset, encode_activity_labels, normalize_dataset
    except ImportError:
        from models import build_cnn
        from preprocessing import PROCESSED_DATA_DIR, SCALER_PATH, build_dataset, encode_activity_labels, normalize_dataset

    parser = argparse.ArgumentParser(description="Train the CNN")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5, a quick smoke test)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print("Building dataset...")
    dataset = build_dataset(save_dir=PROCESSED_DATA_DIR)
    normalized, scaler = normalize_dataset(dataset, save_path=SCALER_PATH)

    y_train = encode_activity_labels(normalized["y_train"])
    y_val = encode_activity_labels(normalized["y_val"])

    print(f"\nTraining for up to {args.epochs} epochs (early stopping may end it sooner)...")
    model = build_cnn()
    history, elapsed = train_cnn(
        model,
        normalized["X_train"],
        y_train,
        normalized["X_val"],
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    n_epochs_run = len(history.history["loss"])
    print(f"\nTraining took {elapsed:.1f}s for {n_epochs_run} epoch(s)")
    print(f"Final train accuracy: {history.history['accuracy'][-1]:.3f}")
    print(f"Final val accuracy:   {history.history['val_accuracy'][-1]:.3f}")
    print(f"Best model saved to:  {MODEL_PATH}")

    history_path = save_history(history.history)
    print(f"Training history saved to: {history_path}")
