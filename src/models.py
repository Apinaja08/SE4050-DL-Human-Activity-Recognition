"""1D CNN model definition for HAR (build_cnn()).

This file is CNN-only. build_lstm(), build_gru(), and build_cnn_lstm() are
owned by other team members and must not be added here.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape=(128, 6), num_classes=12, dropout_rate=0.5, learning_rate=1e-3):
    """Build and compile a lightweight 1D CNN for HAR window classification.

    Input: (128, 6) -- 128 timesteps x 6 sensor channels.
    Conv1D -> BatchNorm -> ReLU -> MaxPool, twice, then a third Conv1D ->
    GlobalAveragePooling1D -> Dense -> Dropout -> Dense(num_classes, softmax).
    """
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv1D(32, kernel_size=5, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(64, kernel_size=5, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
            layers.GlobalAveragePooling1D(),

            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn_har",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    import numpy as np

    model = build_cnn()
    model.summary()

    dummy = np.random.randn(2, 128, 6).astype("float32")
    predictions = model.predict(dummy, verbose=0)
    print(f"\nDummy forward pass output shape: {predictions.shape}")
    print(f"Row sums (should be ~1.0 each, softmax): {predictions.sum(axis=1)}")
