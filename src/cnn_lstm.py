"""CNN-LSTM architecture used by the experiment notebook."""
from tensorflow import keras
from tensorflow.keras import layers

def build_cnn_lstm(input_shape=(128, 6), n_classes=12):
    inp = keras.Input(shape=input_shape, name="sensor_window")

    x = layers.Conv1D(64, 5, padding="same")(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv1D(64, 5, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax", name="activity")(x)
    return keras.Model(inp, out, name="CNN_LSTM")

