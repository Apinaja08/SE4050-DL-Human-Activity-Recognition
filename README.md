# Human Activity Recognition Using Deep Learning

## Project Overview

This project develops a deep learning-based Human Activity Recognition
(HAR) system using smartphone sensor data.

The system uses raw time-series data collected from smartphone:

-   Accelerometer: X, Y, Z
-   Gyroscope: X, Y, Z

The six sensor channels are used to recognize 12 different human
activities and postural transitions.

The project contains four deep learning approaches:

1.  **1D CNN**
2.  **LSTM**
3.  **GRU**
4.  **CNN + LSTM**

Each model uses the same underlying preprocessing and subject-level data
split so that the final comparison is consistent and fair.

------------------------------------------------------------------------

## Dataset

**Dataset:** UCI Smartphone-Based Recognition of Human Activities and
Postural Transitions

**Source:** UCI Machine Learning Repository

Dataset URL:

https://archive.ics.uci.edu/dataset/341/smartphone%2Bbased%2Brecognition%2Bof%2Bhuman%2Bactivities%2Band%2Bpostural%2Btransitions

### Sensor Data

The raw sensor data contains six channels:

  Sensor          Channels
  --------------- ----------
  Accelerometer   X, Y, Z
  Gyroscope       X, Y, Z

The sampling frequency is **50 Hz**.

### Activity Classes

The project recognizes 12 classes:

    ID Activity
  ---- --------------------
     1 WALKING
     2 WALKING_UPSTAIRS
     3 WALKING_DOWNSTAIRS
     4 SITTING
     5 STANDING
     6 LAYING
     7 STAND_TO_SIT
     8 SIT_TO_STAND
     9 SIT_TO_LIE
    10 LIE_TO_SIT
    11 STAND_TO_LIE
    12 LIE_TO_STAND

------------------------------------------------------------------------

## Project Structure

``` text
Human-Activity-Recognition/
│
├── data/
│   ├── raw/
│   │   └── UCI dataset files
│   │
│   └── processed/
│       └── generated datasets
│
├── notebooks/
│   └── existing notebooks
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── windowing.py
│   ├── models.py
│   ├── training.py
│   └── evaluation.py
│
├── models/
│   ├── cnn/
│   ├── lstm/
│   ├── gru/
│   └── cnn_lstm/
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── confusion_matrices/
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Directory Description

-   **`data/raw/`** --- Original UCI dataset files. These files should
    not be committed to Git.
-   **`data/processed/`** --- Generated/preprocessed datasets.
-   **`notebooks/`** --- Existing notebooks used for exploration and
    experiments.
-   **`src/`** --- Reusable Python source code for loading,
    preprocessing, windowing, modeling, training, and evaluation.
-   **`models/`** --- Saved trained model files, separated by
    architecture.
-   **`results/figures/`** --- Generated plots and visualizations.
-   **`results/metrics/`** --- Model performance metrics.
-   **`results/confusion_matrices/`** --- Confusion matrix outputs.

------------------------------------------------------------------------

## Data Processing Pipeline

The raw sensor data is processed using the following pipeline:

``` text
Raw Accelerometer
        +
Raw Gyroscope
        |
        v
Combine 6 Sensor Channels
        |
        v
Activity Segmentation
        |
        v
128-Sample Windows
        |
        v
50% Overlap
        |
        v
128 x 6 Time-Series Input
        |
        v
Deep Learning Models
```

### Window Configuration

The input window contains:

-   **128 time steps**
-   **6 sensor channels**
-   **50% overlap**
-   **64-sample step**

Since the sensors operate at 50 Hz:

``` text
128 / 50 = 2.56 seconds
```

Therefore, each input window represents approximately **2.56 seconds of
sensor activity**.

Windows are created only inside individual labelled activity segments. A
window must not cross from one activity into another.

------------------------------------------------------------------------

## Subject-Level Data Split

To avoid subject-level data leakage, training and testing are separated
by subject.

### Training Subjects

``` text
1, 3, 5, 6, 7, 8, 11, 14, 15, 16,
17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30
```

### Test Subjects

``` text
2, 4, 9, 10, 12, 13, 18, 20, 24
```

There are:

-   **21 training subjects**
-   **9 test subjects**

Validation data is created from the training subjects only.

The same subject must not appear in multiple splits.

------------------------------------------------------------------------

## Normalization

Sensor normalization is performed without data leakage.

The normalization parameters are fitted **only on the training data**.

The same fitted transformation is then applied to:

-   Training data
-   Validation data
-   Test data

Test data is never used to calculate normalization statistics.

------------------------------------------------------------------------

## Deep Learning Models

### 1. 1D CNN

The 1D CNN processes the multivariate sensor sequence:

``` text
Input: 128 x 6
        |
     Conv1D
        |
 Batch Normalization
        |
   Max Pooling
        |
     Conv1D
        |
 Batch Normalization
        |
   Max Pooling
        |
     Conv1D
        |
Global Average Pooling
        |
      Dense
        |
     Dropout
        |
   12-Class Softmax
```

The CNN is designed to learn local temporal patterns from accelerometer
and gyroscope signals.

### 2. LSTM

The LSTM model processes the same `128 x 6` input sequence and learns
temporal dependencies.

### 3. GRU

The GRU model processes the same `128 x 6` input sequence using gated
recurrent units.

### 4. CNN + LSTM

The CNN + LSTM architecture combines convolutional feature extraction
with recurrent temporal modeling.

------------------------------------------------------------------------

## Evaluation

The models are evaluated using the same test subjects.

The following metrics are used:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Macro F1
-   Weighted F1
-   Per-class classification report
-   Confusion matrix

Accuracy alone is not sufficient because the dataset contains class
imbalance, particularly among the postural transition classes.

------------------------------------------------------------------------

## Class Imbalance

The dataset contains significantly different numbers of samples across
activities.

The postural transition classes generally contain fewer samples than
several basic activities.

Therefore, model performance is analyzed using macro-level and per-class
metrics in addition to overall accuracy.

Class weighting may be used when appropriate.

------------------------------------------------------------------------

## Computational Environment

The project is designed to run on CPU.

A GPU is optional and not required.

The models are intentionally kept reasonably sized so that development
and experimentation can be performed without requiring high-end
hardware.

------------------------------------------------------------------------

## Source Code Organization

### `src/data_loader.py`

Responsible for loading:

-   Accelerometer data
-   Gyroscope data
-   Activity labels
-   Activity names
-   Subject information

### `src/preprocessing.py`

Responsible for:

-   Sensor preprocessing
-   Subject-level splitting
-   Normalization
-   Label preparation

### `src/windowing.py`

Responsible for:

-   Activity segmentation
-   Fixed-size window generation
-   50% overlapping windows
-   Preventing windows from crossing activity boundaries

### `src/models.py`

Contains model-building functions for the project's deep learning
architectures.

### `src/training.py`

Contains reusable training utilities, callbacks, training history
handling, and timing.

### `src/evaluation.py`

Contains evaluation metrics, classification reports, confusion matrices,
and visualization utilities.

------------------------------------------------------------------------

## Git and Data Management

The original dataset should not be committed to the repository.

The `.gitignore` should exclude:

``` text
data/raw/
data/processed/
__pycache__/
.ipynb_checkpoints/
*.pyc
```

Large generated model files and datasets should also be excluded when
they are not required in the repository.

------------------------------------------------------------------------

## Reproducibility

The project should use fixed random seeds where practical so that
preprocessing, training, and evaluation are as reproducible as possible.

All models should use the same:

-   Dataset
-   Subject-level split
-   Window configuration
-   Normalization strategy
-   Test set
-   Evaluation metrics

This allows meaningful comparison between the different architectures.

------------------------------------------------------------------------

## Current CNN Development

The raw-data pipeline has been verified using Experiment 1 / User 1.

Verified shapes:

``` text
Accelerometer: (20598, 3)
Gyroscope:     (20598, 3)
Combined:      (20598, 6)
```

Using:

``` text
Window size = 128
Step = 64
```

The verified CNN prototype produces:

``` text
X shape = (185, 128, 6)
y shape = (185,)
```

This confirms that the basic raw sensor windowing pipeline is
functioning correctly.

------------------------------------------------------------------------

## Main Project Goal

The overall system aims to determine human activities from smartphone
sensor time-series data and compare different deep learning
architectures using a consistent preprocessing and evaluation pipeline.

The CNN component focuses on learning temporal patterns directly from
the six raw sensor channels.
