# SE4050 -- Deep Learning

# Human Activity Recognition using Deep Learning Models

This project implements deep learning models for **Human Activity
Recognition (HAR)** using the **UCI Smartphone-Based Recognition of
Human Activities and Postural Transitions (HAPT)** dataset.

The project uses raw smartphone **accelerometer and gyroscope
time-series data** to classify 12 human activities and postural
transitions. The models are evaluated using a common preprocessing and
subject-independent evaluation protocol.

## 1. Dataset

The HAPT data contains smartphone inertial sensor recordings collected
at **50 Hz**.

### Sensor Channels

Six sensor channels are used:

-   Accelerometer X
-   Accelerometer Y
-   Accelerometer Z
-   Gyroscope X
-   Gyroscope Y
-   Gyroscope Z

### Activity Classes

    ID Activity             Group
  ---- -------------------- -----------------
     1 WALKING              Basic / Dynamic
     2 WALKING_UPSTAIRS     Basic / Dynamic
     3 WALKING_DOWNSTAIRS   Basic / Dynamic
     4 SITTING              Basic / Static
     5 STANDING             Basic / Static
     6 LAYING               Basic / Static
     7 STAND_TO_SIT         Transition
     8 SIT_TO_STAND         Transition
     9 SIT_TO_LIE           Transition
    10 LIE_TO_SIT           Transition
    11 STAND_TO_LIE         Transition
    12 LIE_TO_STAND         Transition

The dataset is highly imbalanced. Basic activities account for
approximately **95.3%** of the generated windows, while transition
activities account for approximately **4.7%**.

## 2. Data Preprocessing

### 2.1 Sensor Fusion

Accelerometer and gyroscope signals are stacked into a single
six-channel time series.

### 2.2 Segmentation and Windowing

Activity boundaries from `labels.txt` are used to segment the signals.
Windows are created only inside individual labelled activity segments.

-   Window size: **128 samples**
-   Window duration: **2.56 seconds**
-   Step size: **64 samples**
-   Overlap: **50%**
-   Channels: **6**

### 2.3 Subject-Level Split

The dataset is divided by subject so that the same subject does not
appear in more than one partition.

-   **Training subjects:** 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22,
    23, 26, 28, 29, 30
-   **Validation subjects:** 1, 3, 25, 27
-   **Test subjects:** 2, 4, 9, 10, 12, 13, 18, 20, 24

The subject-overlap check passed.

### 2.4 Normalization

Per-channel standardization is applied.

The scaler is fitted using **training windows only** and then applied
unchanged to validation and test data. This prevents data leakage.

### 2.5 Labels and Class Weights

Activity IDs 1--12 are mapped to class indices 0--11.

Balanced class weights are calculated from **training labels only** and
used during model training to address class imbalance.

## 3. Models

The project evaluates deep learning approaches for the same 12-class HAR
task.

### 3.1 CNN

A compact **1D CNN** learns local temporal patterns directly from the
normalized six-channel sensor windows.

The CNN experiment uses convolutional layers, Batch Normalization,
Dropout, and Global Average Pooling before the final classifier.

### 3.2 GRU

The **GRU** model learns temporal dependencies from the sensor sequence
using gated recurrent units.

Architecture:

``` text
Input (128, 6)
    ↓
GRU (64, return_sequences=True)
    ↓
Dropout (0.30)
    ↓
GRU (32, return_sequences=False)
    ↓
Dropout (0.30)
    ↓
Dense (64, ReLU)
    ↓
Dropout (0.50)
    ↓
Dense (12, Softmax)
```

### 3.3 LSTM

The **LSTM** model is used to learn longer-term temporal dependencies in
the six-channel sensor sequence through recurrent memory and gating
mechanisms.

The exact LSTM configuration and results should be taken from the
executed LSTM notebook.

### 3.4 MLP

The **MLP** model provides a fully connected neural-network approach for
the same classification task.

The exact MLP configuration and results should be taken from the
executed MLP notebook.

## 4. Training Configuration

The common training protocol uses:

  Setting           Value
  ----------------- ----------------------------------
  Optimizer         Adam
  Learning Rate     0.001
  Loss              Sparse Categorical Cross-Entropy
  Batch Size        64
  Maximum Epochs    50
  Early Stopping    Validation loss
  Patience          10
  Checkpoint        Best validation-loss epoch
  Class Weighting   Balanced training-data weights

The model checkpoint is selected using validation performance. The test
set is not used for model selection.

## 5. Experimental Environment

The reported CNN experiment was executed using:

  Component          Configuration
  ------------------ -------------------
  Operating System   Windows 11
  CPU                Intel Core 7 150U
  Logical CPUs       12
  GPU                None
  Python             3.12.10
  TensorFlow         2.21.0
  Keras              3.15.1

## 6. Keeping the Test Set Unseen

The evaluation follows a subject-independent protocol.

-   Test subjects are separated before model development.
-   No test subject contributes training or validation windows.
-   Normalization statistics are calculated from training windows only.
-   Class weights are calculated from training labels only.
-   Early stopping and checkpoint selection use validation loss.
-   The final test set is used only for evaluation.

## 7. Evaluation Metrics

The models are evaluated using metrics suitable for the imbalanced
12-class problem.

  Metric              Purpose
  ------------------- ------------------------------------------
  Accuracy            Overall classification performance
  Macro Precision     Gives equal importance to every class
  Macro Recall        Shows recall across all classes
  Macro F1            Highlights minority-class performance
  Weighted F1         Reflects the observed class distribution
  Balanced Accuracy   Mean recall across classes
  ROC-AUC             Measures class discrimination
  PR-AUC              Useful for imbalanced classes
  Confusion Matrix    Shows class-specific errors

Macro-level metrics are important because basic activities dominate the
dataset.

## 8. Model Evaluation and Analysis

The analysis includes:

-   Overall classification performance
-   Basic activity performance
-   Postural-transition performance
-   Confusion matrix analysis
-   Largest confusion pairs
-   Training and validation curves
-   Class-weight ablation
-   Training stability
-   Computational efficiency

### Basic Activities and Transitions

Basic activities have substantially more samples than postural
transitions. Therefore, transition-class performance is considered
separately rather than relying only on overall accuracy.

### Largest Confusions

The CNN report identifies **SITTING vs STANDING** as a major confusion
pair and also observes confusion between **STAND_TO_LIE and
SIT_TO_LIE**.

## 9. Class-Weight Ablation

The class-weight experiment compares:

-   Model with class weights
-   Model without class weights

The comparison evaluates the effect of class weighting on:

-   Validation Macro-F1
-   Validation Accuracy
-   Test Accuracy
-   Test Macro-F1
-   Test Balanced Accuracy
-   Transition Accuracy
-   Transition Macro-F1

For the reported CNN experiment, class weighting produced higher mean
transition accuracy and transition Macro-F1, while the unweighted setup
had slightly higher mean test accuracy.

## 10. Training Stability

Training stability is evaluated across multiple random seeds.

The analysis reports mean, standard deviation, and range for important
metrics such as:

-   Test accuracy
-   Test Macro-F1
-   Test weighted-F1
-   Test ROC-AUC
-   Test PR-AUC
-   Validation accuracy
-   Validation Macro-F1
-   Training time

## 11. Computational Efficiency

Computational evaluation includes:

-   Number of parameters
-   FLOPs per window
-   Model weight size
-   Saved model size
-   Training time
-   Time per epoch
-   Compiled inference latency
-   Eager inference latency
-   Batched inference throughput
-   New-window interval

For the reported CNN experiment:

  Measure                                         Result
  --------------------- --------------------------------
  Parameters                                      45,420
  FLOPs/window                                   ≈3.19 M
  Weights                                         177 KB
  Saved model                                     591 KB
  Training time                             29.9 ± 6.4 s
  Time/epoch                               1.28 ± 0.11 s
  Compiled latency        0.707 ms median / 0.904 ms p95
  Eager latency                          6.757 ms median
  Batched inference                     0.0402 ms/window
  Approx. throughput                    24,869 windows/s
  New-window interval                           1,280 ms

## 12. Reported CNN Result

The final reported CNN achieved:

-   **Test Accuracy:** 87.67%
-   **Test Macro-F1:** 76.91%

The CNN was evaluated on **3,162 unseen-subject test windows**.

Basic activities were recognized more consistently than transition
activities. The main observed confusions were SITTING vs STANDING and
STAND_TO_LIE vs SIT_TO_LIE.

## 13. Limitations

The reported experiment has the following limitations:

-   Transition classes have very small test support.
-   Some transition segments are shorter than the 2.56-second window.
-   The fixed waist-mounted smartphone position limits generalization to
    other placements.
-   The data was collected using a scripted protocol.
-   Subject variability affects performance.
-   Validation contains only four subjects, so its estimate can be
    noisy.
-   Unlabelled portions of recordings are not used.

## 14. Conclusion

This project applies deep learning to raw smartphone accelerometer and
gyroscope time-series data for 12-class Human Activity Recognition.

The common pipeline uses sensor fusion, labelled-segment windowing,
subject-level separation, training-only normalization, balanced class
weighting, early stopping, and imbalance-aware evaluation.

CNN, GRU, LSTM, and MLP approaches are used within the project to study
different neural-network approaches to the same HAR classification
problem. Model-specific numerical results are obtained from their
respective executed experiments.

## References

1.  Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L.
    (2012). Human activity recognition on smartphones using a multiclass
    hardware-friendly support vector machine. IWAAL.
2.  Cho, K., et al. (2014). Learning phrase representations using RNN
    encoder-decoder for statistical machine translation. EMNLP.
3.  Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical
    evaluation of gated recurrent neural networks on sequence modeling.
4.  Hammerla, N. Y., Halloran, S., & Plötz, T. (2016). Deep,
    convolutional, and recurrent models for human activity recognition
    using wearables. IJCAI.
5.  Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
    Neural Computation.
6.  Murad, A., & Pyun, J.-Y. (2017). Deep recurrent neural networks for
    human activity recognition. Sensors.
7.  Ordóñez, F. J., & Roggen, D. (2016). Deep convolutional and LSTM
    recurrent neural networks for multimodal wearable activity
    recognition. Sensors.
8.  Reyes-Ortiz, J. L., Oneto, L., Samà, A., Parra, X., & Anguita, D.
    (2016). Transition-aware human activity recognition using
    smartphones. Neurocomputing.
