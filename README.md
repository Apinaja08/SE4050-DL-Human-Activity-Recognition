# SE4050 – Deep Learning: Human Activity and Postural Transition Recognition (HAPT)

A deep learning project implementing a **Stacked Bidirectional Long Short-Term Memory (Bi-LSTM)** network to classify 12 human activities and postural transitions using raw tri-axial accelerometer and gyroscope smartphone sensor signals.

---

## Project Structure

The project is structured in two parallel, fully synchronized formats:
1. **Standalone End-to-End Notebook (`notebooks/LSTM_Human_Activity_Recognition.ipynb`)**: Runs completely self-contained in VS Code, local Jupyter Notebook, or Google Colab with zero external script dependencies.
2. **Modular Python Pipeline (`src/`)**: Clean, production-grade modular codebase separated into specialized components.

```text
Saru's DL Project/
├── data/
│   ├── raw/                       # Downloaded UCI HAPT dataset
│   └── processed/                 # Segmented windows (X_train, y_train, etc.)
├── models/
│   └── best_lstm_model.pth        # Saved best PyTorch model checkpoint
├── notebooks/
│   └── LSTM_Human_Activity_Recognition.ipynb  # Standalone end-to-end Master Notebook
├── outputs/
│   ├── figures/                   # Learning curves, confusion matrix, EDA plots
│   └── metrics/                   # Test evaluation results (JSON)
├── src/
│   ├── __init__.py                # Package initialization
│   ├── download_data.py           # Automated dataset download & extraction
│   ├── dataset.py                 # Sliding windowing & subject-based partitioning
│   ├── models.py                  # ActivityLSTM architecture definition
│   ├── utils.py                   # Device setup, seeds, and plotting utilities
│   ├── train.py                   # Model training loop with early stopping & LR scheduler
│   └── evaluate.py                # Comprehensive test evaluation & metrics computation
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## Two Ways to Run the Project

### Option A: Standalone Notebook (VS Code, Local Jupyter, or Google Colab)

The notebook at [`notebooks/LSTM_Human_Activity_Recognition.ipynb`](notebooks/LSTM_Human_Activity_Recognition.ipynb) contains the entire end-to-end pipeline in a single file:
- Automatically downloads and extracts the raw dataset from UCI Repository.
- Segments signals into 128-timestep sliding windows (50% overlap).
- Partitions subjects into Train (1–21), Val (22–25), and Unseen Test (26–30).
- Trains the `ActivityLSTM` with class-weighted cross-entropy loss.
- Computes comprehensive evaluation metrics and renders visualizations.

**To run in VS Code / Local Jupyter:**
1. Open [`notebooks/LSTM_Human_Activity_Recognition.ipynb`](notebooks/LSTM_Human_Activity_Recognition.ipynb).
2. Select your Python kernel (with dependencies from `requirements.txt` installed).
3. Click **Run All**.

**To run in Google Colab:**
1. Upload `LSTM_Human_Activity_Recognition.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime → Change runtime type → T4 GPU** (or CPU).
3. Click **Runtime → Run all**.

---

### Option B: Modular Python Pipeline (CLI Execution)

Run each step modularly using standard Python commands from the project root:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and extract UCI HAPT dataset
python src/download_data.py

# 3. Generate sliding windows and subject-wise splits
python src/dataset.py

# 4. Train the ActivityLSTM model
python src/train.py --epochs 25 --batch-size 64 --lr 0.001

# 5. Evaluate the trained model on unseen test subjects (Users 26-30)
python src/evaluate.py
```

---

## Experimental Protocol & Architecture

| Component | Specification |
|---|---|
| **Dataset** | UCI HAPT (Smartphone-Based Recognition of Human Activities and Postural Transitions, ID: 341) |
| **Input Signals** | 6 raw sensor channels: Total Acc ($X,Y,Z$) + Gyroscope ($X,Y,Z$) sampled at 50 Hz |
| **Windowing** | $T=128$ timesteps (2.56 s), 50% overlap (stride = 64 timesteps) |
| **Target Classes** | 12 classes: 6 basic activities + 6 postural transitions |
| **Split Strategy** | Subject-wise split (Train: Subjects 1–21, Val: Subjects 22–25, Test: Subjects 26–30) to prevent data leakage |
| **Model Architecture** | Stacked Bi-LSTM (128 units/dir) → Dropout (0.3) → Bi-LSTM (64 units/dir) → BatchNorm1d (128) → Linear (64) + ReLU → Dropout (0.4) → Linear (12) |
| **Total Parameters** | 313,420 trainable parameters (1.20 MB checkpoint) |
| **Loss Function** | Inverse-frequency weighted Cross-Entropy Loss (combats class imbalance) |
| **Optimizer & Schedule**| Adam ($lr=10^{-3}$, weight decay $= 10^{-4}$) with `ReduceLROnPlateau` |

---

## Results on Unseen Test Subjects (Users 26–30)

| Metric | Score |
|---|---|
| **Test Accuracy** | **94.39%** |
| **Macro F1-Score** | **89.68%** |
| **Weighted F1-Score** | **94.39%** |
| **Macro Precision** | **89.72%** |
| **Macro Recall** | **90.72%** |
| **Matthews Correlation Coefficient (MCC)** | **93.34%** |
| **Macro ROC-AUC (OvR)** | **99.09%** |
| **Inference Latency** | **0.209 ms / sample** |

---

## References

1. Reyes-Ortiz, J., Anguita, D., Oneto, L., & Parra, X. (2015). *Smartphone-Based Recognition of Human Activities and Postural Transitions* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C54G7M](https://doi.org/10.24432/C54G7M).
2. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.

