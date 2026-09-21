# SE4050 - Deep Learning Lab Assignment
## Human Activity Recognition and Postural Transitions (HAPT)
### Architecture: Gated Recurrent Unit (GRU)

**Student Name:** Nikshan Pathmaseelan  
**Student ID:** IT23264434  

This repository contains the complete Google Colab notebook for the **GRU** model component of the SE4050 Deep Learning Human Activity Recognition project.

---

### Team Project Scope
Our team is benchmarking four deep learning architectures on the exact same dataset split:
1. **MLP (Multilayer Perceptron)**
2. **CNN (Convolutional Neural Network)**
3. **LSTM (Long Short-Term Memory)**
4. **GRU (Gated Recurrent Unit)** — *Implemented in this repository*

---

### Dataset Details
- **Dataset:** UCI Smartphone-Based Recognition of Human Activities and Postural Transitions (HAPT)
- **Google Drive Link:** [Dataset Folder](https://drive.google.com/drive/folders/1R6LEVxyyITQerRzSpV_Xc_0WyRYTqvRb?usp=sharing) (Folder ID: `1R6LEVxyyITQerRzSpV_Xc_0WyRYTqvRb`)
- **Key Files:**
  - `Train/X_train.txt`: 561 engineered features for training
  - `Train/y_train.txt`: Activity labels for training (1–12)
  - `Test/X_test.txt`: 561 engineered features for testing (kept unseen until final evaluation)
  - `Test/y_test.txt`: Official test labels (1–12)
  - `activity_labels.txt`: Mapping of 12 activity classes (6 basic ADLs + 6 postural transitions)

---

### Files in this Repository
- `SE4050_GRU_Human_Activity_Recognition.ipynb`: Complete, self-contained Google Colab notebook with all 16 required academic sections.
- `README.md`: Project overview and execution instructions.
- `.gitignore`: Rules ignoring downloaded datasets, weights, and plots.

---

### How to Run in Google Colab
1. Upload `SE4050_GRU_Human_Activity_Recognition.ipynb` to [Google Colab](https://colab.research.google.com).
2. (Optional) Set the hardware accelerator to **T4 GPU** (`Runtime` -> `Change runtime type` -> `T4 GPU`).
3. Click `Runtime` -> `Run all`.
4. The notebook automatically connects to the team Google Drive folder or downloads the files seamlessly, creates a stratified 80/20 train/validation split from the official training data, normalizes features without data leakage, trains the GRU with callbacks, and performs final evaluation on the official unseen test set.
5. All summary tables, figures (`training_curves.png`, `confusion_matrix.png`), and model files (`gru_har_model.keras`) are saved automatically.
