# 🧠 EEG Visual Learner Classification (LSTM-CNN)

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=tuhinbiswas404076-ai/EEG-LSTM-CNN-Learner-Classification&branch=main&mainModule=app.py)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg?logo=tensorflow)](https://tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.0%2B-red.svg?logo=keras)](https://keras.io/)
[![MNE-Python](https://img.shields.io/badge/MNE-1.6%2B-green.svg)](https://mne.tools/)
[![Dataset: PhysioNet EEGMMIDB](https://img.shields.io/badge/Dataset-PhysioNet_EEGMMIDB-yellow.svg)](https://physionet.org/content/eegmmidb/1.0.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **Research Reference:**  
> Jawed, Faye & Malik — *Deep Learning-Based Assessment Model for Visual Learner Identification using Raw EEG*, **IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)**, Vol. 32, 2024.

---

## 🚀 1-Click Live Cloud Deployment

Click the badge below to immediately deploy and launch this application on **Streamlit Community Cloud** with all configurations pre-filled:

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=tuhinbiswas404076-ai/EEG-LSTM-CNN-Learner-Classification&branch=main&mainModule=app.py)

👉 **[Direct 1-Click Deploy Link](https://share.streamlit.io/deploy?repository=tuhinbiswas404076-ai/EEG-LSTM-CNN-Learner-Classification&branch=main&mainModule=app.py)**

---

## 📌 Project Overview

This repository provides an end-to-end, production-ready implementation and interactive web deployment of an **LSTM-CNN deep learning architecture** designed to classify resting-state raw Electroencephalography (EEG) signals into **Visual Learners** vs. **Non-Visual Learners** based on neurophysiological oscillatory dynamics.

- **Baseline Neural Core**: Preserves the exact LSTM-CNN architecture and signal preprocessing pipeline from the peer-reviewed research study.
- **Zero-Retraining Cloud Serving**: The pre-trained model weights (`model/lstm_cnn_model.keras`) are packaged directly for instant cloud inference without requiring dataset downloads or retraining.
- **Interactive Streamlit Web Dashboard**: Allows researchers and educators to test instant 1-click benchmark recordings, upload custom multi-channel EEG files (`.edf`, `.npy`, `.npz`, `.csv`), inspect real-time waveforms, and visualize frequency band Power Spectral Density (PSD).

---

## 🌊 EEG Signal & Spectral Characteristics

### 1. Multichannel Raw EEG Waveforms
EEG recordings are captured across 64 scalp electrodes (10-10 international montage) at 160 Hz. Below are representative 4-second traces across Frontal (Fp1, Fz), Central (Cz), Parietal (Pz), and Occipital (O1, Oz, O2) regions:

![Multichannel EEG Traces](assets/eeg_multichannel_waveforms.png)

### 2. Power Spectral Density (PSD) & Alpha Oscillations
Visual learner identification relies on distinct occipital-parietal rhythms. Synchronization in the **Alpha band (8–13 Hz)** during eyes-closed resting state indicates visual cortex idling (Visual Learner), whereas alpha desynchronization occurs during eyes-open cognitive processing (Non-Visual Learner):

![Power Spectral Density Decomposition](assets/power_spectral_density.png)

---

## 🏗️ Neural Architecture & Pipeline

```
Raw Multichannel EEG (64 Channels)
               │
               ▼
┌──────────────────────────────────────────┐
│      Signal Preprocessing Pipeline       │
│  • Bandpass IIR Filter: 1.0 – 45.0 Hz    │
│  • Notch Filter: 60.0 Hz (Powerline)     │
│  • Frequency Resampling: 160 Hz          │
│  • Artifact Threshold Clipping: ±500 µV  │
│  • Window Slicing: 4.0 sec (640 samples) │
│  • Z-score Standardization               │
└──────────────────────────────────────────┘
               │
               ▼  Tensor Shape: (Batch, 640, 64)
┌──────────────────────────────────────────┐
│         LSTM-CNN Neural Network          │
│  1. LSTM Layer (64 units, sequences=True)│
│  2. Spatial Dropout (30%)                │
│  3. Conv1D Block 1 (64 filters, k=3, relu│
│  4. MaxPooling1D (pool=2, stride=2)      │
│  5. Conv1D Block 2 (32 filters, k=3, relu│
│  6. GlobalAveragePooling1D               │
│  7. Dense Layer (64 units, ReLU)         │
│  8. Dropout (30%)                        │
│  9. Dense Softmax Output (2 Classes)     │
└──────────────────────────────────────────┘
               │
               ▼
 🎯 Softmax Output: [P(Non-Visual), P(Visual)]
```

---

## 📈 Model Training & Benchmark Results

### 1. Training & Validation Curves
Convergence curves across epochs illustrating categorical cross-entropy loss reduction and accuracy improvement for all tested architectures:

![Training and Validation History](assets/training_history.png)

### 2. Confusion Matrices
Subject-wise holdout validation performance for Baseline LSTM-CNN, CNN-LSTM, CNN-BiLSTM, and CNN-BiLSTM-Attention:

![Confusion Matrices](assets/confusion_matrices.png)

### 3. Receiver Operating Characteristic (ROC-AUC) Curves
ROC analysis demonstrating high discriminative capacity with area under the curve (AUC) reaching **0.9738** for the baseline model:

![ROC Curves](assets/roc_curves.png)

### 4. Benchmark Performance Comparison (PhysioNet EEGMMIDB)

Evaluation was performed with a strict **subject-wise train/val/test split** (60% train / 15% validation / 25% test) to prevent subject data leakage:

| Model Architecture | Validation Accuracy | Validation Balanced Acc | Validation ROC-AUC | Test Accuracy | Test ROC-AUC | 3-Fold CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline LSTM-CNN** | **87.12%** | **87.12%** | **0.9738** | **80.45%** | **0.9278** | **82.9 ± 4.0%** |
| **CNN-LSTM** | 83.33% | 83.33% | 0.9311 | 80.00% | 0.8669 | 81.3 ± 2.3% |
| **CNN-BiLSTM** | 87.12% | 87.12% | 0.9295 | 85.91% | 0.9034 | 83.2 ± 4.5% |
| **CNN-BiLSTM-Attention** | 92.42% | 92.42% | 0.9490 | 85.91% | 0.9518 | 87.9 ± 3.9% |

### 5. Prediction Output Breakdown
Real-time softmax class probabilities generated by the inference engine:

![Prediction Confidence Breakdown](assets/prediction_confidence_gauge.png)

---

## 📁 Repository Structure

```
EEG-LSTM-CNN-Learner-Classification/
├── app.py                                  # Production Streamlit Web Dashboard
├── requirements.txt                        # Lightweight cloud deployment dependencies
├── README.md                               # Comprehensive documentation & visual benchmarks
├── LICENSE                                 # MIT Open-Source License
├── .gitignore                              # Git exclusions (data, cache, checkpoints)
│
├── .streamlit/
│   └── config.toml                         # Streamlit server and theme configurations
│
├── assets/                                 # Research visual plots & figures
│   ├── eeg_multichannel_waveforms.png      # Scalp montage traces
│   ├── power_spectral_density.png          # Alpha band decomposition
│   ├── training_history.png                # Accuracy & loss convergence
│   ├── confusion_matrices.png              # Classification matrices
│   ├── roc_curves.png                      # ROC-AUC curves
│   └── prediction_confidence_gauge.png     # Softmax probability gauge
│
├── model/
│   ├── lstm_cnn_model.keras                # Pre-trained Keras model weights (deployable)
│   └── model_config.json                   # Hyperparameters, architecture & class metadata
│
├── src/
│   ├── __init__.py                         # Package initializer
│   ├── preprocessing.py                    # MNE & SciPy preprocessing pipeline
│   ├── inference.py                        # Cached prediction engine & metric aggregation
│   └── visualizations.py                   # Waveforms, PSD & confidence visualizations
│
├── samples/
│   ├── sample_eyes_open_non_visual.npy     # Benchmark sample (Class 0)
│   └── sample_eyes_closed_visual.npy       # Benchmark sample (Class 1)
│
└── notebook/
    └── lstm_cnn_eeg_eegmmidb.ipynb         # Complete research training notebook
```

---

## 🚀 Quickstart (Local Installation)

### 1. Clone the Repository
```bash
git clone https://github.com/tuhinbiswas404076-ai/EEG-LSTM-CNN-Learner-Classification.git
cd EEG-LSTM-CNN-Learner-Classification
```

### 2. Create and Activate a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🌐 Deploying to Streamlit Community Cloud

Deploy your application for free with a permanent public URL:

### 1-Click Deployment (Recommended)
1. Click here: 👉 **[Deploy on Streamlit Community Cloud](https://share.streamlit.io/deploy?repository=tuhinbiswas404076-ai/EEG-LSTM-CNN-Learner-Classification&branch=main&mainModule=app.py)**
2. Sign in with GitHub (**`tuhinbiswas404076-ai`**).
3. The repository (`tuhinbiswas404076-ai/EEG-LSTM-CNN-Learner-Classification`), branch (`main`), and file (`app.py`) will already be filled in.
4. Click **"Deploy!"**.

Within 1–2 minutes, your web application will be live with a permanent public URL:
$$\text{https://eeg-lstm-cnn-learner-classification.streamlit.app}$$

### Continuous Deployment (Auto-Updates)
Whenever you push new commits to your GitHub repository:
```bash
git add .
git commit -m "update: refine UI and documentation"
git push origin main
```
Streamlit Community Cloud **automatically detects the commit and updates the live web application** without downtime or manual intervention!

---

## 🧪 Supported File Formats

| Format | Extension | Specification |
| :--- | :--- | :--- |
| **European Data Format** | `.edf` | Standard multi-channel recording with at least 64 channels |
| **NumPy Array** | `.npy` / `.npz` | Array of shape `(640, 64)`, `(time, 64)`, or `(N, 640, 64)` |
| **Comma-Separated Values**| `.csv` | Tabular data with at least 64 numeric columns (channels) |

---

## ⚠️ Research & Educational Disclaimer

> **IMPORTANT:**  
> This software and associated trained models are developed strictly for **academic research, scientific evaluation, and educational demonstration**.  
> This platform is **NOT a medical diagnostic device**, clinical tool, or psychological assessment instrument. It must **not** be used to diagnose, treat, monitor, or manage any medical, neurological, or cognitive condition.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
