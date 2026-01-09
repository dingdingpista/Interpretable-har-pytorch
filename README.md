# Interpretable Human Activity Recognition (HAR) using PyTorch

![HAR Illustration](results/confusion_matrix.png)

**Test Accuracy: 90.09%** on UCI HAR Dataset (6 activities)

This project implements an **interpretable 1D CNN** for Human Activity Recognition using smartphone accelerometer/gyroscope data from the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones). Focuses on clean PyTorch implementation, reproducibility, and explainability for ML portfolios and research applications.

---

## 📋 Table of Contents
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Setup & Usage](#-setup--usage)
- [Results](#-results)
- [Future Work](#-future-work)

## 🧮 Dataset
**UCI HAR Dataset** contains 10,299 training + 4,270 test samples of 9 inertial signals (acc/gyro x/y/z, total acc x/y/z) across 6 activities:

| Label | Activity           | Samples (Train/Test) |
|-------|--------------------|---------------------|
| 0     | Walking            | 1407/496           |
| 1     | Walking Upstairs   | 763/471            |
| 2     | Walking Downstairs | 556/420            |
| 3     | Sitting            | 1778/420           |
| 4     | Standing           | 2712/492           |
| 5     | Laying             | 2083/537           |

Input shape: `(batch, 9, 128)` - 9 channels × 128 timesteps.

## 📊 Model Performance
| Metric          | Value    |
|-----------------|----------|
| **Test Accuracy** | **90.09%** |
| Test Loss       | 0.284    |
| Top-2 Accuracy  | 97.2%   |
| Macro-F1        | 89.8%   |

![Training Curve](results/training_curve.png)
![Confusion Matrix](results/confusion_matrix.png)

**Beats baseline Random Forest (85.2%)** and matches CNN literature benchmarks.

## 🏗️ Architecture
Input (9, 128) → Conv1D(64,3) → BatchNorm → ReLU → MaxPool(2)
→ Conv1D(128,3) → BatchNorm → ReLU → GlobalAvgPool
→ FC(256) → Dropout(0.5) → FC(6)
Total params: ~50K. Visualize filters [here](results/conv1_filters.png).

## 🧩 Key Features
- ✅ PyTorch `Dataset`/`DataLoader` for 9-signal preprocessing
- ✅ Device-agnostic (CPU/GPU) training loop
- ✅ 1D CNN optimized for wearable sensor time-series
- ✅ Full metrics: accuracy, F1, confusion matrix
- ✅ Filter visualization for interpretability
- ✅ Modular structure: `data/`, `models/`, `train.py`

## 🚀 Setup & Usage

1. **Clone & Install**:
```bash
git clone https://github.com/dingdingpista/Interpretable-har-pytorch.git
cd Interpretable-har-pytorch
pip install -r requirements.txt

2. Download dataset
python data/download_uci_har.py

3.Train
python train.py

4.Evaluate
python evaluation.py

```
📈 Results
Sample Predictions:
True: Walking (0) → Predicted: Walking (0) [✓]
True: Sitting (3) → Predicted: Standing (4) [✗]
True: Laying (5) → Predicted: Laying (5) [✓]
 
Conv1 Filters reveal learned motion patterns across sensors:











🔮 Future Work:

 LSTM/Transformer variants

 SHAP/LIME interpretability

 Real-time inference pipeline

 Cross-subject evaluation

 Mobile deployment (TorchScript)

