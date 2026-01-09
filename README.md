# Interpretable Human Activity Recognition (HAR) using PyTorch

![HAR Illustration](results/confusion_matrix.png)  

This project implements an **interpretable deep learning pipeline** for Human Activity Recognition (HAR) using smartphone inertial sensor data. The focus is on **clear reasoning, reproducibility, and explainability**, making it suitable for research-oriented and SOP-driven projects.

---

## 🎯 Project Overview

Human Activity Recognition is the task of identifying physical activities (e.g., walking, sitting, standing) from sensor signals such as accelerometer and gyroscope data. This project uses the **UCI HAR Dataset** to train and evaluate a **1D Convolutional Neural Network (CNN)** that predicts six activity classes:

| Label | Activity           |
|-------|------------------|
| 0     | Walking           |
| 1     | Walking Upstairs  |
| 2     | Walking Downstairs|
| 3     | Sitting           |
| 4     | Standing          |
| 5     | Laying            |

The model is **device-agnostic** (works on CPU or GPU) and designed to provide a **clean, interpretable workflow** for anyone learning deep learning and time-series analysis.

---

## 🧩 Key Features

- **Data Loading & Preprocessing**:  
  - PyTorch `Dataset` and `DataLoader` classes separate data logic from training logic.  
  - Signals are stacked into `(samples, channels, timesteps)` for CNN input.

- **Model Architecture**:  
  - 1D Convolutional Neural Network (CNN) tailored for time-series sensor data.  
  - Global average pooling and fully connected layers for classification.  

- **Training**:  
  - Device-agnostic code supports CPU/GPU.  
  - Standard cross-entropy loss and Adam optimizer.  
  - Training workflow is simple, reproducible, and clear.

- **Evaluation & Metrics**:  
  - Overall metrics: Accuracy, Precision, Recall, F1-score.  
  - Per-class metrics for deeper analysis.  
  - Confusion matrix visualization for interpretability.

- **Interpretability**:  
  - The CNN architecture allows inspection of learned temporal patterns.  
  - Easy extension to feature importance or Grad-CAM style interpretability for time-series.

---

## 🛠️ Setup Instructions

1. **Clone the repository**:

```bash
git clone https://github.com/dingdingpista/Interpretable-har-pytorch.git
cd Interpretable-har-pytorch
