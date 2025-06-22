# lightweight_epileptic_seizure_prediction
RF, lightweight, epileptic seizure prediction

以下是根据您提供的工作流程编写的项目 `README.md` 文件，内容涵盖 **CHB-MIT数据集的使用流程、特征提取、模型训练与部署流程**，包括在 Jetson Nano 上的离线推理与预测模块说明。

---

## 🔬 Epileptic Seizure Prediction Pipeline (Based on CHB-MIT Dataset)

This project provides an **end-to-end pipeline** for epileptic seizure prediction using EEG data from the **CHB-MIT dataset**, including online channel selection, nonlinear feature extraction with Serial-EMD, global feature ranking, and final model deployment on **Jetson Nano**.

---

### 📁 Dataset

* **Source:** CHB-MIT Scalp EEG Database
* **Preprocessing:**

  * Seizure segmentation and slicing code is assumed to be completed **(not included here)**.
  * EEG segments are labeled as:

    * **Positive (1)**: preictal
    * **Negative (0)**: interictal
    * **Excluded**: ictal, postictal, undefined (not used for training)

---

### 🧠 Pipeline Overview

#### 1. 🔋 Energy-Based Feature Extraction (Online Phase)

* Script: `online_power_features.py`
* Purpose:

  * To extract frequency-domain energy characteristics
  * Serve as inputs for channel selection

#### 2. 📡 Channel Selection and Pre-Training

* Script: `online_train.py`
* Function:

  * Train a lightweight classifier to assess per-channel contribution
  * Select top **3 / 6 / 9** most informative channels
* Output:

  * `sorted_channel_*.txt`: Channel index list based on contribution

#### 3. 🌊 Nonlinear Feature Extraction using Serial-EMD

* Method: Serial Empirical Mode Decomposition (SEMD)
* Applied only on **selected channels**
* Extracted Features:


#### 4. 🌐 Global Feature Contribution via Random Forest

* Method: Random Forest feature importance ranking
* Output:

  * `feature_importance.npy`: Global feature contribution matrix used for interpretability and final feature selection

#### 5. 🧩 Training the 1D-CNN for Segment Classification

* Model: Lightweight **1D Convolutional Neural Network**
* Input: Selected nonlinear features
* Output: Trained model `best_model_*.h5`
* Target: Binary classification of EEG slice as **preictal** or **interictal**

---

### 🚀 Deployment on Jetson Nano (Offline Phase)

#### 6. 🧪 Model and Artifact Deployment

* Script: `offline_test.py`
* Deployed components:

  * `sorted_channel_*.npy`: Selected channel index list
  * `feature_importance.npy`: Global feature contribution matrix
  * `best_model_*.h5`: Trained 1D-CNN model

#### 7. ⏱️ Real-time Seizure Prediction

* Script: `offline_pred.py`
* Function:

  * Load EEG segments
  * Perform slice classification
  * Predict imminent seizure events using threshold-based logic

---

