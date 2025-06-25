# lightweight_epileptic_seizure_prediction

---

## 🔬 Epileptic Seizure Prediction Pipeline (Based on CHB-MIT Dataset)

This project provides an **end-to-end pipeline** for epileptic seizure prediction using EEG data from the **CHB-MIT dataset**, including online channel selection, nonlinear feature extraction with Serial-EMD, global feature ranking, and final model deployment on **Jetson Nano**.

---

### 📁 Dataset

* **Source:** CHB-MIT Scalp EEG Database
* **Preprocessing:**

  * Seizure segmentation and slicing code in 'Preprocessing'

---

### 🧠 Pipeline Overview

#### 1. 🔋 Power-Based Feature Extraction (Online Phase)

* Script: `online_power_features.py`
* Purpose:

  * To extract frequency-domain energy characteristics
  * Serve as inputs for channel selection

#### 2. 📡 Channel reduction and Pre-Training

* Script: `Jetson Nano test/online_train.py`
* Function:

  * Train a lightweight classifier to assess per-channel contribution
  * Select top **3 / 6 / 9** most informative channels
* Output:

  * `sorted_channel_*.npy`: Channel index list based on contribution

#### 3. 🌊 Nonlinear Feature Extraction using Serial-EMD

* Method: Serial Empirical Mode Decomposition (SEMD)
* Applied only on **selected channels**
* Extracted Features:


#### 4. 🧩 Training the 1D-CNN for Segment Classification

* Model: Lightweight **1D Convolutional Neural Network**
* Input: Selected nonlinear features
* Output: Trained model `best_model_*.h5`
* Target: Binary classification of EEG slice as **preictal** or **interictal**

---

### 🚀 Deployment on Jetson Nano (Offline Phase)

#### 5. 🧪 Model and Artifact Deployment

* Script: `Jetson Nano test/offline_test.py`
* Deployed components:

  * `sorted_channel_*.npy`: Selected channel index list
  * `feature_importance.npy`: Global feature contribution matrix
  * `best_model_*.h5`: Trained 1D-CNN model

#### 6. ⏱️ Real-time Seizure Prediction

* Script: `Jetson Nano test/offline_pred.py`
* Function:

  * Load EEG segments
  * Perform slice classification
  * Predict imminent seizure events using threshold-based logic

---
### 📈 Visualizing classification results

Use the MATLAB files in the `Visualizing classification results/` folder:

- `draw*.m`: Visualizes the classification result for each EEG segment  

> With variants across methods like `1EMD`, `2Channel+EMD`, `3Channel+SEMD`, etc.

---

### 🔍 Automatic Threshold Estimation

To assist seizure prediction threshold determination, you can run:

```bash
python Prediction/Pred-K-Means.py
```
---

###  🛠 Requirements

Install required dependencies:

```bash
pip install -r requirements.txt
```
