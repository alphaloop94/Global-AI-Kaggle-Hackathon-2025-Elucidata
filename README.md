# 🔬 Elucidata AI Challenge 2025 – Spatial Cell-Type Composition Prediction

Ranked **Top 204** in the **Global AI Kaggle Hackathon 2025** (Mar–May 2025) organized by Elucidata.

This repository contains my solution for predicting spatial cell-type compositions from histology images using a deep learning pipeline built with TensorFlow and Keras. The model leverages CNNs (EfficientNet & ResNet backbones) to learn from image patches and spot coordinates.

📁 [Kaggle Competition Page](https://www.kaggle.com/competitions/el-hackathon-2025/overview)

---

## 🧠 Problem Statement

Predict the composition of 35 different cell types at specific spatial locations (spots) on whole-slide histology images, combining image and coordinate features.

---

## 🏗️ Project Structure

- `load_data()`: Loads image and spatial data from `.h5` format.
- `extract_patch()`: Extracts and rescales image patches around spots at multiple scales.
- `prepare_dataset()`: Prepares image patches, normalized coordinates, and target vectors.
- `build_model()`: Builds CNN models using EfficientNetB0/B1/B2 or ResNet50 backbones + coordinate fusion.
- `train_model()`: Trains a single CNN model with early stopping, checkpointing, and LR scheduling.
- `train_ensemble()`: Trains a diverse ensemble of 4 CNNs with different seeds and architectures.
- `predict_and_save()`: Generates ensemble predictions and creates submission file.

---

## 🛠️ Model Details

- **Backbones**: EfficientNetB0, EfficientNetB1, EfficientNetB2, ResNet50
- **Fusion**: CNN features + XY coordinates → concatenated → dense layers
- **Loss**: Mean Absolute Error (MAE)
- **Optimizer**: Adam with learning rate scheduling and gradient clipping
- **Augmentations**: Random flip, rotation, zoom, brightness, contrast, translation
- **Training Strategy**: 4-model ensemble with different seeds and backbones

---

## 🚀 How to Run

1. **Install dependencies:**  
   ```bash
   pip install tensorflow h5py pandas opencv-python scikit-learn
   ```

2. **Place the dataset file:**  
   Download `elucidata_ai_challenge_data.h5` from Kaggle and place it under:
   ```bash
   /kaggle/input/el-hackathon-2025/
   ```

3. **Run the training & prediction pipeline:**  
   ```bash
   python your_script.py
   ```

4. **Output:**  
   The final output will be saved as:
   ```bash
   submission.csv
   ```

---

## 📈 Results

- ✅ Trained a 4-model ensemble combining **EfficientNetB0**, **B1**, **B2**, and **ResNet50**
- 🧠 Incorporated both **histological features** and **spatial coordinates**
- 📊 Achieved competitive performance with strong generalization on the test slide
- 🏆 Final Ranking: **Top 204** among global participants

---

## 📂 Dataset

The dataset consists of:

- 🖼️ Whole-slide images in **HDF5** format
- 📍 Spot-level annotations with:
  - `x, y` coordinates
  - Relative cell type composition (**35 types**)
- 🔬 One test slide (`S_7`) for final evaluation

---

## 📦 Files

- `your_script.py`: Main script for data preparation, model training, and prediction
- `models/`: Directory where trained models are saved (generated automatically)
- `submission.csv`: Final output file in Kaggle submission format
