🩻 Fracture Classification with PyTorch

A CNN-based image classification model built with PyTorch to detect fractured vs. normal medical images.

🚀 Features

Custom 3-layer CNN for binary classification.

Uses PyTorch & TorchVision for modeling and data loading.

Includes training & validation loops with accuracy/loss tracking.

Optuna integration for hyperparameter tuning.

📦 Installation
git clone https://github.com/your-username/fracture-classification.git
cd fracture-classification
pip install -r requirements.txt

🗂️ Dataset Structure
dataset/
├── train/
│   ├── fractured/
│   ├── normal/
├── val/
│   ├── fractured/
│   ├── normal/

🧠 Model Overview

3 Conv2D layers → ReLU → MaxPooling

Fully connected layer → 2-class output

🔄 Usage

Open and run the notebook:

jupyter notebook Fracture_Classification_PyTorch.ipynb

📊 Results

Training Accuracy: ~95%

Validation Accuracy: ~90%
