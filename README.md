# Fracture Classification using PyTorch

A deep learning project for classifying X-ray images to detect fractured vs. non-fractured bones using Convolutional Neural Networks (CNNs) implemented in PyTorch.

## 🏥 Project Overview

This project implements an automated bone fracture detection system using X-ray imaging data. The system can classify X-ray images into two categories:
- **Fractured bones** 
- **Non-fractured bones**

The project demonstrates the complete machine learning pipeline from data preprocessing to model optimization, including techniques to address overfitting and improve generalization.

## 🚀 Features

- **Custom CNN Architecture**: Implemented from scratch with configurable layers
- **Data Augmentation**: Comprehensive augmentation techniques to improve model robustness
- **Hyperparameter Optimization**: Automated tuning using Optuna framework
- **Overfitting Prevention**: Batch normalization, dropout, and regularization techniques
- **Performance Visualization**: Training and validation metrics plotting
- **Model Comparison**: Side-by-side analysis of different model versions

## 🏗️ Architecture

### Base CNN Model
- 3 convolutional layers with ReLU activation and max pooling
- 2 fully connected layers
- Output: Binary classification (2 classes)

### Enhanced CNN Model
- Batch normalization after each convolutional layer
- Dropout regularization (0.5)
- Improved generalization capabilities

### Hyperparameter-Optimized Model
- Configurable convolutional layer dimensions
- Tunable dropout rates
- Optimized learning rates and batch sizes

## 📊 Dataset

The project uses the **X-ray Images of Fractured and Healthy Bones** dataset from Kaggle, containing:
- Training set: 80% of data
- Validation set: 20% of data
- Grayscale images resized to 128x128 pixels
- Data augmentation factor: 10x (synthetic data generation)

**Dataset Source**: [X-ray Images of Fractured and Healthy Bones](https://www.kaggle.com/datasets/foyez767/x-ray-images-of-fractured-and-healthy-bones?resource=download) by [foyez767](https://www.kaggle.com/foyez767) on Kaggle

### Data Augmentation Techniques
- Random horizontal flips
- Color jittering (brightness, contrast)
- Random affine transformations (rotation, translation, scaling)
- Random perspective distortion

## 🛠️ Dependencies

```bash
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.0
scikit-learn==1.5.0
torch==2.3.1
torchvision==0.18.1
optuna
PIL (Pillow)
```

## 📁 Project Structure

```
Fracture_Classification-main/
├── Fracture_Classification_PyTorch (3).ipynb  # Main Jupyter notebook
├── README.md                                   # This file
└── Downloads/
    └── X-ray Imaging Dataset for Detecting Fractured vs. Non-Fractured Bones/
        └── Original Dataset/                   # Dataset directory
```

## 🚀 Usage

### 1. Environment Setup
```bash
# Install required packages
pip install pandas==2.2.2 numpy==1.26.4 matplotlib==3.8.0
pip install scikit-learn==1.5.0 torch==2.3.1 torchvision==0.18.1
pip install optuna
```

### 2. Data Preparation
- Place your X-ray dataset in the appropriate directory structure
- Ensure images are organized in subdirectories by class (fractured/non-fractured)

### 3. Model Training
1. Open the Jupyter notebook
2. Run cells sequentially to:
   - Load and preprocess data
   - Train the base model
   - Implement data augmentation
   - Train the enhanced model
   - Perform hyperparameter optimization
   - Train the final optimized model

### 4. Performance Analysis
- View training/validation loss and accuracy plots
- Compare model performances
- Analyze overfitting patterns

## 📈 Results

### Model Performance Improvements
- **Base Model**: Validation accuracy ~78% (overfitting observed)
- **Enhanced Model**: Improved generalization with data augmentation
- **Optimized Model**: ~25% improvement in validation accuracy, no overfitting

### Key Optimizations
- Data augmentation (10x factor)
- Batch normalization
- Dropout regularization (0.7)
- Hyperparameter tuning via Optuna
- Optimal learning rate: ~0.0058
- Optimal batch size: 32

## 🔬 Technical Details

### Training Parameters
- **Epochs**: 30
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 16 (base), 64 (augmented), 32 (optimized)

### Model Architecture Details
- **Input**: 1x128x128 grayscale images
- **Convolutional Layers**: 3 layers with increasing channel dimensions
- **Pooling**: MaxPool2d (2x2)
- **Fully Connected**: 128 → 2 neurons
- **Activation**: ReLU

## 🎯 Future Improvements

- Transfer learning with pre-trained models (ResNet, VGG)
- Multi-class classification for different fracture types
- Real-time inference capabilities
- Web application interface
- Model deployment and API development

## 📚 References

- PyTorch Documentation
- Optuna Hyperparameter Optimization
- Medical Image Analysis Best Practices
- CNN Architecture Design Principles

## 👥 Contributing

Feel free to contribute to this project by:
- Improving the model architecture
- Adding new data augmentation techniques
- Optimizing hyperparameters further
- Creating additional visualization tools

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contact

For questions or contributions, please open an issue or submit a pull request.

---

**Note**: This project is for educational and research purposes. For clinical applications, ensure proper validation and regulatory compliance.
