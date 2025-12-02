# MediClassify - Data-Efficient Skin Disease Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of a data-efficient framework for skin disease classification using transfer learning and improved training techniques.

## 📜 Overview

This project tackles the challenge of medical image classification with limited data by leveraging:
- **Transfer Learning**: ResNet-50 pretrained on ImageNet
- **Fine-tuning**: Unfreezing deeper layers for better feature adaptation
- **Class Balancing**: Weighted loss for imbalanced datasets
- **Regularization**: Dropout and weight decay to prevent overfitting
- **Smart Training**: Learning rate scheduling and early stopping

## ✨ Features

- ResNet-50 baseline classifier with enhanced architecture
- Automatic handling of imbalanced classes
- Early stopping to prevent overfitting
- Learning rate scheduling for optimal convergence
- Confusion matrix and classification report generation
- Model checkpointing (saves best model)

## 📂 Repository Structure

```
├── config.py           # Configuration and hyperparameters
├── dataset.py          # Dataset class for loading skin lesion data
├── model.py            # Model architecture (ResNet50)
├── train.py            # Training script
├── utils.py            # Training and evaluation utilities
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/MediClassify.git
cd MediClassify
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

This project uses the **PAD-UFES-20 dataset**. Due to its size, it is not included in this repository.

Download the dataset from Kaggle or the official repository. You will need the images and the `metadata.csv` file.

Place the data in the following structure:
```
data/
├── metadata.csv
└── images/
    ├── img_001.png
    ├── img_002.png
    └── ...
```

### 4. Train the Model

```bash
python train.py
```

The script will:
- Load and preprocess the data
- Train the model with data augmentation
- Apply early stopping and learning rate scheduling
- Save the best model to `best_model.pth`
- Generate confusion matrix and classification report

## 🎯 Model Improvements

- **Deeper classifier head**: Added dropout and intermediate layers
- **Partial unfreezing**: Fine-tunes layer4 of ResNet-50
- **Class weights**: Handles imbalanced dataset automatically
- **AdamW optimizer**: Better weight decay implementation
- **ReduceLROnPlateau**: Adaptive learning rate reduction
- **Early stopping**: Prevents overfitting (patience=5)

## 📊 Configuration

Edit `config.py` to adjust:
- Batch size, learning rate, epochs
- Enable/disable class weights, scheduler, early stopping
- Novel classes for few-shot learning experiments

## 🙏 Acknowledgments

This project uses the PAD-UFES-20 dataset. We thank the authors and contributors for making this valuable resource publicly available.

## 📄 License

This project is licensed under the MIT License.
