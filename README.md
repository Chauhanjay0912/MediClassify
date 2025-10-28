# MediClassify - Data-Efficient Medical Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MediClassify** is a comprehensive framework for skin disease classification that demonstrates the power of few-shot learning. This project starts by building a standard multi-class classifier as a performance benchmark and then implements a data-efficient prototypical network to classify new diseases using only a handful of examples.



## 📜 Overview

In many real-world medical imaging scenarios, collecting large, labeled datasets for every possible disease is impractical. This project tackles this challenge by leveraging transfer learning and metric-based few-shot learning.

1.  **Baseline Model**: A fully supervised ResNet-50 model is trained on 6 different skin diseases to establish a benchmark for performance.
2.  **Few-Shot Learning**: We simulate a more realistic scenario where two diseases are "novel" (unseen). A "generalist" feature extractor is trained on the remaining four "base" diseases. This extractor is then used to power a **Prototypical Network**, which can learn to identify the novel diseases from just 5 examples per class (**5-shot learning**).

This framework provides a practical and data-efficient solution for adapting AI models to new, rare diseases where data is scarce.

## ✨ Features

-   **Baseline Classifier**: A powerful ResNet-50 model for multi-class classification.
-   **Generalist Feature Extractor**: Learns robust and transferable features from a set of base classes.
-   **Prototypical Network**: A metric-based few-shot learning implementation for rapid adaptation to new classes.
-   **Performance Evaluation**: Detailed comparison between the fully supervised baseline and the data-efficient few-shot model, including classification reports and confusion matrices.
-   **Feature Space Visualization**: t-SNE plots to visualize how the model organizes and separates different disease classes in its learned feature space.
-   **Sensitivity Analysis**: An analysis of how the few-shot model's accuracy changes with the number of "shots" (examples per class).

## 📂 Repository Structure

```
├── .gitignore          # Files to be ignored by Git
├── LICENSE             # The MIT License file
├── README.md           # This file
├── requirements.txt    # Project dependencies
├── notebooks/
│   └── skin_disease_classification.ipynb   # Main Colab/Jupyter notebook
└── saved_models/       # Saved model weights (ignored by git)
```

## 🚀 Getting Started

Follow these instructions to set up and run the project. This project is designed to run in a Google Colab environment with GPU acceleration.

### Prerequisites

-   Python 3.8+
-   A Google Account (for using Google Colab and Google Drive)

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/MediClassify.git](https://github.com/your-username/MediClassify.git)
cd MediClassify
```

### 2. Set Up the Environment

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the required libraries
pip install -r requirements.txt
```

### 3. Download the Dataset

This project uses the **PAD-UFES-20 dataset**. Due to its size, it is not included in this repository.

1.  Download the dataset from a source like Kaggle or the official repository. You will need the images and the `metadata.csv` file.
2.  Upload the data to your Google Drive with the following structure:
    ```
    My Drive/
    └── skin_disease_project/
        ├── metadata.csv
        └── images/
            ├── img_001.png
            ├── img_002.png
            └── ...
    ```

### 4. How to Run

1.  Upload the `notebooks/skin_disease_classification.ipynb` file to Google Colab.
2.  **Enable GPU Acceleration**: In Colab, go to `Runtime` -> `Change runtime type` and select `GPU` as the hardware accelerator.
3.  Run the first code cell to mount your Google Drive. You will be prompted to authorize access.
4.  Ensure the `BASE_PATH` variable in the notebook correctly points to your project folder in Google Drive:
    ```python
    BASE_PATH = '/content/drive/My Drive/skin_disease_project/'
    ```
5.  Run all cells sequentially to execute the entire pipeline from data loading to model training, evaluation, and visualization.

## 📊 Results

The project demonstrates that few-shot learning can achieve competitive performance with a fraction of the data required by traditional supervised models.

-   **Fully Supervised Baseline (on all data)**: Achieves high accuracy, serving as the upper-bound benchmark.
-   **Few-Shot Model (5-shot)**: Reaches a significant percentage of the baseline's accuracy on novel classes using only 5 examples per class for training.
-   **t-SNE Visualization**: The feature space plot shows that the generalist model learns to create meaningful clusters, separating different diseases effectively—even for the novel classes it has never been trained on.
-   **K-Shot Sensitivity**: Accuracy improves as the number of shots increases, with significant gains observed even from 1 to 5 shots.

## 💡 Future Work

-   Experiment with more advanced few-shot learning algorithms like Relation Networks or MAML.
-   Incorporate metadata (e.g., patient age, lesion location) into the model to potentially improve accuracy.
-   Test the framework on other medical imaging datasets to evaluate its generalizability.

## 🙏 Acknowledgments

This project uses the PAD-UFES-20 dataset. We thank the authors and contributors for making this valuable resource publicly available.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.