# ML-Based-DOS-Attack-Prediction-in-WSN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A Machine Learning-based project for predicting Denial-of-Service (DoS) attacks in Wireless Sensor Networks (WSN) using multiple classifiers and advanced feature selection techniques. This project leverages various techniques to preprocess data, train models, visualize results, and predict attack types on new datasets.

---

## 🚀 Project Overview

This project implements a robust pipeline to detect DoS attacks in WSNs using a dataset (`WSN-DS.csv`). It employs **Gini Impurity-based feature selection** to identify significant features and evaluates four machine learning models—**Decision Tree**, **K-Nearest Neighbors (KNN)**, **Random Forest**, and **XGBoost**—using 10-fold cross-validation. The results are visualized through ROC curves, accuracy per fold, attack-wise accuracy, and classifier error plots. A prediction module allows users to input new data (CSV files without target labels) to classify potential attacks.

---

## 🌟 Features

- **Data Preprocessing**: Loads and preprocesses the WSN-DS dataset, removes extraneous spaces, and performs feature selection using Gini Impurity.
- **Feature Selection**: Selects features with Gini scores > 0.01 to optimize model performance.
- **Model Training**: Trains four classifiers (Decision Tree, KNN, Random Forest, XGBoost) with StandardScaler preprocessing.
- **Cross-Validation**: Implements 10-fold stratified cross-validation to ensure robust model evaluation.
- **Visualization**: Generates insightful plots:
  - ROC curves (micro-average AUC)
  - Accuracy per fold
  - Attack-wise accuracy comparison
  - Classifier error rates
- **Prediction Module**: Allows predictions on new datasets (CSV files without target labels) and displays results in a clean console UI.
- **OOP Design**: Modular code structure using classes for data handling, model training, visualization, and prediction.

---

## 📋 Prerequisites

- **Python 3.8+**
- **Dependencies**:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost

Install dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## 🛠 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mahasarabesh/ML-Based-DOS-Attack-Prediction-in-WSN.git
   cd ML-Based-DOS-Attack-Prediction-in-WSN
   ```

2. **Install Dependencies**

3. **Prepare the Dataset**:
   - Give the `WSN-DS.csv` file path as the first argument.
   - For predictions, prepare a CSV file with the same features (excluding the target label).

---

## 📖 Usage

The project is executed via the `results.py` script, which orchestrates data preprocessing, model training, evaluation, visualization, and prediction.

### Running the Project

1. **Train Models and Visualize Results**:
   Run the main script with the training dataset path:
   ```bash
   python experiment.py
   ```
   This will:
   - Preprocess the dataset and select features.
   - Train models using 10-fold cross-validation.
   - Display performance metrics (accuracy, precision, recall, F1-score, processing time).
   - Generate plots (ROC curves, accuracy per fold, attack-wise accuracy, classifier errors).

2. **Predict on New Data**:
   Specify a new CSV file (without target labels) for predictions:
   ```bash
   python results.py path/to/WSN-DS.csv path/to/new_data.csv
   ```

### Example Dataset
- **Training Dataset** (`WSN-DS.csv`): Contains features and the target column (`Attack type`).
- **Prediction Dataset**: A CSV with the same features as `WSN-DS.csv` but without the `Attack type` column.

---

## 📂 Project Structure

```
ML-Based-DOS-Attack-Prediction-in-WSN/
├── data_loader.py     # Data loading, preprocessing
├── model_trainer.py    # Model training and cross-validation 
├── experiment.py       # to perform model training on multple ML Models 
├── feature_selecter.py # Feature selection on dataset 
├── visualizer.py       # Plotting functions (ROC, accuracy, errors)
├── predictor.py        # Prediction on new data
├── results.py          # Main script to orchestrate the workflow
└── README.md           # Project documentation
```

---

## 📊 Results

The project outputs:
- **Gini Scores**: Displays feature importance based on Gini Impurity.
- **Performance Metrics**: Mean accuracy, precision, recall, and F1-score for each model.
- **Processing Times**: Average time per fold for each model.
- **Visualizations**:
  - **ROC Curves**: Micro-average ROC curves with AUC scores.
  - **Accuracy per Fold**: Line plot of accuracy across 10 folds.
  - **Attack-wise Accuracy**: Bar plot comparing model performance per attack type.
  - **Classifier Errors**: Bar plot of error rates (100 - mean accuracy).

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **Star this repository if you find it useful!** ⭐
