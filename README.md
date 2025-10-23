# Heart Disease Prediction

Predicting the presence of heart disease using classical machine learning models and a PyTorch neural network. The project also includes explainability using SHAP.

---

## Table of Contents

* [Dataset](#dataset)
* [Project Overview](#project-overview)
* [Usage](#usage)
* [Modeling](#modeling)
* [Explainability](#explainability)
* [Results](#results)
* [Saving Artifacts](#saving-artifacts)
* [References](#references)

---

## Dataset

* **Source:** [UCI Heart Disease Dataset on Kaggle](https://www.kaggle.com/code/moatazmohamed8804/predicting-heart-disease-presence)
* **Description:** The dataset contains patient information (age, sex, blood pressure, cholesterol, etc.) and whether they have heart disease.
* **Target variable:** `num` 

---

## Project Overview

This project performs the following steps:

1. **Exploratory Data Analysis (EDA):** Understanding target distribution, feature correlations, and missing values.
2. **Data Preprocessing:** Imputation of missing values, scaling numerical features, and one-hot encoding categorical features.
3. **Modeling:** Training and evaluating three models:

   * RandomForest
   * XGBoost
   * PyTorch Neural Network
4. **Explainability:** Feature importance and SHAP analysis for model interpretability.

---

**Dependencies include:**

* Python
* pandas, numpy, matplotlib, seaborn
* scikit-learn
* xgboost
* torch
* shap

---

## Usage

1. Place the dataset in `/kaggle/input/heart-disease-data/heart_disease_uci.csv` (or adjust the path in the notebook).
2. Run the notebook step by step to:

   * Preprocess data
   * Train models
   * Evaluate performance
   * Generate SHAP explainability plots

---

## Modeling

* **RandomForest:** Basic ensemble classifier for feature importance and accuracy.
* **XGBoost:** Gradient boosting classifier, better handling of tabular data.
* **PyTorch Neural Network:** Feed-forward network with two hidden layers, batch normalization, dropout, and early stopping.

**Metrics:** Accuracy, ROC-AUC, confusion matrix, classification report.

---

## Explainability

* **TreeExplainer:** Fast SHAP explanations for XGBoost model.
* **KernelExplainer:** SHAP explanation for PyTorch neural network (slower, approximate).
* Visualizations include SHAP summary bar and beeswarm plots to identify top features contributing to predictions.

---

## Results

The models were evaluated on the held-out test set using **accuracy**, **ROC-AUC**, and for the PyTorch neural network, a detailed classification report.

### Model Performance

| Model            | Accuracy | ROC-AUC | Notes                                                                          |
| ---------------- | -------- | ------- | ------------------------------------------------------------------------------ |
| **RandomForest** | 0.826    | 0.907   | Strong baseline, interpretable via feature importance.                         |
| **XGBoost**      | 0.855    | 0.889   | Slightly higher accuracy, good overall performance.                            |
| **PyTorch NN**   | 0.848    | 0.909   | Neural network with two hidden layers; highest ROC-AUC, good class separation. |

### PyTorch Neural Network Classification Report

| Class | Precision | Recall | F1-score | Support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.90      | 0.74   | 0.81     | 62      |
| 1     | 0.82      | 0.93   | 0.87     | 76      |

* **Macro avg:** Precision 0.86, Recall 0.84, F1-score 0.84
* **Weighted avg:** Precision 0.85, Recall 0.85, F1-score 0.85

### Observations

* The **PyTorch neural network** achieved the **highest ROC-AUC** (0.909), suggesting better distinction between patients with and without heart disease.
* **XGBoost** achieved the **highest accuracy** (0.855), slightly outperforming RandomForest.
* **RandomForest** is a reliable baseline model and provides interpretable feature importance, highlighting the most influential factors in heart disease prediction.
* Across all models, features like `cp` (chest pain type), `thal`, `ca` (number of major vessels), and `oldpeak` were among the most predictive.

---

## Saving Artifacts

* Preprocessor pipeline saved as `preprocessor.joblib`
* PyTorch model weights saved as `best_model_weights.pt`

These artifacts can be loaded later for inference on new data.

---

## References

* [UCI Heart Disease Dataset](https://www.kaggle.com/ronitf/heart-disease-uci)
* [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

**Author:** Amira Qadry
**Date:** October 2025
