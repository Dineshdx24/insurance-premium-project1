# Insurance Premium Prediction

## 📌 Project Overview

This project predicts insurance premium amounts based on customer demographic, health, financial, lifestyle, and policy-related information.

The project covers the complete machine learning workflow, including:

- Data loading
- Data cleaning
- Missing value handling
- Exploratory Data Analysis (EDA)
- Feature engineering
- Categorical encoding
- Feature transformation
- Model training
- Model evaluation
- Model serialization
- Streamlit deployment

---

## 🎯 Problem Statement

The goal of this project is to develop a machine learning regression model that can predict the insurance premium amount for a customer based on their available information.

This can help insurance companies estimate premiums more efficiently and provide data-driven pricing predictions.

---

## 📊 Dataset

The dataset contains customer and policy-related information such as:

- Age
- Gender
- Annual Income
- Marital Status
- Number of Dependents
- Education Level
- Occupation
- Health Score
- Previous Claims
- Vehicle Age
- Credit Score
- Insurance Duration
- Location
- Policy Type
- Customer Feedback
- Smoking Status
- Exercise Frequency
- Property Type

The target variable is:

**Premium Amount**

The original training and testing datasets are not included in this repository because of their large file sizes.

---

## 🔧 Data Preprocessing

The following preprocessing techniques were applied:

- Missing value treatment
- Median imputation for numerical variables
- Mode imputation for categorical variables
- Categorical variable encoding
- Feature engineering
- Skewness analysis
- Kurtosis analysis
- Feature transformation
- Train-test splitting

---

## ⚙️ Feature Engineering

Additional features were created to improve model performance, including:

- Income per dependent
- Claims per year
- Income × Age
- Credit Score × Annual Income
- Health-related interaction features
- Vehicle/policy duration features

---

## 🤖 Machine Learning Models

Several regression models were evaluated, including:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- AdaBoost Regressor

The models were compared using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

---

## 🏆 Model Performance

The best-performing model during experimentation was based on the evaluation results obtained on the validation/test data.

| Model | MSE | MAE | R² Score |
|---|---:|---:|---:|
| Decision Tree | 1,537,581 | 902.97 | -1.058 |
| Linear Regression | 745,044 | 667.17 | 0.003 |
| Random Forest | 733,776 | 660.92 | 0.018 |
| Gradient Boosting | 728,384 | 656.98 | 0.025 |
| XGBoost | 717,999 | 644.94 | 0.039 |
| AdaBoost | 840,690 | 767.78 | -0.125 |

> Note: These are the model evaluation results obtained during experimentation. Further feature engineering and hyperparameter optimization may improve performance.

---

## 💾 Saved Model Files

The repository contains serialized files used by the application:

- `model.pkl` — trained machine learning model
- `encoders.pkl` — categorical encoders
- `feature_columns.pkl` — feature column information

---

## 🌐 Streamlit Application

The project includes a Streamlit application for making insurance premium predictions.

The application is contained in:

```text
app.py
