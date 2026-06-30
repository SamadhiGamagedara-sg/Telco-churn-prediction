# Telco Customer Churn Prediction

An end-to-end machine learning pipeline that predicts telecom customer churn, covering data cleaning, feature engineering, model comparison with class imbalance handling, and a deployed interactive prediction app.

**Live app:** https://telco-churn-prediction-hxk8jdobpkem4yrpurhiom.streamlit.app/

## Overview

This project uses the Telco Customer Churn dataset to build a classification model that predicts whether a customer is likely to churn, then deploys that model as an interactive web app where a user can enter customer details and get a live prediction.

## Project Workflow

1. **Data cleaning** — handled missing values, blank `TotalCharges` entries, duplicate records, and dropped non-predictive identifier columns.
2. **Exploratory analysis & outlier review** — analyzed distributions of `tenure`, `MonthlyCharges`, and `TotalCharges` using KDE/box plots and a 3-sigma rule. No outliers were removed: flagged values represented legitimate long-tenure, high-value customers rather than data errors, and removing them would have biased the model against the customer segment most valuable to retain.
3. **Feature engineering** — binned `tenure` into Newer / Medium / Long-term groups.
4. **Preprocessing** — built a `ColumnTransformer` pipeline applying StandardScaler to numerical features and One-Hot/Ordinal encoding to categorical features, then split the data into train/test sets.
5. **Model training** — trained and tuned three models (Logistic Regression, Decision Tree, Random Forest) using `GridSearchCV` with stratified k-fold cross-validation, combined with SMOTE to address class imbalance (~27% churn rate in the data).
6. **Model evaluation & selection** — compared models on accuracy, precision, recall, F1, and ROC-AUC, prioritizing recall and F1 over raw accuracy given the class imbalance and the business cost of missing a true churner.
7. **Deployment** — saved the final preprocessing pipeline, model, and label encoder, then built a Streamlit app for live predictions.

## Model Comparison

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.732 | 0.50 | 0.76 | 0.60 | 0.829 |
| Decision Tree | 0.751 | 0.53 | 0.67 | 0.59 | 0.805 |
| **Random Forest (selected)** | **0.764** | **0.54** | **0.73** | **0.62** | **0.829** |

**Why Random Forest:** while Logistic Regression achieves slightly higher recall, Random Forest offers the best overall balance of precision and recall (highest F1 score) and matches the best ROC-AUC, making it the most reliable choice for flagging at-risk customers without an excessive false-positive rate.

## App Preview

**Input form:**

![App input form](screenshots/app_form.png)

**Prediction result:**

![App prediction result](screenshots/app_result.png)

## Tech Stack

- **Data & modeling:** pandas, scikit-learn, imbalanced-learn (SMOTE)
- **App:** Streamlit
- **Deployment:** Streamlit Community Cloud

## Project Structure
├── app.py                          # Streamlit app
├── requirements.txt                # Dependencies for deployment
├── artifacts/                      # Saved model, preprocessor, label encoder
├── data/
│   ├── Telco-Customer-Churn.csv    # Raw dataset
│   └── processed/                  # Cleaned/processed data and later-stage notebooks
├── 02_handling_outliers.ipynb
├── handling_missing_and_duplicate_values.ipynb
└── screenshots/                    # App screenshots used in this README

## Running Locally

```bash
git clone https://github.com/SamadhiGamagedara-sg/Telco-churn-prediction.git
cd Telco-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Key Takeaways

- Class imbalance (27% churn) was addressed with SMOTE inside the training pipeline rather than naively oversampling before the train/test split, avoiding data leakage.
- Model selection was based on F1 and ROC-AUC rather than accuracy, since accuracy is misleading on imbalanced classification problems.
- The full preprocessing pipeline was saved alongside the model so that raw, unprocessed input can be transformed consistently at inference time.
