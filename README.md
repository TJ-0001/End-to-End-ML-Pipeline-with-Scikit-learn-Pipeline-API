# Task — Telco Customer Churn Prediction Pipeline

## Objective

The objective of this task is to build a **reusable, end-to-end machine learning pipeline** for predicting customer churn using the Telco Churn dataset. The pipeline includes data preprocessing, model training, hyperparameter tuning, evaluation, and export for production use.

---

## Dataset

**Name:** Telco Customer Churn
**Source:** Kaggle
**Features (21 columns):**

1. `customerID`
2. `gender`
3. `SeniorCitizen`
4. `Partner`
5. `Dependents`
6. `tenure`
7. `PhoneService`
8. `MultipleLines`
9. `InternetService`
10. `OnlineSecurity`
11. `OnlineBackup`
12. `DeviceProtection`
13. `TechSupport`
14. `StreamingTV`
15. `StreamingMovies`
16. `Contract`
17. `PaperlessBilling`
18. `PaymentMethod`
19. `MonthlyCharges`
20. `TotalCharges`
21. `Churn` (target variable)

---

## Steps Performed in Code

### 1. Dataset Loading

* Loaded dataset using `pandas.read_csv()`.
* Displayed first few rows using `df.head()`.
* Printed dataset shape and column names.
* Checked dataset info with `df.info()` to inspect data types and null values.
* Converted `TotalCharges` to numeric (handling errors) and `Churn` to numeric (0 = No, 1 = Yes).

---

### 2. Data Preprocessing

* Split features and target variable (`X` and `y`).
* Identified **numerical columns**: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`.
* Identified **categorical columns**: remaining object-type features.
* Built preprocessing pipelines using `ColumnTransformer`:

  * **Numerical:** `StandardScaler` for scaling.
  * **Categorical:** `OneHotEncoder` for encoding categorical variables.

---

### 3. Model Development & Training

* Created ML pipelines combining preprocessing and classifier:

  * **Logistic Regression**
  * **Random Forest Classifier**
* Split dataset into training and test sets (`80/20` split, stratified by target).
* Used `GridSearchCV` to tune hyperparameters:

  * Logistic Regression: regularization strength `C`.
  * Random Forest: number of estimators, max depth, min samples split.

---

### 4. Model Evaluation

* Evaluated models using:

  * Classification report (`precision`, `recall`, `f1-score`)
  * ROC-AUC score
  * Confusion matrix
* Random Forest generally showed better performance compared to Logistic Regression.

---

### 5. Pipeline Export

* Exported the **best-performing pipeline** (Random Forest with preprocessing) using `joblib`.
* Saved pipeline can be loaded and used for prediction on new data with a single command.

---

## Key Findings

* `TotalCharges` had some non-numeric entries that needed correction.
* One-Hot Encoding and scaling ensured that all features were usable in models.
* Random Forest performed better than Logistic Regression in predicting churn (based on ROC-AUC and classification metrics).
* Using a full ML pipeline ensures **reproducibility** and **production-readiness**.

---

## Files Included

* `Task-2-Telco-Churn-Pipeline.ipynb` — Jupyter Notebook with full code, preprocessing, model training, and evaluation.
* `Telco-Customer-Churn.csv` — Dataset used for analysis and modeling.
* `churn_prediction_pipeline.pkl` — Exported pipeline ready for production.

---

## References

* [Telco Customer Churn Dataset — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
