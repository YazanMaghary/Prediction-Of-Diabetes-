# 🩺 Diabetes Diagnostic Prediction

**Author:** Yazan Maghary
**Date:** 05/01/2026
**Data Dictionary:** [Data Dictionary](https://docs.google.com/document/d/1QNZHliysHgTINXNo-IUZHDhzKnpI7mfia4gaKJ-97tg/edit?tab=t.0#heading=h.oo5xo3ir4vvl)

---

## 📋 Project Overview

This project builds a machine learning model to predict whether or not a patient has diabetes, based on certain diagnostic measurements. The goal is to maximize **recall** — ensuring diabetic patients are rarely missed, as missing a diagnosis is more dangerous than a false alarm.

---


## 📊 Dataset

| Property | Value |
|---|---|
| **Rows** | 642 |
| **Features** | 9 |
| **Target** | Outcome (0 = No Diabetes, 1 = Diabetes) |
| **Feature Types** | 4 string, 5 numeric |
| **Null Values** | Glucose, BloodPressure, SkinThickness, Insulin |

### Features

| Feature | Type | Description |
|---|---|---|
| Pregnancies | int | Number of pregnancies |
| Glucose | float | Plasma glucose concentration |
| BloodPressure | float | Diastolic blood pressure (mm Hg) |
| SkinThickness | float | Triceps skin fold thickness (mm) |
| Insulin | float | 2-Hour serum insulin (mu U/ml) |
| DiabetesPedigreeFunction | float | Diabetes pedigree function |
| WeightGroup | str | BMI weight category |
| AgeGroup | str | Age group (18-44 / 45-64 / >65) |
| Gender | str | Patient gender |

### Target Distribution

```
No Diabetes (0) → 57.8%  (371 patients)
Diabetes    (1) → 42.2%  (271 patients)
```
> Mild imbalance — SMOTE applied on training data only ✅

---

## 🔍 Table of Contents

1. [Importing Libraries](#importing-libraries)
2. [Load The Data Set](#load-the-data-set)
3. [Check And Display The Data](#check-and-display-the-data)
4. [Data Cleaning](#data-cleaning)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Feature Engineering](#feature-engineering)
7. [Modeling And Evaluation](#modeling-and-evaluation)
8. [Conclusion](#conclusion)

---

## 🧹 Data Cleaning

- Null values imputed with **median** (robust to outliers):
  - `Glucose` — 4 nulls (0.6%)
  - `BloodPressure` — 26 nulls (4.0%)
  - `SkinThickness` — 187 nulls (29.1%)
  - `Insulin` — 311 nulls (48.4%)
- Fixed ordinal category typo: `obsese_3` → `obese_3`
- `Gender` identified as quasi-constant (88% Female) — kept but encoded

---

## 📈 Exploratory Data Analysis

### Key Findings

- **Glucose** is the strongest predictor — diabetic patients have significantly higher glucose levels
- **WeightGroup** shows clear pattern — obese/overweight patients have higher diabetes rates
- **AgeGroup** — patients aged 45-64 have the highest diabetes probability
- **Pregnancies** — higher number of pregnancies correlates with higher diabetes risk
- **Gender** — females have slightly higher diabetes rate (dataset is 88% female)

### Feature vs Target Visualizations

#### AgeGroup vs Outcome

<img width="590" height="390" alt="age" src="https://github.com/user-attachments/assets/8536165b-d0e7-4cde-8c7e-1039bfcbebcc" />

- Feature vs. Target Observations:
  - Based on your business understanding, would you expect this feature to be a predictor of the target?
    - `...` Yes
  - Does this feature appear to be a predictor of the target?
    - `...` Yes , people in age interval 45-64 have higher chance of having diabetes than others

#### Glucose vs Outcome

<img width="590" height="390" alt="output" src="https://github.com/user-attachments/assets/ae1dcb24-9f67-48ce-b8d4-d745bb912738" />

- Feature vs. Target Observations:
  - Based on your business understanding, would you expect this feature to be a predictor of the target?
    - `...` Yest , Glucose is a predictor of Diabetes
  - Does this feature appear to be a predictor of the target?
    - `...` Yes , we can notice that who have diabetes have a higher Glucose level

---

## ⚙️ Feature Engineering

### Preprocessing Pipeline

| Column Type | Columns | Treatment |
|---|---|---|
| **Numeric** | Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, DiabetesPedigreeFunction | Median Imputer + StandardScaler |
| **Ordinal** | WeightGroup, AgeGroup | OrdinalEncoder + StandardScaler |
| **Categorical** | Gender | OneHotEncoder |

```python
# Ordinal orders defined:
WeightGroup : underweight → healthy weight → obese_1 → obese_2 → obese_3 → overweight
AgeGroup    : 18-44 → 45-64 → >65
```

---

## 🤖 Modeling

Two models were trained and evaluated — both focused on maximizing **recall**:

| Model | Approach |
|---|---|
| **Logistic Regression** | SMOTE + GridSearchCV + Threshold tuning |
| **Random Forest Classifier** | SMOTE + GridSearchCV + Threshold tuning |

### Logistic Regression — Best Params

```python
LogisticRegression(
    C       = 0.1,
    penalty = 'l2',
    solver  = 'lbfgs',
    max_iter= 1000
)
```

### Random Forest — Best Params

```python
RandomForestClassifier(
    n_estimators      = 100,
    max_depth         = 5,
    min_samples_leaf  = 8,
    min_samples_split = 10,
    max_features      = 'sqrt'
)
```

---

## 📉 Model Evaluation

### Logistic Regression — With Threshold = 0.3 ⭐

| Metric | Training | Testing | Gap |
|---|---|---|---|
| **Accuracy** | ~0.61 | ~0.71 | ~0.1 ✅ |
| **Diabetes Recall** | **0.94** | **0.89** | 0.05 ✅ |
| **Diabetes Precision** | ~0.44 | ~0.63 | ~0.19 ✅ |

### Random Forest — With Threshold = 0.3

| Metric | Training | Testing | Gap |
|---|---|---|---|
| **Accuracy** | 0.76 | 0.61 | -0.15 ✅ |
| **Diabetes Recall** | **0.97** | **0.94** | -0.03 ✅ |
| **Diabetes Precision** | 0.66 | 0.43 | -0.23 ✅ |

---

### Key Observations

* **Threshold Tuning was the biggest win** — lowering the threshold from 0.5 to 0.3 improved Diabetes Recall from 0.69 → 0.94 (a 25%+ jump) 🎯

* **No Overfitting** — Train and Test metrics are very close across both models, indicating strong generalization ✅

* **Recall is our priority** — For diabetes prediction, missing a real case is dangerous 🏥. The Logistic Regression model achieves **0.94 recall** — catching 94% of all diabetic patients.

* **Trade-off accepted** — Precision dropped to ~0.44, meaning some false alarms exist. This is an acceptable trade-off in a medical context where missing a diagnosis is worse than an extra test.

---

## ✅ Conclusion

> The best model is **Logistic Regression with threshold = 0.3**, achieving a **Diabetes Recall of 0.94** with no overfitting. This means 94% of diabetic patients are correctly identified — only 6% are missed. The threshold tuning proved to be the most impactful technique, improving recall by over 25% compared to the default threshold. 🎯

---

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-lightblue)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11-red)
