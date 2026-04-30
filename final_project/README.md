# Sleep Disorder Classification — Final Project

A machine learning project that classifies sleep disorders (None, Insomnia, Sleep Apnea) using the [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) from Kaggle. Two models (Random Forest and XGBoost) are trained, compared, and audited for fairness across demographic groups.

---

## Project Overview

**Goal:** Predict whether a patient has no sleep disorder, insomnia, or sleep apnea based on lifestyle and health features.

**Models:** Random Forest Classifier vs. XGBoost Classifier

**Evaluation Metrics:** Accuracy, Weighted F1-Score, ROC-AUC (One-vs-Rest), and per-class precision/recall

**Fairness Audit:** Per-group performance breakdown by Gender and BMI Category

---

## Dataset

- **Source:** `Sleep_health_and_lifestyle_dataset.csv` (place in `./data/`)
- **Target variable:** `Sleep Disorder` (None / Insomnia / Sleep Apnea)
- **Features used:**

| Feature | Type |
|---|---|
| Age | Numerical |
| Gender | Categorical |
| Occupation | Categorical |
| Sleep Duration | Numerical |
| Quality of Sleep | Numerical |
| Physical Activity Level | Numerical |
| Stress Level | Numerical |
| BMI Category | Categorical |
| Heart Rate | Numerical |
| Daily Steps | Numerical |
| Systolic BP | Numerical (parsed from Blood Pressure) |
| Diastolic BP | Numerical (parsed from Blood Pressure) |

---

## Project Structure

```
.
├── data/
│   └── Sleep_health_and_lifestyle_dataset.csv
├── analysis.ipynb          # Main notebook
└── README.md
```

---

## Notebook Sections

### 1. Basic Info & Cleaning
- Load dataset and inspect shape, dtypes, and missing values
- Fill missing `Sleep Disorder` values with `"None"`
- Parse `Blood Pressure` string into separate `Systolic` and `Diastolic` columns
- Check for and report duplicate entries

### 2. Exploratory Data Analysis
- Histograms of all numerical features (Age, Sleep Duration, Stress Level, etc.)
- Bar charts of categorical features (Gender, Occupation, BMI Category, Sleep Disorder)
- Correlation heatmap of numerical features
- Box plots of each numerical feature broken down by sleep disorder class
- Line plot of average sleep quality by age group and disorder
- Count plot of BMI category distribution by disorder

### 3. Model Training & Comparison

**Preprocessing:**
- Drop non-feature columns (`Person ID`, `Blood Pressure`, `Age_Group`)
- Label encode categorical features and the target variable
- Standardize features with `StandardScaler`
- 80/20 stratified train/test split

**Random Forest:**
- `n_estimators=100`, `max_depth=15`, `min_samples_split=5`, `min_samples_leaf=2`
- 5-fold stratified cross-validation on training data

**XGBoost:**
- `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`
- 5-fold stratified cross-validation on training data

**Visualizations:**
- Confusion matrices for both models
- Side-by-side accuracy and F1-score bar chart
- Top-10 feature importances for each model
- ROC curves (One-vs-Rest) per class for both models

### 4. Fairness Auditing
- Per-gender accuracy and weighted F1 for both models
- Per-BMI-category accuracy and weighted F1 for both models (groups with fewer than 5 samples are flagged)
- Bar chart comparing model accuracy across gender groups

---

## Results Summary

Both models are evaluated on accuracy, weighted F1-score, ROC-AUC, and cross-validation stability. The fairness audit checks whether model performance is consistent across gender and BMI category subgroups to ensure the classifier does not disproportionately underperform for any demographic.
