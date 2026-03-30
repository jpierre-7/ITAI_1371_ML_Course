# Midterm Project: Vitamin Deficiency Disease Analysis

## Project Overview

This midterm project investigates vitamin deficiency and its relationship with selected health outcomes using the `vitamin_deficiency_disease_dataset_20260123.csv` dataset from Kaggle. The goal is to build a multi-class disease classification model using patient features, with a focus on training and comparing machine learning models including Random Forest Classifier, Linear Regression (as baseline), and XGBoost.

## Repository Structure

- `data_analysis.ipynb`: Main Jupyter Notebook containing data loading, analysis, visualization, model development, and results.
- `data/`
  - `vitamin_deficiency_disease_dataset_20260123.csv`: Primary dataset for analysis.
  - `COPY_vitamin_deficiency_disease_dataset_20260123.csv`: Duplicate of the dataset for backup or experimentation.

## Key Tasks Completed

1. Data Loading and Overview
2. Data Cleaning and Missing Value Handling (correcting mistaken nulls, dropping problematic columns)
3. Exploratory Data Analysis (EDA) with visualizations:
   - Disease distribution countplot
   - Histograms for demographic and nutritional features
   - Correlation heatmap
   - Disease vs. diet type analysis
4. Feature Engineering and Transformation
5. Model Development and Comparison:
   - Random Forest Classifier
   - Linear Regression (baseline)
   - XGBoost Classifier
6. Model Evaluation and Interpretation

## Models and Results Summary

- **Random Forest Classifier**: Primary model with balanced class weighting, achieving reasonable performance with variations by class due to dataset imbalance.
- **Linear Regression**: Used as an unconventional baseline, predicting encoded labels as continuous values.
- **XGBoost Classifier**: Gradient boosting approach showing competitive performance.

The Random Forest was selected as the main model due to its balance of performance, interpretability, and robustness. Evaluation includes accuracy, classification reports, and confusion matrices on a stratified 20% test split.

## How to Run

1. Open `midterm_project/data_analysis.ipynb` in Jupyter Notebook / JupyterLab.
2. Run all cells sequentially.
3. Confirm the dataset is present in `midterm_project/data/`.

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost (optional, falls back to GradientBoostingClassifier)
- joblib

## Learning Outcomes

- Practical data science workflow from raw data to results
- EDA and visualization insights
- Application and comparison of classification algorithms
- Interpretation of model performance and limitations
- Handling imbalanced datasets and feature engineering
