# Telco Customer Churn Prediction & Strategy


![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Effective-green.svg)
![Imbalanced-Learn](https://img.shields.io/badge/SMOTE-Applied-red.svg)

## Project Overview
In the highly competitive telecommunications industry, retaining existing customers is significantly more cost-effective than acquiring new ones. This project utilizes machine learning to identify customers at high risk of churning. By analyzing demographic data, account information, and service usage, we provide actionable business insights to improve retention rates.

## Project Objectives
- **Identify Key Drivers:** Determine which factors (contract type, monthly charges, tenure) most influence customer attrition.
- **Maximize Recall:** Ensure the model captures the highest possible percentage of actual churners (minimizing False Negatives).
- **Business Strategy:** Translate model predictions into a financial ROI and retention roadmap.

## Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-Learn`, `XGBoost`
- **Imbalanced Data:** `Imbalanced-Learn (SMOTE)`

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Analyzed distribution of churn (Target variable).
- Identified correlations between tenure, monthly charges, and total charges.
- Visualized categorical impacts such as contract types and internet service providers.

### 2. Data Preprocessing
- **Cleaning:** Handled missing values in `TotalCharges` and converted data types.
- **Encoding:** Applied Label Encoding for binary features and One-Hot Encoding for multi-category features.
- **Scaling:** Used `StandardScaler` to normalize numerical data for distance-based algorithms.

### 3. Feature Engineering & Selection
- **New Features:** Created `AvgChargePerTenureMonth`, `TenureGroup`, and `ServiceCount`.
- **Selection:** Used a hybrid approach combining **ANOVA F-tests** and **Random Forest Feature Importance** to select the most predictive subset of features.

### 4. Advanced Modeling & Optimization
- **Handling Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the training set.
- **Model Suite:** Trained Logistic Regression, KNN, Random Forest, Gradient Boosting, and XGBoost.
- **Tuning:** Performed `RandomizedSearchCV` to optimize hyperparameters for top-performing models.
- **Ensemble:** Built a **Soft Voting Classifier** to combine the strengths of the best models.

### 5. Evaluation
- Utilized **Stratified K-Fold Cross-Validation** to ensure model stability.
- Evaluated performance using ROC-AUC curves and Confusion Matrices.
- Focused on **Recall** and **F1-Score** as primary business metrics.

## Contributors
- **Mahbod Khalati**
- **Baran Hosseini**
- **Benyamin Karimizadeh**

---
*Note: This project was developed as part of a Machine Learning course initiative.*

