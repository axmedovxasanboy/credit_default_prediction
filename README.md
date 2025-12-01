# Credit Default Prediction Model (Bank Customer Dataset)

## Project Goal

The main goal of this project was to build, improve, and test a strong machine learning model that predicts whether a customer will default on credit (binary classification). Its performance was measured using the ROC-AUC score.

**Target Variable:** `default` (1 = Default, 0 = Non-Default)

## Initial Challenge & Baseline
1. **Severe Class Imbalance:** About **95%** of cases were non-defaults (0) and only **5%** were defaults (1), so special techniques were needed.
2. **Data Quality Issues:** Important numeric features (like `loan_amount`, `annual_income`) were stored as strings because of symbols and commas.
3. **High Cardinality:** Non-useful features such as `referral_code` and extra ID columns added unnecessary noise.


---

## Modeling Strategy and Achievements

The project followed a meticulous three-phase strategy to maximize predictive performance:

### Phase 1: Data Cleansing and Engineering

| Step | Action |
| :--- | :--- |
| **Data Cleaning** | Cleaned and converted 11 financial columns from `object` to `float` (e.g., `loan_amount`, `annual_income`). |
| **Missing Values** | Imputed missing values in `employment_length`, `revolving_balance`, and `num_delinquencies_2yrs` using the **median**. |
| **Feature Engineering** | Engineered **three highly predictive financial ratios**, including the effective `cash_flow_to_income_ratio`. |
| **Encoding** | Applied **One-Hot Encoding (OHE)** to 10 cleaned categorical features (e.g., `loan_type`, `employment_type`) and dropped 9 non-predictive ID columns. |

### Phase 2: Model Training and Optimization

| Model | Technique | ROC-AUC Score |
| :--- | :--- | :--- |
| **Baseline XGBoost** | Training with **Scale Position Weight ($\approx 18.59$ to handle imbalance)**. | $0.7870$ |
| **Tuned XGBoost (Final)**| **Randomized Search Cross-Validation** (50 Iterations). | **$\mathbf{0.8073}$** |
| **Model Comparison**| Tested against CatBoost, Logistic Regression, and Decision Tree. | - |

---

## Key Results and Business Insights

The final Tuned XGBoost model achieved a significant improvement over the baseline.

### Final Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Final ROC-AUC** | **$\mathbf{0.8073}$** |
| **Cross-Validation Score** | $0.8028$ | 

### Top 15 Predictive Features

Understanding these features provides immediate, actionable intelligence for the bank's underwriting and risk management teams.

| Rank | Feature | Importance |
| :--- | :--- |  :--- |
| **1** | `credit_score` | 427 (Most important factor) |
| **2** | `age` | 224 |
| **3** | `credit_utilization` | 208 | 
| **4** | `monthly_free_cash_flow` | 178 |
| **9** | `cash_flow_to_income_ratio` | 106 (**Engineered Ratio**) |
| **11** | `loan_type_Personal` | 82 |
| **13** | `regional_unemployment_rate` | 79 |
| *... other debt/income ratios.* | | | | |

---

## Repository Contents

This repository contains the following critical files:

* **`credit_default_model.ipynb`:** The full Jupyter Notebook containing all steps: data loading, cleansing, feature engineering, model training, tuning, and evaluation.
* **`best_xgb_model.json` (To be saved):** The serialized, production-ready final model object (XGBoost) saved with its optimal hyperparameters.
* **`requirements.txt`:** A list of all necessary Python libraries (`pandas`, `scikit-learn`, `xgboost`, `catboost`, etc.).

### Next Steps (Post-Tuning)

1.  **Implement LightGBM and Feature Selection (Current Run):** Compare performance against the $0.8073$ XGBoost score.
2.  **Model Serialization:** Save the best performing model (XGBoost or LightGBM).
3.  **Optimal Threshold Selection:** Conduct a business cost/benefit analysis (Cost of False Positives vs. False Negatives) to determine the best probability threshold for operational deployment.
