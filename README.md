# 🏦 Credit Default Prediction Model (Bank Customer Dataset)

## 🎯 Project Goal

The primary objective of this project was to develop, optimize, and evaluate a robust machine learning model capable of predicting customer credit default (binary classification) with high discriminatory power, measured by the Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

**Target Variable:** `default` (1 = Default, 0 = Non-Default)

## 📊 Initial Challenge & Baseline

The initial model achieved a baseline ROC-AUC of **$0.7770$**. Key challenges identified were:

1.  **Severe Class Imbalance:** The dataset contained approximately $95\%$ non-default cases ($\text{0}$) and only $5\%$ default cases ($\text{1}$), requiring specialized handling.
2.  **Data Quality Issues:** Multiple critical numerical features (e.g., `loan_amount`, `annual_income`) were incorrectly stored as `object` (string) types due to currency symbols and commas.
3.  **High Cardinality:** Non-predictive features like `referral_code` and multiple redundant ID columns added noise.

---

## 🛠️ Modeling Strategy and Achievements

The project followed a meticulous three-phase strategy to maximize predictive performance:

### Phase 1: Data Cleansing and Engineering

| Step | Action | Impact |
| :--- | :--- | :--- |
| **Data Cleaning** | Cleaned and converted 11 financial columns from `object` to `float` (e.g., `loan_amount`, `annual_income`). | Unlocked the use of core financial data for the model. |
| **Missing Values** | Imputed missing values in `employment_length`, `revolving_balance`, and `num_delinquencies_2yrs` using the **median**. | Ensured a complete, usable dataset. |
| **Feature Engineering** | Engineered **three highly predictive financial ratios**, including the effective `cash_flow_to_income_ratio`. | Increased predictive signal by capturing customer financial leverage and liquidity. |
| **Encoding** | Applied **One-Hot Encoding (OHE)** to 10 cleaned categorical features (e.g., `loan_type`, `employment_type`) and dropped 9 non-predictive ID columns. | Created a clean, 118-feature matrix suitable for ML. |

### Phase 2: Model Training and Optimization

| Model | Technique | Key Achievement | ROC-AUC Score |
| :--- | :--- | :--- | :--- |
| **Baseline XGBoost** | Training with **Scale Position Weight ($\approx 18.59$ to handle imbalance)**. | Successfully addressed class imbalance, boosting initial performance. | $0.7870$ |
| **Tuned XGBoost (Final)**| **Randomized Search Cross-Validation** (50 Iterations). | Optimal hyperparameter tuning led to the highest ranking score. | **$\mathbf{0.8073}$** |
| **Model Comparison**| Tested against CatBoost, Logistic Regression, and Decision Tree. | Confirmed that the **Tuned XGBoost** provided the strongest ROC-AUC, validating the ensemble tree approach. | - |

---

## 🔬 Key Results and Business Insights

The final Tuned XGBoost model achieved a significant improvement over the baseline.

### Final Performance Metrics

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Final ROC-AUC** | **$\mathbf{0.8073}$** | The model's ability to correctly rank customers by default risk. **($+3.03\%$ gain)** |
| **Cross-Validation Score** | $0.8028$ | Confirms the model's stability and robustness against overfitting. |

### Top 15 Predictive Features

Understanding these features provides immediate, actionable intelligence for the bank's underwriting and risk management teams.

| Rank | Feature | Category | Importance | Business Implication |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `credit_score` | Primary Risk | 427 | Remains the single most important factor. |
| **2** | `age` | Demographics | 224 | Proxy for financial stability and experience. |
| **3** | `credit_utilization` | Credit Health | 208 | High usage relative to credit limit is a major stress indicator. |
| **4** | `monthly_free_cash_flow` | Liquidity | 178 | Raw cash left after expenses is highly predictive. |
| **9** | `cash_flow_to_income_ratio` | **Engineered Ratio** | 106 | **Confirms the success of feature engineering.** |
| **11** | `loan_type_Personal` | Loan Type | 82 | Suggests higher inherent risk for personal loans vs. mortgage/card. |
| **13** | `regional_unemployment_rate` | Economic Factor | 79 | Highlights the importance of macro-economic context. |
| *... and other core debt/income ratios.* | | | | |

---

## 💻 Repository Contents

This repository contains the following critical files:

* **`credit_default_model.ipynb`:** The full Jupyter Notebook containing all steps: data loading, cleansing, feature engineering, model training, tuning, and evaluation.
* **`best_xgb_model.json` (To be saved):** The serialized, production-ready final model object (XGBoost) saved with its optimal hyperparameters.
* **`requirements.txt`:** A list of all necessary Python libraries (`pandas`, `scikit-learn`, `xgboost`, `catboost`, etc.).

### Next Steps (Post-Tuning)

1.  **Implement LightGBM and Feature Selection (Current Run):** Compare performance against the $0.8073$ XGBoost score.
2.  **Model Serialization:** Save the best performing model (XGBoost or LightGBM).
3.  **Optimal Threshold Selection:** Conduct a business cost/benefit analysis (Cost of False Positives vs. False Negatives) to determine the best probability threshold for operational deployment.
