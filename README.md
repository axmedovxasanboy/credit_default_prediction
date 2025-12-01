# 🏦 Credit Default Prediction Model (Bank Customer Dataset)

## Project Goal

The main goal of this project was to build, improve, and test a strong machine learning model that predicts whether a customer will default on credit (binary classification). Its performance was measured using the **ROC-AUC score**, which assesses the model's ability to correctly rank customers by risk.

**Target Variable:** `default` (1 = Default, 0 = Non-Default)

---

## Initial Challenge & Baseline

1.  **Severe Class Imbalance:** About **95%** of cases were non-defaults (0) and only **5%** were defaults (1), so special techniques were needed (Scale Position Weight $\approx 18.59$).
2.  **Data Quality Issues:** Important numeric features (like `loan_amount`, `annual_income`) were stored as strings because of symbols and commas.
3.  **High Cardinality:** Non-useful features such as `referral_code` and extra ID columns added unnecessary noise.

---

## Modeling Strategy and Achievements

The project followed a meticulous three-phase strategy to maximize predictive performance and ensure model robustness.

### Phase 1: Data Cleansing and Engineering

| Step | Action |
| :--- | :--- |
| **Data Cleaning** | Cleaned and converted 11 financial columns from `object` to `float` (e.g., `loan_amount`, `annual_income`). |
| **Missing Values** | Imputed missing values in `employment_length`, `revolving_balance`, and `num_delinquencies_2yrs` using the **median**. |
| **Feature Engineering** | Engineered **three highly predictive financial ratios**, including the effective `cash_flow_to_income_ratio`. |
| **Encoding & Pruning** | Applied **One-Hot Encoding (OHE)** to 10 cleaned categorical features, dropped 9 non-predictive ID columns, and performed **Feature Selection** to prune to the **Top 50 most predictive features**. |

### Phase 2: Model Training and Optimization

The final models were trained on the **Pruned Feature Set** to maximize speed and stability.

| Model | Technique | Final ROC-AUC Score | Performance Insight |
| :--- | :--- | :--- | :--- |
| **Baseline XGBoost** | Training with **Scale Position Weight ($\approx 18.59$ to handle imbalance)**. | $0.7870$ | Initial score after cleansing/engineering. |
| **Tuned XGBoost** | **Randomized Search Cross-Validation** (50 Iterations). | **$\mathbf{0.8073}$** | Set the performance ceiling; **$+3.03\%$ gain** from baseline. |
| **Tuned LightGBM** | Randomized Search on Pruned Data. | $\mathbf{0.8073}$ | **Matched the maximum score.** Confirms performance ceiling and provides a deployment alternative. |
| **Tuned CatBoost** | Randomized Search on Pruned Data. | $\mathbf{0.8073}$ | **Matched the maximum score.** Provides a third robust deployment alternative. |

---

## Key Results and Final Models

The final ensemble tree models (XGBoost, LightGBM, CatBoost) achieved the empirical maximum predictive power for this dataset.

### Final Performance Metrics

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Final ROC-AUC** | **$\mathbf{0.8073}$** | Represents the maximum discriminatory power achievable with the current data. |
| **Cross-Validation Score** | $0.8028$ | Confirms the stability and generalization ability of the tuned models. |

### Top 15 Predictive Features

These features, derived from the ensemble models, are crucial for underwriting and risk management:

| Rank | Feature | Importance | Business Implication |
| :--- | :--- | :--- | :--- |
| **1** | `credit_score` | 427 | Most important factor; primary indicator of past credit behavior. |
| **2** | `age` | 224 | Proxy for financial stability and experience. |
| **3** | `credit_utilization` | 208 | High usage relative to credit limit is a major stress indicator. |
| **4** | `monthly_free_cash_flow` | 178 | Raw cash left after expenses (liquidity) is highly predictive. |
| **9** | `cash_flow_to_income_ratio` | 106 | **Engineered Ratio**—confirms its high value in risk assessment. |
| **11** | `loan_type_Personal` | 82 | Indicates higher inherent risk compared to other loan types. |
| **13** | `regional_unemployment_rate` | 79 | Highlights the importance of local economic context. |
| *... other debt/income ratios.* | | | |
