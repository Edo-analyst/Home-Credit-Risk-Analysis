# 🏦 Home Credit Default Risk Prediction

> Predicting the probability that a loan applicant will have repayment difficulties, using an end-to-end Machine Learning workflow on 307,511 real loan applications.

## 📌 Project Overview

A lender has to decide which applicants can repay a loan. Approving a future defaulter costs money; rejecting a good applicant costs a customer. This project builds a binary classifier that estimates $P(\text{TARGET}=1 \mid X)$, the probability of payment difficulties, so applicants can be ranked by risk.

| Phase | Status |
| --- | --- |
| Exploratory Data Analysis | ✅ Done |
| Preprocessing and Feature Engineering | ✅ Done |
| Model Development and Tuning (5 models) | ✅ Done |
| Final Evaluation and Threshold Selection | 🔜 Planned |
| Interpretability (SHAP) | 🔜 Planned |

## 📊 Dataset

[Home Credit Default Risk – Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk): `application_train.csv`, 307,511 applicants × 122 columns. The Kaggle test file has no target, so the training file is split **80/20** (stratified) into training and held-out test sets.

Only **8.1%** of applicants defaulted. A model that approves everyone would be 92% accurate, so accuracy is not used.

## 🔍 Approach

- **[EDA](notebooks/01_eda.ipynb):** missing values are mostly structural, not random. `DAYS_EMPLOYED` hides a placeholder value. External credit scores are by far the strongest predictors.
- **[Preprocessing](notebooks/02_preprocessing.ipynb):** credit-to-income and instalment-to-income ratios, plus flags for informative missing values. There are two scikit-learn pipelines, one for linear models and one for tree models. Both are refitted inside every cross-validation fold to avoid **data leakage**.
- **[Modeling](notebooks/03_modeling.ipynb):** five models tuned on the same 5-fold stratified cross-validation, with class weights for the imbalance. They are ranked by **PR-AUC**, which focuses on the minority class (a random model scores 0.081), with **ROC-AUC** for comparison with Kaggle.
- **Interpretability:** Logistic Regression odds ratios and Random Forest feature importance so far; SHAP is planned.

## 📈 Results

| Model | CV PR-AUC | CV ROC-AUC | Train–CV gap |
| --- | --- | --- | --- |
| **LightGBM** | **0.241 ± 0.005** | **0.757** | +0.046 |
| XGBoost | 0.240 ± 0.005 | 0.755 | +0.124 |
| Random Forest | 0.223 ± 0.004 | 0.743 | +0.036 |
| Logistic Regression | 0.220 ± 0.004 | 0.745 | +0.001 |
| Decision Tree | 0.195 ± 0.002 | 0.720 | +0.012 |

LightGBM ranks first, at about 3× a random model, and overfits much less than XGBoost. Logistic Regression is close behind and remains a strong, interpretable baseline.

🔜 The selected model will be tested once on the held-out set. Precision, recall and the confusion matrix will be reported at a threshold chosen on business costs.

## 📝 Future Work

- [ ] Test-set evaluation and threshold optimization
- [ ] Probability calibration
- [ ] SHAP explainability

## 📂 Repository Structure

```text
├── notebooks/
│   ├── 01_eda.ipynb                # ✅ exploratory analysis
│   ├── 02_preprocessing.ipynb      # ✅ cleaning, features, pipelines
│   ├── 03_modeling.ipynb           # ✅ model comparison
│   ├── 04_evaluation.ipynb         # 🔜 test set, threshold, calibration
│   └── 05_interpretability.ipynb   # 🔜 SHAP
├── data/  
    ├── raw                     # raw data from Kaggle
    ├── processed               # processed data (train/test) 
├── models/                     # saved pipelines & models
├── reports/figures/
└── requirements.txt
```

## 💻 Installation and Usage

```bash
git clone https://github.com/Edo-analyst/home-credit-default-risk.git
cd home-credit-default-risk
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download `application_train.csv` from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data) into `data/raw/`, then run the notebooks in order.

**Tech stack:** Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · SHAP · Matplotlib · Seaborn · Jupyter

---

**Author:** [Edoardo Matterazzo](https://github.com/Edo-analyst) · Data Science | Machine Learning | Credit Risk
