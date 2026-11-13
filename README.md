# 🏦 Home Credit Default Risk Prediction

> Predicting the probability that a loan applicant will have repayment difficulties, using an end-to-end Machine Learning workflow on 307,511 real loan applications.

## 📌 Project Overview

A lender has to decide which applicants can repay a loan. Approving a future defaulter costs money; rejecting a good applicant costs a customer. This project builds a binary classifier that estimates $P(\text{TARGET}=1 \mid X)$, the probability of payment difficulties, so applicants can be ranked by risk.

| Phase | Status |
| --- | --- |
| Exploratory Data Analysis | ✅ Done |
| Preprocessing and Feature Engineering | ✅ Done |
| Model Development and Tuning (5 models) | ✅ Done |
| Final Evaluation on the Test Set | ✅ Done |
| Threshold Selection | 🔜 Planned |
| Interpretability (SHAP) | 🔜 Planned |

## 📊 Dataset

[Home Credit Default Risk – Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk): `application_train.csv`, 307,511 applicants × 122 columns. The Kaggle test file has no target, so the training file is split **70/15/15** (stratified) into training, validation, and test sets.

Only **8.1%** of applicants defaulted. A model that approves everyone would be 92% accurate, so accuracy is not used.

## 🔍 Approach

- **[EDA](notebooks/01_eda.ipynb):** missing values are mostly structural, not random. `DAYS_EMPLOYED` hides a placeholder value. External credit scores are by far the strongest predictors.
- **[Preprocessing](notebooks/02_preprocessing.ipynb):** credit-to-income and instalment-to-income ratios, plus flags for informative missing values. There are two scikit-learn pipelines, one for linear models and one for tree models. Both are refitted inside every cross-validation fold to avoid **data leakage**.
- **[Modeling](notebooks/03_modeling.ipynb):** five models tuned on the same 5-fold stratified cross-validation inside the training set, with class weights for the imbalance. They are then compared on the held-back validation set by **AP** (Average Precision), which focuses on the minority class (a random model scores 0.081), with **ROC-AUC** for comparison with Kaggle.
- **Interpretability:** Logistic Regression odds ratios and Random Forest feature importance so far; SHAP is planned.

## 📈 Results

Hyperparameters are tuned by 5-fold cross-validation **inside the training set**. The validation set is used to compare the five tuned models.

| Model | CV AP | Val AP | Val PR-AUC | CV ROC-AUC | Val ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| **LightGBM** | **0.240 ± 0.007** | **0.239** | **0.239** | **0.757 ± 0.006** | **0.753** |
| XGBoost | 0.239 ± 0.007 | 0.236 | 0.236 | 0.754 ± 0.006 | 0.752 |
| Random Forest | 0.223 ± 0.008 | 0.221 | 0.221 | 0.744 ± 0.007 | 0.740 |
| Logistic Regression | 0.221 ± 0.010 | 0.217 | 0.216 | 0.746 ± 0.007 | 0.742 |
| Decision Tree | 0.193 ± 0.007 | 0.192 | 0.202 | 0.718 ± 0.007 | 0.714 |

*± is the standard deviation across the five cross-validation folds.*

**LightGBM is the model carried to the test set.** It ranks first on every ranking metric — CV AP 0.240, validation AP 0.239, PR-AUC 0.239, ROC-AUC 0.753, roughly 3× the 0.081 no-skill AP — and it ranks first in cross-validation and again on validation, so the ordering is not an artefact of the fold means the search optimised. No model loses more than 0.005 AP moving from folds to validation, so the whole comparison is stable rather than lucky. Its lead over XGBoost is narrow, at 0.003 AP and 0.001 ROC-AUC.

Three practical points favour it beyond the score. It tunes in 5.2 minutes against XGBoost's 8.2 for the same 10 candidates. Random Forest costs 18 minutes and a 12.3 MB artifact to land 0.017 AP lower — the worst trade in the table. And LightGBM's winning configuration sits on two edges of its grid (`min_child_samples` at the maximum 800, `colsample_bytree` at the minimum 0.3) with only 10 of 81 combinations sampled, so the search is still asking for more regularisation than it was offered and 0.240 should be read as a floor.

**Logistic Regression holds up.** It is 0.005 AP behind Random Forest (0.217 against 0.221) but ahead on ROC-AUC (0.742 against 0.740), and it scores only 0.002 higher on the data it was fitted to than on held-out folds, against roughly 0.05 for the boosters. The ~0.02 AP the boosters add is what interpretability costs here. The elastic net zeroed none of its 97 features at the winning `C = 1`, `l1_ratio = 0.5` — the penalty regularised without selecting.

AP decides the ranking rather than the trapezoidal area under the PR curve. The two agree to 0.0003 wherever a model scores each applicant distinctly, but the Decision Tree emits only 157 distinct scores — one per leaf — and interpolating across so few points lifts its PR-AUC to 0.202 against an AP of 0.192.

### Held-out test set

The test set was opened once to check LightGBM performance on unseen data.

| | AP | PR-AUC | ROC-AUC |
| --- | --- | --- | --- |
| **Test** | **0.238** | **0.237** | **0.753** |

At the default 0.50 cutoff the model identifies 2,418 of 3,724 defaults — 64.9% recall — and predicts default for 14,289 applicants, of whom 16.9% go on to default, about twice the 8.1% base rate. **That cutoff is a placeholder, not a decision.** It has not been weighed against the relative cost of a missed default and a rejected good applicant, which is what the threshold work below will do.

## 📝 Future Work

- [ ] Threshold selection against business costs
- [ ] Probability calibration
- [ ] SHAP explainability

## 📂 Repository Structure

```text
├── notebooks/
│   ├── 01_eda.ipynb                # ✅ exploratory analysis
│   ├── 02_preprocessing.ipynb      # ✅ cleaning, features, pipelines
│   ├── 03_modeling.ipynb           # ✅ modeling, comparison
│   └── 04_interpretability.ipynb   # 🔜 SHAP
├── data/  
    ├── raw                     # raw data from Kaggle
    └── processed               # processed data (train/val/test)
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
