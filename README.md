# Home Credit Default Risk 

End-to-end machine learning project that estimates the probability a loan applicant will default, built on 307,511 real loan applications from [Home Credit Group](https://www.kaggle.com/competitions/home-credit-default-risk).

## Business Problem
Home Credit wants to predict which loan applicants are likely to default, enabling better lending decisions and reduced financial risk.

## Tech Stack
- Python
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn (Pipeline, ColumnTransformer, LogisticRegression, DecisionTreeClassifier)
- XGBoost, LightGBM
- SHAP for model explainability

## Repository structure

```
├── data/
│   ├── raw/                      
│   └── processed/ 
├── notebooks/
│   ├── 01_eda.ipynb             
│   ├── 02_preprocessing.ipynb   
│   └── 03_modeling.ipynb                     
├── models/                      
├── reports/figures/             
└── requirements.txt
```


