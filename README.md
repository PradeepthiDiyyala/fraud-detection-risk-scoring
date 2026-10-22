# Fraud Detection & Risk Scoring Platform

End-to-end machine-learning application for detecting suspicious financial transactions and converting model probabilities into operational risk scores.

## Features
- Reproducible transaction data generation
- Data validation and preprocessing
- Transaction-level feature engineering
- Logistic Regression and Random Forest classification
- Class-imbalance aware training
- Cross-validation and threshold tuning
- Accuracy, precision, recall, F1, ROC-AUC and PR-AUC evaluation
- Persisted model and preprocessing pipeline
- Batch and single-transaction prediction
- FastAPI REST service
- Streamlit monitoring dashboard
- Automated tests

## Run
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/train.py
python scripts/evaluate.py
uvicorn api.main:app --reload
streamlit run dashboard/app.py
```

The repository uses synthetic transactions so the complete pipeline can be reproduced without private customer data.
