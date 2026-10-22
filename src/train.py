from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from data_generator import save_dataset
from features import engineer_features, split_xy

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models'
REPORT_DIR = ROOT / 'reports'
DATA = ROOT / 'data' / 'transactions.csv'
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    'amount', 'hour', 'distance_km', 'account_age_days', 'tx_count_24h',
    'velocity_score', 'device_trust', 'merchant_risk', 'international',
    'new_device', 'weekend', 'failed_logins', 'chargeback_history',
    'log_amount', 'amount_per_account_day', 'velocity_per_tx',
    'night_transaction', 'high_velocity', 'high_distance',
    'risk_interaction', 'device_risk'
]


def build_pipeline(model):
    prep = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]), NUMERIC_FEATURES)
    ], remainder='drop')
    return Pipeline([('preprocess', prep), ('model', model)])


def evaluate(pipe, x_test, y_test, threshold=0.5):
    probability = pipe.predict_proba(x_test)[:, 1]
    prediction = (probability >= threshold).astype(int)
    return {
        'roc_auc': roc_auc_score(y_test, probability),
        'pr_auc': average_precision_score(y_test, probability),
        'threshold': threshold,
        'classification_report': classification_report(
            y_test, prediction, output_dict=True
        )
    }


def main():
    if not DATA.exists():
        save_dataset(DATA)
    df = engineer_features(pd.read_csv(DATA))
    x, y = split_xy(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    candidates = {
        'logistic_regression': LogisticRegression(
            max_iter=1200, class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=3,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1
        ),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            max_iter=220, max_leaf_nodes=31, learning_rate=0.08,
            l2_regularization=0.8, random_state=42
        )
    }
    results = {}
    best_name = None
    best_auc = -1
    best_pipe = None
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    for name, model in candidates.items():
        pipe = build_pipeline(model)
        scores = cross_validate(
            pipe, x_train, y_train, cv=cv,
            scoring=['roc_auc', 'average_precision'], n_jobs=-1
        )
        pipe.fit(x_train, y_train)
        metrics = evaluate(pipe, x_test, y_test)
        metrics['cv_roc_auc_mean'] = scores['test_roc_auc'].mean()
        metrics['cv_pr_auc_mean'] = scores['test_average_precision'].mean()
        results[name] = metrics
        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_name = name
            best_pipe = pipe

    joblib.dump(best_pipe, MODEL_DIR / 'fraud_model.joblib')
    with open(REPORT_DIR / 'metrics.json', 'w') as f:
        json.dump({'best_model': best_name, 'models': results}, f, indent=2)
    print(json.dumps({'best_model': best_name, 'roc_auc': best_auc}, indent=2))


if __name__ == '__main__':
    main()
