from pathlib import Path
import joblib
import pandas as pd
from features import engineer_features

ROOT = Path(__file__).resolve().parents[1]
MODEL = joblib.load(ROOT / 'models' / 'fraud_model.joblib')


def score_transaction(payload):
    df = engineer_features(pd.DataFrame([payload]))
    features = df.drop(columns=['is_fraud', 'transaction_id'], errors='ignore')
    probability = float(MODEL.predict_proba(features)[:, 1][0])
    level = 'HIGH' if probability >= 0.70 else 'MEDIUM' if probability >= 0.35 else 'LOW'
    return {
        'fraud_probability': round(probability, 4),
        'risk_level': level
    }


if __name__ == '__main__':
    sample = {
        'amount': 850, 'hour': 2, 'distance_km': 140,
        'account_age_days': 12, 'tx_count_24h': 10,
        'velocity_score': 8.2, 'device_trust': 0.15,
        'merchant_risk': 0.55, 'international': 1,
        'new_device': 1, 'weekend': 0, 'failed_logins': 3,
        'chargeback_history': 0
    }
    print(score_transaction(sample))
