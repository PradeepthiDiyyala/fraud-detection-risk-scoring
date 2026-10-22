import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
import pandas as pd
from features import engineer_features


def test_engineering_adds_columns():
    df = pd.DataFrame([{
        'amount': 100, 'hour': 2, 'distance_km': 10,
        'account_age_days': 20, 'tx_count_24h': 2, 'velocity_score': 3,
        'device_trust': 0.8, 'merchant_risk': 0.1, 'international': 0,
        'new_device': 0, 'weekend': 0, 'failed_logins': 0,
        'chargeback_history': 0, 'is_fraud': 0
    }])
    out = engineer_features(df)
    assert 'log_amount' in out
    assert 'night_transaction' in out
    assert out.loc[0, 'night_transaction'] == 1
