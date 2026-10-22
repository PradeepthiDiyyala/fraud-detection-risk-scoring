import numpy as np
import pandas as pd
from pathlib import Path


def generate_transactions(n=12000, seed=42):
    rng = np.random.default_rng(seed)
    amount = np.exp(rng.normal(4.2, 1.0, n)).clip(2, 5000)
    hour = rng.integers(0, 24, n)
    distance = rng.gamma(2.0, 18.0, n).clip(0, 300)
    account_age = rng.integers(1, 240, n)
    tx_count_24h = rng.poisson(3, n).clip(0, 30)
    velocity = rng.gamma(2, 2, n)
    device_trust = rng.beta(6, 2, n)
    merchant_risk = rng.beta(2, 8, n)
    international = rng.binomial(1, 0.18, n)
    new_device = rng.binomial(1, 0.16, n)
    weekend = rng.binomial(1, 0.29, n)
    failed_logins = rng.poisson(0.35, n).clip(0, 8)
    chargeback_history = rng.binomial(1, 0.06, n)
    log_amount = np.log1p(amount)
    amount_z = (log_amount - log_amount.mean()) / log_amount.std()
    night = ((hour < 6) | (hour > 22)).astype(int)
    score = (
        -4.2 + 0.65 * amount_z + 0.55 * merchant_risk
        + 0.75 * international + 0.9 * new_device + 0.18 * velocity
        + 0.16 * failed_logins + 0.8 * chargeback_history
        + 0.55 * night + 0.025 * distance + 0.18 * tx_count_24h
        - 1.1 * device_trust
    )
    probability = 1 / (1 + np.exp(-score))
    fraud = rng.binomial(1, np.clip(probability, 0.005, 0.95))
    return pd.DataFrame({
        'transaction_id': [f'TX{100000 + i}' for i in range(n)],
        'amount': amount.round(2), 'hour': hour,
        'distance_km': distance.round(2), 'account_age_days': account_age,
        'tx_count_24h': tx_count_24h, 'velocity_score': velocity.round(3),
        'device_trust': device_trust.round(3), 'merchant_risk': merchant_risk.round(3),
        'international': international, 'new_device': new_device,
        'weekend': weekend, 'failed_logins': failed_logins,
        'chargeback_history': chargeback_history, 'is_fraud': fraud
    })


def save_dataset(path='data/transactions.csv'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = generate_transactions()
    df.to_csv(path, index=False)
    return df


if __name__ == '__main__':
    df = save_dataset()
    print(f'Generated {len(df):,} transactions; fraud rate={df.is_fraud.mean():.2%}')
