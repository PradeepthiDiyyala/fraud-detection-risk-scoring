import numpy as np
import pandas as pd

TARGET = "is_fraud"
ID_COL = "transaction_id"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amount"] = np.log1p(out["amount"])
    out["amount_per_account_day"] = out["amount"] / (out["account_age_days"] + 1)
    out["velocity_per_tx"] = out["velocity_score"] / (out["tx_count_24h"] + 1)
    out["night_transaction"] = ((out["hour"] < 6) | (out["hour"] > 22)).astype(int)
    out["high_velocity"] = (out["tx_count_24h"] >= 7).astype(int)
    out["high_distance"] = (out["distance_km"] >= 80).astype(int)
    out["risk_interaction"] = out["merchant_risk"] * (1 + out["international"])
    out["device_risk"] = 1 - out["device_trust"]
    return out


def split_xy(df):
    x = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    y = df[TARGET].astype(int)
    return x, y
