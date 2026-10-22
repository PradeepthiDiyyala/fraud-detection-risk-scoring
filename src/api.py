from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predict import score_transaction

app = FastAPI(title='Fraud Risk Scoring API', version='1.0.0')


class Transaction(BaseModel):
    amount: float = Field(gt=0)
    hour: int = Field(ge=0, le=23)
    distance_km: float = Field(ge=0)
    account_age_days: int = Field(ge=0)
    tx_count_24h: int = Field(ge=0)
    velocity_score: float = Field(ge=0)
    device_trust: float = Field(ge=0, le=1)
    merchant_risk: float = Field(ge=0, le=1)
    international: int = Field(ge=0, le=1)
    new_device: int = Field(ge=0, le=1)
    weekend: int = Field(ge=0, le=1)
    failed_logins: int = Field(ge=0)
    chargeback_history: int = Field(ge=0, le=1)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(tx: Transaction):
    try:
        return score_transaction(tx.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
