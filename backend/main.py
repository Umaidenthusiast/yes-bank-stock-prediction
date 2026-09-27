from __future__ import annotations

from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "linear_regression_v3.pkl"
SCALER_PATH = ROOT / "models" / "scaler_v3.pkl"
DATA_PATH = ROOT / "data" / "yes_bank_stock.csv"
FEATURES = ["Open", "High", "Low", "Previous_Close", "Return", "MA_3", "Volatility_3"]
MODEL_NAME = "Linear Regression V3"
METRICS = {"mae": 9.8376, "mse": 245.8189, "rmse": 15.6786, "r2": 0.9850}


def load_artifacts():
    try:
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required model artifact is missing: {exc.filename}") from exc


model, scaler = load_artifacts()

app = FastAPI(title="YES Bank Stock Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    open: FiniteNumber
    high: FiniteNumber
    low: FiniteNumber
    previous_close: FiniteNumber
    return_value: FiniteNumber
    ma_3: FiniteNumber
    volatility_3: FiniteNumber

    @model_validator(mode="after")
    def validate_market_range(self):
        for name in ("open", "high", "low", "previous_close"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name.replace('_', ' ')} must be greater than zero")
        if self.high < self.low:
            raise ValueError("high should not be lower than low")
        return self


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/model-info")
def model_info():
    return {
        "model": MODEL_NAME,
        "dataset": "YES Bank historical stock data",
        "dataset_period": "2005–2020",
        "training_samples": 145,
        "testing_samples": 37,
        "features": FEATURES,
        "metrics": METRICS,
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    values = [[
        payload.open, payload.high, payload.low, payload.previous_close,
        payload.return_value, payload.ma_3, payload.volatility_3,
    ]]
    try:
        input_frame = pd.DataFrame(values, columns=FEATURES)
        scaled_input = scaler.transform(input_frame)
        prediction = float(model.predict(scaled_input)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The model could not generate a prediction.") from exc
    return {"prediction": round(prediction, 2), "model": MODEL_NAME}


@app.get("/predictions")
def predictions():
    """Return the real chronological test-set predictions used in the final plot."""
    try:
        frame = pd.read_csv(DATA_PATH)
        frame["Date"] = pd.to_datetime(frame["Date"], format="%b-%y")
        frame = frame.sort_values("Date").reset_index(drop=True)
        frame["Previous_Close"] = frame["Close"].shift(1)
        frame["Return"] = frame["Close"].pct_change()
        frame["MA_3"] = frame["Close"].rolling(window=3).mean()
        frame["Volatility_3"] = frame["Return"].rolling(window=3).std()
        frame = frame.dropna().reset_index(drop=True)
        test_frame = frame.iloc[int(len(frame) * 0.80):].copy()
        predicted = model.predict(scaler.transform(test_frame[FEATURES]))
        return {"series": [
            {
                "date": row.Date.strftime("%b %Y"),
                "actual": round(float(row.Close), 2),
                "predicted": round(float(value), 2),
            }
            for row, value in zip(test_frame.itertuples(index=False), predicted)
        ]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Historical performance data is unavailable.") from exc
