# src/api/main.py
import os
import time
from typing import List
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import tensorflow as tf
import xgboost as xgb

app = FastAPI(
    title="PM2.5 Modular Early Warning API",
    description="Modular inference service serving both XGBoost and Deep Recurrent (LSTM) models.",
    version="1.0.0",
)

# Global model and preprocessor artifacts
xgb_model = None
xgb_features = None
lstm_model = None
feature_scaler = None
target_scaler = None

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models/saved")


@app.on_event("startup")
def load_artifacts():
  global xgb_model, xgb_features, lstm_model, feature_scaler, target_scaler

  xgb_model_path = os.path.join(MODELS_DIR, "xgb_model.json")
  xgb_feat_path = os.path.join(MODELS_DIR, "xgb_features.pkl")
  lstm_model_path = os.path.join(MODELS_DIR, "lstm_model.h5")
  feat_scaler_path = os.path.join(MODELS_DIR, "feature_scaler.pkl")
  target_scaler_path = os.path.join(MODELS_DIR, "target_scaler.pkl")

  # Load XGBoost artifacts
  if os.path.exists(xgb_model_path) and os.path.exists(xgb_feat_path):
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_model_path)
    xgb_features = joblib.load(xgb_feat_path)
    print("XGBoost model loaded successfully.")

  # Load LSTM and Scaler artifacts
  if os.path.exists(lstm_model_path):
    lstm_model = tf.keras.models.load_model(lstm_model_path)
    print("LSTM network loaded successfully.")

  if os.path.exists(feat_scaler_path) and os.path.exists(target_scaler_path):
    feature_scaler = joblib.load(feat_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    print("Feature and target scalers loaded successfully.")


# Single-step input schema (for XGBoost baseline)
class TimestepData(BaseModel):
  dew: float
  temp: float
  press: float
  wnd_spd: float
  snow: float
  rain: float
  cbwd_NE: int = 0
  cbwd_NW: int = 0
  cbwd_SE: int = 0
  cbwd_cv: int = 0


# Sequence schema for 24-hour look-back window (for LSTM deep network)
class SequenceData(BaseModel):
  timesteps: List[TimestepData] = Field(
      ...,
      description="24 consecutive hourly readings ordered chronologically.",
  )


@app.get("/")
def health_check():
  return {
      "status": "online",
      "xgb_loaded": xgb_model is not None,
      "lstm_loaded": lstm_model is not None,
      "docs_url": "/docs",
  }


@app.post("/predict/xgboost")
def predict_xgboost(data: TimestepData):
  if xgb_model is None:
    raise HTTPException(status_code=503, detail="XGBoost model not loaded.")

  start_time = time.perf_counter()
  input_df = pd.DataFrame([data.dict()])

  try:
    if xgb_features:
      input_df = input_df[xgb_features]
  except KeyError as err:
    raise HTTPException(status_code=400, detail=f"Missing feature: {err}")

  prediction = xgb_model.predict(input_df)
  pm25_val = max(0.0, float(prediction[0]))
  latency_ms = (time.perf_counter() - start_time) * 1000

  return {
      "model": "XGBoost",
      "predicted_pm25": round(pm25_val, 2),
      "unit": "µg/m³",
      "inference_latency_ms": round(latency_ms, 2),
  }


@app.post("/predict/lstm")
def predict_lstm(payload: SequenceData):
  if lstm_model is None:
    raise HTTPException(status_code=503, detail="LSTM model not loaded.")

  if len(payload.timesteps) != 24:
    raise HTTPException(
        status_code=422,
        detail=(
            "Look-back mismatch: Exactly 24 sequential hourly timesteps are"
            f" required, got {len(payload.timesteps)}."
        ),
    )

  start_time = time.perf_counter()

  # Convert 24-step payload to array
  raw_records = [step.dict() for step in payload.timesteps]
  df = pd.DataFrame(raw_records)

  # Scale features if pre-trained scaler is present
  if feature_scaler is not None:
    scaled_features = feature_scaler.transform(df)
  else:
    scaled_features = df.values

  # Reshape to (batch_size=1, timesteps=24, features=num_features)
  model_input = np.expand_dims(scaled_features, axis=0)

  # Run inference
  scaled_prediction = lstm_model.predict(model_input, verbose=0)

  # Inverse scale target
  if target_scaler is not None:
    pm25_pred = target_scaler.inverse_transform(scaled_prediction)[0][0]
  else:
    pm25_pred = scaled_prediction[0][0]

  pm25_val = max(0.0, float(pm25_pred))
  latency_ms = (time.perf_counter() - start_time) * 1000

  return {
      "model": "LSTM-2Layer",
      "predicted_pm25": round(pm25_val, 2),
      "unit": "µg/m³",
      "lookback_window": "24h",
      "inference_latency_ms": round(latency_ms, 2),
  }
