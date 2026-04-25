# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import joblib
import os

app = FastAPI(
    title="PM2.5 Forecasting API",
    description="Modular Early Warning API based on XGBoost/LSTM",
    version="1.0.0"
)

# Load global model variables
xgb_model = None
xgb_features = None

# Startup event to load models into memory
@app.on_event("startup")
def load_models():
    global xgb_model, xgb_features
    model_path = "src/models/saved/xgb_model.json"
    feature_path = "src/models/saved/xgb_features.pkl"
    
    if os.path.exists(model_path) and os.path.exists(feature_path):
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(model_path)
        xgb_features = joblib.load(feature_path)
        print("Models loaded successfully.")
    else:
        print("Warning: Models not found. Please run the training scripts first.")

# Define input schema based on standard meteorological data
class MeteorologicalData(BaseModel):
    dew: float
    temp: float
    press: float
    wnd_spd: float
    snow: float
    rain: float
    # Assuming one-hot encoded wind directions
    cbwd_NE: int = 0
    cbwd_NW: int = 0
    cbwd_SE: int = 0
    cbwd_cv: int = 0

@app.get("/")
def read_root():
    return {"message": "Welcome to the PM2.5 API. Visit /docs for interactive testing."}

@app.post("/predict/xgboost")
def predict_pm25_xgb(data: MeteorologicalData):
    if xgb_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Ensure columns match the training features exactly
    try:
        input_df = input_df[xgb_features]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")
        
    # Predict
    prediction = xgb_model.predict(input_df)
    
    # Ensure prediction is not negative
    pm25_value = max(0.0, float(prediction[0]))
    
    return {
        "model": "XGBoost",
        "predicted_pm25": pm25_value,
        "unit": "µg/m³"
    }