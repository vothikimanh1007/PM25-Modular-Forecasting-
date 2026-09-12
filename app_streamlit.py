import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

st.set_page_config(page_title="PM2.5 Early Warning Demo", layout="wide")
st.title("PM2.5 Hourly Forecasting System")
st.write("Modular Deep Learning (LSTM) & XGBoost Inference Gateway")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(BASE_DIR, "src", "models", "saved")

# Artifact paths
WEIGHTS_PKL_PATH = os.path.join(SAVED_DIR, "lstm_model_weights.pkl")
WEIGHTS_H5_PATH = os.path.join(SAVED_DIR, "lstm_model_weights.h5")
MODEL_H5_PATH = os.path.join(SAVED_DIR, "lstm_model.h5")
FEAT_SCALER_PATH = os.path.join(SAVED_DIR, "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(SAVED_DIR, "target_scaler.pkl")
XGB_FEATURES_PATH = os.path.join(SAVED_DIR, "xgb_features.pkl")

def build_lstm_architecture(timesteps=24, features=14):
    model = Sequential([
        Input(shape=(timesteps, features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    return model

@st.cache_resource
def load_models():
    scaler = joblib.load(FEAT_SCALER_PATH)
    
    if os.path.exists(XGB_FEATURES_PATH):
        xgb_features = joblib.load(XGB_FEATURES_PATH)
    else:
        xgb_features = []

    total_features = getattr(scaler, "n_features_in_", len(xgb_features) + 1)
    lstm = build_lstm_architecture(timesteps=24, features=total_features)

    # Version-safe weight injection
    if os.path.exists(WEIGHTS_PKL_PATH):
        weights = joblib.load(WEIGHTS_PKL_PATH)
        lstm.set_weights(weights)
    elif os.path.exists(WEIGHTS_H5_PATH):
        lstm.load_weights(WEIGHTS_H5_PATH)
    elif os.path.exists(MODEL_H5_PATH):
        lstm.load_weights(MODEL_H5_PATH)
    else:
        raise FileNotFoundError("No valid model weights found in src/models/saved/")

    return lstm, scaler, xgb_features

# Safe state initialization
lstm_model, scaler, xgb_features = None, None, []
try:
    lstm_model, scaler, xgb_features = load_models()
    st.sidebar.success("Artifacts loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")

st.sidebar.header("Current Meteorological Telemetry")
temp = st.sidebar.slider("Temperature (°C)", -20.0, 40.0, 15.0)
dew = st.sidebar.slider("Dew Point (°C)", -30.0, 30.0, -5.0)
press = st.sidebar.slider("Pressure (hPa)", 980.0, 1040.0, 1015.0)
wnd_spd = st.sidebar.slider("Wind Speed (m/s)", 0.0, 50.0, 4.2)
snow = st.sidebar.slider("Cumulated Snow (hours)", 0.0, 20.0, 0.0)
rain = st.sidebar.slider("Cumulated Rain (hours)", 0.0, 50.0, 0.0)
wind_dir = st.sidebar.selectbox("Wind Direction", ["NW", "SE", "NE", "cv"])

if st.button("Run PM2.5 Forecast"):
    if lstm_model is None or scaler is None:
        st.error("Inference halted: Model artifacts are not loaded.")
    else:
        input_mapping = {
            "dew": dew, "DEWP": dew,
            "temp": temp, "TEMP": temp,
            "press": press, "PRES": press,
            "wnd_spd": wnd_spd, "Iws": wnd_spd,
            "snow": snow, "Is": snow,
            "rain": rain, "Ir": rain,
            "cbwd_NE": 1 if wind_dir == "NE" else 0,
            "cbwd_NW": 1 if wind_dir == "NW" else 0,
            "cbwd_SE": 1 if wind_dir == "SE" else 0,
            "cbwd_cv": 1 if wind_dir == "cv" else 0,
            "month": 6
        }

        # Build feature vector matching training columns
        all_cols = ["pm2.5"] + [col for col in xgb_features if col != "pm2.5"]

        step_data = {}
        for col in all_cols:
            if col == "pm2.5":
                step_data[col] = 50.0  # Seed baseline autoregressive level
            else:
                step_data[col] = input_mapping.get(col, 0.0)

        sequence_df = pd.DataFrame([step_data for _ in range(24)])

        # Ensure exact column dimensionality required by StandardScaler
        if sequence_df.shape[1] < scaler.n_features_in_:
            missing_count = scaler.n_features_in_ - sequence_df.shape[1]
            for i in range(missing_count):
                sequence_df[f"extra_{i}"] = 0.0
        elif sequence_df.shape[1] > scaler.n_features_in_:
            sequence_df = sequence_df.iloc[:, :scaler.n_features_in_]

        # Normalize across channels
        scaled_seq = scaler.transform(sequence_df.values)
        model_input = np.expand_dims(scaled_seq, axis=0)

        # Predict
        scaled_pred = lstm_model.predict(model_input, verbose=0)

        # Invert to native physical concentration (ug/m3) using target channel parameters
        pred_val = float(scaled_pred[0][0] * np.sqrt(scaler.var_[0]) + scaler.mean_[0])
        pred_val = max(0.0, pred_val)

        col1, col2 = st.columns(2)
        col1.metric("Predicted PM2.5", f"{pred_val:.1f} µg/m³")

        if pred_val <= 35:
            col2.success("Air Quality Level: Good / Moderate")
        elif pred_val <= 75:
            col2.warning("Air Quality Level: Unhealthy for Sensitive Groups")
        else:
            col2.error("Air Quality Level: Hazardous")
