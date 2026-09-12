import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential

st.set_page_config(page_title="PM2.5 Early Warning Demo", layout="wide")
st.title("PM2.5 Hourly Forecasting System")
st.write("Modular Deep Learning (LSTM) & XGBoost Inference Gateway")

# Absolute path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(BASE_DIR, "src", "models", "saved")

WEIGHTS_PATH = os.path.join(SAVED_DIR, "lstm_model_weights.h5")
MODEL_H5_PATH = os.path.join(SAVED_DIR, "lstm_model.h5")
FEAT_SCALER_PATH = os.path.join(SAVED_DIR, "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(SAVED_DIR, "target_scaler.pkl")


def build_lstm_architecture(timesteps=24, features=15):
  """Rebuilds the exact architecture matching the verified paper training setup."""
  model = Sequential([
      Input(shape=(timesteps, features)),
      LSTM(64, return_sequences=True),
      Dropout(0.2),
      LSTM(32),
      Dropout(0.2),
      Dense(16, activation="relu"),
      Dense(1),
  ])
  return model


@st.cache_resource
def load_models():
  # Determine number of features from scaler if available
  feat_scaler = joblib.load(FEAT_SCALER_PATH)
  target_scaler = joblib.load(TARGET_SCALER_PATH)

  num_features = getattr(feat_scaler, "n_features_in_", 15)
  lstm = build_lstm_architecture(timesteps=24, features=num_features)

  # Load weights from either lstm_model_weights.h5 or the saved lstm_model.h5
  target_weights = (
      WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else MODEL_H5_PATH
  )
  if not os.path.exists(target_weights):
    raise FileNotFoundError(f"Weights file not found at: {target_weights}")

  lstm.load_weights(target_weights)
  return lstm, feat_scaler, target_scaler


try:
  lstm_model, feat_scaler, target_scaler = load_models()
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

# Generate synthetic 24-step sequence ending with slider inputs
if st.button("Run PM2.5 Forecast"):
  # Default prior PM2.5 seed to feed autoregressive lookback
  baseline_pm25 = 50.0

  base_step = {
      "pm2.5": baseline_pm25,
      "DEWP": dew,
      "TEMP": temp,
      "PRES": press,
      "Iws": wnd_spd,
      "Is": snow,
      "Ir": rain,
      "cbwd_NE": 1 if wind_dir == "NE" else 0,
      "cbwd_NW": 1 if wind_dir == "NW" else 0,
      "cbwd_SE": 1 if wind_dir == "SE" else 0,
      "cbwd_cv": 1 if wind_dir == "cv" else 0,
  }

  sequence_df = pd.DataFrame([base_step for _ in range(24)])

  # Align columns to feature_scaler expectation
  if hasattr(feat_scaler, "feature_names_in_"):
    for col in feat_scaler.feature_names_in_:
      if col not in sequence_df.columns:
        sequence_df[col] = 0
    sequence_df = sequence_df[feat_scaler.feature_names_in_]

  scaled_seq = feat_scaler.transform(sequence_df)
  inp = np.expand_dims(scaled_seq, axis=0)

  scaled_pred = lstm_model.predict(inp, verbose=0)

  # Inverse transform target using target scaler parameters
  if hasattr(target_scaler, "var_"):
    pred_val = float(
        scaled_pred[0][0] * np.sqrt(target_scaler.var_[0])
        + target_scaler.mean_[0]
    )
  else:
    pred_val = float(target_scaler.inverse_transform(scaled_pred)[0][0])

  pred_val = max(0.0, pred_val)

  col1, col2 = st.columns(2)
  col1.metric("Predicted PM2.5", f"{pred_val:.1f} µg/m³")

  if pred_val <= 35:
    col2.success("Air Quality Level: Good / Moderate")
  elif pred_val <= 75:
    col2.warning("Air Quality Level: Unhealthy for Sensitive Groups")
  else:
    col2.error("Air Quality Level: Hazardous")
