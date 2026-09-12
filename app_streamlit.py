import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

st.set_page_config(page_title="PM2.5 Early Warning Demo", layout="wide")
st.title("PM2.5 Hourly Forecasting System")
st.write("Modular Deep Learning (LSTM) & XGBoost Inference Gateway")

# Load artifacts
@st.cache_resource
def load_models():
    lstm = tf.keras.models.load_model("src/models/saved/lstm_model.h5")
    feat_scaler = joblib.load("src/models/saved/feature_scaler.pkl")
    target_scaler = joblib.load("src/models/saved/target_scaler.pkl")
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
    base_step = {
        "dew": dew, "temp": temp, "press": press, "wnd_spd": wnd_spd,
        "snow": snow, "rain": rain,
        "cbwd_NE": 1 if wind_dir == "NE" else 0,
        "cbwd_NW": 1 if wind_dir == "NW" else 0,
        "cbwd_SE": 1 if wind_dir == "SE" else 0,
        "cbwd_cv": 1 if wind_dir == "cv" else 0
    }
    sequence_df = pd.DataFrame([base_step for _ in range(24)])
    
    # Scale and predict
    scaled_seq = feat_scaler.transform(sequence_df)
    inp = np.expand_dims(scaled_seq, axis=0)
    scaled_pred = lstm_model.predict(inp, verbose=0)
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
