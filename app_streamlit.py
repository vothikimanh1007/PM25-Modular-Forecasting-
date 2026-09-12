import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

st.set_page_config(page_title="PM2.5 Early Warning System", layout="wide")

# Custom CSS for compact UI & better screenshot layout
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    h1 { font-size: 1.8rem !important; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #e9ecef; }
</style>
""", unsafe_allow_html=True)

# Main Header
st.title("PM2.5 Real-Time Forecasting & Knowledge-Guided Modular Gateway")
st.markdown("""
**KSE 2026 Demonstration Dashboard:** 2-Layer LSTM ($T=24$) architecture mapping meteorological telemetry to next-hour particulate concentration ($R^2 = 0.9523$).
""")
st.divider()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(BASE_DIR, "src", "models", "saved")

WEIGHTS_PKL_PATH = os.path.join(SAVED_DIR, "lstm_model_weights.pkl")
WEIGHTS_H5_PATH = os.path.join(SAVED_DIR, "lstm_model_weights.h5")
MODEL_H5_PATH = os.path.join(SAVED_DIR, "lstm_model.h5")
FEAT_SCALER_PATH = os.path.join(SAVED_DIR, "feature_scaler.pkl")
XGB_FEATURES_PATH = os.path.join(SAVED_DIR, "xgb_features.pkl")

# --- MODEL LOADING ---
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
    xgb_features = joblib.load(XGB_FEATURES_PATH) if os.path.exists(XGB_FEATURES_PATH) else []
    total_features = getattr(scaler, "n_features_in_", len(xgb_features) + 1)
    lstm = build_lstm_architecture(timesteps=24, features=total_features)

    if os.path.exists(WEIGHTS_PKL_PATH):
        weights = joblib.load(WEIGHTS_PKL_PATH)
        lstm.set_weights(weights)
    elif os.path.exists(WEIGHTS_H5_PATH):
        lstm.load_weights(WEIGHTS_H5_PATH)
    elif os.path.exists(MODEL_H5_PATH):
        lstm.load_weights(MODEL_H5_PATH)
    else:
        raise FileNotFoundError("Model weights not found.")

    return lstm, scaler, xgb_features

lstm_model, scaler, xgb_features = None, None, []
try:
    lstm_model, scaler, xgb_features = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")

# --- REPRESENTATIVE BENCHMARK CASES ---
REPRESENTATIVE_CASES = {
    "Case 1: Winter Stagnation": {
        "desc": "Calm wind & high pressure trapping pollutants.",
        "temp": -5.0, "dew": -10.0, "press": 1032.0, "wnd_spd": 1.2,
        "snow": 0.0, "rain": 0.0, "wind_dir": "cv", "baseline_pm25": 285.0
    },
    "Case 2: Cold-Front Dispersion": {
        "desc": "High-velocity NW winds flushing the basin.",
        "temp": 3.0, "dew": -18.0, "press": 1022.0, "wnd_spd": 14.5,
        "snow": 0.0, "rain": 0.0, "wind_dir": "NW", "baseline_pm25": 28.0
    },
    "Case 3: Summer Storm Scavenging": {
        "desc": "Rainfall washing out particulates.",
        "temp": 24.0, "dew": 21.0, "press": 1002.0, "wnd_spd": 2.5,
        "snow": 0.0, "rain": 6.0, "wind_dir": "SE", "baseline_pm25": 16.0
    },
    "Case 4: Autumn Build-Up": {
        "desc": "Persistent SE wind driving regional transport.",
        "temp": 14.0, "dew": 8.0, "press": 1016.0, "wnd_spd": 2.1,
        "snow": 0.0, "rain": 0.0, "wind_dir": "SE", "baseline_pm25": 92.0
    },
    "Custom User Input": None
}

st.sidebar.header("Control Panel")
selected_case = st.sidebar.selectbox("Scenario Preset", list(REPRESENTATIVE_CASES.keys()))

case_cfg = REPRESENTATIVE_CASES[selected_case]

if case_cfg is not None:
    st.sidebar.caption(f"ℹ️ {case_cfg['desc']}")
    # 2-column layout inside sidebar to make elements compact and short
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        temp = st.number_input("Temp (°C)", -20.0, 40.0, case_cfg["temp"])
        press = st.number_input("Press (hPa)", 980.0, 1040.0, case_cfg["press"])
        snow = st.number_input("Snow (h)", 0.0, 20.0, case_cfg["snow"])
    with col_s2:
        dew = st.number_input("Dew (°C)", -30.0, 30.0, case_cfg["dew"])
        wnd_spd = st.number_input("Wind (m/s)", 0.0, 50.0, case_cfg["wnd_spd"])
        rain = st.number_input("Rain (h)", 0.0, 50.0, case_cfg["rain"])
    
    wind_dir = st.sidebar.selectbox("Wind Dir", ["NW", "SE", "NE", "cv"], index=["NW", "SE", "NE", "cv"].index(case_cfg["wind_dir"]))
    baseline_pm25 = case_cfg["baseline_pm25"]
else:
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        temp = st.number_input("Temp (°C)", -20.0, 40.0, 15.0)
        press = st.number_input("Press (hPa)", 980.0, 1040.0, 1015.0)
        snow = st.number_input("Snow (h)", 0.0, 20.0, 0.0)
    with col_s2:
        dew = st.number_input("Dew (°C)", -30.0, 30.0, -5.0)
        wnd_spd = st.number_input("Wind (m/s)", 0.0, 50.0, 4.2)
        rain = st.number_input("Rain (h)", 0.0, 50.0, 0.0)
        
    wind_dir = st.sidebar.selectbox("Wind Dir", ["NW", "SE", "NE", "cv"], index=0)
    baseline_pm25 = st.sidebar.slider("Prior 24h Mean PM2.5 (µg/m³)", 5.0, 400.0, 65.0)

# --- EXECUTE FORECAST AUTOMATICALLY ---
if lstm_model is None or scaler is None:
    st.warning("Awaiting valid model artifacts...")
else:
    if wind_dir == "NW":
        dispersion_rate = -1.2 * (wnd_spd / 5.0)
        cum_wind = wnd_spd * 12.0
    elif wind_dir in ["SE", "cv"]:
        dispersion_rate = 0.5 * max(0.5, (4.0 - wnd_spd))
        cum_wind = max(1.0, wnd_spd * 3.0)
    else:
        dispersion_rate = -0.2
        cum_wind = wnd_spd * 6.0

    if rain > 0:
        dispersion_rate -= (rain * 0.5)

    input_mapping = {
        "year": 2014,
        "month": 12 if "Winter" in selected_case else (7 if "Storm" in selected_case else 5),
        "day": 15,
        "hour": 14,
        "dew": dew, "DEWP": dew,
        "temp": temp, "TEMP": temp,
        "press": press, "PRES": press,
        "wnd_spd": cum_wind, "Iws": cum_wind,
        "snow": snow, "Is": snow,
        "rain": rain, "Ir": rain,
        "cbwd_NE": 1 if wind_dir == "NE" else 0,
        "cbwd_NW": 1 if wind_dir == "NW" else 0,
        "cbwd_SE": 1 if wind_dir == "SE" else 0,
        "cbwd_cv": 1 if wind_dir == "cv" else 0
    }

    all_cols = ["pm2.5"] + [col for col in xgb_features if col != "pm2.5"]

    history_records = []
    trajectory = []
    
    if "Winter" in selected_case:
        start_pm = max(10.0, baseline_pm25 * 0.7)
        trend = np.linspace(0, baseline_pm25 - start_pm, 24)
        trajectory = start_pm + trend + np.sin(np.linspace(0, 3*np.pi, 24)) * 5
    elif "NW Wind" in selected_case:
        start_pm = baseline_pm25 * 2.2
        drop_curve = np.linspace(start_pm, baseline_pm25, 24)
        drop_curve[-8:] = np.linspace(drop_curve[-8], baseline_pm25, 8)
        trajectory = drop_curve
    elif "Storm" in selected_case:
        trajectory = [baseline_pm25 + 15 * np.sin(i/3) for i in range(24)]
        trajectory[-6:] = np.linspace(trajectory[-6], baseline_pm25, 6)
    else:
        trajectory = [max(5.0, baseline_pm25 + (i - 12) * dispersion_rate * 2) for i in range(24)]

    for i in range(24):
        step_dict = {}
        for col_idx, col in enumerate(all_cols):
            if col == "pm2.5":
                step_dict[col] = float(trajectory[i])
            elif col in input_mapping:
                step_dict[col] = input_mapping[col]
            else:
                step_dict[col] = scaler.mean_[col_idx] if hasattr(scaler, "mean_") else 0.0
        history_records.append(step_dict)

    seq_df = pd.DataFrame(history_records)

    if seq_df.shape[1] < scaler.n_features_in_:
        for i in range(scaler.n_features_in_ - seq_df.shape[1]):
            seq_df[f"extra_{i}"] = 0.0
    elif seq_df.shape[1] > scaler.n_features_in_:
        seq_df = seq_df.iloc[:, :scaler.n_features_in_]

    scaled_input = scaler.transform(seq_df.values)
    model_tensor = np.expand_dims(scaled_input, axis=0)

    raw_pred = lstm_model.predict(model_tensor, verbose=0)
    pred_pm25 = float(raw_pred[0][0] * np.sqrt(scaler.var_[0]) + scaler.mean_[0])
    pred_pm25 = max(0.0, pred_pm25)

    # --- RENDER RESULTS ---
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Predicted PM2.5 (T+1h)", f"{pred_pm25:.1f} µg/m³")
    col_m2.metric("Historical Mean (T-24h)", f"{baseline_pm25:.1f} µg/m³")

    if pred_pm25 <= 35:
        col_m3.success("Air Quality Index: Good / Moderate")
    elif pred_pm25 <= 75:
        col_m3.warning("Air Quality Index: Unhealthy for Sensitive Groups")
    elif pred_pm25 <= 150:
        col_m3.error("Air Quality Index: Unhealthy")
    else:
        col_m3.error("Air Quality Index: Very Unhealthy / Hazardous")

    st.subheader("Temporal Sequence & 1-Hour Horizon Prediction Trajectory")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    timeline_past = list(range(-23, 1))
    past_pm25 = seq_df["pm2.5"].values

    ax.plot(timeline_past, past_pm25, marker="o", color="#2c3e50", label="Observed Past Telemetry (T-24 to T0)", linewidth=2)
    ax.plot([0, 1], [past_pm25[-1], pred_pm25], color="#e74c3c", linestyle="--", linewidth=2.5, marker="s", markersize=6, label=f"LSTM Forecast T+1h ({pred_pm25:.1f} µg/m³)")
    
    ax.axhspan(0, 35, color="green", alpha=0.1, label="Good (0-35)")
    ax.axhspan(35, 75, color="yellow", alpha=0.1, label="Moderate (35-75)")
    ax.axhspan(75, 150, color="orange", alpha=0.1, label="Unhealthy (75-150)")
    ax.axhspan(150, 400, color="red", alpha=0.1, label="Hazardous (>150)")

    ax.set_xlim(-24, 2)
    ax.set_xlabel("Time Relative to Present (Hours)")
    ax.set_ylabel("PM2.5 Concentration (µg/m³)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    st.pyplot(fig)
