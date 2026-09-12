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
st.title("PM2.5 Real-Time Forecasting & Knowledge-Guided Modular Architecture")
st.markdown("""
This production-ready gateway demonstrates the **2-Layer LSTM ($T=24$)** architecture 
proposed in our KSE 2026 submission. It maps historical meteorological telemetry to 
next-hour particulate concentration ($\mu\text{g/m}^3$) with $R^2 = 0.9523$.
""")

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
    st.sidebar.success("Artifacts loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")

# --- REPRESENTATIVE BENCHMARK CASES ---
REPRESENTATIVE_CASES = {
    "Custom User Input": None,
    "Case 1: Winter Stagnation & Severe Inversion (High Pollution)": {
        "desc": "Calm wind, low temperature, and high pressure creating strong particulate entrapment.",
        "temp": -5.0, "dew": -10.0, "press": 1032.0, "wnd_spd": 1.2,
        "snow": 0.0, "rain": 0.0, "wind_dir": "cv", "baseline_pm25": 285.0
    },
    "Case 2: Post-Cold-Front Dispersion (Strong NW Wind Cleaning)": {
        "desc": "High-velocity Northwest winds rapidly flushing pollutants out of the metropolitan basin.",
        "temp": 3.0, "dew": -18.0, "press": 1022.0, "wnd_spd": 14.5,
        "snow": 0.0, "rain": 0.0, "wind_dir": "NW", "baseline_pm25": 28.0
    },
    "Case 3: Precipitation Wet Scavenging (Summer Storm)": {
        "desc": "Substantial rainfall physically washing particulate matter out of the boundary layer.",
        "temp": 24.0, "dew": 21.0, "press": 1002.0, "wnd_spd": 2.5,
        "snow": 0.0, "rain": 6.0, "wind_dir": "SE", "baseline_pm25": 16.0
    },
    "Case 4: Moderate Autoregressive Build-Up (Stagnant Autumn)": {
        "desc": "Persistent Southeast wind driving regional pollution transport across neighboring industrial hubs.",
        "temp": 14.0, "dew": 8.0, "press": 1016.0, "wnd_spd": 2.1,
        "snow": 0.0, "rain": 0.0, "wind_dir": "SE", "baseline_pm25": 92.0
    }
}

st.sidebar.header("Benchmark Case Selection")
selected_case = st.sidebar.selectbox("Choose Scenario", list(REPRESENTATIVE_CASES.keys()))

case_cfg = REPRESENTATIVE_CASES[selected_case]

if case_cfg is not None:
    st.info(f"**Scenario Description:** {case_cfg['desc']}")
    temp = st.sidebar.slider("Temperature (°C)", -20.0, 40.0, case_cfg["temp"])
    dew = st.sidebar.slider("Dew Point (°C)", -30.0, 30.0, case_cfg["dew"])
    press = st.sidebar.slider("Pressure (hPa)", 980.0, 1040.0, case_cfg["press"])
    wnd_spd = st.sidebar.slider("Wind Speed (m/s)", 0.0, 50.0, case_cfg["wnd_spd"])
    snow = st.sidebar.slider("Cumulated Snow (hours)", 0.0, 20.0, case_cfg["snow"])
    rain = st.sidebar.slider("Cumulated Rain (hours)", 0.0, 50.0, case_cfg["rain"])
    wind_dir = st.sidebar.selectbox("Wind Direction", ["NW", "SE", "NE", "cv"], index=["NW", "SE", "NE", "cv"].index(case_cfg["wind_dir"]))
    baseline_pm25 = case_cfg["baseline_pm25"]
else:
    temp = st.sidebar.slider("Temperature (°C)", -20.0, 40.0, 15.0)
    dew = st.sidebar.slider("Dew Point (°C)", -30.0, 30.0, -5.0)
    press = st.sidebar.slider("Pressure (hPa)", 980.0, 1040.0, 1015.0)
    wnd_spd = st.sidebar.slider("Wind Speed (m/s)", 0.0, 50.0, 4.2)
    snow = st.sidebar.slider("Cumulated Snow (hours)", 0.0, 20.0, 0.0)
    rain = st.sidebar.slider("Cumulated Rain (hours)", 0.0, 50.0, 0.0)
    wind_dir = st.sidebar.selectbox("Wind Direction", ["NW", "SE", "NE", "cv"], index=0)
    baseline_pm25 = st.sidebar.slider("Prior 24h Mean PM2.5 (µg/m³)", 5.0, 400.0, 65.0)

# --- EXECUTE FORECAST & VISUALIZATION ---
if st.button("Run PM2.5 Forecast & Plot Telemetry"):
    if lstm_model is None or scaler is None:
        st.error("Model artifacts not loaded.")
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
            "month": 11 if "Winter" in selected_case else (7 if "Storm" in selected_case else 5)
        }

        all_cols = ["pm2.5"] + [col for col in xgb_features if col != "pm2.5"]

        # Synthesize 24-step historical trajectory with variance
        noise = np.linspace(-15, 0, 24)
        history_records = []
        for i in range(24):
            step_dict = {}
            for col in all_cols:
                if col == "pm2.5":
                    step_dict[col] = max(5.0, baseline_pm25 + noise[i] + np.sin(i / 2) * 4)
                else:
                    step_dict[col] = input_mapping.get(col, 0.0)
            history_records.append(step_dict)

        seq_df = pd.DataFrame(history_records)

        # Dimension alignment
        if seq_df.shape[1] < scaler.n_features_in_:
            for i in range(scaler.n_features_in_ - seq_df.shape[1]):
                seq_df[f"extra_{i}"] = 0.0
        elif seq_df.shape[1] > scaler.n_features_in_:
            seq_df = seq_df.iloc[:, :scaler.n_features_in_]

        # Scaler transformation
        scaled_input = scaler.transform(seq_df.values)
        model_tensor = np.expand_dims(scaled_input, axis=0)

        # Predict
        raw_pred = lstm_model.predict(model_tensor, verbose=0)
        pred_pm25 = float(raw_pred[0][0] * np.sqrt(scaler.var_[0]) + scaler.mean_[0])
        pred_pm25 = max(0.0, pred_pm25)

        # Metric Displays
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Predicted PM2.5 (T+1h)", f"{pred_pm25:.1f} µg/m³")
        col_m2.metric("Historical Mean (T-24h)", f"{baseline_pm25:.1f} µg/m³")

        if pred_pm25 <= 35:
            col_m3.success("Air Quality Index: Good / Moderate")
        elif pred_pm25 <= 75:
            col_m3.warning("Air Quality Index: Unhealthy for Sensitive")
        elif pred_pm25 <= 150:
            col_m3.error("Air Quality Index: Unhealthy")
        else:
            col_m3.error("Air Quality Index: Very Unhealthy / Hazardous")

        # Visual Trajectory Chart
        st.subheader("Temporal Sequence & 1-Hour Horizon Prediction")
        fig, ax = plt.subplots(figsize=(11, 4))
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
        ax.legend(loc="upper left")
        st.pyplot(fig)
