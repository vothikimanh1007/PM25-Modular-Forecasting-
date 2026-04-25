# src/models/lstm_module.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import process_pipeline, normalize_features

def create_dataset(X, y, time_steps=24):
    """
    Creates the look-back window structure required for LSTM.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape):
    """
    Builds the 2-Layer LSTM Architecture.
    """
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(filepath="data/raw/pollution.csv", time_steps=24):
    """
    Trains the advanced LSTM model for PM2.5 forecasting.
    """
    print("--- Training Advanced LSTM Model ---")
    
    df = pd.read_csv(filepath, index_col=0)
    X, y = process_pipeline(df)
    
    # Time series split (do not shuffle to preserve temporal order)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Normalize Phase 2
    X_train, X_test, scaler = normalize_features(X_train_raw, X_test_raw, method='min-max')
    
    # Create Time Windows
    print(f"Creating {time_steps}-hour look-back windows...")
    X_train_seq, y_train_seq = create_dataset(X_train, y_train_raw, time_steps)
    X_test_seq, y_test_seq = create_dataset(X_test, y_test_raw, time_steps)
    
    # Build and Train Phase 3
    print("Building and training architecture...")
    model = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=30, # Keep low for quick demonstration
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate Phase 4
    y_pred = model.predict(X_test_seq)
    rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
    r2 = r2_score(y_test_seq, y_pred)
    
    print("\n--- Evaluation Results ---")
    print(f"RMSE: {rmse:.2f} µg/m³")
    print(f"R-squared: {r2:.4f}")
    
    # Save models for API
    os.makedirs("src/models/saved", exist_ok=True)
    model.save("src/models/saved/lstm_model.keras")
    joblib.dump(scaler, "src/models/saved/lstm_scaler.pkl")
    print("\nModel and Scaler saved.")
    
    return model

if __name__ == "__main__":
    train_lstm()