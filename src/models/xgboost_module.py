# src/models/xgboost_module.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Add parent directory to path so we can import preprocessing
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import process_pipeline

def train_xgboost(filepath="data/raw/pollution.csv", save_model_path="src/models/saved/xgb_model.json"):
    """
    Trains the baseline XGBoost model for PM2.5 forecasting.
    """
    print("--- Training Baseline XGBoost Model ---")
    
    # Load data
    try:
        df = pd.read_csv(filepath, index_col=0)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}. Please run data_ingestion.py first.")
        return None
        
    # Preprocess
    X, y = process_pipeline(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    print("Training model...")
    model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    
    # Evaluate Phase 4
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Evaluation Results ---")
    print(f"RMSE: {rmse:.2f} µg/m³")
    print(f"R-squared: {r2:.4f}")
    
    # Save Model
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    model.save_model(save_model_path)
    
    # Save feature names for the API
    joblib.dump(X.columns.tolist(), "src/models/saved/xgb_features.pkl")
    print(f"\nModel saved to {save_model_path}")
    
    return model

if __name__ == "__main__":
    train_xgboost()
