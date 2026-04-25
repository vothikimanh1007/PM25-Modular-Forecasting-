# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def clean_and_impute(df):
    """
    Handles missing values and parses dates if applicable.
    """
    print("Cleaning and imputing missing data...")
    # Drop rows where the target (PM2.5) is missing
    if 'pm2.5' in df.columns:
        initial_len = len(df)
        df = df.dropna(subset=['pm2.5'])
        print(f"Dropped {initial_len - len(df)} rows with missing PM2.5 values.")
    
    # Forward fill remaining meteorological variables
    df = df.fillna(method='ffill')
    return df

def encode_categorical(df, cat_columns=['cbwd']):
    """
    One-hot encodes categorical variables like wind direction.
    """
    print(f"Encoding categorical variables: {cat_columns}")
    for col in cat_columns:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col)
    return df

def normalize_features(X_train, X_test, method='z-score'):
    """
    Applies normalization to the feature sets.
    method: 'z-score' (StandardScaler) or 'min-max' (MinMaxScaler)
    """
    print(f"Normalizing features using {method}...")
    if method == 'z-score':
        scaler = StandardScaler()
    elif method == 'min-max':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'z-score' or 'min-max'")
        
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Return scaled data as DataFrames to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, scaler

def process_pipeline(df, target_col='pm2.5'):
    """
    Executes the full preprocessing pipeline (Phase 2).
    """
    df_clean = clean_and_impute(df)
    df_encoded = encode_categorical(df_clean)
    
    # Separate features and target
    y = df_encoded[target_col]
    X = df_encoded.drop(columns=[target_col])
    
    # If the dataset has non-predictive columns like 'No' (index) or 'year', drop them
    cols_to_drop = [c for c in ['No', 'year'] if c in X.columns]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
        
    return X, y

if __name__ == "__main__":
    # Test the preprocessing module
    from data_ingestion import download_benchmark_data
    raw_df = download_benchmark_data()
    X, y = process_pipeline(raw_df)
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)