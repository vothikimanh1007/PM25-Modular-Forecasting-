# benchmark.py
import os
import time
import requests
import numpy as np

BASE_URL = "http://127.0.0.1:8000"
NUM_REQUESTS = 500  # Number of iterations for statistical stability

# Dummy single-step record matching schema
sample_record = {
    "dew": -12.0,
    "temp": -4.0,
    "press": 1020.0,
    "wnd_spd": 2.5,
    "snow": 0.0,
    "rain": 0.0,
    "cbwd_NE": 0,
    "cbwd_NW": 1,
    "cbwd_SE": 0,
    "cbwd_cv": 0
}

# 24-hour sequence for LSTM
sequence_payload = {"timesteps": [sample_record for _ in range(24)]}

def get_file_size_mb(filepath):
    if os.path.exists(filepath):
        return round(os.path.getsize(filepath) / (1024 * 1024), 2)
    return "N/A"

def benchmark_endpoint(endpoint: str, payload: dict):
    latencies = []
    url = f"{BASE_URL}{endpoint}"
    
    # Warm-up call
    try:
        requests.post(url, json=payload, timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"Error: API server is not running at {BASE_URL}. Start it with 'uvicorn src.api.main:app' first.")
        return None

    # Benchmark loop
    wall_start = time.perf_counter()
    for _ in range(NUM_REQUESTS):
        req_start = time.perf_counter()
        resp = requests.post(url, json=payload)
        req_end = time.perf_counter()
        
        if resp.status_code == 200:
            latencies.append((req_end - req_start) * 1000)
        else:
            print(f"Request failed with code {resp.status_code}: {resp.text}")
            break
            
    wall_time = time.perf_counter() - wall_start
    
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    throughput = len(latencies) / wall_time
    
    return {
        "mean_latency_ms": round(mean_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_req_s": round(throughput, 1)
    }

def main():
    print(f"Starting API Benchmark ({NUM_REQUESTS} iterations per model)...")
    
    xgb_results = benchmark_endpoint("/predict/xgboost", sample_record)
    lstm_results = benchmark_endpoint("/predict/lstm", sequence_payload)
    
    if not xgb_results or not lstm_results:
        return

    # Check model file sizes
    xgb_size = get_file_size_mb("src/models/saved/xgb_model.json")
    lstm_size = get_file_size_mb("src/models/saved/lstm_model.h5")

    print("\n" + "="*80)
    print("EMPIRICAL SYSTEM PROFILING RESULTS (Insert into Table in Section V-B)")
    print("="*80)
    print(f"{'Model Module':<18} | {'Latency (Mean)':<14} | {'Latency (P95)':<14} | {'Throughput':<16} | {'Disk Size'}")
    print("-"*80)
    print(f"XGBoost (Baseline) | {xgb_results['mean_latency_ms']} ms{'':<7} | {xgb_results['p95_latency_ms']} ms{'':<7} | {xgb_results['throughput_req_s']} req/s{'':<6} | {xgb_size} MB")
    print(f"LSTM (2-Layer)     | {lstm_results['mean_latency_ms']} ms{'':<7} | {lstm_results['p95_latency_ms']} ms{'':<7} | {lstm_results['throughput_req_s']} req/s{'':<6} | {lstm_size} MB")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
