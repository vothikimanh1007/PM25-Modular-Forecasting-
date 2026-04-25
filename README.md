# Modular API-Ready Deep Learning Architecture for PM2.5 Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation and experimental notebooks for the paper: **"From Knowledge Mapping to Modular Systems: An API-Ready Deep Learning Architecture for PM2.5 Forecasting"**.

This project bridges the gap between environmental science and practical software engineering by providing a modular, highly reusable framework for predicting PM2.5 fine particulate matter concentrations.

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [Citation](#citation)

## 🌍 Project Overview
Accurately forecasting PM2.5 concentrations is critical for urban environmental management. Based on a comprehensive bibliometric analysis of modern AI applications in air quality forecasting, this repository provides a component-based system designed for:
1. **High Reusability:** Easily swap out local IoT monitoring data to fine-tune the pre-trained models.
2. **API Interoperability:** Deploy models seamlessly into existing smart city management dashboards.
3. **High Accuracy:** Utilizing a Long Short-Term Memory (LSTM) neural network optimized for time-series meteorological data.

## 🏗️ System Architecture
The framework is divided into four independent microservices/phases:
* **Phase 1:** Multi-Source Data Acquisition (Local IoT, Open Meteorological, Remote Sensing)
* **Phase 2:** Data Engineering Modularization (Imputation, Z-score/Max normalization)
* **Phase 3:** Component-Based Model Training (XGBoost vs. 2-Layer LSTM with 24h look-back)
* **Phase 4:** Evaluation & API Deployment

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/YourUsername/PM25-Modular-Forecasting.git](https://github.com/YourUsername/PM25-Modular-Forecasting.git)
   cd PM25-Modular-Forecasting
2. Create a virtual environment and install dependencies:
   ```bash
   # Train the baseline Tree-based model
   python src/models/xgboost_module.py --data_path data/raw/beijing_pm25.csv

   # Train the advanced Deep Learning model
   python src/models/lstm_module.py --data_path data/raw/beijing_pm25.csv --lookback 24
3. Deploying the API (Smart City Integration)
To deploy the trained LSTM model as a local API endpoint:
   ```bash
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

📊 Experimental Results
The models were evaluated using an independent test set from the Beijing PM2.5 benchmark dataset. The deep learning module demonstrated a significant breakthrough in capturing complex temporal pollution accumulations.
| Model | Architecture | $R^2$ Score | RMSE ($\mu g/m^3$) |
| :--- | :--- | :---: | :---: |
| **XGBoost** | Baseline Tree-based | 0.6391 | 56.39 |
| **LSTM** | 2-Layer (24h Window) | **0.9523** | **20.61** |

Note: Feature importance analysis via XGBoost confirmed that meteorological variables (Northwest/Southeast wind direction, Dew point, and Temperature) play a decisive role in the dispersion of fine dust.

## Citation

If you use this code or framework in your research, please cite our paper:
  ```bibtex
  @inproceedings{Vo2026PM25,
   title={From Knowledge Mapping to Modular Systems: An API-Ready Deep Learning Architecture for PM2.5 Forecasting},
   author={Vo, Thi Kim Anh},
   booktitle={...},
   year={2026},
   organization={IEEE}
 }

## Contact
For questions or collaboration opportunities, please contact Vo Thi Kim Anh.

### 3. Next Steps to integrate into your Paper:
Once you create this repository on GitHub, add a sentence to your paper's Introduction or Methodology section, such as: 
> *"To ensure reproducibility and facilitate immediate technological transfer to local municipalities, the complete modular source code, deployment API, and experimental datasets have been made publicly available at: [Insert GitHub Link Here]."*





   
