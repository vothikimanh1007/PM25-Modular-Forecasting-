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
