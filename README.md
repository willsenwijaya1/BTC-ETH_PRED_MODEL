# BTC-ETH Forecast Dashboard

A Streamlit-based cryptocurrency forecasting application that predicts the next-day closing price of Bitcoin (BTC) and Ethereum (ETH) using machine learning and deep learning models. The system automatically retrieves real-time market data, performs feature engineering with technical and macro-financial indicators, generates predictions, and stores historical forecasts in a Supabase database for monitoring and analysis.

---

## Overview

This project was developed as part of an undergraduate thesis focusing on cryptocurrency price prediction using macro-financial indicators and machine learning techniques.

The application provides:

- Real-time Bitcoin and Ethereum market data retrieval
- Automated feature engineering pipeline
- Next-day closing price prediction
- Interactive forecasting dashboard
- Historical prediction logging
- Prediction performance monitoring
- Cloud deployment using Streamlit

---

## Features

### Real-Time Data Collection

The application automatically downloads the latest market data using Yahoo Finance, including:

- Bitcoin (BTC-USD)
- Ethereum (ETH-USD)
- Gold
- S&P 500
- U.S. Dollar Index (DXY)

### Feature Engineering

Features are generated automatically from market data, including:

- Technical indicators
  - RSI
  - MACD
  - EMA
  - SMA
  - Volatility
  - Momentum

- Lag Features
  - Previous-day price values
  - Historical returns
  - Lagged macro-financial indicators

- Macro-financial variables
  - Gold price
  - S&P 500 index
  - U.S. Dollar Index (DXY)

### Machine Learning Models

#### Bitcoin Forecasting

Model:
- XGBoost Regressor

Target:
- Next-day Bitcoin log closing price

#### Ethereum Forecasting

Model:
- GRU (Gated Recurrent Unit)

Target:
- Next-day Ethereum log closing price

### Historical Prediction Logging

Every generated forecast is automatically saved to Supabase, including:

- Prediction timestamp
- Asset type
- Predicted close price
- Predicted percentage change
- Model information

### Interactive Dashboard

The dashboard provides:

- Latest prediction summary
- Historical price visualization
- Forecast trajectory chart
- Historical prediction chart
- Prediction log table
- Model metadata and evaluation metrics

## Technology Stack

### Frontend

- Streamlit
- Plotly

### Data Processing

- Pandas
- NumPy

### Machine Learning

- XGBoost
- TensorFlow
- Scikit-learn

### Financial Data

- Yahoo Finance (yfinance)
- TA (Technical Analysis Library)

### Database

- Supabase PostgreSQL

### Deployment

- GitHub
- Streamlit Community Cloud
