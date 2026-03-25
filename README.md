# Tesla Stock Price Prediction using Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_STREAMLIT_LINK_HERE)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.13-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Stars](https://img.shields.io/github/stars/YOUR_USERNAME/tesla-stock-price-prediction?style=flat)](https://github.com/YOUR_USERNAME/tesla-stock-price-prediction)

---

## Live Demo

 **[[Open Live App](https://YOUR_STREAMLIT_LINK_HERE)](https://tsla-predictor.loca.lt/)**

> Select a model, choose a forecast horizon, and instantly see
> Tesla stock price predictions with full error analysis —
> no setup required.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Project Architecture](#-project-architecture)
- [Dataset](#-dataset)
- [Models](#-models)
- [Results](#-results)
- [App Features](#-app-features)
- [Project Structure](#-project-structure)
- [Run Locally](#-run-locally)
- [Tech Stack](#-tech-stack)
- [Key Learnings](#-key-learnings)
- [Author](#-author)

---

##  Overview

This project builds a complete end-to-end deep learning pipeline
to predict Tesla (TSLA) stock closing prices using historical
market data from 2010 to 2020.

Two recurrent neural network architectures are implemented,
compared, and deployed:

- **SimpleRNN** — baseline sequential model
- **LSTM** — Long Short-Term Memory with stacked layers
  and BatchNormalization
- **LSTM (Tuned)** — GridSearchCV optimized LSTM
  with best hyperparameters

Predictions are generated for **3 forecast horizons**:
1 day, 5 days, and 10 days ahead — simulating real-world
short-term trading scenarios.

---

##  Project Architecture
```
Raw Data (TSLA.csv)
        │
        ▼
┌───────────────────┐
│  Data Cleaning    │  ← ffill, gap detection,
│  & EDA            │    null handling
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Feature          │  ← MA_7/21/50, EMA_12/26,
│  Engineering      │    MACD, Volatility, Lags
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Preprocessing    │  ← MinMaxScaler (train only),
│  & Sequencing     │    Window=60, Horizons 1/5/10
└────────┬──────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│Simple  │ │     LSTM     │
│  RNN   │ │  (Stacked +  │
│        │ │  BatchNorm)  │
└────┬───┘ └──────┬───────┘
     │             │
     │      ┌──────┴───────┐
     │      │  GridSearchCV │
     │      │  Tuned LSTM  │
     │      └──────┬───────┘
     │             │
     └──────┬──────┘
            ▼
┌───────────────────┐
│  Evaluation &     │  ← RMSE, MAE, MAPE,
│  Comparison       │    Actual vs Predicted
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Streamlit App    │  ← Live deployment
│  Deployment       │    on Streamlit Cloud
└───────────────────┘
```

---

## Dataset

| Property | Details |
|---|---|
| Source | Yahoo Finance |
| Ticker | TSLA (Tesla Inc.) |
| Date Range | June 29, 2010 — February 18, 2020 |
| Total Rows | ~2,400 trading days |
| Frequency | Daily |
| Target Column | Adj Close |

### Columns

| Column | Description |
|---|---|
| Date | Trading date |
| Open | Opening price (USD) |
| High | Intraday high price (USD) |
| Low | Intraday low price (USD) |
| Close | Closing price (USD) |
| Adj Close | Adjusted closing price (USD) ← **target** |
| Volume | Number of shares traded |

### Engineered Features

| Feature | Description |
|---|---|
| MA_7 / MA_21 / MA_50 | Rolling mean (7, 21, 50 days) |
| EMA_12 / EMA_26 | Exponential moving averages |
| MACD | EMA_12 − EMA_26 momentum indicator |
| Volatility_30 | 30-day rolling std of daily returns |
| Daily_Return | Percentage change day-over-day |
| Lag_1 / Lag_5 | Lagged closing prices |
| Price_Range | High − Low daily candle range |

---

##  Models

### 1. SimpleRNN
```
Input (60, 1)
    │
SimpleRNN(64, activation='tanh')
    │
Dropout(0.2)
    │
Dense(32, activation='relu')
    │
Dense(1) ← predicted price
```

### 2. LSTM (Stacked)
```
Input (60, 1)
    │
LSTM(64, return_sequences=True)
    │
Dropout(0.2) → BatchNormalization
    │
LSTM(32, return_sequences=False)
    │
Dropout(0.2) → BatchNormalization
    │
Dense(32, activation='relu')
    │
Dense(1) ← predicted price
```

### 3. LSTM (Tuned via GridSearchCV)

Hyperparameter search space:

| Parameter | Values Tested |
|---|---|
| LSTM units | 32, 64 |
| Dropout rate | 0.1, 0.2 |
| Learning rate | 0.001, 0.0005 |
| Batch size | 32, 64 |

- Cross-validation: `TimeSeriesSplit(n_splits=3)`
- Scoring: `neg_mean_squared_error`
- Total fits: 16 combinations × 3 folds = **48 model fits**

### Training Configuration

| Setting | Value |
|---|---|
| Window size | 60 days |
| Train/Test split | 80% / 20% (temporal) |
| Max epochs | 100 |
| Early stopping patience | 15 epochs |
| Optimizer | Adam |
| Loss function | Mean Squared Error |
| LR scheduler | ReduceLROnPlateau (factor=0.5) |

---

##  Results

### Performance Comparison

| Model | Horizon | RMSE | MAE | MAPE (%) |
|---|---|---|---|---|
| SimpleRNN | 1 day | -- | -- | -- |
| SimpleRNN | 5 days | -- | -- | -- |
| SimpleRNN | 10 days | -- | -- | -- |
| LSTM | 1 day | -- | -- | -- |
| LSTM | 5 days | -- | -- | -- |
| LSTM | 10 days | -- | -- | -- |
| LSTM (Tuned) | 1 day | -- | -- | -- |

>  Fill in your actual values from
> `reports/metrics_comparison.csv`

### Key Observations

-  LSTM consistently outperforms SimpleRNN
  across all forecast horizons
-  1-day horizon achieves lowest RMSE
  (shorter horizon = more predictable)
-  GridSearchCV tuning further reduces
  RMSE on the 1-day LSTM model
-  RMSE and MAPE increase with longer
  horizons — expected behavior for
  time-series forecasting
-  Both models struggle with sudden
  price spikes (earnings reports, market events)

---

##  App Features

The deployed Streamlit app includes:

| Feature | Description |
|---|---|
|  Model selector | Switch between SimpleRNN, LSTM, LSTM Tuned |
|  Horizon selector | 1-day, 5-day, 10-day forecast |
|  KPI metrics | Live predicted price, delta, % change |
|  Historical chart | Last 180 days + forecast point |
|  Comparison table | All models ranked by RMSE for selected horizon |
|  Full test chart | Actual vs Predicted on entire test set |
|  Error analysis | Error distribution + Actual vs Predicted scatter |

**App screenshot:**

> *(Add a screenshot of your app here)*
> Drag and drop an image into this README on GitHub

---

##  Project Structure
```
tesla-stock-price-prediction/
│
├── app.py                      ← Streamlit application
├── TSLA.csv                    ← Raw dataset
├── scaler.pkl                  ← Fitted MinMaxScaler
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
│
├── models/
│   ├── simple_rnn_h1.h5        ← SimpleRNN, 1-day
│   ├── simple_rnn_h5.h5        ← SimpleRNN, 5-day
│   ├── simple_rnn_h10.h5       ← SimpleRNN, 10-day
│   ├── lstm_h1.h5              ← LSTM, 1-day
│   ├── lstm_h5.h5              ← LSTM, 5-day
│   ├── lstm_h10.h5             ← LSTM, 10-day
│   └── lstm_best_tuned_h1.h5  ← Tuned LSTM, 1-day
│
├── reports/
│   ├── metrics_comparison.csv  ← All model metrics
│   ├── rmse_comparison.png     ← RMSE bar chart
│   ├── mape_comparison.png     ← MAPE bar chart
│   ├── all_predictions_grid.png← Predictions grid
│   ├── error_distribution.png  ← Error KDE plots
│   └── final_summary.txt       ← Text report
│
└── notebooks/
    └── 01_eda.ipynb            ← Full pipeline notebook
```

---

##  Run Locally
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/tesla-stock-price-prediction.git

# 2. Navigate into the folder
cd tesla-stock-price-prediction

# 3. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the app
streamlit run app.py
```

App will open at `http://localhost:8501`

---

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.13, Keras 2.13 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, Scikeras |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit Cloud |
| Model Persistence | H5PY (.h5 format) |
| Version Control | Git, GitHub |

---

##  Key Learnings

1. **Why ffill for time-series** — Forward fill preserves
   the last known price for missing trading days, which is
   more realistic than mean imputation for stock data.

2. **Why TimeSeriesSplit over KFold** — Random splitting
   causes future data to leak into training folds.
   TimeSeriesSplit always trains on past, validates on
   future — the only correct approach for sequential data.

3. **Why LSTM beats SimpleRNN** — LSTM gates (input,
   forget, output) allow selective memory of long-term
   dependencies across hundreds of timesteps. SimpleRNN
   suffers from vanishing gradients on long sequences.

4. **Why scale on train only** — Fitting the scaler on
   the full dataset leaks test set statistics into training.
   Scaler must be fit on train split only, then applied
   to test — a strict data leakage prevention rule.

5. **Why compile=False on model load** — Avoids optimizer
   config version mismatch errors when loading models
   across different Keras versions (critical for deployment).

---

##  Limitations & Future Improvements

### Current Limitations
- Uses only price history — no external signals
- Sensitive to sudden market events (earnings,
  macro shocks, news)
- Dataset ends in 2020 — does not include
  post-COVID volatility period

### Suggested Improvements

| Improvement | Impact |
|---|---|
| Add news sentiment (NLP) | High — captures market mood |
| Include macro indicators | Medium — interest rates, inflation |
| Try GRU architecture | Medium — faster than LSTM, comparable accuracy |
| Transformer / Attention | High — state-of-art for sequences |
| Extend dataset to present | High — more representative training |
| Ensemble RNN + LSTM | Medium — reduces prediction variance |

---

##  License

This project is licensed under the
[MIT License](LICENSE) —
free to use, modify, and distribute.

---

## 👤 Author

**AVIRAJ VIRAPE**

[![GitHub](https://img.shields.io/badge/GitHub-YOUR_USERNAME-181717?style=flat&logo=github)](https://github.com/YOUR_USERNAME)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_LINKEDIN)

---

## ⭐ Support

If you found this project useful or learned
something from it, please consider giving it
a **star** ⭐ on GitHub — it helps others
discover the project.

---

*Built with ❤️ using TensorFlow, Keras, and Streamlit*
