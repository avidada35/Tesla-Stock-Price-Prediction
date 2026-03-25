# ── SECTION 1: Imports ──────────────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, os
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── SECTION 2: Page Config ───────────────────────────────────
st.set_page_config(
    page_title="TSLA Stock Predictor",
    page_icon="",
    layout="wide"
)

# ── SECTION 3: Cache Functions ───────────────────────────────
@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_keras_model(path):
    return load_model(path)

@st.cache_data
def load_data():
    df = pd.read_csv('TSLA.csv', parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)
    return df

@st.cache_data
def load_metrics():
    return pd.read_csv('reports/metrics_comparison.csv')

# ── SECTION 4: Sidebar Controls ─────────────────────────────
st.sidebar.title("TSLA Predictor")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["SimpleRNN", "LSTM", "LSTM (Tuned)"]
)

horizon_choice = st.sidebar.selectbox(
    "Forecast Horizon",
    [1, 5, 10],
    format_func=lambda x: f"{x} Day{'s' if x > 1 else ''}"
)

# Tuned model only exists for horizon=1 — fall back gracefully
if model_choice == "LSTM (Tuned)" and horizon_choice != 1:
    st.sidebar.warning(
        "Tuned model only available for 1-day horizon. "
        "Switching to default LSTM."
    )
    model_choice = "LSTM"

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Window size:** 60 days\n\n"
    "**Train/Test split:** 80/20\n\n"
    "**Dataset:** 2010–2020"
)

# ── SECTION 5: Model Path Resolver ──────────────────────────
def get_model_path(model_choice, horizon):
    if model_choice == "SimpleRNN":
        return f"models/simple_rnn_h{horizon}.keras"
    elif model_choice == "LSTM (Tuned)" and horizon == 1:
        return "models/lstm_best_tuned_h1.keras"
    else:
        return f"models/lstm_h{horizon}.keras"

# ── SECTION 6: Main Page Header ─────────────────────────────
st.title("Tesla Stock Price Prediction")
st.markdown(
    "Deep Learning forecasting using **SimpleRNN** and **LSTM** networks "
    "trained on TSLA historical data (2010–2020)."
)
st.markdown("---")

# ── SECTION 7: Load Data & Model ────────────────────────────
df     = load_data()
scaler = load_scaler()
model  = load_keras_model(get_model_path(model_choice, horizon_choice))

# ── SECTION 8: Generate Prediction ──────────────────────────
def predict(df, scaler, model, horizon, window=60):
    # Extract the most recent window of prices needed
    close_vals = df['Adj Close'].values.reshape(-1, 1)
    # Scale using already-fitted scaler — never fit_transform
    scaled     = scaler.transform(close_vals)
    # Take the last `window` steps as the input sequence
    input_seq  = scaled[-window:].reshape(1, window, 1)
    y_pred_scaled = model.predict(input_seq, verbose=0)
    y_pred        = scaler.inverse_transform(y_pred_scaled)
    return float(y_pred[0][0])

predicted_price = predict(df, scaler, model, horizon_choice)
last_actual     = float(df['Adj Close'].iloc[-1])
price_delta     = predicted_price - last_actual
pct_change      = (price_delta / last_actual) * 100

# ── SECTION 9: KPI Metrics Row ──────────────────────────────
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Last Actual Price",
    value=f"${last_actual:.2f}"
)
col2.metric(
    label=f"Predicted Price (+{horizon_choice}d)",
    value=f"${predicted_price:.2f}",
    delta=f"{price_delta:+.2f} ({pct_change:+.2f}%)"
)
col3.metric(
    label="Model",
    value=model_choice
)
col4.metric(
    label="Horizon",
    value=f"{horizon_choice} Day(s)"
)

# ── SECTION 10: Historical Chart with Prediction ────────────
st.subheader("Historical Price + Forecast Point")

last_180      = df['Adj Close'].iloc[-180:]
last_date     = last_180.index[-1]
forecast_date = last_date + pd.Timedelta(days=horizon_choice)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(last_180.index, last_180.values,
        color='blue', linewidth=1.4, label='Actual Price')

# Red star at the forecast point
ax.plot(forecast_date, predicted_price,
        marker='*', markersize=16,
        color='red', zorder=5, label='Predicted Point')
ax.annotate(
    f"Predicted\n${predicted_price:.2f}",
    xy=(forecast_date, predicted_price),
    xytext=(10, 10), textcoords='offset points',
    fontsize=9, color='red',
    arrowprops=dict(arrowstyle='->', color='red', lw=1)
)

# Vertical dashed line at last known date
ax.axvline(last_date, color='grey', linestyle='--',
           linewidth=1, alpha=0.7, label='Last known date')

ax.set_title(
    f"TSLA Adj Close — Last 180 Days + {horizon_choice}d Forecast",
    fontsize=12
)
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── SECTION 11: Model Performance Table ─────────────────────
st.subheader("Model Performance Comparison")

metrics_df = load_metrics()
filtered   = metrics_df[
    metrics_df['Horizon'] == horizon_choice
].reset_index(drop=True)

st.dataframe(
    filtered.style.highlight_min(
        subset=['RMSE', 'MAE', 'MAPE(%)'],
        color='#d4edda'
    ).format({'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'MAPE(%)': '{:.4f}'}),
    use_container_width=True
)

# ── SECTION 12: Full Test Predictions Chart ──────────────────
st.subheader("Full Test Set — Actual vs Predicted")

def get_full_predictions(df, scaler, model, horizon, window=60):
    close_vals = df['Adj Close'].values.reshape(-1, 1)
    scaled     = scaler.transform(close_vals)      # transform only

    # Inline sequence builder — no external dependency
    X, y = [], []
    for i in range(window, len(scaled) - horizon):
        X.append(scaled[i - window:i])
        y.append(scaled[i + horizon - 1])

    X = np.array(X)   # (samples, window, 1)
    y = np.array(y)   # (samples, 1)

    # 80/20 train-test split — match training pipeline
    split   = int(len(X) * 0.8)
    X_test  = X[split:]
    y_test  = y[split:]

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred        = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true        = scaler.inverse_transform(y_test).flatten()
    return y_true, y_pred

y_true, y_pred = get_full_predictions(df, scaler, model, horizon_choice)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(y_true, color='blue',  label='Actual',    linewidth=1.2)
ax2.plot(y_pred, color='red',   linestyle='--',
         label='Predicted', alpha=0.85,             linewidth=1.2)
ax2.fill_between(range(len(y_true)), y_true, y_pred,
                 alpha=0.1, color='red')
ax2.set_title(
    f"{model_choice} | Horizon {horizon_choice}d | "
    f"RMSE={rmse:.2f} | MAE={mae:.2f} | MAPE={mape:.2f}%",
    fontsize=11
)
ax2.set_xlabel("Test Time Steps")
ax2.set_ylabel("Price (USD)")
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)
plt.close()

# ── SECTION 13: Error Analysis ───────────────────────────────
st.subheader("Prediction Error Analysis")

col_a, col_b = st.columns(2)
errors = y_true - y_pred

with col_a:
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.hist(errors, bins=50,
             color='steelblue', edgecolor='white', alpha=0.8)
    ax3.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax3.set_title("Error Distribution")
    ax3.set_xlabel("Error (USD)")
    ax3.set_ylabel("Frequency")
    plt.tight_layout()
    col_a.pyplot(fig3)
    plt.close()

with col_b:
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.scatter(y_true, y_pred, alpha=0.4,
                color='steelblue', s=10)
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    ax4.plot([min_v, max_v], [min_v, max_v],
             'r--', linewidth=1.5, label='Perfect prediction')
    ax4.set_title("Actual vs Predicted Scatter")
    ax4.set_xlabel("Actual Price (USD)")
    ax4.set_ylabel("Predicted Price (USD)")
    ax4.legend()
    plt.tight_layout()
    col_b.pyplot(fig4)
    plt.close()

# ── SECTION 14: Footer ───────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Project:** Tesla Stock Price Prediction  |  "
    "**Models:** SimpleRNN & LSTM  |  "
    "**Framework:** TensorFlow / Keras  |  "
    "**Data:** Yahoo Finance (2010–2020)"
)
