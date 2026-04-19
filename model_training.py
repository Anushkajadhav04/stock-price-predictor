"""
Phase 3: Model Training — LSTM + ARIMA
Stock / Crypto Price Predictor Project
---------------------------------------
Run AFTER feature_engineering.py
Run:  pip install tensorflow statsmodels scikit-learn joblib
Then: python model_training.py
"""

import numpy as np
import pandas as pd
import os
import joblib
import matplotlib
matplotlib.use("Agg")               # headless — swap to "TkAgg" if you want live plots
import matplotlib.pyplot as plt

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────

PREPARED_DIR = "prepared"       # output of feature_engineering.py
MODELS_DIR   = "models"         # where trained models are saved
PLOTS_DIR    = "plots"          # where evaluation charts are saved

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

EPOCHS       = 100              # max epochs; EarlyStopping will cut this short
BATCH_SIZE   = 32
LSTM_UNITS   = 50               # units per LSTM layer
DROPOUT      = 0.2
LEARNING_RATE= 0.001


# ─────────────────────────────────────────────
# 2. LOAD PREPARED DATA
# ─────────────────────────────────────────────

def load_prepared(ticker: str):
    """Loads the .npy arrays and scalers saved by feature_engineering.py"""
    safe = ticker.replace(".", "_").replace("-", "_")
    p    = PREPARED_DIR

    X_train = np.load(os.path.join(p, f"{safe}_X_train.npy"))
    y_train = np.load(os.path.join(p, f"{safe}_y_train.npy"))
    X_test  = np.load(os.path.join(p, f"{safe}_X_test.npy"))
    y_test  = np.load(os.path.join(p, f"{safe}_y_test.npy"))
    scaler_close = joblib.load(os.path.join(p, f"{safe}_scaler_close.pkl"))

    print(f"  Loaded  X_train={X_train.shape}  X_test={X_test.shape}")
    return X_train, y_train, X_test, y_test, scaler_close


# ─────────────────────────────────────────────
# 3. BUILD LSTM MODEL
# ─────────────────────────────────────────────

def build_lstm(seq_len: int, n_features: int) -> tf.keras.Model:
    """
    Two stacked LSTM layers with Dropout regularisation.

    Architecture:
        Input  → (seq_len, n_features)
        LSTM 1 → 50 units, return_sequences=True  (passes sequences to next LSTM)
        Dropout → 0.2  (randomly zeros 20% of neurons → prevents overfitting)
        LSTM 2 → 50 units, return_sequences=False (only final timestep output)
        Dropout → 0.2
        Dense  → 1  (single predicted price value)

    Why two LSTM layers?
        First layer learns short-term patterns (days).
        Second layer learns longer-term dependencies (weeks/months).
    """
    model = Sequential([
        Input(shape=(seq_len, n_features)),

        LSTM(LSTM_UNITS, return_sequences=True),
        Dropout(DROPOUT),

        LSTM(LSTM_UNITS, return_sequences=False),
        Dropout(DROPOUT),

        Dense(25, activation="relu"),   # small bottleneck layer
        Dense(1),                       # output: next-day price (scaled)
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",      # MSE is standard for regression
    )

    model.summary()
    return model


# ─────────────────────────────────────────────
# 4. TRAIN LSTM
# ─────────────────────────────────────────────

def train_lstm(ticker: str, X_train, y_train, X_test, y_test):
    """
    Trains the LSTM with three callbacks:
      - EarlyStopping   : stops if val_loss doesn't improve for 15 epochs
                          (saves time; avoids overfitting)
      - ModelCheckpoint : saves the best model weights automatically
      - ReduceLROnPlateau: halves learning rate if stuck (helps escape plateaus)
    """
    safe  = ticker.replace(".", "_").replace("-", "_")
    seq_len, n_features = X_train.shape[1], X_train.shape[2]

    model = build_lstm(seq_len, n_features)

    model_path = os.path.join(MODELS_DIR, f"{safe}_lstm.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path, monitor="val_loss",
            save_best_only=True, verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        ),
    ]

    print(f"\n  Training LSTM for {ticker} ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"  Saved model → {model_path}")
    return model, history


# ─────────────────────────────────────────────
# 5. EVALUATE LSTM
# ─────────────────────────────────────────────

def evaluate_lstm(model, X_test, y_test, scaler_close, ticker: str) -> dict:
    """
    Evaluates the LSTM on test data and returns metrics.

    Metrics:
      - RMSE  : Root Mean Squared Error (in original price units)
      - MAPE  : Mean Absolute Percentage Error (%)
      - Directional Accuracy : % of times model correctly predicted
                               whether price went UP or DOWN
                               (most useful metric for trading signals)
    """
    preds_scaled = model.predict(X_test, verbose=0)

    # Inverse-transform back to real price values
    preds  = scaler_close.inverse_transform(preds_scaled).flatten()
    actual = scaler_close.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(actual, preds))
    mape = mean_absolute_percentage_error(actual, preds) * 100

    # Directional accuracy: did the model predict the right direction?
    actual_dir = np.sign(np.diff(actual))
    pred_dir   = np.sign(np.diff(preds))
    dir_acc    = np.mean(actual_dir == pred_dir) * 100

    metrics = {
        "ticker":      ticker,
        "RMSE":        round(rmse, 4),
        "MAPE_%":      round(mape, 2),
        "Dir_Acc_%":   round(dir_acc, 2),
        "preds":       preds,
        "actual":      actual,
    }

    print(f"\n  ── LSTM Results: {ticker} ──────────────────")
    print(f"     RMSE               : {rmse:.4f}")
    print(f"     MAPE               : {mape:.2f}%")
    print(f"     Directional Acc.   : {dir_acc:.2f}%  ← key metric for trading")

    return metrics


# ─────────────────────────────────────────────
# 6. ARIMA MODEL (baseline comparison)
# ─────────────────────────────────────────────

def check_stationarity(series: pd.Series) -> bool:
    """
    ADF test: if p-value < 0.05, series is stationary (no unit root).
    ARIMA needs a stationary series — if not, we difference it (d=1 or d=2).
    """
    result = adfuller(series.dropna())
    return result[1] < 0.05   # True = stationary


def train_arima(ticker: str, data_dir: str = "data") -> dict:
    """
    Trains an ARIMA(5,1,0) model on the Close price series.
    ARIMA order (p, d, q):
        p=5 : uses last 5 observations (autoregressive terms)
        d=1 : first-difference to make series stationary
        q=0 : no moving-average terms (keep it simple for baseline)

    We use a rolling forecast: retrain on each new day's data
    for a realistic out-of-sample evaluation.
    """
    safe     = ticker.replace(".", "_").replace("-", "_")
    csv_path = os.path.join(data_dir, f"{safe}.csv")

    if not os.path.exists(csv_path):
        print(f"  SKIP ARIMA for {ticker}: CSV not found")
        return {}

    df    = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    close = df["Close"].dropna()

    train_size = int(len(close) * 0.80)
    train      = close.iloc[:train_size].values
    test       = close.iloc[train_size:].values

    is_stationary = check_stationarity(pd.Series(train))
    d = 0 if is_stationary else 1
    print(f"\n  ARIMA for {ticker} | Stationary={is_stationary} → d={d}")

    # Rolling forecast — more realistic than one-shot forecast
    history = list(train)
    preds   = []

    print(f"  Rolling ARIMA forecast over {len(test)} test days ...")
    for i, obs in enumerate(test):
        if i % 100 == 0:
            print(f"    Day {i}/{len(test)} ...")
        try:
            model  = ARIMA(history, order=(5, d, 0))
            result = model.fit()
            yhat   = result.forecast(steps=1)[0]
        except Exception:
            yhat = history[-1]     # fallback: persist last value
        preds.append(yhat)
        history.append(obs)        # add real observation for next iteration

    preds  = np.array(preds)
    actual = test

    rmse = np.sqrt(mean_squared_error(actual, preds))
    mape = mean_absolute_percentage_error(actual, preds) * 100

    actual_dir = np.sign(np.diff(actual))
    pred_dir   = np.sign(np.diff(preds))
    dir_acc    = np.mean(actual_dir == pred_dir) * 100

    metrics = {
        "ticker":    ticker,
        "RMSE":      round(rmse, 4),
        "MAPE_%":    round(mape, 2),
        "Dir_Acc_%": round(dir_acc, 2),
        "preds":     preds,
        "actual":    actual,
    }

    print(f"  ── ARIMA Results: {ticker} ──────────────────")
    print(f"     RMSE               : {rmse:.4f}")
    print(f"     MAPE               : {mape:.2f}%")
    print(f"     Directional Acc.   : {dir_acc:.2f}%")

    # Save ARIMA predictions for dashboard use
    arima_df = pd.DataFrame({"actual": actual, "predicted": preds})
    arima_df.to_csv(os.path.join(MODELS_DIR, f"{safe}_arima_preds.csv"), index=False)

    return metrics


# ─────────────────────────────────────────────
# 7. BUY / SELL / HOLD SIGNALS
# ─────────────────────────────────────────────

def generate_signals(preds: np.ndarray, actual: np.ndarray,
                     buy_threshold: float = 0.01,
                     sell_threshold: float = -0.01) -> pd.DataFrame:
    """
    Generates BUY / SELL / HOLD signals and a confidence score.

    Logic:
        predicted_return = (predicted_price - current_price) / current_price
        if predicted_return >  1%  → BUY
        if predicted_return < -1%  → SELL
        else                       → HOLD

    Confidence = abs(predicted_return) normalised to 0–100%
    (crude but interpretable — good enough for a portfolio project)
    """
    current    = actual[:-1]
    future_hat = preds[1:]

    pred_return = (future_hat - current) / current

    signals    = []
    confidence = []
    for r in pred_return:
        conf = min(abs(r) / 0.05, 1.0) * 100      # 5% move → 100% confidence
        if r > buy_threshold:
            signals.append("BUY")
        elif r < sell_threshold:
            signals.append("SELL")
        else:
            signals.append("HOLD")
        confidence.append(round(conf, 1))

    return pd.DataFrame({
        "current_price":   current,
        "predicted_price": future_hat,
        "predicted_return_%": (pred_return * 100).round(2),
        "signal":          signals,
        "confidence_%":    confidence,
    })


# ─────────────────────────────────────────────
# 8. PLOT RESULTS
# ─────────────────────────────────────────────

def plot_predictions(lstm_metrics: dict, arima_metrics: dict, ticker: str):
    """Saves a comparison chart of LSTM vs ARIMA predictions vs actual prices."""
    safe = ticker.replace(".", "_").replace("-", "_")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"{ticker} — Predictions vs Actual", fontsize=14)

    for ax, m, label in zip(
        axes,
        [lstm_metrics, arima_metrics],
        ["LSTM", "ARIMA"]
    ):
        if not m:
            ax.set_title(f"{label} — no data")
            continue
        n = min(len(m["actual"]), len(m["preds"]))
        ax.plot(m["actual"][:n],  label="Actual",    color="#374151", linewidth=1.2)
        ax.plot(m["preds"][:n],   label=f"{label} Predicted",
                color="#3b82f6" if label == "LSTM" else "#f59e0b",
                linewidth=1.0, linestyle="--")
        ax.set_title(
            f"{label}  |  RMSE={m['RMSE']}  MAPE={m['MAPE_%']}%  "
            f"Dir.Acc={m['Dir_Acc_%']}%"
        )
        ax.legend(fontsize=9)
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{safe}_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Chart saved → {path}")


def plot_training_history(history, ticker: str):
    """Plots training vs validation loss over epochs."""
    safe = ticker.replace(".", "_").replace("-", "_")

    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"],     label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.title(f"{ticker} — LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{safe}_loss.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Loss chart saved → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    TICKERS = [
        "TCS.NS",        # start with one; add others once this works
        # "RELIANCE.NS",
        # "BTC-USD",
    ]

    all_results = []

    for ticker in TICKERS:
        print(f"\n{'='*55}")
        print(f"  TICKER: {ticker}")
        print(f"{'='*55}")

        # ── LSTM ─────────────────────────────────────────
        try:
            X_train, y_train, X_test, y_test, scaler_close = load_prepared(ticker)
            model, history = train_lstm(ticker, X_train, y_train, X_test, y_test)
            lstm_m = evaluate_lstm(model, X_test, y_test, scaler_close, ticker)
            plot_training_history(history, ticker)
        except FileNotFoundError:
            print(f"  Prepared data not found for {ticker}. Run feature_engineering.py first.")
            lstm_m = {}

        # ── ARIMA ────────────────────────────────────────
        arima_m = train_arima(ticker)

        # ── Signals (from LSTM predictions) ──────────────
        if lstm_m:
            signals_df = generate_signals(lstm_m["preds"], lstm_m["actual"])
            safe = ticker.replace(".", "_").replace("-", "_")
            signals_df.to_csv(os.path.join(MODELS_DIR, f"{safe}_signals.csv"), index=False)
            print(f"\n  Last 5 signals for {ticker}:")
            print(signals_df.tail())

        # ── Comparison chart ──────────────────────────────
        if lstm_m or arima_m:
            plot_predictions(lstm_m, arima_m, ticker)

        all_results.append({"ticker": ticker, "lstm": lstm_m, "arima": arima_m})

    # ── Final summary table ───────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'Ticker':<15} {'Model':<8} {'RMSE':>10} {'MAPE%':>8} {'Dir.Acc%':>10}")
    print(f"{'='*65}")
    for r in all_results:
        for label, m in [("LSTM", r["lstm"]), ("ARIMA", r["arima"])]:
            if m:
                print(f"{r['ticker']:<15} {label:<8} {m['RMSE']:>10} "
                      f"{m['MAPE_%']:>8} {m['Dir_Acc_%']:>10}")
    print(f"{'='*65}")
    print("\nDone! Check models/ and plots/ folders.\n")
    print("Next: Phase 4 — Streamlit Dashboard")
