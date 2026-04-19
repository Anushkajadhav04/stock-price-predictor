"""
Phase 2: Feature Engineering & LSTM Data Preparation
Stock / Crypto Price Predictor Project
---------------------------------------
Run AFTER data_collection.py
Run:  pip install scikit-learn ta
Then: python feature_engineering.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib                          # to save/load scalers
import ta                              # technical indicators library

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────

DATA_DIR      = "data"           # where CSVs from Phase 1 are saved
OUTPUT_DIR    = "prepared"       # where processed numpy arrays go
SEQUENCE_LEN  = 30               # use last 30 days to predict next day
TRAIN_RATIO   = 0.80             # 80% train, 20% test
TARGET_COL    = "Close"          # what we are predicting

# Features the model will learn from
FEATURE_COLS = [
    "Close", "Volume",
    "daily_return", "log_return",
    "MA7", "MA21", "MA50",
    "EMA12", "EMA26",
    "volatility_21d", "price_range",
    # Technical indicators added below:
    "RSI", "MACD", "MACD_signal",
    "BB_upper", "BB_lower", "BB_width",
    "OBV",                             # On-Balance Volume
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 2. ADD TECHNICAL INDICATORS
# ─────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds RSI, MACD, Bollinger Bands, and OBV using the `ta` library.
    These are the most common indicators used by traders and ML models.
    """
    df = df.copy()
    close  = df["Close"]
    volume = df["Volume"]

    # ── RSI (Relative Strength Index) ──────────────────────────────
    # Measures momentum: >70 = overbought (possible sell), <30 = oversold (possible buy)
    df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # ── MACD (Moving Average Convergence Divergence) ────────────────
    # Trend-following indicator; signal line crossovers = buy/sell signals
    macd_obj         = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]       = macd_obj.macd()
    df["MACD_signal"]= macd_obj.macd_signal()
    df["MACD_diff"]  = macd_obj.macd_diff()   # histogram

    # ── Bollinger Bands ─────────────────────────────────────────────
    # Price volatility bands: price near upper band = overbought, near lower = oversold
    bb_obj         = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["BB_upper"] = bb_obj.bollinger_hband()
    df["BB_lower"] = bb_obj.bollinger_lband()
    df["BB_mid"]   = bb_obj.bollinger_mavg()
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]  # normalised width

    # ── OBV (On-Balance Volume) ─────────────────────────────────────
    # Cumulative volume indicator: rising OBV with rising price = strong trend
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

    return df


# ─────────────────────────────────────────────
# 3. SCALE FEATURES
# ─────────────────────────────────────────────

def scale_features(df: pd.DataFrame, ticker: str, feature_cols: list):
    """
    Scales each feature to [0, 1] using MinMaxScaler.

    IMPORTANT: We fit the scaler ONLY on training data, then transform
    both train and test. Fitting on all data would cause data leakage —
    the model would indirectly "see" future data during training.

    Returns:
        X_train, X_test  — shaped (samples, SEQUENCE_LEN, n_features)
        y_train, y_test  — shaped (samples,)  — next-day Close price (scaled)
        scaler_close     — scaler for Close only (to inverse-transform predictions)
    """
    # Drop rows with NaN (from rolling window indicators)
    df = df[feature_cols].dropna().copy()

    n = len(df)
    train_size = int(n * TRAIN_RATIO)

    train_df = df.iloc[:train_size]
    test_df  = df.iloc[train_size:]

    # Fit scalers on TRAINING data only
    scaler_all   = MinMaxScaler(feature_range=(0, 1))
    scaler_close = MinMaxScaler(feature_range=(0, 1))

    train_scaled = scaler_all.fit_transform(train_df)
    test_scaled  = scaler_all.transform(test_df)       # transform only, no fit!

    # Also fit a separate scaler just for Close (for inverse-transforming predictions)
    close_idx = feature_cols.index("Close")
    scaler_close.fit(train_df[["Close"]])

    # Save scalers so we can reuse them in the dashboard without refitting
    safe_name = ticker.replace(".", "_").replace("-", "_")
    joblib.dump(scaler_all,   os.path.join(OUTPUT_DIR, f"{safe_name}_scaler_all.pkl"))
    joblib.dump(scaler_close, os.path.join(OUTPUT_DIR, f"{safe_name}_scaler_close.pkl"))

    return train_scaled, test_scaled, scaler_close, close_idx


# ─────────────────────────────────────────────
# 4. CREATE SEQUENCES (the key LSTM reshape)
# ─────────────────────────────────────────────

def create_sequences(data: np.ndarray, seq_len: int, close_idx: int):
    """
    Converts a 2D array [timesteps, features] into LSTM-ready sequences.

    For each position t, we take:
        X[t] = data[t - seq_len : t]        → shape (seq_len, n_features)
        y[t] = data[t, close_idx]           → next-day Close (scaled)

    Final shapes:
        X → (n_samples, seq_len, n_features)   ← what LSTM expects
        y → (n_samples,)

    Think of it like a sliding window of 30 days moving one day at a time.
    """
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len : i])        # 30 days of all features
        y.append(data[i, close_idx])            # next day's Close (scaled)

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# 5. FULL PIPELINE FOR ONE TICKER
# ─────────────────────────────────────────────

def prepare_ticker(ticker: str) -> dict:
    """
    Runs the full pipeline for one ticker:
      1. Load CSV from Phase 1
      2. Add technical indicators
      3. Select & scale features
      4. Create LSTM sequences
      5. Save arrays to disk
    Returns a summary dict.
    """
    safe_name = ticker.replace(".", "_").replace("-", "_")
    csv_path  = os.path.join(DATA_DIR, f"{safe_name}.csv")

    if not os.path.exists(csv_path):
        print(f"  SKIP {ticker}: CSV not found at {csv_path}")
        return {}

    print(f"\n  Processing {ticker} ...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Step A: add indicators
    df = add_technical_indicators(df)

    # Step B: keep only available feature columns
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        print(f"    Note: missing columns {missing} — skipping them")

    # Step C: scale
    train_scaled, test_scaled, scaler_close, close_idx = scale_features(
        df, ticker, available_features
    )

    # Step D: create sequences
    X_train, y_train = create_sequences(train_scaled, SEQUENCE_LEN, close_idx)
    X_test,  y_test  = create_sequences(test_scaled,  SEQUENCE_LEN, close_idx)

    # Step E: save arrays
    np.save(os.path.join(OUTPUT_DIR, f"{safe_name}_X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, f"{safe_name}_y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, f"{safe_name}_X_test.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, f"{safe_name}_y_test.npy"),  y_test)

    summary = {
        "ticker":     ticker,
        "features":   available_features,
        "n_features": len(available_features),
        "seq_len":    SEQUENCE_LEN,
        "X_train":    X_train.shape,
        "X_test":     X_test.shape,
        "train_rows": len(train_scaled),
        "test_rows":  len(test_scaled),
    }

    print(f"    Features : {len(available_features)}")
    print(f"    X_train  : {X_train.shape}  → (samples, {SEQUENCE_LEN} days, {len(available_features)} features)")
    print(f"    X_test   : {X_test.shape}")

    return summary


# ─────────────────────────────────────────────
# 6. SANITY CHECK — print a few values
# ─────────────────────────────────────────────

def sanity_check(ticker: str):
    """
    Loads saved arrays back and prints basic stats.
    Run this to confirm data looks right before model training.
    """
    safe_name = ticker.replace(".", "_").replace("-", "_")
    X_train = np.load(os.path.join(OUTPUT_DIR, f"{safe_name}_X_train.npy"))
    y_train = np.load(os.path.join(OUTPUT_DIR, f"{safe_name}_y_train.npy"))

    print(f"\n── Sanity Check: {ticker} ─────────────────────")
    print(f"  X_train shape : {X_train.shape}")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  X min / max   : {X_train.min():.4f} / {X_train.max():.4f}  (should be 0–1)")
    print(f"  y min / max   : {y_train.min():.4f} / {y_train.max():.4f}  (should be 0–1)")
    print(f"  No NaNs in X  : {not np.isnan(X_train).any()}")
    print(f"  No NaNs in y  : {not np.isnan(y_train).any()}")
    print(f"  Sample X[0] (first sequence, last day, first 5 features):")
    print(f"    {X_train[0, -1, :5].round(4)}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    TICKERS = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS",
        "HDFCBANK.NS", "WIPRO.NS",
        "BTC-USD", "ETH-USD",
    ]

    all_summaries = []
    for ticker in TICKERS:
        result = prepare_ticker(ticker)
        if result:
            all_summaries.append(result)

    print("\n" + "=" * 55)
    print("All tickers prepared. Files saved in prepared/ folder.")
    print("=" * 55)
    for s in all_summaries:
        print(f"  {s['ticker']:<15}  X_train={s['X_train']}  X_test={s['X_test']}")

    # Run sanity check on first successful ticker
    if all_summaries:
        sanity_check(all_summaries[0]["ticker"])

    print("\nReady for Phase 3: Model Training (LSTM + ARIMA)\n")
