"""
Phase 1: Data Collection & Cleaning
Stock / Crypto Price Predictor Project
--------------------------------------
Run:  pip install yfinance pandas numpy plotly ta
Then: python data_collection.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# 1. CONFIG — edit these to your liking
# ─────────────────────────────────────────────

STOCKS = [
    "RELIANCE.NS",   # Reliance Industries
    "TCS.NS",        # Tata Consultancy Services
    "INFY.NS",       # Infosys
    "HDFCBANK.NS",   # HDFC Bank
    "WIPRO.NS",      # Wipro
]

CRYPTOS = [
    "BTC-USD",       # Bitcoin
    "ETH-USD",       # Ethereum
]

ALL_TICKERS  = STOCKS + CRYPTOS
START_DATE   = "2019-01-01"
END_DATE     = datetime.today().strftime("%Y-%m-%d")
DATA_DIR     = "data"          # folder where CSVs will be saved


# ─────────────────────────────────────────────
# 2. FETCH RAW DATA
# ─────────────────────────────────────────────

def fetch_data(tickers: list, start: str, end: str) -> dict:
    """
    Download OHLCV data for all tickers.
    Returns a dict: { ticker: DataFrame }
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    data = {}

    for ticker in tickers:
        print(f"  Fetching {ticker} ...")
        df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            print(f"  WARNING: No data returned for {ticker}. Skipping.")
            continue

        # yfinance sometimes returns MultiIndex columns — flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        data[ticker] = df
        print(f"    {len(df)} rows  |  {df.index[0].date()} → {df.index[-1].date()}")

    return data


# ─────────────────────────────────────────────
# 3. CLEAN DATA
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Cleans a single ticker's OHLCV DataFrame:
      - Drops duplicate dates
      - Forward-fills small gaps (weekends/holidays ≤ 3 days)
      - Drops rows still missing after fill
      - Removes obvious price errors (zero or negative prices)
      - Resets to a clean DatetimeIndex
    """
    df = df.copy()

    # Remove duplicate index entries
    df = df[~df.index.duplicated(keep="first")]

    # Sort chronologically (always!)
    df = df.sort_index()

    # Drop rows where Close is zero or negative — bad data
    before = len(df)
    df = df[df["Close"] > 0]
    dropped = before - len(df)
    if dropped:
        print(f"  [{ticker}] Dropped {dropped} rows with invalid Close price")

    # Forward-fill gaps up to 3 consecutive trading days (holiday gaps)
    df = df.ffill(limit=3)

    # Drop any remaining NaNs
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    return df


# ─────────────────────────────────────────────
# 4. ADD FEATURES (Moving Averages + Returns)
# ─────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds commonly used derived columns used as model features later:
      - Daily return %
      - Log return (better for modelling — normally distributed)
      - Simple moving averages: MA7, MA21, MA50, MA200
      - Exponential moving average: EMA12, EMA26
      - Volatility: rolling 21-day standard deviation of log returns
      - Price range: (High - Low) / Close
    """
    df = df.copy()

    close = df["Close"]

    # Returns
    df["daily_return"]  = close.pct_change() * 100          # % change
    df["log_return"]    = np.log(close / close.shift(1))    # log return

    # Simple Moving Averages
    for window in [7, 21, 50, 200]:
        df[f"MA{window}"] = close.rolling(window=window).mean()

    # Exponential Moving Averages (used in MACD)
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()

    # Volatility — rolling std of log returns (annualised)
    df["volatility_21d"] = df["log_return"].rolling(21).std() * np.sqrt(252)

    # Intraday price range (useful feature for models)
    df["price_range"] = (df["High"] - df["Low"]) / close

    return df


# ─────────────────────────────────────────────
# 5. SAVE TO CSV
# ─────────────────────────────────────────────

def save_csv(data: dict):
    """Saves each ticker's cleaned DataFrame to data/<TICKER>.csv"""
    for ticker, df in data.items():
        # Replace characters that are invalid in filenames
        safe_name = ticker.replace(".", "_").replace("-", "_")
        path = os.path.join(DATA_DIR, f"{safe_name}.csv")
        df.to_csv(path)
        print(f"  Saved → {path}  ({len(df)} rows, {len(df.columns)} columns)")


# ─────────────────────────────────────────────
# 6. QUICK VISUALISATION — Candlestick + Volume
# ─────────────────────────────────────────────

def plot_candlestick(df: pd.DataFrame, ticker: str, last_n_days: int = 180):
    """
    Plots an interactive candlestick chart with volume bars and MA lines.
    Opens in your browser automatically.
    """
    df_plot = df.tail(last_n_days)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot["Open"],
        high=df_plot["High"],
        low=df_plot["Low"],
        close=df_plot["Close"],
        name="Price",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
    ), row=1, col=1)

    # Moving averages
    for ma, color in [("MA21", "#f59e0b"), ("MA50", "#60a5fa"), ("MA200", "#a78bfa")]:
        if ma in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot[ma],
                name=ma, line=dict(color=color, width=1.2),
            ), row=1, col=1)

    # Volume bars
    colors = ["#22c55e" if r >= 0 else "#ef4444"
              for r in df_plot["daily_return"].fillna(0)]
    fig.add_trace(go.Bar(
        x=df_plot.index, y=df_plot["Volume"],
        name="Volume", marker_color=colors, opacity=0.6,
    ), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — Last {last_n_days} Days",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.show()


# ─────────────────────────────────────────────
# 7. CORRELATION HEATMAP (across assets)
# ─────────────────────────────────────────────

def plot_correlation_heatmap(data: dict):
    """
    Plots a heatmap of daily log-return correlations across all tickers.
    Useful for understanding how assets move together.
    """
    # Align all tickers on the same date index using log returns
    returns = pd.DataFrame({
        ticker: df["log_return"]
        for ticker, df in data.items()
        if "log_return" in df.columns
    })

    corr = returns.corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        colorbar=dict(title="Correlation"),
    ))

    fig.update_layout(
        title="Log-Return Correlation Across Assets",
        template="plotly_dark",
        height=500,
    )
    fig.show()


# ─────────────────────────────────────────────
# 8. SUMMARY STATS
# ─────────────────────────────────────────────

def print_summary(data: dict):
    """Prints a clean summary table for all tickers."""
    print("\n" + "=" * 65)
    print(f"{'Ticker':<15} {'Rows':>6} {'Start':>12} {'End':>12} {'Last Close':>12}")
    print("=" * 65)
    for ticker, df in data.items():
        print(
            f"{ticker:<15} {len(df):>6} "
            f"{str(df.index[0].date()):>12} "
            f"{str(df.index[-1].date()):>12} "
            f"{df['Close'].iloc[-1]:>11.2f}"
        )
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Step 1: Fetching data ──────────────────────")
    raw_data = fetch_data(ALL_TICKERS, START_DATE, END_DATE)

    print("\n── Step 2: Cleaning & adding features ────────")
    clean = {}
    for ticker, df in raw_data.items():
        df = clean_data(df, ticker)
        df = add_features(df)
        clean[ticker] = df
        print(f"  [{ticker}] Cleaned → {len(df)} rows, {len(df.columns)} columns")

    print("\n── Step 3: Saving CSVs ───────────────────────")
    save_csv(clean)

    print_summary(clean)

    # ── Visualise (comment out if running headless) ──
    # Pick one ticker to plot
    demo_ticker = "TCS.NS"
    if demo_ticker in clean:
        print(f"── Step 4: Plotting {demo_ticker} candlestick ──")
        plot_candlestick(clean[demo_ticker], demo_ticker, last_n_days=180)

    print("── Step 5: Correlation heatmap ───────────────")
    plot_correlation_heatmap(clean)

    print("\nDone! Check the data/ folder for your CSVs.\n")
