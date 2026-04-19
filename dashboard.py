"""
Phase 4: Streamlit Dashboard
Stock / Crypto Price Predictor Project
---------------------------------------
Run AFTER model_training.py
Run:  pip install streamlit plotly tensorflow joblib ta yfinance
Then: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import joblib
import os
import ta
from datetime import datetime, timedelta
def load_lstm_model(ticker: str):
    return None
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Stock & Crypto Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

MODELS_DIR   = "models"
PREPARED_DIR = "prepared"
DATA_DIR     = "data"
SEQUENCE_LEN = 30

TICKER_LABELS = {
    "RELIANCE.NS" : "Reliance Industries",
    "TCS.NS"      : "Tata Consultancy Services",
    "INFY.NS"     : "Infosys",
    "HDFCBANK.NS" : "HDFC Bank",
    "WIPRO.NS"    : "Wipro",
    "BTC-USD"     : "Bitcoin",
    "ETH-USD"     : "Ethereum",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def safe_name(ticker: str) -> str:
    return ticker.replace(".", "_").replace("-", "_")


@st.cache_data(ttl=3600)          # cache 1 hour — don't re-download on every interaction
def fetch_live_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetches fresh OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df.dropna()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"]
    volume = df["Volume"]
    df["MA21"]    = close.rolling(21).mean()
    df["MA50"]    = close.rolling(50).mean()
    df["EMA12"]   = close.ewm(span=12, adjust=False).mean()
    df["EMA26"]   = close.ewm(span=26, adjust=False).mean()
    df["RSI"]     = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd          = ta.trend.MACD(close=close)
    df["MACD"]    = macd.macd()
    df["MACD_sig"]= macd.macd_signal()
    bb            = ta.volatility.BollingerBands(close=close)
    df["BB_upper"]= bb.bollinger_hband()
    df["BB_lower"]= bb.bollinger_lband()
    df["BB_mid"]  = bb.bollinger_mavg()
    df["OBV"]     = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df["daily_return"]   = close.pct_change() * 100
    df["log_return"]     = np.log(close / close.shift(1))
    df["volatility_21d"] = df["log_return"].rolling(21).std() * np.sqrt(252)
    df["price_range"]    = (df["High"] - df["Low"]) / close
    df["MA7"]            = close.rolling(7).mean()
    df["MA200"]          = close.rolling(200).mean()
    return df


@st.cache_resource                 # cache model in memory across reruns
def load_lstm_model(ticker: str):
    path = os.path.join(MODELS_DIR, f"{safe_name(ticker)}_lstm.keras")
    if os.path.exists(path):
        return load_model(path)
    return None


def load_scaler(ticker: str):
    path = os.path.join(PREPARED_DIR, f"{safe_name(ticker)}_scaler_close.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_scaler_all(ticker: str):
    path = os.path.join(PREPARED_DIR, f"{safe_name(ticker)}_scaler_all.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


FEATURE_COLS = [
    "Close", "Volume", "daily_return", "log_return",
    "MA7", "MA21", "MA50", "EMA12", "EMA26",
    "volatility_21d", "price_range",
    "RSI", "MACD", "MACD_sig", "BB_upper", "BB_lower", "BB_width",
    "OBV",
]


def prepare_latest_sequence(df: pd.DataFrame, scaler_all, feature_cols: list) -> np.ndarray:
    """Prepares the most recent 30-day window for live prediction."""
    df = df.copy()
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
    df["MACD_sig"] = df["MACD_sig"] if "MACD_sig" in df.columns else df.get("MACD_signal", 0)

    available = [c for c in feature_cols if c in df.columns]
    df_feat   = df[available].dropna().tail(SEQUENCE_LEN)

    if len(df_feat) < SEQUENCE_LEN:
        return None

    scaled = scaler_all.transform(df_feat)
    return scaled.reshape(1, SEQUENCE_LEN, len(available))


def predict_next_days(model, df: pd.DataFrame, scaler_all, scaler_close,
                      feature_cols: list, days: int = 7) -> list:
    """
    Iterative multi-step forecast: predict day 1, append to history,
    predict day 2, and so on. Returns list of predicted prices.
    """
    df_work = df.copy()
    preds   = []

    for _ in range(days):
        df_work = add_indicators(df_work)
        seq     = prepare_latest_sequence(df_work, scaler_all, feature_cols)
        if seq is None:
            break
        pred_scaled = model.predict(seq, verbose=0)
        pred_price  = scaler_close.inverse_transform(pred_scaled)[0][0]
        preds.append(pred_price)

        # Append a synthetic row so the next iteration has one more real-ish day
        new_row = df_work.iloc[[-1]].copy()
        new_row.index = [df_work.index[-1] + timedelta(days=1)]
        new_row["Close"] = pred_price
        new_row["Open"]  = pred_price
        new_row["High"]  = pred_price * 1.005
        new_row["Low"]   = pred_price * 0.995
        df_work = pd.concat([df_work, new_row[["Open", "High", "Low", "Close", "Volume"]]])

    return preds


def get_signal(current: float, predicted: float) -> tuple:
    """Returns (signal, confidence, color) based on predicted return."""
    ret = (predicted - current) / current * 100
    conf = min(abs(ret) / 5.0, 1.0) * 100

    if ret > 1.0:
        return "BUY",  round(conf, 1), "#22c55e"
    elif ret < -1.0:
        return "SELL", round(conf, 1), "#ef4444"
    else:
        return "HOLD", round(conf, 1), "#f59e0b"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Predictor")
    st.markdown("---")

    ticker = st.selectbox(
        "Select asset",
        options=list(TICKER_LABELS.keys()),
        format_func=lambda t: f"{TICKER_LABELS[t]} ({t})",
    )

    period = st.selectbox(
        "Chart period",
        options=["3mo", "6mo", "1y", "2y", "5y"],
        index=2,
    )

    forecast_days = st.slider("Forecast days", min_value=1, max_value=30, value=7)

    model_choice = "ARIMA (Classical)"

    st.markdown("---")
    st.caption("Data: Yahoo Finance  |  Refresh: every 1h")
    st.caption("⚠️ For educational use only. Not financial advice.")


# ─────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────

with st.spinner(f"Fetching live data for {ticker} ..."):
    df_raw = fetch_live_data(ticker, period)
    df     = add_indicators(df_raw.copy())

model       = load_lstm_model(ticker)
scaler_close= load_scaler(ticker)
scaler_all  = load_scaler_all(ticker)
model_ready = (model is not None and scaler_close is not None and scaler_all is not None)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

latest_close = float(df["Close"].iloc[-1])
prev_close   = float(df["Close"].iloc[-2])
day_change   = latest_close - prev_close
day_change_p = day_change / prev_close * 100
arrow        = "▲" if day_change >= 0 else "▼"
color_day    = "green" if day_change >= 0 else "red"

st.title(f"{TICKER_LABELS[ticker]}")
st.markdown(
    f"**{ticker}** &nbsp;|&nbsp; "
    f"Last close: **₹{latest_close:,.2f}** &nbsp; "
    f":{color_day}[{arrow} {abs(day_change_p):.2f}% today]"
)

# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Live Chart", "🔮 Forecast", "📡 Signal"])


# ══════════════════════════════════════════════
# TAB 1 — CANDLESTICK CHART
# ══════════════════════════════════════════════

with tab1:
    st.subheader("Price Chart")

    show_ma   = st.toggle("Moving Averages", value=True)
    show_bb   = st.toggle("Bollinger Bands", value=False)
    show_vol  = st.toggle("Volume", value=True)

    rows   = 3 if show_vol else 2
    heights= [0.55, 0.25, 0.20] if show_vol else [0.65, 0.35]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.04,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
    ), row=1, col=1)

    # Moving averages
    if show_ma:
        for ma, color in [("MA21","#f59e0b"),("MA50","#60a5fa"),("MA200","#a78bfa")]:
            if ma in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[ma], name=ma,
                    line=dict(color=color, width=1.2),
                ), row=1, col=1)

    # Bollinger Bands
    if show_bb:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color="#94a3b8", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color="#94a3b8", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(148,163,184,0.1)",
        ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#f472b6", width=1.2),
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444",
                  line_width=0.8, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e",
                  line_width=0.8, row=2, col=1)

    # Volume
    if show_vol:
        vol_colors = ["#22c55e" if r >= 0 else "#ef4444"
                      for r in df["daily_return"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=vol_colors, opacity=0.6,
        ), row=3, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_yaxes(title_text="Price",  row=1, col=1)
    fig.update_yaxes(title_text="RSI",    row=2, col=1)
    if show_vol:
        fig.update_yaxes(title_text="Volume", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Quick stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("52W High",  f"₹{df['Close'].max():,.2f}")
    c2.metric("52W Low",   f"₹{df['Close'].min():,.2f}")
    c3.metric("RSI (14)",  f"{df['RSI'].iloc[-1]:.1f}")
    c4.metric("Volatility", f"{df['volatility_21d'].iloc[-1]*100:.1f}% ann.")


# ══════════════════════════════════════════════
# TAB 2 — FORECAST
# ══════════════════════════════════════════════

with tab2:
    st.subheader(f"{forecast_days}-Day Price Forecast")

    if model_choice.startswith("LSTM"):
        if not model_ready:
            st.warning(
                f"No trained LSTM model found for {ticker}. "
                "Run `model_training.py` first, then come back."
            )
        else:
            with st.spinner("Running LSTM forecast ..."):
                future_prices = predict_next_days(
                    model, df_raw, scaler_all, scaler_close,
                    FEATURE_COLS, days=forecast_days
                )

            if future_prices:
                future_dates = pd.date_range(
                    start=df.index[-1] + timedelta(days=1),
                    periods=len(future_prices), freq="B"
                )

                # Chart: last 60 days actual + forecast
                hist = df["Close"].tail(60)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=hist.index, y=hist.values,
                    name="Actual", line=dict(color="#60a5fa", width=2),
                ))
                fig2.add_trace(go.Scatter(
                    x=future_dates, y=future_prices,
                    name="Forecast", line=dict(color="#f59e0b", width=2, dash="dash"),
                    mode="lines+markers",
                ))
                # Confidence band (±2% simple estimate)
                upper = [p * 1.02 for p in future_prices]
                lower = [p * 0.98 for p in future_prices]
                fig2.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates[::-1]),
                    y=upper + lower[::-1],
                    fill="toself", fillcolor="rgba(245,158,11,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="±2% band", showlegend=True,
                ))

                fig2.update_layout(
                    template="plotly_dark", height=400,
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.01),
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Forecast table
                st.markdown("**Forecast values**")
                forecast_df = pd.DataFrame({
                    "Date"           : future_dates.strftime("%d %b %Y"),
                    "Predicted Price": [f"₹{p:,.2f}" for p in future_prices],
                    "Change from today": [
                        f"{((p - latest_close)/latest_close*100):+.2f}%"
                        for p in future_prices
                    ],
                })
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    else:
        # ARIMA predictions from saved CSV
        arima_path = os.path.join(MODELS_DIR, f"{safe_name(ticker)}_arima_preds.csv")
        if not os.path.exists(arima_path):
            st.warning("No ARIMA predictions found. Run `model_training.py` first.")
        else:
            arima_df = pd.read_csv(arima_path)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                y=arima_df["actual"].values, name="Actual",
                line=dict(color="#60a5fa", width=1.5),
            ))
            fig3.add_trace(go.Scatter(
                y=arima_df["predicted"].values, name="ARIMA Predicted",
                line=dict(color="#f59e0b", width=1.5, dash="dash"),
            ))
            fig3.update_layout(
                template="plotly_dark", height=400,
                xaxis_title="Test days", yaxis_title="Price",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("Showing ARIMA rolling forecast on the held-out test set.")


# ══════════════════════════════════════════════
# TAB 3 — BUY / SELL / HOLD SIGNAL
# ══════════════════════════════════════════════

with tab3:
    st.subheader("Trading Signal")

    if not model_ready:
        st.warning(f"Train the LSTM model for {ticker} to see signals.")
    else:
        with st.spinner("Generating signal ..."):
            next_day = predict_next_days(
                model, df_raw, scaler_all, scaler_close,
                FEATURE_COLS, days=1
            )

        if next_day:
            predicted_tomorrow = next_day[0]
            signal, confidence, sig_color = get_signal(latest_close, predicted_tomorrow)

            # Big signal display
            st.markdown(
                f"""
                <div style="
                    background: {sig_color}22;
                    border: 2px solid {sig_color};
                    border-radius: 12px;
                    padding: 2rem;
                    text-align: center;
                    margin-bottom: 1.5rem;
                ">
                    <div style="font-size: 3rem; font-weight: 700;
                                color: {sig_color};">{signal}</div>
                    <div style="font-size: 1rem; color: #94a3b8; margin-top: 0.5rem;">
                        Confidence: {confidence:.0f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price",   f"₹{latest_close:,.2f}")
            col2.metric("Predicted (1d)",  f"₹{predicted_tomorrow:,.2f}",
                        delta=f"{((predicted_tomorrow-latest_close)/latest_close*100):+.2f}%")
            col3.metric("Signal Strength", f"{confidence:.0f}%")

            st.markdown("---")
            st.markdown("**What's driving this signal?**")

            rsi_val   = df["RSI"].iloc[-1]
            macd_val  = df["MACD"].iloc[-1]
            macd_sig  = df["MACD_sig"].iloc[-1] if "MACD_sig" in df.columns else df.get("MACD_signal", pd.Series([0])).iloc[-1]
            close_val = df["Close"].iloc[-1]
            bb_upper  = df["BB_upper"].iloc[-1]
            bb_lower  = df["BB_lower"].iloc[-1]

            reasons = []
            if rsi_val > 70:
                reasons.append(f"RSI is {rsi_val:.1f} — overbought zone (>70), suggests caution")
            elif rsi_val < 30:
                reasons.append(f"RSI is {rsi_val:.1f} — oversold zone (<30), potential rebound")
            else:
                reasons.append(f"RSI is {rsi_val:.1f} — neutral zone")

            if macd_val > macd_sig:
                reasons.append("MACD is above signal line — bullish momentum")
            else:
                reasons.append("MACD is below signal line — bearish momentum")

            if close_val > bb_upper * 0.98:
                reasons.append("Price near upper Bollinger Band — may be overextended")
            elif close_val < bb_lower * 1.02:
                reasons.append("Price near lower Bollinger Band — potential support")

            for r in reasons:
                st.markdown(f"• {r}")

            # Historical signals from saved CSV
            signals_path = os.path.join(MODELS_DIR, f"{safe_name(ticker)}_signals.csv")
            if os.path.exists(signals_path):
                st.markdown("---")
                st.markdown("**Recent historical signals (test period)**")
                sig_df = pd.read_csv(signals_path).tail(10)
                st.dataframe(sig_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Built with Python · TensorFlow/Keras · Streamlit · Plotly · yfinance  |  "
    "⚠️ This is a student portfolio project. Not financial advice."
)
