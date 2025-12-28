import time
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from duckduckgo_search import DDGS

# --- Configuration ---
MODEL_ID = "tngtech/deepseek-r1t2-chimera:free"

# [CHANGED] Try to load key from Streamlit Secrets
try:
    API_KEY = st.secrets["OPENROUTER_API_KEY"]
except FileNotFoundError:
    API_KEY = None
    # Use a placeholder or handle the error gracefully if running locally without the file
except KeyError:
    API_KEY = None

# --- Technical Analysis Functions (Same as before) ---
def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(df, window=20, num_std=2):
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

# --- Data Fetching ---
def get_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty: return None, "No data found."
        
        df['RSI'] = calculate_rsi(df)
        df['MACD'], df['Signal'], df['Hist'] = calculate_macd(df)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df)
        return df, None
    except Exception as e:
        return None, str(e)

# --- DeepSeek AI Analysis ---
def get_deepseek_analysis(ticker, df):
    # [CHANGED] Uses the global API_KEY retrieved from st.secrets
    if not API_KEY:
        return "⚠️ API Key missing. Please set OPENROUTER_API_KEY in .streamlit/secrets.toml"

    news_summary = ""
    try:
        results = DDGS().text(f"{ticker} stock news analysis finance", max_results=3)
        if results:
            for r in results:
                news_summary += f"- {r['title']}: {r['body']}\n"
        else:
            news_summary = "No immediate news found."
    except Exception as e:
        news_summary = f"Could not fetch news: {e}"

    current_price = df['Close'].iloc[-1]
    recent_data = df.tail(10)[['Close', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']].to_string()

    prompt = f"""
    Role: Senior Quant. Analyze {ticker}.
    
    [DATA]
    Price: ${current_price:.2f}
    Last 10 Days:
    {recent_data}

    [NEWS]
    {news_summary}

    [TASK]
    Provide a Bullish/Bearish verdict, probability, and trade setup based on indicators + news.
    """

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY, 
        )
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            extra_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "StockAnalyzer"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# --- Dashboard ---
def main():
    st.set_page_config(page_title="Tangible Stock Analyzer", layout="wide")
    st.title("📈 Tangible Stock Market Analyzer")

    with st.sidebar:
        st.header("Settings")
        
        # [CHANGED] Check if secrets loaded successfully
        if API_KEY:
            st.success("✅ API Key Loaded from Secrets")
        else:
            st.error("❌ Key Missing (.streamlit/secrets.toml)")
            
        ticker = st.text_input("Ticker", value="NVDA").upper()
        if st.button("Analyze"):
            st.session_state['analyze'] = True

    if st.session_state.get('analyze'):
        df, error = get_stock_data(ticker)
        if error:
            st.error(error)
        else:
            # Graph
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.2, 0.7])
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], fill='tonexty', line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

            # AI Report
            st.subheader("🧠 DeepSeek Analysis")
            if API_KEY:
                with st.spinner("Analyzing..."):
                    st.markdown(get_deepseek_analysis(ticker, df))
            else:
                st.warning("Configure secrets.toml to see AI analysis.")

if __name__ == "__main__":
    main()