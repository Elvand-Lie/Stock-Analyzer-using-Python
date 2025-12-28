import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Load variables from .env if present
MODEL_ID = "tngtech/deepseek-r1t2-chimera:free"

# --- Helper: Smart Key Loader ---
def get_api_key():
    """
    Tries to find the API key in this order:
    1. Streamlit Secrets (.streamlit/secrets.toml)
    2. Environment Variables (.env)
    3. Returns None if not found (triggers manual input)
    """
    # Check Streamlit Secrets
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"], "Secrets File"
    except FileNotFoundError:
        pass # No secrets file found, continue
        
    # Check OS Environment (.env)
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key, "Environment (.env)"
        
    return None, None

# --- Technical Analysis Functions ---
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
def get_deepseek_analysis(ticker, df, api_key):
    if not api_key:
        return "⚠️ Please enter an API Key to see the report."

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
            api_key=api_key, 
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
        
        # [LOGIC FIX] 1. Try to load key automatically
        api_key, source = get_api_key()
        
        # 2. If found, show success message
        if api_key:
            st.success(f"✅ Key Loaded from {source}")
            
        # 3. If NOT found, show the manual input box
        else:
            st.warning("No .env or secrets found.")
            api_key = st.text_input("OpenRouter API Key", type="password")
            
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
            if api_key:
                with st.spinner("Analyzing..."):
                    st.markdown(get_deepseek_analysis(ticker, df, api_key))
            else:
                st.warning("⚠️ Enter API Key in sidebar to see the AI report.")

if __name__ == "__main__":
    main()