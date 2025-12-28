import os
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

# --- Technical Analysis Functions (Raw Pandas) ---

def calculate_rsi(df, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculates MACD, Signal line, and Histogram."""
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(df, window=20, num_std=2):
    """Calculates Bollinger Bands (Upper, Middle/SMA, Lower)."""
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

# --- Data Fetching ---

def get_stock_data(ticker, period="1y"):
    """Fetches stock data and calculates indicators."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return None, "No data found for ticker."

        # Calculate Indicators
        df['RSI'] = calculate_rsi(df)
        df['MACD'], df['Signal'], df['Hist'] = calculate_macd(df)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df)
        
        return df, None
    except Exception as e:
        return None, str(e)

# --- DeepSeek AI Analysis ---

def get_news_summary(ticker):
    """Fetches top 3 news results using DuckDuckGo."""
    try:
        results = DDGS().text(f"{ticker} stock news", max_results=3)
        if not results:
            return "No recent news found."
        
        news_summary = ""
        for i, res in enumerate(results, 1):
            news_summary += f"{i}. {res['title']}: {res['body']} (Source: {res['href']})\n"
        return news_summary
    except Exception as e:
        return f"Error fetching news: {str(e)}"

def get_deepseek_analysis(ticker, df, api_key):
    # Logic Fix: Use the passed api_key argument, not a global variable
    if not api_key:
        return "⚠️ Please enter an OpenRouter API Key in the sidebar to generate the report."

    # 1. Search Grounding (DuckDuckGo)
    news_summary = ""
    try:
        with st.spinner(f"Searching news for {ticker}..."):
            results = DDGS().text(f"{ticker} stock news analysis finance", max_results=3)
            if results:
                for r in results:
                    news_summary += f"- {r['title']}: {r['body']}\n"
            else:
                news_summary = "No immediate news found."
    except Exception as e:
        news_summary = f"Could not fetch news: {e}"

    # 2. Prepare Data for AI
    current_price = df['Close'].iloc[-1]
    recent_data = df.tail(10)[['Close', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']].to_string()

    prompt = f"""
    Role: Senior Quant & Technical Analyst.
    Task: Analyze {ticker} based on Math + News.
    
    [MARKET DATA]
    Price: ${current_price:.2f}
    Recent Indicators (Last 10 Days):
    {recent_data}

    [LATEST NEWS]
    {news_summary}

    [INSTRUCTIONS]
    1. MATH: Analyze the RSI/MACD divergence. Is the volatility (Bollinger) expanding or contracting?
    2. SYNTHESIS: Combine the math with the news sentiment.
    3. VERDICT: Bullish / Bearish / Neutral (with Probability %).
    4. STRATEGY: Suggest a specific entry/exit based on the math.
    """

    try:
        # Connect to OpenRouter using the user-provided key
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key, 
        )

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            extra_headers={
                "HTTP-Referer": "https://localhost:8501", 
                "X-Title": "StockAnalyzer"
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to OpenRouter: {str(e)}"

# --- Streamlit Dashboard ---

def main():
    st.set_page_config(page_title="Tangible Stock Analyzer", layout="wide")
    
    st.title("📈 Tangible Stock Market Analyzer")
    st.markdown("Powered by **DeepSeek R1** (via OpenRouter) & **DuckDuckGo**")

    # Sidebar Inputs
    with st.sidebar:
        st.header("Settings")
        
        # API Key Handling (User Input Only)
        api_key = st.text_input("OpenRouter API Key", type="password", help="Get your free key at openrouter.ai")
        
        if not api_key:
            st.warning("Enter API Key to enable AI Report")
            
        ticker = st.text_input("Ticker Symbol", value="NVDA").upper()
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
        
        if st.button("Analyze Stock"):
            st.session_state['analyze'] = True

    if st.session_state.get('analyze'):
        with st.spinner(f"Fetching data for {ticker}..."):
            df, error = get_stock_data(ticker, period)
            
        if error:
            st.error(f"Error: {error}")
        else:
            # --- Layout: Graph on Top, AI Report Below ---
            
            # 1. Interactive Graph (Plotly)
            st.subheader(f"Technical Chart: {ticker}")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, subplot_titles=(f"{ticker} Price & Bollinger Bands", "RSI & MACD"),
                                row_width=[0.2, 0.7])

            # Price
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)
            
            # Bollinger Bands
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='rgba(255, 0, 0, 0.3)'), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='rgba(255, 0, 0, 0.3)'), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name='SMA 20', line=dict(color='orange', dash='dash')), row=1, col=1)

            # MACD
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='purple')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='orange')), row=2, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Hist'], name='Histogram'), row=2, col=1)
            
            fig.update_layout(height=800, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, width=None, use_container_width=True)

            # 2. DeepSeek Intelligence Report
            st.subheader("🧠 DeepSeek R1 Intelligence Report")
            
            if api_key:
                with st.spinner("DeepSeek is thinking (Chain of Thought) and searching news..."):
                    analysis = get_deepseek_analysis(ticker, df, api_key)
                st.markdown(analysis)
            else:
                st.info("ℹ️ Enter an OpenRouter API Key in the sidebar to generate the AI analysis.")
            
            # 3. Raw Data (Optional)
            with st.expander("View Raw Data"):
                st.dataframe(df.tail(10))

if __name__ == "__main__":
    main()