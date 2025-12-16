import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from fpdf import FPDF
import pandas as pd

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Strategic Comparison Engine", layout="wide")

# --- 2. SECURITY SYSTEM ---
def check_password():
    """Simple password protection."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    def password_entered():
        if st.session_state["password"] == "Orbittal2025": 
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.text_input("Enter Access Code:", type="password", on_change=password_entered, key="password")
        return False
    return True

if not check_password():
    st.stop()

# --- 3. THE ANALYST ENGINE (With Caching to prevent Errors) ---
@st.cache_data(ttl=24*3600) 
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        if df.empty: return None, None
        
        # Calculate Technicals
        rsi = RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
        ema = EMAIndicator(close=df["Close"], window=200).ema_indicator().iloc[-1]
        current_price = df["Close"].iloc[-1]
        
        # Get Fundamentals (P/E) safely
        pe = stock.info.get('trailingPE', 0)
        if pe is None: pe = 0
        
        # --- THE SCORING LOGIC ---
        # We give 1 point for each positive signal
        score = 0
        if current_price > ema: score += 1      # Uptrend (+1)
        if 40 < rsi < 70: score += 1            # Healthy Momentum (+1)
        if 0 < pe < 40: score += 1              # Undervalued (+1)
        
        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "trend": "BULLISH 🟢" if current_price > ema else "BEARISH 🔴",
            "rsi": round(rsi, 2),
            "pe": pe,
            "score": score
        }, df
    except Exception:
        return None, None

# --- 4. THE COMPARISON DASHBOARD ---
st.title("⚖️ Strategic Comparison Engine")
st.markdown("### Benchmarking Analysis: Competitor A vs. Competitor B")

with st.sidebar:
    st.header("Select Competitors")
    # Two Inputs for comparison
    stock_a = st.text_input("Competitor A (NSE):", value="TATASTEEL.NS")
    stock_b = st.text_input("Competitor B (NSE):", value="JSWSTEEL.NS")
    
    st.caption("Tip: Use .NS for Indian Stocks")
    
    compare_btn = st.button("Run Comparison", type="primary")

if compare_btn:
    with st.spinner("Analyzing Market Data..."):
        # 1. Get Data for BOTH
        data_a, hist_a = analyze_stock(stock_a)
        data_b, hist_b = analyze_stock(stock_b)
        
        if data_a and data_b:
            # 2. Create Two Columns Side-by-Side
            col1, col2 = st.columns(2)
            
            # --- LEFT COLUMN (Stock A) ---
            with col1:
                st.subheader(f"🔹 {stock_a}")
                st.metric("Price", f"₹{data_a['price']}")
                st.metric("Trend", data_a['trend'])
                st.metric("RSI (Momentum)", data_a['rsi'])
                st.metric("P/E Ratio", data_a['pe'])
                st.line_chart(hist_a['Close'])
            
            # --- RIGHT COLUMN (Stock B) ---
            with col2:
                st.subheader(f"🔸 {stock_b}")
                st.metric("Price", f"₹{data_b['price']}")
                st.metric("Trend", data_b['trend'])
                st.metric("RSI (Momentum)", data_b['rsi'])
                st.metric("P/E Ratio", data_b['pe'])
                st.line_chart(hist_b['Close'])
            
            # 3. The Verdict
            st.divider()
            st.subheader("🏆 The Principal's Verdict")
            
            # Compare Scores
            if data_a['score'] > data_b['score']:
                winner = stock_a
                details = f"{stock_a} has a stronger technical score ({data_a['score']}/3)."
            elif data_b['score'] > data_a['score']:
                winner = stock_b
                details = f"{stock_b} has a stronger technical score ({data_b['score']}/3)."
            else:
                winner = "It's a TIE"
                details = "Both companies show similar strength indicators."
            
            st.success(f"**WINNER:** {winner}")
            st.info(f"**Reasoning:** {details} We analyzed Trend Alignment, RSI Momentum, and Valuation.")
            
        else:
            st.error("Could not fetch data. Please check ticker symbols and try again.")