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

# --- 3. THE ANALYST ENGINE (Now with Caching!) ---
@st.cache_data(ttl=24*3600) # <--- THIS LINE FIXES THE ERROR (Caches data for 24 hours)
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Fetch 1 year of data
        df = stock.history(period="1y")
        
        if df.empty: return None, None
        
        # Calculate Metrics
        rsi = RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
        ema = EMAIndicator(close=df["Close"], window=200).ema_indicator().iloc[-1]
        current_price = df["Close"].iloc[-1]
        
        # Get P/E (handle missing data safely)
        pe = stock.info.get('trailingPE', 0)
        if pe is None: pe = 0
        
        # Simple "Score" for Winner Logic
        score = 0
        if current_price > ema: score += 1      # Point for Uptrend
        if 30 < rsi < 70: score += 1            # Point for Safe Momentum
        if 0 < pe < 40: score += 1              # Point for Good Value
        
        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "trend": "BULLISH 🟢" if current_price > ema else "BEARISH 🔴",
            "rsi": round(rsi, 2),
            "pe": pe,
            "score": score
        }, df
    except Exception as e:
        return None, None

# --- 4. THE DASHBOARD UI ---
st.title("⚖️ Strategic Comparison Engine")
st.markdown("### Benchmarking Analysis: Company A vs. Company B")

with st.sidebar:
    st.header("Select Competitors")
    # Using TATASTEEL and JSWSTEEL as defaults to test
    stock_a = st.text_input("Competitor A (NSE):", value="TATASTEEL.NS")
    stock_b = st.text_input("Competitor B (NSE):", value="JSWSTEEL.NS")
    
    # Add a "Clear Cache" button in case data gets stuck
    if st.button("Refresh Data"):
        st.cache_data.clear()
        
    compare_btn = st.button("Run Comparison", type="primary")

if compare_btn:
    with st.spinner("Analyzing Market Data..."):
        # Analyze BOTH stocks
        data_a, hist_a = analyze_stock(stock_a)
        data_b, hist_b = analyze_stock(stock_b)
        
        if data_a and data_b:
            # --- DISPLAY SIDE-BY-SIDE COLUMNS ---
            col1, col2 = st.columns(2)
            
            # Stock A Column
            with col1:
                st.subheader(f"🔹 {stock_a}")
                st.metric("Price", f"₹{data_a['price']}")
                st.metric("Trend", data_a['trend'])
                st.metric("P/E Ratio", data_a['pe'])
                st.line_chart(hist_a['Close'])
            
            # Stock B Column
            with col2:
                st.subheader(f"🔸 {stock_b}")
                st.metric("Price", f"₹{data_b['price']}")
                st.metric("Trend", data_b['trend'])
                st.metric("P/E Ratio", data_b['pe'])
                st.line_chart(hist_b['Close'])
            
            # --- THE "WINNER" VERDICT ---
            st.divider()
            st.subheader("🏆 The Principal's Verdict")
            
            if data_a['score'] > data_b['score']:
                winner = stock_a
                reason = "better technical momentum and valuation structure."
            elif data_b['score'] > data_a['score']:
                winner = stock_b
                reason = "superior fundamentals and stronger trend alignment."
            else:
                winner = "TIE"
                reason = "both companies showing similar strength scores."
            
            st.success(f"**WINNER:** {winner}")
            st.info(f"**Reasoning:** Based on our 3-point scoring system (Trend, RSI, Valuation), {winner} currently demonstrates {reason}")
            
        else:
            st.error("⚠️ Yahoo Finance is busy or Ticker is invalid. Please click 'Refresh Data' or try again in 1 minute.")