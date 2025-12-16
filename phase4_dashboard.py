import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
import pandas as pd

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Investment Agent", layout="wide")

st.title("🤖 AI Investment Principal Agent")
st.markdown("Enter an Indian Stock Ticker to generate an institutional-grade memo.")

# --- 2. SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.header("Configuration")
    ticker_input = st.text_input("Enter Ticker (NSE):", value="TITAN.NS")
    
    st.divider()
    
    ai_mode = st.radio("AI Mode:", ["Simulation (Free)", "Real AI (Requires Key)"])
    
    api_key = ""
    if ai_mode == "Real AI (Requires Key)":
        api_key = st.text_input("Enter OpenAI/Gemini Key:", type="password")
        st.caption("Your key is not saved anywhere.")

# --- 3. THE ANALYST ENGINE (Hidden Logic) ---
def analyze_stock(ticker):
    # Fetch Data
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    
    if df.empty: return None, None
    
    # Calculate Indicators
    rsi = RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
    ema = EMAIndicator(close=df["Close"], window=200).ema_indicator().iloc[-1]
    current_price = df["Close"].iloc[-1]
    
    # Simple Logic
    trend = "BULLISH 🟢" if current_price > ema else "BEARISH 🔴"
    valuation = stock.info.get('trailingPE', 0)
    
    data = {
        "price": current_price,
        "trend": trend,
        "rsi": rsi,
        "pe": valuation
    }
    return data, df

def write_report(data, ticker, use_real_ai, key):
    prompt = f"""
    Analyze {ticker}.
    Price: {data['price']:.2f} | Trend: {data['trend']} | RSI: {data['rsi']:.2f} | P/E: {data['pe']}
    Write a 3-bullet investment verdict.
    """
    
    if use_real_ai and key:
        try:
            # Try connecting to AI (Using LangChain logic simply here)
            # Note: For simplicity in this specific file, we simulate the "Call" 
            # to avoid complex library imports if you haven't set them up perfectly.
            return f"🤖 (AI Response Placeholder) The stock {ticker} shows a {data['trend']} trend. RSI is {data['rsi']:.1f}. Recommendation: Watch carefully." 
        except:
            return "Error connecting to AI. Check Key."
    else:
        # Simulation Mode
        return f"""
        **SIMULATED VERDICT:**
        * **Trend:** The stock is currently {data['trend']}.
        * **Momentum:** RSI is at {data['rsi']:.1f}.
        * **Valuation:** P/E ratio is {data['pe']}.
        *(Select 'Real AI' in sidebar to use ChatGPT/Gemini)*
        """

# --- 4. THE MAIN DASHBOARD UI ---
if st.button("Generate Analysis", type="primary"):
    with st.spinner(f"Analyzing {ticker_input}..."):
        
        # A. Run Analysis
        metrics, history_df = analyze_stock(ticker_input)
        
        if metrics:
            # B. Display Top Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"₹{metrics['price']:.2f}")
            col2.metric("Trend", metrics['trend'])
            col3.metric("RSI (Momentum)", f"{metrics['rsi']:.1f}")
            col4.metric("P/E Ratio", metrics['pe'])
            
            # C. Show Chart
            st.subheader("Price Chart (1 Year)")
            st.line_chart(history_df['Close'])
            
            # D. AI Memo
            st.subheader("📝 Principal's Memo")
            report = write_report(metrics, ticker_input, ai_mode == "Real AI (Requires Key)", api_key)
            st.info(report)
            
        else:
            st.error("Ticker not found. Please try 'RELIANCE.NS' or 'TCS.NS'")