import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from fpdf import FPDF
import pandas as pd

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Principal AI Agent", layout="wide")

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

# --- 3. ANALYTICS ENGINE (Cached) ---
@st.cache_data(ttl=24*3600)
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if df.empty: return None, None
        
        rsi = RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
        ema = EMAIndicator(close=df["Close"], window=200).ema_indicator().iloc[-1]
        current_price = df["Close"].iloc[-1]
        pe = stock.info.get('trailingPE', 0)
        if pe is None: pe = 0
        
        score = 0
        if current_price > ema: score += 1
        if 30 < rsi < 70: score += 1
        if 0 < pe < 40: score += 1
        
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

# --- 4. PDF GENERATOR (For Single Mode) ---
def create_pdf_report(ticker, data, report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Investment Memo: {ticker}", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Price: Rs. {data['price']} | Trend: {data['trend']}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=report_text)
    return pdf.output(dest='S').encode('latin-1')

# --- 5. DASHBOARD UI ---
st.title("🤖 Principal Consultant's Dashboard")

# === SIDEBAR MENU ===
with st.sidebar:
    st.header("Control Panel")
    # This Switcher controls the entire app view!
    mode = st.radio("Select Analysis Mode:", ["Single Stock Deep Dive", "Competitor Comparison"])
    st.markdown("---")

# === MODE 1: SINGLE STOCK ===
if mode == "Single Stock Deep Dive":
    st.subheader("🔍 Deep Dive Analysis")
    ticker = st.text_input("Enter Ticker (NSE):", value="RELIANCE.NS")
    
    if st.button("Generate Report"):
        metrics, history = analyze_stock(ticker)
        if metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Price", f"₹{metrics['price']}")
            c2.metric("Trend", metrics['trend'])
            c3.metric("RSI", f"{metrics['rsi']:.1f}")
            
            st.line_chart(history['Close'])
            
            # Smart Logic
            verdict = f"The stock is currently in a {metrics['trend']} trend. With an RSI of {metrics['rsi']}, momentum is {'strong' if metrics['rsi']>50 else 'weak'}. Valuation P/E is {metrics['pe']}."
            st.info(f"**AI Verdict:** {verdict}")
            
            # PDF Button
            pdf_data = create_pdf_report(ticker, metrics, verdict)
            st.download_button("📄 Download PDF", data=pdf_data, file_name=f"{ticker}_Report.pdf", mime="application/pdf")
        else:
            st.error("Ticker not found.")

# === MODE 2: COMPARISON ===
elif mode == "Competitor Comparison":
    st.subheader("⚖️ Head-to-Head Benchmarking")
    col1, col2 = st.columns(2)
    with col1:
        stock_a = st.text_input("Competitor A:", value="TCS.NS")
    with col2:
        stock_b = st.text_input("Competitor B:", value="INFY.NS")
        
    if st.button("Compare Competitors"):
        data_a, hist_a = analyze_stock(stock_a)
        data_b, hist_b = analyze_stock(stock_b)
        
        if data_a and data_b:
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader(f"🔹 {stock_a}")
                st.metric("Price", f"₹{data_a['price']}")
                st.metric("Score", f"{data_a['score']}/3")
                st.line_chart(hist_a['Close'])
                
            with c2:
                st.subheader(f"🔸 {stock_b}")
                st.metric("Price", f"₹{data_b['price']}")
                st.metric("Score", f"{data_b['score']}/3")
                st.line_chart(hist_b['Close'])
            
            # Winner Logic
            st.divider()
            if data_a['score'] > data_b['score']:
                st.success(f"🏆 **WINNER:** {stock_a} (Stronger Technicals)")
            elif data_b['score'] > data_a['score']:
                st.success(f"🏆 **WINNER:** {stock_b} (Stronger Technicals)")
            else:
                st.warning("⚖️ **RESULT:** It's a Tie.")