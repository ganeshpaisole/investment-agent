import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from fpdf import FPDF
import pandas as pd

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Principal AI Agent", layout="wide")

# --- 2. USER DATABASE (Define Users & Roles Here) ---
# Format: "username": {"password": "pwd", "role": "admin/viewer"}
USERS = {
    "admin": {
        "password": "Orbittal2025",
        "role": "admin",
        "name": "Principal Consultant"
    },
    "client_demo": {
        "password": "welcome123",
        "role": "viewer",
        "name": "Valued Client"
    },
    "staff": {
        "password": "staff",
        "role": "viewer",
        "name": "Staff Member"
    }
}

# --- 3. SECURITY SYSTEM ---
def check_login():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["user_role"] = None
        st.session_state["user_name"] = None

    if not st.session_state["logged_in"]:
        st.title("🔒 Secure Login System")
        st.markdown("Enter your credentials to access the AI Investment Engine.")
        
        c1, c2 = st.columns(2)
        with c1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
        
            if st.button("Login"):
                # Check if user exists and password matches
                if username in USERS and USERS[username]["password"] == password:
                    st.session_state["logged_in"] = True
                    st.session_state["user_role"] = USERS[username]["role"]
                    st.session_state["user_name"] = USERS[username]["name"]
                    st.success("Login Successful!")
                    st.rerun()
                else:
                    st.error("❌ Access Denied: Invalid Credentials")
        
        st.stop() # Stop app if not logged in

check_login()

# =========================================================
# 🔓 AUTHORIZED ZONE
# =========================================================

# --- 4. ANALYTICS ENGINE (Cached) ---
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

# --- 5. PDF GENERATOR ---
def create_pdf_report(ticker, data, report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Investment Memo: {ticker}", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Price: Rs. {data['price']} | Trend: {data['trend']}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=report_text)
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, txt="DISCLAIMER: Not SEBI Registered. For educational purposes only.")
    return pdf.output(dest='S').encode('latin-1')

# --- 6. DASHBOARD UI ---
st.title(f"👋 Welcome, {st.session_state['user_name']}")

# === SIDEBAR CONTROL PANEL ===
with st.sidebar:
    st.header("Control Panel")
    
    # Show Role Badge
    if st.session_state["user_role"] == "admin":
        st.success("🔑 ADMIN ACCESS")
    else:
        st.info("👁️ VIEWER ACCESS")
        
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()
    st.markdown("---")
    
    # --- LOGIC TO RESTRICT FEATURES ---
    # Admins see BOTH options. Viewers see ONLY "Single Stock".
    if st.session_state["user_role"] == "admin":
        options = ["Single Stock Deep Dive", "Competitor Comparison"]
    else:
        options = ["Single Stock Deep Dive"] # <--- RESTRICTED LIST
        
    mode = st.radio("Select Analysis Mode:", options)

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
            
            verdict = f"Stock is in {metrics['trend']} trend. RSI: {metrics['rsi']:.1f}. P/E: {metrics['pe']}."
            st.info(f"**AI Verdict:** {verdict}")
            
            # Allow PDF download for everyone
            pdf_data = create_pdf_report(ticker, metrics, verdict)
            st.download_button("📄 Download PDF", data=pdf_data, file_name=f"{ticker}_Report.pdf", mime="application/pdf")
        else:
            st.error("Ticker not found.")

# === MODE 2: COMPARISON (Only accessible if selected) ===
elif mode == "Competitor Comparison":
    st.subheader("⚖️ Head-to-Head Benchmarking")
    col1, col2 = st.columns(2)
    with col1: stock_a = st.text_input("Competitor A:", value="TCS.NS")
    with col2: stock_b = st.text_input("Competitor B:", value="INFY.NS")
        
    if st.button("Compare Competitors"):
        data_a, hist_a = analyze_stock(stock_a)
        data_b, hist_b = analyze_stock(stock_b)
        
        if data_a and data_b:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"🔹 {stock_a}")
                st.metric("Score", f"{data_a['score']}/3")
                st.line_chart(hist_a['Close'])
            with c2:
                st.subheader(f"🔸 {stock_b}")
                st.metric("Score", f"{data_b['score']}/3")
                st.line_chart(hist_b['Close'])
            
            st.divider()
            if data_a['score'] > data_b['score']: st.success(f"🏆 WINNER: {stock_a}")
            elif data_b['score'] > data_a['score']: st.success(f"🏆 WINNER: {stock_b}")
            else: st.warning("It's a Tie.")

# SEBI Disclaimer
st.markdown("---")
st.warning("**SEBI DISCLAIMER:** Educational purposes only. Not financial advice.")