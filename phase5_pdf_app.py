import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from fpdf import FPDF
import pandas as pd

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Principal AI Agent", layout="wide")

# --- 2. SECURITY SYSTEM (Username + Password) ---
def check_login():
    """Authenticates the user with Username & Password."""
    # Initialize session state for login
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # Login Logic
    if not st.session_state["logged_in"]:
        st.title("🔒 Consultant Login")
        st.markdown("Please sign in to access the Presales Dashboard.")
        
        # Input fields
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            # <--- SET YOUR CREDENTIALS HERE --->
            if username == "admin" and password == "Orbittal2025":
                st.session_state["logged_in"] = True
                st.rerun() # Refresh to show the dashboard
            else:
                st.error("❌ Invalid Username or Password")
        
        # Stop the app here if not logged in
        st.stop()

# Run the login check before anything else
check_login()

# =========================================================
# 🔓 SECURE ZONE: App only loads below this line
# =========================================================

# --- 3. SEBI DISCLAIMER (Compliance) ---
def show_disclaimer():
    st.markdown("---")
    st.warning(
        """
        **⚠️ SEBI DISCLAIMER & COMPLIANCE NOTICE:**
        
        1. **Not a SEBI Registered Advisor:** This AI tool is for **educational and presales demonstration purposes only**. It does not constitute financial advice, investment recommendations, or a solicitation to buy/sell any securities.
        2. **Market Risk:** Investments in securities market are subject to market risks. Read all the related documents carefully before investing.
        3. **No Assurance:** Past performance of the algorithms or stocks shown here does not guarantee future returns.
        4. **Consult a Professional:** Please consult a SEBI registered investment advisor before making any actual investment decisions.
        """
    )

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
    
    # Add Footer Disclaimer to PDF
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, txt="DISCLAIMER: Not SEBI Registered. This report is computer-generated for educational purposes only. Investments are subject to market risks.")
    
    return pdf.output(dest='S').encode('latin-1')

# --- 6. DASHBOARD UI ---
st.title("🤖 Principal Consultant's Dashboard")

# === SIDEBAR MENU ===
with st.sidebar:
    st.header("Control Panel")
    st.write(f"User: **admin**") # Show who is logged in
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()
        
    st.markdown("---")
    mode = st.radio("Select Analysis Mode:", ["Single Stock Deep Dive", "Competitor Comparison"])

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
            
            verdict = f"The stock is currently in a {metrics['trend']} trend. With an RSI of {metrics['rsi']:.1f}, momentum is {'strong' if metrics['rsi']>50 else 'weak'}. Valuation P/E is {metrics['pe']}."
            st.info(f"**AI Verdict:** {verdict}")
            
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
            
            st.divider()
            if data_a['score'] > data_b['score']:
                st.success(f"🏆 **WINNER:** {stock_a}")
            elif data_b['score'] > data_a['score']:
                st.success(f"🏆 **WINNER:** {stock_b}")
            else:
                st.warning("⚖️ **RESULT:** It's a Tie.")

# Show SEBI Disclaimer at the very bottom
show_disclaimer()