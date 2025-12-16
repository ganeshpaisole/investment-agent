import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from fpdf import FPDF

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Principal AI Agent", layout="wide")

# --- 2. USER DATABASE ---
USERS = {
    "admin": {"password": "Orbittal2025", "role": "admin", "name": "Principal Consultant"},
    "client": {"password": "guest", "role": "viewer", "name": "Valued Client"}
}

# --- 3. SECURITY SYSTEM ---
def check_login():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["user_role"] = None
    
    if not st.session_state["logged_in"]:
        st.title("🔒 Institutional Login")
        c1, c2 = st.columns(2)
        with c1:
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.button("Login"):
                if user in USERS and USERS[user]["password"] == pwd:
                    st.session_state["logged_in"] = True
                    st.session_state["user_role"] = USERS[user]["role"]
                    st.session_state["user_name"] = USERS[user]["name"]
                    st.rerun()
                else:
                    st.error("Invalid Credentials")
        st.stop()

check_login()

# --- 4. HYBRID ANALYTICS ENGINE ---
@st.cache_data(ttl=24*3600)
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if df.empty: return None, None, None
        
        info = stock.info
        
        # === A. TECHNICAL ANALYSIS (TIMING) ===
        # 1. Trend & Momentum
        ema_200 = EMAIndicator(close=df["Close"], window=200).ema_indicator().iloc[-1]
        rsi = RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
        stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"]).stoch().iloc[-1]
        
        # 2. Volatility (ATR)
        atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"]).average_true_range().iloc[-1]
        
        # 3. MACD
        macd = MACD(close=df["Close"])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # 4. Bollinger Bands (For Charting)
        bb = BollingerBands(close=df["Close"], window=20)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        
        current_price = df["Close"].iloc[-1]

        # === B. FUNDAMENTAL ANALYSIS (QUALITY) ===
        # Safe extraction of data (handles missing values)
        pe_ratio = info.get('trailingPE', 0) or 0
        forward_pe = info.get('forwardPE', 0) or 0
        peg_ratio = info.get('pegRatio', 0) or 0
        price_to_book = info.get('priceToBook', 0) or 0
        profit_margins = info.get('profitMargins', 0) or 0
        debt_to_equity = info.get('debtToEquity', 0) or 0
        roe = info.get('returnOnEquity', 0) or 0
        
        # === C. SCORING SYSTEM (The "Hybrid" Logic) ===
        
        # Technical Score (Max 5)
        tech_score = 0
        if current_price > ema_200: tech_score += 1     # Long term Uptrend
        if 40 < rsi < 70: tech_score += 1               # Healthy Momentum
        if stoch < 80: tech_score += 1                  # Not Overbought
        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]: tech_score += 1 # Bullish Cross
        if current_price > df['Close'].iloc[-50]: tech_score += 1 # Short term Uptrend
        
        # Fundamental Score (Max 5)
        fund_score = 0
        if 0 < pe_ratio < 40: fund_score += 1           # Reasonable Valuation
        if peg_ratio < 2: fund_score += 1               # Growth at reasonable price
        if profit_margins > 0.10: fund_score += 1       # Healthy Margins (>10%)
        if debt_to_equity < 100: fund_score += 1        # Low Debt
        if roe > 0.15: fund_score += 1                  # Strong Return on Equity (>15%)

        metrics = {
            "price": round(current_price, 2),
            "tech_score": tech_score,
            "fund_score": fund_score,
            "total_score": tech_score + fund_score,
            "rsi": round(rsi, 2),
            "pe": round(pe_ratio, 2),
            "margins": round(profit_margins * 100, 2),
            "roe": round(roe * 100, 2),
            "debt": round(debt_to_equity, 2),
            "trend": "UP 🟢" if current_price > ema_200 else "DOWN 🔴"
        }
        
        return metrics, df, info
        
    except Exception as e:
        return None, None, None

# --- 5. PLOTLY CHARTING ---
def plot_advanced_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='gray', width=1), name='BB Upper'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', width=1), name='BB Lower'))
    fig.update_layout(title=f"{ticker} - Interactive Analysis", xaxis_rangeslider_visible=False, height=500)
    return fig

# --- 6. PDF GENERATOR ---
def create_pdf(ticker, data, text):
    # Sanitize inputs
    clean_trend = data['trend'].replace("🟢", "").replace("🔴", "").strip()
    clean_price = str(data['price']).replace("₹", "Rs. ")
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt=f"Hybrid Investment Memo: {ticker}", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Price: Rs. {clean_price} | Tech Score: {data['tech_score']}/5 | Fund Score: {data['fund_score']}/5", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=clean_text)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(200, 10, txt="Key Financials:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 8, txt=f"ROE: {data['roe']}% | Margins: {data['margins']}% | P/E: {data['pe']}", ln=True)
    
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, txt="Generated by AI Agent. Educational Use Only.")
    
    return pdf.output(dest='S').encode('latin-1')

# --- 7. DASHBOARD UI ---
st.title(f"📊 Principal Hybrid Engine ({st.session_state['user_name']})")

with st.sidebar:
    st.header("Controls")
    if st.session_state["user_role"] == "admin":
        mode = st.radio("Mode:", ["Hybrid Deep Dive", "Competitor Comparison"])
    else:
        mode = st.radio("Mode:", ["Hybrid Deep Dive"])
        
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

# === MODE 1: HYBRID DEEP DIVE ===
if mode == "Hybrid Deep Dive":
    st.subheader("🔍 Technical + Fundamental Analysis")
    ticker = st.text_input("Ticker Symbol:", value="RELIANCE.NS")
    
    if st.button("Run Hybrid Analysis"):
        with st.spinner("Analyzing Balance Sheet & Charts..."):
            metrics, history, info = analyze_stock(ticker)
            
            if metrics:
                # --- EXECUTIVE SUMMARY ---
                c1, c2, c3 = st.columns(3)
                c1.metric("Overall Score", f"{metrics['total_score']}/10")
                c2.metric("Technical Strength", f"{metrics['tech_score']}/5", delta="Timing")
                c3.metric("Fundamental Health", f"{metrics['fund_score']}/5", delta="Quality")
                
                # --- TABS FOR DETAIL ---
                tab1, tab2 = st.tabs(["📈 Technical Chart", "🏢 Fundamental Health"])
                
                with tab1:
                    st.plotly_chart(plot_advanced_chart(history, ticker), use_container_width=True)
                    st.info(f"Trend: {metrics['trend']} | RSI: {metrics['rsi']} (Momentum)")
                
                with tab2:
                    f1, f2, f3 = st.columns(3)
                    f1.metric("Profit Margins", f"{metrics['margins']}%", help="Higher is better")
                    f2.metric("Return on Equity (ROE)", f"{metrics['roe']}%", help="Efficiency of using capital")
                    f3.metric("Debt-to-Equity", metrics['debt'], help="Risk level (Lower is usually better)")
                    
                    st.write("### Business Description")
                    st.caption(info.get('longBusinessSummary', 'No description available.'))
                
                # --- VERDICT ---
                st.divider()
                st.subheader("🤖 The AI Verdict")
                
                verdict = ""
                if metrics['total_score'] >= 8:
                    verdict = f"STRONG BUY: {ticker} is a high-quality company (Score {metrics['fund_score']}/5) currently showing strong technical momentum (Score {metrics['tech_score']}/5)."
                    st.success(verdict)
                elif metrics['fund_score'] >= 4 and metrics['tech_score'] <= 2:
                    verdict = f"WATCHLIST: {ticker} is a great company (Fundamentals {metrics['fund_score']}/5) but the trend is currently weak. Wait for a reversal."
                    st.warning(verdict)
                elif metrics['fund_score'] <= 2 and metrics['tech_score'] >= 4:
                    verdict = f"SPECULATIVE TRADING BUY: {ticker} has strong momentum, but weak fundamentals ({metrics['fund_score']}/5). Use strict stop-losses."
                    st.warning(verdict)
                else:
                    verdict = f"NEUTRAL / AVOID: {ticker} shows average performance in both quality and trend."
                    st.error(verdict)

                # PDF Download
                pdf = create_pdf(ticker, metrics, verdict)
                st.download_button("📄 Download Hybrid Memo", data=pdf, file_name=f"{ticker}_HybridReport.pdf", mime="application/pdf")

# === MODE 2: COMPARISON ===
elif mode == "Competitor Comparison":
    st.subheader("⚖️ Head-to-Head Benchmarking")
    c1, c2 = st.columns(2)
    with c1: s1 = st.text_input("Stock A", "TCS.NS")
    with c2: s2 = st.text_input("Stock B", "INFY.NS")
    
    if st.button("Compare"):
        m1, h1, f1 = analyze_stock(s1)
        m2, h2, f2 = analyze_stock(s2)
        if m1 and m2: