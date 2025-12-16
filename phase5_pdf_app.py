import streamlit as st
import yfinance as yf
import pandas as pd
import math
import plotly.graph_objects as go
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from fpdf import FPDF

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Principal AI Agent", layout="wide")

# --- 2. USER DATABASE ---
USERS = {
    "admin": {"password": "Orbittal2025", "role": "admin", "name": "Principal Consultant"},
    "client": {"password": "guest", "role": "viewer", "name": "Valued Client"}
}

# --- 3. SECTOR DATABASE ---
SECTORS = {
    "Blue Chips (Top 20)": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LICI.NS", "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "KOTAKBANK.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS"],
    "IT Sector": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "OFSS.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "IDFCFIRSTB.NS", "BAJAJFINSV.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BHARATFORG.NS", "TIINDIA.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS", "AUROPHARMA.NS", "ALKEM.NS", "TORNTPHARM.NS", "MANKIND.NS", "ZYDUSLIFE.NS"]
}

# --- 4. LOGIN SYSTEM ---
def check_login():
    if "logged_in" not in st.session_state: st.session_state.update({"logged_in": False, "user_role": None})
    if not st.session_state["logged_in"]:
        st.title("🔒 Institutional Login")
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if u in USERS and USERS[u]["password"] == p:
                    st.session_state.update({"logged_in": True, "user_role": USERS[u]["role"], "user_name": USERS[u]["name"]})
                    st.rerun()
                else: st.error("Invalid Credentials")
        st.stop()
check_login()

# --- 5. MARKET PULSE ---
def get_market_pulse():
    try:
        df = yf.Ticker("^NSEI").history(period="5d", interval="15m")
        if df.empty: return None
        price = df["Close"].iloc[-1]
        prev = df["Close"].iloc[0]
        return {"price": round(price, 2), "change": round(price-prev, 2), "pct": round(((price-prev)/prev)*100, 2), "trend": "BULLISH 🐂" if price > df["Close"].mean() else "BEARISH 🐻", "data": df}
    except: return None

# --- 6. CORE ANALYTICS (Debugged) ---
@st.cache_data(ttl=24*3600)
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if df.empty: return None, None, None
        
        info = stock.info
        current_price = df["Close"].iloc[-1]
        
        # Technicals
        ema_200 = EMAIndicator(close=df["Close"], window=200).ema_indicator().iloc[-1]
        rsi = RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
        stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"]).stoch().iloc[-1]
        macd = MACD(close=df["Close"])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        bb = BollingerBands(close=df["Close"], window=20)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()

        # Fundamentals
        eps = info.get('trailingEps', 0) or 0
        book_value = info.get('bookValue', 0) or 0
        pe = info.get('trailingPE', 0) or 0
        margins = info.get('profitMargins', 0) or 0
        debt = info.get('debtToEquity', 0) or 0
        roe = info.get('returnOnEquity', 0) or 0
        
        # --- FIXED: Graham Number Safety Check ---
        intrinsic_value = 0
        # Only calculate if BOTH are positive to avoid Math Domain Error (Sqrt of negative)
        if eps > 0 and book_value > 0:
            intrinsic_value = math.sqrt(22.5 * eps * book_value)
        
        # Scoring
        t_score = 0
        if current_price > ema_200: t_score += 1
        if 40 < rsi < 70: t_score += 1
        if stoch < 80: t_score += 1
        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]: t_score += 1
        if current_price > df['Close'].iloc[-50]: t_score += 1

        f_score = 0
        if 0 < pe < 40: f_score += 1
        if margins > 0.10: f_score += 1
        if debt < 100: f_score += 1
        if roe > 0.15: f_score += 1

        metrics = {
            "price": round(current_price, 2),
            "tech_score": t_score, 
            "fund_score": f_score, 
            "total_score": t_score + f_score,
            "rsi": round(rsi, 2), 
            "pe": round(pe, 2), 
            "margins": round(margins*100, 2),
            "roe": round(roe*100, 2), 
            "debt": round(debt, 2),
            "trend": "UP 🟢" if current_price > ema_200 else "DOWN 🔴",
            "intrinsic": round(intrinsic_value, 2),
            "eps": eps, 
            "book_value": book_value
        }
        return metrics, df, info
    except Exception as e:
        # Return the error message to help debug
        return None, None, str(e)

# --- 7. HELPER FUNCTIONS ---
def run_scanner(tickers):
    res = []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period="1y")
            if h.empty: continue
            p, l = h["Close"].iloc[-1], h["Low"].min()
            res.append({"Ticker": t, "Price": round(p, 2), "52W Low": round(l, 2), "Dist 52W Low (%)": round(((p-l)/l)*100, 2)})
        except: continue
    return pd.DataFrame(res)

def plot_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='gray', width=1), name='BB Upper'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', width=1), name='BB Lower'))
    fig.update_layout(title=f"{ticker} - Analysis", xaxis_rangeslider_visible=False, height=500)
    return fig

def create_pdf(ticker, data, text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Investment Memo: {ticker}", ln=True, align='C')
    pdf.ln(10)
    
    fair_val_text = f"Fair Value (Graham): Rs. {data['intrinsic']}" if data['intrinsic'] > 0 else "Fair Value: N/A"
    clean_price = str(data['price']).replace("₹", "")
    pdf.cell(200, 10, txt=f"Price: Rs. {clean_price} | {fair_val_text}", ln=True)
    
    pdf.ln(10)
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean_text)
    
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, txt="Generated by AI Agent. Educational Use Only.")
    return pdf.output(dest='S').encode('latin-1')

# --- 8. DASHBOARD UI ---
with st.sidebar:
    st.title(f"👤 {st.session_state['user_name']}")
    st.markdown("---")
    
    # Mode Selection
    if st.session_state["user_role"] == "admin":
        mode = st.radio("Mode:", ["Market Scanner", "Deep Dive Valuation", "Compare"]) 
    else:
        mode = st.radio("Mode:", ["Deep Dive Valuation"])
        
    if st.button("Logout"): 
        st.session_state.update({"logged_in": False}) 
        st.rerun()

st.title("📊 Principal Hybrid Engine")
with st.expander("🇮🇳 NSE Market Pulse", expanded=True):
    p = get_market_pulse()
    if p:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nifty 50", p['price'], f"{p['pct']}%")
        c2.metric("Trend", p['trend'])
        c3.metric("Change", p['change'])
        with c4: st.line_chart(p['data']['Close'], height=100)

# ==========================================
# MODE 1: MARKET SCANNER
# ==========================================
if mode == "Market Scanner":
    st.subheader("📡 Market Radar")
    t1, t2 = st.tabs(["Sector Leaders", "Value Hunters"])
    
    with t1:
        with st.form("scanner_form"):
            sec = st.selectbox("Select Sector:", list(SECTORS.keys()))
            submitted = st.form_submit_button("Scan Sector")
            
        if submitted:
            with st.spinner(f"Scanning {sec}..."):
                d = run_scanner(SECTORS[sec])
                st.dataframe(d)

    with t2:
        if st.button("Find 52-Week Lows"):
            with st.spinner("Hunting for value..."):
                all_s = list(set([i for s in SECTORS.values() for i in s]))
                d = run_scanner(all_s)
                st.dataframe(d.sort_values("Dist 52W Low (%)").head(10))

# ==========================================
# MODE 2: DEEP DIVE (With Debugging)
# ==========================================
elif mode == "Deep Dive Valuation":
    st.subheader("🔍 Valuation & Analysis")
    
    # 1. INPUT FORM (Fixes the "Button doesn't work" issue)
    with st.form("analysis_form"):
        ticker = st.text_input("Enter Ticker (e.g. RELIANCE.NS):", value="RELIANCE.NS")
        submitted = st.form_submit_button("Run Analysis")
    
    # 2. RESULT DISPLAY
    if submitted:
        with st.spinner(f"Analyzing {ticker}..."):
            metrics, history, info = analyze_stock(ticker)
            
            # ERROR HANDLING
            if metrics is None:
                st.error(f"❌ Could not analyze {ticker}.")
                st.warning("Possible reasons: 1. Invalid Ticker 2. Stock Delisted 3. Yahoo Finance Timeout.")
                if info: st.caption(f"Debug Info: {info}") # Show specific error if captured
            
            else:
                # SUCCESS - SHOW DASHBOARD
                c1, c2, c3 = st.columns(3)
                c1.metric("Overall Score", f"{metrics['total_score']}/10")
                c2.metric("Tech Strength", f"{metrics['tech_score']}/5")
                c3.metric("Fund Health", f"{metrics['fund_score']}/5")
                
                # TABS
                tab1, tab2, tab3 = st.tabs(["📈 Chart", "🎩 Warren Buffett Way", "🏢 Financials"])
                
                with tab1: 
                    st.plotly_chart(plot_chart(history, ticker), use_container_width=True)
                
                # WARREN BUFFETT TAB
                with tab2:
                    st.markdown("### 🎩 The Warren Buffett Valuation Model")
                    st.caption("Based on Benjamin Graham's Intrinsic Value Formula.")
                    
                    if metrics['intrinsic'] > 0:
                        col_a, col_b = st.columns(2)
                        col_a.metric("Current Market Price", f"₹{metrics['price']}")
                        col_b.metric("Calculated Fair Value", f"₹{metrics['intrinsic']}", 
                             delta=f"{round(((metrics['intrinsic']-metrics['price'])/metrics['price'])*100, 1)}% Potential" if metrics['intrinsic'] > metrics['price'] else "Premium")
                        
                        st.divider()
                        # Verdict Logic
                        if metrics['price'] < metrics['intrinsic'] * 0.7:
                            st.success(f"💎 **DEEP VALUE BUY:** Trading significantly below fair value (₹{metrics['intrinsic']}).")
                        elif metrics['price'] < metrics['intrinsic']:
                            st.success(f"✅ **UNDERVALUED:** Price is below fair value. Good entry.")
                        else:
                            st.warning(f"⚠️ **OVERVALUED:** Price is above Graham's fair value (₹{metrics['intrinsic']}).")
                    else:
                        st.error("⚠️ Cannot Calculate Intrinsic Value.")
                        st.info("Reason: This company likely has **Negative Earnings** or **Negative Book Value**.")
                        st.write(f"EPS: {metrics['eps']} | Book Value: {metrics['book_value']}")

                with tab3:
                    st.write(info.get('longBusinessSummary', 'No summary available.'))
                
                # PDF Generation
                verdict = f"Fair Value: {metrics['intrinsic']}. Score: {metrics['total_score']}/10."
                pdf = create_pdf(ticker, metrics, verdict)
                st.download_button("Download PDF Report", data=pdf, file_name=f"{ticker}_Report.pdf", mime="application/pdf")

# ==========================================
# MODE 3: COMPARE (Fixed)
# ==========================================
elif mode == "Compare":
    st.subheader("⚖️ Head-to-Head Comparison")
    
    with st.form("compare_form"):
        c1, c2 = st.columns(2)
        s1 = c1.text_input("Stock A", "TCS.NS")
        s2 = c2.text_input("Stock B", "INFY.NS")
        submitted = st.form_submit_button("Compare Stocks")
    
    if submitted:
        m1, _, _ = analyze_stock(s1)
        m2, _, _ = analyze_stock(s2)
        
        if m1 and m2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(s1, f"{m1['total_score']}/10", delta=f"Fair Val: {m1['intrinsic']}")
                st.info(f"Tech: {m1['tech_score']} | Fund: {m1['fund_score']}")
            
            with col2:
                st.metric(s2, f"{m2['total_score']}/10", delta=f"Fair Val: {m2['intrinsic']}")
                st.info(f"Tech: {m2['tech_score']} | Fund: {m2['fund_score']}")
                
            st.success(f"🏆 Winner: {s1 if m1['total_score'] > m2['total_score'] else s2}")
        else:
            st.error("Could not fetch data. Please check if tickers are correct.")