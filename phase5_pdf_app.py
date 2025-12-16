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

# --- 3. EXPANDED STOCK DATABASE ---
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
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
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
        price, prev = df["Close"].iloc[-1], df["Close"].iloc[0]
        return {"price": round(price, 2), "change": round(price-prev, 2), "pct": round(((price-prev)/prev)*100, 2), "trend": "BULLISH 🐂" if price > df["Close"].mean() else "BEARISH 🐻", "data": df}
    except: return None

# --- 6. CORE ANALYTICS (With Intrinsic Data) ---
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
        df['MACD'], df['MACD_Signal'] = macd.macd(), macd.macd_signal()
        bb = BollingerBands(close=df["Close"], window=20)
        df['BB_High'], df['BB_Low'] = bb.bollinger_hband(), bb.bollinger_lband()

        # Fundamentals & Intrinsic Logic
        eps = info.get('trailingEps', 0) or 0
        book_value = info.get('bookValue', 0) or 0
        pe, margins, debt, roe = info.get('trailingPE', 0) or 0, info.get('profitMargins', 0) or 0, info.get('debtToEquity', 0) or 0, info.get('returnOnEquity', 0) or 0
        
        # Graham Number Calculation (Fair Value)
        # Formula: Sqrt(22.5 * EPS * Book Value)
        intrinsic_value = 0
        if eps > 0 and book_value > 0:
            intrinsic_value = math.sqrt(22.5 * eps * book_value)
        
        # Scoring
        t_score = sum([current_price > ema_200, 40 < rsi < 70, stoch < 80, df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1], current_price > df['Close'].iloc[-50]])
        f_score = sum([0 < pe < 40, margins > 0.10, debt < 100, roe > 0.15])

        metrics = {
            "price": round(current_price, 2),
            "tech_score": t_score, "fund_score": f_score, "total_score": t_score + f_score,
            "rsi": round(rsi, 2), "pe": round(pe, 2), "margins": round(margins*100, 2),
            "roe": round(roe*100, 2), "debt": round(debt, 2),
            "trend": "UP 🟢" if current_price > ema_200 else "DOWN 🔴",
            "intrinsic": round(intrinsic_value, 2),
            "eps": eps, "book_value": book_value
        }
        return metrics, df, info
    except: return None, None, None

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
    
    # Add Intrinsic Value to PDF
    fair_val_text = f"Fair Value (Graham): Rs. {data['intrinsic']}" if data['intrinsic'] > 0 else "Fair Value: N/A"
    pdf.cell(200, 10, txt=f"Price: Rs. {str(data['price']).replace('₹','')} | {fair_val_text}", ln=True)
    
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=text.encode('latin-1', 'replace').decode('latin-1'))
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, txt="Generated by AI Agent. Educational Use Only.")
    return pdf.output(dest='S').encode('latin-1')

# --- 8. DASHBOARD UI ---
with st.sidebar:
    st.title(f"👤 {st.session_state['user_name']}")
    st.markdown("---")
    mode = st.radio("Mode:", ["Market Scanner", "Deep Dive Valuation", "Compare"]) if st.session_state["user_role"] == "admin" else st.radio("Mode:", ["Deep Dive Valuation"])
    if st.button("Logout"): st.session_state.update({"logged_in": False}); st.rerun()

st.title("📊 Principal Hybrid Engine")
with st.expander("🇮🇳 NSE Market Pulse", expanded=True):
    p = get_market_pulse()
    if p:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nifty 50", p['price'], f"{p['pct']}%")
        c2.metric("Trend", p['trend'])
        c3.metric("Change", p['change'])
        with c4: st.line_chart(p['data']['Close'], height=100)

if mode == "Market Scanner":
    st.subheader("📡 Market Radar")
    t1, t2 = st.tabs(["Sector Leaders", "Value Hunters"])
    with t1:
        sec = st.selectbox("Sector:", list(SECTORS.keys()))
        if st.button("Scan"):
            d = run_scanner(SECTORS[sec])
            st.dataframe(d)
    with t2:
        if st.button("Find Lows"):
            all_s = list(set([i for s in SECTORS.values() for i in s]))
            d = run_scanner(all_s)
            st.dataframe(d.sort_values("Dist 52W Low (%)").head(10))

elif mode == "Deep Dive Valuation":
    st.subheader("🔍 Valuation & Analysis")
    ticker = st.text_input("Ticker:", value="RELIANCE.NS")
    if st.button("Analyze"):
        with st.spinner("Calculating Intrinsic Value..."):
            m, h, i = analyze_stock(ticker)
            if m:
                c1, c2, c3 = st.columns(3)
                c1.metric("Score", f"{m['total_score']}/10")
                c2.metric("Tech Strength", f"{m['tech_score']}/5")
                c3.metric("Fund Health", f"{m['fund_score']}/5")
                
                # --- RENAMED TABS HERE ---
                tab1, tab2, tab3 = st.tabs(["📈 Chart", "🎩 Warren Buffett Way", "🏢 Financials"])
                
                with tab1: st.plotly_chart(plot_chart(h, ticker), use_container_width=True)
                
                # --- THE WARREN BUFFETT WAY TAB ---
                with tab2:
                    st.markdown("### 🎩 The Warren Buffett Valuation Model")
                    st.caption("Based on Benjamin Graham's Intrinsic Value Formula (The Mentor of Warren Buffett)")
                    
                    if m['intrinsic'] > 0:
                        col_a, col_b = st.columns(2)
                        col_a.metric("Current Market Price", f"₹{m['price']}")
                        col_b.metric("Calculated Fair Value", f"₹{m['intrinsic']}", 
                                     delta=f"{round(((m['intrinsic']-m['price'])/m['price'])*100, 1)}% Upside" if m['intrinsic'] > m['price'] else f"{round(((m['intrinsic']-m['price'])/m['price'])*100, 1)}% Premium")
                        
                        st.progress(min(m['price'] / (m['intrinsic'] * 1.5), 1.0))
                        st.caption("Visual scale of Price vs Fair Value")

                        # Suggestion Logic
                        st.divider()
                        if m['price'] < m['intrinsic'] * 0.7:
                            st.success(f"💎 **DEEP VALUE BUY:** Stock is trading significantly below its fair value (₹{m['intrinsic']}). Buffett considers this a safe margin.")
                        elif m['price'] < m['intrinsic']:
                            st.success(f"✅ **UNDERVALUED:** Current price is below fair value. Good long-term entry point.")
                        elif m['price'] < m['intrinsic'] * 1.2:
                            st.warning(f"⚠️ **FAIRLY VALUED:** Price is close to fair value. Returns depend on future growth, not valuation discount.")
                        else:
                            st.error(f"❌ **OVERVALUED:** Stock is trading well above fair value. Buying now implies high risk or high growth expectations.")
                    else:
                        st.warning("Could not calculate Intrinsic Value (Company might have negative Earnings or Book Value).")
                        
                    st.info(f"**Formula Used:** Graham Number = √(22.5 × EPS {round(m['eps'],1)} × Book Value {round(m['book_value'],1)})")

                with tab3:
                    st.write(i.get('longBusinessSummary'))
                
                verdict = f"Fair Value: {m['intrinsic']}. Score: {m['total_score']}/10."
                pdf = create_pdf(ticker, m, verdict)
                st.download_button("Download Report", data=pdf, file_name=f"{ticker}.pdf", mime="application/pdf")

elif mode == "Compare":
    c1, c2 = st.columns(2)
    s1 = c1.text_input("A", "TCS.NS"); s2 = c2.text_input("B", "INFY.NS")
    if st.button("Compare"):
        m1, _, _ = analyze_stock(s1); m2, _, _ = analyze_stock(s2)
        if m1 and m2:
            c1.metric(s1, m1['total_score']); c2.metric(s2, m2['total_score'])
            st.success(f"Winner: {s1 if m1['total_score']>m2['total_score'] else s2}")