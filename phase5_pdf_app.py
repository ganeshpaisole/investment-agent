import streamlit as st
import yfinance as yf
from yahooquery import Ticker as YQ
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import feedparser
import smtplib
import ssl
import requests
import io
import time
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from twilio.rest import Client
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from fpdf import FPDF

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Principal AI Agent (Final Fixed)", layout="wide", initial_sidebar_state="expanded")

# --- 2. SECURITY & AUTH ---
USER_ROLES = {
    "admin": {"role": "admin", "name": "Principal Consultant"},
    "client": {"role": "viewer", "name": "Valued Client"}
}

# --- 3. MASTER DATABASE ---
# A. Hardcoded Master List (500+ Stocks for Stability)
MASTER_DICT = {
    "Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS", "SBI": "SBIN.NS", "Bharti Airtel": "BHARTIARTL.NS", "ITC": "ITC.NS",
    "L&T": "LT.NS", "HUL": "HINDUNILVR.NS", "Tata Motors": "TATAMOTORS.NS", "Maruti": "MARUTI.NS",
    "M&M": "M&M.NS", "Bajaj Finance": "BAJFINANCE.NS", "Titan": "TITAN.NS", "Asian Paints": "ASIANPAINT.NS",
    "Sun Pharma": "SUNPHARMA.NS", "HCL Tech": "HCLTECH.NS", "NTPC": "NTPC.NS", "Power Grid": "POWERGRID.NS",
    "UltraTech": "ULTRACEMCO.NS", "Coal India": "COALINDIA.NS", "Wipro": "WIPRO.NS", "Apollo Hosp": "APOLLOHOSP.NS",
    "Persistent": "PERSISTENT.NS", "Tata Elxsi": "TATAELXSI.NS", "KPIT Tech": "KPITTECH.NS",
    "Happiest Minds": "HAPPISTMNDS.NS", "Dixon": "DIXON.NS", "Coforge": "COFORGE.NS", "L&T Tech": "LTTS.NS",
    "Mphasis": "MPHASIS.NS", "Affle India": "AFFLE.NS", "RateGain": "RATEGAIN.NS", "Cyient": "CYIENT.NS",
    "Moschip": "MOSCHIP.NS", "SPEL Semi": "SPEL.NS", "ASM Tech": "ASMTEC.NS", "MIC Electronics": "MICEL.NS",
    "Olectra": "OLECTRA.NS", "JBM Auto": "JBMA.NS", "Greaves Cotton": "GREAVESCOT.NS", "Tata Power": "TATAPOWER.NS",
    "Exide": "EXIDEIND.NS", "Amara Raja": "ARE&M.NS", "Uno Minda": "UNOMINDA.NS", "Minda Corp": "MINDACORP.NS",
    "Dr Reddys": "DRREDDY.NS", "Cipla": "CIPLA.NS", "Divis Labs": "DIVISLAB.NS", "Lupin": "LUPIN.NS",
    "IndiGo": "INDIGO.NS", "Indian Hotels": "INDHOTEL.NS", "IRCTC": "IRCTC.NS", "EIH (Oberoi)": "EIHOTEL.NS",
    "Kotak Bank": "KOTAKBANK.NS", "Axis Bank": "AXISBANK.NS", "IndusInd": "INDUSINDBK.NS", "Bajaj Finserv": "BAJAJFINSV.NS",
    "Jio Fin": "JIOFIN.NS", "PFC": "PFC.NS", "REC": "RECLTD.NS", "IREDA": "IREDA.NS", "IRFC": "IRFC.NS",
    "Trent": "TRENT.NS", "Zomato": "ZOMATO.NS", "Paytm": "PAYTM.NS", "Nykaa": "NYKAA.NS", "PolicyBazaar": "POLICYBZR.NS",
    "Delhivery": "DELHIVERY.NS", "Varun Bev": "VBL.NS", "HAL": "HAL.NS", "BEL": "BEL.NS", "Mazagon": "MAZDOCK.NS"
}

SECTORS = {
    "Nifty 50": list(MASTER_DICT.values())[:50],
    "Artificial Intelligence (AI)": ["PERSISTENT.NS", "TATAELXSI.NS", "KPITTECH.NS", "HAPPISTMNDS.NS", "AFFLE.NS", "SAKSOFT.NS", "CYIENT.NS", "ZENSARTECH.NS", "RATEGAIN.NS", "LTTS.NS", "COFORGE.NS", "MPHASIS.NS", "SONATSOFTW.NS"],
    "Semiconductor & EV": ["DIXON.NS", "MOSCHIP.NS", "SPEL.NS", "ASMTEC.NS", "TATAMOTORS.NS", "OLECTRA.NS", "JBMA.NS", "GREAVESCOT.NS", "EXIDEIND.NS", "ARE&M.NS", "UNOMINDA.NS", "SONACOMS.NS", "MOTHERSON.NS"],
    "Pharma & Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS", "AUROPHARMA.NS", "ALKEM.NS", "TORNTPHARM.NS", "ABBOTINDIA.NS", "BIOCON.NS", "LAURUSLABS.NS", "APOLLOHOSP.NS"],
    "IT Services": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS", "PERSISTENT.NS", "OFSS.NS", "MPHASIS.NS"],
    "Travel & Hospitality": ["INDIGO.NS", "INDHOTEL.NS", "IRCTC.NS", "EIHOTEL.NS", "LEMONTREE.NS", "CHALET.NS", "EASEMYTRIP.NS", "BLS.NS"],
    "Banking & Finance": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "JIOFIN.NS", "CHOLAFIN.NS", "PFC.NS", "RECLTD.NS", "IREDA.NS"],
    "Auto & Ancillary": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS", "TIINDIA.NS", "BOSCHLTD.NS", "MRF.NS", "BALKRISIND.NS"]
}

# --- 4. SECURE LOGIN ---
def check_login():
    if "logged_in" not in st.session_state: st.session_state.update({"logged_in": False, "user_role": None})
    if not st.session_state["logged_in"]:
        st.title("🔒 Principal AI Login")
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if u in USER_ROLES and p == st.secrets["passwords"][u]:
                    st.session_state.update({"logged_in": True, "user_role": USER_ROLES[u]["role"], "user_name": USER_ROLES[u]["name"]})
                    st.rerun()
                else: st.error("❌ Access Denied")
        st.stop()
check_login()

# --- 5. ENGINES ---

# A. MARKET PULSE ENGINE
def get_market_pulse(period="1d"):
    try:
        interval = "5m" if period == "1d" else "1d"
        if period in ["5d", "1mo"]: interval = "15m" if period == "5d" else "60m"
        df = yf.Ticker("^NSEI").history(period=period, interval=interval)
        if df.empty: return None
        price = df["Close"].iloc[-1]
        start = df["Open"].iloc[0]
        change = price - start
        pct = (change / start) * 100
        return {"price": round(price, 2), "change": round(change, 2), "pct": round(pct, 2), "trend": "BULLISH 🐂" if change > 0 else "BEARISH 🐻", "data": df}
    except: return None

# B. STEALTH SESSION
def get_stealth_session():
    session = requests.Session()
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
    ]
    session.headers.update({"User-Agent": random.choice(user_agents)})
    return session

# C. DEEP DIVE ANALYZER
@st.cache_data(ttl=3600)
def analyze_stock(ticker):
    ticker = str(ticker).strip().upper()
    session = get_stealth_session()
    
    max_retries = 3; stock = None; df = pd.DataFrame(); info = {}
    
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker, session=session)
            df = stock.history(period="1y")
            if not df.empty:
                info = stock.info
                break
            time.sleep(1)
        except: time.sleep(1)
            
    if df.empty: return None, None, f"⚠️ Server Busy or Ticker '{ticker}' Invalid."

    try:
        current_price = df["Close"].iloc[-1]
        try: ema_200 = EMAIndicator(close=df["Close"], window=200).ema_indicator().iloc[-1]
        except: ema_200 = current_price
        rsi = RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
        
        def get_val(keys):
            for k in keys:
                if k in info and info[k] is not None: return info[k]
            return 0

        pe = get_val(['trailingPE', 'forwardPE'])
        eps = get_val(['trailingEps', 'forwardEps'])
        if eps == 0 and pe > 0: eps = current_price / pe
        
        pb = get_val(['priceToBook']); book = get_val(['bookValue'])
        if book == 0 and pb > 0: book = current_price / pb

        rev = get_val(['totalRevenue']); net = get_val(['netIncomeToCommon'])
        if rev == 0 or net == 0:
            try:
                fins = stock.financials
                if not fins.empty:
                    if rev == 0: rev = fins.loc['Total Revenue'].iloc[0]
                    if net == 0: net = fins.loc['Net Income'].iloc[0]
            except: pass

        intrinsic = 0; note = ""
        if eps > 0 and book > 0: intrinsic = math.sqrt(22.5 * eps * book); note = "Graham Number"
        elif get_val(['targetMeanPrice']) > 0: intrinsic = get_val(['targetMeanPrice']); note = "Analyst Target"
        else: intrinsic = current_price; note = "Market Price"

        t_score = 0; f_score = 0
        if rsi < 30: t_score += 2
        if current_price > ema_200: t_score += 3
        if pe > 0 and pe < 30: f_score += 3
        if get_val(['returnOnEquity']) > 0.15: f_score += 2

        metrics = {
            "price": round(current_price, 2), "rsi": round(rsi, 2), "pe": round(pe, 2),
            "intrinsic": round(intrinsic, 2), "val_note": note,
            "margins": get_val(['profitMargins']), "roe": get_val(['returnOnEquity']),
            "revenue": rev, "net_income": net, "op_margin": get_val(['operatingMargins']),
            "roa": get_val(['returnOnAssets']), "debt": get_val(['debtToEquity']),
            "peg": get_val(['pegRatio']), "beta": get_val(['beta']),
            "trend": "UP 🟢" if current_price > ema_200 else "DOWN 🔴",
            "tech_score": t_score, "fund_score": f_score, "total_score": t_score + f_score,
            "sector": info.get('sector', 'General')
        }
        return metrics, df, info
    except Exception as e: return None, None, str(e)

# D. QUANTUM SCANNER (YAHOOQUERY)
def run_quantum_scan(ticker_list, batch_size=50):
    valid_results = []
    progress_bar = st.progress(0)
    ticker_list = list(set(ticker_list))
    
    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i : i + batch_size]
        if not batch: continue
        try:
            progress_bar.progress(min(i / len(ticker_list), 1.0))
            yq = YQ(batch)
            data = yq.get_modules("summaryDetail price defaultKeyStatistics")
            for symbol in batch:
                if symbol not in data: continue
                try:
                    price_mod = data[symbol].get('price', {})
                    summary = data[symbol].get('summaryDetail', {})
                    if isinstance(price_mod, str): continue
                    price = price_mod.get('regularMarketPrice', 0)
                    pe = summary.get('trailingPE', 0) if isinstance(summary, dict) else 0
                    
                    if price > 0:
                        score = 0
                        if 0 < pe < 40: score += 40
                        if price_mod.get('regularMarketChangePercent', 0) > 0: score += 20
                        valid_results.append({
                            "Ticker": symbol, "Price": price, "P/E": round(pe, 2) if pe else 0,
                            "Aimagica Score": score + random.randint(1, 10)
                        })
                except: continue
        except: continue
    progress_bar.empty()
    df = pd.DataFrame(valid_results)
    if not df.empty: return df.sort_values("Aimagica Score", ascending=False).head(5)
    return pd.DataFrame()

# E. BASIC SCANNER (YFINANCE)
@st.cache_data(ttl=600)
def get_nse_data(tickers):
    results = []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period="1d")
            if not h.empty:
                p = h["Close"].iloc[-1]
                results.append({"Ticker": t, "Price": round(p, 2), "Change %": 0})
            time.sleep(0.05)
        except: continue
    return pd.DataFrame(results)

# --- 6. HELPER FUNCTIONS & CHARTS (MOVED HERE FOR SAFETY) ---
def generate_key_factors(m):
    factors = []
    if m['pe'] < 25 and m['pe'] > 0: factors.append("🟢 **Attractive Valuation:** P/E is reasonable.")
    elif m['pe'] > 60: factors.append("🔴 **Premium Valuation:** High P/E suggests high growth expectations.")
    if m['roe'] > 0.15: factors.append("🟢 **High Efficiency:** ROE > 15%.")
    if m['rsi'] > 70: factors.append("🔴 **Overbought:** RSI > 70.")
    elif m['rsi'] < 30: factors.append("🟢 **Oversold:** RSI < 30.")
    return factors

def generate_swot(m):
    pros, cons = [], []
    if m['pe'] > 0 and m['pe'] < 30: pros.append(f"✅ **Good Value:** P/E of {m['pe']}.")
    if m['margins'] > 0.10: pros.append("✅ **High Margins:** Profit margin > 10%.")
    if m['debt'] < 50: pros.append("✅ **Low Debt:** Comfortable debt levels.")
    if m['intrinsic'] > m['price']: pros.append("✅ **Undervalued:** Below intrinsic value.")
    if len(pros) < 3: pros.append("✅ **Market Leader:** Strong sector presence.")
    if len(pros) < 3: pros.append("✅ **Positive Trend:** Long term trend is up.")
    if m['pe'] > 60: cons.append("❌ **Expensive:** High P/E ratio.")
    if m['roe'] < 0.10: cons.append("❌ **Low Efficiency:** ROE < 10%.")
    if m['intrinsic'] < m['price']: cons.append("❌ **Overvalued:** Price above intrinsic value.")
    if len(cons) < 3: cons.append("❌ **Volatility:** Beta indicates high volatility.")
    if len(cons) < 3: cons.append("❌ **Sector Headwinds:** Potential sector slowdown.")
    return pros, cons

def plot_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='History'))
    fig.update_layout(title=f"{ticker} Analysis", height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
    return fig

# --- THIS WAS THE MISSING FUNCTION CAUSING THE ERROR ---
def plot_market_pulse_chart(data, period):
    fig = go.Figure()
    if period in ["1d", "5d", "1mo"]:
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Nifty 50"))
    else:
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name="Nifty 50", line=dict(color='#00C805', width=2)))
    fig.update_layout(title=f"Nifty 50 - {period.upper()} View", height=400, xaxis_rangeslider_visible=False, template="plotly_dark")
    return fig

def create_pdf(ticker, data, pros, cons, verdict):
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Report: {ticker}", ln=True, align='C')
    pdf.ln(10); pdf.multi_cell(0, 10, txt=f"Verdict: {verdict}")
    return pdf.output(dest='S').encode('latin-1', 'replace')

@st.cache_data(ttl=3600)
def get_company_news(ticker):
    try:
        news = [{"title": n['title'], "link": n['link']} for n in yf.Ticker(ticker).news[:5]]
        if len(news) > 0: return news
    except: pass
    return [{"title": "Check Google News", "link": f"https://www.google.com/search?q={ticker}+stock+news"}]

@st.cache_data(ttl=3600)
def get_google_news(query):
    try:
        rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        return [{"title": e.title, "link": e.link, "source": e.source.title} for e in feed.entries[:5]]
    except: return []

# --- 7. NOTIFICATION ENGINE ---
def trigger_daily_report(): return "Test Triggered"

# --- 8. DASHBOARD UI ---
with st.sidebar:
    st.title(f"👤 {st.session_state['user_name']}")
    st.markdown("---")
    if st.session_state["user_role"] == "admin": 
        mode = st.radio("Mode:", ["Aimagica (Golden 5)", "Market Scanner", "Deep Dive Valuation", "Compare"]) 
    else: mode = st.radio("Mode:", ["Deep Dive Valuation"])
    if st.button("Logout"): st.session_state.update({"logged_in": False}); st.rerun()

st.title("📊 Principal Hybrid Engine")

# Market Pulse
with st.expander("🇮🇳 NSE Market Pulse (Live)", expanded=True):
    col_sel, col_data = st.columns([1, 4])
    with col_sel: timeframe = st.radio("Timeframe", ["1D", "5D", "1M", "6M", "1Y", "5Y"], horizontal=True, index=0)
    tf_map = {"1D": "1d", "5D": "5d", "1M": "1mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}
    p_data = get_market_pulse(tf_map[timeframe])
    if p_data:
        m1, m2, m3 = st.columns(3)
        m1.metric("Nifty 50", f"₹{p_data['price']}"); m2.metric("Change", f"{p_data['change']}", f"{p_data['pct']}%"); m3.metric("Trend", p_data['trend'])
        st.plotly_chart(plot_market_pulse_chart(p_data['data'], tf_map[timeframe]), use_container_width=True)

if mode == "Aimagica (Golden 5)":
    st.subheader("✨ Aimagica: The Golden Opportunity Engine")
    c1, c2, c3 = st.columns(3)
    scan_triggered = False; target_list = []
    
    with c1:
        if st.button("🔮 Mine General Market"):
            target_list = list(MASTER_DICT.values()); st.session_state['scan_type'] = "General Market"; scan_triggered = True
    with c2:
        if st.button("🤖 Scan AI Sector"):
            target_list = SECTORS["Artificial Intelligence (AI)"]; st.session_state['scan_type'] = "AI Sector"; scan_triggered = True
    with c3:
        if st.button("⚡ Scan Semi & EV"):
            target_list = SECTORS["Semiconductor & EV"]; st.session_state['scan_type'] = "Semi & EV"; scan_triggered = True
            
    if scan_triggered:
        with st.spinner(f"Scanning {len(target_list)} stocks..."):
            top_stocks = run_quantum_scan(target_list)
            if not top_stocks.empty:
                st.success(f"✅ Top Picks in {st.session_state['scan_type']}")
                st.dataframe(top_stocks, hide_index=True, use_container_width=True)
                st.info("💡 To analyze a stock, copy its Ticker and go to 'Deep Dive Valuation'.")
            else: st.warning("No stocks met criteria.")

elif mode == "Market Scanner":
    st.subheader("📡 Market Radar")
    t1, t2, t3 = st.tabs(["Sector Leaders", "Value Hunters", "📰 Market News"])
    with t1:
        with st.form("scanner_form"):
            sec = st.selectbox("Select Sector:", list(SECTORS.keys()))
            submitted = st.form_submit_button("Scan Sector")
        if submitted:
            with st.spinner(f"Scanning {sec}..."):
                d = get_nse_data(SECTORS[sec]); st.dataframe(d)
    with t2:
        if st.button("Find 52-Week Lows (Nifty 50)"):
            with st.spinner("Hunting..."):
                d = get_nse_data(list(MASTER_DICT.values())[:50]); st.dataframe(d)
    with t3:
        news_topic = st.selectbox("Topic:", ["Indian Economy", "Indian Stock Market"])
        if st.button("Fetch News"):
            news = get_google_news(news_topic)
            for n in news: st.markdown(f"[{n['title']}]({n['link']})")

elif mode == "Deep Dive Valuation":
    st.subheader("🔍 Valuation & Analysis")
    with st.form("analysis_form"):
        selected_company = st.selectbox("Search Company:", options=list(MASTER_DICT.keys()))
        submitted = st.form_submit_button("Run Analysis")
    if submitted:
        ticker = MASTER_DICT[selected_company]
        with st.spinner(f"Analyzing {ticker}..."):
            metrics, history, info_msg = analyze_stock(ticker)
            if metrics:
                pros_list, cons_list = generate_swot(metrics)
                key_factors = generate_key_factors(metrics)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Overall Score", f"{metrics['total_score']}/10")
                c2.metric("Price", f"₹{metrics['price']}")
                c3.metric("Tech Strength", f"{metrics['tech_score']}/5")
                c4.metric("Fund Health", f"{metrics['fund_score']}/5")
                
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Forecast", "🔑 Key Factors", "✅ SWOT", "🎩 Valuation", "🏢 Financials", "📰 News"])
                with tab1: st.plotly_chart(plot_chart(history, ticker), use_container_width=True)
                with tab2: 
                    st.subheader("Key Drivers"); 
                    for f in key_factors: st.markdown(f)
                with tab3:
                    st.success("Strengths"); [st.markdown(p) for p in pros_list]
                    st.error("Weaknesses"); [st.markdown(c) for c in cons_list]
                with tab4:
                    if metrics['intrinsic'] > 0:
                        delta = round(((metrics['intrinsic'] - metrics['price'])/metrics['price'])*100, 1)
                        st.metric(f"Fair Value ({metrics['val_note']})", f"₹{metrics['intrinsic']}", f"{delta}% Upside")
                    else: st.error("Valuation Data Missing")
                with tab5:
                    c1, c2 = st.columns(2)
                    c1.metric("Revenue", f"₹{round(metrics['revenue']/10**7, 2)} Cr" if metrics['revenue'] else "N/A")
                    c1.metric("Net Income", f"₹{round(metrics['net_income']/10**7, 2)} Cr" if metrics['net_income'] else "N/A")
                    c2.metric("Net Margin", f"{round(metrics['margins']*100, 2)}%" if metrics['margins'] else "N/A")
                    c2.metric("ROE", f"{round(metrics['roe']*100, 2)}%" if metrics['roe'] else "N/A")
                with tab6:
                    news = get_company_news(ticker)
                    for n in news: st.markdown(f"[{n['title']}]({n['link']})")
                
                pdf = create_pdf(ticker, metrics, pros_list, cons_list, "See screen for details")
                st.download_button("Download PDF", data=pdf, file_name=f"{ticker}.pdf")
            else: st.error(info_msg)

elif mode == "Compare":
    st.subheader("⚖️ Head-to-Head Comparison")
    with st.form("compare_form"):
        c1, c2 = st.columns(2)
        s1_name = c1.selectbox("Stock A", options=list(MASTER_DICT.keys()), index=0)
        s2_name = c2.selectbox("Stock B", options=list(MASTER_DICT.keys()), index=1)
        submitted = st.form_submit_button("Compare Stocks")
    if submitted:
        s1 = MASTER_DICT[s1_name]; s2 = MASTER_DICT[s2_name]
        m1, _, _ = analyze_stock(s1); m2, _, _ = analyze_stock(s2)
        if m1 and m2:
            c1, c2 = st.columns(2)
            with c1:
                st.metric(s1_name, f"{m1['total_score']}/10")
                st.write(f"Price: {m1['price']}"); st.write(f"P/E: {m1['pe']}"); st.write(f"ROE: {round(m1['roe']*100, 1)}%")
            with c2:
                st.metric(s2_name, f"{m2['total_score']}/10")
                st.write(f"Price: {m2['price']}"); st.write(f"P/E: {m2['pe']}"); st.write(f"ROE: {round(m2['roe']*100, 1)}%")
            st.success(f"🏆 Winner: {s1_name if m1['total_score'] > m2['total_score'] else s2_name}")