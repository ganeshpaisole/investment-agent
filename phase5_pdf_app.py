import streamlit as st
import yfinance as yf
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

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Principal AI Agent", layout="wide")

# --- 2. SECURITY CONFIG ---
USER_ROLES = {
    "admin": {"role": "admin", "name": "Principal Consultant"},
    "client": {"role": "viewer", "name": "Valued Client"}
}

# --- 3. DATABASE ENGINE ---
FAILSAFE_COMPANIES = {
    "Reliance Industries (RELIANCE)": "RELIANCE.NS", "TCS (TCS)": "TCS.NS", 
    "HDFC Bank (HDFCBANK)": "HDFCBANK.NS", "ICICI Bank (ICICIBANK)": "ICICIBANK.NS", 
    "Infosys (INFY)": "INFY.NS", "State Bank of India (SBIN)": "SBIN.NS",
    "Bharti Airtel (BHARTIARTL)": "BHARTIARTL.NS", "ITC Ltd (ITC)": "ITC.NS", 
    "Larsen & Toubro (LT)": "LT.NS", "Hindustan Unilever (HINDUNILVR)": "HINDUNILVR.NS",
    "Maruti Suzuki (MARUTI)": "MARUTI.NS", "Mahindra & Mahindra (M&M)": "M&M.NS", 
    "Tata Motors (TATAMOTORS)": "TATAMOTORS.NS", "Bajaj Auto (BAJAJ-AUTO)": "BAJAJ-AUTO.NS", 
    "Eicher Motors (EICHERMOT)": "EICHERMOT.NS", "Hero MotoCorp (HEROMOTOCO)": "HEROMOTOCO.NS",
    "Asian Paints (ASIANPAINT)": "ASIANPAINT.NS", "Titan Company (TITAN)": "TITAN.NS", 
    "Nestle India (NESTLEIND)": "NESTLEIND.NS", "Britannia (BRITANNIA)": "BRITANNIA.NS", 
    "Tata Consumer (TATACONSUM)": "TATACONSUM.NS", "Trent (TRENT)": "TRENT.NS",
    "Bajaj Finance (BAJFINANCE)": "BAJFINANCE.NS", "Bajaj Finserv (BAJAJFINSV)": "BAJAJFINSV.NS", 
    "Kotak Bank (KOTAKBANK)": "KOTAKBANK.NS", "Axis Bank (AXISBANK)": "AXISBANK.NS", 
    "IndusInd Bank (INDUSINDBK)": "INDUSINDBK.NS", "HDFC Life (HDFCLIFE)": "HDFCLIFE.NS",
    "SBI Life (SBILIFE)": "SBILIFE.NS", "Shriram Finance (SHRIRAMFIN)": "SHRIRAMFIN.NS",
    "Jio Financial (JIOFIN)": "JIOFIN.NS", "REC Ltd (REC)": "RECLTD.NS",
    "Power Finance Corp (PFC)": "PFC.NS", "IREDA (IREDA)": "IREDA.NS",
    "HCL Tech (HCLTECH)": "HCLTECH.NS", "Wipro (WIPRO)": "WIPRO.NS", 
    "Tech Mahindra (TECHM)": "TECHM.NS", "LTIMindtree (LTIM)": "LTIM.NS",
    "Sun Pharma (SUNPHARMA)": "SUNPHARMA.NS", "Dr Reddys Labs (DRREDDY)": "DRREDDY.NS", 
    "Cipla (CIPLA)": "CIPLA.NS", "Divis Labs (DIVISLAB)": "DIVISLAB.NS", 
    "Apollo Hospitals (APOLLOHOSP)": "APOLLOHOSP.NS",
    "Tata Steel (TATASTEEL)": "TATASTEEL.NS", "JSW Steel (JSWSTEEL)": "JSWSTEEL.NS", 
    "Hindalco (HINDALCO)": "HINDALCO.NS", "NTPC (NTPC)": "NTPC.NS", 
    "Power Grid (POWERGRID)": "POWERGRID.NS", "ONGC (ONGC)": "ONGC.NS",
    "Coal India (COALINDIA)": "COALINDIA.NS", "BPCL (BPCL)": "BPCL.NS", 
    "Adani Enterprises (ADANIENT)": "ADANIENT.NS", "Adani Ports (ADANIPORTS)": "ADANIPORTS.NS", 
    "Grasim Industries (GRASIM)": "GRASIM.NS", "UltraTech Cement (ULTRACEMCO)": "ULTRACEMCO.NS",
    "Bharat Electronics (BEL)": "BEL.NS", "HAL (HAL)": "HAL.NS", 
    "Zomato (ZOMATO)": "ZOMATO.NS", "Paytm (PAYTM)": "PAYTM.NS", 
    "Varun Beverages (VBL)": "VBL.NS", "PB Fintech (POLICYBZR)": "POLICYBZR.NS"
}

@st.cache_data(ttl=24*3600)
def load_nse_master_list():
    master_dict = FAILSAFE_COMPANIES.copy()
    try:
        url = "https://raw.githubusercontent.com/sfini/NSE-Data/master/EQUITY_L.csv"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            if 'SYMBOL' in df.columns and 'NAME OF COMPANY' in df.columns:
                for index, row in df.iterrows():
                    symbol = row['SYMBOL']
                    name = row['NAME OF COMPANY']
                    if symbol == "VARUN": yahoo = "VBL.NS"
                    elif symbol == "REC": yahoo = "RECLTD.NS"
                    else: yahoo = f"{symbol}.NS"
                    master_dict[f"{name} ({symbol})"] = yahoo
    except: pass
    return master_dict

NSE_COMPANIES = load_nse_master_list()

SECTORS = {
    "Nifty 50 (All)": list(FAILSAFE_COMPANIES.values()),
    "Banks": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS"],
    "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS"]
}

# --- 4. SECURE LOGIN ---
def check_login():
    if "logged_in" not in st.session_state: st.session_state.update({"logged_in": False, "user_role": None})
    if not st.session_state["logged_in"]:
        st.title("🔒 Institutional Login")
        st.caption("Protected by Streamlit Secrets Manager")
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if u in USER_ROLES:
                    try:
                        if p == st.secrets["passwords"][u]:
                            st.session_state.update({"logged_in": True, "user_role": USER_ROLES[u]["role"], "user_name": USER_ROLES[u]["name"]})
                            st.rerun()
                        else: st.error("❌ Invalid Password")
                    except: st.error("⚠️ Secrets not configured.")
                else: st.error("❌ Invalid Username")
        st.stop()
check_login()

# --- 5. MARKET PULSE ---
def get_market_pulse():
    try:
        df = yf.Ticker("^NSEI").history(period="5d")
        if df.empty: return None
        price = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2] if len(df) > 1 else df["Open"].iloc[-1]
        change_val = price - prev_close
        pct_val = (change_val / prev_close) * 100
        return {"price": round(price, 2), "change": round(change_val, 2), "pct": round(pct_val, 2), "trend": "BULLISH 🐂" if change_val > 0 else "BEARISH 🐻", "data": df}
    except: return None

# --- 6. CORE ANALYTICS (SMART VALUATION) ---
@st.cache_data(ttl=3600)
def analyze_stock(ticker):
    ticker = str(ticker).strip().upper()
    
    # Retry Logic for Rate Limits
    max_retries = 3
    stock = None
    df = pd.DataFrame()
    info = {}
    
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            if not df.empty:
                info = stock.info
                break
            time.sleep(random.uniform(1.0, 3.0)) 
        except Exception as e:
            time.sleep(1)
            
    if df.empty:
        return None, None, f"⚠️ Server Busy. Please try '{ticker}' again later."

    try:
        current_price = df["Close"].iloc[-1]
        
        # Technicals
        try: ema_200 = EMAIndicator(close=df["Close"], window=200).ema_indicator().iloc[-1]
        except: ema_200 = current_price
        
        rsi = RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
        stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"]).stoch().iloc[-1]
        macd = MACD(close=df["Close"])
        df['MACD'] = macd.macd(); df['MACD_Signal'] = macd.macd_signal()
        bb = BollingerBands(close=df["Close"], window=20)
        df['BB_High'] = bb.bollinger_hband(); df['BB_Low'] = bb.bollinger_lband()

        # --- SMART FUNDAMENTALS EXTRACTION ---
        def get_val(keys):
            # Tries multiple keys to find data (Yahoo is inconsistent)
            for k in keys:
                if k in info and info[k] is not None:
                    return info[k]
            return 0

        pe = get_val(['trailingPE', 'forwardPE'])
        pb = get_val(['priceToBook'])
        
        # 1. Smart EPS: Try direct -> Try calculate from PE
        eps = get_val(['trailingEps', 'forwardEps'])
        if eps == 0 and pe > 0: 
            eps = current_price / pe
            
        # 2. Smart Book Value: Try direct -> Try calculate from PB
        book_value = get_val(['bookValue'])
        if book_value == 0 and pb > 0:
            book_value = current_price / pb

        margins = get_val(['profitMargins'])
        debt = get_val(['debtToEquity'])
        roe = get_val(['returnOnEquity'])
        peg = get_val(['pegRatio'])
        
        # 3. Fair Value Calculation (Graham Number)
        intrinsic_value = 0
        valuation_note = ""
        
        if eps > 0 and book_value > 0:
            intrinsic_value = math.sqrt(22.5 * eps * book_value)
        elif eps < 0:
            valuation_note = "Company is Loss-Making (Negative EPS). Fair Value N/A."
        else:
            valuation_note = "Insufficient Data (Missing EPS/Book Value)."

        # Scores
        t_score = sum([current_price > ema_200, 40 < rsi < 70, stoch < 80, df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]])
        f_score = sum([0 < pe < 40, margins > 0.10, debt < 100, roe > 0.15])

        metrics = {
            "price": round(current_price, 2),
            "tech_score": t_score, "fund_score": f_score, "total_score": t_score + f_score,
            "rsi": round(rsi, 2), "pe": round(pe, 2), "margins": round(margins*100, 2),
            "roe": round(roe*100, 2), "debt": round(debt, 2), "peg": peg,
            "trend": "UP 🟢" if current_price > ema_200 else "DOWN 🔴",
            "intrinsic": round(intrinsic_value, 2),
            "val_note": valuation_note, # <--- Pass the reason to UI
            "eps": eps, "book_value": book_value, "sector": info.get('sector', 'General')
        }
        return metrics, df, info
    except Exception as e: 
        return None, None, f"⚠️ Analysis Error: {str(e)}"

# --- 7. FAST SCANNER ---
@st.cache_data(ttl=600)
def get_nse_data(tickers):
    results = []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period="1d")
            if not h.empty:
                p = h["Close"].iloc[-1]
                results.append({"Ticker": t, "Price": round(p, 2), "Change %": 0})
            time.sleep(0.05) # Tiny pause
        except: continue
    return pd.DataFrame(results)

# --- 8. AIMAGICA ---
def run_aimagica_scan(stock_list):
    results = []
    for ticker in stock_list:
        try:
            m, _, _ = analyze_stock(ticker)
            if not m: continue
            
            val_score = 20 if (m['intrinsic'] > 0 and m['price'] < m['intrinsic']) else 0
            if m['price'] < m['intrinsic'] * 0.7: val_score += 10
            rev_score = 10 if m['rsi'] < 40 else 0
            if "UP" in m['trend']: rev_score += 15
            qual_score = 10 if m['margins'] > 15 else 0
            grow_score = 15 if (0 < m['peg'] < 1.5) else 0
            
            final_score = val_score + rev_score + qual_score + grow_score
            
            if final_score > 40:
                results.append({
                    "Ticker": ticker, "Price": m['price'], "Aimagica Score": final_score,
                    "Why": f"Val: {val_score}/30 | Mom: {rev_score}/25",
                    "Upside": round(((m['intrinsic'] - m['price'])/m['price'])*100, 1) if m['intrinsic'] > 0 else 0
                })
        except: continue
    df_res = pd.DataFrame(results)
    if not df_res.empty: df_res = df_res.sort_values("Aimagica Score", ascending=False).head(5)
    return df_res

# --- 9. HELPER FUNCTIONS ---
def generate_swot(m):
    pros, cons = [], []
    if m['pe'] > 0 and m['pe'] < 25: pros.append(f"Valuation is attractive (P/E {m['pe']}).")
    elif m['pe'] > 50: cons.append(f"Stock is expensive (High P/E {m['pe']}).")
    if m['margins'] > 15: pros.append(f"High Profit Margins ({m['margins']}%).")
    if m['debt'] < 50: pros.append("Company has low debt levels.")
    if m['rsi'] < 30: pros.append("Technically Oversold.")
    if m['intrinsic'] > 0 and m['price'] < m['intrinsic']: pros.append("Below Intrinsic Value.")
    return pros, cons

def plot_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='History'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='gray', width=1), name='BB Upper'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', width=1), name='BB Lower'))
    try:
        recent_df = df.tail(90).copy()
        if len(recent_df) > 20:
            x = np.arange(len(recent_df))
            y = recent_df['Close'].values
            coeffs = np.polyfit(x, y, 2)
            poly_curve = np.poly1d(coeffs)
            future_x = np.arange(len(recent_df), len(recent_df) + 30)
            future_prices = poly_curve(future_x)
            fig.add_trace(go.Scatter(x=pd.date_range(start=df.index[-1], periods=31)[1:], y=future_prices, mode='lines', line=dict(color='#FFA500', width=2, dash='dash'), name='AI Projected Path'))
    except: pass
    fig.update_layout(title=f"{ticker} - Analysis & Forecast", xaxis_rangeslider_visible=False, height=600)
    return fig

def create_pdf(ticker, data, pros, cons, verdict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Investment Memo: {ticker}", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Price: {data['price']} | Score: {data['total_score']}/10", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"VERDICT: {verdict}")
    return pdf.output(dest='S').encode('latin-1', 'replace')

@st.cache_data(ttl=3600)
def get_google_news(query):
    try:
        rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        return [{"title": e.title, "link": e.link, "source": e.source.title} for e in feed.entries[:5]]
    except: return []

@st.cache_data(ttl=3600)
def get_company_news(ticker):
    try:
        return [{"title": n['title'], "link": n['link'], "publisher": n.get('publisher', 'Yahoo')} for n in yf.Ticker(ticker).news[:5]]
    except: return []

# --- 10. NOTIFICATION ENGINE ---
def send_email_alert(subject, body):
    try:
        sender = st.secrets["notifications"]["email_sender"]
        password = st.secrets["notifications"]["email_password"]
        receiver = st.secrets["notifications"]["email_receiver"]
        msg = MIMEMultipart()
        msg['From'] = sender; msg['To'] = receiver; msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender, password); server.sendmail(sender, receiver, msg.as_string())
        return True, "Email Sent"
    except Exception as e: return False, str(e)

def send_whatsapp_alert(body):
    try:
        sid = st.secrets["notifications"]["twilio_sid"]
        token = st.secrets["notifications"]["twilio_token"]
        from_num = st.secrets["notifications"]["twilio_from"]
        to_num = st.secrets["notifications"]["twilio_to"]
        client = Client(sid, token)
        message = client.messages.create(body=body, from_=from_num, to=to_num)
        return True, f"WhatsApp Sent (SID: {message.sid})"
    except Exception as e: return False, str(e)

def trigger_daily_report():
    top_5 = run_aimagica_scan(list(FAILSAFE_COMPANIES.values()))
    if not top_5.empty:
        msg_body = "🚀 *Golden 5 Report* 🚀\n"
        for i, row in top_5.iterrows():
            msg_body += f"{i+1}. {row['Ticker']} (Score: {int(row['Aimagica Score'])})\n"
        e_ok, e_msg = send_email_alert("Golden 5 Stocks", msg_body)
        w_ok, w_msg = send_whatsapp_alert(msg_body)
        return f"{e_msg}. {w_msg}"
    return "No opportunities."

if 'scheduler' not in st.session_state:
    scheduler = BackgroundScheduler()
    scheduler.add_job(trigger_daily_report, 'cron', hour=9, minute=30)
    scheduler.start()
    st.session_state['scheduler'] = scheduler

# --- 11. DASHBOARD UI ---
with st.sidebar:
    st.title(f"👤 {st.session_state['user_name']}")
    st.markdown("---")
    if st.session_state["user_role"] == "admin": mode = st.radio("Mode:", ["Aimagica (Golden 5)", "Market Scanner", "Deep Dive Valuation", "Compare"]) 
    else: mode = st.radio("Mode:", ["Deep Dive Valuation"])
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

if mode == "Aimagica (Golden 5)":
    st.subheader("✨ Aimagica: The Golden Opportunity Engine")
    c1, c2 = st.columns([3, 1])
    with c1:
        if st.button("🔮 Reveal Top 5 Opportunities"):
            with st.spinner("Scanning Nifty 50..."):
                top_5 = run_aimagica_scan(list(FAILSAFE_COMPANIES.values()))
                if not top_5.empty:
                    st.balloons()
                    cols = st.columns(5)
                    for i, row in top_5.iterrows():
                        with cols[i]:
                            st.markdown(f"### {row['Ticker']}")
                            st.metric("Price", f"₹{row['Price']}", delta=f"{row['Upside']}% Upside")
                            st.progress(row['Aimagica Score']/100)
                    st.divider(); st.dataframe(top_5, hide_index=True)
                else: st.warning("No Golden opportunities found.")
    with c2:
        st.info("🔔 **Automation**")
        if st.button("📧 Test Alerts"):
            with st.spinner("Sending..."):
                result = trigger_daily_report(); st.success(result)

elif mode == "Market Scanner":
    st.subheader("📡 Market Radar")
    t1, t2, t3 = st.tabs(["Sector Leaders", "Value Hunters", "📰 Market News"])
    with t1:
        with st.form("scanner_form"):
            sec = st.selectbox("Select Sector:", list(SECTORS.keys()))
            submitted = st.form_submit_button("Scan Sector")
        if submitted:
            with st.spinner(f"Scanning {sec}..."):
                d = get_nse_data(SECTORS[sec])
                st.dataframe(d)
    with t2:
        if st.button("Find 52-Week Lows (Nifty 50)"):
            with st.spinner("Hunting..."):
                d = get_nse_data(list(FAILSAFE_COMPANIES.values()))
                st.dataframe(d)
    with t3:
        news_topic = st.selectbox("Topic:", ["Indian Economy", "Indian Stock Market"])
        if st.button("Fetch News"):
            news = get_google_news(news_topic)
            for n in news: st.markdown(f"[{n['title']}]({n['link']})")

elif mode == "Deep Dive Valuation":
    st.subheader("🔍 Valuation & Analysis")
    with st.form("analysis_form"):
        selected_company = st.selectbox("Search Company:", options=list(NSE_COMPANIES.keys()))
        submitted = st.form_submit_button("Run Analysis")
    if submitted:
        ticker = NSE_COMPANIES[selected_company]
        with st.spinner(f"Analyzing {ticker}..."):
            metrics, history, info_msg = analyze_stock(ticker)
            if metrics:
                pros_list, cons_list = generate_swot(metrics)
                c1, c2, c3 = st.columns(3)
                c1.metric("Overall Score", f"{metrics['total_score']}/10")
                c2.metric("Tech Strength", f"{metrics['tech_score']}/5")
                c3.metric("Fund Health", f"{metrics['fund_score']}/5")
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Forecast", "✅ SWOC", "🎩 Valuation", "🏢 Financials", "📰 News & Events"])
                with tab1: st.plotly_chart(plot_chart(history, ticker), use_container_width=True)
                with tab2: 
                    st.success("✅ STRENGTHS"); [st.write(p) for p in pros_list]
                    st.error("❌ WEAKNESSES"); [st.write(c) for c in cons_list]
                with tab3:
                    if metrics['intrinsic'] > 0: 
                        st.metric("Fair Value", f"₹{metrics['intrinsic']}")
                    else: 
                        st.error(f"Cannot calculate Fair Value. Reason: {metrics.get('val_note', 'Data Missing')}")
                with tab4: st.write(metrics.get('sector', 'No summary.'))
                with tab5:
                    company_news = get_company_news(ticker)
                    if company_news: [st.markdown(f"**[{n['title']}]({n['link']})**") for n in company_news]
                verdict = f"Fair Value: {metrics['intrinsic']}. Score: {metrics['total_score']}/10."
                pdf = create_pdf(ticker, metrics, pros_list, cons_list, verdict)
                st.download_button("Download Report", data=pdf, file_name=f"{ticker}_Report.pdf", mime="application/pdf")
            else: 
                st.error(info_msg)

elif mode == "Compare":
    st.subheader("⚖️ Head-to-Head Comparison")
    with st.form("compare_form"):
        c1, c2 = st.columns(2)
        s1_name = c1.selectbox("Stock A", options=list(NSE_COMPANIES.keys()), index=0)
        s2_name = c2.selectbox("Stock B", options=list(NSE_COMPANIES.keys()), index=1)
        submitted = st.form_submit_button("Compare Stocks")
    if submitted:
        s1 = NSE_COMPANIES[s1_name]; s2 = NSE_COMPANIES[s2_name]
        m1, _, _ = analyze_stock(s1); m2, _, _ = analyze_stock(s2)
        if m1 and m2:
            col1, col2 = st.columns(2)
            with col1: st.metric(s1, f"{m1['total_score']}/10")
            with col2: st.metric(s2, f"{m2['total_score']}/10")
            st.success(f"🏆 Winner: {s1 if m1['total_score'] > m2['total_score'] else s2}")