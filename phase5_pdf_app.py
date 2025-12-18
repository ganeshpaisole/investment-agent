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

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Principal AI Agent", layout="wide")

# --- 2. SECURITY CONFIG ---
USER_ROLES = {
    "admin": {"role": "admin", "name": "Principal Consultant"},
    "client": {"role": "viewer", "name": "Valued Client"}
}

# --- 3. DATABASE ENGINE ---
AI_COMPANIES = [
    "PERSISTENT.NS", "HAPPISTMNDS.NS", "TATAELXSI.NS", "AFFLE.NS", "SAKSOFT.NS", 
    "OFSS.NS", "CYIENT.NS", "ZENSARTECH.NS", "RATEGAIN.NS", "KPITTECH.NS", 
    "LTTS.NS", "COFORGE.NS", "MPHASIS.NS", "SONATSOFTW.NS"
]

SEMI_EV_COMPANIES = [
    "TATAMOTORS.NS", "OLECTRA.NS", "JBMA.NS", "DIXON.NS", "MOSCHIP.NS", 
    "SPEL.NS", "ASMTEC.NS", "MICEL.NS", "EXIDEIND.NS", "AMARAJABAT.NS", 
    "KAYNES.NS", "CGPOWER.NS", "BEL.NS", "TIINDIA.NS", "GREAVESCOT.NS", 
    "HIMATSEIDE.NS", "ELECTCAST.NS", "JBM_AUTO.NS"
]

FAILSAFE_COMPANIES = {
    "Reliance Industries": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS", "Infosys": "INFY.NS", "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "ITC": "ITC.NS", "Larsen & Toubro": "LT.NS",
    "Hindustan Unilever": "HINDUNILVR.NS", "Maruti Suzuki": "MARUTI.NS",
    "Mahindra & Mahindra": "M&M.NS", "Tata Motors": "TATAMOTORS.NS", "Bajaj Finance": "BAJFINANCE.NS",
    "Persistent Systems": "PERSISTENT.NS", "Happiest Minds": "HAPPISTMNDS.NS", 
    "Tata Elxsi": "TATAELXSI.NS", "KPIT Tech": "KPITTECH.NS",
    "Dixon Tech": "DIXON.NS", "Olectra Greentech": "OLECTRA.NS", "Moschip": "MOSCHIP.NS",
    "HAL": "HAL.NS", "BEL": "BEL.NS", "Zomato": "ZOMATO.NS", "Trent": "TRENT.NS"
}

@st.cache_data(ttl=24*3600)
def load_all_nse_tickers():
    master_dict = FAILSAFE_COMPANIES.copy()
    ticker_list = list(FAILSAFE_COMPANIES.values())
    try:
        url = "https://raw.githubusercontent.com/sfini/NSE-Data/master/EQUITY_L.csv"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            for index, row in df.iterrows():
                symbol = row['SYMBOL']
                name = row['NAME OF COMPANY']
                if symbol == "VARUN": y_ticker = "VBL.NS"
                elif symbol == "REC": y_ticker = "RECLTD.NS"
                else: y_ticker = f"{symbol}.NS"
                master_dict[f"{name} ({symbol})"] = y_ticker
                ticker_list.append(y_ticker)
    except: pass
    return master_dict, list(set(ticker_list))

NSE_COMPANIES, ALL_NSE_TICKERS = load_all_nse_tickers()

SECTORS = {
    "Blue Chips": list(FAILSAFE_COMPANIES.values())[:10],
    "Artificial Intelligence (AI)": AI_COMPANIES,
    "Semiconductor & EV": SEMI_EV_COMPANIES,
    "Banks": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "TVSMOTOR.NS"],
    "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"]
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
def get_market_pulse(period="1d"):
    try:
        interval = "5m" if period == "1d" else "1d"
        if period in ["5d", "1mo"]: interval = "15m" if period == "5d" else "60m"
        df = yf.Ticker("^NSEI").history(period=period, interval=interval)
        if df.empty: return None
        price = df["Close"].iloc[-1]
        start_price = df["Open"].iloc[0]
        change_val = price - start_price
        pct_val = (change_val / start_price) * 100
        return {"price": round(price, 2), "change": round(change_val, 2), "pct": round(pct_val, 2), "trend": "BULLISH 🐂" if change_val > 0 else "BEARISH 🐻", "data": df}
    except: return None

def plot_market_pulse_chart(data, period):
    fig = go.Figure()
    if period in ["1d", "5d", "1mo"]:
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Nifty 50"))
    else:
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name="Nifty 50", line=dict(color='#00C805', width=2)))
    fig.update_layout(title=f"Nifty 50 - {period.upper()} View", height=400, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark")
    return fig

# --- 6. CORE ANALYTICS ---
@st.cache_data(ttl=3600)
def analyze_stock(ticker):
    ticker = str(ticker).strip().upper()
    max_retries = 3
    stock = None; df = pd.DataFrame(); info = {}
    
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            if not df.empty:
                info = stock.info
                break
            time.sleep(random.uniform(1.0, 3.0)) 
        except: time.sleep(1)
            
    if df.empty: return None, None, f"⚠️ Server Busy. Please try '{ticker}' again later."

    try:
        current_price = df["Close"].iloc[-1]
        try: ema_200 = EMAIndicator(close=df["Close"], window=200).ema_indicator().iloc[-1]
        except: ema_200 = current_price
        rsi = RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
        stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"]).stoch().iloc[-1]
        macd = MACD(close=df["Close"])
        df['MACD'] = macd.macd(); df['MACD_Signal'] = macd.macd_signal()
        bb = BollingerBands(close=df["Close"], window=20)
        df['BB_High'] = bb.bollinger_hband(); df['BB_Low'] = bb.bollinger_lband()

        def get_val(keys):
            for k in keys:
                if k in info and info[k] is not None: return info[k]
            return 0

        pe = get_val(['trailingPE', 'forwardPE'])
        pb = get_val(['priceToBook'])
        eps = get_val(['trailingEps', 'forwardEps'])
        if eps == 0 and pe > 0: eps = current_price / pe
        book_value = get_val(['bookValue'])
        if book_value == 0 and pb > 0: book_value = current_price / pb
        margins = get_val(['profitMargins'])
        debt = get_val(['debtToEquity'])
        roe = get_val(['returnOnEquity'])
        peg = get_val(['pegRatio'])
        beta = get_val(['beta'])
        revenue = get_val(['totalRevenue'])
        net_income = get_val(['netIncomeToCommon'])
        if revenue == 0 or net_income == 0:
            try:
                fins = stock.financials
                if not fins.empty:
                    if revenue == 0: revenue = fins.loc['Total Revenue'].iloc[0]
                    if net_income == 0: net_income = fins.loc['Net Income'].iloc[0]
            except: pass
        op_margin = get_val(['operatingMargins'])
        roa = get_val(['returnOnAssets'])

        intrinsic_value = 0
        valuation_note = ""
        if eps > 0 and book_value > 0:
            intrinsic_value = math.sqrt(22.5 * eps * book_value)
            valuation_note = "Graham Number"
        if intrinsic_value == 0:
            target = get_val(['targetMeanPrice', 'targetMedianPrice'])
            if target > 0:
                intrinsic_value = target
                valuation_note = "Analyst Target"
            else:
                intrinsic_value = current_price 
                valuation_note = "Market Price"

        t_score = sum([current_price > ema_200, 40 < rsi < 70, stoch < 80, df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]])
        f_score = sum([0 < pe < 40, margins > 0.10, debt < 100, roe > 0.15])

        metrics = {
            "price": round(current_price, 2), "tech_score": t_score, "fund_score": f_score, "total_score": t_score + f_score,
            "rsi": round(rsi, 2), "pe": round(pe, 2), "margins": round(margins*100, 2), "roe": round(roe*100, 2),
            "debt": round(debt, 2), "peg": peg, "beta": beta, "trend": "UP 🟢" if current_price > ema_200 else "DOWN 🔴",
            "intrinsic": round(intrinsic_value, 2), "val_note": valuation_note, "eps": eps, "book_value": book_value,
            "sector": info.get('sector', 'General'), "revenue": revenue, "net_income": net_income, "op_margin": op_margin, "roa": roa
        }
        return metrics, df, info
    except Exception as e: return None, None, f"⚠️ Analysis Error: {str(e)}"

# --- 7. HELPER FUNCTIONS ---
def generate_key_factors(m):
    factors = []
    if m['pe'] < 20 and m['pe'] > 0: factors.append("🟢 **Attractive Valuation:** P/E Ratio is low.")
    elif m['pe'] > 50: factors.append("🔴 **Expensive Valuation:** P/E Ratio is high.")
    if m['roe'] > 15: factors.append(f"🟢 **High Efficiency:** ROE is {m['roe']}%.")
    else: factors.append(f"🟠 **Low Efficiency:** ROE is {m['roe']}%.")
    if m['rsi'] > 70: factors.append("🔴 **Overbought:** RSI > 70.")
    elif m['rsi'] < 30: factors.append("🟢 **Oversold:** RSI < 30.")
    else: factors.append("⚪ **Neutral Momentum:** RSI is healthy.")
    return factors

def generate_swot(m):
    pros, cons = [], []
    if m['pe'] > 0 and m['pe'] < 25: pros.append(f"✅ **Attractive Valuation:** P/E of {m['pe']} is reasonable.")
    if m['margins'] > 10: pros.append(f"✅ **High Profitability:** Net margins of {m['margins']}% are healthy.")
    if m['roe'] > 15: pros.append(f"✅ **Efficient Management:** Return on Equity is strong at {m['roe']}%.")
    if m['debt'] < 50: pros.append("✅ **Low Debt:** Company has a safe Debt-to-Equity ratio.")
    if m['intrinsic'] > m['price']: pros.append(f"✅ **Undervalued:** Trading below fair value ({m['val_note']}).")
    if m['trend'] == "UP 🟢": pros.append("✅ **Uptrend:** Stock is trading above its 200-Day Moving Average.")
    if len(pros) < 3:
        if m['beta'] < 1: pros.append("✅ **Low Volatility:** Stock is less volatile than the market.")
        if m['rsi'] < 40: pros.append("✅ **Oversold Zone:** RSI indicates potential for a bounce.")
        if m['revenue'] > 0: pros.append("✅ **Revenue Generating:** Company has established revenue streams.")
    
    if m['pe'] > 50: cons.append(f"❌ **Expensive:** P/E of {m['pe']} is quite high.")
    if m['margins'] < 5: cons.append(f"❌ **Thin Margins:** Net margins are low ({m['margins']}%) or negative.")
    if m['roe'] < 10: cons.append(f"❌ **Low Efficiency:** ROE of {m['roe']}% is below par.")
    if m['debt'] > 100: cons.append(f"❌ **High Debt:** Debt-to-Equity ratio is high ({m['debt']}%).")
    if m['intrinsic'] < m['price']: cons.append("❌ **Overvalued:** Trading above calculated fair value.")
    if m['trend'] == "DOWN 🔴": cons.append("❌ **Downtrend:** Stock is trading below its 200-Day Moving Average.")
    if len(cons) < 3:
        if m['beta'] > 1.5: cons.append("❌ **High Volatility:** Stock is significantly more volatile than the market.")
        if m['rsi'] > 70: cons.append("❌ **Overbought:** RSI indicates stock might correct soon.")
        if m['val_note'] == "Market Price": cons.append("❌ **Lack of Fundamental Data:** Valuation is uncertain.")
    return pros[:5], cons[:5]

def plot_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='History'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='gray', width=1), name='BB Upper'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', width=1), name='BB Lower'))
    try:
        recent_df = df.tail(90).copy()
        if len(recent_df) > 20:
            x = np.arange(len(recent_df))
            y = recent_df['Close'].values; coeffs = np.polyfit(x, y, 2); poly_curve = np.poly1d(coeffs)
            future_x = np.arange(len(recent_df), len(recent_df) + 30); future_prices = poly_curve(future_x)
            fig.add_trace(go.Scatter(x=pd.date_range(start=df.index[-1], periods=31)[1:], y=future_prices, mode='lines', line=dict(color='#FFA500', width=2, dash='dash'), name='AI Projected Path'))
    except: pass
    fig.update_layout(title=f"{ticker} - Analysis & Forecast", xaxis_rangeslider_visible=False, height=600)
    return fig

def create_pdf(ticker, data, pros, cons, verdict):
    pdf = FPDF()
    pdf.add_page(); pdf.set_font("Arial", size=12)
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
        news = [{"title": n['title'], "link": n['link'], "publisher": n.get('publisher', 'Yahoo')} for n in yf.Ticker(ticker).news[:5]]
        if len(news) > 0: return news
    except: pass
    return get_google_news(f"{ticker} stock news")

# --- 8. SCANNER ENGINES ---
# A. QUANTUM SCANNER (YahooQuery) - For Aimagica
def run_quantum_scan(ticker_list, batch_size=50):
    valid_results = []
    progress_bar = st.progress(0)
    
    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i : i + batch_size]
        if not batch: continue
        try:
            progress_bar.progress((i / len(ticker_list)))
            yq = YQ(batch)
            data = yq.get_modules("summaryDetail price defaultKeyStatistics")
            for symbol in batch:
                if symbol not in data: continue
                try:
                    summary = data[symbol].get('summaryDetail', {})
                    price_mod = data[symbol].get('price', {})
                    if isinstance(summary, str) or isinstance(price_mod, str): continue
                    price = price_mod.get('regularMarketPrice', 0)
                    pe = summary.get('trailingPE', 0)
                    volume = summary.get('volume', 0)
                    
                    if price > 0 and volume > 5000: # Loose criteria
                        score = 0
                        if 0 < pe < 30: score += 40
                        elif 0 < pe < 50: score += 20
                        if price_mod.get('regularMarketChangePercent', 0) > 0: score += 20
                        
                        valid_results.append({
                            "Ticker": symbol,
                            "Price": price,
                            "P/E": round(pe, 2) if pe else "N/A",
                            "Aimagica Score": score + random.randint(1, 10)
                        })
                except: continue
        except: continue
            
    progress_bar.empty()
    df = pd.DataFrame(valid_results)
    if not df.empty: return df.sort_values("Aimagica Score", ascending=False).head(5)
    return pd.DataFrame()

# B. BASIC SCANNER (YFinance) - For Market Scanner Tab (RESTORED)
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

# --- 9. NOTIFICATION ENGINE ---
def send_email_alert(subject, body):
    try:
        sender = st.secrets["notifications"]["email_sender"]
        password = st.secrets["notifications"]["email_password"]
        receiver = st.secrets["notifications"]["email_receiver"]
        msg = MIMEMultipart(); msg['From'] = sender; msg['To'] = receiver; msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender, password); server.sendmail(sender, receiver, msg.as_string())
        return True, "Email Sent"
    except Exception as e: return False, str(e)

def send_whatsapp_alert(body):
    try:
        sid = st.secrets["notifications"]["twilio_sid"]; token = st.secrets["notifications"]["twilio_token"]
        from_num = st.secrets["notifications"]["twilio_from"]; to_num = st.secrets["notifications"]["twilio_to"]
        client = Client(sid, token)
        message = client.messages.create(body=body, from_=from_num, to=to_num)
        return True, f"WhatsApp Sent (SID: {message.sid})"
    except Exception as e: return False, str(e)

def trigger_daily_report():
    return "Test Triggered"

# --- 10. DASHBOARD UI ---
with st.sidebar:
    st.title(f"👤 {st.session_state['user_name']}")
    st.markdown("---")
    if st.session_state["user_role"] == "admin": 
        mode = st.radio("Mode:", ["Aimagica (Golden 5)", "Market Scanner", "Deep Dive Valuation", "Compare"]) 
    else: mode = st.radio("Mode:", ["Deep Dive Valuation"])
    if st.button("Logout"): st.session_state.update({"logged_in": False}); st.rerun()

st.title("📊 Principal Hybrid Engine")

with st.expander("🇮🇳 NSE Market Pulse (Live)", expanded=True):
    col_sel, col_data = st.columns([1, 4])
    with col_sel: timeframe = st.radio("Timeframe", ["1D", "5D", "1M", "6M", "1Y", "5Y"], horizontal=True, index=0)
    tf_map = {"1D": "1d", "5D": "5d", "1M": "1mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}
    p_data = get_market_pulse(tf_map[timeframe])
    if p_data:
        m1, m2, m3 = st.columns(3)
        m1.metric("Nifty 50 Level", f"₹{p_data['price']}"); m2.metric("Change", f"{p_data['change']}", f"{p_data['pct']}%"); m3.metric("Trend", p_data['trend'])
        st.plotly_chart(plot_market_pulse_chart(p_data['data'], tf_map[timeframe]), use_container_width=True)

if mode == "Aimagica (Golden 5)":
    st.subheader("✨ Aimagica: The Golden Opportunity Engine")
    c1, c2, c3 = st.columns(3)
    scan_triggered = False; target_list = []
    
    with c1:
        if st.button("🔮 Mine General Market"):
            target_list = list(NSE_COMPANIES.values())
            if len(ALL_NSE_TICKERS) > 100: target_list += random.sample(ALL_NSE_TICKERS, 50)
            st.session_state['scan_type'] = "General Market"
            scan_triggered = True
    with c2:
        if st.button("🤖 Scan AI Sector"):
            target_list = AI_COMPANIES
            st.session_state['scan_type'] = "AI Sector"
            scan_triggered = True
    with c3:
        if st.button("⚡ Scan Semi & EV"):
            target_list = SEMI_EV_COMPANIES
            st.session_state['scan_type'] = "Semi & EV"
            scan_triggered = True
            
    if scan_triggered:
        with st.spinner(f"Scanning {len(target_list)} stocks..."):
            top_stocks = run_quantum_scan(target_list)
            if not top_stocks.empty:
                st.success(f"✅ Found Top Opportunities in {st.session_state['scan_type']}")
                st.dataframe(top_stocks, hide_index=True, use_container_width=True)
                st.markdown("### 🔎 Quick Analysis of Winner")
                winner = top_stocks.iloc[0]['Ticker']
                if st.button(f"Deep Dive into {winner}"): st.write(f"Go to 'Deep Dive Valuation' tab and select {winner} for details.")
            else: st.warning("No stocks met the criteria right now. Try again later.")

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
                key_factors = generate_key_factors(metrics)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Overall Score", f"{metrics['total_score']}/10")
                c2.metric("Current Price", f"₹{metrics['price']}")
                c3.metric("Tech Strength", f"{metrics['tech_score']}/5")
                c4.metric("Fund Health", f"{metrics['fund_score']}/5")
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Forecast", "🔑 Key Factors", "✅ SWOT", "🎩 Valuation", "🏢 Financials", "📰 News & Events"])
                with tab1: st.plotly_chart(plot_chart(history, ticker), use_container_width=True)
                with tab2: 
                    st.subheader("Drivers of Stock Value")
                    for factor in key_factors: st.markdown(factor)
                with tab3:
                    st.success("✅ STRENGTHS"); [st.markdown(p) for p in pros_list]
                    st.error("❌ WEAKNESSES"); [st.markdown(c) for c in cons_list]
                with tab4:
                    if metrics['intrinsic'] > 0: 
                        delta_val = round(((metrics['intrinsic'] - metrics['price']) / metrics['price']) * 100, 1)
                        st.metric(label=f"Fair Value ({metrics['val_note']})", value=f"₹{metrics['intrinsic']}", delta=f"{delta_val}% {'Upside' if delta_val > 0 else 'Downside'}")
                    else: st.error(f"Cannot calculate Fair Value. Reason: {metrics.get('val_note', 'Data Missing')}")
                with tab5: 
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Total Revenue", f"₹{round(metrics['revenue']/10000000, 2)} Cr" if metrics['revenue'] else "N/A")
                        st.metric("Net Income", f"₹{round(metrics['net_income']/10000000, 2)} Cr" if metrics['net_income'] else "N/A")
                    with col_b:
                        st.metric("Operating Margin", f"{round(metrics['op_margin']*100, 2)}%" if metrics['op_margin'] else "N/A")
                        st.metric("Return on Assets", f"{round(metrics['roa']*100, 2)}%" if metrics['roa'] else "N/A")
                with tab6:
                    company_news = get_company_news(ticker)
                    if company_news:
                        for n in company_news: st.markdown(f"**[{n['title']}]({n['link']})**")
                    else: st.write("No recent news found.")
                verdict = f"Fair Value: {metrics['intrinsic']}. Score: {metrics['total_score']}/10."
                pdf = create_pdf(ticker, metrics, pros_list, cons_list, verdict)
                st.download_button("Download Report", data=pdf, file_name=f"{ticker}_Report.pdf", mime="application/pdf")
            else: st.error(info_msg)

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
            with col1: 
                st.metric(s1_name, f"{m1['total_score']}/10")
                st.write(f"Price: {m1['price']}")
                st.write(f"P/E: {m1['pe']}")
                st.write(f"ROE: {m1['roe']}%")
            with col2: 
                st.metric(s2_name, f"{m2['total_score']}/10")
                st.write(f"Price: {m2['price']}")
                st.write(f"P/E: {m2['pe']}")
                st.write(f"ROE: {m2['roe']}%")
            st.success(f"🏆 Winner: {s1_name if m1['total_score'] > m2['total_score'] else s2_name}")