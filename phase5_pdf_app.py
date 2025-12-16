import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import feedparser # <--- NEW: For Google News RSS
from datetime import datetime
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

# --- 6. NEWS ENGINE (NEW!) ---
@st.cache_data(ttl=3600) # Cache news for 1 hour
def get_google_news(query):
    """Fetches news from Google News RSS Feed"""
    try:
        # URL encode the query
        encoded_query = query.replace(" ", "+")
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        
        news_items = []
        for entry in feed.entries[:8]: # Top 8 items
            news_items.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published,
                "source": entry.source.title
            })
        return news_items
    except Exception:
        return []

@st.cache_data(ttl=3600)
def get_company_news(ticker):
    """Fetches specific company news from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        cleaned_news = []
        for n in news[:5]:
            cleaned_news.append({
                "title": n['title'],
                "link": n['link'],
                "publisher": n.get('publisher', 'Yahoo Finance'),
                "time": datetime.fromtimestamp(n['providerPublishTime']).strftime('%Y-%m-%d')
            })
        return cleaned_news
    except:
        return []

# --- 7. CORE ANALYTICS ---
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
        
        intrinsic_value = 0
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
            "book_value": book_value,
            "sector": info.get('sector', 'General')
        }
        return metrics, df, info
    except Exception as e:
        return None, None, str(e)

# --- 8. HELPER FUNCTIONS ---
def generate_swot(m):
    pros = []
    cons = []
    if m['pe'] > 0 and m['pe'] < 25: pros.append(f"Valuation is attractive (P/E {m['pe']}).")
    elif m['pe'] > 50: cons.append(f"Stock is expensive (High P/E {m['pe']}).")
    if m['margins'] > 15: pros.append(f"High Profit Margins ({m['margins']}%).")
    elif m['margins'] < 5: cons.append(f"Thin Profit Margins ({m['margins']}%).")
    if m['debt'] < 50: pros.append("Company has low debt levels.")
    elif m['debt'] > 150: cons.append(f"High Debt-to-Equity ratio ({m['debt']}%).")
    if m['rsi'] < 30: pros.append("Technically Oversold (Good entry point?).")
    elif m['rsi'] > 70: cons.append("Technically Overbought (Risk of correction).")
    if "UP" in m['trend']: pros.append("Trading above 200-Day EMA (Long-term Uptrend).")
    else: cons.append("Trading below 200-Day EMA (Long-term Downtrend).")
    if m['intrinsic'] > 0 and m['price'] < m['intrinsic']: pros.append("Trading below Graham's Intrinsic Value.")
    return pros, cons

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
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date, periods=31)[1:]
            fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', line=dict(color='#FFA500', width=2, dash='dash'), name='AI Projected Path'))
    except: pass
    fig.update_layout(title=f"{ticker} - Analysis & Forecast", xaxis_rangeslider_visible=False, height=600)
    return fig

def create_pdf(ticker, data, pros, cons, verdict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Investment Memo: {ticker}", ln=True, align='C')
    pdf.ln(10)
    clean_price = str(data['price']).replace("₹", "")
    pdf.cell(200, 10, txt=f"Price: Rs. {clean_price} | Score: {data['total_score']}/10", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(200, 10, txt="PROS:", ln=True)
    pdf.set_font("Arial", size=10)
    for p in pros: pdf.cell(200, 6, txt=f"- {p}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(200, 10, txt="CONS:", ln=True)
    pdf.set_font("Arial", size=10)
    for c in cons: pdf.cell(200, 6, txt=f"- {c}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, txt=f"VERDICT: {verdict}")
    return pdf.output(dest='S').encode('latin-1', 'replace')

# --- 9. DASHBOARD UI ---
with st.sidebar:
    st.title(f"👤 {st.session_state['user_name']}")
    st.markdown("---")
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
# MODE 1: MARKET SCANNER (With News)
# ==========================================
if mode == "Market Scanner":
    st.subheader("📡 Market Radar")
    t1, t2, t3 = st.tabs(["Sector Leaders", "Value Hunters", "📰 Market News"])
    
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

    # --- NEW NEWS TAB ---
    with t3:
        st.markdown("### 🌍 Global & Local Market Updates")
        news_topic = st.selectbox("Select Topic:", ["Indian Economy", "Indian Stock Market", "International Markets", "Stock Market Key Events"])
        
        if st.button("Fetch News"):
            with st.spinner(f"Fetching news for {news_topic}..."):
                news_list = get_google_news(news_topic)
                if news_list:
                    for news in news_list:
                        st.markdown(f"**[{news['title']}]({news['link']})**")
                        st.caption(f"Source: {news['source']} | {news['published']}")
                        st.divider()
                else:
                    st.error("No news found. Google News RSS might be blocked.")

# ==========================================
# MODE 2: DEEP DIVE (With News)
# ==========================================
elif mode == "Deep Dive Valuation":
    st.subheader("🔍 Valuation & Analysis")
    with st.form("analysis_form"):
        ticker = st.text_input("Enter Ticker (e.g. RELIANCE.NS):", value="RELIANCE.NS")
        submitted = st.form_submit_button("Run Analysis")
    
    if submitted:
        with st.spinner(f"Analyzing {ticker}..."):
            metrics, history, info = analyze_stock(ticker)
            if metrics:
                pros_list, cons_list = generate_swot(metrics)
                c1, c2, c3 = st.columns(3)
                c1.metric("Overall Score", f"{metrics['total_score']}/10")
                c2.metric("Tech Strength", f"{metrics['tech_score']}/5")
                c3.metric("Fund Health", f"{metrics['fund_score']}/5")
                
                # --- ADDED "NEWS & EVENTS" TAB ---
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Forecast", "✅ SWOC", "🎩 Valuation", "🏢 Financials", "📰 News & Events"])
                
                with tab1: 
                    st.plotly_chart(plot_chart(history, ticker), use_container_width=True)
                
                with tab2:
                    c_pros, c_cons = st.columns(2)
                    with c_pros:
                        st.success("✅ STRENGTHS")
                        for p in pros_list: st.write(f"• {p}")
                    with c_cons:
                        st.error("❌ WEAKNESSES")
                        for c in cons_list: st.write(f"• {c}")

                with tab3:
                    if metrics['intrinsic'] > 0:
                        st.metric("Fair Value", f"₹{metrics['intrinsic']}", 
                             delta=f"{round(((metrics['intrinsic']-metrics['price'])/metrics['price'])*100, 1)}% Potential" if metrics['intrinsic'] > metrics['price'] else "Premium")
                    else: st.error("Cannot calculate Fair Value.")
                    
                with tab4: st.write(info.get('longBusinessSummary', 'No summary.'))

                # --- NEW NEWS CONTENT ---
                with tab5:
                    st.markdown(f"### 📰 Latest News for {ticker}")
                    company_news = get_company_news(ticker)
                    if company_news:
                        for n in company_news:
                            st.markdown(f"**[{n['title']}]({n['link']})**")
                            st.caption(f"Source: {n['publisher']} | {n['time']}")
                            st.divider()
                    else: st.info("No specific company news found.")
                    
                    st.markdown(f"### 🏭 Industry Trends ({metrics['sector']})")
                    # Fetch Industry News using the Sector Name
                    ind_news = get_google_news(f"Indian {metrics['sector']} Sector")
                    if ind_news:
                        for n in ind_news[:3]:
                            st.markdown(f"**[{n['title']}]({n['link']})**")
                            st.caption(f"Source: {n['source']}")

                verdict = f"Fair Value: {metrics['intrinsic']}. Score: {metrics['total_score']}/10."
                pdf = create_pdf(ticker, metrics, pros_list, cons_list, verdict)
                st.download_button("Download Report", data=pdf, file_name=f"{ticker}_Report.pdf", mime="application/pdf")

elif mode == "Compare":
    st.subheader("⚖️ Head-to-Head Comparison")
    with st.form("compare_form"):
        c1, c2 = st.columns(2)
        s1 = c1.text_input("Stock A", "TCS.NS"); s2 = c2.text_input("B", "INFY.NS")
        submitted = st.form_submit_button("Compare Stocks")
    if submitted:
        m1, _, _ = analyze_stock(s1); m2, _, _ = analyze_stock(s2)
        if m1 and m2:
            col1, col2 = st.columns(2)
            with col1: st.metric(s1, f"{m1['total_score']}/10"); st.caption(f"Fair Val: {m1['intrinsic']}")
            with col2: st.metric(s2, f"{m2['total_score']}/10"); st.caption(f"Fair Val: {m2['intrinsic']}")
            st.success(f"🏆 Winner: {s1 if m1['total_score'] > m2['total_score'] else s2}")