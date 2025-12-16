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

# --- 3. EXPANDED STOCK DATABASE (Top 10 per Sector) ---
SECTORS = {
    "Blue Chips (Top 20)": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LICI.NS",
        "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS"
    ],
    "IT Sector": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", 
        "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "OFSS.NS"
    ],
    "Banking & Finance": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", 
        "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "IDFCFIRSTB.NS", "BAJAJFINSV.NS"
    ],
    "Auto": [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", 
        "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BHARATFORG.NS", "TIINDIA.NS"
    ],
    "Pharma": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS", 
        "AUROPHARMA.NS", "ALKEM.NS", "TORNTPHARM.NS", "MANKIND.NS", "ZYDUSLIFE.NS"
    ],
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS",
        "GODREJCP.NS", "DABUR.NS", "MARICO.NS", "COLPAL.NS", "VARUN.NS"
    ]
}

# --- 4. SECURITY SYSTEM ---
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

# --- 5. ANALYTICS ENGINES ---

# A. MARKET PULSE (Top Banner)
def get_market_pulse():
    try:
        market = yf.Ticker("^NSEI")
        df = market.history(period="5d", interval="15m")
        if df.empty: return None
        current_price = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[0]
        change = current_price - prev_close
        pct = (change / prev_close) * 100
        trend = "BULLISH 🐂" if current_price > df["Close"].mean() else "BEARISH 🐻"
        return {"price": round(current_price, 2), "change": round(change, 2), "pct": round(pct, 2), "trend": trend, "data": df}
    except: return None

# B. CORE ANALYZER (Deep Dive)
@st.cache_data(ttl=24*3600)
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if df.empty: return None, None, None
        
        info = stock.info
        ema_200 = EMAIndicator(close=df["Close"], window=200).ema_indicator().iloc[-1]
        rsi = RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
        stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"]).stoch().iloc[-1]
        macd = MACD(close=df["Close"])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        bb = BollingerBands(close=df["Close"], window=20)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        current_price = df["Close"].iloc[-1]

        # Fundamentals
        pe = info.get('trailingPE', 0) or 0
        margins = info.get('profitMargins', 0) or 0
        debt = info.get('debtToEquity', 0) or 0
        roe = info.get('returnOnEquity', 0) or 0
        
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
            "margins": round(margins * 100, 2),
            "roe": round(roe * 100, 2),
            "debt": round(debt, 2),
            "trend": "UP 🟢" if current_price > ema_200 else "DOWN 🔴"
        }
        return metrics, df, info
    except: return None, None, None

# C. SCANNER ENGINE (Enhanced for All-Time/52W)
@st.cache_data(ttl=12*3600)
def run_scanner(category_list):
    results = []
    for ticker in category_list:
        try:
            stock = yf.Ticker(ticker)
            # 5y history is enough to approximate "Long Term Lows" for speed
            hist = stock.history(period="5y") 
            if hist.empty: continue
            
            price = hist["Close"].iloc[-1]
            
            # 52 Week Logic
            last_1y = hist.tail(252)
            low_52 = last_1y["Low"].min()
            high_52 = last_1y["High"].max()
            
            # All Time Logic (approx 5y)
            low_all = hist["Low"].min()
            high_all = hist["High"].max()
            
            dist_52_low = ((price - low_52) / low_52) * 100
            dist_all_low = ((price - low_all) / low_all) * 100
            
            results.append({
                "Ticker": ticker,
                "Price": round(price, 2),
                "52W Low": round(low_52, 2),
                "Dist 52W Low (%)": round(dist_52_low, 2),
                "5Y Low": round(low_all, 2),
                "Dist 5Y Low (%)": round(dist_all_low, 2)
            })
        except: continue
        
    return pd.DataFrame(results)

# --- 6. PLOTTING & PDF ---
def plot_advanced_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='gray', width=1), name='BB Upper'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', width=1), name='BB Lower'))
    fig.update_layout(title=f"{ticker} - Interactive Analysis", xaxis_rangeslider_visible=False, height=500)
    return fig

def create_pdf(ticker, data, text):
    clean_trend = data['trend'].replace("🟢", "").replace("🔴", "").strip()
    clean_price = str(data['price']).replace("₹", "Rs. ")
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Investment Memo: {ticker}", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Price: Rs. {clean_price} | Tech Score: {data['tech_score']}/5 | Fund Score: {data['fund_score']}/5", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=clean_text)
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, txt="Generated by AI Agent. Educational Use Only.")
    return pdf.output(dest='S').encode('latin-1')

# --- 7. DASHBOARD UI ---

# === SIDEBAR ===
with st.sidebar:
    st.title(f"👤 {st.session_state['user_name']}")
    st.markdown("---")
    
    if st.session_state["user_role"] == "admin":
        mode = st.radio("Mode:", ["Market Scanner (New!)", "Hybrid Deep Dive", "Competitor Comparison"])
    else:
        mode = st.radio("Mode:", ["Hybrid Deep Dive"])
        
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

# === TOP BANNER ===
st.title("📊 Principal Hybrid Engine")
with st.expander("🇮🇳 NSE NIFTY 50 Live Pulse", expanded=True):
    pulse = get_market_pulse()
    if pulse:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Nifty 50", f"{pulse['price']}", f"{pulse['pct']}%")
        m2.metric("Market Sentiment", pulse['trend'])
        m3.metric("5-Day Change", f"{pulse['change']}")
        with m4: st.line_chart(pulse['data']['Close'], height=100)

# === MODE 1: MARKET SCANNER ===
if mode == "Market Scanner (New!)":
    st.subheader("📡 Market Discovery Radar")
    
    # Define tabs for the different views requested
    tab1, tab2, tab3 = st.tabs(["🏢 Sector Leaders", "📉 52-Week Lows", "⚠️ All-Time Lows (5Y)"])
    
    # Flatten list for broad scans
    all_stocks = list(set([item for sublist in SECTORS.values() for item in sublist]))
    
    # --- TAB 1: SECTORS & BLUE CHIPS ---
    with tab1:
        st.markdown("### Top Companies by Sector")
        sector_choice = st.selectbox("Select List:", list(SECTORS.keys()))
        
        if st.button(f"Scan {sector_choice}"):
            with st.spinner(f"Scanning {sector_choice}..."):
                tickers = SECTORS[sector_choice]
                # Quick Cards for top 4
                cols = st.columns(4)
                for i, ticker in enumerate(tickers[:4]):
                    m, _, _ = analyze_stock(ticker)
                    if m:
                        with cols[i]:
                            st.metric(ticker, f"₹{m['price']}", delta=f"Score: {m['total_score']}/10")
                # Full Table
                st.dataframe(run_scanner(tickers))

    # --- TAB 2: 52-WEEK LOWS ---
    with tab2:
        st.markdown("### 💎 Value Hunter: Near 52-Week Low")
        st.caption("Stocks from our database trading within 5% of their yearly low.")
        
        if st.button("Find Value Buys"):
            with st.spinner("Scanning market for discounts..."):
                df_scan = run_scanner(all_stocks)
                # Filter: Price is within 5% of 52W Low (Dist < 5)
                df_lows = df_scan[df_scan["Dist 52W Low (%)"] < 10].sort_values("Dist 52W Low (%)")
                
                if not df_lows.empty:
                    st.success(f"Found {len(df_lows)} stocks near 52-week lows.")
                    st.dataframe(df_lows[["Ticker", "Price", "52W Low", "Dist 52W Low (%)"]], hide_index=True)
                else:
                    st.info("No stocks are currently near their 52-week lows. Market is strong!")

    # --- TAB 3: ALL-TIME LOWS ---
    with tab3:
        st.markdown("### ⚠️ Historic Lows (5-Year View)")
        st.caption("Stocks trading near their lowest price in 5 years (High Risk / Reversal candidates).")
        
        if st.button("Scan Historic Lows"):
            with st.spinner("Analyzing deep history..."):
                df_scan = run_scanner(all_stocks)
                # Filter: Price is near 5Y Low
                df_historic = df_scan[df_scan["Dist 5Y Low (%)"] < 5].sort_values("Dist 5Y Low (%)")
                
                if not df_historic.empty:
                    st.warning(f"Found {len(df_historic)} stocks near 5-year lows.")
                    st.dataframe(df_historic[["Ticker", "Price", "5Y Low", "Dist 5Y Low (%)"]], hide_index=True)
                else:
                    st.success("Good news! No major Blue Chip stocks are at historic lows right now.")

# === MODE 2: HYBRID DEEP DIVE ===
elif mode == "Hybrid Deep Dive":
    st.subheader("🔍 Deep Dive Analysis")
    ticker = st.text_input("Ticker Symbol:", value="RELIANCE.NS")
    if st.button("Run Analysis"):
        with st.spinner("Analyzing..."):
            metrics, history, info = analyze_stock(ticker)
            if metrics:
                c1, c2, c3 = st.columns(3)
                c1.metric("Overall Score", f"{metrics['total_score']}/10")
                c2.metric("Technical", f"{metrics['tech_score']}/5")
                c3.metric("Fundamental", f"{metrics['fund_score']}/5")
                
                tab1, tab2 = st.tabs(["Chart", "Financials"])
                with tab1: st.plotly_chart(plot_advanced_chart(history, ticker), use_container_width=True)
                with tab2: st.write(info.get('longBusinessSummary'))
                
                verdict = f"Score: {metrics['total_score']}/10. Trend: {metrics['trend']}."
                pdf = create_pdf(ticker, metrics, verdict)
                st.download_button("Download Report", data=pdf, file_name=f"{ticker}.pdf", mime="application/pdf")

# === MODE 3: COMPARISON ===
elif mode == "Competitor Comparison":
    st.subheader("⚖️ Benchmarking")
    c1, c2 = st.columns(2)
    with c1: s1 = st.text_input("Stock A", "TCS.NS")
    with c2: s2 = st.text_input("Stock B", "INFY.NS")
    if st.button("Compare"):
        m1, h1, f1 = analyze_stock(s1)
        m2, h2, f2 = analyze_stock(s2)
        if m1 and m2:
            col1, col2 = st.columns(2)
            col1.metric(s1, m1['total_score'], delta=f"Tech: {m1['tech_score']}")
            col2.metric(s2, m2['total_score'], delta=f"Tech: {m2['tech_score']}")
            st.success(f"Winner: {s1 if m1['total_score'] > m2['total_score'] else s2}")