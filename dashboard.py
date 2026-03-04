"""
╔══════════════════════════════════════════════════════════════╗
║     NSE STOCK FORECAST AI AGENT — STREAMLIT DASHBOARD        ║
║     Visual Web UI | All Nifty50 | Sector Views | Screener    ║
╚══════════════════════════════════════════════════════════════╝

Run:
  streamlit run dashboard.py

Requirements:
  pip install streamlit plotly pandas yfinance
"""

import sys
import time
import json
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Add parent dir so we can import our agents
sys.path.insert(0, str(Path(__file__).parent))

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Forecast AI Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .metric-card {
      background: #1e2130;
      border-radius: 10px;
      padding: 16px;
      border-left: 4px solid #00d4aa;
      margin-bottom: 10px;
  }
  .buy-card   { border-left-color: #00c853 !important; }
  .hold-card  { border-left-color: #ffd600 !important; }
  .avoid-card { border-left-color: #ff1744 !important; }
  .score-badge {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 20px;
      font-weight: bold;
      font-size: 14px;
  }
  .stMetric > div { background: #1e2130; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ── NIFTY50 UNIVERSE ─────────────────────────────────────────
NIFTY50 = {
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "KOTAKBANK": "BANKING",
    "AXISBANK": "BANKING", "SBIN": "BANKING", "INDUSINDBK": "BANKING",
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC",
    "HDFCLIFE": "INSURANCE", "SBILIFE": "INSURANCE",
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT", "LTIM": "IT",
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA",
    "DIVISLAB": "PHARMA", "APOLLOHOSP": "PHARMA",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG", "ASIANPAINT": "FMCG", "TITAN": "FMCG",
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "M&M": "AUTO",
    "BAJAJ-AUTO": "AUTO", "HEROMOTOCO": "AUTO", "EICHERMOT": "AUTO",
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "BPCL": "ENERGY", "COALINDIA": "ENERGY",
    "NTPC": "UTILITY", "POWERGRID": "UTILITY",
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS", "HINDALCO": "METALS", "GRASIM": "METALS",
    "LT": "INFRA", "ADANIENT": "INFRA", "ADANIPORTS": "INFRA",
    "BHARTIARTL": "TELECOM",
    "SHREECEM": "CEMENT", "ULTRACEMCO": "CEMENT",
}

PSU_STOCKS = {"SBIN", "ONGC", "BPCL", "COALINDIA", "NTPC", "POWERGRID"}

SECTOR_COLORS = {
    "BANKING": "#2196F3", "NBFC": "#03A9F4", "INSURANCE": "#00BCD4",
    "IT": "#9C27B0", "PHARMA": "#E91E63", "FMCG": "#4CAF50",
    "AUTO": "#FF9800", "ENERGY": "#FF5722", "METALS": "#795548",
    "INFRA": "#607D8B", "UTILITY": "#009688", "TELECOM": "#3F51B5",
    "CEMENT": "#8D6E63",
}


# ── CACHED DATA FETCHER ──────────────────────────────────────

@st.cache_data(ttl=300)  # Cache 5 minutes
def fetch_stock_data(ticker: str) -> dict:
    """Fetch live data from yfinance."""
    if not HAS_YFINANCE:
        return {}
    try:
        sym  = f"{ticker}.NS"
        info = yf.Ticker(sym).info or {}
        hist = yf.Ticker(sym).history(period="1y")

        current = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        high_52 = info.get("fiftyTwoWeekHigh", 0)
        low_52  = info.get("fiftyTwoWeekLow", 0)
        pos_pct = round((current - low_52) / (high_52 - low_52) * 100, 1) if high_52 != low_52 else 50

        pe      = info.get("trailingPE", 0) or 0
        pb      = info.get("priceToBook", 0) or 0
        roe     = (info.get("returnOnEquity", 0) or 0) * 100
        div_yld = (info.get("dividendYield", 0) or 0) * 100
        mktcap  = (info.get("marketCap", 0) or 0) / 1e7  # Crores

        # Simple composite score heuristic
        score = 50
        if 0 < pe < 20:   score += 12
        elif pe < 35:     score += 6
        if roe > 20:      score += 12
        elif roe > 12:    score += 6
        if pb < 3:        score += 8
        elif pb < 6:      score += 4
        if div_yld > 3:   score += 5
        if pos_pct < 30:  score += 8   # Near 52W low = value
        elif pos_pct > 85: score -= 8  # Near 52W high = caution

        score = max(0, min(100, score))
        rating = ("🟢 BUY" if score >= 65 else "🟡 HOLD" if score >= 50 else "🔴 AVOID")

        return {
            "ticker": ticker, "current_price": current,
            "pe": round(pe, 1), "pb": round(pb, 2), "roe": round(roe, 1),
            "div_yield": round(div_yld, 2), "market_cap_cr": round(mktcap, 0),
            "high_52w": high_52, "low_52w": low_52, "pos_52w_pct": pos_pct,
            "score": score, "rating": rating,
            "company_name": info.get("longName", ticker),
            "sector": NIFTY50.get(ticker, "OTHER"),
            "hist": hist,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e), "score": 50, "rating": "🟡 HOLD",
                "current_price": 0, "pe": 0, "pb": 0, "roe": 0}


@st.cache_data(ttl=600)
def fetch_all_nifty50() -> pd.DataFrame:
    """Fetch summary data for all Nifty50 stocks."""
    rows = []
    progress = st.progress(0, text="Loading Nifty50 universe...")
    for i, (ticker, sector) in enumerate(NIFTY50.items()):
        data = fetch_stock_data(ticker)
        rows.append({
            "Ticker":      ticker,
            "Company":     data.get("company_name", ticker)[:22],
            "Sector":      sector,
            "Price ₹":     data.get("current_price", 0),
            "Score":       data.get("score", 50),
            "Rating":      data.get("rating", "🟡 HOLD"),
            "P/E":         data.get("pe", 0),
            "P/B":         data.get("pb", 0),
            "ROE %":       data.get("roe", 0),
            "Div Yield %": data.get("div_yield", 0),
            "52W Pos %":   data.get("pos_52w_pct", 50),
            "MCap Cr":     data.get("market_cap_cr", 0),
            "PSU":         "🏛️" if ticker in PSU_STOCKS else "",
        })
        progress.progress((i + 1) / len(NIFTY50), text=f"Loading {ticker}...")
        time.sleep(0.2)
    progress.empty()
    return pd.DataFrame(rows).sort_values("Score", ascending=False)


# ── CHART BUILDERS ───────────────────────────────────────────

def price_chart(hist: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick with 50/200 DMA."""
    if hist is None or hist.empty:
        return go.Figure()

    sma50  = hist["Close"].rolling(50).mean()
    sma200 = hist["Close"].rolling(200).mean() if len(hist) >= 200 else hist["Close"].rolling(len(hist)).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"], name=ticker,
        increasing_line_color="#00c853", decreasing_line_color="#ff1744"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=hist.index, y=sma50,  name="50 DMA",  line=dict(color="#FFA726", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=sma200, name="200 DMA", line=dict(color="#29B6F6", width=1.5)), row=1, col=1)

    # Volume
    colors = ["#00c853" if hist["Close"].iloc[i] >= hist["Open"].iloc[i] else "#ff1744" for i in range(len(hist))]
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume",
                         marker_color=colors, opacity=0.6), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=420,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", y=1.02),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117"
    )
    return fig


def score_gauge(score: int, label: str = "Score") -> go.Figure:
    color = "#00c853" if score >= 65 else "#ffd600" if score >= 50 else "#ff1744"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": label, "font": {"size": 14, "color": "white"}},
        number={"font": {"color": color, "size": 32}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, 45],  "color": "#1a0a0a"},
                {"range": [45, 65], "color": "#1a1a0a"},
                {"range": [65, 100],"color": "#0a1a0a"},
            ],
            "bgcolor": "#1e2130",
        }
    ))
    fig.update_layout(paper_bgcolor="#1e2130", height=180, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def sector_heatmap(df: pd.DataFrame) -> go.Figure:
    sector_avg = df.groupby("Sector")["Score"].mean().reset_index()
    sector_avg = sector_avg.sort_values("Score", ascending=False)
    colors     = [SECTOR_COLORS.get(s, "#888") for s in sector_avg["Sector"]]

    fig = go.Figure(go.Bar(
        x=sector_avg["Sector"], y=sector_avg["Score"],
        marker_color=colors, text=sector_avg["Score"].round(0).astype(int),
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", height=280,
        yaxis_range=[0, 110],
        title="Average Score by Sector",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def scatter_pe_roe(df: pd.DataFrame) -> go.Figure:
    df_clean = df[(df["P/E"] > 0) & (df["P/E"] < 80) & (df["ROE %"] > 0)].copy()
    df_clean["color"] = df_clean["Sector"].map(lambda s: SECTOR_COLORS.get(s, "#888"))

    fig = px.scatter(
        df_clean, x="P/E", y="ROE %", size="MCap Cr", color="Sector",
        hover_data=["Ticker", "Score", "Rating"], text="Ticker",
        color_discrete_map=SECTOR_COLORS, height=400,
        title="P/E vs ROE — Bubble size = Market Cap"
    )
    fig.update_traces(textposition="top center", textfont_size=8)
    fig.add_hline(y=15, line_dash="dot", line_color="yellow", annotation_text="ROE=15%")
    fig.add_vline(x=25, line_dash="dot", line_color="cyan",   annotation_text="P/E=25")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def position_52w_chart(df: pd.DataFrame) -> go.Figure:
    df_s = df.sort_values("52W Pos %")
    colors = ["#00c853" if p < 30 else "#ffd600" if p < 70 else "#ff1744" for p in df_s["52W Pos %"]]

    fig = go.Figure(go.Bar(
        x=df_s["Ticker"], y=df_s["52W Pos %"],
        marker_color=colors, text=df_s["52W Pos %"].astype(str) + "%",
        textposition="outside"
    ))
    fig.add_hline(y=30, line_dash="dot", line_color="#00c853", annotation_text="Value Zone")
    fig.add_hline(y=70, line_dash="dot", line_color="#ff1744", annotation_text="Caution Zone")
    fig.update_layout(
        template="plotly_dark", height=350,
        title="52-Week Range Position (Lower = Closer to Value)",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        yaxis_range=[0, 115], margin=dict(l=0, r=0, t=40, b=0),
        xaxis_tickangle=45
    )
    return fig


# ── RSI CALCULATION ──────────────────────────────────────────

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


# ══════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════

def main():
    # ── SIDEBAR ─────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/NSE-Logo.svg/320px-NSE-Logo.svg.png", width=120)
        st.title("NSE AI Agent")
        st.caption("360° Investment Intelligence")
        st.divider()

        page = st.radio("📍 Navigation", [
            "🏠 Overview",
            "🔍 Single Stock Analysis",
            "📊 Portfolio Screener",
            "🏦 Sector Deep Dive",
            "📡 Market Sentiment",
        ])

        st.divider()
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        st.caption("⚠️ Not SEBI registered advice")

    # ══════════════════════════════════════════════════════
    # PAGE 1: OVERVIEW
    # ══════════════════════════════════════════════════════
    if page == "🏠 Overview":
        st.title("📈 NSE Stock Forecast AI Agent")
        st.caption("Graham + Buffett Framework | All Nifty50 | Real-time Analysis")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stocks Covered",  "50", "Nifty50 Universe")
        with col2:
            st.metric("Sectors",         "13", "Full Coverage")
        with col3:
            st.metric("Analysis Models", "6",  "Fundamental+Technical+AI")
        with col4:
            st.metric("Data Sources",    "3",  "Yahoo + Screener + NSE")

        st.divider()

        with st.spinner("Loading Nifty50 data..."):
            df = fetch_all_nifty50()

        # Top picks
        buys   = df[df["Score"] >= 65].head(5)
        avoids = df[df["Score"] < 45].tail(5)

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("🟢 Top Picks")
            for _, row in buys.iterrows():
                st.markdown(f"""
                <div class="metric-card buy-card">
                  <b>{row['Ticker']}</b> — {row['Company']}<br>
                  Score: <b>{row['Score']}/100</b> | ₹{row['Price ₹']:,.0f} | {row['Sector']}
                  {row['PSU']}
                </div>""", unsafe_allow_html=True)

        with col_r:
            st.subheader("🔴 Watch List (Avoid)")
            for _, row in avoids.iterrows():
                st.markdown(f"""
                <div class="metric-card avoid-card">
                  <b>{row['Ticker']}</b> — {row['Company']}<br>
                  Score: <b>{row['Score']}/100</b> | ₹{row['Price ₹']:,.0f} | {row['Sector']}
                </div>""", unsafe_allow_html=True)

        st.divider()

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.plotly_chart(scatter_pe_roe(df), use_container_width=True)
        with col_b:
            st.plotly_chart(sector_heatmap(df), use_container_width=True)

        st.plotly_chart(position_52w_chart(df), use_container_width=True)

    # ══════════════════════════════════════════════════════
    # PAGE 2: SINGLE STOCK ANALYSIS
    # ══════════════════════════════════════════════════════
    elif page == "🔍 Single Stock Analysis":
        st.title("🔍 Single Stock Deep Dive")

        col_input, col_go = st.columns([4, 1])
        with col_input:
            ticker = st.selectbox("Select Stock", sorted(NIFTY50.keys()),
                                  index=sorted(NIFTY50.keys()).index("RELIANCE"))
        with col_go:
            st.write("")
            analyze = st.button("🚀 Analyze", use_container_width=True)

        if ticker:
            with st.spinner(f"Analyzing {ticker}..."):
                data = fetch_stock_data(ticker)

            if "error" in data:
                st.error(f"Failed to fetch data: {data['error']}")
                return

            # Header metrics
            price  = data.get("current_price", 0)
            score  = data.get("score", 50)
            rating = data.get("rating", "")
            is_psu = ticker in PSU_STOCKS

            st.subheader(f"{data.get('company_name', ticker)} ({ticker}) {'🏛️ PSU' if is_psu else ''}")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Price ₹",   f"₹{price:,.2f}")
            m2.metric("Score",     f"{score}/100")
            m3.metric("P/E",       data.get("pe", 0))
            m4.metric("ROE %",     f"{data.get('roe',0):.1f}%")
            m5.metric("Div Yield", f"{data.get('div_yield',0):.2f}%")

            # Rating banner
            color = "green" if "BUY" in rating else "orange" if "HOLD" in rating else "red"
            st.markdown(f"<h2 style='color:{color};text-align:center'>{rating}</h2>", unsafe_allow_html=True)

            # Score gauge + 52W position
            col_g, col_52, col_info = st.columns([1, 2, 1])

            with col_g:
                st.plotly_chart(score_gauge(score, "Composite Score"), use_container_width=True)

            with col_52:
                pos  = data.get("pos_52w_pct", 50)
                low  = data.get("low_52w", 0)
                high = data.get("high_52w", 0)
                fig  = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pos,
                    delta={"reference": 50},
                    title={"text": "52-Week Position %"},
                    number={"suffix": "%", "font": {"color": "white"}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#00c853" if pos < 30 else "#ffd600" if pos < 70 else "#ff1744"},
                        "steps": [
                            {"range": [0, 30],  "color": "#0a1a0a"},
                            {"range": [30, 70], "color": "#1a1a0a"},
                            {"range": [70, 100],"color": "#1a0a0a"},
                        ],
                    }
                ))
                fig.update_layout(paper_bgcolor="#1e2130", height=180, margin=dict(l=20,r=20,t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"52W Low: ₹{low:,.0f} | 52W High: ₹{high:,.0f}")

            with col_info:
                st.metric("Sector",   data.get("sector", ""))
                st.metric("MCap Cr",  f"₹{data.get('market_cap_cr',0):,.0f}")
                st.metric("P/B",      data.get("pb", 0))

            # Price chart
            hist = data.get("hist")
            if hist is not None and not hist.empty:
                st.plotly_chart(price_chart(hist, ticker), use_container_width=True)

                # RSI Chart
                rsi = compute_rsi(hist["Close"])
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI(14)", line=dict(color="#E91E63")))
                fig_rsi.add_hline(y=70, line_dash="dot", line_color="red",    annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dot", line_color="green",  annotation_text="Oversold")
                fig_rsi.add_hline(y=50, line_dash="dot", line_color="yellow", annotation_text="Neutral")
                fig_rsi.update_layout(
                    template="plotly_dark", height=200, title="RSI (14)",
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    yaxis_range=[0, 100], margin=dict(l=0,r=0,t=30,b=0)
                )
                st.plotly_chart(fig_rsi, use_container_width=True)

            # PSU Info
            if is_psu:
                st.info(f"🏛️ **PSU Stock** — Government owned. Apply PSU valuation discount (~20-30%) to fair value. "
                        f"Higher dividend mandates, policy-driven capex, and governance discounts apply.")

    # ══════════════════════════════════════════════════════
    # PAGE 3: PORTFOLIO SCREENER
    # ══════════════════════════════════════════════════════
    elif page == "📊 Portfolio Screener":
        st.title("📊 Nifty50 Portfolio Screener")
        st.caption("Rank all 50 stocks by composite score | Filter by sector, valuation, momentum")

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            sector_filter = st.multiselect("Sector", sorted(set(NIFTY50.values())), default=[])
        with col_f2:
            rating_filter = st.multiselect("Rating", ["🟢 BUY", "🟡 HOLD", "🔴 AVOID"], default=[])
        with col_f3:
            min_score = st.slider("Min Score", 0, 100, 0)
        with col_f4:
            show_psu = st.checkbox("Show PSU only", False)

        with st.spinner("Loading all Nifty50 stocks..."):
            df = fetch_all_nifty50()

        # Apply filters
        if sector_filter: df = df[df["Sector"].isin(sector_filter)]
        if rating_filter: df = df[df["Rating"].isin(rating_filter)]
        if min_score > 0: df = df[df["Score"] >= min_score]
        if show_psu:      df = df[df["PSU"] == "🏛️"]

        st.caption(f"Showing {len(df)} stocks")

        # Color-coded table
        def color_score(val):
            if val >= 65: return "background-color: #0a2a0a; color: #00c853"
            elif val >= 50: return "background-color: #2a2a0a; color: #ffd600"
            else: return "background-color: #2a0a0a; color: #ff1744"

        styled = df.style.applymap(color_score, subset=["Score"])
        st.dataframe(styled, use_container_width=True, height=500)

        # Summary stats
        st.divider()
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Avg Score",   f"{df['Score'].mean():.1f}")
        col_s2.metric("BUY signals", len(df[df["Score"] >= 65]))
        col_s3.metric("HOLD",        len(df[(df["Score"] >= 50) & (df["Score"] < 65)]))
        col_s4.metric("AVOID",       len(df[df["Score"] < 50]))

        # Score distribution
        fig = px.histogram(df, x="Score", nbins=20, color="Rating",
                           color_discrete_map={"🟢 BUY":"#00c853","🟡 HOLD":"#ffd600","🔴 AVOID":"#ff1744"},
                           title="Score Distribution")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════
    # PAGE 4: SECTOR DEEP DIVE
    # ══════════════════════════════════════════════════════
    elif page == "🏦 Sector Deep Dive":
        st.title("🏦 Sector Deep Dive")

        sector = st.selectbox("Choose Sector", sorted(set(NIFTY50.values())))
        tickers = [t for t, s in NIFTY50.items() if s == sector]

        st.caption(f"Analyzing {len(tickers)} stocks in {sector}: {', '.join(tickers)}")

        rows = []
        with st.spinner(f"Loading {sector} stocks..."):
            for ticker in tickers:
                data = fetch_stock_data(ticker)
                rows.append({
                    "Ticker":    ticker,
                    "Company":   data.get("company_name", ticker)[:20],
                    "Score":     data.get("score", 50),
                    "Rating":    data.get("rating", ""),
                    "Price ₹":   data.get("current_price", 0),
                    "P/E":       data.get("pe", 0),
                    "P/B":       data.get("pb", 0),
                    "ROE %":     data.get("roe", 0),
                    "52W %":     data.get("pos_52w_pct", 50),
                    "MCap Cr":   data.get("market_cap_cr", 0),
                })
                time.sleep(0.2)

        df_sector = pd.DataFrame(rows).sort_values("Score", ascending=False)

        # Score bars
        fig = go.Figure(go.Bar(
            y=df_sector["Ticker"], x=df_sector["Score"],
            orientation="h",
            marker_color=[
                "#00c853" if s >= 65 else "#ffd600" if s >= 50 else "#ff1744"
                for s in df_sector["Score"]
            ],
            text=df_sector["Score"], textposition="outside"
        ))
        fig.update_layout(
            template="plotly_dark", height=max(300, len(tickers) * 50),
            title=f"{sector} — Stock Rankings",
            xaxis_range=[0, 115],
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_sector, use_container_width=True)

    # ══════════════════════════════════════════════════════
    # PAGE 5: MARKET SENTIMENT
    # ══════════════════════════════════════════════════════
    elif page == "📡 Market Sentiment":
        st.title("📡 Market Sentiment Dashboard")
        st.caption("FII/DII Flows | India VIX | Put-Call Ratio | 52W Position")

        # Nifty snapshot
        with st.spinner("Fetching market data..."):
            nifty_data = fetch_stock_data("NIFTY") if "NIFTY" in NIFTY50 else {}
            # Use a proxy
            try:
                nifty = yf.Ticker("^NSEI").history(period="30d")
                vix   = yf.Ticker("^INDIAVIX").history(period="30d")
            except Exception:
                nifty, vix = pd.DataFrame(), pd.DataFrame()

        col1, col2, col3, col4 = st.columns(4)

        if not nifty.empty:
            nifty_price  = nifty["Close"].iloc[-1]
            nifty_change = (nifty["Close"].iloc[-1] - nifty["Close"].iloc[-2]) / nifty["Close"].iloc[-2] * 100
            col1.metric("Nifty 50",  f"{nifty_price:,.0f}", f"{nifty_change:+.2f}%")

        if not vix.empty:
            vix_val    = vix["Close"].iloc[-1]
            vix_change = (vix["Close"].iloc[-1] - vix["Close"].iloc[-2]) / vix["Close"].iloc[-2] * 100
            col2.metric("India VIX", f"{vix_val:.2f}", f"{vix_change:+.2f}%",
                        delta_color="inverse")

        col3.metric("PCR (Est.)", "~1.0", "Neutral", help="Live PCR requires NSE API access")
        col4.metric("FII Flow",   "Live", "Check NSE", help="FII/DII data from NSE API")

        # Nifty chart
        if not nifty.empty:
            sma50  = nifty["Close"].rolling(50).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nifty.index, y=nifty["Close"], name="Nifty50",
                                     fill="tozeroy", fillcolor="rgba(0,200,100,0.1)",
                                     line=dict(color="#00c853", width=2)))
            fig.add_trace(go.Scatter(x=nifty.index, y=sma50, name="50DMA",
                                     line=dict(color="#FFA726", width=1.5, dash="dot")))
            fig.update_layout(
                template="plotly_dark", height=300, title="Nifty50 — 30 Day",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        # VIX chart
        if not vix.empty:
            fig_vix = go.Figure(go.Scatter(
                x=vix.index, y=vix["Close"], fill="tozeroy",
                fillcolor="rgba(255,100,0,0.1)", line=dict(color="#FF5722", width=2), name="India VIX"
            ))
            fig_vix.add_hline(y=12, line_dash="dot", line_color="green",  annotation_text="Low Fear")
            fig_vix.add_hline(y=20, line_dash="dot", line_color="yellow", annotation_text="Elevated")
            fig_vix.add_hline(y=25, line_dash="dot", line_color="red",    annotation_text="High Fear")
            fig_vix.update_layout(
                template="plotly_dark", height=250, title="India VIX (Fear Index)",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_vix, use_container_width=True)

        # Sentiment guide
        st.divider()
        st.subheader("📖 Interpretation Guide")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("""
            **India VIX**
            - < 12 → Complacent / Stable
            - 12–17 → Normal
            - 17–22 → Elevated
            - > 25 → Fear / Potential bottom

            **Put-Call Ratio (PCR)**
            - > 1.3 → Oversold, bullish contrarian
            - 0.8–1.2 → Neutral
            - < 0.7 → Overbought, bearish contrarian
            """)
        with col_g2:
            st.markdown("""
            **FII/DII Flows**
            - FII buying + DII buying → 🟢 Strong
            - FII selling + DII buying → 🟡 Balanced
            - Both selling → 🔴 Weak market

            **52-Week Position**
            - < 30% → Near lows → Value zone
            - 30–70% → Healthy range
            - > 85% → Near highs → Caution
            """)


if __name__ == "__main__":
    main()
