import yfinance as yf
import feedparser
import pandas as pd

# --- CONFIGURATION ---
# For Indian stocks, always add '.NS' for NSE or '.BO' for BSE
STOCK_TICKER = "TATAMOTORS.NS" 

def fetch_stock_price(ticker):
    """
    Fetches the latest closing price and simple trend.
    """
    print(f"\n📊 FETCHING PRICE DATA FOR {ticker}...")
    stock = yf.Ticker(ticker)
    
    # Get 1 month of history
    history = stock.history(period="1mo")
    
    if history.empty:
        print("❌ Error: No data found. Check ticker symbol.")
        return None
    
    current_price = history['Close'].iloc[-1]
    prev_price = history['Close'].iloc[-2]
    
    # Calculate simple daily movement
    change = current_price - prev_price
    pct_change = (change / prev_price) * 100
    
    print(f"   Current Price: ₹{current_price:.2f}")
    print(f"   Daily Change: {change:.2f} ({pct_change:.2f}%)")
    return current_price

def fetch_fundamentals(ticker):
    """
    Fetches key ratios like PE, Debt, and ROE.
    """
    print(f"\n⚖️  FETCHING FUNDAMENTALS FOR {ticker}...")
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # We use .get() to avoid crashing if data is missing
    pe_ratio = info.get('trailingPE', 'N/A')
    roe = info.get('returnOnEquity', 'N/A')
    debt_to_equity = info.get('debtToEquity', 'N/A')
    
    print(f"   P/E Ratio: {pe_ratio}")
    print(f"   ROE: {roe}")
    print(f"   Debt/Equity: {debt_to_equity}")

def fetch_latest_news(ticker):
    """
    Fetches top 3 news headlines from Google News India.
    """
    print(f"\n📰 FETCHING NEWS FOR {ticker}...")
    
    # Clean ticker name for search (remove .NS)
    search_term = ticker.replace('.NS', '').replace('.BO', '')
    
    # Google News RSS URL for India
    rss_url = f"https://news.google.com/rss/search?q={search_term}+stock+news&hl=en-IN&gl=IN&ceid=IN:en"
    
    feed = feedparser.parse(rss_url)
    
    if not feed.entries:
        print("   No news found.")
        return

    # Loop through the first 3 news items
    for i, entry in enumerate(feed.entries[:3]):
        print(f"   {i+1}. {entry.title}")
        print(f"      (Link: {entry.link})")

# --- MASTER EXECUTION ---
if __name__ == "__main__":
    print(f"--- STARTING PHASE 1: DATA FOUNDATION ---")
    
    # 1. Get Price
    fetch_stock_price(STOCK_TICKER)
    
    # 2. Get Fundamentals
    fetch_fundamentals(STOCK_TICKER)
    
    # 3. Get News
    fetch_latest_news(STOCK_TICKER)
    
    print("\n--- PHASE 1 COMPLETE ---")