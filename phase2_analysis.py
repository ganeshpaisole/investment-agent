import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# --- CONFIGURATION ---
STOCK_TICKER = "TITAN.NS"

def analyze_technicals(ticker):
    print(f"\n📈 RUNNING TECHNICAL ANALYSIS FOR {ticker}...")
    stock = yf.Ticker(ticker)
    
    # Get 1 year of data
    df = stock.history(period="1y")
    
    if df.empty:
        print("❌ No data found.")
        return

    # --- NEW CALCULATION METHOD (Using 'ta' library) ---
    
    # 1. Calculate RSI (Relative Strength Index)
    # We create an "Indicator" and then ask for the "rsi()" values
    rsi_indicator = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()
    
    # 2. Calculate EMA (Exponential Moving Average)
    ema_indicator = EMAIndicator(close=df["Close"], window=200)
    df["EMA_200"] = ema_indicator.ema_indicator()
    
    # 3. The "Judge" Logic
    # We look at the very last row (iloc[-1]) to get today's data
    current_price = df['Close'].iloc[-1]
    current_rsi = df['RSI'].iloc[-1]
    ema_200 = df['EMA_200'].iloc[-1]
    
    print(f"   Price: ₹{current_price:.2f}")
    print(f"   RSI Level: {current_rsi:.2f}")
    
    # DECISION 1: TREND
    if current_price > ema_200:
        print("   ✅ TREND VERDICT: BULLISH (Price > 200 EMA)")
    else:
        print("   ⚠️ TREND VERDICT: BEARISH (Price < 200 EMA)")

    # DECISION 2: MOMENTUM
    if current_rsi > 70:
        print("   ⚠️ MOMENTUM: OVERBOUGHT (High risk of pullback)")
    elif current_rsi < 30:
        print("   ✅ MOMENTUM: OVERSOLD (Potential buying opportunity)")
    else:
        print("   ℹ️ MOMENTUM: NEUTRAL")

# --- MASTER EXECUTION ---
if __name__ == "__main__":
    analyze_technicals(STOCK_TICKER)
