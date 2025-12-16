import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
import os

# --- 1. SETUP: CHOOSE YOUR MODE ---
# Set this to True if you have an OpenAI Key. Set to False to use Free Simulation.
USE_REAL_AI = False 
OPENAI_API_KEY = "sk-..."  # Paste your key inside these quotes if USE_REAL_AI is True

# --- 2. THE ANALYST (Your Code from Phase 2) ---
def get_stock_data(ticker):
    print(f"   ... Analyzing {ticker} ...")
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    info = stock.info
    
    if df.empty: return None

    # Calculate Indicators
    rsi = RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
    ema = EMAIndicator(close=df["Close"], window=200).ema_indicator().iloc[-1]
    price = df["Close"].iloc[-1]
    
    # Get Fundamentals
    pe = info.get('trailingPE', 0)
    roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
    
    # Simple Logic (The "Reasoning")
    trend = "BULLISH" if price > ema else "BEARISH"
    val_status = "EXPENSIVE" if pe > 40 else "REASONABLE"
    
    return {
        "ticker": ticker,
        "price": round(price, 2),
        "trend": trend,
        "rsi": round(rsi, 2),
        "pe": pe,
        "roe": round(roe, 2),
        "valuation": val_status
    }

# --- 3. THE WRITER (The New Phase 3 Part) ---
def generate_report(data):
    print("\n✍️  WRITING MEMO...")
    
    # The Prompt: We format the data into a text block
    prompt = f"""
    ACT AS: A Senior Investment Consultant.
    TASK: Write a short, punchy investment memo for {data['ticker']}.
    
    DATA:
    - Price: ₹{data['price']}
    - Technical Trend: {data['trend']} (RSI: {data['rsi']})
    - Valuation: {data['pe']} P/E ({data['valuation']})
    - Profitability (ROE): {data['roe']}%
    
    INSTRUCTIONS:
    1. Start with a "VERDICT" (Buy/Watch/Avoid).
    2. Give 2 bullet points explaining why.
    3. Use professional tone.
    """

    if USE_REAL_AI:
        # This part connects to the REAL ChatGPT
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage
        
        chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o")
        response = chat.invoke([HumanMessage(content=prompt)])
        return response.content
        
    else:
        # --- SIMULATION MODE (Free) ---
        # This pretends to be AI by using simple logic templates
        verdict = "BUY" if data['trend'] == "BULLISH" and data['valuation'] == "REASONABLE" else "WATCH"
        
        simulated_report = f"""
        *** AI SIMULATION REPORT ***
        
        VERDICT: {verdict}
        
        REASONING:
        1. The trend is currently {data['trend']} with an RSI of {data['rsi']}.
        2. The valuation is {data['valuation']} (P/E: {data['pe']}), and ROE is {data['roe']}%.
        
        (Note: Enable USE_REAL_AI in code to get a full ChatGPT analysis)
        """
        return simulated_report

# --- 4. MASTER EXECUTION ---
if __name__ == "__main__":
    ticker = "TITAN.NS" # Change this to any Indian stock
    
    print(f"--- STARTING AI AGENT FOR {ticker} ---")
    
    # Step A: Get Data
    stock_data = get_stock_data(ticker)
    
    if stock_data:
        # Step B: Write Report
        final_memo = generate_report(stock_data)
        print("-" * 40)
        print(final_memo)
        print("-" * 40)
    else:
        print("Error: Could not fetch data.")