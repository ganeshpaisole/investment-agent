# 🚀 Automatic Live Data Fetching - No Manual Entry!

## What You Got

✅ **live_data_fetcher.py** - Simple fetcher (works immediately, no setup)  
✅ **auto_fill.py** - Advanced fetcher (needs free API key, gets ALL data)

---

## 🎯 OPTION 1: Simple Fetcher (Start Here)

### **What It Does:**
- ✅ Fetches live prices from Yahoo Finance
- ✅ Updates all your config files automatically
- ✅ No API key needed
- ⚠️ Only gets price (you still need to manually add revenue, margins, etc. once)

### **How to Use:**

```powershell
# Update ALL stocks at once
py live_data_fetcher.py

# Update just one stock
py live_data_fetcher.py RELIANCE

# Update multiple stocks
py live_data_fetcher.py TCS INFY WIPRO
```

### **Example Output:**
```
LIVE DATA UPDATER - NSE Stock Prices
────────────────────────────────────────

Fetching RELIANCE... ✅ ₹2,850.50 (Yahoo Finance)
Fetching TCS... ✅ ₹3,845.20 (Yahoo Finance)
Fetching INFY... ✅ ₹1,452.75 (Yahoo Finance)

✅ Updated: 48 stocks
❌ Failed: 2 stocks
```

### **Schedule It to Run Daily:**

**Windows Task Scheduler:**
- Program: `C:\Python39\python.exe`
- Arguments: `live_data_fetcher.py`
- Trigger: Daily at 9:15 AM (after market opens)

**Or run continuously:**
```powershell
# Update every 1 hour
while($true) { py live_data_fetcher.py; Start-Sleep -Seconds 3600 }
```

---

## 🔥 OPTION 2: Advanced Auto-Fill (Complete Data)

### **What It Does:**
- ✅ Fetches EVERYTHING automatically:
  - Current Price
  - Market Cap
  - Annual Revenue
  - EBITDA Margin
  - Debt
  - Cash
  - P/E Ratio
- ✅ No manual entry needed at all!
- ⚠️ Requires free API key (one-time setup)

### **Setup (5 Minutes):**

#### **Step 1: Get Free API Key**

Go to: https://financialmodelingprep.com/developer/docs/

1. Click "Get Your Free API Key"
2. Sign up (free)
3. Copy your API key (looks like: `abc123xyz...`)

**Free Tier Limits:**
- 250 requests per day
- Enough for 50 NIFTY stocks 5 times/day!

#### **Step 2: Setup**

```powershell
py auto_fill.py --setup

# Paste your API key when prompted
# Enter FMP API key: abc123xyz...
```

#### **Step 3: Auto-Fill Stocks**

```powershell
# Fill one stock completely
py auto_fill.py RELIANCE

# Fill multiple stocks
py auto_fill.py TCS INFY HDFCBANK WIPRO

# Fill all NIFTY 50 (takes ~5 minutes)
py auto_fill_all_nifty50.py
```

### **Example Output:**
```
Auto-filling RELIANCE...
────────────────────────────────────────

Trying Financial Modeling Prep...
✅ Complete data fetched!
   Price: ₹2,850.50
   Revenue: ₹9,20,000 Cr
   EBITDA Margin: 18.2%
   Debt: ₹2,50,000 Cr
   Cash: ₹1,80,000 Cr
```

**No manual data entry needed!** 🎉

---

## 📊 Comparison

| Feature | Simple Fetcher | Auto-Fill |
|---------|----------------|-----------|
| **Setup** | None needed ✅ | 5-min API setup |
| **Price** | ✅ Auto | ✅ Auto |
| **Market Cap** | ✅ Auto | ✅ Auto |
| **Revenue** | ❌ Manual | ✅ Auto |
| **Margins** | ❌ Manual | ✅ Auto |
| **Debt/Cash** | ❌ Manual | ✅ Auto |
| **Speed** | Instant | ~6 sec/stock |
| **API Limits** | Unlimited | 250/day (free) |

---

## 🎯 Recommended Workflow

### **One-Time Setup (Auto-Fill):**

```powershell
# 1. Setup API key (one time)
py auto_fill.py --setup

# 2. Auto-fill all NIFTY 50 stocks (takes 5-10 minutes)
py auto_fill_all_nifty50.py
```

**Now all 50 configs are completely filled!**

### **Daily Updates (Simple Fetcher):**

```powershell
# Just update prices daily (fast)
py live_data_fetcher.py
```

**Why:** Fundamentals don't change daily. Prices do.

---

## 🤖 Integrate with Agent

### **Auto-update before analysis:**

Edit `stock_agent_simple.py`, add this at the top of `run_daily_analysis()`:

```python
def run_daily_analysis(self):
    # Auto-update prices first
    os.system("py live_data_fetcher.py")
    
    # Then analyze with updated data
    print(f"\n🤖 Stock Agent Running...")
    ...
```

Now agent always uses latest prices!

---

## 🔧 Create Auto-Fill Script for All NIFTY 50

Save as `auto_fill_all_nifty50.py`:

```python
#!/usr/bin/env python3
"""Auto-fill all NIFTY 50 stocks"""

import os
import time

NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "HCLTECH", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO",
    "NESTLEIND", "ONGC", "TATAMOTORS", "NTPC", "ADANIENT",
    "TATASTEEL", "M&M", "POWERGRID", "BAJAJFINSV", "TECHM",
    "HINDALCO", "INDUSINDBK", "ADANIPORTS", "COALINDIA", "DRREDDY",
    "JSWSTEEL", "EICHERMOT", "TATACONSUM", "GRASIM", "CIPLA",
    "BRITANNIA", "BPCL", "HEROMOTOCO", "DIVISLAB", "APOLLOHOSP",
    "SHREECEM", "SBILIFE", "HDFCLIFE", "LTIM", "BAJAJ-AUTO"
]

print("Auto-filling all NIFTY 50 stocks...")
print("This will take ~5-10 minutes\n")

for i, ticker in enumerate(NIFTY_50, 1):
    print(f"[{i}/50] ", end="")
    os.system(f"py auto_fill.py {ticker}")
    time.sleep(3)  # Rate limiting

print("\n✅ All NIFTY 50 stocks filled!\n")
```

Run it:
```powershell
py auto_fill_all_nifty50.py
```

---

## 💡 Pro Tips

### **Tip 1: Schedule Daily Price Updates**

Create `update_prices_daily.bat`:

```batch
@echo off
cd C:\Users\drdee\AI_Agent\Desktop\StockAnalysis
C:\Python39\python.exe live_data_fetcher.py
pause
```

**Schedule in Task Scheduler at 9:15 AM daily**

### **Tip 2: Weekly Full Update**

```batch
@echo off
cd C:\Users\drdee\AI_Agent\Desktop\StockAnalysis
C:\Python39\python.exe auto_fill_all_nifty50.py
pause
```

**Schedule for Sunday nights (updates fundamentals too)**

### **Tip 3: Check Data Quality**

After auto-fill, verify a few stocks manually:

```powershell
# Check if data looks reasonable
py nse_generic_forecaster.py nifty50_configs/stock_config_RELIANCE.json
```

If price or revenue looks wrong, fix manually.

### **Tip 4: Fallback to Manual**

If API fails, you always have Screener.in as backup:
1. Go to Screener.in
2. Search stock
3. Copy values
4. Paste into JSON

---

## 🆘 Troubleshooting

### **"Failed to fetch data"**

**Solution 1:** Yahoo Finance might be rate-limiting you
- Add longer delays: `time.sleep(5)`
- Or use API version

**Solution 2:** Stock ticker might be different
- Some stocks use different symbols
- Check Yahoo Finance website for correct symbol

### **"API limit exceeded"**

**Solution:** You hit 250 requests/day limit
- Wait 24 hours, or
- Get paid plan ($14/month = unlimited), or
- Use Simple Fetcher for daily updates

### **"Config not found"**

**Solution:** Make sure you generated configs first
```powershell
py nifty50_generator.py
```

---

## 🎯 Quick Commands Reference

```powershell
# SIMPLE FETCHER (No setup needed)
py live_data_fetcher.py                    # Update all
py live_data_fetcher.py RELIANCE           # Update one
py live_data_fetcher.py TCS INFY WIPRO     # Update multiple

# AUTO-FILL (Needs API key)
py auto_fill.py --setup                    # One-time setup
py auto_fill.py RELIANCE                   # Fill one completely
py auto_fill_all_nifty50.py                # Fill all 50

# VERIFY
py nse_generic_forecaster.py nifty50_configs/stock_config_RELIANCE.json
```

---

## 🎊 You're All Set!

You now have:
1. ✅ Simple price updater (works now)
2. ✅ Advanced auto-filler (complete data)
3. ✅ No more manual data entry!

**Your workflow:**
- **One-time:** `py auto_fill_all_nifty50.py` (fills everything)
- **Daily:** `py live_data_fetcher.py` (updates prices)
- **Weekly:** `py auto_fill_all_nifty50.py` (updates fundamentals)

**Completely automated!** 🚀
