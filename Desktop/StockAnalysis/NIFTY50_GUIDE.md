# 🎯 NIFTY 50 Quick Start Guide

## What You Got

✅ **50 pre-configured template files** (one for each NIFTY 50 stock)  
✅ **Batch analyzer** (analyze multiple stocks at once)  
✅ **Sector analyzer** (compare stocks within sectors)  
✅ **Agent setup** (automated monitoring for all 50)

---

## 🚀 Quick Start (3 Steps)

### **Step 1: Run the Generator** (Already done! ✅)

```powershell
py nifty50_generator.py
```

This created:
- `nifty50_configs/` folder with 50 config files
- `batch_analyzer.py`
- `sector_analyzer.py`

### **Step 2: Fill in Your Favorites (Start with 5)**

**Don't fill all 50!** Start with stocks you're interested in:

#### **Example: Fill RELIANCE**

1. Go to **Screener.in** → Search "RELIANCE"
2. Copy these values:
   - Price: 2,850
   - Market Cap: 19,30,000 Cr
   - Revenue (TTM): 9,20,000 Cr
   - EBITDA Margin: 18%
   - Debt: 2,50,000 Cr
   - Cash: 1,80,000 Cr

3. Open `nifty50_configs/stock_config_RELIANCE.json` in Notepad

4. Replace zeros with real values:
```json
{
    "ticker": "RELIANCE",
    "company_name": "Reliance Industries",
    "current_data": {
        "price": 2850,
        "market_cap_cr": 1930000,
        "annual_revenue_cr": 920000,
        "ebitda_margin": 0.18,
        "debt_cr": 250000,
        "cash_cr": 180000
    }
}
```

5. Save

### **Step 3: Analyze!**

```powershell
# Single stock
py nse_generic_forecaster.py nifty50_configs/stock_config_RELIANCE.json

# All filled stocks at once
py batch_analyzer.py

# Compare all IT stocks
py sector_analyzer.py Technology
```

---

## 📋 Recommended Fill Order

### **Week 1: Top 10 by Index Weight**
1. RELIANCE (16% of NIFTY)
2. HDFCBANK
3. ICICIBANK
4. INFY
5. TCS
6. ITC
7. HINDUNILVR
8. BHARTIARTL
9. LT
10. KOTAKBANK

### **Week 2: Your Current Holdings**
Fill configs for stocks you already own

### **Week 3: Your Watchlist**
Fill configs for stocks you're considering

### **Week 4: Rest (Optional)**
Only fill as needed

---

## 🎯 Usage Examples

### **Example 1: Analyze Top IT Stocks**

```powershell
# Fill these first:
# - stock_config_TCS.json
# - stock_config_INFY.json
# - stock_config_WIPRO.json
# - stock_config_HCLTECH.json
# - stock_config_TECHM.json

# Then compare:
py sector_analyzer.py Technology
```

**Output:**
```
SECTOR ANALYSIS: TECHNOLOGY
─────────────────────────────────────
Rank  Ticker  Company                 Upside    Score
1     INFY    Infosys                 +42.3%    7.8/10
2     TCS     Tata Consultancy        +18.2%    7.5/10
3     HCLTECH HCL Technologies        +35.1%    7.2/10
...

✅ Best in Technology: Infosys
```

### **Example 2: Find Best Banking Stock**

```powershell
# Fill banking stocks:
# HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK, INDUSINDBK

py sector_analyzer.py Financial
```

### **Example 3: Batch Analysis of Your Portfolio**

```powershell
# Fill configs for all your holdings
# Then run:

py batch_analyzer.py
```

**Output:**
```
BATCH STOCK ANALYZER
─────────────────────────────────────
Analyzing 12 stocks...

TOP 5 PICKS (by Quality Score):
1. Infosys (INFY)
   Score: 7.8/10 | Upside: +42.3% | ₹1,450

2. HDFC Bank (HDFCBANK)
   Score: 7.6/10 | Upside: +25.1% | ₹1,600
...
```

---

## 🤖 Set Up Agent for NIFTY 50

### **Option A: Monitor Specific Stocks**

Edit `agent_config.json`:

```json
{
    "watchlist": [
        "nifty50_configs/stock_config_RELIANCE.json",
        "nifty50_configs/stock_config_TCS.json",
        "nifty50_configs/stock_config_INFY.json",
        "nifty50_configs/stock_config_HDFCBANK.json"
    ],
    
    "alert_rules": {
        "RELIANCE": {"buy_below": 2700},
        "TCS": {"buy_below": 3600},
        "INFY": {"buy_below": 1400},
        "HDFCBANK": {"buy_below": 1550}
    }
}
```

### **Option B: Monitor Entire Sector**

```json
{
    "watchlist": [
        "nifty50_configs/stock_config_TCS.json",
        "nifty50_configs/stock_config_INFY.json",
        "nifty50_configs/stock_config_WIPRO.json",
        "nifty50_configs/stock_config_HCLTECH.json",
        "nifty50_configs/stock_config_TECHM.json",
        "nifty50_configs/stock_config_LTIM.json"
    ]
}
```

### **Option C: Monitor All 50 (Advanced)**

Create `agent_config_nifty50.json`:

```json
{
    "watchlist": [
        "nifty50_configs/stock_config_RELIANCE.json",
        "nifty50_configs/stock_config_TCS.json",
        ...all 50 stocks...
    ],
    
    "send_daily_summary": true
}
```

**Run:**
```powershell
py stock_agent_simple.py --config agent_config_nifty50.json
```

---

## 📊 Tools Explained

### **1. nifty50_generator.py**
**What:** Creates template configs for all 50 stocks  
**When:** Run once (already done)  
**Output:** 50 JSON files in `nifty50_configs/` folder

### **2. batch_analyzer.py**
**What:** Analyzes all filled configs at once  
**When:** After filling 5+ configs  
**Output:** Ranked list of best stocks

```powershell
py batch_analyzer.py
```

### **3. sector_analyzer.py**
**What:** Compares stocks within a sector  
**When:** Deciding between similar stocks  
**Output:** Best stock in sector

```powershell
py sector_analyzer.py Technology
py sector_analyzer.py Financial
py sector_analyzer.py Manufacturing
```

### **4. stock_agent_simple.py**
**What:** Monitors stocks and sends alerts  
**When:** Daily automated analysis  
**Output:** Email alerts when opportunities arise

---

## 💡 Pro Tips

### **Tip 1: Use Screener.in Watchlist**

1. Create watchlist on Screener.in
2. Export to Excel
3. Fill configs from Excel data
4. Much faster than manual!

### **Tip 2: Template Customization**

The generator uses sector-based defaults. Adjust if needed:

```json
"growth_assumptions": {
    "base": {
        "revenue_growth_5y": [0.15, 0.13, 0.12, 0.10, 0.09],
        // Increase if you're bullish, decrease if bearish
    }
}
```

### **Tip 3: Partial Fill is OK**

You don't need ALL fields. Minimum required:
- Price ✅
- Market Cap ✅
- Annual Revenue ✅
- EBITDA Margin ✅

Debt and Cash can be 0 if unknown.

### **Tip 4: Automate Updates**

Create a script to update prices weekly:

```python
import pandas as pd
from nsepy import get_history

# Fetch latest prices
# Update config files automatically
```

---

## 🎯 Common Workflows

### **Workflow 1: Portfolio Review**

```powershell
# 1. Fill configs for all your holdings
# 2. Run batch analysis
py batch_analyzer.py

# 3. Check which holdings are underperforming
# 4. Decide to hold/sell
```

### **Workflow 2: New Investment Research**

```powershell
# 1. Fill configs for 5-10 candidates
# 2. Compare within sector
py sector_analyzer.py Technology

# 3. Analyze top 2-3 in detail
py nse_generic_forecaster.py nifty50_configs/stock_config_INFY.json

# 4. Make decision
```

### **Workflow 3: Rebalancing**

```powershell
# 1. Analyze entire portfolio
py batch_analyzer.py

# 2. Sort by score
# 3. Increase allocation to high-scorers
# 4. Reduce allocation to low-scorers
```

---

## 📁 File Structure

```
StockAnalysis/
├── nse_generic_forecaster.py
├── stock_agent_simple.py
├── nifty50_generator.py
├── batch_analyzer.py
├── sector_analyzer.py
│
├── nifty50_configs/
│   ├── stock_config_RELIANCE.json
│   ├── stock_config_TCS.json
│   ├── stock_config_INFY.json
│   └── ... (47 more)
│
└── agent_config.json
```

---

## ⚡ Quick Commands Reference

```powershell
# Generate all 50 configs (one-time)
py nifty50_generator.py

# Analyze single stock
py nse_generic_forecaster.py nifty50_configs/stock_config_RELIANCE.json

# Batch analyze all filled stocks
py batch_analyzer.py

# Compare sector stocks
py sector_analyzer.py Technology

# Monitor with agent
py stock_agent_simple.py

# Continuous monitoring
py stock_agent_simple.py --continuous 24
```

---

## 🎓 Learning Path

**Week 1:**
- Fill top 5 stocks
- Run batch analyzer
- Understand output

**Week 2:**
- Fill 10 more stocks
- Use sector analyzer
- Compare similar stocks

**Week 3:**
- Set up agent
- Add email alerts
- Automate daily analysis

**Month 2:**
- Fill all 50 configs
- Build custom dashboards
- Refine strategy

---

## 🆘 Common Questions

**Q: Do I need to fill all 50?**  
A: No! Start with 5-10 stocks you care about.

**Q: How often should I update data?**  
A: After quarterly results (4 times/year) or when major news.

**Q: Can I add more stocks beyond NIFTY 50?**  
A: Yes! Use the template generator or copy an existing config.

**Q: Which stocks should I fill first?**  
A: Your current holdings + your watchlist.

**Q: How long does it take to fill one config?**  
A: 2-3 minutes per stock if you have Screener.in open.

---

## 🎉 You're Ready!

You now have a complete NIFTY 50 analysis system. Start with 5 stocks today!

**Your action items:**
1. ✅ Pick 5 favorite NIFTY 50 stocks
2. ✅ Go to Screener.in
3. ✅ Fill their configs (10 minutes total)
4. ✅ Run `py batch_analyzer.py`
5. ✅ See which is best!

---

**Happy analyzing!** 📊
