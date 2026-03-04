# ============================================================
# debug_infy.py — Run this ONCE to diagnose the FCF data issue
# Usage: python debug_infy.py
# ============================================================

import yfinance as yf

stock = yf.Ticker("INFY.NS")
cf    = stock.cashflow
info  = stock.info

print("\n" + "="*60)
print("CASH FLOW STATEMENT — ALL ROW NAMES")
print("="*60)
for idx in cf.index:
    try:
        val = cf.loc[idx].iloc[0]
        print(f"  {idx:<45} {val:>20,.0f}")
    except:
        print(f"  {idx:<45} {'N/A':>20}")

print("\n" + "="*60)
print("KEY INFO FIELDS")
print("="*60)
keys = [
    "freeCashflow", "operatingCashflow", "totalRevenue",
    "ebitda", "netIncomeToCommon", "marketCap"
]
for k in keys:
    v = info.get(k, "NOT FOUND")
    if isinstance(v, (int, float)) and v:
        print(f"  {k:<35} {v:>20,.0f}")
    else:
        print(f"  {k:<35} {str(v):>20}")
