# ============================================================
# debug_currency.py — Check exactly why USD detection fails
# Run: python debug_currency.py
# ============================================================

import yfinance as yf

stock = yf.Ticker("INFY.NS")
info  = stock.info

current_price   = info.get("currentPrice") or info.get("regularMarketPrice", 0)
market_cap_info = info.get("marketCap", 0) or 0
shares_info     = info.get("sharesOutstanding", 0) or 0

print(f"\ncurrent_price      : {current_price}")
print(f"marketCap (info)   : {market_cap_info:,.0f}")
print(f"sharesOutstanding  : {shares_info:,.0f}")

if shares_info > 0 and current_price > 0:
    implied_mcap_inr = shares_info * current_price
    ratio = implied_mcap_inr / market_cap_info
    print(f"\nimplied_mcap_inr   : {implied_mcap_inr:,.0f}")
    print(f"ratio              : {ratio:.2f}")
    print(f"USD detected?      : {ratio > 50}")
else:
    print("\nCannot compute ratio — shares or price is zero")

# Also check what cfo_cr and capex_cr look like
cf = stock.cashflow
print("\n=== RAW CASH FLOW KEY ROWS ===")
for key in ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"]:
    if key in cf.index:
        print(f"  {key}: {cf.loc[key].iloc[0]:,.0f}")
