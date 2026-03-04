"""
patch_v2.py — Final fix for INFY USD detection
Detection method: If totalRevenue < 1 Trillion (1e12), it's in USD
INFY revenue: $19.8B = 19,847,000,064 (< 1e12) → USD
TCS revenue:  ₹2,40,893 Cr = 2,408,930,000,000 (> 1e12) → INR

Run: python patch_v2.py
"""

path = 'agents/valuation_agent.py'
content = open(path, 'r', encoding='utf-8').read()

# The old detection block (what's currently in the file)
old = '''        shares_out   = info.get("sharesOutstanding", 1) or 1
        market_cap   = info.get("marketCap", 1) or 1
        implied_mcap = shares_out * current_price
        fx           = 84.0 if market_cap > 0 and (implied_mcap / market_cap) < 0.05 else 1.0
        if fx == 84.0:
            logger.warning(f"USD financials detected for {self.ticker} — converting at Rs.84")
        fcf_from_info = (info.get("freeCashflow", 0) or 0) * fx
        cfo_from_info = (info.get("operatingCashflow", 0) or 0) * fx'''

# The new detection block (revenue-size based)
new = '''        # Detect USD financials: INR revenue for large NSE stocks > 1 Trillion
        # INFY revenue = $19.8B (< 1T) -> USD; TCS = Rs.2.4L Cr (> 1T) -> INR
        _revenue_raw = info.get("totalRevenue", 0) or 0
        fx = 84.0 if 0 < _revenue_raw < 1_000_000_000_000 else 1.0
        if fx == 84.0:
            logger.warning(
                f"USD financials detected for {self.ticker} "
                f"(revenue ${_revenue_raw/1e9:.1f}B < 1T threshold). "
                f"Converting at Rs.84"
            )
        fcf_from_info = (info.get("freeCashflow", 0) or 0) * fx
        cfo_from_info = (info.get("operatingCashflow", 0) or 0) * fx'''

# Try replacing
if old in content:
    content = content.replace(old, new)
    open(path, 'w', encoding='utf-8').write(content)
    print("✅ PATCH APPLIED SUCCESSFULLY")
    print("\nExpected INFY output:")
    print("  USD financials detected for INFY (revenue $19.8B < 1T threshold)")
    print("  Base FCF: ~Rs.26,358 Cr")
    print("  DCF IV:   ~Rs.1,600-1,800/share")
    print("\nNow run: python -B main.py --ticker INFY --no-gpt --mode valuation")
else:
    print("❌ Old block not found. Searching for partial matches...")
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'implied_mcap' in line or 'shares_out' in line or 'fx ' in line:
            print(f"  Line {i:3d}: {repr(line)}")
    print("\nPaste this output so we can match exactly.")
