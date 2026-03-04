"""
patch_fix.py — Run this once to fix USD currency detection in valuation_agent.py
Usage: python patch_fix.py
"""

content = open('agents/valuation_agent.py', 'r', encoding='utf-8').read()

# Find the exact lines
for i, l in enumerate(content.split('\n')):
    if 'freeCashflow' in l or 'operatingCashflow' in l:
        print(f'Line {i}: {repr(l)}')

# Find line numbers to build exact old string
lines = content.split('\n')
fcf_line = next((i for i, l in enumerate(lines) if 'freeCashflow' in l and 'info.get' in l), None)
cfo_line = next((i for i, l in enumerate(lines) if 'operatingCashflow' in l and 'info.get' in l), None)

if fcf_line is None or cfo_line is None:
    print("ERROR: Could not find target lines")
    exit(1)

print(f"\nFound freeCashflow at line {fcf_line}")
print(f"Found operatingCashflow at line {cfo_line}")

# Build old string from actual file content
old = lines[fcf_line] + '\n' + lines[cfo_line]
print(f"\nOLD block:\n{repr(old)}")

# Get the indentation from the existing line
indent = len(lines[fcf_line]) - len(lines[fcf_line].lstrip())
pad = ' ' * indent

new = (
    pad + 'shares_out   = info.get("sharesOutstanding", 1) or 1\n' +
    pad + 'market_cap   = info.get("marketCap", 1) or 1\n' +
    pad + 'implied_mcap = shares_out * current_price\n' +
    pad + 'fx           = 84.0 if market_cap > 0 and (implied_mcap / market_cap) < 0.05 else 1.0\n' +
    pad + 'if fx == 84.0:\n' +
    pad + '    logger.warning(f"USD financials detected for {self.ticker} — converting at Rs.84")\n' +
    pad + 'fcf_from_info = (info.get("freeCashflow", 0) or 0) * fx\n' +
    pad + 'cfo_from_info = (info.get("operatingCashflow", 0) or 0) * fx'
)

if old in content:
    content = content.replace(old, new)
    open('agents/valuation_agent.py', 'w', encoding='utf-8').write(content)
    print("\n✅ PATCH APPLIED SUCCESSFULLY")
    print("Now run: python -B main.py --ticker INFY --no-gpt --mode valuation")
else:
    print("\n❌ PATCH FAILED — old string not found in file")
    print("Try running: python patch_fix.py 2>&1 | more")
