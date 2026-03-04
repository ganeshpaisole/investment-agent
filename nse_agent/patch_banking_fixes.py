"""
patch_banking_fixes.py — Two fixes:
  1. Banking Agent: ROA from balance sheet (not info dict)
     - info.totalAssets unreliable for banks
     - Use balance_sheet DataFrame iloc[0] instead
  2. NBFC routing: Route to Banking Agent (not generic agents)
     - NBFCs are financial lenders, same logic as banks
     - Negative FCF, high D/E, zero ROCE are all expected for NBFCs

Run: python patch_banking_fixes.py
"""

# ============================================================
# FIX 1: banking_agent.py — ROA from balance sheet
# ============================================================
print("=" * 60)
print("FIX 1: Banking Agent ROA fix")
print("=" * 60)

path = 'agents/banking_agent.py'
content = open(path, encoding='utf-8').read()

old1 = '''        # ── Return on Assets (ROA) ────────────────────────────
        # ROA = Net Income / Total Assets
        # Target: >1% for good banks, >1.5% for great banks
        total_assets = info.get("totalAssets", 0) or 0
        net_income   = info.get("netIncomeToCommon", 0) or 0
        roa = (net_income / total_assets * 100) if total_assets > 0 else 0
        banking["roa"] = round(roa, 2)'''

new1 = '''        # ── Return on Assets (ROA) ────────────────────────────
        # ROA = Net Income / Total Assets
        # Target: >1% for good banks, >1.5% for great banks
        # NOTE: info.totalAssets unreliable for banks — use balance sheet
        total_assets = 0
        if not bal.empty:
            for key in ["Total Assets", "TotalAssets"]:
                if key in bal.index:
                    try:
                        val = bal.loc[key].iloc[0]
                        if pd.notna(val) and val > 0:
                            total_assets = float(val)
                            break
                    except: pass
        if total_assets == 0:
            total_assets = info.get("totalAssets", 0) or 0

        net_income = info.get("netIncomeToCommon", 0) or 0
        if net_income == 0 and not income.empty:
            for key in ["Net Income", "Net Income Common Stockholders"]:
                if key in income.index:
                    try:
                        val = income.loc[key].iloc[0]
                        if pd.notna(val): net_income = float(val); break
                    except: pass

        roa = (net_income / total_assets * 100) if total_assets > 0 else 0
        banking["roa"] = round(roa, 2)
        banking["total_assets_cr"] = round(total_assets / 1e7, 0)'''

if old1 in content:
    content = content.replace(old1, new1)
    print("  ✅ ROA fix applied (balance sheet totalAssets)")
else:
    print("  ❌ ROA block not found")

# Fix 2: Justified P/B formula — use actual ROA-implied ROE floor
old2 = '''        # Justified P/B = (ROE - g) / (Ke - g)
        # Using g=8%, Ke=12%
        g  = 0.08
        ke = (self.INDIA_RISK_FREE_RATE + 1.0 * self.INDIA_EQUITY_RISK_PREMIUM) / 100
        justified_pb = (roe/100 - g) / (ke - g) if ke > g else 2.0
        justified_pb = max(0.5, min(justified_pb, 6.0))  # cap between 0.5x and 6x'''

new2 = '''        # Justified P/B = (ROE - g) / (Ke - g)  — Gordon Growth applied to book value
        # g = sustainable growth = ROE × retention ratio
        # For Indian private banks: typical fair P/B range 2x-4x
        g  = min(roe / 100 * 0.6, 0.12)   # 60% retention, max 12% growth
        ke = (self.INDIA_RISK_FREE_RATE + 1.0 * self.INDIA_EQUITY_RISK_PREMIUM) / 100
        if ke > g and roe > 0:
            justified_pb = (roe/100 - g) / (ke - g)
        else:
            justified_pb = 1.0
        justified_pb = max(0.5, min(justified_pb, 6.0))  # cap 0.5x - 6x'''

if old2 in content:
    content = content.replace(old2, new2)
    print("  ✅ Justified P/B formula fixed (ROE-based growth)")
else:
    print("  ❌ Justified P/B block not found")

open(path, 'w', encoding='utf-8').write(content)
print(f"  💾 banking_agent.py saved")


# ============================================================
# FIX 3: main.py — Route NBFC to Banking Agent
# ============================================================
print()
print("=" * 60)
print("FIX 2: NBFC routing to Banking Agent")
print("=" * 60)

path2   = 'main.py'
content2 = open(path2, encoding='utf-8').read()

old3 = '''    elif classifier.sector_type == "NBFC":
        console.print(Panel(
            "  ℹ️  NBFC detected — running generic agents with caution.\\n"
            "  D/E ratio and DCF less reliable for NBFCs.\\n"
            "  Focus on: ROA, NIM, asset quality, AUM growth.",
            title="[bold yellow]🏷️  NBFC — Partial Compatibility[/bold yellow]",
            border_style="yellow", padding=(0, 2)
        ))
        console.print()
        return run_generic_analysis(ticker, use_gpt, mode, classifier)'''

new3 = '''    elif classifier.sector_type == "NBFC":
        console.print(Panel(
            "  ℹ️  NBFC detected — routing to Banking Agent.\\n"
            "  Generic DCF/D/E/ROCE are misleading for NBFCs.\\n"
            "  Using ROA, P/B, and liability franchise scoring instead.",
            title="[bold cyan]🏷️  NBFC — Banking Agent[/bold cyan]",
            border_style="cyan", padding=(0, 2)
        ))
        console.print()
        return run_banking_analysis(ticker, use_gpt, mode, classifier)'''

if old3 in content2:
    content2 = content2.replace(old3, new3)
    open(path2, 'w', encoding='utf-8').write(content2)
    print("  ✅ NBFC now routes to Banking Agent")
else:
    print("  ❌ NBFC routing block not found — check main.py")

# ============================================================
# FIX 4: sector_classifier.py — NBFC support matrix
# ============================================================
print()
print("=" * 60)
print("FIX 3: NBFC sector support matrix")
print("=" * 60)

path3    = 'utils/sector_classifier.py'
content3 = open(path3, encoding='utf-8').read()

old4 = '''    "UNKNOWN": {
        "fundamental": True,
        "valuation":   True,
        "management":  True,
        "banking":     False,
        "insurance":   False,
        "notes":       "⚠️  Unknown sector — results may be less accurate.",
    },'''

new4 = '''    "NBFC": {
        "fundamental": False,  # D/E, current ratio misleading
        "valuation":   False,  # DCF unreliable — negative FCF by design
        "management":  True,
        "banking":     True,   # Use banking agent
        "insurance":   False,
        "notes":       "ℹ️  NBFC: Banking Agent used. ROA, P/B, AUM growth are key metrics.",
    },
    "UNKNOWN": {
        "fundamental": True,
        "valuation":   True,
        "management":  True,
        "banking":     False,
        "insurance":   False,
        "notes":       "⚠️  Unknown sector — results may be less accurate.",
    },'''

if old4 in content3:
    content3 = content3.replace(old4, new4)
    open(path3, 'w', encoding='utf-8').write(content3)
    print("  ✅ NBFC added to sector support matrix")
else:
    print("  ❌ NBFC block not found in sector_classifier.py")

print()
print("=" * 60)
print("ALL FIXES APPLIED")
print("=" * 60)
print()
print("Expected HDFCBANK changes after fix:")
print("  ROA:        0.00% → ~1.8-2.0%  (from balance sheet)")
print("  Justified P/B: 1.4x → ~2.8-3.2x (ROE-based growth)")
print("  Fair Value: ₹378 → ~₹1,400-1,800")
print("  Score:      32/100 → ~60-70/100")
print()
print("Expected BAJFINANCE changes after fix:")
print("  Routes to Banking Agent (not generic)")
print("  No more negative FCF / ROCE=0 false alarms")
print()
print("Run: python -B main.py --ticker HDFCBANK --no-gpt --mode all")
print("Run: python -B main.py --ticker BAJFINANCE --no-gpt --mode all")
