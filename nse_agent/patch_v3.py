"""
patch_v3.py — Fix EPV and Relative Valuation USD conversion
Both models use info.ebitda directly without fx conversion.
Run: python patch_v3.py
"""

path = 'agents/valuation_agent.py'
content = open(path, 'r', encoding='utf-8').read()

fixes = 0

# ── FIX 1: EPV — add fx detection before ebitda usage ──────────────────────
old1 = '''        ebitda  = info.get("ebitda", 0) or 0
        revenue = info.get("totalRevenue", 0) or 0
        ebit    = ebitda - (revenue * 0.03)
        nopat   = ebit * (1 - 0.25)'''

new1 = '''        _rev_epv = info.get("totalRevenue", 0) or 0
        _fx_epv  = 84.0 if 0 < _rev_epv < 1_000_000_000_000 else 1.0
        ebitda  = (info.get("ebitda", 0) or 0) * _fx_epv
        revenue = _rev_epv * _fx_epv
        ebit    = ebitda - (revenue * 0.03)
        nopat   = ebit * (1 - 0.25)'''

if old1 in content:
    content = content.replace(old1, new1)
    fixes += 1
    print("✅ Fix 1 applied: EPV ebitda/revenue USD conversion")
else:
    print("❌ Fix 1 NOT found — searching for EPV ebitda line:")
    for i, l in enumerate(content.split('\n')):
        if 'ebitda' in l and 'epv' not in l.lower() and 'info.get' in l:
            print(f"  Line {i}: {repr(l)}")

# ── FIX 2: Relative — add fx detection before ebitda usage ─────────────────
old2 = '''        eps           = ratios["valuation"]["eps"]
        ebitda        = info.get("ebitda", 0) or 0
        net_debt      = (info.get("totalDebt", 0) or 0) - (info.get("totalCash", 0) or 0)'''

new2 = '''        eps           = ratios["valuation"]["eps"]
        _rev_rel = info.get("totalRevenue", 0) or 0
        _fx_rel  = 84.0 if 0 < _rev_rel < 1_000_000_000_000 else 1.0
        ebitda        = (info.get("ebitda", 0) or 0) * _fx_rel
        net_debt      = (info.get("totalDebt", 0) or 0) - (info.get("totalCash", 0) or 0)'''

if old2 in content:
    content = content.replace(old2, new2)
    fixes += 1
    print("✅ Fix 2 applied: Relative ebitda USD conversion")
else:
    print("❌ Fix 2 NOT found — searching for Relative ebitda line:")
    for i, l in enumerate(content.split('\n')):
        if 'ebitda' in l and 'info.get' in l:
            print(f"  Line {i}: {repr(l)}")

# ── Save ────────────────────────────────────────────────────────────────────
if fixes > 0:
    open(path, 'w', encoding='utf-8').write(content)
    print(f"\n✅ {fixes}/2 fixes applied and saved.")
    print("\nExpected INFY results after fix:")
    print("  EPV IV:      ~₹800-900/share  (was ₹7.84)")
    print("  Relative IV: ~₹1,400-1,600/share  (was ₹845)")
    print("  Composite:   ~₹1,500-1,700/share  → 🟢 BUY")
    print("\nNow run: python -B main.py --ticker INFY --no-gpt --mode valuation")
else:
    print("\n❌ No fixes applied. Paste output above for diagnosis.")
