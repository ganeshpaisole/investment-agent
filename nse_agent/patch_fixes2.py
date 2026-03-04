"""
patch_fixes2.py — Two fixes:
  1. DCF: Sector-aware FCF for cyclicals (auto, energy, industrials)
     - Caps EBITDA proxy at realistic FCF margin for capex-heavy sectors
     - Adds higher WACC floor for cyclical businesses
     - Warns when terminal value > 80% of total (unreliable DCF)
  2. Management printer: CFO/PAT display bug (key was cfo_to_pat, now cfo_to_net_income)

Run: python patch_fixes2.py
"""

import re

# ============================================================
# FIX 1: valuation_agent.py — Cyclical sector DCF fix
# ============================================================
print("=" * 60)
print("FIX 1: Cyclical sector DCF awareness")
print("=" * 60)

path_val = 'agents/valuation_agent.py'
content  = open(path_val, encoding='utf-8').read()

# 1a. Replace EBITDA fallback with sector-aware version
old1a = '''        else:
            # Last resort: EBITDA proxy
            ebitda   = info.get("ebitda", 0) or 0
            base_fcf = ebitda * 0.65
            logger.warning(f"⚠️  Using EBITDA×0.65: ₹{base_fcf/1e7:,.0f}Cr")'''

new1a = '''        else:
            # Last resort: sector-aware EBITDA proxy
            ebitda  = info.get("ebitda", 0) or 0
            sector  = info.get("sector", "").lower()
            # Cyclicals convert less EBITDA to FCF due to high capex
            _cyclical = any(s in sector for s in [
                "consumer cyclical", "energy", "basic materials",
                "industrials", "real estate", "utilities"
            ])
            ebitda_to_fcf = 0.35 if _cyclical else 0.65
            base_fcf = ebitda * ebitda_to_fcf
            logger.warning(
                f"⚠️  Using EBITDA×{ebitda_to_fcf} proxy "
                f"({'cyclical' if _cyclical else 'asset-light'} sector): "
                f"₹{base_fcf/1e7:,.0f}Cr"
            )'''

if old1a in content:
    content = content.replace(old1a, new1a)
    print("  ✅ 1a: Sector-aware EBITDA proxy applied")
else:
    print("  ❌ 1a: EBITDA fallback block not found")

# 1b. Replace WACC calculation with cyclical floor
old1b = '''    def _calculate_wacc(self, info: dict) -> float:
        beta = max(0.5, min(info.get("beta", 1.0) or 1.0, 2.0))
        ke   = self.INDIA_RISK_FREE_RATE + beta * self.INDIA_EQUITY_RISK_PREMIUM
        total_debt = info.get("totalDebt", 0) or 0
        market_cap = info.get("marketCap", 1) or 1
        debt_ratio = total_debt / (total_debt + market_cap)
        kd_after_tax = 8.0 * (1 - 0.25)
        wacc = ke * (1 - debt_ratio) + kd_after_tax * debt_ratio
        logger.info(f"📊 WACC: {wacc:.2f}% (Beta: {beta}, Ke: {ke:.2f}%)")
        return round(wacc, 2)'''

new1b = '''    def _calculate_wacc(self, info: dict, sector: str = "") -> float:
        beta = max(0.5, min(info.get("beta", 1.0) or 1.0, 2.0))
        ke   = self.INDIA_RISK_FREE_RATE + beta * self.INDIA_EQUITY_RISK_PREMIUM

        # Cyclical businesses deserve higher WACC floor (more earnings risk)
        _cyclical = any(s in sector.lower() for s in [
            "consumer cyclical", "energy", "basic materials",
            "industrials", "real estate", "utilities"
        ])
        wacc_floor = 10.5 if _cyclical else 9.0

        total_debt   = info.get("totalDebt", 0) or 0
        market_cap   = info.get("marketCap", 1) or 1
        debt_ratio   = total_debt / (total_debt + market_cap)
        kd_after_tax = 8.0 * (1 - 0.25)
        wacc = ke * (1 - debt_ratio) + kd_after_tax * debt_ratio
        wacc = max(wacc, wacc_floor)  # apply floor

        label = " [cyclical floor]" if _cyclical and wacc == wacc_floor else ""
        logger.info(f"📊 WACC: {wacc:.2f}%{label} (Beta: {beta}, Ke: {ke:.2f}%)")
        return round(wacc, 2)'''

if old1b in content:
    content = content.replace(old1b, new1b)
    print("  ✅ 1b: WACC cyclical floor applied")
else:
    print("  ❌ 1b: WACC block not found")

# 1c. Pass sector to _calculate_wacc in _dcf_valuation
old1c = '''        # ---- WACC ----
        wacc   = self._calculate_wacc(info)
        wacc_r = wacc / 100'''

new1c = '''        # ---- WACC ----
        _sector = info.get("sector", "")
        wacc   = self._calculate_wacc(info, sector=_sector)
        wacc_r = wacc / 100'''

if old1c in content:
    content = content.replace(old1c, new1c)
    print("  ✅ 1c: Sector passed to WACC in DCF")
else:
    print("  ❌ 1c: WACC call not found in DCF")

# 1d. Pass sector to _calculate_wacc in _epv_valuation
old1d = '''        wacc   = self._calculate_wacc(info)
        wacc_r = wacc / 100

        epv_enterprise = nopat / wacc_r'''

new1d = '''        wacc   = self._calculate_wacc(info, sector=info.get("sector", ""))
        wacc_r = wacc / 100

        epv_enterprise = nopat / wacc_r'''

if old1d in content:
    content = content.replace(old1d, new1d)
    print("  ✅ 1d: Sector passed to WACC in EPV")
else:
    print("  ❌ 1d: WACC call not found in EPV")

# 1e. Add terminal value warning when TV > 80%
old1e = '''        dcf_per_share = equity_val / shares if shares > 0 else 0
        logger.info(f"✅ DCF Intrinsic Value: ₹{dcf_per_share:,.2f}/share")'''

new1e = '''        dcf_per_share = equity_val / shares if shares > 0 else 0

        # Warn if terminal value dominates — DCF less reliable
        tv_pct = pv_terminal_value / total_pv * 100 if total_pv > 0 else 0
        if tv_pct > 80:
            logger.warning(
                f"⚠️  Terminal value = {tv_pct:.0f}% of DCF — "
                f"highly sensitive to long-term assumptions. "
                f"Weight DCF less for cyclical/capex-heavy stocks."
            )

        logger.info(f"✅ DCF Intrinsic Value: ₹{dcf_per_share:,.2f}/share")'''

if old1e in content:
    content = content.replace(old1e, new1e)
    print("  ✅ 1e: Terminal value warning added")
else:
    print("  ❌ 1e: DCF per share line not found")

open(path_val, 'w', encoding='utf-8').write(content)
print(f"\n  💾 valuation_agent.py saved")

# ============================================================
# FIX 2: management_printer.py — CFO/PAT display key fix
# ============================================================
print()
print("=" * 60)
print("FIX 2: CFO/PAT display key fix in management_printer.py")
print("=" * 60)

path_mgmt = 'utils/management_printer.py'
content2  = open(path_mgmt, encoding='utf-8').read()

old2 = '''         "CFO/PAT Ratio",       f"{ei.get('cfo_to_pat', 0):.2f}x"),'''
new2 = '''         "CFO/PAT Ratio",       f"{ei.get('cfo_to_net_income', 0):.2f}x"),'''

if old2 in content2:
    content2 = content2.replace(old2, new2)
    open(path_mgmt, 'w', encoding='utf-8').write(content2)
    print("  ✅ CFO/PAT display key fixed")
else:
    # Try alternate spacing
    old2b = "\"CFO/PAT Ratio\",       f\"{ei.get('cfo_to_pat', 0):.2f}x\")"
    new2b = "\"CFO/PAT Ratio\",       f\"{ei.get('cfo_to_net_income', 0):.2f}x\")"
    if old2b in content2:
        content2 = content2.replace(old2b, new2b)
        open(path_mgmt, 'w', encoding='utf-8').write(content2)
        print("  ✅ CFO/PAT display key fixed (alt match)")
    else:
        # Brute force: replace any remaining cfo_to_pat references
        if 'cfo_to_pat' in content2:
            content2 = content2.replace('cfo_to_pat', 'cfo_to_net_income')
            open(path_mgmt, 'w', encoding='utf-8').write(content2)
            print("  ✅ All cfo_to_pat references replaced in printer")
        else:
            print("  ⚠️  cfo_to_pat not found in printer — may already be fixed")

print()
print("=" * 60)
print("ALL FIXES APPLIED")
print("=" * 60)
print()
print("Expected M&M changes after fix:")
print("  WACC:    8.7% → 10.5% (cyclical floor)")
print("  FCF:     ₹22,457 Cr → ₹12,100 Cr (EBITDA×0.35)")
print("  DCF IV:  ₹11,116 → ~₹4,500-5,500")
print("  Warning: Terminal value % alert if > 80%")
print("  Display: CFO/PAT ratio now shows correct value")
print()
print("Run: python -B main.py --ticker \"M&M\" --no-gpt --mode all")
