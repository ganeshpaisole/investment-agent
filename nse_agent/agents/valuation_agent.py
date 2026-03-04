# ============================================================
# agents/valuation_agent.py — VERSION 1.3
# Fixed: Cash flow currency detection using operatingCashflow
# from info dict (INR) vs cashflow DataFrame (may be USD)
# Root cause: INFY marketCap=INR but cashflow DataFrame=USD
# ============================================================

import numpy as np
import pandas as pd
from loguru import logger
from utils.data_fetcher import NSEDataFetcher
from utils.ratio_calculator import RatioCalculator


class ValuationAgent:

    INDIA_RISK_FREE_RATE      = 6.8
    INDIA_EQUITY_RISK_PREMIUM = 5.5
    INDIA_TERMINAL_GROWTH     = 6.5
    MARGIN_OF_SAFETY_TARGET   = 30
    USD_TO_INR                = 84.0

    def __init__(self, ticker: str):
        self.ticker = ticker.upper().strip()
        logger.info(f"\n{'💰'*20}")
        logger.info(f"  DCF VALUATION ENGINE: {self.ticker}")
        logger.info(f"{'💰'*20}")

    def _fetch_data(self) -> tuple:
        fetcher  = NSEDataFetcher(self.ticker)
        raw_data = fetcher.get_all_data()
        calc     = RatioCalculator(raw_data)
        ratios   = calc.calculate_all_ratios()
        return raw_data, ratios

    def _calculate_wacc(self, info: dict, sector: str = "") -> float:
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
        return round(wacc, 2)

    # ----------------------------------------------------------
    # MODEL 1: DCF
    # Key insight: Use info.freeCashflow & info.operatingCashflow
    # as primary source — these are reliably in INR for .NS stocks
    # The cashflow DataFrame may return USD for dual-listed stocks
    # ----------------------------------------------------------

    def _dcf_valuation(self, raw_data: dict, ratios: dict) -> dict:
        logger.info("🔢 Running DCF Valuation...")

        info          = raw_data.get("info", {})
        current_price = raw_data.get("current_price", 0)
        market_cap    = info.get("marketCap", 0) or 0

        # ---- Get FCF: Use info dict fields (reliably INR) ----
        # info.freeCashflow is the most reliable source for .NS stocks
        # Detect USD financials: INR revenue for large NSE stocks > 1 Trillion
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
        cfo_from_info = (info.get("operatingCashflow", 0) or 0) * fx

        # Sanity check: FCF should be reasonable vs market cap
        # For healthy large-caps: FCF yield 2-8% → FCF > MCap*0.02
        if fcf_from_info > market_cap * 0.005:
            base_fcf = fcf_from_info
            logger.info(f"📦 Using info.freeCashflow: ₹{base_fcf/1e7:,.0f} Cr")
        elif cfo_from_info > 0:
            # Estimate capex as 3% of revenue (IT sector benchmark)
            revenue  = info.get("totalRevenue", 0) or 0
            capex_est = revenue * 0.03
            base_fcf  = cfo_from_info - capex_est
            logger.info(
                f"📦 CFO ₹{cfo_from_info/1e7:,.0f}Cr - "
                f"Est.Capex ₹{capex_est/1e7:,.0f}Cr = "
                f"FCF ₹{base_fcf/1e7:,.0f}Cr"
            )
        else:
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
            )

        logger.info(f"✅ Base FCF confirmed: ₹{base_fcf/1e7:,.0f} Cr")

        # ---- Growth rates ----
        rev_growth     = ratios["growth"]["revenue_cagr_3yr"]
        roe            = ratios["profitability"]["roe"]
        is_asset_light = ratios["is_asset_light"]

        if is_asset_light and roe > 30:
            stage1_growth = min(max(rev_growth, 8), 15)
        elif roe > 20:
            stage1_growth = min(max(rev_growth, 6), 18)
        else:
            stage1_growth = min(max(rev_growth, 4), 12)

        stage2_growth   = stage1_growth * 0.6
        terminal_growth = self.INDIA_TERMINAL_GROWTH / 100

        logger.info(
            f"📈 Growth — Stage1: {stage1_growth:.1f}% | "
            f"Stage2: {stage2_growth:.1f}% | Terminal: {self.INDIA_TERMINAL_GROWTH}%"
        )

        # ---- WACC ----
        _sector = info.get("sector", "")
        wacc   = self._calculate_wacc(info, sector=_sector)
        wacc_r = wacc / 100

        # ---- Discount FCFs ----
        pv_cash_flows = 0
        fcf = base_fcf
        for year in range(1, 6):
            fcf = fcf * (1 + stage1_growth / 100)
            pv_cash_flows += fcf / (1 + wacc_r) ** year
        for year in range(6, 11):
            fcf = fcf * (1 + stage2_growth / 100)
            pv_cash_flows += fcf / (1 + wacc_r) ** year

        # ---- Terminal Value ----
        terminal_value    = fcf * (1 + terminal_growth) / (wacc_r - terminal_growth)
        pv_terminal_value = terminal_value / (1 + wacc_r) ** 10
        total_pv          = pv_cash_flows + pv_terminal_value

        # ---- Equity Value ----
        # info.totalDebt and totalCash are in INR for .NS stocks
        net_debt   = (info.get("totalDebt", 0) or 0) - (info.get("totalCash", 0) or 0)
        equity_val = total_pv - net_debt

        # ---- Shares: derive from MCap/Price (most reliable) ----
        shares = market_cap / current_price if current_price > 0 else (
            info.get("sharesOutstanding", 1) or 1
        )

        dcf_per_share = equity_val / shares if shares > 0 else 0

        # Warn if terminal value dominates — DCF less reliable
        tv_pct = pv_terminal_value / total_pv * 100 if total_pv > 0 else 0
        if tv_pct > 80:
            logger.warning(
                f"⚠️  Terminal value = {tv_pct:.0f}% of DCF — "
                f"highly sensitive to long-term assumptions. "
                f"Weight DCF less for cyclical/capex-heavy stocks."
            )

        logger.info(f"✅ DCF Intrinsic Value: ₹{dcf_per_share:,.2f}/share")

        return {
            "model":           "DCF",
            "intrinsic_value": round(dcf_per_share, 2),
            "base_fcf_cr":     round(base_fcf / 1e7, 0),
            "stage1_growth":   stage1_growth,
            "stage2_growth":   round(stage2_growth, 1),
            "terminal_growth": self.INDIA_TERMINAL_GROWTH,
            "wacc":            wacc,
            "pv_fcf_cr":       round(pv_cash_flows / 1e7, 0),
            "pv_terminal_cr":  round(pv_terminal_value / 1e7, 0),
            "tv_pct_of_total": round(pv_terminal_value / total_pv * 100, 1),
        }

    # ----------------------------------------------------------
    # MODEL 2: EPV
    # ----------------------------------------------------------

    def _epv_valuation(self, raw_data: dict, ratios: dict) -> dict:
        logger.info("🔢 Running EPV Valuation...")

        info          = raw_data.get("info", {})
        current_price = raw_data.get("current_price", 0)
        market_cap    = info.get("marketCap", 0) or 0

        _rev_epv = info.get("totalRevenue", 0) or 0
        _fx_epv  = 84.0 if 0 < _rev_epv < 1_000_000_000_000 else 1.0
        ebitda  = (info.get("ebitda", 0) or 0) * _fx_epv
        revenue = _rev_epv * _fx_epv
        ebit    = ebitda - (revenue * 0.03)
        nopat   = ebit * (1 - 0.25)

        wacc   = self._calculate_wacc(info, sector=info.get("sector", ""))
        wacc_r = wacc / 100

        epv_enterprise = nopat / wacc_r if wacc_r > 0 else 0
        net_debt       = (info.get("totalDebt", 0) or 0) - (info.get("totalCash", 0) or 0)
        epv_equity     = epv_enterprise - net_debt

        shares = market_cap / current_price if current_price > 0 else (
            info.get("sharesOutstanding", 1) or 1
        )
        epv_share = epv_equity / shares if shares > 0 else 0
        logger.info(f"✅ EPV Intrinsic Value: ₹{epv_share:,.2f}/share")

        return {
            "model":           "EPV",
            "intrinsic_value": round(epv_share, 2),
            "nopat_cr":        round(nopat / 1e7, 0),
            "wacc":            wacc,
        }

    # ----------------------------------------------------------
    # MODEL 3: DDM
    # ----------------------------------------------------------

    def _ddm_valuation(self, raw_data: dict, ratios: dict) -> dict:
        logger.info("🔢 Running DDM Valuation...")

        info     = raw_data.get("info", {})
        div_data = ratios["dividend"]
        div_rate = div_data["dividend_rate"]

        if div_rate <= 0:
            logger.warning("⚠️ No dividend — DDM not applicable")
            return {"model": "DDM", "intrinsic_value": 0, "applicable": False}

        roe          = ratios["profitability"]["roe"] / 100
        payout_ratio = div_data["payout_ratio"] / 100
        if payout_ratio <= 0 or payout_ratio > 1:
            payout_ratio = 0.47

        retention = 1 - payout_ratio
        g         = min(roe * retention, 0.12)
        beta      = info.get("beta", 1.0) or 1.0
        ke        = (self.INDIA_RISK_FREE_RATE + beta * self.INDIA_EQUITY_RISK_PREMIUM) / 100

        if ke <= g:
            logger.warning("⚠️ Growth rate ≥ cost of equity — DDM gives infinite value, skipping")
            return {"model": "DDM", "intrinsic_value": 0, "applicable": False}

        d1        = div_rate * (1 + g)
        ddm_value = d1 / (ke - g)
        logger.info(f"✅ DDM Intrinsic Value: ₹{ddm_value:,.2f}/share")

        return {
            "model":           "DDM",
            "intrinsic_value": round(ddm_value, 2),
            "dividend_d1":     round(d1, 2),
            "growth_rate_g":   round(g * 100, 2),
            "cost_of_equity":  round(ke * 100, 2),
            "applicable":      True,
        }

    # ----------------------------------------------------------
    # MODEL 4: Relative Valuation
    # ----------------------------------------------------------

    def _relative_valuation(self, raw_data: dict, ratios: dict) -> dict:
        logger.info("🔢 Running Relative Valuation...")

        info          = raw_data.get("info", {})
        current_price = raw_data.get("current_price", 0)
        sector        = info.get("sector", "").lower()
        eps           = ratios["valuation"]["eps"]
        _rev_rel = info.get("totalRevenue", 0) or 0
        _fx_rel  = 84.0 if 0 < _rev_rel < 1_000_000_000_000 else 1.0
        ebitda        = (info.get("ebitda", 0) or 0) * _fx_rel
        net_debt      = (info.get("totalDebt", 0) or 0) - (info.get("totalCash", 0) or 0)
        market_cap    = info.get("marketCap", 0) or 0
        shares        = market_cap / current_price if current_price > 0 else (
            info.get("sharesOutstanding", 1) or 1
        )

        sector_pe = {
            "technology": 24, "information technology": 24,
            "financial services": 18, "consumer defensive": 35,
            "consumer cyclical": 28, "energy": 12, "basic materials": 14,
            "industrials": 22, "healthcare": 28, "real estate": 20,
            "utilities": 16, "communication services": 20,
        }
        sector_ev_ebitda = {
            "technology": 16, "information technology": 16,
            "financial services": 12, "consumer defensive": 22,
            "consumer cyclical": 18, "energy": 8, "basic materials": 9,
            "industrials": 14, "healthcare": 18, "real estate": 14,
            "utilities": 10, "communication services": 12,
        }

        fair_pe        = sector_pe.get(sector, 20)
        fair_ev_ebitda = sector_ev_ebitda.get(sector, 14)
        pe_value       = eps * fair_pe if eps > 0 else 0
        ev_equity      = ebitda * fair_ev_ebitda - net_debt
        ev_value_ps    = ev_equity / shares if shares > 0 else 0

        relative_value = (
            pe_value * 0.5 + ev_value_ps * 0.5 if pe_value > 0 and ev_value_ps > 0
            else pe_value or ev_value_ps
        )
        logger.info(f"✅ Relative Valuation: ₹{relative_value:,.2f}/share")

        return {
            "model":           "Relative",
            "intrinsic_value": round(relative_value, 2),
            "pe_based_value":  round(pe_value, 2),
            "ev_based_value":  round(ev_value_ps, 2),
            "fair_pe_used":    fair_pe,
            "fair_ev_used":    fair_ev_ebitda,
            "sector":          sector,
        }

    # ----------------------------------------------------------
    # COMPOSITE + RECOMMENDATION
    # ----------------------------------------------------------

    def _composite_value(self, dcf, epv, ddm, rel) -> dict:
        dcf_iv = dcf["intrinsic_value"]
        epv_iv = epv["intrinsic_value"]
        rel_iv = rel["intrinsic_value"]
        ddm_iv = ddm.get("intrinsic_value", 0)
        ddm_ok = ddm.get("applicable", False)

        weights = (
            {"dcf": 0.40, "epv": 0.25, "relative": 0.20, "ddm": 0.15}
            if ddm_ok and ddm_iv > 0
            else {"dcf": 0.50, "epv": 0.25, "relative": 0.25, "ddm": 0.0}
        )
        composite = (
            dcf_iv * weights["dcf"] + epv_iv * weights["epv"] +
            rel_iv * weights["relative"] +
            (ddm_iv * weights["ddm"] if ddm_ok else 0)
        )
        return {
            "composite_iv": round(composite, 2),
            "weights": weights,
            "dcf_iv": dcf_iv, "epv_iv": epv_iv,
            "rel_iv": rel_iv, "ddm_iv": ddm_iv if ddm_ok else "N/A",
        }

    def _get_recommendation(self, current_price: float, composite_iv: float) -> dict:
        if composite_iv <= 0:
            return {"signal": "⚪ INSUFFICIENT DATA", "upside_pct": 0,
                    "mos_pct": 0, "entry_price": 0, "target_price": 0}
        upside_pct   = ((composite_iv - current_price) / current_price) * 100
        mos_pct      = ((composite_iv - current_price) / composite_iv) * 100
        entry_price  = composite_iv * (1 - self.MARGIN_OF_SAFETY_TARGET / 100)
        target_price = composite_iv * 1.10
        if upside_pct >= 30:   signal = "🟢 STRONG BUY — Significant discount to intrinsic value"
        elif upside_pct >= 15: signal = "🟢 BUY — Trading below intrinsic value"
        elif upside_pct >= 0:  signal = "🟡 FAIRLY VALUED — Accumulate on dips"
        elif upside_pct >= -15: signal = "🟠 HOLD — Slightly above intrinsic value"
        else:                  signal = "🔴 AVOID — Significantly overvalued vs intrinsic value"
        return {
            "signal": signal, "upside_pct": round(upside_pct, 1),
            "mos_pct": round(mos_pct, 1), "entry_price": round(entry_price, 2),
            "target_price": round(target_price, 2),
        }

    def _valuation_score(self, upside_pct: float, mos_pct: float) -> int:
        if upside_pct >= 40:    return 95
        elif upside_pct >= 30:  return 85
        elif upside_pct >= 20:  return 75
        elif upside_pct >= 10:  return 65
        elif upside_pct >= 0:   return 55
        elif upside_pct >= -10: return 40
        elif upside_pct >= -20: return 25
        else:                   return 10

    def analyze(self) -> dict:
        raw_data, ratios = self._fetch_data()
        current_price    = raw_data.get("current_price", 0)
        info             = raw_data.get("info", {})

        dcf_result = self._dcf_valuation(raw_data, ratios)
        epv_result = self._epv_valuation(raw_data, ratios)
        ddm_result = self._ddm_valuation(raw_data, ratios)
        rel_result = self._relative_valuation(raw_data, ratios)

        composite = self._composite_value(dcf_result, epv_result, ddm_result, rel_result)
        rec       = self._get_recommendation(current_price, composite["composite_iv"])
        score     = self._valuation_score(rec["upside_pct"], rec["mos_pct"])

        result = {
            "ticker": self.ticker, "current_price": current_price,
            "company_name": info.get("longName", self.ticker),
            "sector": info.get("sector", "N/A"),
            "models": {"dcf": dcf_result, "epv": epv_result,
                       "ddm": ddm_result, "relative": rel_result},
            "composite": composite, "recommendation": rec,
            "valuation_score": score, "ratios": ratios,
        }
        logger.info(
            f"✅ Valuation complete — IV: ₹{composite['composite_iv']:,.2f} | "
            f"CMP: ₹{current_price:,.2f} | Upside: {rec['upside_pct']}%"
        )
        return result

