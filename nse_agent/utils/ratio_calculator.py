# ============================================================
# utils/ratio_calculator.py  — VERSION 1.1
# Fixed: Dividend yield anomaly, asset-light detection added
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger


class RatioCalculator:

    def __init__(self, raw_data: dict):
        self.data     = raw_data
        self.info     = raw_data.get("info", {})
        self.income   = raw_data.get("income_statement", pd.DataFrame())
        self.balance  = raw_data.get("balance_sheet", pd.DataFrame())
        self.cashflow = raw_data.get("cash_flow", pd.DataFrame())
        self.price    = raw_data.get("current_price", 0)

    def _get(self, df: pd.DataFrame, *keys) -> float:
        for key in keys:
            try:
                if key in df.index:
                    val = df.loc[key].iloc[0]
                    if pd.notna(val) and val != 0:
                        return float(val)
            except Exception:
                continue
        return 0.0

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        if denominator == 0 or pd.isna(denominator):
            return 0.0
        return numerator / denominator

    def get_valuation_ratios(self) -> dict:
        logger.info("📐 Calculating valuation ratios...")
        pe_ratio  = self.info.get("trailingPE", 0) or 0
        pb_ratio  = self.info.get("priceToBook", 0) or 0
        eps       = self.info.get("trailingEps", 0) or 0
        ev        = self.info.get("enterpriseValue", 0) or 0
        ebitda    = self.info.get("ebitda", 0) or 0
        ev_ebitda = self._safe_divide(ev, ebitda)
        eps_growth = (self.info.get("earningsGrowth", 0) or 0) * 100
        peg_ratio  = self._safe_divide(pe_ratio, eps_growth) if eps_growth > 0 else 0
        earnings_yield = self._safe_divide(eps, self.price) * 100
        return {
            "pe_ratio":       round(pe_ratio, 2),
            "pb_ratio":       round(pb_ratio, 2),
            "ev_ebitda":      round(ev_ebitda, 2),
            "peg_ratio":      round(peg_ratio, 2),
            "earnings_yield": round(earnings_yield, 2),
            "eps":            round(eps, 2),
        }

    def get_profitability_ratios(self) -> dict:
        logger.info("💹 Calculating profitability ratios...")
        roe           = (self.info.get("returnOnEquity", 0) or 0) * 100
        roa           = (self.info.get("returnOnAssets", 0) or 0) * 100
        net_income    = self._get(self.income, "Net Income", "Net Income Common Stockholders")
        revenue       = self._get(self.income, "Total Revenue", "Revenue")
        net_margin    = self._safe_divide(net_income, revenue) * 100
        ebitda        = self.info.get("ebitda", 0) or 0
        ebitda_margin = self._safe_divide(ebitda, revenue) * 100
        ebit             = self._get(self.income, "EBIT", "Operating Income")
        total_assets     = self._get(self.balance, "Total Assets")
        current_liab     = self._get(self.balance, "Current Liabilities", "Total Current Liabilities")
        capital_employed = total_assets - current_liab
        roce             = self._safe_divide(ebit, capital_employed) * 100
        return {
            "roe":           round(roe, 2),
            "roa":           round(roa, 2),
            "roce":          round(roce, 2),
            "net_margin":    round(net_margin, 2),
            "ebitda_margin": round(ebitda_margin, 2),
        }

    def get_leverage_ratios(self) -> dict:
        logger.info("⚖️  Calculating leverage ratios...")
        de_ratio         = (self.info.get("debtToEquity", 0) or 0) / 100
        current_assets   = self._get(self.balance, "Current Assets", "Total Current Assets")
        current_liab     = self._get(self.balance, "Current Liabilities", "Total Current Liabilities")
        current_ratio    = self._safe_divide(current_assets, current_liab)
        ebit             = self._get(self.income, "EBIT", "Operating Income")
        interest_expense = abs(self._get(self.income, "Interest Expense"))
        interest_coverage = self._safe_divide(ebit, interest_expense)
        total_debt       = self.info.get("totalDebt", 0) or 0
        cash             = self.info.get("totalCash", 0) or 0
        net_debt         = total_debt - cash
        ebitda           = self.info.get("ebitda", 1) or 1
        net_debt_ebitda  = self._safe_divide(net_debt, ebitda)
        return {
            "debt_to_equity":    round(de_ratio, 2),
            "current_ratio":     round(current_ratio, 2),
            "interest_coverage": round(interest_coverage, 2),
            "net_debt_cr":       round(net_debt / 1e7, 2),
            "net_debt_ebitda":   round(net_debt_ebitda, 2),
        }

    def get_cashflow_ratios(self) -> dict:
        logger.info("💸 Calculating cash flow ratios...")
        cfo               = self._get(self.cashflow, "Operating Cash Flow", "Cash Flow From Operations")
        capex             = abs(self._get(self.cashflow, "Capital Expenditure"))
        fcf               = cfo - capex
        net_income        = self._get(self.income, "Net Income", "Net Income Common Stockholders")
        cfo_to_net_income = self._safe_divide(cfo, net_income)
        market_cap        = self.info.get("marketCap", 1) or 1
        fcf_yield         = self._safe_divide(fcf, market_cap) * 100
        return {
            "cfo_cr":            round(cfo / 1e7, 2),
            "fcf_cr":            round(fcf / 1e7, 2),
            "capex_cr":          round(capex / 1e7, 2),
            "cfo_to_net_income": round(cfo_to_net_income, 2),
            "fcf_yield":         round(fcf_yield, 2),
        }

    def get_growth_ratios(self) -> dict:
        logger.info("📈 Calculating growth ratios...")
        revenue_growth   = (self.info.get("revenueGrowth", 0) or 0) * 100
        earnings_growth  = (self.info.get("earningsGrowth", 0) or 0) * 100
        revenue_cagr_3yr = 0
        if not self.income.empty and self.income.shape[1] >= 4:
            try:
                revenues = []
                for col in self.income.columns[:4]:
                    for key in ["Total Revenue", "Revenue"]:
                        if key in self.income.index:
                            val = self.income.loc[key, col]
                            if pd.notna(val):
                                revenues.append(float(val))
                                break
                if len(revenues) >= 2:
                    n_years = len(revenues) - 1
                    revenue_cagr_3yr = ((revenues[0] / revenues[-1]) ** (1/n_years) - 1) * 100
            except Exception as e:
                logger.warning(f"Could not calculate Revenue CAGR: {e}")
        return {
            "revenue_growth_yoy":  round(revenue_growth, 2),
            "earnings_growth_yoy": round(earnings_growth, 2),
            "revenue_cagr_3yr":    round(revenue_cagr_3yr, 2),
        }

    def get_dividend_data(self) -> dict:
        """
        ✅ FIXED v1.1: Handles Yahoo Finance dividend yield anomalies.
        Yahoo sometimes includes special one-time dividends in the yield calculation
        which inflates it massively (e.g., TCS showed 234% due to special dividend).
        We recalculate from rate/price when yield looks unrealistic (>20%).
        """
        div_rate  = self.info.get("dividendRate", 0) or 0
        div_yield = (self.info.get("dividendYield", 0) or 0) * 100

        # Recalculate if yield looks like a data error
        if div_yield > 20 and self.price > 0 and div_rate > 0:
            div_yield_corrected = (div_rate / self.price) * 100
            logger.warning(
                f"⚠️  Dividend yield anomaly: {div_yield:.1f}% → "
                f"Corrected to {div_yield_corrected:.2f}% (rate ÷ price)"
            )
            div_yield = div_yield_corrected

        payout_ratio = (self.info.get("payoutRatio", 0) or 0) * 100
        # Payout ratio >100% is usually a data error in Yahoo Finance
        if payout_ratio > 100:
            logger.warning(f"⚠️  Payout ratio {payout_ratio:.1f}% > 100% — capping")
            payout_ratio = 0

        return {
            "dividend_rate":  round(div_rate, 2),
            "dividend_yield": round(div_yield, 2),
            "payout_ratio":   round(payout_ratio, 2),
        }

    def calculate_graham_number(self) -> float:
        """
        Graham Number = √(22.5 × EPS × BVPS)
        Best for asset-heavy businesses. Less relevant for IT/FMCG.
        """
        eps  = self.info.get("trailingEps", 0) or 0
        bvps = self.info.get("bookValue", 0) or 0
        if eps <= 0 or bvps <= 0:
            logger.warning("⚠️ EPS or BVPS is negative/zero — Graham Number not applicable")
            return 0.0
        graham_number = np.sqrt(22.5 * eps * bvps)
        logger.info(f"📐 Graham Number for {self.data['ticker']}: ₹{graham_number:,.2f}")
        return round(graham_number, 2)

    def is_asset_light(self) -> bool:
        """
        Detect asset-light businesses (IT, FMCG, services).
        For these, DCF is more appropriate than Graham Number.
        """
        sector = self.info.get("sector", "").lower()
        roe    = (self.info.get("returnOnEquity", 0) or 0) * 100
        capex  = abs(self._get(self.cashflow, "Capital Expenditure"))
        revenue = self._get(self.income, "Total Revenue", "Revenue")
        capex_to_revenue = self._safe_divide(capex, revenue) * 100

        asset_light_sectors = [
            "technology", "information technology", "software",
            "consumer defensive", "communication services", "financial services"
        ]
        is_light = (
            any(s in sector for s in asset_light_sectors) or
            (roe > 30 and capex_to_revenue < 5)
        )
        if is_light:
            logger.info("🏢 Asset-light business — DCF more appropriate than Graham Number")
        return is_light

    def calculate_all_ratios(self) -> dict:
        logger.info("🔢 Calculating all financial ratios...")
        all_ratios = {
            "valuation":      self.get_valuation_ratios(),
            "profitability":  self.get_profitability_ratios(),
            "leverage":       self.get_leverage_ratios(),
            "cashflow":       self.get_cashflow_ratios(),
            "growth":         self.get_growth_ratios(),
            "dividend":       self.get_dividend_data(),
            "graham_number":  self.calculate_graham_number(),
            "is_asset_light": self.is_asset_light(),
        }
        logger.info("✅ All ratios calculated successfully")
        return all_ratios
