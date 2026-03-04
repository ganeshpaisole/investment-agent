# ============================================================
# agents/fundamental_agent.py — VERSION 1.1
# Fixed: Uses corrected dividend data from ratio_calculator
# Fixed: Asset-light valuation signal updated
# ============================================================

from loguru import logger
from config.settings import GRAHAM_THRESHOLDS, GRADE_MAP
from utils.data_fetcher import NSEDataFetcher
from utils.ratio_calculator import RatioCalculator


class FundamentalAnalysisAgent:
    """
    Evaluates the fundamental health of an NSE-listed company.
    Scores across 4 categories (25 pts each) = 100 pts total.

    Usage:
        agent = FundamentalAnalysisAgent("TCS")
        result = agent.analyze()
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.upper().strip()
        logger.info(f"\n{'🔍'*20}")
        logger.info(f"  FUNDAMENTAL ANALYSIS: {self.ticker}")
        logger.info(f"{'🔍'*20}")

    def _fetch_data(self) -> tuple:
        fetcher  = NSEDataFetcher(self.ticker)
        raw_data = fetcher.get_all_data()
        calc     = RatioCalculator(raw_data)
        ratios   = calc.calculate_all_ratios()
        return raw_data, ratios

    # ----------------------------------------------------------
    # CATEGORY 1: Earnings Quality (25 pts)
    # ----------------------------------------------------------

    def _score_earnings_quality(self, ratios: dict, raw_data: dict) -> dict:
        score     = 0
        breakdown = []
        t         = GRAHAM_THRESHOLDS
        growth    = ratios["growth"]
        cashflow  = ratios["cashflow"]
        valuation = ratios["valuation"]

        # EPS Growth (10 pts)
        eps_growth = growth["earnings_growth_yoy"]
        if eps_growth >= t["min_eps_cagr_10yr"]:
            score += 10
            breakdown.append(f"  ✅ EPS Growth {eps_growth:.1f}% ≥ {t['min_eps_cagr_10yr']}% required [+10]")
        elif eps_growth >= 5:
            score += 5
            breakdown.append(f"  ⚠️  EPS Growth {eps_growth:.1f}% (moderate, target: {t['min_eps_cagr_10yr']}%) [+5]")
        elif eps_growth >= 0:
            score += 2
            breakdown.append(f"  ⚠️  EPS Growth {eps_growth:.1f}% (low but positive) [+2]")
        else:
            breakdown.append(f"  ❌ EPS Growth {eps_growth:.1f}% (YoY dip — check if one-time or structural) [+0]")

        # Cash Flow Quality (8 pts)
        cfo_ratio = cashflow["cfo_to_net_income"]
        if cfo_ratio >= t["min_cfo_to_net_income"]:
            score += 8
            breakdown.append(f"  ✅ CFO/Net Income {cfo_ratio:.2f} ≥ {t['min_cfo_to_net_income']} (real earnings) [+8]")
        elif cfo_ratio >= 0.7:
            score += 4
            breakdown.append(f"  ⚠️  CFO/Net Income {cfo_ratio:.2f} (acceptable, target ≥{t['min_cfo_to_net_income']}) [+4]")
        else:
            breakdown.append(f"  ❌ CFO/Net Income {cfo_ratio:.2f} — earnings quality concern [+0]")

        # Positive EPS (7 pts)
        eps = valuation["eps"]
        if eps > 0:
            score += 7
            breakdown.append(f"  ✅ Positive EPS ₹{eps} [+7]")
        else:
            breakdown.append(f"  ❌ Negative EPS ₹{eps} — company is loss-making [+0]")

        return {"category": "Earnings Quality", "score": score, "max_score": 25, "breakdown": breakdown}

    # ----------------------------------------------------------
    # CATEGORY 2: Balance Sheet Strength (25 pts)
    # ----------------------------------------------------------

    def _score_balance_sheet(self, ratios: dict) -> dict:
        score     = 0
        breakdown = []
        t         = GRAHAM_THRESHOLDS
        leverage  = ratios["leverage"]

        # Current Ratio (8 pts)
        cr = leverage["current_ratio"]
        if cr >= t["min_current_ratio"]:
            score += 8
            breakdown.append(f"  ✅ Current Ratio {cr:.2f} ≥ {t['min_current_ratio']} [+8]")
        elif cr >= 1.5:
            score += 4
            breakdown.append(f"  ⚠️  Current Ratio {cr:.2f} (borderline, target ≥{t['min_current_ratio']}) [+4]")
        else:
            breakdown.append(f"  ❌ Current Ratio {cr:.2f} — liquidity risk [+0]")

        # Debt/Equity (10 pts)
        de = leverage["debt_to_equity"]
        if de <= t["max_debt_to_equity"]:
            score += 10
            breakdown.append(f"  ✅ D/E Ratio {de:.2f} ≤ {t['max_debt_to_equity']} (low debt) [+10]")
        elif de <= 1.0:
            score += 5
            breakdown.append(f"  ⚠️  D/E Ratio {de:.2f} (moderate, target ≤{t['max_debt_to_equity']}) [+5]")
        else:
            breakdown.append(f"  ❌ D/E Ratio {de:.2f} — HIGH DEBT WARNING [+0]")

        # Interest Coverage (7 pts)
        ic = leverage["interest_coverage"]
        if ic >= t["min_interest_coverage"]:
            score += 7
            breakdown.append(f"  ✅ Interest Coverage {ic:.1f}x ≥ {t['min_interest_coverage']}x [+7]")
        elif ic >= 3:
            score += 3
            breakdown.append(f"  ⚠️  Interest Coverage {ic:.1f}x (target ≥{t['min_interest_coverage']}x) [+3]")
        else:
            breakdown.append(f"  ❌ Interest Coverage {ic:.1f}x — debt servicing risk [+0]")

        return {"category": "Balance Sheet Strength", "score": score, "max_score": 25, "breakdown": breakdown}

    # ----------------------------------------------------------
    # CATEGORY 3: Profitability & Returns (25 pts)
    # ----------------------------------------------------------

    def _score_profitability(self, ratios: dict) -> dict:
        score     = 0
        breakdown = []
        t         = GRAHAM_THRESHOLDS
        profit    = ratios["profitability"]

        # ROE (10 pts)
        roe = profit["roe"]
        if roe >= t["min_roe_5yr_avg"]:
            score += 10
            breakdown.append(f"  ✅ ROE {roe:.1f}% ≥ {t['min_roe_5yr_avg']}% (Buffett benchmark) [+10]")
        elif roe >= 10:
            score += 5
            breakdown.append(f"  ⚠️  ROE {roe:.1f}% (below target {t['min_roe_5yr_avg']}%) [+5]")
        else:
            breakdown.append(f"  ❌ ROE {roe:.1f}% — poor returns on equity [+0]")

        # ROCE (8 pts)
        roce = profit["roce"]
        if roce >= t["min_roce_5yr_avg"]:
            score += 8
            breakdown.append(f"  ✅ ROCE {roce:.1f}% ≥ {t['min_roce_5yr_avg']}% [+8]")
        elif roce >= 10:
            score += 4
            breakdown.append(f"  ⚠️  ROCE {roce:.1f}% (below target {t['min_roce_5yr_avg']}%) [+4]")
        else:
            breakdown.append(f"  ❌ ROCE {roce:.1f}% — inefficient capital use [+0]")

        # Net Margin (7 pts)
        nm = profit["net_margin"]
        if nm >= t["min_net_margin"]:
            score += 7
            breakdown.append(f"  ✅ Net Margin {nm:.1f}% ≥ {t['min_net_margin']}% [+7]")
        elif nm >= 5:
            score += 3
            breakdown.append(f"  ⚠️  Net Margin {nm:.1f}% (thin, target ≥{t['min_net_margin']}%) [+3]")
        else:
            breakdown.append(f"  ❌ Net Margin {nm:.1f}% — very thin/negative margins [+0]")

        return {"category": "Profitability & Returns", "score": score, "max_score": 25, "breakdown": breakdown}

    # ----------------------------------------------------------
    # CATEGORY 4: Dividend & Capital Allocation (25 pts)
    # ----------------------------------------------------------

    def _score_dividend_capital(self, ratios: dict) -> dict:
        """✅ FIXED v1.1 — uses corrected dividend data from ratio_calculator"""
        score     = 0
        breakdown = []
        t         = GRAHAM_THRESHOLDS

        # Use the corrected dividend data (not raw info)
        dividend     = ratios["dividend"]
        div_rate     = dividend["dividend_rate"]
        div_yield    = dividend["dividend_yield"]   # ✅ Already corrected
        payout_ratio = dividend["payout_ratio"]

        # FCF (10 pts)
        fcf = ratios["cashflow"]["fcf_cr"]
        if fcf > 0:
            score += 10
            breakdown.append(f"  ✅ Positive Free Cash Flow ₹{fcf:,.0f} Cr [+10]")
        else:
            breakdown.append(f"  ❌ Negative FCF ₹{fcf:,.0f} Cr — burning cash [+0]")

        # Dividend History (8 pts)
        if div_rate > 0 and div_yield > 0:
            score += 8
            breakdown.append(f"  ✅ Dividend Paying — Rate ₹{div_rate:.1f}, Yield {div_yield:.2f}% [+8]")
        elif div_rate > 0:
            score += 4
            breakdown.append(f"  ⚠️  Dividend paid (yield data unavailable) [+4]")
        else:
            breakdown.append(f"  ⚠️  No dividend — check if profits being reinvested wisely [+0]")

        # Payout Ratio (7 pts)
        if t["min_payout_ratio"] <= payout_ratio <= t["max_payout_ratio"]:
            score += 7
            breakdown.append(f"  ✅ Payout Ratio {payout_ratio:.1f}% (healthy 20–60% range) [+7]")
        elif 0 < payout_ratio <= 80:
            score += 3
            breakdown.append(f"  ⚠️  Payout Ratio {payout_ratio:.1f}% (outside ideal range) [+3]")
        elif payout_ratio == 0:
            breakdown.append(f"  ⚠️  Payout ratio not available [+0]")
        else:
            breakdown.append(f"  ❌ Payout Ratio {payout_ratio:.1f}% — unsustainably high [+0]")

        return {"category": "Dividend & Capital Allocation", "score": score, "max_score": 25, "breakdown": breakdown}

    # ----------------------------------------------------------
    # GRADE & VALUATION SUMMARY
    # ----------------------------------------------------------

    def _get_grade(self, total_score: float) -> tuple:
        for (low, high), (grade, recommendation) in GRADE_MAP.items():
            if low <= total_score <= high:
                return grade, recommendation
        return "F", "Poor — Definite Avoid"

    def _get_valuation_summary(self, ratios: dict, raw_data: dict) -> dict:
        current_price  = raw_data.get("current_price", 0)
        graham_number  = ratios["graham_number"]
        is_asset_light = ratios["is_asset_light"]
        val            = ratios["valuation"]

        margin_of_safety = 0
        valuation_signal = "N/A"

        if graham_number > 0 and current_price > 0:
            margin_of_safety = ((graham_number - current_price) / graham_number) * 100

            # ✅ FIXED: Asset-light companies need DCF, not Graham Number
            if is_asset_light:
                valuation_signal = (
                    "🔵 ASSET-LIGHT BUSINESS — Graham Number not applicable. "
                    "Use DCF model for true intrinsic value."
                )
            elif margin_of_safety >= 30:
                valuation_signal = "🟢 UNDERVALUED — Good Margin of Safety"
            elif margin_of_safety >= 10:
                valuation_signal = "🟡 FAIRLY VALUED — Limited Margin of Safety"
            elif margin_of_safety >= 0:
                valuation_signal = "🟠 SLIGHTLY OVERVALUED"
            else:
                valuation_signal = "🔴 OVERVALUED vs Graham Number (run DCF for full picture)"
        elif is_asset_light:
            valuation_signal = "🔵 ASSET-LIGHT BUSINESS — DCF valuation recommended"

        return {
            "current_price":     current_price,
            "graham_number":     graham_number,
            "margin_of_safety":  round(margin_of_safety, 2),
            "valuation_signal":  valuation_signal,
            "is_asset_light":    is_asset_light,
            "pe_ratio":          val["pe_ratio"],
            "pb_ratio":          val["pb_ratio"],
            "earnings_yield":    val["earnings_yield"],
        }

    # ----------------------------------------------------------
    # MAIN ANALYSIS
    # ----------------------------------------------------------

    def analyze(self) -> dict:
        raw_data, ratios = self._fetch_data()

        earnings_score = self._score_earnings_quality(ratios, raw_data)
        balance_score  = self._score_balance_sheet(ratios)
        profit_score   = self._score_profitability(ratios)
        dividend_score = self._score_dividend_capital(ratios)

        total_score = (
            earnings_score["score"] +
            balance_score["score"] +
            profit_score["score"] +
            dividend_score["score"]
        )

        grade, recommendation = self._get_grade(total_score)
        valuation_summary     = self._get_valuation_summary(ratios, raw_data)

        result = {
            "ticker":           self.ticker,
            "total_score":      total_score,
            "grade":            grade,
            "recommendation":   recommendation,
            "categories": {
                "earnings_quality":       earnings_score,
                "balance_sheet_strength": balance_score,
                "profitability_returns":  profit_score,
                "dividend_capital":       dividend_score,
            },
            "ratios":            ratios,
            "valuation_summary": valuation_summary,
            "raw_info":          raw_data.get("info", {}),
        }

        logger.info(f"✅ Fundamental Analysis complete — Score: {total_score}/100 | Grade: {grade}")
        return result
