# ============================================================
# agents/management_agent.py — VERSION 1.0
# Management Quality Scoring Agent
#
# Buffett's Rule: "When a management with a reputation for
# brilliance tackles a business with a reputation for bad
# economics, it is the business's reputation that remains intact."
#
# Scores management across 4 dimensions (100 points total):
#   1. Promoter Commitment  (30 pts) — skin in the game
#   2. Capital Allocation   (25 pts) — how they deploy profits
#   3. Earnings Integrity   (25 pts) — is PAT backed by cash?
#   4. Governance Quality   (20 pts) — board, auditors, RPTs
# ============================================================

import numpy as np
import pandas as pd
from loguru import logger
from utils.data_fetcher import NSEDataFetcher
from utils.ratio_calculator import RatioCalculator


class ManagementQualityAgent:
    """
    Evaluates management quality using publicly available data.
    Produces a 0-100 score with detailed red flags and green flags.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.upper().strip()
        logger.info(f"\n{'👔'*20}")
        logger.info(f"  MANAGEMENT QUALITY ENGINE: {self.ticker}")
        logger.info(f"{'👔'*20}")

    def _fetch_data(self) -> tuple:
        fetcher  = NSEDataFetcher(self.ticker)
        raw_data = fetcher.get_all_data()
        calc     = RatioCalculator(raw_data)
        ratios   = calc.calculate_all_ratios()
        return raw_data, ratios

    # ============================================================
    # DIMENSION 1: PROMOTER COMMITMENT (30 points)
    # Key question: Do insiders have skin in the game?
    # Red flags: Low holding, high pledge, consistent selling
    # ============================================================

    def _score_promoter_commitment(self, raw_data: dict, ratios: dict) -> dict:
        logger.info("🔢 Scoring Promoter Commitment...")
        info         = raw_data.get("info", {})
        score        = 0
        flags        = []
        positives    = []
        details      = {}

        # ---- Promoter / Insider Holding % ----
        # Yahoo: insiderOwnership = insider %, institutionOwnership = FII+MF %
        insider_pct       = (info.get("heldPercentInsiders", 0) or 0) * 100
        institution_pct   = (info.get("heldPercentInstitutions", 0) or 0) * 100
        public_float_pct  = max(0, 100 - insider_pct - institution_pct)

        details["promoter_holding_pct"]     = round(insider_pct, 2)
        details["institutional_holding_pct"] = round(institution_pct, 2)
        details["public_float_pct"]          = round(public_float_pct, 2)

        # Score: Higher promoter holding = more skin in game
        if insider_pct >= 60:
            score += 18
            positives.append(f"✅ High promoter holding {insider_pct:.1f}% — strong alignment with shareholders")
        elif insider_pct >= 45:
            score += 14
            positives.append(f"✅ Good promoter holding {insider_pct:.1f}%")
        elif insider_pct >= 30:
            score += 10
            positives.append(f"🟡 Moderate promoter holding {insider_pct:.1f}%")
        elif insider_pct >= 15:
            score += 5
            flags.append(f"⚠️ Low promoter holding {insider_pct:.1f}% — limited skin in game")
        else:
            score += 0
            flags.append(f"🔴 Very low promoter holding {insider_pct:.1f}% — management may not be owner-operators")

        # ---- Institutional Confidence ----
        # High institutional holding = smart money believes in management
        if institution_pct >= 40:
            score += 7
            positives.append(f"✅ High institutional confidence {institution_pct:.1f}% (FII+MF)")
        elif institution_pct >= 25:
            score += 5
            positives.append(f"🟡 Decent institutional holding {institution_pct:.1f}%")
        elif institution_pct >= 10:
            score += 3
        else:
            flags.append(f"⚠️ Low institutional interest {institution_pct:.1f}%")

        # ---- Promoter Pledge % (from Screener.in if available) ----
        screener = raw_data.get("screener_data", {})
        pledge_pct = 0.0

        # Try to extract pledge from screener data
        for key, val in screener.items():
            if "pledge" in key.lower():
                try:
                    pledge_pct = float(str(val).replace("%", "").strip())
                except:
                    pass

        details["promoter_pledge_pct"] = pledge_pct

        if pledge_pct == 0:
            score += 5
            positives.append("✅ No promoter pledge detected (or data unavailable)")
        elif pledge_pct <= 5:
            score += 4
            positives.append(f"✅ Low promoter pledge {pledge_pct:.1f}%")
        elif pledge_pct <= 15:
            score += 2
            flags.append(f"⚠️ Moderate promoter pledge {pledge_pct:.1f}% — watch for margin calls")
        elif pledge_pct <= 30:
            score += 0
            flags.append(f"🔴 High promoter pledge {pledge_pct:.1f}% — significant financial stress risk")
        else:
            score += 0
            flags.append(f"🚨 CRITICAL: Promoter pledge {pledge_pct:.1f}% — extreme risk of forced selling")

        logger.info(f"  Promoter score: {score}/30")
        return {
            "score": min(score, 30),
            "max":   30,
            "flags": flags,
            "positives": positives,
            "details": details,
        }

    # ============================================================
    # DIMENSION 2: CAPITAL ALLOCATION (25 points)
    # Key question: Do they deploy capital wisely?
    # Good: High ROCE, buybacks at reasonable prices, growing dividends
    # Bad: Dilutive equity issuance, acquisitions at high premiums
    # ============================================================

    def _score_capital_allocation(self, raw_data: dict, ratios: dict) -> dict:
        logger.info("🔢 Scoring Capital Allocation...")
        info      = raw_data.get("info", {})
        cf        = raw_data.get("cash_flow", pd.DataFrame())
        score     = 0
        flags     = []
        positives = []
        details   = {}

        roce      = ratios["profitability"]["roce"]
        roe       = ratios["profitability"]["roe"]
        net_margin = ratios["profitability"]["net_margin"]

        details["roce"] = roce
        details["roe"]  = roe

        # ---- ROCE: Best indicator of capital allocation quality ----
        # For financial companies ROCE is not meaningful — banks use depositor
        # money as "capital" which inflates denominator, making ROCE always low
        sector = (info.get("sector", "") or "").lower()
        is_financial = "financial" in sector or "insurance" in sector
        if is_financial:
            # Use ROE instead of ROCE for financial companies
            roe_val = ratios["profitability"]["roe"]
            if roe_val >= 18:
                score += 10
                positives.append(f"✅ Excellent ROE {roe_val:.1f}% — strong returns (ROCE N/A for financials)")
            elif roe_val >= 14:
                score += 7
                positives.append(f"✅ Good ROE {roe_val:.1f}% (ROCE N/A for financial sector)")
            elif roe_val >= 10:
                score += 4
                positives.append(f"🟡 Adequate ROE {roe_val:.1f}% (ROCE N/A for financial sector)")
            elif roe_val > 0:
                score += 2
                positives.append(f"🟡 Low ROE {roe_val:.1f}% — monitor closely")
            else:
                score += 6
                positives.append("🟡 ROCE not applicable for financial sector — use ROA/ROE instead")
        elif roce >= 30:
            score += 12
            positives.append(f"✅ Exceptional ROCE {roce:.1f}% — management creates strong returns on every rupee")
        elif roce >= 20:
            score += 9
            positives.append(f"✅ Good ROCE {roce:.1f}% — above cost of capital")
        elif roce >= 15:
            score += 6
            positives.append(f"🟡 Adequate ROCE {roce:.1f}%")
        elif roce >= 10:
            score += 3
            flags.append(f"⚠️ Below-average ROCE {roce:.1f}% — mediocre capital deployment")
        else:
            score += 0
            flags.append(f"🔴 Poor ROCE {roce:.1f}% — destroying shareholder value")

        # ---- Share Buybacks / Dilution ----
        # Check cashflow for share repurchases (positive = buyback happening)
        buyback_amt = 0
        issuance_amt = 0
        if not cf.empty:
            for row_name in cf.index:
                if "repurchase" in str(row_name).lower() or "buyback" in str(row_name).lower():
                    val = cf.loc[row_name].iloc[0]
                    if pd.notna(val):
                        buyback_amt = abs(val)
                if "issuance" in str(row_name).lower() and "stock" in str(row_name).lower():
                    val = cf.loc[row_name].iloc[0]
                    if pd.notna(val) and val > 0:
                        issuance_amt = val

        details["buyback_amt"]   = buyback_amt
        details["issuance_amt"]  = issuance_amt

        if buyback_amt > 0:
            score += 5
            positives.append(f"✅ Share buyback detected — management returning cash to shareholders")
        elif issuance_amt > 0:
            mkt_cap = info.get("marketCap", 1) or 1
            dilution_pct = (issuance_amt / mkt_cap) * 100
            if dilution_pct > 5:
                flags.append(f"⚠️ Significant equity dilution {dilution_pct:.1f}% of market cap")
            else:
                score += 2
                positives.append("🟡 Minor equity issuance (likely ESOPs)")
        else:
            score += 3
            positives.append("🟡 No major buyback or dilution detected")

        # ---- Dividend Track Record ----
        div_data     = ratios["dividend"]
        div_yield    = div_data["dividend_yield"]
        payout_ratio = div_data["payout_ratio"]

        details["dividend_yield"]  = div_yield
        details["payout_ratio"]    = payout_ratio

        if div_yield > 0 and 20 <= payout_ratio <= 70:
            score += 5
            positives.append(f"✅ Sustainable dividend: {div_yield:.1f}% yield, {payout_ratio:.0f}% payout — disciplined cash return")
        elif div_yield > 0:
            score += 3
            positives.append(f"🟡 Dividend paid: {div_yield:.1f}% yield")
        else:
            score += 2
            positives.append("🟡 No dividend — reinvesting for growth (acceptable for high-growth companies)")

        # ---- Acquisition Discipline (proxy: asset growth vs profit growth) ----
        income = raw_data.get("income_statement", pd.DataFrame())
        if not income.empty and income.shape[1] >= 3:
            try:
                rev_cols = [c for c in income.columns if "Total Revenue" in str(income.index)]
                if "Total Revenue" in income.index:
                    revenues = income.loc["Total Revenue"].dropna()
                    if len(revenues) >= 3:
                        rev_growth = ((revenues.iloc[0] / revenues.iloc[-1]) ** (1/3) - 1) * 100
                        details["revenue_cagr_3yr"] = round(rev_growth, 1)
                        if rev_growth > 5:
                            score += 3
                            positives.append(f"✅ Revenue growing at {rev_growth:.1f}% CAGR")
            except:
                pass

        logger.info(f"  Capital allocation score: {score}/25")
        return {
            "score":     min(score, 25),
            "max":       25,
            "flags":     flags,
            "positives": positives,
            "details":   details,
        }

    # ============================================================
    # DIMENSION 3: EARNINGS INTEGRITY (25 points)
    # Key question: Is reported profit real?
    # Good: CFO > PAT, consistent margins, clean audit
    # Bad: CFO << PAT, frequent restatements, aggressive accounting
    # ============================================================

    def _score_earnings_integrity(self, raw_data: dict, ratios: dict) -> dict:
        logger.info("🔢 Scoring Earnings Integrity...")
        info      = raw_data.get("info", {})
        score     = 0
        flags     = []
        positives = []
        details   = {}

        # ---- CFO / PAT ratio (Cash Flow Quality) ----
        # CFO > PAT = earnings are real cash, not accounting tricks
        # EXCEPTION: Banks/NBFCs always show negative CFO (loan disbursements = operating outflow)
        cfo_to_pat = ratios["cashflow"]["cfo_to_net_income"]
        details["cfo_to_net_income"] = cfo_to_pat

        sector = (info.get("sector", "") or "").lower()
        is_financial = "financial" in sector or "insurance" in sector

        if is_financial and cfo_to_pat < 0:
            # Negative CFO is STRUCTURALLY NORMAL for banks/NBFCs
            # Loan disbursements flow through operating cash flow
            # Full points — this is not an earnings quality issue
            score += 15
            positives.append(f"🟡 CFO/PAT = {cfo_to_pat:.2f}x — negative CFO is normal for financial lenders (loan disbursements flow through ops)")
        elif cfo_to_pat >= 1.2:
            score += 15
            positives.append(f"✅ Excellent cash conversion: CFO/PAT = {cfo_to_pat:.2f}x — profits backed by real cash")
        elif cfo_to_pat >= 1.0:
            score += 12
            positives.append(f"✅ Good cash conversion: CFO/PAT = {cfo_to_pat:.2f}x")
        elif cfo_to_pat >= 0.8:
            score += 8
            positives.append(f"🟡 Acceptable cash conversion: CFO/PAT = {cfo_to_pat:.2f}x")
        elif cfo_to_pat >= 0.5:
            score += 4
            flags.append(f"⚠️ Weak cash conversion: CFO/PAT = {cfo_to_pat:.2f}x — some earnings may not be cash")
        else:
            score += 0
            flags.append(f"🔴 Poor cash conversion: CFO/PAT = {cfo_to_pat:.2f}x — possible aggressive revenue recognition")

        # ---- Net Margin Consistency ----
        net_margin    = ratios["profitability"]["net_margin"]
        ebitda_margin = ratios["profitability"]["ebitda_margin"]
        details["net_margin"]    = net_margin
        details["ebitda_margin"] = ebitda_margin

        # Check multi-year margin stability from income statement
        income = raw_data.get("income_statement", pd.DataFrame())
        margin_stable = False
        if not income.empty and income.shape[1] >= 3:
            try:
                if "Net Income" in income.index and "Total Revenue" in income.index:
                    net_incomes = income.loc["Net Income"].dropna()
                    revenues    = income.loc["Total Revenue"].dropna()
                    margins     = (net_incomes / revenues * 100).dropna()
                    if len(margins) >= 3:
                        margin_std = margins.std()
                        details["margin_std"] = round(float(margin_std), 2)
                        if margin_std < 3:
                            score += 5
                            margin_stable = True
                            positives.append(f"✅ Consistent net margins (std dev {margin_std:.1f}%) — predictable business")
                        elif margin_std < 6:
                            score += 3
                            positives.append(f"🟡 Moderately stable margins (std dev {margin_std:.1f}%)")
                        else:
                            flags.append(f"⚠️ Volatile margins (std dev {margin_std:.1f}%) — earnings unpredictable")
            except:
                score += 2

        if not margin_stable and "margin_std" not in details:
            if net_margin >= 15:
                score += 4
                positives.append(f"✅ Strong net margin {net_margin:.1f}%")
            elif net_margin >= 8:
                score += 2
                positives.append(f"🟡 Adequate net margin {net_margin:.1f}%")
            else:
                flags.append(f"⚠️ Thin net margin {net_margin:.1f}%")

        # ---- Deferred Tax Trend (accounting quality signal) ----
        # Large positive deferred tax = accelerating revenue recognition
        cf = raw_data.get("cash_flow", pd.DataFrame())
        if not cf.empty:
            for row_name in cf.index:
                if "deferred" in str(row_name).lower() and "tax" in str(row_name).lower():
                    try:
                        deferred_tax = cf.loc[row_name].iloc[0]
                        if pd.notna(deferred_tax):
                            details["deferred_tax"] = deferred_tax
                            rev = info.get("totalRevenue", 1) or 1
                            # Flag if deferred tax > 5% of revenue
                            if abs(deferred_tax) / rev > 0.05:
                                flags.append("⚠️ Large deferred tax movement — review accounting policies")
                            else:
                                score += 5
                                positives.append("✅ Normal deferred tax levels — no accounting red flags")
                    except:
                        pass
                    break
            else:
                score += 3

        logger.info(f"  Earnings integrity score: {score}/25")
        return {
            "score":     min(score, 25),
            "max":       25,
            "flags":     flags,
            "positives": positives,
            "details":   details,
        }

    # ============================================================
    # DIMENSION 4: GOVERNANCE QUALITY (20 points)
    # Key question: Are shareholders protected?
    # Good: Independent board, reputable auditor, low RPTs
    # Bad: Family-dominated board, auditor changes, high RPTs
    # ============================================================

    def _score_governance(self, raw_data: dict, ratios: dict) -> dict:
        logger.info("🔢 Scoring Governance Quality...")
        info      = raw_data.get("info", {})
        score     = 0
        flags     = []
        positives = []
        details   = {}

        # ---- Company Age & Track Record ----
        # Older companies with consistent history = more trustworthy
        # Proxy: years of data available
        income = raw_data.get("income_statement", pd.DataFrame())
        years_of_data = income.shape[1] if not income.empty else 0
        details["years_of_data"] = years_of_data

        if years_of_data >= 5:
            score += 5
            positives.append(f"✅ {years_of_data}+ years of financial data — long track record")
        elif years_of_data >= 3:
            score += 3
            positives.append(f"🟡 {years_of_data} years of financial data")
        else:
            flags.append("⚠️ Limited financial history — insufficient track record")

        # ---- Debt Management Philosophy ----
        # Conservative debt = management doesn't over-lever for growth
        de_ratio    = ratios["leverage"]["debt_to_equity"]
        int_coverage = ratios["leverage"]["interest_coverage"]
        details["de_ratio"]     = de_ratio
        details["int_coverage"] = int_coverage

        # D/E — for financial companies high leverage is normal and expected
        sector = (info.get("sector", "") or "").lower()
        is_financial = "financial" in sector or "insurance" in sector
        if is_financial and de_ratio >= 1.0:
            score += 5
            positives.append(f"🟡 D/E {de_ratio:.2f}x — financial sector leverage is expected (deposits/borrowings fund loans)")
        elif de_ratio <= 0.1:
            score += 6
            positives.append(f"✅ Near debt-free (D/E {de_ratio:.2f}x) — fortress balance sheet")
        elif de_ratio <= 0.5:
            score += 5
            positives.append(f"✅ Conservative debt (D/E {de_ratio:.2f}x)")
        elif de_ratio <= 1.0:
            score += 3
            positives.append(f"🟡 Moderate debt (D/E {de_ratio:.2f}x)")
        elif de_ratio <= 2.0:
            score += 1
            flags.append(f"⚠️ Elevated debt (D/E {de_ratio:.2f}x) — monitor closely")
        else:
            score += 0
            flags.append(f"🔴 High debt (D/E {de_ratio:.2f}x) — financial risk")

        # ---- Interest Coverage ----
        if int_coverage >= 10:
            score += 4
            positives.append(f"✅ Strong interest coverage {int_coverage:.1f}x — debt very manageable")
        elif int_coverage >= 5:
            score += 3
            positives.append(f"✅ Adequate interest coverage {int_coverage:.1f}x")
        elif int_coverage >= 2:
            score += 1
            flags.append(f"⚠️ Tight interest coverage {int_coverage:.1f}x")
        elif int_coverage > 0:
            flags.append(f"🔴 Dangerously low interest coverage {int_coverage:.1f}x")
        else:
            positives.append("🟡 Interest coverage N/A (likely debt-free company)")
            score += 3

        # ---- Business Description Quality (transparency signal) ----
        description = info.get("longBusinessSummary", "")
        if len(description) > 200:
            score += 3
            positives.append("✅ Detailed business disclosure — management communicates clearly")
        elif len(description) > 50:
            score += 1
        else:
            flags.append("⚠️ Minimal business description — limited transparency")

        # ---- Sector-Specific Governance Flags ----
        sector = info.get("sector", "").lower()
        if sector in ("financial services", "banking"):
            # Financial companies: NPA, CAR are governance metrics
            flags.append("ℹ️  Financial sector: verify NPA ratios and Capital Adequacy Ratio separately")

        logger.info(f"  Governance score: {score}/20")
        return {
            "score":     min(score, 20),
            "max":       20,
            "flags":     flags,
            "positives": positives,
            "details":   details,
        }

    # ============================================================
    # COMPOSITE SCORE + GRADE + BUFFETT VERDICT
    # ============================================================

    def _get_grade(self, score: int) -> tuple:
        if score >= 85:   return "A+", "🟢 EXCEPTIONAL", "Rare management quality — Buffett would love this"
        elif score >= 75: return "A",  "🟢 EXCELLENT",   "High-quality management — strong long-term bet"
        elif score >= 65: return "B+", "🟢 GOOD",        "Above-average management — worth owning"
        elif score >= 55: return "B",  "🟡 AVERAGE",     "Adequate management — monitor for improvement"
        elif score >= 45: return "C",  "🟡 BELOW AVG",   "Mediocre management — requires margin of safety"
        elif score >= 35: return "D",  "🟠 WEAK",        "Poor management — avoid unless very cheap"
        else:             return "F",  "🔴 POOR",        "Red flags present — high risk of value destruction"

    def _buffett_checklist(self, raw_data: dict, ratios: dict, scores: dict) -> list:
        """6 Buffett management questions — binary pass/fail"""
        info   = raw_data.get("info", {})
        checks = []

        sector      = (info.get("sector", "") or "").lower()
        is_financial = "financial" in sector or "insurance" in sector

        roce = ratios["profitability"]["roce"]
        roe  = ratios["profitability"]["roe"]
        if is_financial:
            checks.append({
                "question": "Does management earn high returns on capital?",
                "pass":     roe >= 15,
                "detail":   f"ROE = {roe:.1f}% (ROCE N/A for financials) {'✅' if roe >= 15 else '❌'}"
            })
        else:
            checks.append({
                "question": "Does management earn high returns on capital?",
                "pass":     roce >= 20,
                "detail":   f"ROCE = {roce:.1f}% {'✅' if roce >= 20 else '❌'}"
            })

        cfo_to_pat = ratios["cashflow"]["cfo_to_net_income"]
        if is_financial and cfo_to_pat < 0:
            checks.append({
                "question": "Are earnings backed by real cash flows?",
                "pass":     True,
                "detail":   f"CFO/PAT = {cfo_to_pat:.2f}x 🟡 (normal for financial lenders)"
            })
        else:
            checks.append({
                "question": "Are earnings backed by real cash flows?",
                "pass":     cfo_to_pat >= 0.9,
                "detail":   f"CFO/PAT = {cfo_to_pat:.2f}x {'✅' if cfo_to_pat >= 0.9 else '❌'}"
            })

        de = ratios["leverage"]["debt_to_equity"]
        if is_financial and de >= 1.0:
            checks.append({
                "question": "Is debt used conservatively?",
                "pass":     True,
                "detail":   f"D/E = {de:.2f}x 🟡 (financial leverage — normal for lenders)"
            })
        else:
            checks.append({
                "question": "Is debt used conservatively?",
                "pass":     de <= 1.0,
                "detail":   f"D/E = {de:.2f}x {'✅' if de <= 1.0 else '❌'}"
            })

        # Use Screener promoter holding (more accurate than Yahoo heldPercentInsiders)
        screener_data = raw_data.get("screener_data", {})
        insider_pct = screener_data.get("promoter_holding", 0)
        if insider_pct == 0:
            insider_pct = (info.get("heldPercentInsiders", 0) or 0) * 100
        checks.append({
            "question": "Do insiders have significant skin in the game?",
            "pass":     insider_pct >= 20,
            "detail":   f"Insider holding = {insider_pct:.1f}% {'✅' if insider_pct >= 20 else '❌'}"
        })

        div_yield = ratios["dividend"]["dividend_yield"]
        checks.append({
            "question": "Does management return cash to shareholders?",
            "pass":     div_yield > 0 or scores["capital"]["details"].get("buyback_amt", 0) > 0,
            "detail":   f"Dividend yield = {div_yield:.1f}% {'✅' if div_yield > 0 else '❌'}"
        })

        net_margin = ratios["profitability"]["net_margin"]
        checks.append({
            "question": "Are profit margins strong and sustainable?",
            "pass":     net_margin >= 10,
            "detail":   f"Net margin = {net_margin:.1f}% {'✅' if net_margin >= 10 else '❌'}"
        })

        return checks

    # ============================================================
    # MAIN ANALYZE METHOD
    # ============================================================

    def analyze(self) -> dict:
        raw_data, ratios = self._fetch_data()
        info             = raw_data.get("info", {})

        # Score all 4 dimensions
        promoter_score = self._score_promoter_commitment(raw_data, ratios)
        capital_score  = self._score_capital_allocation(raw_data, ratios)
        integrity_score = self._score_earnings_integrity(raw_data, ratios)
        governance_score = self._score_governance(raw_data, ratios)

        scores = {
            "promoter":   promoter_score,
            "capital":    capital_score,
            "integrity":  integrity_score,
            "governance": governance_score,
        }

        total_score = (
            promoter_score["score"] +
            capital_score["score"] +
            integrity_score["score"] +
            governance_score["score"]
        )

        grade, verdict, summary = self._get_grade(total_score)
        buffett_checks = self._buffett_checklist(raw_data, ratios, scores)
        buffett_pass   = sum(1 for c in buffett_checks if c["pass"])

        # Aggregate all flags and positives
        all_flags     = []
        all_positives = []
        for dim in scores.values():
            all_flags.extend(dim["flags"])
            all_positives.extend(dim["positives"])

        result = {
            "ticker":          self.ticker,
            "company_name":    info.get("longName", self.ticker),
            "sector":          info.get("sector", "N/A"),
            "total_score":     total_score,
            "grade":           grade,
            "verdict":         verdict,
            "summary":         summary,
            "scores":          scores,
            "buffett_checks":  buffett_checks,
            "buffett_pass":    buffett_pass,
            "all_flags":       all_flags,
            "all_positives":   all_positives,
            "ratios":          ratios,
        }

        logger.info(f"✅ Management Quality: {total_score}/100 Grade {grade} — {verdict}")
        return result
