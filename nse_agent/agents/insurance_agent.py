# ============================================================
# agents/insurance_agent.py — VERSION 1.0
# Specialized analysis for Life & General Insurance companies
#
# Why standard agents fail for insurance:
#   - Revenue = Premiums earned (not product sales)
#   - "Profit" includes unrealized investment gains
#   - D/E meaningless — insurers hold float (policyholder money)
#   - Key metrics: Claims ratio, Combined ratio, Embedded Value
#
# Correct valuation: P/EV (Price to Embedded Value)
# Embedded Value = Net Asset Value + Present Value of Future Profits
# ============================================================

import numpy as np
import pandas as pd
from loguru import logger
from utils.data_fetcher import NSEDataFetcher
from utils.ratio_calculator import RatioCalculator


class InsuranceAgent:
    """
    Specialized analysis agent for Life and General Insurance companies.
    Uses claims ratio, combined ratio, VNB, and P/EV valuation.
    """

    INDIA_RISK_FREE_RATE      = 6.8
    INDIA_EQUITY_RISK_PREMIUM = 5.5

    def __init__(self, ticker: str):
        self.ticker = ticker.upper().strip()
        logger.info(f"\n{'🛡️'*20}")
        logger.info(f"  INSURANCE ANALYSIS ENGINE: {self.ticker}")
        logger.info(f"{'🛡️'*20}")

    def _fetch_data(self) -> tuple:
        fetcher  = NSEDataFetcher(self.ticker)
        raw_data = fetcher.get_all_data()
        calc     = RatioCalculator(raw_data)
        ratios   = calc.calculate_all_ratios()
        return raw_data, ratios

    def _detect_insurance_type(self, info: dict) -> str:
        """Detect if life insurance or general insurance."""
        name     = (info.get("longName", "") or "").lower()
        industry = (info.get("industry", "") or "").lower()
        if "life" in name or "life" in industry:
            return "LIFE"
        elif "general" in name or "general" in industry or "non-life" in industry:
            return "GENERAL"
        return "LIFE"  # default

    def _get_insurance_ratios(self, raw_data: dict, ratios: dict) -> dict:
        """Extract insurance-specific metrics."""
        info     = raw_data.get("info", {})
        screener = raw_data.get("screener_data", {})
        ins      = {}

        ins["insurance_type"]  = self._detect_insurance_type(info)
        ins["roe"]             = ratios["profitability"]["roe"]
        ins["net_margin"]      = ratios["profitability"]["net_margin"]
        ins["revenue_cagr"]    = ratios["growth"]["revenue_cagr_3yr"]
        ins["eps_growth"]      = ratios["growth"]["earnings_growth_yoy"]
        ins["pe_ratio"]        = info.get("trailingPE", 0) or 0
        ins["pb_ratio"]        = info.get("priceToBook", 0) or 0
        ins["bv_ps"]           = info.get("bookValue", 0) or 0
        ins["dividend_yield"]  = ratios["dividend"]["dividend_yield"]
        ins["dividend_rate"]   = ratios["dividend"]["dividend_rate"]

        # Try to get insurance-specific ratios from screener
        claims_ratio   = 0.0
        combined_ratio = 0.0
        solvency_ratio = 0.0
        vnb_margin     = 0.0

        for key, val in screener.items():
            k = key.lower()
            if "claims ratio" in k or "loss ratio" in k:
                try: claims_ratio = float(str(val).replace("%","").strip())
                except: pass
            if "combined ratio" in k:
                try: combined_ratio = float(str(val).replace("%","").strip())
                except: pass
            if "solvency" in k:
                try: solvency_ratio = float(str(val).replace("%","").strip())
                except: pass
            if "vnb" in k and "margin" in k:
                try: vnb_margin = float(str(val).replace("%","").strip())
                except: pass

        ins["claims_ratio"]   = claims_ratio
        ins["combined_ratio"] = combined_ratio
        ins["solvency_ratio"] = solvency_ratio
        ins["vnb_margin"]     = vnb_margin

        return ins

    # ============================================================
    # SCORING (100 points)
    # ============================================================

    def _score_underwriting(self, ins: dict) -> dict:
        """Underwriting Quality (30 pts) — Core insurance business."""
        logger.info("🔢 Scoring Underwriting Quality...")
        score = 0; flags = []; positives = []

        claims  = ins["claims_ratio"]
        combined = ins["combined_ratio"]

        if claims > 0:
            # Claims ratio: lower = better (paid claims / premiums earned)
            if claims <= 65:
                score += 15
                positives.append(f"✅ Excellent claims ratio {claims:.1f}% — underwriting discipline")
            elif claims <= 75:
                score += 11
                positives.append(f"✅ Good claims ratio {claims:.1f}%")
            elif claims <= 85:
                score += 6
                positives.append(f"🟡 Adequate claims ratio {claims:.1f}%")
            else:
                flags.append(f"🔴 High claims ratio {claims:.1f}% — poor underwriting or adverse claims")

            if combined > 0:
                # Combined ratio: <100% = profitable underwriting
                if combined <= 95:
                    score += 15
                    positives.append(f"✅ Combined ratio {combined:.1f}% — profitable underwriting")
                elif combined <= 100:
                    score += 10
                    positives.append(f"✅ Combined ratio {combined:.1f}% — breakeven underwriting")
                elif combined <= 105:
                    score += 5
                    flags.append(f"⚠️ Combined ratio {combined:.1f}% > 100% — relying on investment income")
                else:
                    flags.append(f"🔴 Combined ratio {combined:.1f}% — underwriting losses")
        else:
            # No underwriting data — use net margin as proxy
            nm = ins["net_margin"]
            if nm >= 20:
                score += 25
                positives.append(f"✅ Strong net margin {nm:.1f}% suggests good underwriting")
            elif nm >= 12:
                score += 18
                positives.append(f"✅ Good net margin {nm:.1f}%")
            elif nm >= 5:
                score += 10
                positives.append(f"🟡 Moderate net margin {nm:.1f}%")
            else:
                flags.append(f"⚠️ Low net margin {nm:.1f}% — profitability concerns")
            positives.append("🟡 Claims/Combined ratio not available — check annual report")

        return {"score": min(score,30), "max": 30, "flags": flags, "positives": positives}

    def _score_growth_quality(self, ins: dict) -> dict:
        """Growth & Franchise (25 pts) — Premium growth + VNB."""
        logger.info("🔢 Scoring Growth Quality...")
        score = 0; flags = []; positives = []

        rev_growth = ins["revenue_cagr"]
        vnb        = ins["vnb_margin"]

        if rev_growth >= 20:
            score += 15
            positives.append(f"✅ Strong premium growth {rev_growth:.1f}% CAGR — expanding franchise")
        elif rev_growth >= 12:
            score += 11
            positives.append(f"✅ Good growth {rev_growth:.1f}% CAGR")
        elif rev_growth >= 6:
            score += 6
            positives.append(f"🟡 Moderate growth {rev_growth:.1f}% CAGR")
        else:
            flags.append(f"⚠️ Slow growth {rev_growth:.1f}% CAGR")

        # VNB Margin (life insurance only) — quality of new business
        if vnb > 0:
            if vnb >= 25:
                score += 10
                positives.append(f"✅ High VNB margin {vnb:.1f}% — high-quality new business mix")
            elif vnb >= 18:
                score += 7
                positives.append(f"✅ Good VNB margin {vnb:.1f}%")
            elif vnb >= 12:
                score += 4
                positives.append(f"🟡 Average VNB margin {vnb:.1f}%")
            else:
                flags.append(f"⚠️ Low VNB margin {vnb:.1f}% — poor new business quality")
        else:
            score += 5
            positives.append("🟡 VNB margin not available — check investor presentation")

        return {"score": min(score,25), "max": 25, "flags": flags, "positives": positives}

    def _score_financial_strength(self, ins: dict) -> dict:
        """Financial Strength (25 pts) — Solvency + ROE."""
        logger.info("🔢 Scoring Financial Strength...")
        score = 0; flags = []; positives = []

        solvency = ins["solvency_ratio"]
        roe      = ins["roe"]

        # Solvency ratio (IRDAI minimum: 150%)
        if solvency > 0:
            if solvency >= 250:
                score += 12
                positives.append(f"✅ Strong solvency {solvency:.0f}% (IRDAI min 150%) — very safe")
            elif solvency >= 180:
                score += 9
                positives.append(f"✅ Good solvency {solvency:.0f}%")
            elif solvency >= 150:
                score += 5
                flags.append(f"⚠️ Near minimum solvency {solvency:.0f}% — limited buffer")
            else:
                flags.append(f"🔴 Below regulatory solvency {solvency:.0f}% — serious risk")
        else:
            score += 6
            positives.append("🟡 Solvency ratio not available — check IRDAI filings")

        # ROE for insurance
        if roe >= 20:
            score += 13
            positives.append(f"✅ Exceptional ROE {roe:.1f}% — highly profitable insurer")
        elif roe >= 15:
            score += 10
            positives.append(f"✅ Good ROE {roe:.1f}%")
        elif roe >= 10:
            score += 6
            positives.append(f"🟡 Adequate ROE {roe:.1f}%")
        else:
            flags.append(f"⚠️ Low ROE {roe:.1f}%")

        return {"score": min(score,25), "max": 25, "flags": flags, "positives": positives}

    def _score_valuation(self, raw_data: dict, ins: dict) -> dict:
        """Valuation (20 pts) — P/EV, P/B, DDM."""
        logger.info("🔢 Scoring Insurance Valuation...")
        info  = raw_data.get("info", {})
        score = 0; flags = []; positives = []
        details = {}

        current_price = raw_data.get("current_price", 0)
        pb   = ins["pb_ratio"]
        roe  = ins["roe"]
        pe   = ins["pe_ratio"]

        details["pb_ratio"]      = pb
        details["pe_ratio"]      = pe
        details["current_price"] = current_price

        # For life insurers: P/EV is key metric
        # Typical fair P/EV: 2-4x for quality life insurers
        # Proxy: If P/B > 5x, likely pricing in EV premium
        if ins["insurance_type"] == "LIFE":
            if pb <= 2.5:
                score += 12
                positives.append(f"✅ Attractive P/B {pb:.1f}x for life insurer — possible discount to EV")
            elif pb <= 4.0:
                score += 8
                positives.append(f"✅ Reasonable P/B {pb:.1f}x — fairly valued")
            elif pb <= 6.0:
                score += 4
                flags.append(f"⚠️ Premium P/B {pb:.1f}x — pricing in high growth expectations")
            else:
                score += 1
                flags.append(f"🔴 Very high P/B {pb:.1f}x — expensive for life insurer")
        else:
            # General insurance: P/B 1.5-3x is fair
            if pb <= 1.5:
                score += 12
                positives.append(f"✅ Attractive P/B {pb:.1f}x for general insurer")
            elif pb <= 2.5:
                score += 8
                positives.append(f"✅ Reasonable P/B {pb:.1f}x")
            elif pb <= 4.0:
                score += 4
                flags.append(f"⚠️ Elevated P/B {pb:.1f}x")
            else:
                flags.append(f"🔴 Expensive P/B {pb:.1f}x")

        # PE ratio check
        if 0 < pe <= 25:
            score += 5
            positives.append(f"✅ Reasonable P/E {pe:.1f}x")
        elif 25 < pe <= 40:
            score += 3
            positives.append(f"🟡 Growth P/E {pe:.1f}x — acceptable for quality insurer")
        elif pe > 40:
            flags.append(f"⚠️ High P/E {pe:.1f}x — growth priced in")

        # DDM for dividend paying insurers
        div_rate = ins["dividend_rate"]
        if div_rate > 0:
            score += 3
            positives.append(f"✅ Dividend paying: {ins['dividend_yield']:.1f}% yield")

        # Estimate fair value (P/B based)
        bv_ps  = ins["bv_ps"]
        fair_pb = 3.0 if ins["insurance_type"] == "LIFE" else 2.0
        fair_value = fair_pb * bv_ps if bv_ps > 0 else 0
        details["fair_value"] = round(fair_value, 2)
        details["fair_pb"]    = fair_pb

        return {
            "score": min(score,20), "max": 20,
            "flags": flags, "positives": positives, "details": details
        }

    def _buffett_insurance_checklist(self, ins: dict) -> list:
        checks = []
        checks.append({
            "question": "Is underwriting profitable (Combined ratio < 100%)?",
            "pass":     0 < ins["combined_ratio"] < 100 or ins["combined_ratio"] == 0,
            "detail":   f"Combined = {ins['combined_ratio']:.1f}% {'✅' if ins['combined_ratio'] < 100 else '❌'}"
                        if ins["combined_ratio"] > 0 else "Combined ratio N/A"
        })
        checks.append({
            "question": "Is ROE strong (>15%)?",
            "pass":     ins["roe"] >= 15,
            "detail":   f"ROE = {ins['roe']:.1f}% {'✅' if ins['roe'] >= 15 else '❌'}"
        })
        checks.append({
            "question": "Is premium growing at >12% CAGR?",
            "pass":     ins["revenue_cagr"] >= 12,
            "detail":   f"Revenue CAGR = {ins['revenue_cagr']:.1f}% {'✅' if ins['revenue_cagr'] >= 12 else '❌'}"
        })
        checks.append({
            "question": "Is VNB margin healthy (>18% for life)?",
            "pass":     ins["vnb_margin"] >= 18 or ins["vnb_margin"] == 0,
            "detail":   f"VNB margin = {ins['vnb_margin']:.1f}% {'✅' if ins['vnb_margin'] >= 18 else '❌'}"
                        if ins["vnb_margin"] > 0 else "VNB margin N/A"
        })
        checks.append({
            "question": "Is solvency ratio above 200%?",
            "pass":     ins["solvency_ratio"] >= 200 or ins["solvency_ratio"] == 0,
            "detail":   f"Solvency = {ins['solvency_ratio']:.0f}% {'✅' if ins['solvency_ratio'] >= 200 else '❌'}"
                        if ins["solvency_ratio"] > 0 else "Solvency N/A"
        })
        checks.append({
            "question": "Is P/B reasonable (< 4x for life, < 2.5x for general)?",
            "pass":     ins["pb_ratio"] <= 4.0,
            "detail":   f"P/B = {ins['pb_ratio']:.1f}x {'✅' if ins['pb_ratio'] <= 4.0 else '❌'}"
        })
        return checks

    def _get_grade(self, score: int) -> tuple:
        if score >= 85:   return "A+", "🟢 EXCEPTIONAL INSURER"
        elif score >= 75: return "A",  "🟢 EXCELLENT INSURER"
        elif score >= 65: return "B+", "🟢 GOOD INSURER"
        elif score >= 55: return "B",  "🟡 AVERAGE INSURER"
        elif score >= 45: return "C",  "🟡 BELOW AVERAGE"
        elif score >= 35: return "D",  "🟠 WEAK INSURER"
        else:             return "F",  "🔴 AVOID"

    def analyze(self) -> dict:
        raw_data, ratios = self._fetch_data()
        info             = raw_data.get("info", {})
        current_price    = raw_data.get("current_price", 0)

        ins = self._get_insurance_ratios(raw_data, ratios)

        uw_score     = self._score_underwriting(ins)
        growth_score = self._score_growth_quality(ins)
        fin_score    = self._score_financial_strength(ins)
        val_score    = self._score_valuation(raw_data, ins)

        total_score = (
            uw_score["score"] + growth_score["score"] +
            fin_score["score"] + val_score["score"]
        )

        grade, verdict = self._get_grade(total_score)
        checks         = self._buffett_insurance_checklist(ins)
        buffett_pass   = sum(1 for c in checks if c["pass"])

        all_flags = []; all_positives = []
        for s in [uw_score, growth_score, fin_score, val_score]:
            all_flags.extend(s["flags"])
            all_positives.extend(s["positives"])

        vd        = val_score.get("details", {})
        iv        = vd.get("fair_value", 0)
        upside    = ((iv - current_price) / current_price * 100) if current_price > 0 and iv > 0 else 0

        result = {
            "ticker":          self.ticker,
            "company_name":    info.get("longName", self.ticker),
            "sector":          "Insurance",
            "industry":        ins["insurance_type"] + " Insurance",
            "total_score":     total_score,
            "grade":           grade,
            "verdict":         verdict,
            "scores": {
                "underwriting": uw_score,
                "growth":       growth_score,
                "financial":    fin_score,
                "valuation":    val_score,
            },
            "insurance_ratios": ins,
            "buffett_checks":   checks,
            "buffett_pass":     buffett_pass,
            "all_flags":        all_flags,
            "all_positives":    all_positives,
            "current_price":    current_price,
            "intrinsic_value":  round(iv, 2),
            "upside_pct":       round(upside, 1),
            "valuation_score":  min(95, max(10, int(50 + upside))),
        }

        logger.info(
            f"✅ Insurance Analysis: {total_score}/100 Grade {grade} — {verdict} | "
            f"Fair Value ₹{iv:,.0f}"
        )
        return result
