"""
╔══════════════════════════════════════════════════════════╗
║     PHARMA ANALYSIS AGENT — NSE Specialized              ║
║     Patent Pipeline | R&D Intensity | FDA Alerts          ║
║     USFDA Warning Letters | ANDA Filings | Compliance     ║
╚══════════════════════════════════════════════════════════╝

Why standard agents fail for Pharma:
  - Revenue lumpy due to patent expirations
  - R&D is a cost today but an asset tomorrow
  - USFDA observations (483s, Warning Letters) are tail risks
  - Specialty vs generic vs CDMO business models differ hugely
  - US generics pricing erosion not captured in DCF
"""

import re
import time
import requests
import warnings
from datetime import datetime
from loguru import logger

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# ─────────────────────────────────────────────────────────
# PHARMA PEER BENCHMARKS (NSE-listed, India context)
# ─────────────────────────────────────────────────────────
PHARMA_BENCHMARKS = {
    "rd_to_revenue_pct": {"excellent": 10, "good": 7, "average": 4, "poor": 2},
    "ebitda_margin":     {"excellent": 28, "good": 22, "average": 16, "poor": 10},
    "roe":               {"excellent": 22, "good": 18, "average": 12, "poor": 8},
    "us_revenue_mix_pct":{"excellent": 40, "good": 25, "average": 15, "poor": 5},
    "anda_filings":      {"excellent": 50, "good": 25, "average": 10, "poor": 3},
}

# Known USFDA warning letter history (hardcoded for major Indian pharma)
# In production: scrape https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/compliance-actions-and-activities/warning-letters
FDA_RISK_REGISTRY = {
    "SUNPHARMA":  {"warning_letters": 2, "483_observations": 5, "import_alerts": 1, "last_issue": "2023"},
    "DRREDDY":    {"warning_letters": 1, "483_observations": 3, "import_alerts": 0, "last_issue": "2022"},
    "CIPLA":      {"warning_letters": 0, "483_observations": 2, "import_alerts": 0, "last_issue": "2021"},
    "DIVISLAB":   {"warning_letters": 1, "483_observations": 4, "import_alerts": 0, "last_issue": "2022"},
    "LUPIN":      {"warning_letters": 2, "483_observations": 6, "import_alerts": 1, "last_issue": "2023"},
    "AUROPHARMA": {"warning_letters": 3, "483_observations": 8, "import_alerts": 2, "last_issue": "2023"},
    "GLENMARK":   {"warning_letters": 1, "483_observations": 3, "import_alerts": 0, "last_issue": "2022"},
    "TORNTPHARM": {"warning_letters": 0, "483_observations": 1, "import_alerts": 0, "last_issue": "N/A"},
    "APOLLOHOSP": {"warning_letters": 0, "483_observations": 0, "import_alerts": 0, "last_issue": "N/A"},
}

# Business model classification
PHARMA_BUSINESS_MODEL = {
    "SUNPHARMA":  "Specialty + Branded Generic",
    "DRREDDY":    "US Generic + Biosimilar",
    "CIPLA":      "Branded Generic + Inhaler",
    "DIVISLAB":   "CDMO + API",
    "APOLLOHOSP": "Hospital Network",
    "LUPIN":      "US Generic + India Branded",
    "AUROPHARMA": "US Generic Heavy",
    "TORNTPHARM": "India Branded + Europe",
    "GLENMARK":   "Specialty + Branded Generic",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


class PharmaAgent:
    """
    Specialized analysis engine for NSE-listed Pharmaceutical companies.

    Scoring (100 points):
      - Business Quality & Model     (20 pts)
      - R&D Intensity & Pipeline     (25 pts)
      - Regulatory Risk (FDA)        (20 pts)
      - Financial Performance        (20 pts)
      - Valuation                    (15 pts)
    """

    INDIA_RISK_FREE   = 6.8
    INDIA_ERP         = 5.5
    PHARMA_WACC_MIN   = 10.0
    PHARMA_WACC_MAX   = 13.0

    def __init__(self, ticker: str):
        self.ticker = ticker.upper().replace(".NS", "")
        logger.info(f"💊 PharmaAgent initialized for {self.ticker}")

    # ─────────────────────────────────────────────────────
    # DATA FETCHING
    # ─────────────────────────────────────────────────────
    def _fetch_data(self) -> dict:
        raw = {"info": {}, "income": None, "balance": None, "cashflow": None, "screener": {}}
        if not HAS_YFINANCE:
            return raw
        try:
            ticker_obj  = yf.Ticker(f"{self.ticker}.NS")
            raw["info"] = ticker_obj.info or {}
            try: raw["income"]  = ticker_obj.financials
            except Exception: pass
            try: raw["balance"] = ticker_obj.balance_sheet
            except Exception: pass
            try: raw["cashflow"]= ticker_obj.cashflow
            except Exception: pass
            raw["screener"] = self._fetch_screener()
        except Exception as e:
            logger.warning(f"Data fetch error: {e}")
        return raw

    def _fetch_screener(self) -> dict:
        """Scrape Screener.in for R&D and key ratios."""
        result = {}
        try:
            url  = f"https://www.screener.in/company/{self.ticker}/consolidated/"
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")

            # Parse top-ratios section
            ratios_div = soup.find(id="top-ratios")
            if ratios_div:
                for li in ratios_div.find_all("li"):
                    name_tag  = li.find("span", class_="name")
                    value_tag = li.find("span", class_="nowrap")
                    if name_tag and value_tag:
                        key = name_tag.get_text(strip=True).lower().replace(" ", "_")
                        val = value_tag.get_text(strip=True).replace(",", "").replace("%", "").replace("₹", "").strip()
                        try:
                            result[key] = float(val)
                        except ValueError:
                            result[key] = val

            # Try to find R&D from annual report section
            for row in soup.find_all("tr"):
                cells = row.find_all("td")
                if cells and "r&d" in cells[0].get_text(strip=True).lower():
                    try:
                        result["rd_expense"] = float(cells[1].get_text(strip=True).replace(",",""))
                    except Exception:
                        pass
                    break

            time.sleep(1)
        except Exception as e:
            logger.warning(f"Screener fetch failed for {self.ticker}: {e}")
        return result

    # ─────────────────────────────────────────────────────
    # SCORING MODULES
    # ─────────────────────────────────────────────────────
    def _score_business_quality(self, info: dict) -> tuple:
        """
        20 points — Business model, domestic vs export mix,
        branded vs generic, moat assessment.
        """
        score = 0
        flags = []

        model = PHARMA_BUSINESS_MODEL.get(self.ticker, "Generic Pharma")

        # Moat signals via margins
        ebitda_margin = info.get("ebitdaMargins", 0) * 100 if info.get("ebitdaMargins") else 0
        net_margin    = info.get("profitMargins", 0) * 100  if info.get("profitMargins") else 0

        if ebitda_margin >= PHARMA_BENCHMARKS["ebitda_margin"]["excellent"]:
            score += 10; flags.append(f"✅ Excellent EBITDA margin {ebitda_margin:.1f}%")
        elif ebitda_margin >= PHARMA_BENCHMARKS["ebitda_margin"]["good"]:
            score += 7;  flags.append(f"✅ Good EBITDA margin {ebitda_margin:.1f}%")
        elif ebitda_margin >= PHARMA_BENCHMARKS["ebitda_margin"]["average"]:
            score += 4;  flags.append(f"⚠️ Average EBITDA margin {ebitda_margin:.1f}%")
        else:
            score += 1;  flags.append(f"❌ Weak EBITDA margin {ebitda_margin:.1f}%")

        # Business model bonus
        if "Specialty" in model or "CDMO" in model:
            score += 6;  flags.append(f"✅ Premium model: {model}")
        elif "Branded" in model:
            score += 4;  flags.append(f"✅ Branded generic: {model}")
        else:
            score += 2;  flags.append(f"⚠️ Generic heavy: {model}")

        # Beta (stability)
        beta = info.get("beta", 1.0) or 1.0
        if beta < 0.7:
            score += 4; flags.append("✅ Low beta — defensive stock")
        elif beta > 1.3:
            flags.append("⚠️ High beta — volatile")

        return min(score, 20), flags, model

    def _score_rd_pipeline(self, info: dict, screener: dict, income) -> tuple:
        """
        25 points — R&D intensity, ANDA filings proxy, patent pipeline,
        biosimilar/specialty pipeline assessment.
        """
        score = 0
        flags = []

        # R&D as % of revenue
        try:
            revenue = 0
            rd_exp  = 0
            if income is not None and not income.empty:
                rev_rows = ["Total Revenue", "Revenue"]
                for r in rev_rows:
                    if r in income.index:
                        revenue = income.loc[r].iloc[0] / 1e7  # Crores
                        break
                if "Research And Development" in income.index:
                    rd_exp = income.loc["Research And Development"].iloc[0] / 1e7
                elif "ResearchAndDevelopment" in income.index:
                    rd_exp = income.loc["ResearchAndDevelopment"].iloc[0] / 1e7

            # Fallback from screener
            if rd_exp == 0 and "rd_expense" in screener:
                rd_exp = screener["rd_expense"]

            rd_pct = (rd_exp / revenue * 100) if revenue > 0 else 0

            if rd_pct >= PHARMA_BENCHMARKS["rd_to_revenue_pct"]["excellent"]:
                score += 15; flags.append(f"✅ High R&D intensity: {rd_pct:.1f}% of revenue")
            elif rd_pct >= PHARMA_BENCHMARKS["rd_to_revenue_pct"]["good"]:
                score += 10; flags.append(f"✅ Good R&D intensity: {rd_pct:.1f}%")
            elif rd_pct >= PHARMA_BENCHMARKS["rd_to_revenue_pct"]["average"]:
                score += 6;  flags.append(f"⚠️ Average R&D: {rd_pct:.1f}%")
            elif rd_pct > 0:
                score += 3;  flags.append(f"❌ Low R&D: {rd_pct:.1f}%")
            else:
                flags.append("⚠️ R&D data unavailable")
                score += 5  # Neutral default

        except Exception as e:
            logger.warning(f"R&D calc failed: {e}")
            score += 5
            flags.append("⚠️ R&D data unavailable — using neutral score")

        # Pipeline bonus by company (known data)
        pipeline_bonus = {
            "SUNPHARMA":  8,  # Strong specialty pipeline (Ilumya, Cequa)
            "DRREDDY":    7,  # Biosimilars pipeline
            "DIVISLAB":   6,  # CDMO pipeline
            "CIPLA":      6,  # Respiratory pipeline
            "LUPIN":      5,  # US generics pipeline
            "AUROPHARMA": 4,
            "TORNTPHARM": 5,
            "GLENMARK":   4,
            "APOLLOHOSP": 3,
        }
        bonus = pipeline_bonus.get(self.ticker, 5)
        score += bonus
        flags.append(f"📋 Pipeline quality score: {bonus}/8")

        # Biosimilar/Specialty flag
        specialty_cos = ["SUNPHARMA", "CIPLA", "DIVISLAB", "DRREDDY"]
        if self.ticker in specialty_cos:
            score += 2
            flags.append("✅ Specialty/Biosimilar pipeline presence")

        return min(score, 25), flags, rd_pct if 'rd_pct' in dir() else 0

    def _score_fda_compliance(self) -> tuple:
        """
        20 points — USFDA warning letters, Form 483 observations,
        import alerts, compliance track record.
        Lower regulatory risk = higher score.
        """
        score = 20  # Start full, deduct for issues
        flags = []

        fda = FDA_RISK_REGISTRY.get(self.ticker, {"warning_letters": 0, "483_observations": 0, "import_alerts": 0})

        wl  = fda.get("warning_letters", 0)
        obs = fda.get("483_observations", 0)
        imp = fda.get("import_alerts", 0)
        last = fda.get("last_issue", "N/A")

        # Warning letters: -4 pts each (severe)
        score -= wl * 4
        if wl > 0: flags.append(f"🔴 {wl} USFDA Warning Letter(s) — last: {last}")
        else:       flags.append("✅ No USFDA Warning Letters")

        # 483 observations: -1 pt each (moderate)
        score -= min(obs, 6)
        if obs > 3: flags.append(f"🟠 {obs} Form 483 observations — compliance risk")
        elif obs > 0: flags.append(f"⚠️ {obs} Form 483 observations")
        else:         flags.append("✅ Clean Form 483 record")

        # Import alerts: -3 pts each (severe)
        score -= imp * 3
        if imp > 0: flags.append(f"🔴 {imp} Import Alert(s) — US revenue at risk")
        else:        flags.append("✅ No Import Alerts")

        # Time since last issue (recovery bonus)
        try:
            if last != "N/A":
                years_ago = datetime.now().year - int(last)
                if years_ago >= 3:
                    score += 3; flags.append(f"✅ {years_ago}yr since last FDA issue — recovering")
        except Exception:
            pass

        score = max(0, min(20, score))
        fda_grade = (
            "🟢 LOW RISK"    if score >= 18 else
            "🟡 MEDIUM RISK" if score >= 12 else
            "🟠 HIGH RISK"   if score >= 6  else
            "🔴 CRITICAL RISK"
        )

        return score, flags, fda_grade

    def _score_financials(self, info: dict, screener: dict) -> tuple:
        """20 points — ROE, revenue growth, FCF generation."""
        score = 0
        flags = []

        roe            = (info.get("returnOnEquity", 0) or 0) * 100
        revenue_growth = (info.get("revenueGrowth", 0) or 0) * 100
        fcf_yield      = (info.get("freeCashflow", 0) or 0) / max(info.get("marketCap", 1), 1) * 100

        # ROE
        if roe >= PHARMA_BENCHMARKS["roe"]["excellent"]:
            score += 8; flags.append(f"✅ Strong ROE: {roe:.1f}%")
        elif roe >= PHARMA_BENCHMARKS["roe"]["good"]:
            score += 6; flags.append(f"✅ Good ROE: {roe:.1f}%")
        elif roe >= PHARMA_BENCHMARKS["roe"]["average"]:
            score += 4; flags.append(f"⚠️ Average ROE: {roe:.1f}%")
        else:
            score += 1; flags.append(f"❌ Weak ROE: {roe:.1f}%")

        # Revenue growth
        if revenue_growth >= 15:
            score += 7; flags.append(f"✅ Strong revenue growth: {revenue_growth:.1f}%")
        elif revenue_growth >= 10:
            score += 5; flags.append(f"✅ Good revenue growth: {revenue_growth:.1f}%")
        elif revenue_growth >= 5:
            score += 3; flags.append(f"⚠️ Moderate revenue growth: {revenue_growth:.1f}%")
        else:
            flags.append(f"❌ Low revenue growth: {revenue_growth:.1f}%")

        # FCF yield
        if fcf_yield >= 5:
            score += 5; flags.append(f"✅ Strong FCF yield: {fcf_yield:.1f}%")
        elif fcf_yield >= 2:
            score += 3; flags.append(f"✅ Decent FCF yield: {fcf_yield:.1f}%")
        else:
            score += 1; flags.append(f"⚠️ Low FCF yield: {fcf_yield:.1f}%")

        return min(score, 20), flags

    def _score_valuation(self, info: dict) -> tuple:
        """15 points — P/E, P/B, EV/EBITDA vs pharma norms."""
        score = 0
        flags = []

        pe  = info.get("trailingPE", 0) or 0
        pb  = info.get("priceToBook", 0) or 0
        peg = info.get("pegRatio", 0) or 0

        # Pharma sector norms: P/E 20-35x is fair
        if 0 < pe <= 20:
            score += 6; flags.append(f"✅ Attractive P/E: {pe:.1f}x")
        elif pe <= 30:
            score += 4; flags.append(f"🟡 Fair P/E: {pe:.1f}x")
        elif pe <= 45:
            score += 2; flags.append(f"⚠️ Premium P/E: {pe:.1f}x")
        elif pe > 45:
            flags.append(f"❌ Very expensive P/E: {pe:.1f}x")

        # P/B
        if 0 < pb <= 3:
            score += 5; flags.append(f"✅ Attractive P/B: {pb:.1f}x")
        elif pb <= 6:
            score += 3; flags.append(f"🟡 Fair P/B: {pb:.1f}x")
        else:
            score += 1; flags.append(f"⚠️ High P/B: {pb:.1f}x")

        # PEG
        if 0 < peg <= 1:
            score += 4; flags.append(f"✅ Excellent PEG: {peg:.2f}")
        elif peg <= 2:
            score += 2; flags.append(f"🟡 Fair PEG: {peg:.2f}")

        # Fair value estimate (DCF-light)
        eps = info.get("trailingEps", 0) or 0
        growth_rate = min(info.get("earningsGrowth", 0.12) or 0.12, 0.20)
        fair_pe = 25 + (growth_rate * 100 - 10) * 0.5  # Pharma base PE of 25
        fair_value = eps * fair_pe if eps > 0 else 0

        return min(score, 15), flags, fair_value

    # ─────────────────────────────────────────────────────
    # MAIN ANALYZE
    # ─────────────────────────────────────────────────────
    def analyze(self) -> dict:
        logger.info(f"💊 Running PharmaAgent.analyze() for {self.ticker}")
        raw = self._fetch_data()
        info     = raw.get("info", {})
        income   = raw.get("income")
        screener = raw.get("screener", {})

        # Run all scoring modules
        biz_score,  biz_flags,  model      = self._score_business_quality(info)
        rd_score,   rd_flags,   rd_pct     = self._score_rd_pipeline(info, screener, income)
        fda_score,  fda_flags,  fda_grade  = self._score_fda_compliance()
        fin_score,  fin_flags              = self._score_financials(info, screener)
        val_score,  val_flags,  fair_value = self._score_valuation(info)

        total_score = biz_score + rd_score + fda_score + fin_score + val_score
        grade, verdict = self._get_grade(total_score)

        current_price = info.get("currentPrice", 0) or info.get("regularMarketPrice", 0)
        upside = round((fair_value - current_price) / current_price * 100, 1) if current_price and fair_value else 0

        all_flags = biz_flags + rd_flags + fda_flags + fin_flags + val_flags

        result = {
            "ticker":          self.ticker,
            "company_name":    info.get("longName", self.ticker),
            "sector":          "PHARMA",
            "business_model":  model,
            "current_price":   current_price,
            "fair_value":      round(fair_value, 2),
            "upside_pct":      upside,
            "score":           total_score,
            "grade":           grade,
            "verdict":         verdict,
            "fda_grade":       fda_grade,
            "rd_pct":          round(rd_pct, 1),
            "scores_breakdown": {
                "business_quality": biz_score,
                "rd_pipeline":      rd_score,
                "fda_compliance":   fda_score,
                "financials":       fin_score,
                "valuation":        val_score,
            },
            "flags":           all_flags,
            "fda_details":     FDA_RISK_REGISTRY.get(self.ticker, {}),
            "fundamental_score": total_score,
            "valuation_score":   val_score * (100 / 15),
            "management_score":  (fda_score / 20) * 100,
        }

        self._print_summary(result)
        return result

    def _get_grade(self, score: int) -> tuple:
        if score >= 85: return "A+", "🟢 EXCEPTIONAL PHARMA COMPANY"
        elif score >= 75: return "A", "🟢 EXCELLENT — Strong pipeline & compliance"
        elif score >= 65: return "B+", "🟢 GOOD — Solid fundamentals"
        elif score >= 55: return "B", "🟡 AVERAGE — Monitor FDA risks"
        elif score >= 45: return "C", "🟡 BELOW AVERAGE — Regulatory concerns"
        elif score >= 35: return "D", "🟠 WEAK — High FDA/pipeline risk"
        else:             return "F", "🔴 AVOID — Critical risks present"

    def _print_summary(self, r: dict):
        print(f"\n{'='*60}")
        print(f"💊 PHARMA ANALYSIS: {r['company_name']} ({r['ticker']})")
        print(f"{'='*60}")
        print(f"Business Model  : {r['business_model']}")
        print(f"Score           : {r['score']}/100 ({r['grade']}) — {r['verdict']}")
        print(f"FDA Risk        : {r['fda_grade']}")
        print(f"R&D Intensity   : {r['rd_pct']}% of revenue")
        print(f"Current Price   : ₹{r['current_price']:,.2f}")
        print(f"Fair Value Est. : ₹{r['fair_value']:,.2f} (Upside: {r['upside_pct']}%)")
        print(f"\nScore Breakdown :")
        for k, v in r["scores_breakdown"].items():
            print(f"  {k:22}: {v}")
        print(f"\nKey Flags:")
        for flag in r["flags"]:
            print(f"  {flag}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    agent = PharmaAgent("SUNPHARMA")
    result = agent.analyze()
