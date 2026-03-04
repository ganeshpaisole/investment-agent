# ============================================================
# utils/bse_parser.py — BSE Filing Parser v1.0
# ============================================================
# Auto-downloads quarterly investor presentations and results
# from BSE (bseindia.com) and extracts key metrics that are
# unavailable from Yahoo Finance or Screener.in:
#
#   Banks:     CASA ratio, CAR, NIM, GNPA, NNPA, Credit Cost
#   Insurance: VNB margin, EV, Solvency ratio, Claims ratio
#   NBFC:      AUM, Stage-2/3 assets, Credit Cost, CoF
#   Generic:   Order book, capacity utilisation (infra/capital goods)
#
# Architecture:
#   1. BSE search API  → find latest filing URLs for ticker
#   2. requests        → download PDF bytes
#   3. pdfplumber      → extract raw text from PDF
#   4. Regex patterns  → extract structured metrics
#   5. Cache layer     → skip re-download if file exists today
#
# No paid APIs. No auth. Fully public BSE data.
# ============================================================

import os
import re
import time
import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger


# ── Cache directory ───────────────────────────────────────────
CACHE_DIR  = Path("data/bse_cache")
CACHE_DAYS = 7          # Re-download after 7 days
PDF_DIR    = Path("data/bse_pdfs")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ── HTTP headers (mimic browser to avoid bot-blocking) ────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Referer": "https://www.bseindia.com/",
}

# ── BSE Security Code map for common Nifty 50 stocks ─────────
# BSE uses numeric codes, Yahoo uses NSE ticker symbols
# We maintain a lookup for the most common stocks
BSE_CODE_MAP = {
    # Banking
    "HDFCBANK":   "500180",
    "ICICIBANK":  "532174",
    "SBIN":       "500112",
    "KOTAKBANK":  "500247",
    "AXISBANK":   "532215",
    "INDUSINDBK": "532187",
    "BANDHANBNK": "541153",
    "IDFCFIRSTB": "539437",
    "FEDERALBNK": "500469",
    "YESBANK":    "532648",
    # NBFC
    "BAJFINANCE": "500034",
    "BAJAJFINSV": "532978",
    "HDFCAMC":    "541729",
    "MUTHOOTFIN": "533398",
    "CHOLAFIN":   "590005",
    # Insurance
    "HDFCLIFE":   "540777",
    "SBILIFE":    "540719",
    "ICICIPRU":   "540133",
    "STARHEALTH": "543412",
    "LICI":       "543526",
    # IT
    "TCS":        "532540",
    "INFY":       "500209",
    "WIPRO":      "507685",
    "HCLTECH":    "532281",
    "TECHM":      "532755",
    "LTIM":       "540005",
    # FMCG
    "HINDUNILVR": "500696",
    "ITC":        "500875",
    "NESTLEIND":  "500790",
    "BRITANNIA":  "500825",
    "MARICO":     "531642",
    "DABUR":      "500096",
    "TATACONSUM": "500800",
    # Auto
    "MARUTI":     "532500",
    "TATAMOTORS": "500570",
    "BAJAJ-AUTO": "532977",
    "EICHERMOT":  "505200",
    "HEROMOTOCO": "500182",
    "M&M":        "500520",
    # Pharma
    "SUNPHARMA":  "524715",
    "DRREDDY":    "500124",
    "CIPLA":      "500087",
    "DIVISLAB":   "532488",
    "APOLLOHOSP": "508869",
    # Industrials
    "LT":         "500510",
    "NTPC":       "532555",
    "POWERGRID":  "532898",
    "ONGC":       "500312",
    "ADANIPORTS": "532921",
    # Materials
    "TATASTEEL":  "500470",
    "JSWSTEEL":   "500228",
    "HINDALCO":   "500440",
    "COALINDIA":  "533278",
    # Telecom / Energy
    "BHARTIARTL": "532454",
    "RELIANCE":   "500325",
}

# ── Filing categories on BSE ──────────────────────────────────
# These are the BSE category codes for different filing types
FILING_CATEGORIES = {
    "results":       "Result",          # Quarterly financial results
    "presentation":  "Investor Presentation",  # Investor day / analyst presentations
    "annual":        "Annual Report",   # Annual reports
}


# ── Direct investor presentation URLs for CASA ratio ─────────
# BSE filings don't contain CASA ratio — banks host presentations
# on their own IR pages. These URLs are updated quarterly.
# Format: ticker → list of URLs to try (most recent first)
CASA_URL_MAP = {
    "HDFCBANK": [
        # Confirmed working — hdfc.bank.in domain, Q3FY26
        "https://www.hdfc.bank.in/content/dam/hdfcbankpws/in/en/pdf/financial-results/2025-2026/quarter-3/Q3FY26-earnings-presentation.pdf",
        # Q2FY26 fallback
        "https://www.hdfc.bank.in/content/dam/hdfcbankpws/in/en/pdf/financial-results/2025-2026/quarter-2/Q2FY26-earnings-presentation.pdf",
        # Q4FY25 fallback
        "https://www.hdfc.bank.in/content/dam/hdfcbankpws/in/en/pdf/financial-results/2024-2025/quarter-4/Q4FY25-earnings-presentation.pdf",
    ],
    "ICICIBANK": [
        "https://www.icicibank.com/content/dam/icicibank/india/managed-assets/docs/investor/quarterly-financial-results/2025/q3-fy2025-analyst-presentation.pdf",
    ],
    "SBIN": [
        "https://bank.sbi/documents/16012/0/Analyst+Presentation+Q3FY25.pdf",
    ],
    "KOTAKBANK": [
        "https://ir.kotak.com/downloads/quarterly-results/FY2025/Q3-FY2025-Analyst-Presentation.pdf",
    ],
    "AXISBANK": [
        "https://www.axisbank.com/docs/default-source/investor-relations/quarterly-results/fy2025/q3fy25/investor-presentation.pdf",
    ],
    "BAJFINANCE": [
        "https://www.bajajfinserv.in/finance/downloads/investor-relations/2025/Q3-FY25-BFL-Investor-Presentation.pdf",
    ],
}


# ============================================================
# BSE FILING SEARCH
# ============================================================

class BSEFilingParser:
    """
    Downloads and parses BSE quarterly filings to extract
    key banking/insurance/NBFC metrics unavailable elsewhere.

    Usage:
        parser = BSEFilingParser("HDFCBANK")
        data = parser.get_filing_data()
        # Returns: {"casa": 38.5, "car": 18.2, "nim": 3.6, ...}
    """

    def __init__(self, ticker: str):
        self.ticker   = ticker.upper().strip()
        self.bse_code = BSE_CODE_MAP.get(self.ticker)
        self.session  = requests.Session()
        self.session.headers.update(HEADERS)

    # ──────────────────────────────────────────────────────────
    # CACHE HELPERS
    # ──────────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path:
        safe = hashlib.md5(key.encode()).hexdigest()[:12]
        return CACHE_DIR / f"{self.ticker}_{safe}.json"

    def _cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < timedelta(days=CACHE_DAYS)

    def _cache_read(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _cache_write(self, path: Path, data: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ──────────────────────────────────────────────────────────
    # BSE API: FIND FILINGS
    # ──────────────────────────────────────────────────────────

    def _get_bse_filings(self, category: str = "Result", max_results: int = 5) -> list:
        """
        Query BSE API to get list of recent filings for this stock.
        Returns list of filing dicts with URL, date, description.
        """
        if not self.bse_code:
            logger.warning(f"⚠️ BSE code not found for {self.ticker} — add to BSE_CODE_MAP")
            return []

        cache_key = f"filings_{category}_{self.bse_code}"
        cache_path = self._cache_path(cache_key)
        if self._cache_valid(cache_path):
            return self._cache_read(cache_path)

        try:
            # BSE filing search API (public endpoint)
            url = (
                f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
                f"?strCat={category}&strType=C&strScrip={self.bse_code}"
                f"&strSearch=P&strToDate=&strFromDate=&myClient="
            )
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            filings = []
            for item in data.get("Table", [])[:max_results]:
                # NSURL = BSE stock page (HTML) — never use for PDF download
                # ATTACHMENTNAME = actual PDF filename — always use this
                attach = item.get("ATTACHMENTNAME", "").strip()
                if not attach:
                    continue
                # AttachHis is the correct BSE PDF path (confirmed working)
                filing_url = f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{attach}"
                filings.append({
                    "url":         filing_url,
                    "date":        item.get("News_submission_dt", ""),
                    "description": item.get("NEWSSUB", ""),
                    "category":    category,
                })

            self._cache_write(cache_path, filings)
            logger.info(f"📂 BSE: Found {len(filings)} '{category}' filings for {self.ticker}")
            return filings

        except Exception as e:
            logger.warning(f"⚠️ BSE filing search failed for {self.ticker}: {e}")
            return []

    # ──────────────────────────────────────────────────────────
    # PDF DOWNLOAD
    # ──────────────────────────────────────────────────────────

    def _download_pdf(self, url: str):
        """Download a PDF from BSE and save locally. Returns local path."""
        # Use URL hash as filename to avoid re-downloads
        fname = hashlib.md5(url.encode()).hexdigest()[:16] + ".pdf"
        local = PDF_DIR / fname

        # Already downloaded?
        if local.exists() and local.stat().st_size > 1000:
            logger.info(f"📄 PDF cached: {local.name}")
            return local

        try:
            logger.info(f"⬇️  Downloading PDF: {url[:70]}...")
            resp = self.session.get(url, timeout=30, stream=True)
            resp.raise_for_status()

            if "pdf" not in resp.headers.get("Content-Type", "").lower():
                logger.warning(f"⚠️ Not a PDF: {url}")
                return None

            with open(local, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_kb = local.stat().st_size // 1024
            logger.info(f"✅ Downloaded {size_kb} KB → {local.name}")
            return local

        except Exception as e:
            logger.warning(f"⚠️ PDF download failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    # PDF TEXT EXTRACTION
    # ──────────────────────────────────────────────────────────

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract all text from a PDF file."""
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:40]:  # First 40 pages (enough for results)
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            logger.info(f"📝 Extracted {len(text):,} chars from PDF")
            return text
        except ImportError:
            logger.warning("⚠️ pdfplumber not installed — run: pip install pdfplumber")
            return ""
        except Exception as e:
            logger.warning(f"⚠️ PDF text extraction failed: {e}")
            return ""

    # ──────────────────────────────────────────────────────────
    # REGEX EXTRACTION — BANKING
    # ──────────────────────────────────────────────────────────

    def _extract_banking_metrics(self, text: str) -> dict:
        """
        Extract key banking metrics from PDF text using regex patterns.
        Patterns verified against actual BSE quarterly results PDFs.

        HDFC Bank format examples (from real Q4FY25 PDF):
          Capital Adequacy Ratio 19.55% 19.97% 18.80% 19.55% 18.80%
          % of Gross NPA s to Gross Advances 1.33% 1.42% 1.24% 1.33% 1.24%
          % of Net NPAs to Net Advances 0.43% 0.46% 0.33% 0.43% 0.33%
          Return on assets (average) -not annualized 0.48% 0.47% 0.49% 1.91% 1.98%
          Net interest margin was at 3.54% on total assets
          CASA deposits were ... growth of 5.7%  (amount not ratio in results PDF)
        """
        result = {}

        def find_pct(patterns, text, max_val=100):
            """Try multiple regex patterns, return first valid percentage match."""
            for pattern in patterns:
                for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                    try:
                        val = float(m.group(1).replace(",", ""))
                        if 0 < val < max_val:
                            return round(val, 2)
                    except:
                        pass
            return None

        def find_first_pct_after(keyword, text, max_val=100):
            """Find the first percentage after a keyword — handles table format."""
            idx = text.upper().find(keyword.upper())
            if idx < 0:
                return None
            snippet = text[idx:idx+200]
            m = re.search(r"([\d.]+)\s*%", snippet)
            if m:
                try:
                    val = float(m.group(1))
                    if 0 < val < max_val:
                        return round(val, 2)
                except:
                    pass
            return None

        def find_annual_pct_after(keyword, text, col=4, max_val=100):
            """
            For table rows like: 'Label  Q4  Q3  Q2  YTD  PY'
            col=4 picks the YTD/annual figure (4th number after keyword).
            """
            idx = text.upper().find(keyword.upper())
            if idx < 0:
                return None
            snippet = text[idx:idx+300]
            nums = re.findall(r"([\d.]+)\s*%", snippet)
            if len(nums) >= col:
                try:
                    val = float(nums[col-1])
                    if 0 < val < max_val:
                        return round(val, 2)
                except:
                    pass
            # Fallback to first percentage
            if nums:
                try:
                    val = float(nums[0])
                    if 0 < val < max_val:
                        return round(val, 2)
                except:
                    pass
            return None

        # ── CAR / Capital Adequacy Ratio ──────────────────────
        # Format: "Capital Adequacy Ratio 19.55% 19.97% 18.80% 19.55% 18.80%"
        car = find_first_pct_after("Capital Adequacy Ratio", text, max_val=50)
        if not car:
            car = find_pct([
                r"CRAR\s*[:\-]?\s*([\d.]+)\s*%",
                r"\bCAR\b\s*[:\-]?\s*([\d.]+)\s*%",
            ], text, max_val=50)
        if car:
            result["car"] = car
            logger.info(f"  ✅ CAR: {car}%")

        # ── GNPA % ────────────────────────────────────────────
        # HDFC:  "% of Gross NPA s to Gross Advances 1.33% 1.42%"  ← has % sign
        # ICICI: "gross NPA ratio was 1.67% at June 30, 2025"      ← prose
        #        "% of gross non-performing customer assets ... 1.67% 1.67%"
        # Axis:  "% of Gross NPAs 1.46 l.57 l.44"                  ← NO % sign, 'l' OCR artefact
        gnpa = find_first_pct_after("% of Gross NPA", text, max_val=30)
        if not gnpa:
            gnpa = find_pct([
                r"gross\s+NPA\s+ratio\s+was\s+([\d.]+)\s*%",
                r"[Gg]ross\s+NPA\s+ratio\s+of\s+([\d.]+)\s*%",
                r"Gross\s+NPA[s]?\s+ratio\s*[:\-]?\s*([\d.]+)\s*%",
            ], text, max_val=30)
        if not gnpa:
            # Axis format: "% of Gross NPAs 1.46 1.57 1.44" — no % sign
            # OCR sometimes renders "1" as "l", so match both
            m = re.search(
                r"%\s+of\s+Gross\s+NPA[s]?\s+([\dl][.,\d]*)\s",
                text, re.IGNORECASE
            )
            if m:
                try:
                    val = float(m.group(1).replace("l", "1").replace(",", ""))
                    if 0.1 < val < 30:
                        gnpa = round(val, 2)
                except:
                    pass
        if gnpa:
            result["gnpa"] = gnpa
            logger.info(f"  ✅ GNPA: {gnpa}%")

        # ── NNPA % ────────────────────────────────────────────
        # HDFC:  "% of Net NPAs to Net Advances 0.43%"
        # ICICI: "Net NPA ratio was 0.41% at June 30"
        #        "% of net non-performing customer assets ... 0.41%"
        # Axis:  "% of Net NPAs 0.44 0.45 0.34"  ← no % sign
        nnpa = find_first_pct_after("% of Net NPA", text, max_val=15)
        if not nnpa:
            nnpa = find_pct([
                r"[Nn]et\s+NPA\s+ratio\s+was\s+([\d.]+)\s*%",
                r"[Nn]et\s+NPA\s+ratio\s+of\s+([\d.]+)\s*%",
                r"NNPA\s*[:\-]?\s*([\d.]+)\s*%",
            ], text, max_val=15)
        if not nnpa:
            # Axis format: "% of Net NPAs 0.44 0.45 0.34"
            m = re.search(
                r"%\s+of\s+Net\s+NPA[s]?\s+([\dl][.,\d]*)\s",
                text, re.IGNORECASE
            )
            if m:
                try:
                    val = float(m.group(1).replace("l", "1").replace(",", ""))
                    if 0.01 < val < 15:
                        nnpa = round(val, 2)
                except:
                    pass
        if nnpa:
            result["nnpa"] = nnpa
            logger.info(f"  ✅ NNPA: {nnpa}%")

        # ── NIM / Net Interest Margin ─────────────────────────
        # HDFC:  "Net interest margin was at 3.54% on total assets"
        # ICICI: "Net interest margin was 4.34% in Q1-2026"
        # Axis:  NOT in results PDF (only in investor presentation)
        nim = find_pct([
            r"[Nn]et\s+interest\s+margin\s+was\s+(?:at\s+)?([\d.]+)\s*%",
            r"[Nn]et\s+interest\s+margin\s+(?:of\s+)?([\d.]+)\s*%",
            r"[Nn]et\s+[Ii]nterest\s+[Mm]argin\s+(?:was\s+)?(?:at\s+)?([\d.]+)\s*%",
            r"NIM\s*[:\-]?\s*([\d.]+)\s*%",
            r"NIM\s+\(annualised\)\s*[:\-]?\s*([\d.]+)",
        ], text, max_val=15)
        if nim:
            result["nim"] = nim
            logger.info(f"  ✅ NIM: {nim}%")

        # ── ROA ───────────────────────────────────────────────
        # HDFC:  "Return on assets (average) -not annualized 0.48%...1.91% 1.98%"  ← col4=annual
        # ICICI: "Return on assets (annualised) 2.44% 2.52% 2.36%"                 ← col1
        # Axis:  "Return on Assets (annualized) % 1.23 1.47 1.84"                  ← NO % sign
        roa = find_first_pct_after("Return on assets", text, max_val=10)
        if roa and roa < 0.5:
            # Very small = quarterly not-annualized (HDFC) — try col4 for annual
            roa = find_annual_pct_after("Return on assets", text, col=4, max_val=10)
        if not roa:
            # Axis format: "Return on Assets (annualized) % 1.23 1.47 1.84"
            m = re.search(
                r"Return\s+on\s+Assets?\s*\([^)]*\)\s*%\s+([\dl][.,\d]*)",
                text, re.IGNORECASE
            )
            if m:
                try:
                    val = float(m.group(1).replace("l", "1").replace(",", ""))
                    if 0.1 < val < 10:
                        roa = round(val, 2)
                except:
                    pass
        if not roa:
            roa = find_pct([
                r"ROA\s*[:\-]?\s*([\d.]+)\s*%",
                r"RoA\s*\(annualised\)\s*[:\-]?\s*([\d.]+)",
            ], text, max_val=10)
        if roa:
            result["roa_pdf"] = roa
            logger.info(f"  ✅ ROA (PDF): {roa}%")

        # ── CASA Ratio ────────────────────────────────────────
        # ICICI: "CASA ratio was 38.7% in Q1-2026"  ← IN results PDF!
        # HDFC/Axis/SBI: only in investor presentations
        casa = find_pct([
            r"CASA\s+ratio\s+was\s+([\d.]+)\s*%",
            r"CASA\s+[Rr]atio\s*[:\-]?\s*([\d.]+)\s*%",
            r"CASA\s*[:\-]\s*([\d.]+)\s*%",
            r"CASA\s+(?:deposits?\s+)?(?:stood\s+at|was|were|at)\s+([\d.]+)\s*%",
            r"(?:^|\s)CASA\s+([\d.]+)\s*%",
            r"Current\s+and\s+[Ss]avings\s+.*?([\d.]+)\s*%",
        ], text, max_val=80)
        if casa:
            result["casa"] = casa
            logger.info(f"  ✅ CASA: {casa}%")
        else:
            logger.info("  ℹ️  CASA ratio not in results PDF — check investor presentation")

        # ── ROE ───────────────────────────────────────────────
        roe = find_pct([
            r"Return\s+on\s+[Ee]quity\s*[:\-]?\s*([\d.]+)\s*%",
            r"ROE\s*[:\-]?\s*([\d.]+)\s*%",
            r"RoE\s*\(annualised\)\s*[:\-]?\s*([\d.]+)",
        ], text, max_val=50)
        if roe:
            result["roe_pdf"] = roe
            logger.info(f"  ✅ ROE (PDF): {roe}%")

        # ── Cost of Funds ─────────────────────────────────────
        cof = find_pct([
            r"Cost\s+of\s+[Ff]unds?\s*[:\-]?\s*([\d.]+)\s*%",
            r"Cost\s+of\s+[Bb]orrowings?\s*[:\-]?\s*([\d.]+)\s*%",
            r"Cost\s+of\s+[Ll]iabilities\s*[:\-]?\s*([\d.]+)\s*%",
            r"Cost\s+of\s+[Dd]eposits?\s*[:\-]?\s*([\d.]+)\s*%",
        ], text, max_val=20)
        if cof:
            result["cost_of_funds"] = cof
            logger.info(f"  ✅ Cost of Funds: {cof}%")

        return result

    # ──────────────────────────────────────────────────────────
    # REGEX EXTRACTION — INSURANCE
    # ──────────────────────────────────────────────────────────

    def _extract_insurance_metrics(self, text: str) -> dict:
        result = {}

        def find_pct(patterns, text):
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    try:
                        val = float(match.group(1).replace(",", ""))
                        if 0 < val < 200:
                            return round(val, 2)
                    except:
                        pass
            return None

        def find_num(patterns, text, scale=1):
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    try:
                        return round(float(match.group(1).replace(",", "")) * scale, 2)
                    except:
                        pass
            return None

        # ── VNB Margin ────────────────────────────────────────
        vnb_margin = find_pct([
            r"VNB\s+Margin\s*[:\-]?\s*([\d.]+)\s*%",
            r"Value\s+of\s+New\s+Business\s+Margin\s*[:\-]?\s*([\d.]+)\s*%",
            r"NBM\s*[:\-]?\s*([\d.]+)\s*%",
            r"New\s+Business\s+Margin\s*[:\-]?\s*([\d.]+)\s*%",
        ], text)
        if vnb_margin:
            result["vnb_margin"] = vnb_margin
            logger.info(f"  ✅ VNB Margin: {vnb_margin}%")

        # ── Solvency Ratio ────────────────────────────────────
        solvency = find_pct([
            r"Solvency\s+Ratio\s*[:\-]?\s*([\d.]+)\s*%",
            r"Solvency\s+Margin\s*[:\-]?\s*([\d.]+)\s*%",
            r"Solvency\s*[:\-]?\s*([\d.]+)x",
        ], text)
        if solvency:
            result["solvency_ratio"] = solvency
            logger.info(f"  ✅ Solvency: {solvency}%")

        # ── Claims Ratio ──────────────────────────────────────
        claims = find_pct([
            r"Claims\s+Ratio\s*[:\-]?\s*([\d.]+)\s*%",
            r"Loss\s+Ratio\s*[:\-]?\s*([\d.]+)\s*%",
            r"Incurred\s+Claims?\s+Ratio\s*[:\-]?\s*([\d.]+)\s*%",
            r"(?:Net\s+)?Claims?\s+paid\s+ratio\s*[:\-]?\s*([\d.]+)",
        ], text)
        if claims:
            result["claims_ratio"] = claims
            logger.info(f"  ✅ Claims Ratio: {claims}%")

        # ── Combined Ratio ────────────────────────────────────
        combined = find_pct([
            r"Combined\s+Ratio\s*[:\-]?\s*([\d.]+)\s*%",
            r"Expense\s+of\s+Management\s+Ratio\s*[:\-]?\s*([\d.]+)\s*%",
        ], text)
        if combined:
            result["combined_ratio"] = combined
            logger.info(f"  ✅ Combined Ratio: {combined}%")

        # ── Embedded Value ────────────────────────────────────
        ev = find_num([
            r"Embedded\s+Value\s*[:\-]?\s*(?:Rs\.?|₹|INR)?\s*([\d,]+(?:\.\d+)?)\s*(?:Cr|crore|Bn|billion)?",
            r"EV\s+(?:per\s+share)?\s*[:\-]?\s*(?:Rs\.?|₹)?\s*([\d,]+)",
        ], text)
        if ev and ev > 100:  # Must be > ₹100 to be plausible
            result["embedded_value_cr"] = ev
            logger.info(f"  ✅ Embedded Value: ₹{ev} Cr")

        # ── VNB ───────────────────────────────────────────────
        vnb = find_num([
            r"Value\s+of\s+New\s+Business\s*[:\-]?\s*(?:Rs\.?|₹|INR)?\s*([\d,]+(?:\.\d+)?)\s*(?:Cr|crore)?",
            r"VNB\s*[:\-]?\s*(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)\s*(?:Cr|crore)?",
        ], text)
        if vnb and vnb > 10:
            result["vnb_cr"] = vnb
            logger.info(f"  ✅ VNB: ₹{vnb} Cr")

        # ── 13th Month Persistency ────────────────────────────
        persistency = find_pct([
            r"13th\s+Month\s+Persistency\s*[:\-]?\s*([\d.]+)\s*%",
            r"Persistency\s+\(13th\s+Month\)\s*[:\-]?\s*([\d.]+)\s*%",
        ], text)
        if persistency:
            result["persistency_13m"] = persistency
            logger.info(f"  ✅ 13M Persistency: {persistency}%")

        return result

    # ──────────────────────────────────────────────────────────
    # REGEX EXTRACTION — NBFC
    # ──────────────────────────────────────────────────────────

    def _extract_nbfc_metrics(self, text: str) -> dict:
        result = {}

        def find_pct(patterns, text):
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    try:
                        val = float(match.group(1).replace(",", ""))
                        if 0 < val < 100:
                            return round(val, 2)
                    except:
                        pass
            return None

        def find_num(patterns, text):
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    try:
                        return float(match.group(1).replace(",", ""))
                    except:
                        pass
            return None

        # ── AUM ───────────────────────────────────────────────
        aum = find_num([
            r"AUM\s*[:\-]?\s*(?:Rs\.?|₹|INR)?\s*([\d,]+(?:\.\d+)?)\s*(?:Cr|crore)",
            r"Assets\s+Under\s+Management\s*[:\-]?\s*(?:Rs\.?|₹)?\s*([\d,]+)",
            r"Loan\s+Book\s*[:\-]?\s*(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)\s*(?:Cr|crore)",
        ], text)
        if aum and aum > 1000:
            result["aum_cr"] = aum
            logger.info(f"  ✅ AUM: ₹{aum:,.0f} Cr")

        # ── Yield on Assets ───────────────────────────────────
        yield_on_assets = find_pct([
            r"Yield\s+on\s+Assets?\s*[:\-]?\s*([\d.]+)\s*%",
            r"Yield\s+on\s+Loan\s+Book\s*[:\-]?\s*([\d.]+)\s*%",
            r"Interest\s+Yield\s*[:\-]?\s*([\d.]+)\s*%",
        ], text)
        if yield_on_assets:
            result["yield_on_assets"] = yield_on_assets
            logger.info(f"  ✅ Yield on Assets: {yield_on_assets}%")

        # ── Stage-2 Assets ────────────────────────────────────
        stage2 = find_pct([
            r"Stage\s*[-\s]?2\s*[:\-]?\s*([\d.]+)\s*%",
            r"Stage\s+II\s*[:\-]?\s*([\d.]+)\s*%",
        ], text)
        if stage2:
            result["stage2_pct"] = stage2
            logger.info(f"  ✅ Stage-2: {stage2}%")

        # ── Stage-3 / NPA ─────────────────────────────────────
        stage3 = find_pct([
            r"Stage\s*[-\s]?3\s*[:\-]?\s*([\d.]+)\s*%",
            r"Stage\s+III\s*[:\-]?\s*([\d.]+)\s*%",
        ], text)
        if stage3:
            result["stage3_pct"] = stage3
            logger.info(f"  ✅ Stage-3: {stage3}%")

        # ── Credit Cost ───────────────────────────────────────
        credit_cost = find_pct([
            r"Credit\s+Cost\s*[:\-]?\s*([\d.]+)\s*%",
            r"Loan\s+Loss\s+Provision\s*[:\-]?\s*([\d.]+)\s*%",
        ], text)
        if credit_cost:
            result["credit_cost"] = credit_cost
            logger.info(f"  ✅ Credit Cost: {credit_cost}%")

        return result

    # ──────────────────────────────────────────────────────────
    # MAIN: GET FILING DATA
    # ──────────────────────────────────────────────────────────

    def _fetch_casa_from_ir_page(self):
        """
        Try direct investor presentation URLs for CASA ratio.
        Banks host quarterly presentations on their own IR pages.
        Falls back gracefully — returns None if unreachable.
        """
        urls = CASA_URL_MAP.get(self.ticker, [])
        if not urls:
            return None

        logger.info(f"  🌐 Trying direct IR page for CASA ({len(urls)} URLs)...")

        for url in urls:
            try:
                # Use domain-specific headers — some banks block generic requests
                domain = url.split("/")[2]
                ir_session = requests.Session()
                ir_session.headers.update({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Referer": f"https://{domain}/",
                    "Accept": "application/pdf,*/*",
                })

                resp = ir_session.get(url, timeout=30, allow_redirects=True)
                if resp.status_code != 200:
                    logger.warning(f"  ⚠️ IR URL returned {resp.status_code}: {url[:60]}")
                    continue

                if resp.content[:4] != b"%PDF":
                    logger.warning(f"  ⚠️ Not a PDF from IR URL: {url[:60]}")
                    continue

                # Save and extract
                pdf_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                pdf_path = PDF_DIR / f"{self.ticker}_ir_{pdf_hash}.pdf"
                pdf_path.write_bytes(resp.content)
                size_kb = len(resp.content) // 1024
                logger.info(f"  ✅ IR PDF downloaded: {size_kb} KB")

                text = self._extract_text(pdf_path)
                if len(text) < 200:
                    continue

                import re
                # HDFC Bank format: "CASA ratio 38% 34% 35% 34% 34% 34%"
                # First number after "CASA ratio" is oldest quarter, last is most recent
                # We want the LAST percentage (most recent quarter)
                patterns = [
                    # Table row — grab ALL percentages, take the last one
                    r"CASA\s+ratio\s+((?:[\d.]+%\s*){2,})",
                    # Simple formats
                    r"CASA\s+[Rr]atio\s*[:\-]?\s*([\d.]+)\s*%",
                    r"\*\s*CASA\s+ratio\s+([\d.]+)%",
                    r"CASA\s*[:\-]\s*([\d.]+)\s*%",
                    r"(?:^|\s)CASA\s+([\d.]+)\s*%",
                ]
                for pattern in patterns:
                    m = re.search(pattern, text, re.IGNORECASE)
                    if m:
                        group = m.group(1)
                        # If multiple percentages in group, take the last one
                        all_pcts = re.findall(r"([\d.]+)%", group)
                        val_str = all_pcts[-1] if all_pcts else group.replace("%","").strip()
                        try:
                            val = float(val_str)
                            if 20 < val < 80:
                                logger.info(f"  ✅ CASA found in IR page: {val}%")
                                return round(val, 2)
                        except:
                            pass

                logger.info(f"  ℹ️  Downloaded IR PDF but CASA pattern not matched")

            except Exception as e:
                logger.warning(f"  ⚠️ IR page failed: {e}")
                continue

        logger.info(f"  ℹ️  CASA not found in IR pages — may need URL update")
        return None

    def get_filing_data(self, sector_type: str = "BANKING") -> dict:
        """
        Main entry point. Downloads and parses the latest quarterly
        filing for this stock and returns extracted metrics.

        Args:
            sector_type: "BANKING", "NBFC", "INSURANCE", or "GENERIC"

        Returns:
            dict of extracted metrics, e.g.:
            {
                "casa": 42.3, "car": 18.1, "nim": 3.6,
                "gnpa": 1.3, "nnpa": 0.4,
                "source": "BSE Q3FY25 results",
                "date": "2025-01-15",
            }
        """
        if not self.bse_code:
            logger.warning(f"⚠️ {self.ticker}: BSE code unknown — add to BSE_CODE_MAP in bse_parser.py")
            return {"source": "BSE code not mapped"}

        # Check JSON cache first (avoid re-parsing same PDF)
        cache_key = f"parsed_{sector_type}_{self.bse_code}"
        cache_path = self._cache_path(cache_key)
        if self._cache_valid(cache_path):
            data = self._cache_read(cache_path)
            logger.info(f"📦 BSE data from cache ({len(data)} fields) for {self.ticker}")
            return data

        logger.info(f"🏛️  BSE Parser: Searching filings for {self.ticker} ({self.bse_code})...")

        all_metrics = {}

        # ── Search order ──────────────────────────────────────
        # 1. Quarterly Results PDF — has CAR, GNPA, NNPA, NIM, ROA
        # 2. Analyst/Investor Presentation — has CASA ratio, NIM, Credit Cost
        # 3. Press Release — sometimes has CASA
        categories = ["Result", "Analyst Meet/Con. Call", "Company Update", "Press Release / Media Release"]

        for category in categories:
            # Skip non-Result categories if we already have core metrics
            # (avoids downloading ESOP/governance PDFs which have no financial data)
            core_found = all(k in all_metrics for k in ["car", "gnpa", "nnpa", "nim"])
            if core_found and category in ("Company Update", "Press Release / Media Release"):
                logger.info(f"  ⏭️  Skipping '{category}' — core metrics already found")
                continue

            filings = self._get_bse_filings(category=category, max_results=5)
            if not filings:
                continue

            logger.info(f"  📂 {category}: {len(filings)} filings found")

            for filing in filings[:3]:
                url  = filing["url"]
                date = filing["date"][:10] if filing["date"] else "unknown"
                desc = filing["description"][:60]

                logger.info(f"  📎 {desc}")

                pdf_path = self._download_pdf(url)
                if not pdf_path:
                    continue

                text = self._extract_text(pdf_path)
                if len(text) < 200:
                    logger.warning(f"     ⚠️ Too little text ({len(text)} chars) — skipping")
                    continue

                logger.info(f"  🔍 Parsing {sector_type} metrics ({len(text):,} chars)...")

                if sector_type == "BANKING":
                    metrics = self._extract_banking_metrics(text)
                elif sector_type == "NBFC":
                    metrics = {**self._extract_nbfc_metrics(text),
                               **self._extract_banking_metrics(text)}
                elif sector_type == "INSURANCE":
                    metrics = self._extract_insurance_metrics(text)
                else:
                    metrics = {}

                n_found = len(metrics)
                if n_found:
                    logger.info(f"     → {n_found} metrics: {list(metrics.keys())}")
                    metrics["source"] = f"BSE {desc}"
                    metrics["date"]   = date
                    # Merge — don't overwrite already-found metrics
                    for k, v in metrics.items():
                        if k not in all_metrics:
                            all_metrics[k] = v
                    logger.info(f"  ✅ Running total: {len([k for k in all_metrics if k not in ('source','date')])} metrics")
                else:
                    logger.info(f"     → 0 metrics found in this PDF")

                time.sleep(1)

                # Stop early if we have everything
                if sector_type == "BANKING" and all(
                    k in all_metrics for k in ["casa", "car", "nim", "gnpa", "nnpa"]
                ):
                    logger.info("  ✅ All key banking metrics found — stopping")
                    break
                elif sector_type == "INSURANCE" and all(
                    k in all_metrics for k in ["vnb_margin", "solvency_ratio"]
                ):
                    break

        # ── Direct IR page fallback for CASA ─────────────────
        # If CASA still missing, try bank's own investor presentation
        if sector_type in ("BANKING", "NBFC") and "casa" not in all_metrics:
            casa = self._fetch_casa_from_ir_page()
            if casa:
                all_metrics["casa"] = casa
                logger.info(f"  ✅ CASA: {casa}% (from IR page)")

        if not all_metrics:
            logger.warning(f"⚠️ {self.ticker}: No metrics extracted from any BSE filing")
            all_metrics = {"source": "BSE: no data extracted", "date": ""}
        else:
            n = len([k for k in all_metrics if k not in ("source", "date")])
            logger.info(f"✅ BSE Parser complete: {n} metrics for {self.ticker}")
            if sector_type == "BANKING":
                missing = [k for k in ["casa","car","nim","gnpa","nnpa","roa_pdf"]
                           if k not in all_metrics]
                if missing:
                    logger.info(f"  ℹ️  Still missing: {missing}")

        # Cache the result
        self._cache_write(cache_path, all_metrics)
        return all_metrics


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def get_bse_data(ticker: str, sector_type: str = "BANKING") -> dict:
    """
    One-line helper to get BSE filing data for any stock.

    Usage:
        from utils.bse_parser import get_bse_data
        data = get_bse_data("HDFCBANK", "BANKING")
        casa = data.get("casa")   # e.g. 42.3
        car  = data.get("car")    # e.g. 18.1
    """
    parser = BSEFilingParser(ticker)
    return parser.get_filing_data(sector_type=sector_type)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    import sys
    import shutil
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()

    # Parse flags vs tickers
    no_cache = "--no-cache" in sys.argv
    tickers  = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not tickers:
        tickers = ["HDFCBANK", "BAJFINANCE", "HDFCLIFE"]

    if no_cache:
        console.print("[yellow]⚠️  --no-cache: clearing BSE cache...[/yellow]")
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✅ Cache cleared[/green]")

    for ticker in tickers:
        console.print(f"\n[bold cyan]━━━ Testing BSE Parser: {ticker} ━━━[/bold cyan]")

        # Auto-detect sector for test
        sector_map = {
            "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "SBIN": "BANKING",
            "KOTAKBANK": "BANKING", "AXISBANK": "BANKING",
            "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC",
            "HDFCLIFE": "INSURANCE", "SBILIFE": "INSURANCE", "ICICIPRU": "INSURANCE",
        }
        sector = sector_map.get(ticker, "BANKING")

        data = get_bse_data(ticker, sector)

        t = Table(title=f"BSE Data — {ticker}", box=box.ROUNDED,
                  header_style="bold white")
        t.add_column("Metric", width=24)
        t.add_column("Value",  width=14)
        t.add_column("Status", width=10)

        key_metrics = {
            "BANKING":   ["casa", "car", "nim", "gnpa", "nnpa", "credit_cost", "roa_pdf", "roe_pdf", "cost_of_funds"],
            "NBFC":      ["aum_cr", "yield_on_assets", "stage2_pct", "stage3_pct", "credit_cost", "gnpa", "nnpa"],
            "INSURANCE": ["vnb_margin", "solvency_ratio", "claims_ratio", "combined_ratio", "embedded_value_cr", "vnb_cr", "persistency_13m"],
        }

        for metric in key_metrics.get(sector, []):
            val = data.get(metric)
            status = "✅" if val else "❌"
            val_str = f"{val}%" if val and "cr" not in metric else (f"₹{val:,.0f} Cr" if val else "—")
            t.add_row(metric, val_str, status)

        t.add_row("source", data.get("source", "—")[:40], "ℹ️")
        t.add_row("date",   data.get("date", "—"), "ℹ️")
        console.print(t)
