"""
screener_scraper.py — Standalone Screener.in scraper (v2.0)
============================================================
Fixed based on diagnostic findings:
  - ul#top-ratios (was div#top-ratios — Screener changed the tag)
  - Extracts 10-year tables from section#profit-loss, section#ratios
  - Extracts shareholding from section#shareholding
  - Parses span.number within li items correctly

Run standalone to test:
    python screener_scraper.py HDFCBANK
    python screener_scraper.py TCS
    python screener_scraper.py BAJFINANCE
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import sys
from loguru import logger

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.screener.in/",
}


def _clean_number(text: str) -> str:
    """Remove commas, ₹, Cr, %, extra spaces from a number string."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"[₹,\xa0]", "", text)   # remove ₹ comma nbsp
    text = re.sub(r"\s+", " ", text).strip()
    # Handle "Cr." suffix
    text = text.replace("Cr.", "").replace("Cr", "").strip()
    return text


def _parse_number(text: str) -> float:
    """Parse a cleaned number string to float. Returns 0 on failure."""
    try:
        cleaned = _clean_number(text)
        if not cleaned or cleaned in ["-", "—", "N/A", ""]:
            return 0.0
        return float(cleaned)
    except:
        return 0.0


def scrape_screener(ticker: str) -> dict:
    """
    Scrape Screener.in for a given NSE ticker.

    Returns a flat dictionary with all available data:
    {
        "market_cap": 142116.0,
        "pe": 18.5,
        "book_value": 367.0,
        "roce": 15.2,
        "roe": 14.0,
        "dividend_yield": 1.2,
        "promoter_holding": 0.2,
        "fii_holding": 32.1,
        "dii_holding": 23.3,
        "promoter_pledge": 0.0,
        "gnpa": 1.24,          # banks only
        "nnpa": 0.31,          # banks only
        ...
    }
    """
    result = {}

    # Try consolidated first, then standalone
    urls = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/",
    ]

    soup = None
    for url in urls:
        try:
            logger.info(f"🌐 Screener.in → {url}")
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "lxml")
                logger.info(f"✅ Fetched successfully")
                break
            else:
                logger.warning(f"⚠️ Status {resp.status_code} for {url}")
        except Exception as e:
            logger.warning(f"⚠️ Request failed: {e}")

    if soup is None:
        logger.error("❌ Could not fetch Screener.in page")
        return {}

    # ── BLOCK 1: Top Ratios (ul#top-ratios) ──────────────────────
    # Structure: <ul id="top-ratios">
    #              <li>
    #                <span class="name">Market Cap</span>
    #                <span class="nowrap value">₹ <span class="number">14,21,166</span> Cr.</span>
    #              </li>
    # KEY FIX: It's a <ul> tag, NOT <div> tag

    top_ratios_ul = soup.find("ul", {"id": "top-ratios"})
    if top_ratios_ul:
        for li in top_ratios_ul.find_all("li"):
            # Get label
            name_tag = li.find("span", class_="name")
            if not name_tag:
                # Try direct text before value span
                name_tag = li.find("span")
            name = name_tag.get_text(strip=True) if name_tag else ""

            # Get value — look for span.number inside the value span
            value_span = li.find("span", class_="value")
            if value_span:
                number_span = value_span.find("span", class_="number")
                if number_span:
                    raw_val = number_span.get_text(strip=True)
                else:
                    raw_val = value_span.get_text(strip=True)
            else:
                # Fallback: any span.number in the li
                number_span = li.find("span", class_="number")
                raw_val = number_span.get_text(strip=True) if number_span else ""

            val_str = _clean_number(raw_val)
            val_num = _parse_number(raw_val)

            # Map label → result key
            name_lower = name.lower()
            if "market cap" in name_lower:
                result["market_cap_cr"] = val_num
            elif name_lower in ["current price", "price"]:
                result["current_price"] = val_num
            elif "high / low" in name_lower or "52 week" in name_lower:
                # Format: "3,767 / 2,579"
                parts = raw_val.replace(",", "").split("/")
                if len(parts) == 2:
                    result["high_52w"] = _parse_number(parts[0])
                    result["low_52w"]  = _parse_number(parts[1])
            elif name_lower.startswith("p/e") or name_lower == "pe":
                result["pe"] = val_num
            elif "book value" in name_lower:
                result["book_value"] = val_num
            elif "dividend yield" in name_lower:
                result["dividend_yield"] = val_num
            elif "roce" in name_lower:
                result["roce"] = val_num
            elif name_lower == "roe" or name_lower.startswith("return on equity"):
                result["roe"] = val_num
            elif "face value" in name_lower:
                result["face_value"] = val_num

        logger.info(f"✅ Top ratios: {list(result.keys())}")
    else:
        logger.warning("⚠️ ul#top-ratios not found")

    # ── BLOCK 2: 10-Year Ratios Table (section#ratios) ────────────
    # Contains: Debtor Days, Inventory, ROCE%, ROE%, etc. by year
    ratios_section = soup.find("section", {"id": "ratios"})
    if ratios_section:
        table = ratios_section.find("table", class_="data-table")
        if table:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue
                label = cells[0].get_text(strip=True).lower()
                # Get most recent year value (second column = oldest, last = newest)
                # Screener shows oldest to newest left to right
                recent_cell = cells[-1]  # last column = most recent TTM
                val = _parse_number(recent_cell.get_text(strip=True))

                if "roce" in label:
                    result["roce_10y_latest"] = val
                elif "roe" in label:
                    result["roe_10y_latest"] = val
                elif "debtor days" in label:
                    result["debtor_days"] = val
                elif "inventory days" in label or "inventory turnover" in label:
                    result["inventory_days"] = val
                elif "cash conversion" in label:
                    result["cash_conversion_cycle"] = val
                elif "working capital" in label:
                    result["working_capital_days"] = val
                elif "dividend payout" in label:
                    result["dividend_payout"] = val

        logger.info(f"✅ Ratios section parsed")

    # ── BLOCK 3: Profit & Loss Table (section#profit-loss) ────────
    # Contains: Sales, Expenses, OPM%, Net Profit, EPS by year
    pl_section = soup.find("section", {"id": "profit-loss"})
    if pl_section:
        table = pl_section.find("table", class_="data-table")
        if table:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue
                label = cells[0].get_text(strip=True).lower()
                recent_val = _parse_number(cells[-1].get_text(strip=True))

                if "sales" in label or "revenue" in label:
                    result["revenue_ttm_cr"] = recent_val
                elif "operating profit" in label or "opm" in label:
                    result["opm_pct"] = recent_val
                elif "net profit" in label:
                    result["net_profit_ttm_cr"] = recent_val
                elif label.strip() == "eps":
                    result["eps_ttm"] = recent_val
                elif "interest" in label and "coverage" not in label:
                    result["interest_expense_cr"] = recent_val
                elif "tax" in label and "%" not in label:
                    result["tax_cr"] = recent_val

        logger.info(f"✅ P&L section parsed")

    # ── BLOCK 4: Balance Sheet Table (section#balance-sheet) ──────
    bs_section = soup.find("section", {"id": "balance-sheet"})
    if bs_section:
        table = bs_section.find("table", class_="data-table")
        if table:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue
                label = cells[0].get_text(strip=True).lower()
                recent_val = _parse_number(cells[-1].get_text(strip=True))

                if "equity capital" in label:
                    result["equity_capital_cr"] = recent_val
                elif "reserves" in label:
                    result["reserves_cr"] = recent_val
                elif "borrowings" in label:
                    result["borrowings_cr"] = recent_val
                elif "total assets" in label or "total liabilities" in label:
                    result["total_assets_cr"] = recent_val
                elif "investments" in label:
                    result["investments_cr"] = recent_val
                elif "fixed assets" in label or "tangible" in label:
                    result["fixed_assets_cr"] = recent_val

        logger.info(f"✅ Balance sheet section parsed")

    # ── BLOCK 5: Shareholding (section#shareholding) ──────────────
    # Contains: Promoter %, FII %, DII %, Public %, Pledge %
    shp_section = soup.find("section", {"id": "shareholding"})
    if shp_section:
        # Quarterly table — we want the most recent quarter (last column)
        table = shp_section.find("table", class_="data-table")
        if table:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue
                label = cells[0].get_text(strip=True).lower()
                recent_val = _parse_number(cells[-1].get_text(strip=True))

                if "promoter" in label and "pledge" not in label:
                    result["promoter_holding"] = recent_val
                elif "fii" in label or "foreign" in label:
                    result["fii_holding"] = recent_val
                elif "dii" in label or "domestic inst" in label:
                    result["dii_holding"] = recent_val
                elif "public" in label and "promoter" not in label:
                    result["public_holding"] = recent_val
                elif "pledge" in label:
                    result["promoter_pledge"] = recent_val

        logger.info(f"✅ Shareholding section parsed")

    # ── BLOCK 6: Bank-specific data from analysis section ─────────
    # NPA, CASA, CAR often appear in text paragraphs or analysis section
    analysis_section = soup.find("section", {"id": "analysis"})
    if analysis_section:
        text = analysis_section.get_text()
        # Extract GNPA
        gnpa_match = re.search(r"(?:GNPA|Gross NPA)[^\d]*([\d.]+)\s*%", text, re.IGNORECASE)
        if gnpa_match:
            result["gnpa"] = float(gnpa_match.group(1))
        # Extract NNPA
        nnpa_match = re.search(r"(?:NNPA|Net NPA)[^\d]*([\d.]+)\s*%", text, re.IGNORECASE)
        if nnpa_match:
            result["nnpa"] = float(nnpa_match.group(1))
        # Extract CASA
        casa_match = re.search(r"CASA[^\d]*([\d.]+)\s*%", text, re.IGNORECASE)
        if casa_match:
            result["casa"] = float(casa_match.group(1))
        # Extract CAR
        car_match = re.search(r"(?:CAR|Capital Adequacy)[^\d]*([\d.]+)\s*%", text, re.IGNORECASE)
        if car_match:
            result["car"] = float(car_match.group(1))

    # ── BLOCK 7: Full page text search for banking ratios ─────────
    full_text = soup.get_text()
    # Attempt to find NIM (Net Interest Margin) in page text
    nim_match = re.search(r"NIM[^\d]*([\d.]+)\s*%", full_text, re.IGNORECASE)
    if nim_match and "nim" not in result:
        result["nim"] = float(nim_match.group(1))

    # ── BLOCK 8: Derive ROE/ROCE from 10-year table if missing ────
    # Use 10-year latest if top-ratios didn't yield them
    if result.get("roce") == 0 and result.get("roce_10y_latest", 0) > 0:
        result["roce"] = result["roce_10y_latest"]
    if result.get("roe") == 0 and result.get("roe_10y_latest", 0) > 0:
        result["roe"] = result["roe_10y_latest"]

    logger.info(f"✅ Screener.in: fetched {len(result)} data points for {ticker}")
    return result


# ============================================================
# STANDALONE TEST
# ============================================================
if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "HDFCBANK"

    print(f"\n{'='*60}")
    print(f"SCREENER.IN SCRAPER v2.0 — Testing: {ticker}")
    print(f"{'='*60}\n")

    data = scrape_screener(ticker)

    if data:
        print(f"✅ Successfully extracted {len(data)} data points:\n")
        # Print in categories
        categories = {
            "Valuation": ["market_cap_cr", "pe", "book_value", "current_price",
                          "high_52w", "low_52w", "face_value"],
            "Profitability": ["roce", "roe", "opm_pct", "dividend_yield",
                              "roce_10y_latest", "roe_10y_latest"],
            "Financials": ["revenue_ttm_cr", "net_profit_ttm_cr", "eps_ttm",
                           "total_assets_cr", "borrowings_cr", "reserves_cr",
                           "interest_expense_cr"],
            "Shareholding": ["promoter_holding", "fii_holding", "dii_holding",
                             "public_holding", "promoter_pledge"],
            "Banking": ["gnpa", "nnpa", "casa", "car", "nim"],
            "Efficiency": ["debtor_days", "inventory_days", "working_capital_days",
                           "dividend_payout"],
        }

        for cat, keys in categories.items():
            vals = {k: data[k] for k in keys if k in data}
            if vals:
                print(f"  📊 {cat}:")
                for k, v in vals.items():
                    print(f"     {k:35s} = {v}")
                print()
    else:
        print("❌ No data extracted")
        print("Check if Screener.in is accessible and HTML files were saved")
