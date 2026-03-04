# ============================================================
# utils/data_fetcher.py
# Responsible for fetching all raw financial data from:
#   1. Yahoo Finance (via yfinance) — prices + financials
#   2. Screener.in — deeper Indian market data
#
# Think of this as the "data collection" layer.
# All other agents use this to get their numbers.
# ============================================================

import yfinance as yf          # Stock data from Yahoo Finance
import pandas as pd            # For data tables
import numpy as np             # For math operations
import requests                # For HTTP calls to Screener.in
from bs4 import BeautifulSoup  # For parsing HTML from websites
import re                      # For regex parsing of text data
import time                    # For delays between requests
import os
import json
from pathlib import Path
from loguru import logger      # For clean logging messages
from config.settings import NSE_SUFFIX


class NSEDataFetcher:
    """
    Fetches all financial data for a given NSE stock ticker.

    Usage:
        fetcher = NSEDataFetcher("RELIANCE")
        data = fetcher.get_all_data()
    """

    def __init__(self, ticker: str):
        """
        Initialize with the NSE ticker symbol.

        Args:
            ticker: NSE stock symbol WITHOUT .NS suffix
                    Example: "RELIANCE", "TCS", "INFY", "HDFC"
        """
        # Store the original ticker (e.g., "RELIANCE")
        self.ticker = ticker.upper().strip()

        # Yahoo Finance needs ".NS" for NSE stocks
        self.yf_ticker = self.ticker + NSE_SUFFIX  # e.g., "RELIANCE.NS"

        # Create the yfinance Ticker object — this is our data connection
        self.stock = yf.Ticker(self.yf_ticker)

        logger.info(f"📡 Initialized data fetcher for {self.ticker}")
        # Ensure cache directory exists
        self._ensure_cache_dir()

    # ----------------------------------------------------------
    # SECTION 1: PRICE DATA
    # ----------------------------------------------------------

    def get_current_price(self) -> float:
        """
        Get the current market price of the stock.

        Returns:
            Current price in INR (₹)
        """
        # Try yfinance once; if it fails with rate-limit or returns no price,
        # immediately use Screener.in as a reliable fallback to avoid long waits.
        try:
            info = self.stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            if price and float(price) > 0:
                logger.info(f"💰 Current price of {self.ticker}: ₹{price:,.2f}")
                return float(price)
            else:
                logger.warning("⚠️ yfinance returned empty price data, will try Screener.in fallback")
        except Exception as e:
            msg = str(e)
            logger.warning(f"❌ yfinance price fetch error: {msg}")
            # If this looks like a rate-limit, prefer Screener.in immediately
            if "Too Many Requests" in msg or "429" in msg:
                try:
                    screener = self.get_screener_data()
                    sp = screener.get("current_price")
                    if sp:
                        logger.info(f"💡 Using Screener.in price for {self.ticker}: ₹{sp:,.2f}")
                        return float(sp)
                except Exception:
                    logger.warning("⚠️ Screener fallback also failed")

        # Final fallbacks: cached price or Screener.in if not tried yet
        cache = self._load_cache()
        cache_price = cache.get("current_price") if cache else 0.0
        if cache_price:
            logger.info(f"💾 Loaded cached price for {self.ticker}: ₹{cache_price:,.2f}")
            return float(cache_price)

        try:
            # Try Screener.in if yfinance didn't provide price earlier
            screener = self.get_screener_data()
            sp = screener.get("current_price")
            if sp:
                logger.info(f"💡 Using Screener.in price for {self.ticker}: ₹{sp:,.2f}")
                return float(sp)
        except Exception:
            pass

        logger.error(f"❌ Could not fetch price for {self.ticker}")
        return 0.0

    def get_price_history(self, period: str = "10y") -> pd.DataFrame:
        """
        Get historical price data.

        Args:
            period: Time period — "1y", "3y", "5y", "10y"

        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume
        """
        # Prefer cached history first to avoid yfinance network issues
        csv_path = self._cache_path("history.csv")
        if os.path.exists(csv_path):
            try:
                history = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                logger.info(f"💾 Loaded cached history ({len(history)} rows)")
                return history
            except Exception:
                logger.warning("⚠️ Failed to read cached history, will attempt live fetch")

        # Respect environment toggle to disable yfinance when rate-limited
        use_yf = os.environ.get("USE_YFINANCE", "true").lower() != "false"
        if not use_yf:
            logger.info("⚠️ USE_YFINANCE=false — skipping live price history fetch")
            return pd.DataFrame()

        # If no cache available, attempt a single, quick fetch (no retries)
        try:
            history = self.stock.history(period=period)
            if history is None or history.empty:
                logger.warning("⚠️ yfinance returned empty history")
            else:
                logger.info(f"📈 Fetched {len(history)} days of price history")
                try:
                    history.to_csv(self._cache_path("history.csv"))
                except Exception:
                    pass
                return history
        except Exception as e:
            logger.warning(f"❌ Live price history fetch failed: {e}")

        # As a final fallback, return empty DataFrame rather than block
        logger.error("❌ Could not fetch price history — returning empty DataFrame")
        return pd.DataFrame()

    # ----------------------------------------------------------
    # SECTION 2: FINANCIAL STATEMENTS
    # ----------------------------------------------------------

    def get_income_statement(self) -> pd.DataFrame:
        """
        Get the Profit & Loss statement (annual).
        Contains: Revenue, Gross Profit, EBITDA, Net Income, EPS

        Returns:
            DataFrame with years as columns, metrics as rows
        """
        try:
            # .financials gives annual P&L data
            income = self.stock.financials
            logger.info(f"📊 Income statement fetched — {income.shape[1]} years of data")
            return income
        except Exception as e:
            logger.error(f"❌ Income statement fetch failed: {e}")
            return pd.DataFrame()

    def get_balance_sheet(self) -> pd.DataFrame:
        """
        Get the Balance Sheet (annual).
        Contains: Assets, Liabilities, Equity, Debt, Cash

        Returns:
            DataFrame with balance sheet line items
        """
        try:
            balance = self.stock.balance_sheet
            logger.info(f"🏦 Balance sheet fetched — {balance.shape[1]} years of data")
            return balance
        except Exception as e:
            logger.error(f"❌ Balance sheet fetch failed: {e}")
            return pd.DataFrame()

    def get_cash_flow(self) -> pd.DataFrame:
        """
        Get the Cash Flow Statement (annual).
        Contains: Operating CF, Investing CF, Financing CF, Free Cash Flow

        Returns:
            DataFrame with cash flow items
        """
        try:
            cashflow = self.stock.cashflow
            logger.info(f"💸 Cash flow statement fetched")
            return cashflow
        except Exception as e:
            logger.error(f"❌ Cash flow fetch failed: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------
    # SECTION 3: KEY INFO (one-stop shop for ratios)
    # ----------------------------------------------------------

    def get_stock_info(self) -> dict:
        """
        Get a comprehensive dictionary of stock info including:
        - Market cap, P/E, P/B, ROE, Debt/Equity
        - Company description, sector, industry
        - 52-week high/low, beta

        Returns:
            Dictionary with 100+ data points
        """
        # Allow disabling yfinance (useful when rate-limited)
        use_yf = os.environ.get("USE_YFINANCE", "true").lower() != "false"

        if use_yf:
            try:
                info = self.stock.info
                if info and len(info) > 0:
                    logger.info(f"ℹ️  Stock info fetched — {len(info)} data points")
                    # save some to cache
                    cache = self._load_cache() or {}
                    cache["info"] = info
                    self._save_cache(cache)
                    return info
                else:
                    logger.warning("⚠️ yfinance returned empty info dict")
            except Exception as e:
                logger.warning(f"❌ yfinance stock info fetch error: {e}")

        # Fallback: load from cache or Screener.in
        cache = self._load_cache()
        if cache and cache.get("info"):
            logger.info(f"💾 Loaded cached info for {self.ticker}")
            return cache.get("info")

        try:
            screener = self.get_screener_data()
            # Map screener fields into a minimal info dict
            info = {}
            if screener.get("market_cap_cr"):
                info["marketCap"] = float(screener.get("market_cap_cr")) * 1e7
            if screener.get("current_price"):
                info["currentPrice"] = float(screener.get("current_price"))
            if screener.get("pe"):
                info["trailingPE"] = screener.get("pe")
            if screener.get("roe"):
                info["returnOnEquity"] = screener.get("roe")
            if info:
                logger.info(f"💡 Built minimal info dict from Screener.in — {len(info)} keys")
                cache = cache or {}
                cache["info"] = info
                self._save_cache(cache)
                return info
        except Exception:
            logger.warning("⚠️ Screener.in info fallback failed")

        logger.error("❌ Stock info fetch failed and no cache available")
        return {}

    # ----------------------------------------------------------
    # SECTION 4: SCREENER.IN DATA (deeper Indian market data)
    # ----------------------------------------------------------

    def get_screener_data(self) -> dict:
        """
        Scrape key financial data from Screener.in (v2.0 — fixed selectors).

        Changes from v1:
        - ul#top-ratios instead of div#top-ratios (Screener changed the tag)
        - Now extracts 10-year P&L, balance sheet, ratios tables
        - Extracts shareholding (promoter %, FII %, pledge %)
        - Extracts banking data (NPA, CASA, CAR) from text

        Returns:
            Dictionary with all available financial data
        """
        logger.info(f"🌐 Scraping Screener.in for {self.ticker}...")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IN,en;q=0.9",
            "Referer": "https://www.screener.in/",
        }

        soup = None
        for url in [
            f"https://www.screener.in/company/{self.ticker}/consolidated/",
            f"https://www.screener.in/company/{self.ticker}/",
        ]:
            try:
                resp = requests.get(url, headers=headers, timeout=20)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.content, "lxml")
                    break
                else:
                    logger.warning(f"⚠️ Screener.in status {resp.status_code}")
            except requests.exceptions.Timeout:
                logger.warning("⚠️ Screener.in request timed out")
            except Exception as e:
                logger.warning(f"⚠️ Screener.in request error: {e}")

        if soup is None:
            logger.warning("⚠️ Screener.in not accessible")
            return {}

        result = {}

        def clean_num(text):
            text = str(text).strip()
            text = re.sub(r"[₹,\xa0]", "", text)
            text = text.replace("Cr.", "").replace("Cr", "").strip()
            try:
                return float(text)
            except:
                return 0.0

        # ── BLOCK 1: ul#top-ratios ────────────────────────────────
        # KEY FIX: Screener changed <div id="top-ratios"> to <ul id="top-ratios">
        top_ul = soup.find("ul", {"id": "top-ratios"})
        if top_ul:
            for li in top_ul.find_all("li"):
                name_tag = li.find("span", class_="name")
                name = name_tag.get_text(strip=True).lower() if name_tag else ""

                # Value is inside span.number within span.value
                val_span = li.find("span", class_="value")
                num_span = li.find("span", class_="number")
                raw = num_span.get_text(strip=True) if num_span else (
                      val_span.get_text(strip=True) if val_span else "")

                val = clean_num(raw)

                if "market cap" in name:
                    result["market_cap_cr"] = val
                elif "current price" in name:
                    result["current_price"] = val
                elif "high / low" in name or "52" in name:
                    parts = re.sub(r"[,\s]", "", raw).split("/")
                    if len(parts) == 2:
                        result["high_52w"] = clean_num(parts[0])
                        result["low_52w"]  = clean_num(parts[1])
                elif name.startswith("p/e") or name == "pe":
                    result["pe"] = val
                elif "book value" in name:
                    result["book_value"] = val
                elif "dividend yield" in name:
                    result["dividend_yield"] = val
                elif "roce" in name:
                    result["roce"] = val
                elif name == "roe":
                    result["roe"] = val
                elif "face value" in name:
                    result["face_value"] = val

        # ── BLOCK 2: 10-year ratios table ────────────────────────
        ratios_sec = soup.find("section", {"id": "ratios"})
        if ratios_sec:
            table = ratios_sec.find("table", class_="data-table")
            if table:
                for row in table.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 2:
                        continue
                    label = cells[0].get_text(strip=True).lower()
                    val   = clean_num(cells[-1].get_text(strip=True))
                    if "roce" in label and not result.get("roce"):
                        result["roce"] = val
                    elif "roe" in label and not result.get("roe"):
                        result["roe"] = val
                    elif "debtor days" in label:
                        result["debtor_days"] = val
                    elif "dividend payout" in label:
                        result["dividend_payout"] = val
                    elif "working capital" in label:
                        result["working_capital_days"] = val

        # ── BLOCK 3: Profit & Loss table ─────────────────────────
        pl_sec = soup.find("section", {"id": "profit-loss"})
        if pl_sec:
            table = pl_sec.find("table", class_="data-table")
            if table:
                for row in table.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 2:
                        continue
                    label = cells[0].get_text(strip=True).lower()
                    val   = clean_num(cells[-1].get_text(strip=True))
                    if "sales" in label or "revenue" in label:
                        result["revenue_ttm_cr"] = val
                    elif "operating profit" in label and "%" not in label:
                        result["ebitda_ttm_cr"] = val
                    elif "opm" in label:
                        result["opm_pct"] = val
                    elif "net profit" in label:
                        result["net_profit_ttm_cr"] = val
                    elif label.strip() == "eps":
                        result["eps_ttm"] = val

        # ── BLOCK 4: Balance sheet table ─────────────────────────
        bs_sec = soup.find("section", {"id": "balance-sheet"})
        if bs_sec:
            table = bs_sec.find("table", class_="data-table")
            if table:
                for row in table.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 2:
                        continue
                    label = cells[0].get_text(strip=True).lower()
                    val   = clean_num(cells[-1].get_text(strip=True))
                    if "borrowings" in label:
                        result["borrowings_cr"] = val
                    elif "total assets" in label:
                        result["total_assets_cr"] = val
                    elif "reserves" in label:
                        result["reserves_cr"] = val
                    elif "equity capital" in label:
                        result["equity_capital_cr"] = val

        # ── BLOCK 5: Shareholding table ───────────────────────────
        shp_sec = soup.find("section", {"id": "shareholding"})
        if shp_sec:
            table = shp_sec.find("table", class_="data-table")
            if table:
                for row in table.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 2:
                        continue
                    label = cells[0].get_text(strip=True).lower()
                    val   = clean_num(cells[-1].get_text(strip=True))
                    if "promoter" in label and "pledge" not in label:
                        result["promoter_holding"] = val
                    elif "fii" in label or "foreign" in label:
                        result["fii_holding"] = val
                    elif "dii" in label:
                        result["dii_holding"] = val
                    elif "pledge" in label:
                        result["promoter_pledge"] = val

        # ── BLOCK 6: Banking ratios from page text ────────────────
        full_text = soup.get_text()
        for pattern, key in [
            (r"(?:GNPA|Gross NPA)[^\d]*([\d.]+)\s*%", "gnpa"),
            (r"(?:NNPA|Net NPA)[^\d]*([\d.]+)\s*%",   "nnpa"),
            (r"CASA[^\d]*([\d.]+)\s*%",                "casa"),
            (r"(?:CAR|Capital Adequacy)[^\d]*([\d.]+)\s*%", "car"),
            (r"NIM[^\d]*([\d.]+)\s*%",                 "nim"),
        ]:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                result[key] = float(match.group(1))

        logger.info(f"✅ Screener.in: fetched {len(result)} ratios")
        time.sleep(1)
        return result

    # ----------------------------------------------------------
    # SECTION 5: MASTER DATA COLLECTION
    # ----------------------------------------------------------

    def get_all_data(self, sector_type: str = "GENERIC") -> dict:
        """
        Fetch ALL data in one call. This is the main entry point.
        Pass sector_type="BANKING"/"NBFC"/"INSURANCE" to also fetch BSE filing data.
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"🔍 Starting data collection for {self.ticker}")
        logger.info(f"{'='*50}")

        # Fetch Screener.in first — it often contains a quoted current price
        screener = self.get_screener_data()

        current_price = self.get_current_price()
        # If yfinance failed to return price, try Screener.in value
        if (not current_price or float(current_price) == 0) and screener.get("current_price"):
            try:
                current_price = float(screener.get("current_price"))
                logger.info(f"💡 Using Screener.in price for {self.ticker}: ₹{current_price:,.2f}")
            except Exception:
                pass

        data = {
            "ticker":           self.ticker,
            "current_price":    current_price,
            "info":             self.get_stock_info(),
            "income_statement": self.get_income_statement(),
            "balance_sheet":    self.get_balance_sheet(),
            "cash_flow":        self.get_cash_flow(),
            "price_history":    self.get_price_history(),
            "screener_data":    screener,
        }

        # BSE filing data — only for financial sector (has CASA/CAR/NIM/VNB)
        if sector_type in ("BANKING", "NBFC", "INSURANCE"):
            data["bse_data"] = self.get_bse_data(sector_type=sector_type)
        else:
            data["bse_data"] = {}

        logger.info(f"✅ Data collection complete for {self.ticker}")
        # Save to cache for future runs if we have useful data
        try:
            cache = {
                "ticker": self.ticker,
                "current_price": data.get("current_price", 0),
                "info": data.get("info", {}),
                "screener_data": data.get("screener_data", {}),
            }
            self._save_cache(cache)
        except Exception:
            pass

        return data

    # ----------------------------------------------------------
    # CACHE HELPERS
    # ----------------------------------------------------------
    def _ensure_cache_dir(self):
        try:
            base = Path(__file__).resolve().parents[1]
            self.cache_dir = os.path.join(base, "data_cache")
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception:
            self.cache_dir = os.path.join(os.getcwd(), "data_cache")
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, name: str) -> str:
        return os.path.join(self.cache_dir, f"{self.ticker}_{name}")

    def _load_cache(self) -> dict:
        path = os.path.join(self.cache_dir, f"{self.ticker}_cache.json")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_cache(self, data: dict):
        path = os.path.join(self.cache_dir, f"{self.ticker}_cache.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str)
        except Exception:
            pass

    def get_bse_data(self, sector_type: str = "BANKING") -> dict:
        """Fetch data from BSE quarterly filings — CASA, CAR, NIM, VNB, Solvency, AUM."""
        try:
            from utils.bse_parser import BSEFilingParser
            logger.info(f"🏛️  Fetching BSE filing data for {self.ticker}...")
            parser = BSEFilingParser(self.ticker)
            data   = parser.get_filing_data(sector_type=sector_type)
            n = len([k for k in data if k not in ("source", "date")])
            if n > 0:
                logger.info(f"✅ BSE filings: {n} metrics extracted")
            return data
        except Exception as e:
            logger.warning(f"⚠️ BSE parser failed for {self.ticker}: {e}")
            return {}
