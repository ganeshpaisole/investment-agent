"""
╔══════════════════════════════════════════════════════╗
║     MARKET SENTIMENT AGENT — NSE Intelligence        ║
║     FII/DII Flows | Put-Call Ratio | 52W Position    ║
║     Breadth | VIX | Sentiment Score                  ║
╚══════════════════════════════════════════════════════╝
"""

import time
import requests
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from nsepython import fnolist, nse_optionchain_scrapper
    HAS_NSE = True
except ImportError:
    HAS_NSE = False

# ─────────────────────────────────────────────────────────
# NSE PUBLIC API HEADERS (no auth needed for most endpoints)
# ─────────────────────────────────────────────────────────
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

NSE_SESSION = requests.Session()
NSE_SESSION.headers.update(NSE_HEADERS)

def _nse_get(url: str) -> dict:
    """Fetch NSE API with session cookies."""
    try:
        # Prime session
        NSE_SESSION.get("https://www.nseindia.com", timeout=10)
        time.sleep(0.5)
        resp = NSE_SESSION.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"NSE API error [{url}]: {e}")
        return {}


class MarketSentimentAgent:
    """
    Comprehensive market sentiment analysis for NSE.

    Covers:
      1. FII / DII cash flow activity
      2. Put-Call Ratio (PCR) from Nifty options chain
      3. India VIX (fear gauge)
      4. 52-week position of Nifty & individual stocks
      5. Market breadth (advances vs declines)
      6. Composite Sentiment Score (0-100)
    """

    def __init__(self, ticker: str = "NIFTY"):
        self.ticker = ticker.upper().replace(".NS", "")
        self.nse_ticker = f"{self.ticker}.NS" if self.ticker not in ("NIFTY", "SENSEX") else "^NSEI"

    # ─────────────────────────────────────────────────────
    # 1. FII / DII FLOWS
    # ─────────────────────────────────────────────────────
    def get_fii_dii_flows(self) -> dict:
        """
        Fetch FII/DII cash segment activity from NSE.
        Returns net buy/sell for last 5 trading days.
        """
        result = {
            "fii_net_cr": 0, "dii_net_cr": 0,
            "fii_5d_trend": "NEUTRAL", "dii_5d_trend": "NEUTRAL",
            "institutional_flow": "NEUTRAL", "status": "estimated"
        }

        try:
            url = "https://www.nseindia.com/api/fiidiiTradeReact"
            data = _nse_get(url)

            if data and isinstance(data, list) and len(data) > 0:
                fii_totals, dii_totals = [], []

                for day in data[:5]:  # Last 5 days
                    try:
                        fii_net = float(str(day.get("fiiNet", "0")).replace(",", ""))
                        dii_net = float(str(day.get("diiNet", "0")).replace(",", ""))
                        fii_totals.append(fii_net)
                        dii_totals.append(dii_net)
                    except Exception:
                        pass

                if fii_totals:
                    result["fii_net_cr"]    = round(sum(fii_totals), 0)
                    result["fii_daily_avg"] = round(sum(fii_totals) / len(fii_totals), 0)
                    result["fii_5d_trend"]  = "BUYING" if sum(fii_totals) > 0 else "SELLING"
                    result["fii_5d_data"]   = fii_totals

                if dii_totals:
                    result["dii_net_cr"]    = round(sum(dii_totals), 0)
                    result["dii_daily_avg"] = round(sum(dii_totals) / len(dii_totals), 0)
                    result["dii_5d_trend"]  = "BUYING" if sum(dii_totals) > 0 else "SELLING"

                result["status"] = "live"

            # Institutional consensus
            fii = result["fii_net_cr"]
            dii = result["dii_net_cr"]
            if fii > 500 and dii > 0:
                result["institutional_flow"] = "🟢 STRONG BULLISH"
            elif fii > 0 or dii > 500:
                result["institutional_flow"] = "🟢 BULLISH"
            elif fii < -500 and dii < 0:
                result["institutional_flow"] = "🔴 STRONG BEARISH"
            elif fii < 0:
                result["institutional_flow"] = "🟡 CAUTIOUS — FII SELLING"
            else:
                result["institutional_flow"] = "🟡 NEUTRAL"

        except Exception as e:
            logger.warning(f"FII/DII fetch failed: {e}")
            # Fallback: use yfinance proxy signals
            result["status"] = "estimated"

        return result

    # ─────────────────────────────────────────────────────
    # 2. PUT-CALL RATIO (PCR)
    # ─────────────────────────────────────────────────────
    def get_put_call_ratio(self) -> dict:
        """
        Fetch Nifty options chain and compute PCR.
        PCR > 1.2 = Oversold/Bullish contrarian
        PCR < 0.7 = Overbought/Bearish contrarian
        PCR 0.8–1.1 = Neutral
        """
        result = {
            "pcr_oi": 0, "pcr_volume": 0,
            "total_put_oi": 0, "total_call_oi": 0,
            "pcr_signal": "NEUTRAL", "status": "estimated"
        }

        try:
            url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
            data = _nse_get(url)

            if data and "filtered" in data:
                records = data["filtered"].get("data", [])
                total_call_oi = sum(r.get("CE", {}).get("openInterest", 0) for r in records)
                total_put_oi  = sum(r.get("PE", {}).get("openInterest", 0) for r in records)
                total_call_vol = sum(r.get("CE", {}).get("totalTradedVolume", 0) for r in records)
                total_put_vol  = sum(r.get("PE", {}).get("totalTradedVolume", 0) for r in records)

                pcr_oi  = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 1.0
                pcr_vol = round(total_put_vol / total_call_vol, 2) if total_call_vol > 0 else 1.0

                result.update({
                    "pcr_oi": pcr_oi,
                    "pcr_volume": pcr_vol,
                    "total_put_oi":  round(total_put_oi / 1e6, 2),
                    "total_call_oi": round(total_call_oi / 1e6, 2),
                    "status": "live"
                })

                # PCR signal (contrarian interpretation)
                if pcr_oi >= 1.3:
                    result["pcr_signal"] = "🟢 BULLISH (Oversold — Puts dominate)"
                elif pcr_oi >= 0.9:
                    result["pcr_signal"] = "🟡 NEUTRAL"
                elif pcr_oi >= 0.7:
                    result["pcr_signal"] = "🟠 MILDLY BEARISH (Calls dominate)"
                else:
                    result["pcr_signal"] = "🔴 BEARISH (Extreme call buying)"

        except Exception as e:
            logger.warning(f"PCR fetch failed: {e}")
            result["pcr_signal"] = "⚠️ Unavailable"

        return result

    # ─────────────────────────────────────────────────────
    # 3. INDIA VIX
    # ─────────────────────────────────────────────────────
    def get_india_vix(self) -> dict:
        """Fetch India VIX — NSE's fear index."""
        result = {"vix": 0, "vix_signal": "NEUTRAL", "vix_change_pct": 0}

        try:
            if HAS_YFINANCE:
                vix = yf.Ticker("^INDIAVIX")
                hist = vix.history(period="5d")
                if not hist.empty:
                    current_vix  = hist["Close"].iloc[-1]
                    prev_vix     = hist["Close"].iloc[-2] if len(hist) > 1 else current_vix
                    change_pct   = round((current_vix - prev_vix) / prev_vix * 100, 2)
                    result["vix"]            = round(current_vix, 2)
                    result["vix_change_pct"] = change_pct

                    if current_vix < 12:
                        result["vix_signal"] = "🟢 VERY LOW FEAR — Complacent market"
                    elif current_vix < 17:
                        result["vix_signal"] = "🟢 LOW FEAR — Stable market"
                    elif current_vix < 22:
                        result["vix_signal"] = "🟡 MODERATE FEAR — Normal volatility"
                    elif current_vix < 28:
                        result["vix_signal"] = "🟠 HIGH FEAR — Caution advised"
                    else:
                        result["vix_signal"] = "🔴 EXTREME FEAR — Potential opportunity"
        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")

        return result

    # ─────────────────────────────────────────────────────
    # 4. 52-WEEK POSITION
    # ─────────────────────────────────────────────────────
    def get_52w_position(self) -> dict:
        """
        Computes where the stock/index sits in its 52-week range.
        Also signals proximity to highs/lows.
        """
        result = {
            "current_price": 0, "high_52w": 0, "low_52w": 0,
            "position_pct": 0, "signal": "NEUTRAL"
        }

        try:
            if HAS_YFINANCE:
                ticker_sym = "^NSEI" if self.ticker == "NIFTY" else f"{self.ticker}.NS"
                hist = yf.Ticker(ticker_sym).history(period="1y")

                if not hist.empty:
                    current = hist["Close"].iloc[-1]
                    high_52 = hist["High"].max()
                    low_52  = hist["Low"].min()
                    pos_pct = round((current - low_52) / (high_52 - low_52) * 100, 1) if high_52 != low_52 else 50

                    result.update({
                        "current_price": round(current, 2),
                        "high_52w":      round(high_52, 2),
                        "low_52w":       round(low_52, 2),
                        "position_pct":  pos_pct,
                        "pct_from_high": round((current - high_52) / high_52 * 100, 1),
                        "pct_from_low":  round((current - low_52) / low_52 * 100, 1),
                    })

                    # Signal based on position
                    if pos_pct >= 90:
                        result["signal"] = "🔴 NEAR 52W HIGH — Caution (overbought zone)"
                    elif pos_pct >= 70:
                        result["signal"] = "🟡 UPPER RANGE — Momentum strong"
                    elif pos_pct >= 40:
                        result["signal"] = "🟢 MID RANGE — Healthy zone"
                    elif pos_pct >= 20:
                        result["signal"] = "🟢 LOWER RANGE — Potential value"
                    else:
                        result["signal"] = "🟢 NEAR 52W LOW — Deep value / high risk"

        except Exception as e:
            logger.warning(f"52W position fetch failed: {e}")

        return result

    # ─────────────────────────────────────────────────────
    # 5. MARKET BREADTH (Advances vs Declines)
    # ─────────────────────────────────────────────────────
    def get_market_breadth(self) -> dict:
        """
        Approximates market breadth using Nifty500 constituents
        or falls back to Nifty50 via yfinance.
        """
        result = {"advance_decline_ratio": 1.0, "breadth_signal": "NEUTRAL",
                  "advances": 0, "declines": 0}

        try:
            if HAS_YFINANCE:
                # Use Nifty50 stocks as proxy for breadth
                from nse_master_agent import NIFTY50_UNIVERSE
                advances, declines = 0, 0

                for ticker in list(NIFTY50_UNIVERSE.keys())[:20]:  # Sample 20 for speed
                    try:
                        hist = yf.Ticker(f"{ticker}.NS").history(period="2d")
                        if len(hist) >= 2:
                            if hist["Close"].iloc[-1] > hist["Close"].iloc[-2]:
                                advances += 1
                            else:
                                declines += 1
                        time.sleep(0.1)
                    except Exception:
                        pass

                total = advances + declines
                adr   = round(advances / declines, 2) if declines > 0 else 1.0
                result.update({
                    "advances": advances, "declines": declines,
                    "advance_decline_ratio": adr,
                    "breadth_pct": round(advances / total * 100, 1) if total > 0 else 50
                })

                if adr >= 2.5:   result["breadth_signal"] = "🟢 VERY STRONG BREADTH"
                elif adr >= 1.5: result["breadth_signal"] = "🟢 STRONG BREADTH"
                elif adr >= 0.8: result["breadth_signal"] = "🟡 NEUTRAL BREADTH"
                elif adr >= 0.5: result["breadth_signal"] = "🟠 WEAK BREADTH"
                else:            result["breadth_signal"] = "🔴 VERY WEAK BREADTH"

        except Exception as e:
            logger.warning(f"Breadth calculation failed: {e}")

        return result

    # ─────────────────────────────────────────────────────
    # 6. COMPOSITE SENTIMENT SCORE
    # ─────────────────────────────────────────────────────
    def compute_sentiment_score(self, fii: dict, pcr: dict, vix: dict, pos: dict) -> int:
        """
        Blends all sentiment signals into a 0-100 score.
        >65 = Bullish | 45-65 = Neutral | <45 = Bearish
        """
        score = 50  # Neutral base

        # FII/DII (30 pts)
        fii_net = fii.get("fii_net_cr", 0)
        dii_net = fii.get("dii_net_cr", 0)
        if fii_net > 2000:  score += 15
        elif fii_net > 500: score += 10
        elif fii_net > 0:   score += 5
        elif fii_net < -2000: score -= 15
        elif fii_net < -500:  score -= 10
        else:                 score -= 5

        if dii_net > 0: score += 5
        else:           score -= 5

        # PCR (25 pts) — contrarian
        pcr_val = pcr.get("pcr_oi", 1.0)
        if pcr_val >= 1.3:   score += 15   # Lots of puts = bottom signal
        elif pcr_val >= 1.0: score += 8
        elif pcr_val >= 0.8: score += 0    # Neutral
        elif pcr_val >= 0.6: score -= 8
        else:                score -= 15   # Too many calls = top signal

        # VIX (20 pts) — inverse
        vix_val = vix.get("vix", 18)
        if vix_val < 12:     score += 10   # Complacency (caution) or stability
        elif vix_val < 17:   score += 15   # Sweet spot
        elif vix_val < 22:   score += 5
        elif vix_val < 28:   score -= 10
        else:                score -= 15   # Panic = contrarian buy

        # 52W Position (25 pts) — mid-range best
        pos_pct = pos.get("position_pct", 50)
        if 30 <= pos_pct <= 70:  score += 15   # Healthy zone
        elif pos_pct < 20:       score += 10   # Deep value zone
        elif pos_pct > 85:       score -= 10   # Near top — caution
        else:                    score += 5

        return max(0, min(100, score))

    # ─────────────────────────────────────────────────────
    # MASTER ANALYZE
    # ─────────────────────────────────────────────────────
    def analyze(self) -> dict:
        """Run all sentiment modules and return unified result."""
        logger.info(f"📡 Market Sentiment Agent starting for {self.ticker}...")

        fii_data   = self.get_fii_dii_flows()
        pcr_data   = self.get_put_call_ratio()
        vix_data   = self.get_india_vix()
        pos_data   = self.get_52w_position()
        # breadth = self.get_market_breadth()  # Slower — uncomment if needed

        sentiment_score = self.compute_sentiment_score(fii_data, pcr_data, vix_data, pos_data)

        overall = (
            "🟢 BULLISH"          if sentiment_score >= 65 else
            "🟡 NEUTRAL"          if sentiment_score >= 45 else
            "🟠 CAUTIOUS"         if sentiment_score >= 30 else
            "🔴 BEARISH"
        )

        return {
            "ticker":           self.ticker,
            "sentiment_score":  sentiment_score,
            "overall_sentiment": overall,
            "fii_dii":          fii_data,
            "put_call_ratio":   pcr_data,
            "india_vix":        vix_data,
            "position_52w":     pos_data,
            "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


if __name__ == "__main__":
    agent = MarketSentimentAgent("NIFTY")
    result = agent.analyze()
    import json
    print(json.dumps(result, indent=2, default=str))
