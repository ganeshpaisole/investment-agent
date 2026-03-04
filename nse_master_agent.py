"""
NSE STOCK FORECAST AI AGENT - MASTER ORCHESTRATOR v3.0
360 Degree Investment Intelligence | Graham + Buffett Framework

ALL MODULES INTEGRATED:
  - Fundamental Agent (sector-aware scoring)
  - Valuation Engine (DCF + Graham + EPV + Relative)
  - Technical Agent (RSI, MACD, DMA, Volume, Bollinger)
  - Sentiment Agent (FII/DII, Put-Call Ratio, India VIX, 52W)
  - Pharma Module (FDA alerts, R&D intensity, pipeline)
  - PSU Module (government ownership corrections)
  - Peer Comparison (auto sector peers)
  - Macro Agent (INR, Crude, VIX)
  - AI Analyst (Claude-powered narrative)
  - Portfolio Screener (all Nifty50 ranked)

Usage:
  python nse_master_agent.py --ticker RELIANCE
  python nse_master_agent.py --sector BANKING
  python nse_master_agent.py --portfolio TCS,INFY,HDFCBANK
  python nse_master_agent.py --all-nifty50
  python nse_master_agent.py --sentiment
  python nse_master_agent.py --list-sectors

Install:
  pip install yfinance pandas numpy requests beautifulsoup4 loguru rich anthropic ta
"""

import os, sys, json, time, argparse, warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

warnings.filterwarnings("ignore")
console = Console()

try:    import yfinance as yf;  HAS_YFINANCE = True
except: HAS_YFINANCE = False;   logger.warning("pip install yfinance")

try:    import ta;               HAS_TA = True
except: HAS_TA = False;          logger.warning("pip install ta")

try:    import anthropic;        HAS_ANTHROPIC = True
except: HAS_ANTHROPIC = False;   logger.warning("pip install anthropic")

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
_NSE_SESSION = requests.Session()
_NSE_SESSION.headers.update(NSE_HEADERS)

def _nse_get(url):
    try:
        _NSE_SESSION.get("https://www.nseindia.com", timeout=8)
        time.sleep(0.4)
        r = _NSE_SESSION.get(url, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.debug(f"NSE API error: {e}")
        return {}

# ==============================================================================
# UNIVERSE
# ==============================================================================

NIFTY50_UNIVERSE = {
    "HDFCBANK":   {"sector":"BANKING",   "sub":"Private Bank"},
    "ICICIBANK":  {"sector":"BANKING",   "sub":"Private Bank"},
    "KOTAKBANK":  {"sector":"BANKING",   "sub":"Private Bank"},
    "AXISBANK":   {"sector":"BANKING",   "sub":"Private Bank"},
    "SBIN":       {"sector":"BANKING",   "sub":"PSU Bank",       "psu":True},
    "INDUSINDBK": {"sector":"BANKING",   "sub":"Private Bank"},
    "BAJFINANCE": {"sector":"NBFC",      "sub":"Consumer Finance"},
    "BAJAJFINSV": {"sector":"NBFC",      "sub":"Financial Holding"},
    "HDFCLIFE":   {"sector":"INSURANCE", "sub":"Life Insurance"},
    "SBILIFE":    {"sector":"INSURANCE", "sub":"Life Insurance"},
    "TCS":        {"sector":"IT",        "sub":"IT Services"},
    "INFY":       {"sector":"IT",        "sub":"IT Services"},
    "HCLTECH":    {"sector":"IT",        "sub":"IT Services"},
    "WIPRO":      {"sector":"IT",        "sub":"IT Services"},
    "TECHM":      {"sector":"IT",        "sub":"IT Services"},
    "LTIM":       {"sector":"IT",        "sub":"IT Services"},
    "SUNPHARMA":  {"sector":"PHARMA",    "sub":"Specialty Pharma"},
    "DRREDDY":    {"sector":"PHARMA",    "sub":"US Generic"},
    "CIPLA":      {"sector":"PHARMA",    "sub":"Branded Generic"},
    "DIVISLAB":   {"sector":"PHARMA",    "sub":"CDMO + API"},
    "APOLLOHOSP": {"sector":"PHARMA",    "sub":"Hospitals"},
    "HINDUNILVR": {"sector":"FMCG",      "sub":"FMCG"},
    "ITC":        {"sector":"FMCG",      "sub":"FMCG"},
    "NESTLEIND":  {"sector":"FMCG",      "sub":"FMCG"},
    "BRITANNIA":  {"sector":"FMCG",      "sub":"FMCG"},
    "TATACONSUM": {"sector":"FMCG",      "sub":"FMCG"},
    "ASIANPAINT": {"sector":"FMCG",      "sub":"Paints"},
    "TITAN":      {"sector":"FMCG",      "sub":"Consumer Goods"},
    "MARUTI":     {"sector":"AUTO",      "sub":"Passenger Vehicles"},
    "TATAMOTORS": {"sector":"AUTO",      "sub":"Commercial Vehicles"},
    "M&M":        {"sector":"AUTO",      "sub":"Utility Vehicles"},
    "BAJAJ-AUTO": {"sector":"AUTO",      "sub":"Two Wheelers"},
    "HEROMOTOCO": {"sector":"AUTO",      "sub":"Two Wheelers"},
    "EICHERMOT":  {"sector":"AUTO",      "sub":"Two Wheelers"},
    "RELIANCE":   {"sector":"ENERGY",    "sub":"Conglomerate"},
    "ONGC":       {"sector":"ENERGY",    "sub":"Oil & Gas",       "psu":True},
    "BPCL":       {"sector":"ENERGY",    "sub":"Oil Refining",    "psu":True},
    "COALINDIA":  {"sector":"ENERGY",    "sub":"Mining",          "psu":True},
    "NTPC":       {"sector":"UTILITY",   "sub":"Power Generation", "psu":True},
    "POWERGRID":  {"sector":"UTILITY",   "sub":"Power Transmission","psu":True},
    "TATASTEEL":  {"sector":"METALS",    "sub":"Steel"},
    "JSWSTEEL":   {"sector":"METALS",    "sub":"Steel"},
    "HINDALCO":   {"sector":"METALS",    "sub":"Aluminium"},
    "GRASIM":     {"sector":"METALS",    "sub":"Cement/Textiles"},
    "LT":         {"sector":"INFRA",     "sub":"Engineering"},
    "ADANIENT":   {"sector":"INFRA",     "sub":"Ports/Energy"},
    "ADANIPORTS": {"sector":"INFRA",     "sub":"Ports"},
    "BHARTIARTL": {"sector":"TELECOM",   "sub":"Telecom"},
    "SHREECEM":   {"sector":"CEMENT",    "sub":"Cement"},
    "ULTRACEMCO": {"sector":"CEMENT",    "sub":"Cement"},
}

SECTORS = {}
for t, v in NIFTY50_UNIVERSE.items():
    SECTORS.setdefault(v["sector"], []).append(t)

PSU_REGISTRY = {
    "SBIN":      {"govt_stake":57.5, "discount":25, "div_risk":"LOW",
                  "moat":"Largest bank network in India"},
    "ONGC":      {"govt_stake":58.9, "discount":35, "div_risk":"LOW",
                  "moat":"Largest upstream oil/gas company"},
    "BPCL":      {"govt_stake":52.9, "discount":30, "div_risk":"MEDIUM",
                  "moat":"Refining + fuel retail network"},
    "COALINDIA": {"govt_stake":63.1, "discount":20, "div_risk":"LOW",
                  "moat":"Near-monopoly in Indian coal"},
    "NTPC":      {"govt_stake":51.1, "discount":20, "div_risk":"LOW",
                  "moat":"Largest power generator, regulated ROE"},
    "POWERGRID": {"govt_stake":51.3, "discount":15, "div_risk":"LOW",
                  "moat":"Monopoly power transmission"},
}

FDA_RISK = {
    "SUNPHARMA": {"wl":2, "obs":5, "alerts":1, "last":"2023"},
    "DRREDDY":   {"wl":1, "obs":3, "alerts":0, "last":"2022"},
    "CIPLA":     {"wl":0, "obs":2, "alerts":0, "last":"2021"},
    "DIVISLAB":  {"wl":1, "obs":4, "alerts":0, "last":"2022"},
    "APOLLOHOSP":{"wl":0, "obs":0, "alerts":0, "last":"N/A"},
}

SCORE_WEIGHTS = {
    "fundamental":0.30,"valuation":0.25,"management":0.20,
    "technical":0.10,"sentiment":0.08,"peer":0.05,"macro":0.02,
}

SECTOR_PE   = {"IT":28,"PHARMA":30,"FMCG":45,"BANKING":14,"INSURANCE":25,
               "AUTO":18,"ENERGY":12,"METALS":10,"NBFC":20,"UTILITY":16,"DEFAULT":20}
SECTOR_WACC = {"IT":11,"PHARMA":12,"FMCG":11,"BANKING":13,"ENERGY":12,
               "METALS":13,"AUTO":12,"DEFAULT":12}

# ==============================================================================
# MODULE 1: DATA FETCHER
# ==============================================================================

class DataFetcher:
    def __init__(self, ticker):
        self.ticker = ticker.upper().replace(".NS","")
        self.sym    = f"{self.ticker}.NS"

    def get_all(self):
        if not HAS_YFINANCE:
            return {"ticker":self.ticker,"info":{},"current_price":0}
        try:
            obj  = yf.Ticker(self.sym)
            info = obj.info or {}
            def safe(fn):
                try:
                    r = fn()
                    return r if r is not None and not (hasattr(r,"empty") and r.empty) else pd.DataFrame()
                except: return pd.DataFrame()

            return {
                "ticker":       self.ticker,
                "info":         info,
                "income":       safe(lambda: obj.financials),
                "balance":      safe(lambda: obj.balance_sheet),
                "cashflow":     safe(lambda: obj.cashflow),
                "history_1y":   obj.history(period="1y"),
                "current_price":info.get("currentPrice") or info.get("regularMarketPrice",0),
                "screener":     self._screener(),
            }
        except Exception as e:
            logger.warning(f"DataFetcher {self.ticker}: {e}")
            return {"ticker":self.ticker,"info":{},"current_price":0}

    def _screener(self):
        result = {}
        try:
            from bs4 import BeautifulSoup
            url  = f"https://www.screener.in/company/{self.ticker}/consolidated/"
            resp = requests.get(url, headers={"User-Agent":NSE_HEADERS["User-Agent"]}, timeout=12)
            soup = BeautifulSoup(resp.text,"html.parser")
            div  = soup.find(id="top-ratios")
            if div:
                for li in div.find_all("li"):
                    n = li.find("span",class_="name")
                    v = li.find("span",class_="nowrap")
                    if n and v:
                        key = n.get_text(strip=True).lower().replace(" ","_")
                        val = v.get_text(strip=True).replace(",","").replace("%","").replace("₹","").strip()
                        try:    result[key] = float(val)
                        except: result[key] = val
            time.sleep(0.8)
        except Exception as e:
            logger.debug(f"Screener {self.ticker}: {e}")
        return result

# ==============================================================================
# MODULE 2: RATIO CALCULATOR
# ==============================================================================

class RatioCalc:
    def __init__(self, raw):
        self.info = raw.get("info",{})
        self.price = raw.get("current_price",0)

    def all(self):
        i = self.info
        return {
            "pe":            i.get("trailingPE",0) or 0,
            "pb":            i.get("priceToBook",0) or 0,
            "ev_ebitda":     min(i.get("enterpriseToEbitda",0) or 0, 100),
            "peg":           i.get("pegRatio",0) or 0,
            "eps":           i.get("trailingEps",0) or 0,
            "roe":           (i.get("returnOnEquity",0) or 0)*100,
            "roa":           (i.get("returnOnAssets",0) or 0)*100,
            "net_margin":    (i.get("profitMargins",0) or 0)*100,
            "ebitda_margin": (i.get("ebitdaMargins",0) or 0)*100,
            "de_ratio":      i.get("debtToEquity",0) or 0,
            "current_ratio": i.get("currentRatio",0) or 0,
            "revenue_growth":(i.get("revenueGrowth",0) or 0)*100,
            "earnings_growth":(i.get("earningsGrowth",0) or 0)*100,
            "div_yield":     (i.get("dividendYield",0) or 0)*100,
            "beta":          i.get("beta",1.0) or 1.0,
            "market_cap":    i.get("marketCap",0) or 0,
            "book_value":    i.get("bookValue",0) or 0,
            "fcf":           i.get("freeCashflow",0) or 0,
            "high_52w":      i.get("fiftyTwoWeekHigh",0) or 0,
            "low_52w":       i.get("fiftyTwoWeekLow",0) or 0,
            "shares":        i.get("sharesOutstanding",0) or 0,
            "total_revenue": i.get("totalRevenue",0) or 0,
        }

# ==============================================================================
# MODULE 3: FUNDAMENTAL AGENT
# ==============================================================================

class FundamentalAgent:
    SECTOR_ADJ = {
        "BANKING":  {"de_skip":True, "roe_min":15},
        "INSURANCE":{"de_skip":True, "roe_min":15},
        "PHARMA":   {"de_skip":False,"roe_min":18},
        "IT":       {"de_skip":False,"roe_min":20},
        "FMCG":     {"de_skip":False,"roe_min":20},
        "AUTO":     {"de_skip":False,"roe_min":15},
        "ENERGY":   {"de_skip":False,"roe_min":12},
        "METALS":   {"de_skip":False,"roe_min":12},
        "DEFAULT":  {"de_skip":False,"roe_min":15},
    }

    def __init__(self, ticker, ratios, sector):
        self.ticker = ticker
        self.r      = ratios
        self.sector = sector
        self.adj    = self.SECTOR_ADJ.get(sector, self.SECTOR_ADJ["DEFAULT"])

    def score(self):
        r = self.r; s = 0; flags = []

        # Earnings quality (25)
        if r["eps"] > 0:              s += 8;  flags.append("Positive EPS")
        eg = r["earnings_growth"]
        if eg >= 15:                  s += 10; flags.append(f"Strong earnings growth {eg:.1f}%")
        elif eg >= 8:                 s += 6;  flags.append(f"Good earnings growth {eg:.1f}%")
        elif eg >= 0:                 s += 3;  flags.append(f"Slow growth {eg:.1f}%")
        nm = r["net_margin"]
        if nm >= 15:                  s += 7;  flags.append(f"Strong net margin {nm:.1f}%")
        elif nm >= 8:                 s += 4;  flags.append(f"OK margin {nm:.1f}%")

        # Balance sheet (25)
        if not self.adj["de_skip"]:
            de = r["de_ratio"]
            if de < 0.5:              s += 12; flags.append(f"Low D/E {de:.2f}")
            elif de < 1.0:            s += 7;  flags.append(f"Moderate D/E {de:.2f}")
            elif de < 2.0:            s += 3;  flags.append(f"High D/E {de:.2f}")
            else:                              flags.append(f"Very high D/E {de:.2f}")
        else:
            s += 8; flags.append("D/E not applicable (Banking/Insurance)")
        cr = r["current_ratio"]
        if cr >= 2.0:                 s += 8;  flags.append(f"Strong current ratio {cr:.2f}")
        elif cr >= 1.0:               s += 5;  flags.append(f"OK current ratio {cr:.2f}")
        s += 5

        # Profitability (25)
        roe = r["roe"]; roe_min = self.adj["roe_min"]
        if roe >= roe_min * 1.5:      s += 12; flags.append(f"Excellent ROE {roe:.1f}%")
        elif roe >= roe_min:          s += 8;  flags.append(f"Good ROE {roe:.1f}%")
        elif roe >= roe_min * 0.7:    s += 4;  flags.append(f"Average ROE {roe:.1f}%")
        else:                                  flags.append(f"Weak ROE {roe:.1f}%")
        em = r["ebitda_margin"]
        if em >= 25:                  s += 8;  flags.append(f"Strong EBITDA margin {em:.1f}%")
        elif em >= 15:                s += 5;  flags.append(f"OK EBITDA margin {em:.1f}%")
        elif em >= 8:                 s += 2
        s += 5

        # Growth & dividends (25)
        rg = r["revenue_growth"]
        if rg >= 15:                  s += 12; flags.append(f"Strong revenue growth {rg:.1f}%")
        elif rg >= 8:                 s += 8;  flags.append(f"Good revenue growth {rg:.1f}%")
        elif rg >= 0:                 s += 4;  flags.append(f"Slow revenue growth {rg:.1f}%")
        else:                                  flags.append(f"Revenue declining {rg:.1f}%")
        dy = r["div_yield"]
        if dy > 2:                    s += 8;  flags.append(f"Good dividend {dy:.2f}%")
        elif dy > 0:                  s += 4;  flags.append(f"Dividend {dy:.2f}%")
        s += 5

        return min(s, 100), flags

# ==============================================================================
# MODULE 4: VALUATION ENGINE
# ==============================================================================

class ValuationEngine:
    def __init__(self, ticker, raw, ratios, sector):
        self.ticker = ticker
        self.raw    = raw
        self.r      = ratios
        self.sector = sector
        self.price  = raw.get("current_price",0)

    def dcf(self):
        try:
            fcf    = self.r.get("fcf",0)
            shares = self.r.get("shares",0)
            if fcf <= 0 or shares <= 0: return 0
            fcf_ps = fcf / shares
            g1     = min(max(self.r["earnings_growth"]/100, 0.05), 0.20)
            g2     = 0.065
            wacc   = SECTOR_WACC.get(self.sector,12) / 100
            pv     = sum(fcf_ps*(1+g1)**t/(1+wacc)**t for t in range(1,6))
            tv     = fcf_ps*(1+g1)**5*(1+g2)/(wacc-g2)/(1+wacc)**5
            return round(pv+tv, 2)
        except: return 0

    def graham(self):
        eps = self.r.get("eps",0); bv = self.r.get("book_value",0)
        return round((22.5*eps*bv)**0.5, 2) if eps > 0 and bv > 0 else 0

    def relative(self):
        eps = self.r.get("eps",0)
        return round(eps * SECTOR_PE.get(self.sector,20), 2) if eps > 0 else 0

    def epv(self):
        try:
            em  = self.r.get("ebitda_margin",0)/100
            rev = self.r.get("total_revenue",0)
            shares = self.r.get("shares",1)
            if rev <= 0 or em <= 0 or shares <= 0: return 0
            nopat = rev * em * 0.85 * 0.75
            return round(nopat / (SECTOR_WACC.get(self.sector,12)/100) / shares, 2)
        except: return 0

    def composite(self):
        d,g,e,rel = self.dcf(), self.graham(), self.epv(), self.relative()
        valid = [(n,v) for n,v in [("dcf",d),("graham",g),("epv",e),("relative",rel)] if v > 0]
        if not valid:
            return {"composite_iv":0,"dcf":0,"graham":0,"epv":0,"relative":0,
                    "upside_pct":0,"recommendation":"INSUFFICIENT DATA","valuation_score":40,
                    "entry_price":0,"target_price":0}
        w = {"dcf":0.40,"epv":0.25,"relative":0.25,"graham":0.10}
        iv, tw = 0, 0
        for n,v in valid:
            iv += w[n]*v; tw += w[n]
        comp_iv = round(iv/tw, 2) if tw > 0 else 0
        upside  = round((comp_iv - self.price)/self.price*100, 1) if self.price and comp_iv else 0
        if upside >= 30:    rec, vs = "STRONG BUY", 90
        elif upside >= 15:  rec, vs = "BUY",        75
        elif upside >= 0:   rec, vs = "FAIRLY VALUED",55
        elif upside >= -15: rec, vs = "HOLD",        40
        else:               rec, vs = "AVOID",        20
        return {
            "composite_iv":  comp_iv, "dcf":d, "graham":g, "epv":e, "relative":rel,
            "upside_pct":    upside,  "valuation_score":vs, "recommendation":rec,
            "entry_price":   round(comp_iv*0.70,2), "target_price":round(comp_iv*1.10,2),
        }

# ==============================================================================
# MODULE 5: TECHNICAL AGENT
# ==============================================================================

class TechnicalAgent:
    def __init__(self, ticker, history):
        self.ticker = ticker
        self.hist   = history

    def analyze(self):
        h = self.hist
        if h is None or h.empty or len(h) < 50:
            return {"technical_score":50,"signals":{},"status":"insufficient_data"}

        close  = h["Close"]; volume = h["Volume"]
        score  = 0; max_s = 0; signals = {}

        # Trend (40pts)
        sma50  = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(min(200,len(close))).mean().iloc[-1]
        curr   = close.iloc[-1]
        signals["price_vs_50dma"]  = "ABOVE" if curr > sma50 else "BELOW"
        signals["price_vs_200dma"] = "ABOVE" if curr > sma200 else "BELOW"
        signals["golden_cross"]    = bool(sma50 > sma200)
        if curr > sma50:   score += 10
        if curr > sma200:  score += 15
        if sma50 > sma200: score += 15
        max_s += 40

        h52 = close.max(); l52 = close.min()
        pos = round((curr-l52)/(h52-l52)*100,1) if h52 != l52 else 50
        signals.update({"52w_position_pct":pos,"52w_high":round(h52,2),"52w_low":round(l52,2)})

        # RSI (25pts)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = round((100 - 100/(1+gain/loss)).iloc[-1], 1)
        signals["rsi_14"] = rsi
        if rsi < 35:         score += 20
        elif rsi < 50:       score += 15
        elif rsi < 65:       score += 10
        elif rsi < 75:       score += 5
        max_s += 25

        # MACD (15pts)
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_bull = bool((ema12-ema26).iloc[-1] > 0)
        signals["macd_bullish"] = macd_bull
        if macd_bull: score += 15
        max_s += 15

        # Volume (10pts)
        last5c = close.tail(5); last5v = volume.tail(5)
        up_vol = last5v[last5c.diff()>0].mean()
        dn_vol = last5v[last5c.diff()<0].mean()
        acc = (pd.notna(up_vol) and pd.notna(dn_vol) and up_vol > dn_vol)
        signals["volume_trend"] = "ACCUMULATION" if acc else "DISTRIBUTION"
        if acc: score += 10
        max_s += 10

        # Bollinger
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_up = (sma20+2*std20).iloc[-1]; bb_dn = (sma20-2*std20).iloc[-1]
        bb_pos = round((curr-bb_dn)/(bb_up-bb_dn)*100,1) if bb_up != bb_dn else 50
        signals["bollinger_position_pct"] = bb_pos
        signals["support_1"]    = round(l52*1.05,2)
        signals["resistance_1"] = round(h52*0.95,2)
        signals["entry_zone"]   = f"Rs{round(curr*0.94,0)}-Rs{round(curr*0.98,0)}"
        signals["current_price"]= round(curr,2)

        tech_score = round((score/max_s)*100) if max_s > 0 else 50
        if rsi < 35 and macd_bull:   sig = "STRONG BUY SIGNAL"
        elif rsi < 45:               sig = "BUY SIGNAL (Oversold)"
        elif rsi > 72:               sig = "OVERBOUGHT - WAIT"
        elif signals["golden_cross"]:sig = "HOLD - UPTREND"
        else:                        sig = "NEUTRAL"
        signals["signal_label"] = sig

        return {"technical_score":tech_score,"signals":signals,"rsi":rsi,"status":"ok"}

# ==============================================================================
# MODULE 6: SENTIMENT AGENT (FII/DII, PCR, VIX, 52W)
# ==============================================================================

class SentimentAgent:
    def fii_dii(self):
        res = {"fii_net_cr":0,"dii_net_cr":0,"institutional_flow":"NEUTRAL","status":"estimated"}
        try:
            data = _nse_get("https://www.nseindia.com/api/fiidiiTradeReact")
            if data and isinstance(data,list):
                fii_t = [float(str(d.get("fiiNet","0")).replace(",","")) for d in data[:5] if d.get("fiiNet")]
                dii_t = [float(str(d.get("diiNet","0")).replace(",","")) for d in data[:5] if d.get("diiNet")]
                if fii_t: res["fii_net_cr"] = round(sum(fii_t),0); res["status"]="live"
                if dii_t: res["dii_net_cr"] = round(sum(dii_t),0)
                f,d = res["fii_net_cr"],res["dii_net_cr"]
                if f>500 and d>0:   res["institutional_flow"]="STRONG BULLISH"
                elif f>0 or d>500:  res["institutional_flow"]="BULLISH"
                elif f<-500 and d<0:res["institutional_flow"]="STRONG BEARISH"
                elif f<0:           res["institutional_flow"]="CAUTIOUS - FII SELLING"
        except Exception as e: logger.debug(f"FII/DII: {e}")
        return res

    def pcr(self):
        res = {"pcr_oi":1.0,"pcr_signal":"NEUTRAL","status":"estimated"}
        try:
            data = _nse_get("https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY")
            if data and "filtered" in data:
                recs = data["filtered"].get("data",[])
                call_oi = sum(r.get("CE",{}).get("openInterest",0) for r in recs)
                put_oi  = sum(r.get("PE",{}).get("openInterest",0) for r in recs)
                pcr_val = round(put_oi/call_oi,2) if call_oi > 0 else 1.0
                res["pcr_oi"] = pcr_val; res["status"]="live"
                if pcr_val >= 1.3:   res["pcr_signal"]="BULLISH (Oversold - Puts dominate)"
                elif pcr_val >= 0.9: res["pcr_signal"]="NEUTRAL"
                elif pcr_val >= 0.7: res["pcr_signal"]="MILDLY BEARISH"
                else:                res["pcr_signal"]="BEARISH (Call buying extreme)"
        except Exception as e: logger.debug(f"PCR: {e}")
        return res

    def vix(self):
        res = {"vix":18,"vix_signal":"NEUTRAL","vix_change_pct":0}
        if not HAS_YFINANCE: return res
        try:
            hist = yf.Ticker("^INDIAVIX").history(period="5d")
            if not hist.empty:
                v = hist["Close"].iloc[-1]; p = hist["Close"].iloc[-2] if len(hist)>1 else v
                res["vix"] = round(v,2); res["vix_change_pct"] = round((v-p)/p*100,2)
                if v < 12:   res["vix_signal"]="VERY LOW - Stable"
                elif v < 17: res["vix_signal"]="LOW FEAR - Healthy"
                elif v < 22: res["vix_signal"]="MODERATE FEAR"
                elif v < 28: res["vix_signal"]="HIGH FEAR - Caution"
                else:        res["vix_signal"]="EXTREME FEAR - Watch for bottom"
        except Exception as e: logger.debug(f"VIX: {e}")
        return res

    def nifty_52w(self):
        res = {"nifty_52w_pos":50,"nifty_signal":"NEUTRAL"}
        if not HAS_YFINANCE: return res
        try:
            hist = yf.Ticker("^NSEI").history(period="1y")
            if not hist.empty:
                c = hist["Close"].iloc[-1]; h = hist["High"].max(); l = hist["Low"].min()
                pos = round((c-l)/(h-l)*100,1) if h != l else 50
                res.update({"nifty_52w_pos":pos,"nifty_price":round(c,0),
                             "nifty_52w_high":round(h,0),"nifty_52w_low":round(l,0)})
                if pos>=85:   res["nifty_signal"]="NEAR 52W HIGH - Market expensive"
                elif pos>=60: res["nifty_signal"]="UPPER RANGE - Momentum ok"
                elif pos>=35: res["nifty_signal"]="MID RANGE - Healthy"
                else:         res["nifty_signal"]="NEAR 52W LOW - Deep value zone"
        except Exception as e: logger.debug(f"Nifty52W: {e}")
        return res

    def score(self, fii, pcr, vix, n52):
        s = 50
        fn = fii.get("fii_net_cr",0)
        s += (15 if fn>2000 else 8 if fn>500 else 4 if fn>0 else -8 if fn<-500 else -4)
        s += (5 if fii.get("dii_net_cr",0)>0 else -3)
        pv = pcr.get("pcr_oi",1.0)
        s += (12 if pv>=1.3 else 5 if pv>=0.9 else -5 if pv>=0.7 else -12)
        vv = vix.get("vix",18)
        s += (8 if vv<12 else 12 if vv<17 else 4 if vv<22 else -8 if vv<28 else -12)
        np_ = n52.get("nifty_52w_pos",50)
        s += (10 if 30<=np_<=65 else 8 if np_<30 else -8 if np_>80 else 0)
        return max(0,min(100,s))

    def analyze(self):
        logger.info("Fetching market sentiment...")
        fii = self.fii_dii(); p = self.pcr(); v = self.vix(); n = self.nifty_52w()
        s   = self.score(fii,p,v,n)
        overall = ("BULLISH" if s>=65 else "NEUTRAL" if s>=45 else "CAUTIOUS" if s>=30 else "BEARISH")
        return {"sentiment_score":s,"overall_sentiment":overall,"fii_dii":fii,
                "put_call_ratio":p,"india_vix":v,"nifty_52w":n,
                "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# ==============================================================================
# MODULE 7: PHARMA MODULE
# ==============================================================================

class PharmaModule:
    def score_fda(self, ticker):
        fda = FDA_RISK.get(ticker,{"wl":0,"obs":0,"alerts":0,"last":"N/A"})
        s = 20; flags = []
        s -= fda["wl"]*4; s -= min(fda["obs"],6); s -= fda["alerts"]*3
        flags.append(f"{'No FDA Warning Letters' if fda['wl']==0 else str(fda['wl'])+' FDA Warning Letter(s) last:'+fda['last']}")
        if fda["obs"]>3: flags.append(f"{fda['obs']} Form 483 observations")
        if fda["alerts"]>0: flags.append(f"{fda['alerts']} Import Alert(s)")
        try:
            if fda["last"]!="N/A" and datetime.now().year-int(fda["last"])>=3:
                s+=3; flags.append("3+ years since last FDA issue - recovering")
        except: pass
        grade = ("LOW RISK" if s>=18 else "MEDIUM RISK" if s>=12 else "HIGH RISK" if s>=6 else "CRITICAL RISK")
        return max(0,min(20,s)), flags, grade

    def score_rd(self, ratios):
        em = ratios.get("ebitda_margin",0); s = 0; flags = []
        if em >= 28:   s=20; flags.append(f"Premium EBITDA margin {em:.1f}% - strong R&D leverage")
        elif em >= 20: s=14; flags.append(f"Good EBITDA margin {em:.1f}%")
        elif em >= 14: s=8;  flags.append(f"Average EBITDA margin {em:.1f}%")
        else:          s=4;  flags.append(f"Low EBITDA margin {em:.1f}%")
        return s, flags

    def adjust(self, base, ticker, ratios):
        fda_s, fda_flags, fda_grade = self.score_fda(ticker)
        rd_s,  rd_flags             = self.score_rd(ratios)
        adj = min(100, round(base*0.60 + fda_s*1.25 + rd_s*0.75))
        return {"pharma_adjusted_score":adj,"fda_score":fda_s,"rd_score":rd_s,
                "fda_grade":fda_grade,"flags":fda_flags+rd_flags}

# ==============================================================================
# MODULE 8: PSU MODULE
# ==============================================================================

class PSUModule:
    def is_psu(self, ticker): return ticker in PSU_REGISTRY

    def adjust(self, ticker, scores, iv, price):
        if not self.is_psu(ticker):
            return {"scores":scores,"iv_adjusted":iv,"psu":False}
        p = PSU_REGISTRY[ticker]; adj = scores.copy(); logs = []
        vb = min(p["discount"]*0.3,10)
        adj["valuation"] = min(100,adj.get("valuation",50)+vb)
        logs.append(f"Valuation +{vb:.0f}pts (PSU discount = value opportunity)")
        adj["management"] = max(0,adj.get("management",50)-8)
        logs.append("Management -8pts (government interference risk)")
        adj["fundamental"] = min(100,adj.get("fundamental",50)+5)
        logs.append("Fundamental +5pts (dividend mandate + state backing)")
        if p["div_risk"]=="MEDIUM":
            adj["valuation"] = max(0,adj["valuation"]-8)
            logs.append("Valuation -8pts (divestment overhang)")
        iv_adj = round(iv*(1-p["discount"]/100),2) if iv>0 else 0
        mos    = round((iv_adj-price)/price*100,1) if price and iv_adj else 0
        return {"scores":adj,"iv_adjusted":iv_adj,"mos_adj":mos,
                "psu":True,"psu_info":p,"psu_logs":logs}

# ==============================================================================
# MODULE 9: PEER AGENT
# ==============================================================================

class PeerAgent:
    def __init__(self, ticker, sector):
        self.ticker = ticker
        self.peers  = [p for p in SECTORS.get(sector,[]) if p != ticker][:4]

    def analyze(self, my_ratios):
        if not HAS_YFINANCE or not self.peers:
            return {"peer_score":50,"verdict":"NO PEER DATA"}
        try:
            pdata = {}
            for p in self.peers:
                try:
                    info = yf.Ticker(f"{p}.NS").info
                    pdata[p] = {"pe":info.get("trailingPE",0) or 0,
                                "pb":info.get("priceToBook",0) or 0,
                                "roe":(info.get("returnOnEquity",0) or 0)*100}
                    time.sleep(0.2)
                except: pass
            if not pdata: return {"peer_score":50}
            avg_pe  = np.mean([v["pe"]  for v in pdata.values() if v["pe"]>0]  or [20])
            avg_pb  = np.mean([v["pb"]  for v in pdata.values() if v["pb"]>0]  or [3])
            avg_roe = np.mean([v["roe"] for v in pdata.values() if v["roe"]>0] or [15])
            my_pe=my_ratios.get("pe",0); my_pb=my_ratios.get("pb",0); my_roe=my_ratios.get("roe",0)
            s=50
            if my_pe>0 and my_pe<avg_pe: s+=15
            elif my_pe>avg_pe*1.2:       s-=10
            if my_pb>0 and my_pb<avg_pb: s+=15
            elif my_pb>avg_pb*1.2:       s-=10
            if my_roe>0 and my_roe>avg_roe: s+=20
            elif my_roe<avg_roe*0.7:        s-=10
            s = max(0,min(100,s))
            verdict = ("CHEAPER THAN PEERS" if s>=65 else "IN LINE WITH PEERS" if s>=45 else "EXPENSIVE VS PEERS")
            return {"peer_score":s,"peers":list(pdata.keys()),
                    "peer_averages":{"pe":round(avg_pe,1),"pb":round(avg_pb,2),"roe":round(avg_roe,1)},
                    "verdict":verdict}
        except Exception as e:
            logger.debug(f"Peer: {e}")
            return {"peer_score":50,"verdict":"PEER ERROR"}

# ==============================================================================
# MODULE 10: MACRO AGENT
# ==============================================================================

class MacroAgent:
    _cache = None

    def analyze(self):
        if MacroAgent._cache: return MacroAgent._cache
        s=60; sig={}
        if HAS_YFINANCE:
            try:
                inr=yf.Ticker("USDINR=X").fast_info.get("last_price",84)
                sig["usd_inr"]=round(inr,2)
                s+=(5 if inr<83 else -5 if inr>86 else 0)
            except: pass
            try:
                oil=yf.Ticker("BZ=F").fast_info.get("last_price",80)
                sig["brent_crude"]=round(oil,2)
                s+=(5 if oil<75 else -10 if oil>90 else 0)
            except: pass
            try:
                hist=yf.Ticker("^NSEI").history(period="30d")
                if not hist.empty:
                    vp=hist["Close"].pct_change().dropna().std()*(252**0.5)*100
                    sig["vix_proxy_pct"]=round(vp,1)
                    s+=(10 if vp<15 else -10 if vp>25 else 0)
            except: pass
        s=max(0,min(100,s))
        sig["macro_score"]=s
        sig["macro_env"]=("FAVORABLE" if s>=65 else "NEUTRAL" if s>=45 else "CAUTIOUS")
        MacroAgent._cache=sig
        return sig

# ==============================================================================
# MODULE 11: AI ANALYST
# ==============================================================================

class AIAnalyst:
    def __init__(self):
        self.client = anthropic.Anthropic() if HAS_ANTHROPIC else None

    def narrative(self, report):
        if not self.client:
            return "AI narrative unavailable - set ANTHROPIC_API_KEY and pip install anthropic"
        ticker=report.get("ticker",""); company=report.get("company_name",ticker)
        sector=report.get("sector",""); scores=report.get("scores",{})
        price=report.get("current_price",0); iv=report.get("intrinsic_value",0)
        upside=report.get("upside_pct",0); rec=report.get("recommendation","HOLD")
        psu_note = f"\nGovt Stake: {report.get('psu_info',{}).get('govt_stake','')}% | PSU Discount Applied" if report.get("psu") else ""
        pharma_note = f"\nFDA Grade: {report.get('fda_grade','')}" if sector=="PHARMA" else ""
        prompt = f"""You are a senior equity research analyst at a top Indian brokerage specializing in NSE/BSE markets.
Write a concise investment narrative following Graham and Buffett value investing principles.

Stock: {company} ({ticker}.NS)  |  Sector: {sector}{' (PSU)' if report.get('psu') else ''}
CMP: Rs{price:,.0f}  |  Intrinsic Value: Rs{iv:,.0f}  |  Upside: {upside}%  |  Signal: {rec}

Scores: Fundamental:{scores.get('fundamental','N/A')} | Valuation:{scores.get('valuation','N/A')} | Management:{scores.get('management','N/A')} | Technical:{scores.get('technical','N/A')} | Sentiment:{scores.get('sentiment','N/A')} | Composite:{scores.get('composite','N/A')}{psu_note}{pharma_note}

Write exactly 3 paragraphs:
1. Investment Thesis - moat, growth, India macro context (why buy or avoid)
2. Key Risks - 3 specific risks with numbers (e.g. NPA cycle, FDA warning, crude impact)
3. Final Verdict - entry strategy, price targets, time horizon 1-3 years

End with: WARNING: Research only - not SEBI registered investment advice."""
        try:
            resp = self.client.messages.create(model="claude-sonnet-4-6", max_tokens=700,
                                               messages=[{"role":"user","content":prompt}])
            return resp.content[0].text
        except Exception as e:
            return f"AI narrative error: {e}"

# ==============================================================================
# COMPOSITE SCORER
# ==============================================================================

def composite_score(scores):
    comp = round(sum(scores.get(k,50)*v for k,v in SCORE_WEIGHTS.items()), 1)
    if comp>=80:   rating,stars="STRONG BUY","5 Stars"
    elif comp>=65: rating,stars="BUY","4 Stars"
    elif comp>=55: rating,stars="HOLD","3 Stars"
    elif comp>=40: rating,stars="AVOID","2 Stars"
    else:          rating,stars="STRONG AVOID","1 Star"
    return {"composite_score":comp,"rating":rating,"stars":stars}

# ==============================================================================
# NSE MASTER ORCHESTRATOR v3.0
# ==============================================================================

class NSEMasterAgent:
    def __init__(self, use_ai=True, use_peers=True, use_sentiment=True):
        self.use_ai        = use_ai
        self.use_peers     = use_peers
        self.use_sentiment = use_sentiment
        self.ai            = AIAnalyst() if use_ai else None
        self.pharma        = PharmaModule()
        self.psu           = PSUModule()
        self.macro         = MacroAgent()
        self._sent_cache   = None

    def _sentiment(self):
        if self._sent_cache is None and self.use_sentiment:
            self._sent_cache = SentimentAgent().analyze()
        return self._sent_cache or {}

    # --------------------------------------------------------------------------
    def analyze_stock(self, ticker):
        ticker = ticker.upper().replace(".NS","")
        meta   = NIFTY50_UNIVERSE.get(ticker,{"sector":"UNKNOWN","sub":"Unknown"})
        sector = meta["sector"]; is_psu = meta.get("psu",False)

        console.print(Panel(
            f"[bold cyan]{ticker}[/] | Sector:[yellow]{sector}[/] | {meta['sub']}"
            f"{'  [red]PSU[/]' if is_psu else ''}",
            title="NSE Master Agent v3.0", border_style="cyan"))

        report = {"ticker":ticker,"sector":sector,"subsector":meta["sub"],
                  "psu":is_psu,"scores":{},"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        with Progress(SpinnerColumn(),TextColumn("[progress.description]{task.description}"),transient=True) as prog:
            task = prog.add_task("",total=None)

            # 1. Data
            prog.update(task, description="Fetching data...")
            raw    = DataFetcher(ticker).get_all()
            ratios = RatioCalc(raw).all()
            price  = raw.get("current_price",0)
            info   = raw.get("info",{})
            report["current_price"]  = price
            report["company_name"]   = info.get("longName",ticker)
            report["market_cap_cr"]  = round((info.get("marketCap",0) or 0)/1e7,0)

            # 2. Fundamental
            prog.update(task, description="Fundamental analysis...")
            fund_score, fund_flags = FundamentalAgent(ticker,ratios,sector).score()

            # Try existing nse_agent specialized agents
            mgmt_score = 60
            try:
                agent_root = Path(__file__).parent / "nse_agent"
                if str(agent_root) not in sys.path: sys.path.insert(0,str(agent_root))
                if sector in ("BANKING","NBFC"):
                    from agents.banking_agent import BankingAgent
                    ext = BankingAgent(ticker).analyze()
                    fund_score = ext.get("score",fund_score); mgmt_score = ext.get("management_score",60)
                elif sector == "INSURANCE":
                    from agents.insurance_agent import InsuranceAgent
                    ext = InsuranceAgent(ticker).analyze()
                    fund_score = ext.get("score",fund_score); mgmt_score = ext.get("management_score",60)
                else:
                    from agents.management_agent import ManagementAgent
                    ext = ManagementAgent(ticker).analyze(); mgmt_score = ext.get("score",60)
            except Exception as e:
                logger.debug(f"Existing agents: {e}")

            # Pharma adjustments
            if sector == "PHARMA":
                pharma_adj = self.pharma.adjust(fund_score,ticker,ratios)
                fund_score = pharma_adj["pharma_adjusted_score"]
                report["pharma"]    = pharma_adj
                report["fda_grade"] = pharma_adj["fda_grade"]

            report["scores"]["fundamental"] = fund_score
            report["scores"]["management"]  = mgmt_score
            report["fund_flags"]            = fund_flags

            # 3. Valuation
            prog.update(task, description="Valuation models...")
            val = ValuationEngine(ticker,raw,ratios,sector).composite()
            report.update(val)
            report["intrinsic_value"]    = val.get("composite_iv",0)
            report["recommendation"]     = val.get("recommendation","HOLD")
            report["upside_pct"]         = val.get("upside_pct",0)
            report["scores"]["valuation"]= val.get("valuation_score",50)

            # 4. PSU
            if is_psu:
                prog.update(task, description="PSU corrections...")
                psu_r = self.psu.adjust(ticker,report["scores"],
                                        iv=report["intrinsic_value"],price=price)
                report["scores"]          = psu_r["scores"]
                report["intrinsic_value"] = psu_r["iv_adjusted"]
                report["psu_info"]        = psu_r.get("psu_info",{})
                report["psu_logs"]        = psu_r.get("psu_logs",[])

            # 5. Technical
            prog.update(task, description="Technical analysis...")
            tech = TechnicalAgent(ticker,raw.get("history_1y",pd.DataFrame())).analyze()
            report["scores"]["technical"] = tech.get("technical_score",50)
            report["technical"]           = tech

            # 6. Sentiment
            prog.update(task, description="Market sentiment (FII/DII/PCR/VIX)...")
            sent = self._sentiment()
            report["scores"]["sentiment"] = sent.get("sentiment_score",50)
            report["sentiment"]           = sent

            # 7. Peers
            if self.use_peers:
                prog.update(task, description="Peer comparison...")
                peer = PeerAgent(ticker,sector).analyze(ratios)
                report["scores"]["peer"]  = peer.get("peer_score",50)
                report["peer"]            = peer

            # 8. Macro
            prog.update(task, description="Macro environment...")
            macro = self.macro.analyze()
            report["scores"]["macro"]     = macro.get("macro_score",60)
            report["macro"]               = macro

            # 9. Composite
            comp = composite_score(report["scores"])
            report.update(comp)

            # 10. AI Narrative
            if self.use_ai:
                prog.update(task, description="Generating AI narrative (Claude)...")
                report["ai_narrative"] = self.ai.narrative(report)

        self._print_report(report)
        return report

    # --------------------------------------------------------------------------
    def _print_report(self, r):
        price = r.get("current_price",0); iv = r.get("intrinsic_value",0)
        mos   = r.get("upside_pct",0)
        mos_c = "green" if mos>=30 else "yellow" if mos>=0 else "red"
        rating= r.get("rating","")
        border= "green" if "BUY" in rating else "yellow" if "HOLD" in rating else "red"

        console.print(Panel(
            f"[bold white]{r.get('company_name',r['ticker'])} ({r['ticker']})[/]"
            f"{'  PSU' if r.get('psu') else ''}\n"
            f"Sector:[cyan]{r.get('sector')}[/] - {r.get('subsector','')}\n\n"
            f"[yellow]CMP: Rs{price:,.2f}[/]  |  [green]IV: Rs{iv:,.2f}[/]  |  "
            f"[{mos_c}]MoS: {mos:+.1f}%[/]\n\n"
            f"[bold]{r.get('stars','')}  {rating}[/]  |  "
            f"Composite: [cyan]{r.get('composite_score',0)}/100[/]",
            title="NSE STOCK ANALYSIS REPORT v3.0", border_style=border))

        # Scores table
        t = Table(title="Score Breakdown",box=box.ROUNDED,header_style="bold cyan")
        t.add_column("Dimension",width=20); t.add_column("Score",width=10)
        t.add_column("Weight",width=8);    t.add_column("Bar",width=25)
        for label,key,weight in [
            ("Fundamental","fundamental","30%"),("Valuation","valuation","25%"),
            ("Management","management","20%"),("Technical","technical","10%"),
            ("Sentiment","sentiment","8%"),("Peer Ranking","peer","5%"),("Macro","macro","2%")]:
            s = r["scores"].get(key,50)
            c = "green" if s>=65 else "yellow" if s>=45 else "red"
            t.add_row(label,f"[{c}]{s}/100[/]",weight,f"[{c}]{'X'*(s//10)+'..'*(10-s//10)}[/]")
        console.print(t)

        # Valuation
        console.print(Panel(
            f"DCF:Rs{r.get('dcf',0):,.0f} | Graham:Rs{r.get('graham',0):,.0f} | "
            f"EPV:Rs{r.get('epv',0):,.0f} | Relative:Rs{r.get('relative',0):,.0f}\n"
            f"[bold]Composite IV: Rs{iv:,.0f}[/] | Entry:Rs{r.get('entry_price',0):,.0f} | "
            f"Target:Rs{r.get('target_price',0):,.0f} | Signal: {r.get('recommendation','')}",
            title="Valuation Models (DCF + Graham + EPV + Relative)",border_style="green"))

        # Technical
        tech = r.get("technical",{}).get("signals",{})
        if tech:
            console.print(Panel(
                f"RSI(14):{tech.get('rsi_14','N/A')} | "
                f"MACD:{'Bullish' if tech.get('macd_bullish') else 'Bearish'} | "
                f"50DMA:{tech.get('price_vs_50dma','N/A')} | "
                f"200DMA:{tech.get('price_vs_200dma','N/A')} | "
                f"Golden Cross:{'YES' if tech.get('golden_cross') else 'NO'}\n"
                f"Volume:{tech.get('volume_trend','N/A')} | "
                f"52W Pos:{tech.get('52w_position_pct','N/A')}% | "
                f"Entry Zone:{tech.get('entry_zone','N/A')}\n"
                f"Signal: [bold]{tech.get('signal_label','N/A')}[/]",
                title="Technical Analysis (RSI | MACD | DMA | Volume | Bollinger)",
                border_style="blue"))

        # Sentiment
        sent = r.get("sentiment",{})
        if sent:
            fii=sent.get("fii_dii",{}); vix=sent.get("india_vix",{})
            pcr=sent.get("put_call_ratio",{}); n52=sent.get("nifty_52w",{})
            console.print(Panel(
                f"Overall: [bold]{sent.get('overall_sentiment','N/A')}[/] | "
                f"Score:{sent.get('sentiment_score','N/A')}/100\n"
                f"FII 5D:Rs{fii.get('fii_net_cr',0):,.0f}Cr | "
                f"DII 5D:Rs{fii.get('dii_net_cr',0):,.0f}Cr | "
                f"Flow:{fii.get('institutional_flow','N/A')}\n"
                f"PCR:{pcr.get('pcr_oi','N/A')} - {pcr.get('pcr_signal','N/A')}\n"
                f"India VIX:{vix.get('vix','N/A')} - {vix.get('vix_signal','N/A')}\n"
                f"Nifty 52W Pos:{n52.get('nifty_52w_pos','N/A')}% - {n52.get('nifty_signal','N/A')}",
                title="Market Sentiment (FII/DII | Put-Call Ratio | India VIX | 52W)",
                border_style="magenta"))

        # Pharma
        if r.get("pharma"):
            ph=r["pharma"]
            console.print(Panel(
                f"FDA Grade:{r.get('fda_grade','N/A')} | R&D Score:{ph.get('rd_score',0)}/20 | "
                f"FDA Score:{ph.get('fda_score',0)}/20\n"+"\n".join(ph.get("flags",[])[:6]),
                title="Pharma - FDA Risk & R&D Intensity",border_style="magenta"))

        # PSU
        if r.get("psu") and r.get("psu_logs"):
            pi=r.get("psu_info",{})
            console.print(Panel(
                "\n".join(r["psu_logs"])+
                f"\nGovt Stake:{pi.get('govt_stake','')}% | "
                f"Discount:{pi.get('discount','')}% | Moat:{pi.get('moat','')}",
                title="PSU Adjustments (Govt Ownership Corrections)",border_style="red"))

        # Peer
        peer=r.get("peer",{})
        if peer:
            avgs=peer.get("peer_averages",{})
            console.print(Panel(
                f"Peers:{', '.join(peer.get('peers',[]))}\n"
                f"Sector Avg - P/E:{avgs.get('pe','N/A')}x | P/B:{avgs.get('pb','N/A')}x | "
                f"ROE:{avgs.get('roe','N/A')}%\nVerdict:[bold]{peer.get('verdict','N/A')}[/]",
                title="Peer Comparison",border_style="yellow"))

        # Macro
        macro=r.get("macro",{})
        if macro:
            console.print(Panel(
                f"USD/INR:Rs{macro.get('usd_inr','N/A')} | "
                f"Brent:${macro.get('brent_crude','N/A')} | "
                f"VIX Proxy:{macro.get('vix_proxy_pct','N/A')}%\n"
                f"Environment:[bold]{macro.get('macro_env','N/A')}[/]",
                title="Macro Environment",border_style="magenta"))

        # AI narrative
        if r.get("ai_narrative"):
            console.print(Panel(r["ai_narrative"],title="AI Analyst Narrative (Claude)",border_style="cyan"))

        console.print()

    # --------------------------------------------------------------------------
    def analyze_sector(self, sector):
        tickers = SECTORS.get(sector.upper(),[])
        if not tickers:
            console.print(f"[red]Unknown sector. Available:{list(SECTORS.keys())}[/]"); return []
        console.print(Panel(f"[bold]Sector Scan: {sector}[/] - {len(tickers)} stocks",border_style="cyan"))
        results = []
        for t in tickers:
            try: results.append(self.analyze_stock(t)); time.sleep(1.2)
            except Exception as e: logger.error(f"{t}: {e}")
        results.sort(key=lambda x: x.get("composite_score",0), reverse=True)
        self._leaderboard(sector, results); self._save(results, f"sector_{sector}")
        return results

    def analyze_portfolio(self, tickers):
        results = []
        for t in tickers:
            try: results.append(self.analyze_stock(t)); time.sleep(1)
            except Exception as e: logger.error(f"{t}: {e}")
        avg   = round(np.mean([r.get("composite_score",50) for r in results]),1)
        buys  = [r["ticker"] for r in results if "BUY" in r.get("rating","")]
        avoid = [r["ticker"] for r in results if "AVOID" in r.get("rating","")]
        console.print(Panel(
            f"Portfolio Score: [bold cyan]{avg}/100[/]\n"
            f"Buys:[green]{', '.join(buys) or 'None'}[/]\n"
            f"Avoid:[red]{', '.join(avoid) or 'None'}[/]",
            title="Portfolio Summary",border_style="cyan"))
        return {"results":results,"avg_score":avg,"buys":buys,"avoids":avoid}

    def analyze_all_nifty50(self):
        console.print(Panel("[bold]FULL NIFTY50 UNIVERSE SCAN[/]",border_style="bright_cyan"))
        all_r = []
        for t in NIFTY50_UNIVERSE.keys():
            try: all_r.append(self.analyze_stock(t)); time.sleep(1.5)
            except Exception as e: logger.error(f"{t}: {e}")
        all_r.sort(key=lambda x: x.get("composite_score",0), reverse=True)
        self._leaderboard("NIFTY50 FULL SCAN", all_r); self._save(all_r,"nifty50_full_scan")
        return all_r

    def sentiment_only(self):
        s=SentimentAgent().analyze()
        fii=s.get("fii_dii",{}); pcr=s.get("put_call_ratio",{})
        vix=s.get("india_vix",{}); n52=s.get("nifty_52w",{})
        console.print(Panel(
            f"Overall: [bold]{s.get('overall_sentiment')}[/] | Score:{s.get('sentiment_score')}/100\n\n"
            f"FII/DII (5-day):\n"
            f"  FII Net: Rs{fii.get('fii_net_cr',0):,.0f}Cr | DII Net: Rs{fii.get('dii_net_cr',0):,.0f}Cr\n"
            f"  Flow: {fii.get('institutional_flow')}\n\n"
            f"Options:\n  PCR:{pcr.get('pcr_oi','N/A')} | {pcr.get('pcr_signal')}\n\n"
            f"Volatility:\n  India VIX:{vix.get('vix','N/A')} ({vix.get('vix_change_pct',0):+.2f}%) - {vix.get('vix_signal')}\n\n"
            f"Nifty50:\n  Rs{n52.get('nifty_price','N/A'):,} | 52W Pos:{n52.get('nifty_52w_pos','N/A')}% - {n52.get('nifty_signal')}",
            title=f"Market Sentiment - {s.get('timestamp','')}",border_style="magenta"))
        return s

    def _leaderboard(self, title, results):
        t = Table(title=f"LEADERBOARD: {title}",box=box.ROUNDED)
        t.add_column("Rank",width=5); t.add_column("Ticker",width=12); t.add_column("Sector",width=12)
        t.add_column("Score",width=10); t.add_column("Rating",width=20); t.add_column("MoS%",width=10)
        t.add_column("CMP Rs",width=12); t.add_column("PSU",width=5)
        for i,r in enumerate(results[:20],1):
            s=r.get("composite_score",0)
            c="green" if s>=65 else "yellow" if s>=45 else "red"
            t.add_row(str(i),r.get("ticker",""),r.get("sector",""),
                      f"[{c}]{s}[/]",r.get("rating",""),
                      f"{r.get('upside_pct',0):+.1f}%",
                      f"Rs{r.get('current_price',0):,.0f}",
                      "PSU" if r.get("psu") else "")
        console.print(t)

    def _save(self, results, prefix):
        fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fname,"w") as f: json.dump(results,f,indent=2,default=str)
        console.print(f"[dim]Results saved to {fname}[/]")

# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="NSE Stock Forecast AI Agent v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nse_master_agent.py --ticker RELIANCE
  python nse_master_agent.py --ticker HDFCBANK --no-ai
  python nse_master_agent.py --sector BANKING
  python nse_master_agent.py --sector PHARMA
  python nse_master_agent.py --portfolio TCS,INFY,HDFCBANK,RELIANCE
  python nse_master_agent.py --all-nifty50
  python nse_master_agent.py --sentiment
  python nse_master_agent.py --list-sectors""")
    parser.add_argument("--ticker");      parser.add_argument("--sector")
    parser.add_argument("--portfolio");   parser.add_argument("--all-nifty50",action="store_true")
    parser.add_argument("--sentiment",action="store_true")
    parser.add_argument("--no-ai",   action="store_true"); parser.add_argument("--no-peers",action="store_true")
    parser.add_argument("--list-sectors", action="store_true")
    args = parser.parse_args()

    if args.list_sectors:
        for sec,tickers in sorted(SECTORS.items()):
            console.print(f"[cyan]{sec:12}[/]: {', '.join(tickers)}")
        return

    agent = NSEMasterAgent(use_ai=not args.no_ai, use_peers=not args.no_peers, use_sentiment=True)

    if args.sentiment:         agent.sentiment_only()
    elif args.ticker:          agent.analyze_stock(args.ticker)
    elif args.sector:          agent.analyze_sector(args.sector)
    elif args.portfolio:       agent.analyze_portfolio([t.strip().upper() for t in args.portfolio.split(",")])
    elif getattr(args,"all_nifty50",False): agent.analyze_all_nifty50()
    else:
        console.print(Panel(
            "[bold cyan]NSE Stock Forecast AI Agent v3.0[/]\n"
            "Commands: [ticker] | sector [S] | portfolio [T1,T2] | sentiment | quit",
            title="Welcome",border_style="cyan"))
        while True:
            try:
                cmd = input("\n> ").strip()
                if not cmd: continue
                if cmd.lower() in ("quit","exit","q"): break
                elif cmd.lower()=="sentiment": agent.sentiment_only()
                elif cmd.lower().startswith("sector "):
                    agent.analyze_sector(cmd.split(" ",1)[1].strip().upper())
                elif cmd.lower().startswith("portfolio "):
                    agent.analyze_portfolio([t.strip().upper() for t in cmd.split(" ",1)[1].split(",")])
                else: agent.analyze_stock(cmd.upper())
            except KeyboardInterrupt: break

    console.print("[dim]All outputs are research only - not SEBI registered investment advice.[/]")

if __name__ == "__main__":
    main()
