# ============================================================
# agents/sentiment_agent.py — Market Sentiment Agent v1.0
# ============================================================
# Scores market sentiment from 5 dimensions:
#
#   1. Price Momentum      (30 pts) — 52-week position, trend, SMA
#   2. Volume & Breadth    (20 pts) — volume trend, accumulation/distribution
#   3. Technical Signals   (20 pts) — RSI, MACD, Bollinger Bands
#   4. Institutional Flow  (20 pts) — holding %, beta, relative strength vs Nifty
#   5. Valuation Context   (10 pts) — P/E vs sector avg, price vs 1Y mean
#
# Signal mapping:
#   80-100  🟢 STRONG BUY  — Momentum + institutions aligned
#   65-79   🟢 BUY         — Positive sentiment, accumulate on dips
#   50-64   🟡 NEUTRAL     — Mixed signals, wait for clarity
#   35-49   🟠 CAUTION     — Weak momentum, risk-off
#   0-34    🔴 AVOID       — Bearish momentum, avoid fresh positions
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger
from config.settings import NSE_SUFFIX


class MarketSentimentAgent:

    def __init__(self, ticker: str):
        self.ticker    = ticker.upper().strip()
        self.yf_ticker = self.ticker + NSE_SUFFIX
        self.stock     = yf.Ticker(self.yf_ticker)
        logger.info(f"📡 Sentiment Agent initialized for {self.ticker}")

    # ──────────────────────────────────────────────────────────
    # DATA
    # ──────────────────────────────────────────────────────────

    def _get_data(self) -> dict:
        info = {}
        hist_1y = pd.DataFrame()
        hist_3m = pd.DataFrame()
        try:
            info = self.stock.info
        except Exception as e:
            logger.warning(f"⚠️ Info fetch failed: {e}")
        try:
            hist_1y = self.stock.history(period="1y", interval="1d")
            logger.info(f"📈 1-year history: {len(hist_1y)} days")
        except Exception as e:
            logger.warning(f"⚠️ History fetch failed: {e}")
        hist_3m = hist_1y.tail(63) if not hist_1y.empty else pd.DataFrame()
        return {"info": info, "hist_1y": hist_1y, "hist_3m": hist_3m}

    # ──────────────────────────────────────────────────────────
    # TECHNICAL INDICATORS
    # ──────────────────────────────────────────────────────────

    def _rsi(self, prices: pd.Series, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        delta = prices.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return round(float((100 - 100 / (1 + rs)).iloc[-1]), 1)

    def _macd(self, prices: pd.Series) -> dict:
        if len(prices) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0, "bullish": False}
        ema12  = prices.ewm(span=12, adjust=False).mean()
        ema26  = prices.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        sig    = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - sig
        return {
            "macd":      round(float(macd.iloc[-1]), 2),
            "signal":    round(float(sig.iloc[-1]), 2),
            "histogram": round(float(hist.iloc[-1]), 2),
            "bullish":   bool(hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]),
        }

    def _bollinger(self, prices: pd.Series, period: int = 20) -> dict:
        if len(prices) < period:
            return {"pct_b": 0.5, "position": "middle", "upper": 0, "lower": 0, "middle": 0}
        sma   = prices.rolling(period).mean()
        std   = prices.rolling(period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        curr  = prices.iloc[-1]
        brange = float(upper.iloc[-1] - lower.iloc[-1])
        pct_b  = float((curr - lower.iloc[-1]) / brange) if brange > 0 else 0.5
        pos = "upper" if pct_b > 0.8 else "lower" if pct_b < 0.2 else "middle"
        return {
            "pct_b":    round(pct_b, 3),
            "position": pos,
            "upper":    round(float(upper.iloc[-1]), 2),
            "lower":    round(float(lower.iloc[-1]), 2),
            "middle":   round(float(sma.iloc[-1]), 2),
        }

    def _sma_signals(self, prices: pd.Series) -> dict:
        r = {"sma20": 0, "sma50": 0, "sma200": 0,
             "above_sma20": False, "above_sma50": False, "above_sma200": False,
             "golden_cross": False, "death_cross": False}
        curr = float(prices.iloc[-1])
        if len(prices) >= 20:
            s20 = float(prices.rolling(20).mean().iloc[-1])
            r["sma20"] = round(s20, 2); r["above_sma20"] = curr > s20
        if len(prices) >= 50:
            s50 = float(prices.rolling(50).mean().iloc[-1])
            r["sma50"] = round(s50, 2); r["above_sma50"] = curr > s50
        if len(prices) >= 200:
            s200 = float(prices.rolling(200).mean().iloc[-1])
            r["sma200"] = round(s200, 2); r["above_sma200"] = curr > s200
            s50p  = float(prices.rolling(50).mean().iloc[-2])
            s200p = float(prices.rolling(200).mean().iloc[-2])
            r["golden_cross"] = s50 > s200 and s50p <= s200p
            r["death_cross"]  = s50 < s200 and s50p >= s200p
        return r

    # ──────────────────────────────────────────────────────────
    # DIMENSION 1: PRICE MOMENTUM (30 pts)
    # ──────────────────────────────────────────────────────────

    def _score_momentum(self, hist_1y: pd.DataFrame, info: dict) -> dict:
        score = 0; signals = []; warnings = []; details = {}

        if hist_1y.empty or len(hist_1y) < 20:
            return {"score": 15, "max": 30, "signals": ["⚠️ Insufficient price history"],
                    "warnings": [], "details": {}}

        prices = hist_1y["Close"]
        curr   = float(prices.iloc[-1])
        high52 = float(prices.max())
        low52  = float(prices.min())
        rng    = high52 - low52
        pct52  = (curr - low52) / rng * 100 if rng > 0 else 50

        details.update({"current_price": round(curr, 2),
                         "high_52w": round(high52, 2),
                         "low_52w":  round(low52, 2),
                         "pct_52w_pos": round(pct52, 1)})

        # 52-week position
        if pct52 >= 80:
            score += 4; signals.append(f"⚠️ Near 52-week high ({pct52:.0f}%) — momentum strong but stretched")
        elif pct52 >= 60:
            score += 7; signals.append(f"✅ Strong 52-week position ({pct52:.0f}% of range)")
        elif pct52 >= 40:
            score += 5; signals.append(f"🟡 Mid-range 52-week position ({pct52:.0f}%)")
        elif pct52 >= 20:
            score += 3; warnings.append(f"⚠️ Weak 52-week position ({pct52:.0f}%) — near lows")
        else:
            score += 1; warnings.append(f"🔴 Near 52-week low ({pct52:.0f}%) — bearish trend")

        # Returns
        def ret(d): return (prices.iloc[-1]/prices.iloc[-d]-1)*100 if len(prices)>d else None
        r1m, r3m, r6m = ret(21), ret(63), ret(126)
        details.update({"return_1m": round(r1m,1) if r1m else None,
                         "return_3m": round(r3m,1) if r3m else None,
                         "return_6m": round(r6m,1) if r6m else None})

        if r3m is not None:
            if r3m >= 20:   score += 8; signals.append(f"✅ Strong 3M return +{r3m:.1f}% — powerful momentum")
            elif r3m >= 10: score += 6; signals.append(f"✅ Positive 3M return +{r3m:.1f}%")
            elif r3m >= 0:  score += 4; signals.append(f"🟡 Flat 3M return {r3m:+.1f}%")
            elif r3m >= -10:score += 2; warnings.append(f"⚠️ Negative 3M return {r3m:.1f}%")
            else:           score += 0; warnings.append(f"🔴 Sharp 3M decline {r3m:.1f}% — strong downtrend")

        # SMA alignment
        sma = self._sma_signals(prices)
        details["sma"] = sma
        above = sum([sma["above_sma20"], sma["above_sma50"], sma["above_sma200"]])
        if above == 3:   score += 8; signals.append("✅ Price above SMA20/50/200 — full bullish alignment")
        elif above == 2: score += 6; signals.append("🟡 Price above 2/3 SMAs — moderate uptrend")
        elif above == 1: score += 3; warnings.append("⚠️ Price below most SMAs — weak trend")
        else:            score += 0; warnings.append("🔴 Price below all SMAs — confirmed downtrend")

        if sma["golden_cross"]: score += 3; signals.append("✅ Golden Cross — SMA50 crossed above SMA200 — bullish")
        elif sma["death_cross"]: warnings.append("🔴 Death Cross — SMA50 below SMA200 — bearish")

        if r6m is not None:
            if r6m >= 30:   score += 3; signals.append(f"✅ Exceptional 6M return +{r6m:.1f}%")
            elif r6m >= 15: score += 2; signals.append(f"✅ Strong 6M return +{r6m:.1f}%")
            elif r6m < -20: warnings.append(f"🔴 Major 6M decline {r6m:.1f}% — structural weakness")

        return {"score": min(score,30), "max": 30, "signals": signals, "warnings": warnings, "details": details}

    # ──────────────────────────────────────────────────────────
    # DIMENSION 2: VOLUME & BREADTH (20 pts)
    # ──────────────────────────────────────────────────────────

    def _score_volume(self, hist_1y: pd.DataFrame, hist_3m: pd.DataFrame) -> dict:
        score = 0; signals = []; warnings = []; details = {}

        if hist_1y.empty:
            return {"score": 10, "max": 20, "signals": [], "warnings": [], "details": {}}

        vol   = hist_1y["Volume"]
        close = hist_1y["Close"]
        v20   = float(vol.rolling(20).mean().iloc[-1]) if len(vol)>=20 else float(vol.mean())
        v6m   = float(vol.tail(126).mean()) if len(vol)>=126 else float(vol.mean())
        vtrd  = (v20/v6m-1)*100 if v6m>0 else 0

        details.update({"vol_20d_avg_L": round(v20/1e5,1),
                         "vol_6m_avg_L":  round(v6m/1e5,1),
                         "vol_trend_pct": round(vtrd,1)})

        if vtrd >= 20:   score += 6; signals.append(f"✅ Volume expanding +{vtrd:.0f}% vs 6M avg — conviction rising")
        elif vtrd >= 5:  score += 4; signals.append(f"🟡 Slightly higher volume (+{vtrd:.0f}%)")
        elif vtrd >= -10:score += 3; signals.append("🟡 Stable volume — balanced participation")
        else:            score += 1; warnings.append(f"⚠️ Declining volume {vtrd:.0f}% — weak participation")

        # Accumulation / Distribution
        if len(hist_3m) >= 20:
            recent  = hist_3m.tail(20)
            avg_vol = float(recent["Volume"].mean())
            acc = dist = 0
            for i in range(1, len(recent)):
                up  = recent["Close"].iloc[i] > recent["Close"].iloc[i-1]
                hv  = recent["Volume"].iloc[i] > avg_vol
                if up and hv:   acc  += 1
                elif not up and hv: dist += 1
            details.update({"acc_days": acc, "dist_days": dist})
            ratio = acc / max(dist, 1)
            if ratio >= 2.0:   score += 8; signals.append(f"✅ Strong accumulation: {acc} acc vs {dist} dist days — institutions buying")
            elif ratio >= 1.3: score += 6; signals.append(f"✅ Mild accumulation: {acc} acc vs {dist} dist")
            elif ratio >= 0.8: score += 4; signals.append(f"🟡 Balanced: {acc} acc vs {dist} dist")
            else:              score += 1; warnings.append(f"🔴 Distribution: {dist} dist vs {acc} acc days — selling pressure")

        # Liquidity
        daily_cr = v20 * float(close.iloc[-1]) / 1e7
        details["daily_value_cr"] = round(daily_cr, 1)
        if daily_cr >= 100:  score += 4; signals.append(f"✅ High liquidity ₹{daily_cr:.0f} Cr daily turnover")
        elif daily_cr >= 10: score += 2; signals.append(f"🟡 Adequate liquidity ₹{daily_cr:.0f} Cr daily")
        else:                score += 1; warnings.append(f"⚠️ Low liquidity ₹{daily_cr:.1f} Cr — impact cost risk")

        # Relative volume (last day)
        rel_vol = float(vol.iloc[-1]) / v20 if v20 > 0 else 1
        details["relative_volume"] = round(rel_vol, 2)
        if rel_vol >= 2.0:  score += 2; signals.append(f"✅ High relative volume {rel_vol:.1f}x — unusual activity")
        elif rel_vol < 0.4: warnings.append(f"⚠️ Very low volume {rel_vol:.1f}x — low conviction move")

        return {"score": min(score,20), "max": 20, "signals": signals, "warnings": warnings, "details": details}

    # ──────────────────────────────────────────────────────────
    # DIMENSION 3: TECHNICAL SIGNALS (20 pts)
    # ──────────────────────────────────────────────────────────

    def _score_technicals(self, hist_1y: pd.DataFrame) -> dict:
        score = 0; signals = []; warnings = []; details = {}

        if hist_1y.empty or len(hist_1y) < 30:
            return {"score": 10, "max": 20, "signals": [], "warnings": [], "details": {}}

        prices = hist_1y["Close"]
        rsi    = self._rsi(prices)
        macd   = self._macd(prices)
        bb     = self._bollinger(prices)
        details.update({"rsi": rsi, "macd": macd, "bollinger": bb})

        # RSI (0-100): <30 oversold = buy, >70 overbought = caution
        if rsi <= 30:    score += 8; signals.append(f"✅ RSI {rsi} — oversold, potential reversal entry")
        elif rsi <= 45:  score += 6; signals.append(f"✅ RSI {rsi} — mild oversold, good entry zone")
        elif rsi <= 60:  score += 5; signals.append(f"🟡 RSI {rsi} — neutral, no extreme reading")
        elif rsi <= 70:  score += 3; signals.append(f"🟡 RSI {rsi} — approaching overbought, strong momentum")
        else:            score += 1; warnings.append(f"⚠️ RSI {rsi} — overbought, caution on new entries")

        # MACD
        if macd["bullish"]:         score += 7; signals.append(f"✅ MACD bullish crossover — histogram rising ({macd['histogram']:+.2f})")
        elif macd["histogram"] > 0: score += 5; signals.append(f"🟡 MACD positive ({macd['histogram']:+.2f}) — uptrend intact")
        elif macd["histogram"] > -1:score += 3; warnings.append(f"⚠️ MACD mildly negative ({macd['histogram']:+.2f}) — weakening")
        else:                       score += 0; warnings.append(f"🔴 MACD bearish ({macd['histogram']:+.2f}) — downtrend signal")

        # Bollinger Bands
        if bb["position"] == "lower":   score += 5; signals.append(f"✅ Near lower Bollinger Band (B%={bb['pct_b']:.2f}) — mean reversion opportunity")
        elif bb["position"] == "middle": score += 4; signals.append(f"🟡 Mid Bollinger range (B%={bb['pct_b']:.2f}) — neutral")
        else:                            score += 2; warnings.append(f"⚠️ Near upper Bollinger Band (B%={bb['pct_b']:.2f}) — stretched, pullback risk")

        return {"score": min(score,20), "max": 20, "signals": signals, "warnings": warnings, "details": details}

    # ──────────────────────────────────────────────────────────
    # DIMENSION 4: INSTITUTIONAL FLOW (20 pts)
    # ──────────────────────────────────────────────────────────

    def _score_institutional(self, info: dict, hist_1y: pd.DataFrame) -> dict:
        score = 0; signals = []; warnings = []; details = {}

        inst  = (info.get("heldPercentInstitutions", 0) or 0) * 100
        beta  = info.get("beta", None)
        details.update({"institutional_pct": round(inst,1), "beta": beta})

        # Institutional holding
        if inst >= 50:   score += 8; signals.append(f"✅ Very high institutional holding {inst:.1f}% — strong smart money conviction")
        elif inst >= 30: score += 6; signals.append(f"✅ High institutional holding {inst:.1f}%")
        elif inst >= 15: score += 4; signals.append(f"🟡 Moderate institutional holding {inst:.1f}%")
        elif inst >= 5:  score += 2; warnings.append(f"⚠️ Low institutional interest {inst:.1f}%")
        else:            score += 0; warnings.append(f"🔴 Minimal institutional holding {inst:.1f}% — retail-driven stock")

        # Beta
        if beta is not None:
            if 0.5 <= beta <= 1.2:   score += 5; signals.append(f"✅ Beta {beta:.2f} — moderate market sensitivity")
            elif beta < 0.5:          score += 4; signals.append(f"✅ Low beta {beta:.2f} — defensive, less volatile")
            elif beta <= 1.5:         score += 3; warnings.append(f"🟡 Elevated beta {beta:.2f} — more volatile than market")
            else:                     score += 1; warnings.append(f"⚠️ High beta {beta:.2f} — amplified market moves")

        # Volatility (annualized)
        if not hist_1y.empty and len(hist_1y) >= 20:
            vol_ann = float(hist_1y["Close"].pct_change().tail(20).std() * np.sqrt(252) * 100)
            details["annualized_vol_pct"] = round(vol_ann, 1)
            if vol_ann < 20:    score += 4; signals.append(f"✅ Low volatility {vol_ann:.0f}% ann — stable price action")
            elif vol_ann < 35:  score += 3; signals.append(f"🟡 Moderate volatility {vol_ann:.0f}% — normal for large-cap")
            elif vol_ann < 50:  score += 1; warnings.append(f"⚠️ High volatility {vol_ann:.0f}% — elevated risk")
            else:               score += 0; warnings.append(f"🔴 Extreme volatility {vol_ann:.0f}% — speculative phase")

        # Relative strength vs Nifty 50
        if not hist_1y.empty and len(hist_1y) >= 63:
            try:
                nifty = yf.Ticker("^NSEI").history(period="1y")["Close"]
                if len(nifty) >= 63:
                    sr3m = (hist_1y["Close"].iloc[-1]/hist_1y["Close"].iloc[-63]-1)*100
                    nr3m = (nifty.iloc[-1]/nifty.iloc[-63]-1)*100
                    rs   = sr3m - nr3m
                    details.update({"stock_3m": round(sr3m,1), "nifty_3m": round(nr3m,1), "rel_strength_3m": round(rs,1)})
                    if rs >= 10:   score += 3; signals.append(f"✅ Outperforming Nifty by +{rs:.1f}% (3M) — alpha generating")
                    elif rs >= 0:  score += 2; signals.append(f"🟡 In-line with Nifty ({rs:+.1f}% vs index)")
                    else:          warnings.append(f"⚠️ Underperforming Nifty by {abs(rs):.1f}% (3M) — relative weakness")
            except Exception as e:
                logger.warning(f"⚠️ Nifty comparison failed: {e}")

        return {"score": min(score,20), "max": 20, "signals": signals, "warnings": warnings, "details": details}

    # ──────────────────────────────────────────────────────────
    # DIMENSION 5: VALUATION CONTEXT (10 pts)
    # ──────────────────────────────────────────────────────────

    def _score_valuation_ctx(self, info: dict, hist_1y: pd.DataFrame) -> dict:
        score = 0; signals = []; warnings = []; details = {}

        pe      = info.get("trailingPE", 0) or 0
        sector  = info.get("sector", "")
        details["pe"] = round(pe, 1) if pe else None

        # Sector P/E benchmarks (Nifty sector approximate averages)
        ref_pe_map = {
            "Technology": 28, "Information Technology": 28,
            "Financial Services": 20, "Consumer Defensive": 45,
            "Consumer Cyclical": 35, "Healthcare": 30,
            "Energy": 12, "Basic Materials": 15,
            "Industrials": 25, "Communication Services": 22,
        }
        ref_pe = ref_pe_map.get(sector, 25)
        details["sector_ref_pe"] = ref_pe

        if pe > 0:
            disc = (ref_pe - pe) / ref_pe * 100
            if disc >= 20:    score += 5; signals.append(f"✅ P/E {pe:.1f}x — {disc:.0f}% below sector avg ({ref_pe}x) — value")
            elif disc >= 0:   score += 4; signals.append(f"🟡 P/E {pe:.1f}x — inline with sector avg ({ref_pe}x)")
            elif disc >= -25: score += 2; warnings.append(f"⚠️ P/E {pe:.1f}x — slight premium to sector avg")
            else:             score += 0; warnings.append(f"🔴 P/E {pe:.1f}x — significantly above sector avg — expensive")
        else:
            score += 3; signals.append("🟡 P/E not available (financial sector or loss-making)")

        # Price vs 1-year mean
        if not hist_1y.empty:
            mean1y  = float(hist_1y["Close"].mean())
            curr    = float(hist_1y["Close"].iloc[-1])
            vs_mean = (curr/mean1y-1)*100
            details.update({"vs_1y_mean_pct": round(vs_mean,1), "mean_1y": round(mean1y,2)})
            if vs_mean <= -15:  score += 5; signals.append(f"✅ Price {vs_mean:.0f}% below 1Y mean — mean reversion potential")
            elif vs_mean <= 0:  score += 4; signals.append(f"🟡 Price {vs_mean:+.0f}% below 1Y mean — reasonable level")
            elif vs_mean <= 15: score += 3; signals.append(f"🟡 Price {vs_mean:+.0f}% above 1Y mean — slightly extended")
            else:               score += 1; warnings.append(f"⚠️ Price {vs_mean:+.0f}% above 1Y mean — extended, await pullback")

        return {"score": min(score,10), "max": 10, "signals": signals, "warnings": warnings, "details": details}

    # ──────────────────────────────────────────────────────────
    # SIGNAL + ANALYZE
    # ──────────────────────────────────────────────────────────

    def _get_signal(self, score: int) -> tuple:
        if score >= 80:   return "🟢 STRONG BUY",  "A+", "Powerful momentum + institutional alignment — high conviction entry"
        elif score >= 65: return "🟢 BUY",          "A",  "Positive sentiment — accumulate on dips"
        elif score >= 50: return "🟡 NEUTRAL",      "B",  "Mixed signals — wait for clarity before adding"
        elif score >= 35: return "🟠 CAUTION",      "C",  "Weak momentum — reduce exposure or wait"
        else:             return "🔴 AVOID",        "D",  "Bearish sentiment — avoid fresh positions"

    def analyze(self) -> dict:
        logger.info(f"\n{'='*50}")
        logger.info(f"📡 MARKET SENTIMENT ENGINE: {self.ticker}")
        logger.info(f"{'='*50}")

        data    = self._get_data()
        info    = data["info"]
        hist_1y = data["hist_1y"]
        hist_3m = data["hist_3m"]

        logger.info("🔢 Scoring Price Momentum...")
        mom  = self._score_momentum(hist_1y, info)
        logger.info(f"   Momentum: {mom['score']}/30")

        logger.info("🔢 Scoring Volume & Breadth...")
        vol  = self._score_volume(hist_1y, hist_3m)
        logger.info(f"   Volume: {vol['score']}/20")

        logger.info("🔢 Scoring Technical Signals...")
        tech = self._score_technicals(hist_1y)
        logger.info(f"   Technicals: {tech['score']}/20")

        logger.info("🔢 Scoring Institutional Flow...")
        inst = self._score_institutional(info, hist_1y)
        logger.info(f"   Institutional: {inst['score']}/20")

        logger.info("🔢 Scoring Valuation Context...")
        valc = self._score_valuation_ctx(info, hist_1y)
        logger.info(f"   Valuation Context: {valc['score']}/10")

        total = mom["score"] + vol["score"] + tech["score"] + inst["score"] + valc["score"]
        signal, grade, verdict = self._get_signal(total)
        logger.info(f"✅ Sentiment Analysis: {total}/100 {grade} — {signal}")

        return {
            "ticker":        self.ticker,
            "total_score":   total,
            "grade":         grade,
            "signal":        signal,
            "verdict":       verdict,
            "current_price": mom["details"].get("current_price", 0),
            "dimensions": {
                "momentum":      mom,
                "volume":        vol,
                "technicals":    tech,
                "institutional": inst,
                "valuation_ctx": valc,
            },
            "rsi":          tech["details"].get("rsi", 0),
            "macd_bullish": tech["details"].get("macd", {}).get("bullish", False),
            "bb_position":  tech["details"].get("bollinger", {}).get("position", "middle"),
            "sma_above":    sum([mom["details"].get("sma",{}).get("above_sma20",False),
                                  mom["details"].get("sma",{}).get("above_sma50",False),
                                  mom["details"].get("sma",{}).get("above_sma200",False)]),
            "pct_52w":      mom["details"].get("pct_52w_pos", 50),
            "return_3m":    mom["details"].get("return_3m", 0),
            "beta":         inst["details"].get("beta", 1.0),
            "rel_strength": inst["details"].get("rel_strength_3m", 0),
            "inst_pct":     inst["details"].get("institutional_pct", 0),
        }
