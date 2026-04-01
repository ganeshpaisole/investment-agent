#!/usr/bin/env python3
"""
intraday_monitor.py — Live intraday monitor for NSE stocks.

Runs every 30 min during market hours (09:15–15:30 IST) and:
  1. Fetches live prices for quality stocks (BUY/HOLD signal or score ≥ 60).
     → Sends BUY alert if price drops > 3 % from today's reference price.
  2. Polls BSE for new quarterly-result filings.
     → Auto-re-parses any new filing and emails updated key metrics.

Usage:
    python intraday_monitor.py             # normal run
    python intraday_monitor.py --dry-run   # log only, no emails
    python intraday_monitor.py --force     # ignore market-hours gate
"""

import argparse
import csv
import json
import os
import smtplib
import sys
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils.data_fetcher import NSEDataFetcher  # noqa: E402
from utils.bse_parser import BSEFilingParser   # noqa: E402

# ── timezone (IST = UTC+5:30, no DST) ────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))

# ── thresholds ────────────────────────────────────────────────────────────────
PRICE_DROP_THRESHOLD = 3.0   # % intraday drop to trigger BUY alert
QUALITY_SCORE_FLOOR  = 60    # minimum overall_score for "quality stock"
QUALITY_SIGNALS      = {"BUY", "HOLD"}

MARKET_OPEN_H,  MARKET_OPEN_M  = 9,  15   # IST
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30   # IST

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
REPORTS_DIR = Path("reports")

_today            = date.today().isoformat()
HOLIDAYS_FILE     = DATA_DIR / "bse_holidays_2026.json"
LATEST_SCREENER   = REPORTS_DIR / "nifty50_screener_latest.csv"
ALERTS_FILE       = DATA_DIR / f"intraday_alerts_{_today}.json"
REF_PRICES_FILE   = DATA_DIR / f"intraday_prices_{_today}.json"
SEEN_FILINGS_FILE = DATA_DIR / f"intraday_filings_{_today}.json"
FILING_LOG_FILE   = DATA_DIR / f"intraday_filings_{_today}.csv"

# ── BSE scrip codes (NSE ticker → BSE 6-digit code) ───────────────────────────
BSE_SCRIP_MAP: dict[str, str] = {
    "HDFCBANK":   "500180",
    "ICICIBANK":  "532174",
    "SBIN":       "500112",
    "AXISBANK":   "532215",
    "KOTAKBANK":  "500247",
    "INDUSINDBK": "532187",
    "BAJFINANCE": "500034",
    "TCS":        "532540",
    "INFY":       "500209",
    "WIPRO":      "507685",
    "HCLTECH":    "532281",
    "RELIANCE":   "500325",
    "TATAMOTORS": "500570",
    "MARUTI":     "532500",
    "SUNPHARMA":  "524715",
    "DRREDDY":    "500124",
    "ASIANPAINT": "500820",
    "NESTLEIND":  "500790",
    "HINDUNILVR": "500696",
    "ITC":        "500875",
    "LT":         "500510",
    "ULTRACEMCO": "532538",
    "BHARTIARTL": "532454",
    "TITAN":      "500114",
    "BAJAJ-AUTO": "532977",
    "BAJAJFINSV": "532978",
    "ADANIENT":   "512599",
    "ADANIPORTS": "532921",
    "POWERGRID":  "532898",
    "NTPC":       "532555",
    "ONGC":       "500312",
    "COALINDIA":  "533278",
    "JSWSTEEL":   "500228",
    "TATASTEEL":  "500470",
    "HINDALCO":   "500440",
    "GRASIM":     "500300",
    "TECHM":      "532755",
    "BPCL":       "500547",
    "CIPLA":      "500087",
    "EICHERMOT":  "505200",
    "M&M":        "500520",
    "BRITANNIA":  "500825",
    "APOLLOHOSP": "508869",
    "DIVISLAB":   "532488",
    "HEROMOTOCO": "500182",
    "INDIGO":     "521228",
    "TATACONSUM": "500800",
    "SBILIFE":    "540719",
    "HDFCLIFE":   "540777",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _now_ist() -> datetime:
    return datetime.now(IST)


# ─────────────────────────────────────────────────────────────────────────────
# Market hours gate
# ─────────────────────────────────────────────────────────────────────────────

def _load_holidays() -> set[str]:
    data = _load_json(HOLIDAYS_FILE, {"holidays": []})
    return {h["date"] for h in data.get("holidays", [])}


def is_market_open(force: bool = False) -> bool:
    """Return True only when NSE is currently open (weekday, non-holiday, 09:15–15:30 IST)."""
    if force:
        return True
    now = _now_ist()
    if now.weekday() >= 5:
        logger.info("Weekend — market closed.")
        return False
    today_str = now.date().isoformat()
    if today_str in _load_holidays():
        logger.info(f"Exchange holiday ({today_str}) — market closed.")
        return False
    open_dt  = now.replace(hour=MARKET_OPEN_H,  minute=MARKET_OPEN_M,  second=0, microsecond=0)
    close_dt = now.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
    if not (open_dt <= now <= close_dt):
        logger.info(f"Outside market hours ({now.strftime('%H:%M')} IST) — nothing to do.")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Quality stock loader
# ─────────────────────────────────────────────────────────────────────────────

def load_quality_stocks() -> list[dict]:
    """
    Return stocks from the latest screener CSV that meet the quality bar:
    signal ∈ {BUY, HOLD}  OR  overall_score ≥ QUALITY_SCORE_FLOOR.
    Falls back to the full BSE_SCRIP_MAP if the CSV is missing.
    """
    if not LATEST_SCREENER.exists():
        logger.warning(f"No screener CSV at {LATEST_SCREENER} — watching full Nifty 50.")
        return [{"ticker": t, "signal": "", "score": 0, "price": 0, "name": ""}
                for t in BSE_SCRIP_MAP]

    quality = []
    with LATEST_SCREENER.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            signal = row.get("signal", "").upper().strip()
            try:
                score = float(row.get("overall", row.get("overall_score", 0)) or 0)
            except ValueError:
                score = 0.0
            if signal in QUALITY_SIGNALS or score >= QUALITY_SCORE_FLOOR:
                quality.append({
                    "ticker": row.get("ticker", "").strip(),
                    "signal": signal,
                    "score":  score,
                    "price":  float(row.get("price", 0) or 0),
                    "name":   row.get("name", ""),
                })
    logger.info(f"Loaded {len(quality)} quality stocks from screener CSV.")
    return quality


# ─────────────────────────────────────────────────────────────────────────────
# Reference-price store  (first observed price = today's baseline)
# ─────────────────────────────────────────────────────────────────────────────

def get_reference_price(ticker: str, current_price: float) -> float:
    ref = _load_json(REF_PRICES_FILE, {})
    if ticker not in ref:
        ref[ticker] = current_price
        _save_json(REF_PRICES_FILE, ref)
        logger.debug(f"{ticker}: reference price initialised at ₹{current_price:.2f}")
    return ref[ticker]


# ─────────────────────────────────────────────────────────────────────────────
# Email delivery
# ─────────────────────────────────────────────────────────────────────────────

def _send_email(subject: str, html_body: str) -> None:
    """Try Gmail OAuth first, fall back to SMTP."""
    try:
        from send_email_via_gmail import send_email  # type: ignore
        send_email(subject, html_body)
        return
    except Exception as exc:
        logger.warning(f"Gmail OAuth send failed ({exc}) — trying SMTP fallback.")

    user = os.getenv("SMTP_USER", "")
    pwd  = os.getenv("SMTP_PASS", "")
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", 587))

    if not user or not pwd:
        logger.error("No SMTP credentials — alert not delivered.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"]   = user
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(user, pwd)
        s.send_message(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1 — live price-drop monitoring
# ─────────────────────────────────────────────────────────────────────────────

def _buy_alert_html(
    ticker: str, name: str, drop_pct: float,
    current: float, ref: float, score: float,
) -> str:
    now_str = _now_ist().strftime("%H:%M:%S IST")
    return f"""
<h2>Intraday BUY Alert — {ticker}</h2>
<table border="1" cellpadding="6" cellspacing="0"
       style="border-collapse:collapse;font-family:monospace">
  <tr><td><b>Stock</b></td>           <td>{name or ticker}</td></tr>
  <tr><td><b>Current Price</b></td>   <td>₹{current:.2f}</td></tr>
  <tr><td><b>Reference Price</b></td> <td>₹{ref:.2f}</td></tr>
  <tr><td><b>Intraday Drop</b></td>
      <td style="color:red"><b>▼ {drop_pct:.2f}%</b></td></tr>
  <tr><td><b>Quality Score</b></td>   <td>{score:.0f} / 100</td></tr>
  <tr><td><b>Time</b></td>            <td>{now_str}</td></tr>
</table>
<p>This stock is rated <b>quality</b> (score ≥ {QUALITY_SCORE_FLOOR} or signal BUY/HOLD).
A &gt;{PRICE_DROP_THRESHOLD:.0f}% intraday dip may present an attractive entry.</p>
<hr><small>NSE Agent — Intraday Monitor</small>
"""


def check_price_drops(quality_stocks: list[dict], dry_run: bool) -> list[dict]:
    """
    Fetch live prices and fire a BUY alert when a quality stock drops
    more than PRICE_DROP_THRESHOLD % from today's reference price.
    Each ticker fires at most one alert per day.
    """
    alerts_fired = _load_json(ALERTS_FILE, {})
    triggered = []

    for stock in quality_stocks:
        ticker = stock.get("ticker", "")
        if not ticker:
            continue
        try:
            fetcher = NSEDataFetcher(ticker, cloud=True)
            current = fetcher.get_current_price()
            if not current or current <= 0:
                logger.warning(f"{ticker}: could not fetch live price — skipping.")
                continue

            ref      = get_reference_price(ticker, current)
            drop_pct = (ref - current) / ref * 100

            logger.debug(f"{ticker}: ₹{current:.2f}  ref ₹{ref:.2f}  Δ {drop_pct:+.2f}%")

            if drop_pct < PRICE_DROP_THRESHOLD:
                continue

            alert_key = f"{ticker}_{_today}"
            if alert_key in alerts_fired:
                logger.debug(f"{ticker}: BUY alert already sent today — suppressed.")
                continue

            subject = (
                f"[NSE Intraday] BUY Alert — {ticker} dropped {drop_pct:.1f}%"
            )
            if dry_run:
                logger.info(f"[DRY-RUN] {subject}")
            else:
                _send_email(
                    subject,
                    _buy_alert_html(
                        ticker, stock.get("name", ""), drop_pct,
                        current, ref, stock.get("score", 0),
                    ),
                )
                logger.success(f"BUY alert sent — {ticker} ▼{drop_pct:.2f}%")

            record = {
                "ticker":   ticker,
                "drop_pct": round(drop_pct, 2),
                "current":  current,
                "ref":      ref,
                "time_ist": _now_ist().isoformat(),
            }
            alerts_fired[alert_key] = record
            _save_json(ALERTS_FILE, alerts_fired)
            triggered.append(record)

        except Exception as exc:
            logger.error(f"{ticker}: price-check error — {exc}")

    return triggered


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2 — BSE new-filing detection + auto re-parse
# ─────────────────────────────────────────────────────────────────────────────

_BSE_ANN_URL = (
    "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
    "?strCat=Result&strType=C&strScrip={scrip}&strSearch=P"
)
_BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
    "Accept":     "application/json",
    "Referer":    "https://www.bseindia.com/",
}


def _latest_filing_uuid(scrip: str) -> str | None:
    """Return the attachment UUID of the most recent result filing from BSE API."""
    url = _BSE_ANN_URL.format(scrip=scrip)
    try:
        r = requests.get(url, headers=_BSE_HEADERS, timeout=15)
        r.raise_for_status()
        items = r.json().get("Table", [])
        if items:
            return str(items[0].get("ATTACHMENTNAME", "")).strip() or None
    except Exception as exc:
        logger.debug(f"BSE filing API scrip={scrip}: {exc}")
    return None


def _log_filing_to_csv(ticker: str, uuid: str, metrics: dict) -> None:
    cols = ["timestamp_ist", "ticker", "uuid",
            "nim", "gnpa", "nnpa", "car", "casa", "roe", "roa"]
    write_header = not FILING_LOG_FILE.exists()
    with FILING_LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({
            "timestamp_ist": _now_ist().isoformat(),
            "ticker": ticker,
            "uuid":   uuid,
            **{k: metrics.get(k, "") for k in cols[3:]},
        })


def _filing_alert_html(ticker: str, name: str, uuid: str, metrics: dict) -> str:
    rows = "".join(
        f"<tr><td><b>{k.upper()}</b></td><td>{v}</td></tr>"
        for k, v in metrics.items()
        if v not in (None, "", 0)
    ) or "<tr><td colspan='2'>No structured metrics extracted</td></tr>"
    return f"""
<h2>New BSE Filing — {ticker}</h2>
<p><b>Company:</b> {name or ticker}<br>
<b>Filing UUID:</b> {uuid}<br>
<b>Detected:</b> {_now_ist().strftime("%H:%M:%S IST")}</p>
<h3>Updated Key Metrics</h3>
<table border="1" cellpadding="6" cellspacing="0"
       style="border-collapse:collapse;font-family:monospace">
{rows}
</table>
<hr><small>NSE Agent — Intraday Monitor</small>
"""


def check_new_bse_filings(quality_stocks: list[dict], dry_run: bool) -> list[str]:
    """
    Poll the BSE API for every quality stock that has a known scrip code.
    When the latest filing UUID changes since the last check, re-parse the
    filing and email the updated metrics.
    Returns list of tickers whose filings were refreshed.
    """
    seen = _load_json(SEEN_FILINGS_FILE, {})
    refreshed = []

    for stock in quality_stocks:
        ticker = stock.get("ticker", "")
        scrip  = BSE_SCRIP_MAP.get(ticker)
        if not ticker or not scrip:
            continue
        try:
            uuid = _latest_filing_uuid(scrip)
            if not uuid:
                continue
            if seen.get(ticker) == uuid:
                logger.debug(f"{ticker}: BSE filing unchanged (UUID={uuid[:8]}…)")
                continue

            logger.info(f"{ticker}: new BSE filing detected (UUID={uuid[:8]}…) — re-parsing.")

            # BSEFilingParser fetches and caches the filing internally.
            # A new UUID means a new PDF → parser will download fresh data.
            parser  = BSEFilingParser(ticker)
            metrics = parser.get_filing_data()

            seen[ticker] = uuid
            _save_json(SEEN_FILINGS_FILE, seen)
            _log_filing_to_csv(ticker, uuid, metrics)

            subject = f"[NSE Intraday] New BSE Filing — {ticker}"
            if dry_run:
                logger.info(f"[DRY-RUN] {subject}")
            else:
                _send_email(
                    subject,
                    _filing_alert_html(ticker, stock.get("name", ""), uuid, metrics),
                )
                logger.success(f"Filing alert sent — {ticker}")

            refreshed.append(ticker)

        except Exception as exc:
            logger.error(f"{ticker}: BSE filing check error — {exc}")

    return refreshed


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="NSE Intraday Monitor")
    ap.add_argument("--dry-run", action="store_true",
                    help="Log alerts without sending emails.")
    ap.add_argument("--force",   action="store_true",
                    help="Skip market-hours check and run anyway.")
    args = ap.parse_args()

    # ── logging setup ─────────────────────────────────────────────────────────
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="{time:HH:mm:ss} | {level:<7} | {message}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        REPORTS_DIR / f"intraday_{_today}.log",
        level="DEBUG", rotation="1 day", retention="7 days",
    )

    # ── market-hours gate ─────────────────────────────────────────────────────
    if not is_market_open(force=args.force):
        sys.exit(0)

    logger.info("=== Intraday Monitor — started ===")

    quality = load_quality_stocks()
    if not quality:
        logger.warning("No quality stocks found — exiting.")
        sys.exit(0)

    # ── 1. price-drop check ───────────────────────────────────────────────────
    logger.info(
        f"Checking {len(quality)} quality stocks "
        f"for intraday drops > {PRICE_DROP_THRESHOLD}% ..."
    )
    triggered = check_price_drops(quality, dry_run=args.dry_run)
    if triggered:
        logger.info(f"BUY alerts fired: {[a['ticker'] for a in triggered]}")
    else:
        logger.info("No price-drop alerts this run.")

    # ── 2. BSE filing check ───────────────────────────────────────────────────
    logger.info("Polling BSE for new filings ...")
    refreshed = check_new_bse_filings(quality, dry_run=args.dry_run)
    if refreshed:
        logger.info(f"Refreshed metrics for: {refreshed}")
    else:
        logger.info("No new BSE filings detected.")

    logger.info("=== Intraday Monitor — done ===")


if __name__ == "__main__":
    main()
