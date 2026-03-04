# ============================================================
# screener.py — Nifty 50 Batch Screener v1.0
# ============================================================
# Runs all 50 Nifty stocks through the full agent pipeline,
# ranks them by overall score, and outputs:
#   1. Live terminal leaderboard (rich table, updates as stocks complete)
#   2. CSV report  → reports/nifty50_screener_YYYYMMDD.csv
#   3. Summary     → reports/nifty50_summary_YYYYMMDD.txt
#
# Usage:
#   python screener.py                        # full Nifty 50
#   python screener.py --sector IT            # IT stocks only
#   python screener.py --sector BANKING       # Banking only
#   python screener.py --top 10               # show top 10
#   python screener.py --tickers TCS INFY     # custom list
#   python screener.py --resume               # skip already done
#
# Runtime: ~3-5 min for full Nifty 50 (1 stock ~4-6 seconds)
# ============================================================

import argparse
import csv
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box

# ── silence logger for clean screener output ─────────────────
logger.remove()
logger.add("reports/screener_debug.log", level="DEBUG", rotation="10 MB")

console = Console()

# ── Nifty 50 constituents (as of 2025) ───────────────────────
NIFTY50 = [
    # IT / Technology
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM",

    # Banking
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "INDUSINDBK", "BANDHANBNK",

    # NBFC / Financial
    "BAJFINANCE", "BAJAJFINSV", "HDFCAMC",

    # Insurance
    "HDFCLIFE", "SBILIFE",

    # FMCG / Consumer
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "MARICO", "DABUR", "TATACONSUM",

    # Auto
    "MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "M&M",

    # Pharma / Healthcare
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",

    # Industrials / Infra
    "LT", "ADANIENT", "ADANIPORTS", "NTPC", "POWERGRID", "ONGC",

    # Metals / Materials
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA",

    # Telecom / Media
    "BHARTIARTL",

    # Conglomerate / Energy
    "RELIANCE",
]

# ── Sector groupings for filter ───────────────────────────────
SECTOR_FILTER = {
    "IT":       ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM"],
    "BANKING":  ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "BANDHANBNK"],
    "NBFC":     ["BAJFINANCE", "BAJAJFINSV", "HDFCAMC"],
    "INSURANCE":["HDFCLIFE", "SBILIFE"],
    "FMCG":     ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "MARICO", "DABUR", "TATACONSUM"],
    "AUTO":     ["MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "M&M"],
    "PHARMA":   ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
    "INFRA":    ["LT", "ADANIENT", "ADANIPORTS", "NTPC", "POWERGRID", "ONGC"],
    "METALS":   ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA"],
    "TELECOM":  ["BHARTIARTL"],
    "ENERGY":   ["RELIANCE"],
}


# ============================================================
# SINGLE STOCK ANALYSIS (silent mode — no pretty printing)
# ============================================================

def analyze_stock(ticker: str) -> dict:
    """
    Run the full agent pipeline for one stock.
    Returns a flat result dict for the screener table.
    Suppresses all rich output — screener handles its own display.
    """
    import io
    from contextlib import redirect_stdout

    # Import agents
    from utils.sector_classifier import SectorClassifier
    from agents.fundamental_agent import FundamentalAnalysisAgent
    from agents.valuation_agent   import ValuationAgent
    from agents.management_agent  import ManagementQualityAgent
    from agents.banking_agent     import BankingAgent
    from agents.insurance_agent   import InsuranceAgent
    from agents.sentiment_agent   import MarketSentimentAgent
    from utils.data_fetcher       import NSEDataFetcher

    result = {
        "ticker":         ticker,
        "name":           "—",
        "sector":         "—",
        "sector_type":    "—",
        "price":          0,
        # Scores
        "fundamental":    0,
        "valuation":      0,
        "management":     0,
        "banking":        0,
        "sentiment":      0,
        "overall":        0,
        # Key metrics
        "iv":             0,
        "upside":         0,
        "roe":            0,
        "roce":           0,
        "roa":            0,
        "pe":             0,
        "pb":             0,
        "market_cap_cr":  0,
        "rsi":            0,
        "return_3m":      0,
        # Signal
        "signal":         "—",
        "grade":          "—",
        # Status
        "status":         "pending",
        "error":          "",
        "elapsed":        0,
    }

    t0 = time.time()

    try:
        # ── Sector classification ──────────────────────────────
        fetcher = NSEDataFetcher(ticker)
        info    = fetcher.get_stock_info()
        classifier = SectorClassifier(ticker, info)
        st = classifier.sector_type

        result["name"]        = info.get("longName", info.get("shortName", ticker))[:30]
        result["sector"]      = info.get("sector", "Unknown")
        result["sector_type"] = st
        result["price"]       = info.get("currentPrice", info.get("regularMarketPrice", 0)) or 0
        result["pe"]          = info.get("trailingPE", 0) or 0
        result["pb"]          = info.get("priceToBook", 0) or 0
        result["market_cap_cr"] = round((info.get("marketCap", 0) or 0) / 1e7, 0)

        # ── Run agents based on sector ─────────────────────────
        if st in ("BANKING", "NBFC"):
            bank_agent  = BankingAgent(ticker)
            bank_result = bank_agent.analyze()
            mgmt_agent  = ManagementQualityAgent(ticker)
            mgmt_result = mgmt_agent.analyze()
            sent_agent  = MarketSentimentAgent(ticker)
            sent_result = sent_agent.analyze()

            bs = bank_result["total_score"]
            ms = mgmt_result["total_score"]
            ss = sent_result["total_score"]
            overall = round(bs * 0.45 + ms * 0.35 + ss * 0.20)

            result["banking"]    = bs
            result["management"] = ms
            result["sentiment"]  = ss
            result["overall"]    = overall
            result["iv"]         = bank_result.get("intrinsic_value", 0)
            result["upside"]     = bank_result.get("upside_pct", 0)
            result["roe"]        = bank_result["banking_ratios"].get("roe", 0)
            result["roa"]        = bank_result["banking_ratios"].get("roa", 0)
            result["pb"]         = bank_result["banking_ratios"].get("pb_ratio", 0)
            result["rsi"]        = sent_result.get("rsi", 0)
            result["return_3m"]  = sent_result.get("return_3m", 0) or 0

            # Grade + signal
            result["grade"]  = bank_result["grade"]
            result["signal"] = bank_result["verdict"]

        elif st == "INSURANCE":
            ins_agent   = InsuranceAgent(ticker)
            ins_result  = ins_agent.analyze()
            mgmt_agent  = ManagementQualityAgent(ticker)
            mgmt_result = mgmt_agent.analyze()
            sent_agent  = MarketSentimentAgent(ticker)
            sent_result = sent_agent.analyze()

            is_ = ins_result["total_score"]
            ms  = mgmt_result["total_score"]
            ss  = sent_result["total_score"]
            overall = round(is_ * 0.45 + ms * 0.35 + ss * 0.20)

            result["banking"]    = is_
            result["management"] = ms
            result["sentiment"]  = ss
            result["overall"]    = overall
            result["iv"]         = ins_result.get("intrinsic_value", 0)
            result["upside"]     = ins_result.get("upside_pct", 0)
            result["roe"]        = ins_result["insurance_ratios"].get("roe", 0)
            result["rsi"]        = sent_result.get("rsi", 0)
            result["return_3m"]  = sent_result.get("return_3m", 0) or 0
            result["grade"]      = ins_result["grade"]
            result["signal"]     = ins_result["verdict"]

        else:
            # Generic: IT, FMCG, Auto, Pharma, etc.
            fund_agent  = FundamentalAnalysisAgent(ticker)
            fund_result = fund_agent.analyze()
            val_agent   = ValuationAgent(ticker)
            val_result  = val_agent.analyze()
            mgmt_agent  = ManagementQualityAgent(ticker)
            mgmt_result = mgmt_agent.analyze()
            sent_agent  = MarketSentimentAgent(ticker)
            sent_result = sent_agent.analyze()

            fs = fund_result["total_score"]
            vs = val_result["valuation_score"]
            ms = mgmt_result["total_score"]
            ss = sent_result["total_score"]
            overall = round(fs * 0.35 + vs * 0.25 + ms * 0.25 + ss * 0.15)

            result["fundamental"] = fs
            result["valuation"]   = vs
            result["management"]  = ms
            result["sentiment"]   = ss
            result["overall"]     = overall
            result["iv"]          = val_result["composite"]["composite_iv"]
            result["upside"]      = val_result["recommendation"]["upside_pct"]
            result["signal"]      = val_result["recommendation"]["signal"]
            result["grade"]       = fund_result["grade"]

            # Key ratios from ratio_calculator output
            ratios = fund_result.get("ratios", {})
            prof   = ratios.get("profitability", {})
            result["roe"]  = prof.get("roe", 0)
            result["roce"] = prof.get("roce", 0)
            result["rsi"]  = sent_result.get("rsi", 0)
            result["return_3m"] = sent_result.get("return_3m", 0) or 0

        result["status"]  = "done"
        result["elapsed"] = round(time.time() - t0, 1)

    except Exception as e:
        result["status"]  = "error"
        result["error"]   = str(e)
        result["elapsed"] = round(time.time() - t0, 1)
        logger.error(f"❌ {ticker}: {e}\n{traceback.format_exc()}")

    return result


# ============================================================
# SIGNAL HELPERS
# ============================================================

def _signal_emoji(upside: float, overall: int) -> str:
    if overall >= 70 and upside >= 15:  return "🟢 STRONG BUY"
    if overall >= 60 and upside >= 0:   return "🟢 BUY"
    if overall >= 50:                   return "🟡 HOLD"
    if overall >= 35:                   return "🟠 CAUTION"
    return "🔴 AVOID"

def _score_color(score: int) -> str:
    if score >= 70: return "green"
    if score >= 50: return "yellow"
    return "red"

def _upside_color(upside: float) -> str:
    if upside >= 20:  return "green"
    if upside >= 0:   return "yellow"
    return "red"

def _rsi_fmt(rsi: float) -> str:
    if rsi == 0: return "—"
    if rsi <= 30: return f"[green]{rsi:.0f} 🟢[/green]"
    if rsi >= 70: return f"[red]{rsi:.0f} 🔴[/red]"
    return f"{rsi:.0f}"


# ============================================================
# LEADERBOARD TABLE
# ============================================================

def build_leaderboard(results: list, top_n: int = 0) -> Table:
    """Build the rich leaderboard table from results list."""
    sorted_res = sorted(
        [r for r in results if r["status"] == "done"],
        key=lambda x: x["overall"],
        reverse=True
    )
    if top_n:
        sorted_res = sorted_res[:top_n]

    t = Table(
        title="🏆 Nifty 50 Stock Screener Rankings",
        box=box.ROUNDED,
        header_style="bold white",
        show_lines=True,
        title_style="bold cyan",
    )
    t.add_column("#",          width=3,  justify="right")
    t.add_column("Ticker",     width=12, style="bold cyan")
    t.add_column("Sector",     width=8)
    t.add_column("Overall",    width=9,  justify="center")
    t.add_column("Fund/Bank",  width=9,  justify="center")
    t.add_column("Val/Upsid",  width=11, justify="center")
    t.add_column("Mgmt",       width=7,  justify="center")
    t.add_column("Sent",       width=7,  justify="center")
    t.add_column("ROE%",       width=7,  justify="right")
    t.add_column("P/E",        width=7,  justify="right")
    t.add_column("RSI",        width=8,  justify="center")
    t.add_column("3M Ret",     width=8,  justify="right")
    t.add_column("Signal",     width=16)

    for i, r in enumerate(sorted_res, 1):
        oc   = _score_color(r["overall"])
        fc   = _score_color(r["fundamental"] or r["banking"])
        mc   = _score_color(r["management"])
        sc   = _score_color(r["sentiment"])
        upc  = _upside_color(r["upside"])
        sig  = _signal_emoji(r["upside"], r["overall"])

        fund_bank = r["fundamental"] or r["banking"]
        val_str   = (f"[{upc}]{r['upside']:+.0f}%[/{upc}]"
                     if r["iv"] else "—")

        roe_str   = f"{r['roe']:.1f}%" if r["roe"] else "—"
        pe_str    = f"{r['pe']:.0f}x"  if r["pe"]  else "—"
        ret_str   = (f"[{'green' if r['return_3m']>=0 else 'red'}]{r['return_3m']:+.1f}%[/]"
                     if r["return_3m"] else "—")

        t.add_row(
            str(i),
            r["ticker"],
            r["sector_type"],
            f"[{oc}][bold]{r['overall']}[/bold][/{oc}]",
            f"[{fc}]{fund_bank}[/{fc}]",
            val_str,
            f"[{mc}]{r['management']}[/{mc}]",
            f"[{sc}]{r['sentiment']}[/{sc}]",
            roe_str,
            pe_str,
            _rsi_fmt(r["rsi"]),
            ret_str,
            sig,
        )

    return t


# ============================================================
# CSV EXPORT
# ============================================================

def save_csv(results: list, filepath: str):
    """Save all results to CSV."""
    fields = [
        "rank", "ticker", "name", "sector_type", "price",
        "overall", "fundamental", "valuation", "banking", "management", "sentiment",
        "grade", "iv", "upside", "signal",
        "roe", "roce", "roa", "pe", "pb", "market_cap_cr",
        "rsi", "return_3m", "status", "elapsed",
    ]
    done = sorted(
        [r for r in results if r["status"] == "done"],
        key=lambda x: x["overall"], reverse=True
    )
    errors = [r for r in results if r["status"] == "error"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, r in enumerate(done, 1):
            r["rank"] = i
            writer.writerow(r)
        for r in errors:
            r["rank"] = "ERR"
            writer.writerow(r)

    console.print(f"  [green]✅ CSV saved → {filepath}[/green]")


# ============================================================
# SUMMARY TEXT
# ============================================================

def save_summary(results: list, filepath: str, elapsed_total: float):
    done   = sorted([r for r in results if r["status"] == "done"],
                    key=lambda x: x["overall"], reverse=True)
    errors = [r for r in results if r["status"] == "error"]

    strong_buy = [r for r in done if r["overall"] >= 70 and r["upside"] >= 15]
    buy        = [r for r in done if r["overall"] >= 60 and r["upside"] >= 0
                  and r not in strong_buy]
    caution    = [r for r in done if r["overall"] < 50]
    oversold   = [r for r in done if r["rsi"] and r["rsi"] < 30]

    lines = [
        f"NSE STOCK SCREENER — NIFTY 50 REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Total time: {elapsed_total:.0f}s | Stocks analyzed: {len(done)} | Errors: {len(errors)}",
        "=" * 60,
        "",
        "🟢 STRONG BUY CANDIDATES (Overall ≥70, Upside ≥15%)",
    ]
    for r in strong_buy:
        lines.append(f"  {r['ticker']:12} Overall:{r['overall']:3}/100  "
                     f"Upside:{r['upside']:+.0f}%  ROE:{r['roe']:.1f}%  RSI:{r['rsi']:.0f}")

    lines += ["", "🟢 BUY / ACCUMULATE (Overall ≥60, Upside ≥0%)"]
    for r in buy:
        lines.append(f"  {r['ticker']:12} Overall:{r['overall']:3}/100  "
                     f"Upside:{r['upside']:+.0f}%  ROE:{r['roe']:.1f}%  RSI:{r['rsi']:.0f}")

    lines += ["", "⚠️  CAUTION / AVOID (Overall <50)"]
    for r in caution:
        lines.append(f"  {r['ticker']:12} Overall:{r['overall']:3}/100  "
                     f"Upside:{r['upside']:+.0f}%  Grade:{r['grade']}")

    lines += ["", "📉 OVERSOLD WATCH (RSI <30 — contrarian opportunity)"]
    for r in oversold:
        lines.append(f"  {r['ticker']:12} RSI:{r['rsi']:.0f}  "
                     f"3M:{r['return_3m']:+.1f}%  Overall:{r['overall']:3}/100  "
                     f"Upside:{r['upside']:+.0f}%")

    if errors:
        lines += ["", f"❌ ERRORS ({len(errors)} stocks failed)"]
        for r in errors:
            lines.append(f"  {r['ticker']:12} {r['error'][:60]}")

    lines += ["", "=" * 60,
              "Not SEBI-registered investment advice. Do your own research."]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    console.print(f"  [green]✅ Summary saved → {filepath}[/green]")


# ============================================================
# RESUME: LOAD PREVIOUSLY DONE STOCKS
# ============================================================

def load_previous_results(csv_path: str) -> dict:
    """Load results from a previous run for resume mode."""
    done = {}
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "done":
                done[row["ticker"]] = row
    return done


# ============================================================
# MAIN SCREENER LOOP
# ============================================================

def run_screener(tickers: list, top_n: int = 0, resume: bool = False):
    Path("reports").mkdir(exist_ok=True)
    date_str  = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path  = f"reports/nifty50_screener_{date_str}.csv"
    txt_path  = f"reports/nifty50_summary_{date_str}.txt"
    resume_csv = "reports/nifty50_screener_latest.csv"

    # ── Banner ────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"  [bold cyan]Nifty 50 Batch Screener v1.0[/bold cyan]\n"
        f"  Stocks to analyze: [bold]{len(tickers)}[/bold]\n"
        f"  Mode: {'Resume (skip done)' if resume else 'Fresh run'}\n"
        f"  Est. time: [bold]{len(tickers)*5//60}m {len(tickers)*5%60}s[/bold] (~5s/stock)\n"
        f"  Output: [dim]{csv_path}[/dim]",
        title="[bold yellow]📊 NSE AI Screener[/bold yellow]",
        border_style="cyan", padding=(0, 2)
    ))
    console.print()

    # ── Resume: load previous ─────────────────────────────────
    prev_results = load_previous_results(resume_csv) if resume else {}

    results   = []
    t_start   = time.time()
    n         = len(tickers)

    # ── Progress bar ──────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Scanning Nifty 50...", total=n)

        for i, ticker in enumerate(tickers, 1):
            # Check resume cache
            if resume and ticker in prev_results:
                r = {k: prev_results[ticker].get(k, 0) for k in [
                    "ticker","name","sector","sector_type","price",
                    "fundamental","valuation","banking","management","sentiment",
                    "overall","iv","upside","signal","grade",
                    "roe","roce","roa","pe","pb","market_cap_cr",
                    "rsi","return_3m","status","elapsed","error"
                ]}
                r["status"] = "done (cached)"
                results.append(r)
                progress.update(task, advance=1,
                    description=f"[cached] {ticker:12} Overall: {r['overall']:3}/100")
                continue

            # Live status update
            progress.update(task, description=f"[{i}/{n}] Analyzing {ticker:12}...")

            r = analyze_stock(ticker)
            results.append(r)

            # Progress update with result
            if r["status"] == "done":
                sig_icon = ("🟢" if r["overall"] >= 65 else
                            "🟡" if r["overall"] >= 50 else "🔴")
                progress.update(task, advance=1,
                    description=f"[{i}/{n}] {ticker:12} {sig_icon} {r['overall']:3}/100  "
                                f"({r['elapsed']:.1f}s)")
            else:
                progress.update(task, advance=1,
                    description=f"[{i}/{n}] [red]{ticker:12} ❌ ERROR[/red]")

            # Save intermediate CSV after every 5 stocks (crash safety)
            if i % 5 == 0:
                save_csv(results, resume_csv)

    elapsed_total = time.time() - t_start

    # ── Final leaderboard ─────────────────────────────────────
    console.print()
    console.print(build_leaderboard(results, top_n=top_n))

    # ── Sector breakdown ──────────────────────────────────────
    _print_sector_summary(results)

    # ── Key findings panels ───────────────────────────────────
    _print_key_findings(results)

    # ── Save outputs ──────────────────────────────────────────
    console.print()
    console.rule("[bold]📁 Saving Reports[/bold]")
    save_csv(results, csv_path)
    save_csv(results, resume_csv)   # always update latest
    save_summary(results, txt_path, elapsed_total)

    errors = [r for r in results if r["status"] == "error"]
    done   = [r for r in results if r["status"] == "done"]

    console.print()
    console.print(f"  ✅ [bold green]Done![/bold green] Analyzed {len(done)}/{n} stocks "
                  f"in {elapsed_total:.0f}s")
    if errors:
        console.print(f"  ❌ [red]{len(errors)} failed:[/red] "
                      f"{', '.join(r['ticker'] for r in errors)}")
    console.rule("[dim]Not SEBI-registered investment advice[/dim]")


# ============================================================
# SECTOR SUMMARY
# ============================================================

def _print_sector_summary(results: list):
    done = [r for r in results if r["status"] == "done"]
    if not done: return

    sectors = {}
    for r in done:
        st = r["sector_type"]
        if st not in sectors:
            sectors[st] = []
        sectors[st].append(r)

    console.print()
    console.rule("[bold white]📊 Sector Average Scores[/bold white]")
    t = Table(box=box.SIMPLE, header_style="bold white")
    t.add_column("Sector",    width=12)
    t.add_column("Stocks",    width=7,  justify="center")
    t.add_column("Avg Score", width=10, justify="center")
    t.add_column("Best Stock",width=14)
    t.add_column("Best Score",width=10, justify="center")
    t.add_column("Avg RSI",   width=8,  justify="center")
    t.add_column("Avg 3M Ret",width=10, justify="right")

    for st, stocks in sorted(sectors.items()):
        avg_score = sum(s["overall"] for s in stocks) / len(stocks)
        best      = max(stocks, key=lambda x: x["overall"])
        avg_rsi   = sum(s["rsi"] for s in stocks if s["rsi"]) / max(len([s for s in stocks if s["rsi"]]), 1)
        avg_ret   = sum(s["return_3m"] for s in stocks) / len(stocks)
        col       = _score_color(int(avg_score))
        t.add_row(
            st,
            str(len(stocks)),
            f"[{col}]{avg_score:.0f}[/{col}]",
            best["ticker"],
            f"[bold]{best['overall']}[/bold]",
            f"{avg_rsi:.0f}" if avg_rsi else "—",
            f"[{'green' if avg_ret>=0 else 'red'}]{avg_ret:+.1f}%[/]",
        )
    console.print(t)


# ============================================================
# KEY FINDINGS PANELS
# ============================================================

def _print_key_findings(results: list):
    done = sorted(
        [r for r in results if r["status"] == "done"],
        key=lambda x: x["overall"], reverse=True
    )
    if not done: return

    console.print()
    console.rule("[bold white]🔍 Key Findings[/bold white]")

    # Top 5 overall
    top5 = done[:5]
    top5_str = "  ".join(
        f"[bold cyan]{r['ticker']}[/bold cyan] ({r['overall']}/100)"
        for r in top5
    )

    # Best value (highest upside with good fundamentals)
    value_picks = sorted(
        [r for r in done if r["upside"] and r["overall"] >= 55],
        key=lambda x: x["upside"], reverse=True
    )[:3]
    value_str = "  ".join(
        f"[bold green]{r['ticker']}[/bold green] (+{r['upside']:.0f}%)"
        for r in value_picks
    ) or "None found"

    # Oversold gems (RSI < 35, Overall ≥ 55)
    oversold = sorted(
        [r for r in done if r["rsi"] and r["rsi"] < 35 and r["overall"] >= 55],
        key=lambda x: x["rsi"]
    )[:3]
    oversold_str = "  ".join(
        f"[bold yellow]{r['ticker']}[/bold yellow] (RSI {r['rsi']:.0f}, {r['overall']}/100)"
        for r in oversold
    ) or "None found"

    # Avoid list (lowest scoring)
    avoid = done[-5:][::-1]
    avoid_str = "  ".join(
        f"[red]{r['ticker']}[/red] ({r['overall']}/100)"
        for r in avoid
    )

    # High ROE stocks
    high_roe = sorted(
        [r for r in done if r["roe"] and r["roe"] > 20],
        key=lambda x: x["roe"], reverse=True
    )[:5]
    roe_str = "  ".join(
        f"[bold]{r['ticker']}[/bold] (ROE {r['roe']:.0f}%)"
        for r in high_roe
    ) or "None found"

    findings = (
        f"[bold]🥇 Top 5 Overall:[/bold]\n  {top5_str}\n\n"
        f"[bold]💰 Best Value Picks (highest upside):[/bold]\n  {value_str}\n\n"
        f"[bold]📉 Oversold Gems (RSI<35, Score≥55):[/bold]\n  {oversold_str}\n\n"
        f"[bold]💪 High ROE Leaders (>20%):[/bold]\n  {roe_str}\n\n"
        f"[bold]⚠️  Bottom 5 (lowest score):[/bold]\n  {avoid_str}"
    )
    console.print(Panel(findings, title="🔍 Screener Insights",
                        border_style="cyan", padding=(1, 2)))


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="NSE Nifty 50 Batch Screener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python screener.py                        # Full Nifty 50
  python screener.py --sector IT            # IT stocks only
  python screener.py --sector BANKING       # Banking only
  python screener.py --top 10              # Show top 10
  python screener.py --tickers TCS INFY HDFCBANK
  python screener.py --resume              # Continue interrupted run
        """
    )
    p.add_argument("--sector",  type=str, default=None,
                   help=f"Filter by sector: {', '.join(SECTOR_FILTER.keys())}")
    p.add_argument("--top",     type=int, default=0,
                   help="Show only top N stocks in leaderboard")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Custom list of NSE tickers")
    p.add_argument("--resume",  action="store_true",
                   help="Skip stocks already analyzed (uses latest CSV)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine ticker list
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.sector:
        s = args.sector.upper()
        if s not in SECTOR_FILTER:
            console.print(f"[red]Unknown sector '{s}'. Choose from: {', '.join(SECTOR_FILTER.keys())}[/red]")
            sys.exit(1)
        tickers = SECTOR_FILTER[s]
        console.print(f"[cyan]Filtering to {s} sector: {', '.join(tickers)}[/cyan]")
    else:
        tickers = NIFTY50

    run_screener(tickers=tickers, top_n=args.top, resume=args.resume)
