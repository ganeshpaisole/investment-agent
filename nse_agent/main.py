# ============================================================
# main.py — ENTRY POINT v1.3
# Sector-aware routing to specialized agents
#
# Generic agents (IT, FMCG, Auto, Pharma, Chemicals):
#   --mode fundamental | valuation | management | full | all
#
# Specialized agents:
#   Banking/NBFC  → BankingAgent   (NIM, NPA, CAR, CASA, ROA)
#   Insurance     → InsuranceAgent (Claims ratio, VNB, Solvency)
#
# Usage:
#   python main.py --ticker TCS --mode all --no-gpt
#   python main.py --ticker HDFCBANK --mode all --no-gpt
#   python main.py --ticker HDFCLIFE --mode all --no-gpt
# ============================================================

import argparse
import sys
from loguru import logger
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

from agents.fundamental_agent  import FundamentalAnalysisAgent
from agents.valuation_agent    import ValuationAgent
from agents.management_agent   import ManagementQualityAgent
from agents.banking_agent      import BankingAgent
from agents.insurance_agent    import InsuranceAgent
from agents.sentiment_agent    import MarketSentimentAgent
from agents.gpt_analyst        import GPTAnalyst


from utils.data_fetcher        import NSEDataFetcher
from utils.sector_classifier   import SectorClassifier
from utils.report_printer      import print_fundamental_report
from utils.valuation_printer   import print_valuation_report
from utils.management_printer  import print_management_report
from utils.specialized_printer import print_banking_report, print_insurance_report
from utils.sentiment_printer   import print_sentiment_report
from utils.sentiment_printer   import print_sentiment_report

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level="INFO",
    colorize=True
)

console = Console()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="NSE Stock AI Agent — Sector-Aware Graham + Buffett Framework"
    )
    parser.add_argument("--ticker", type=str,
                        help="NSE stock ticker (e.g. TCS, HDFCBANK, HDFCLIFE)")
    parser.add_argument("--no-gpt", action="store_true", help="Skip GPT-4 analysis")
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["fundamental", "valuation", "management", "sentiment", "full", "all"],
        help="Analysis mode (default: full)"
    )
    return parser.parse_args()


def show_banner():
    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║      NSE STOCK FORECAST AI AGENT  v1.3          ║[/bold cyan]")
    console.print("[bold cyan]║      Sector-Aware Graham + Buffett Framework    ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════╝[/bold cyan]")
    console.print()
    console.print("  Generic  : TCS, INFY, RELIANCE, ITC, MARUTI, SUNPHARMA")
    console.print("  Banking  : HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK")
    console.print("  Insurance: HDFCLIFE, SBILIFE, ICICIPRU, STARHEALTH, LICI")
    console.print()
    console.print("  Modes    : fundamental | valuation | management | sentiment | full | all")
    console.print()


def get_ticker_from_user() -> str:
    return Prompt.ask("[bold yellow]Enter NSE Stock Ticker[/bold yellow]").upper().strip()


# ── Quick sector info fetch (lightweight) ────────────────────
def _get_sector_info(ticker: str) -> dict:
    """Fetch just the info dict for sector classification."""
    try:
        fetcher = NSEDataFetcher(ticker)
        return fetcher.get_stock_info()
    except:
        return {}


# ── Sector warning panel ─────────────────────────────────────
def _show_sector_warnings(classifier: SectorClassifier):
    warnings = classifier.get_warnings()
    if warnings:
        content = "\n".join(f"  {w}" for w in warnings)
        console.print(Panel(
            content,
            title="[bold yellow]⚠️  SECTOR COMPATIBILITY WARNINGS[/bold yellow]",
            border_style="yellow",
            padding=(0, 2),
        ))
        console.print()


# ════════════════════════════════════════════════════════════
# GENERIC ANALYSIS (IT, FMCG, Auto, Pharma, etc.)
# ════════════════════════════════════════════════════════════

def run_generic_analysis(ticker: str, use_gpt: bool, mode: str,
                         classifier: SectorClassifier):

    fund_result = None
    val_result  = None
    mgmt_result = None

    # Step 1: Fundamental
    if mode in ("fundamental", "full", "all") and classifier.should_run("fundamental"):
        console.print("[cyan]━━━ Step 1: Fundamental Analysis Engine ━━━[/cyan]")
        try:
            agent       = FundamentalAnalysisAgent(ticker)
            fund_result = agent.analyze()
        except Exception as e:
            console.print(f"[red]❌ Fundamental analysis failed: {e}[/red]")
            sys.exit(1)
    elif mode in ("fundamental", "full", "all"):
        console.print(f"[yellow]⏭️  Skipping Fundamental Analysis — not suitable for {classifier.sector_type}[/yellow]")

    # Step 2: GPT-4
    buffett_eval = "GPT-4 skipped (--no-gpt flag)"
    narrative    = "GPT-4 skipped (--no-gpt flag)"
    if fund_result and use_gpt and mode in ("fundamental", "full", "all"):
        console.print("[cyan]━━━ Step 2: GPT-4 Expert Analysis ━━━[/cyan]")
        try:
            analyst      = GPTAnalyst()
            buffett_eval = analyst.evaluate_buffett_questions(fund_result)
            narrative    = analyst.generate_narrative(fund_result)
        except ValueError as e:
            console.print(f"[red]❌ {e}[/red]")
            buffett_eval = "⚠️ Add OPENAI_API_KEY to .env to enable GPT-4"
            narrative    = "⚠️ Add OPENAI_API_KEY to .env to enable GPT-4"
        except Exception as e:
            console.print(f"[yellow]⚠️ GPT-4 failed: {e}[/yellow]")

    # Step 3: DCF Valuation
    if mode in ("valuation", "full", "all") and classifier.should_run("valuation"):
        console.print("[cyan]━━━ Step 3: DCF Valuation Engine ━━━[/cyan]")
        try:
            val_agent  = ValuationAgent(ticker)
            val_result = val_agent.analyze()
        except Exception as e:
            console.print(f"[red]❌ Valuation engine failed: {e}[/red]")
    elif mode in ("valuation", "full", "all"):
        console.print(f"[yellow]⏭️  Skipping DCF Valuation — not suitable for {classifier.sector_type}[/yellow]")

    # Step 4: Management
    if mode in ("management", "all") and classifier.should_run("management"):
        console.print("[cyan]━━━ Step 4: Management Quality Agent ━━━[/cyan]")
        try:
            mgmt_agent  = ManagementQualityAgent(ticker)
            mgmt_result = mgmt_agent.analyze()
        except Exception as e:
            console.print(f"[red]❌ Management analysis failed: {e}[/red]")

    # Step 5: Sentiment
    sent_result = None
    if mode in ("sentiment", "all"):
        console.print("[cyan]━━━ Step 5: 📡 Market Sentiment Agent ━━━[/cyan]")
        try:
            sent_agent  = MarketSentimentAgent(ticker)
            sent_result = sent_agent.analyze()
        except Exception as e:
            console.print(f"[red]❌ Sentiment analysis failed: {e}[/red]")

    # Print reports
    if fund_result:
        print_fundamental_report(fund_result, buffett_eval, narrative)
    if val_result:
        print_valuation_report(val_result)
    if mgmt_result:
        print_management_report(mgmt_result)
    if sent_result:
        print_sentiment_report(sent_result)

    # Combined summary
    if mode == "all" and fund_result and val_result and mgmt_result:
        _print_combined_summary(ticker, fund_result, val_result, mgmt_result, sent_result)

    return fund_result, val_result, mgmt_result, sent_result


# ════════════════════════════════════════════════════════════
# BANKING ANALYSIS
# ════════════════════════════════════════════════════════════

def run_banking_analysis(ticker: str, use_gpt: bool, mode: str,
                         classifier: SectorClassifier):

    console.print("[cyan]━━━ 🏦 Banking Analysis Engine ━━━[/cyan]")
    banking_result = None
    mgmt_result    = None

    try:
        agent          = BankingAgent(ticker)
        banking_result = agent.analyze()
    except Exception as e:
        console.print(f"[red]❌ Banking analysis failed: {e}[/red]")
        return None, None

    # Management still works for banks
    if mode in ("management", "all"):
        console.print("[cyan]━━━ 👔 Management Quality Agent ━━━[/cyan]")
        try:
            mgmt_agent  = ManagementQualityAgent(ticker)
            mgmt_result = mgmt_agent.analyze()
        except Exception as e:
            console.print(f"[red]❌ Management analysis failed: {e}[/red]")

    # Print reports
    if banking_result:
        print_banking_report(banking_result)
    if mgmt_result:
        print_management_report(mgmt_result)

    # Sentiment
    sent_result = None
    if mode in ("sentiment", "all"):
        console.print("[cyan]━━━ 📡 Market Sentiment Agent ━━━[/cyan]")
        try:
            sent_agent  = MarketSentimentAgent(ticker)
            sent_result = sent_agent.analyze()
        except Exception as e:
            console.print(f"[red]❌ Sentiment analysis failed: {e}[/red]")
    if sent_result:
        print_sentiment_report(sent_result)

    # Combined summary for banking
    if mode == "all" and banking_result and mgmt_result:
        _print_banking_combined_summary(ticker, banking_result, mgmt_result, sent_result)

    return banking_result, mgmt_result, sent_result


# ════════════════════════════════════════════════════════════
# INSURANCE ANALYSIS
# ════════════════════════════════════════════════════════════

def run_insurance_analysis(ticker: str, use_gpt: bool, mode: str,
                           classifier: SectorClassifier):

    console.print("[cyan]━━━ 🛡️  Insurance Analysis Engine ━━━[/cyan]")
    ins_result  = None
    mgmt_result = None

    try:
        agent      = InsuranceAgent(ticker)
        ins_result = agent.analyze()
    except Exception as e:
        console.print(f"[red]❌ Insurance analysis failed: {e}[/red]")
        return None, None

    if mode in ("management", "all"):
        console.print("[cyan]━━━ 👔 Management Quality Agent ━━━[/cyan]")
        try:
            mgmt_agent  = ManagementQualityAgent(ticker)
            mgmt_result = mgmt_agent.analyze()
        except Exception as e:
            console.print(f"[red]❌ Management analysis failed: {e}[/red]")

    if ins_result:
        print_insurance_report(ins_result)
    if mgmt_result:
        print_management_report(mgmt_result)

    if mode == "all" and ins_result and mgmt_result:
        _print_insurance_combined_summary(ticker, ins_result, mgmt_result)

    return ins_result, mgmt_result


# ════════════════════════════════════════════════════════════
# COMBINED SUMMARY TABLES
# ════════════════════════════════════════════════════════════

def _print_combined_summary(ticker, fund, val, mgmt, sent=None):
    from rich.table import Table
    from rich import box
    console.print()
    console.rule(f"[bold white]⭐ COMBINED ANALYSIS SUMMARY — {ticker}[/bold white]")
    t = Table(box=box.ROUNDED, header_style="bold white")
    t.add_column("Agent",       width=28, style="bold")
    t.add_column("Score",       width=14, justify="center")
    t.add_column("Grade",       width=10, justify="center")
    t.add_column("Key Signal",  width=44)

    fs = fund["total_score"]; fc = "green" if fs>=70 else "yellow" if fs>=50 else "red"
    t.add_row("📊 Fundamental", f"[{fc}]{fs}/100[/{fc}]", fund["grade"], fund.get("verdict","—"))

    vs  = val["valuation_score"]
    iv  = val["composite"]["composite_iv"]
    cmp = val["current_price"]
    up  = val["recommendation"]["upside_pct"]
    vc  = "green" if up>=15 else "yellow" if up>=0 else "red"
    t.add_row("💰 DCF Valuation", f"[{vc}]{vs}/100[/{vc}]", "—",
              f"IV ₹{iv:,.0f} | CMP ₹{cmp:,.0f} | Upside {up:+.1f}%")

    ms = mgmt["total_score"]; mc = "green" if ms>=65 else "yellow" if ms>=45 else "red"
    t.add_row("👔 Management", f"[{mc}]{ms}/100[/{mc}]", mgmt["grade"], mgmt["verdict"])

    if sent:
        ss  = sent["total_score"]
        sc  = "green" if ss>=65 else "yellow" if ss>=40 else "red"
        t.add_row("📡 Sentiment", f"[{sc}]{ss}/100[/{sc}]", "—", sent["signal"])
        overall = round(fs*0.35 + vs*0.25 + ms*0.25 + ss*0.15, 0)
        weight_note = "Fund 35% + Val 25% + Mgmt 25% + Sent 15%"
    else:
        overall = round(fs*0.4 + vs*0.3 + ms*0.3, 0)
        weight_note = "Fund 40% + Val 30% + Mgmt 30%"

    oc = "green" if overall>=65 else "yellow" if overall>=45 else "red"
    t.add_row("[bold]⭐ OVERALL[/bold]",
              f"[{oc}][bold]{overall:.0f}/100[/bold][/{oc}]", "—",
              f"[bold]{weight_note}[/bold]")
    console.print(t)
    console.print()
    console.print(f"  [bold]Signal:[/bold]  {val['recommendation']['signal']}")
    console.print(f"  [bold]Entry Zone (30% MoS):[/bold]  ₹{val['recommendation']['entry_price']:,.2f}")
    console.print(f"  [bold]Target:[/bold]  ₹{val['recommendation']['target_price']:,.2f}")
    console.print(f"  [bold]Buffett:[/bold] {mgmt['buffett_pass']}/6 passed")
    if sent:
        console.print(f"  [bold]Sentiment:[/bold] {sent['signal']} ({sent['total_score']}/100)")
    console.print()
    console.rule("[dim]Not SEBI-registered investment advice[/dim]")


def _print_banking_combined_summary(ticker, banking, mgmt, sent=None):
    from rich.table import Table
    from rich import box
    console.print()
    console.rule(f"[bold cyan]⭐ COMBINED BANKING SUMMARY — {ticker}[/bold cyan]")
    t = Table(box=box.ROUNDED, header_style="bold white")
    t.add_column("Agent",       width=28, style="bold")
    t.add_column("Score",       width=14, justify="center")
    t.add_column("Grade",       width=10, justify="center")
    t.add_column("Key Signal",  width=44)

    bs = banking["total_score"]; bc = "green" if bs>=65 else "yellow" if bs>=45 else "red"
    up = banking["upside_pct"]; iv = banking["intrinsic_value"]
    cmp = banking["current_price"]
    t.add_row("🏦 Banking Analysis", f"[{bc}]{bs}/100[/{bc}]", banking["grade"],
              f"{banking['verdict']}")
    t.add_row("📊 Valuation (P/B)", "—", "—",
              f"IV ₹{iv:,.0f} | CMP ₹{cmp:,.0f} | {up:+.1f}%")

    ms = mgmt["total_score"]; mc = "green" if ms>=65 else "yellow" if ms>=45 else "red"
    t.add_row("👔 Management", f"[{mc}]{ms}/100[/{mc}]", mgmt["grade"], mgmt["verdict"])

    if sent:
        ss = sent["total_score"]
        sc = "green" if ss>=65 else "yellow" if ss>=40 else "red"
        t.add_row("📡 Sentiment", f"[{sc}]{ss}/100[/{sc}]", "—", sent["signal"])
        overall = round(bs*0.45 + ms*0.35 + ss*0.20, 0)
        weight_note = "Banking 45% + Mgmt 35% + Sent 20%"
    else:
        overall = round(bs*0.55 + ms*0.45, 0)
        weight_note = "Banking 55% + Mgmt 45%"

    oc = "green" if overall>=65 else "yellow" if overall>=45 else "red"
    t.add_row("[bold]⭐ OVERALL[/bold]",
              f"[{oc}][bold]{overall:.0f}/100[/bold][/{oc}]", "—",
              f"[bold]{weight_note}[/bold]")
    console.print(t)
    console.print()
    console.print(f"  [bold]ROA:[/bold]  {banking['banking_ratios']['roa']:.2f}%   "
                  f"[bold]ROE:[/bold]  {banking['banking_ratios']['roe']:.1f}%   "
                  f"[bold]P/B:[/bold]  {banking['banking_ratios']['pb_ratio']:.2f}x")
    console.print(f"  [bold]Buffett:[/bold] {banking['buffett_pass']}/6 passed")
    if sent:
        console.print(f"  [bold]Sentiment:[/bold] {sent['signal']} ({sent['total_score']}/100)")
    console.print()
    console.rule("[dim]Not SEBI-registered investment advice[/dim]")


def _print_insurance_combined_summary(ticker, ins_result, mgmt):
    from rich.table import Table
    from rich import box
    console.print()
    console.rule(f"[bold magenta]⭐ COMBINED INSURANCE SUMMARY — {ticker}[/bold magenta]")
    t = Table(box=box.ROUNDED, header_style="bold white")
    t.add_column("Agent",       width=28, style="bold")
    t.add_column("Score",       width=14, justify="center")
    t.add_column("Grade",       width=10, justify="center")
    t.add_column("Key Signal",  width=44)

    is_ = ins_result["total_score"]; ic = "green" if is_>=65 else "yellow" if is_>=45 else "red"
    iv  = ins_result["intrinsic_value"]; cmp = ins_result["current_price"]
    up  = ins_result["upside_pct"]
    t.add_row("🛡️  Insurance Analysis", f"[{ic}]{is_}/100[/{ic}]", ins_result["grade"],
              f"{ins_result['verdict']}")
    t.add_row("📊 Valuation (P/EV)", "—", "—",
              f"Fair Value ₹{iv:,.0f} | CMP ₹{cmp:,.0f} | {up:+.1f}%")

    ms = mgmt["total_score"]; mc = "green" if ms>=65 else "yellow" if ms>=45 else "red"
    t.add_row("👔 Management", f"[{mc}]{ms}/100[/{mc}]", mgmt["grade"], mgmt["verdict"])

    overall = round(is_*0.55 + ms*0.45, 0)
    oc = "green" if overall>=65 else "yellow" if overall>=45 else "red"
    t.add_row("[bold]⭐ OVERALL[/bold]",
              f"[{oc}][bold]{overall:.0f}/100[/bold][/{oc}]", "—",
              "[bold]Insurance 55% + Mgmt 45%[/bold]")
    console.print(t)
    ins = ins_result["insurance_ratios"]
    console.print()
    console.print(f"  [bold]ROE:[/bold]  {ins['roe']:.1f}%   "
                  f"[bold]Rev CAGR:[/bold]  {ins['revenue_cagr']:.1f}%   "
                  f"[bold]P/B:[/bold]  {ins['pb_ratio']:.2f}x")
    console.print(f"  [bold]Buffett:[/bold] {ins_result['buffett_pass']}/6 passed")
    console.print()
    console.rule("[dim]Not SEBI-registered investment advice[/dim]")


# ════════════════════════════════════════════════════════════
# MASTER ROUTER
# ════════════════════════════════════════════════════════════

def run_analysis(ticker: str, use_gpt: bool = True, mode: str = "full"):
    console.print(f"\n[bold green]🚀 Starting {mode.upper()} analysis for {ticker}...[/bold green]\n")

    # ── Sector Classification ────────────────────────────────
    console.print("[dim]🏷️  Detecting sector...[/dim]")
    info       = _get_sector_info(ticker)
    classifier = SectorClassifier(ticker, info)
    classifier.print_sector_banner()
    _show_sector_warnings(classifier)

    # ── Route to appropriate analysis ───────────────────────
    if classifier.sector_type == "BANKING":
        return run_banking_analysis(ticker, use_gpt, mode, classifier)

    elif classifier.sector_type == "INSURANCE":
        return run_insurance_analysis(ticker, use_gpt, mode, classifier)

    elif classifier.sector_type == "NBFC":
        return run_banking_analysis(ticker, use_gpt, mode, classifier)

    elif classifier.sector_type == "REIT":
        console.print(Panel(
            "  ⛔ Real Estate sector — DCF Valuation skipped.\n"
            "  Use NAV-based valuation (Net Asset Value per unit).\n"
            "  Fundamental and Management agents still running.",
            title="[bold yellow]🏷️  REIT — Limited Compatibility[/bold yellow]",
            border_style="yellow", padding=(0, 2)
        ))
        console.print()
        return run_generic_analysis(ticker, use_gpt, mode, classifier)

    else:
        # Generic: IT, FMCG, Pharma, Cyclical, Utility, Unknown
        return run_generic_analysis(ticker, use_gpt, mode, classifier)


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args   = parse_arguments()
    show_banner()
    ticker = args.ticker if args.ticker else get_ticker_from_user()
    run_analysis(ticker=ticker, use_gpt=not args.no_gpt, mode=args.mode)
