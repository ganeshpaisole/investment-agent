# ============================================================
# utils/management_printer.py — VERSION 1.0
# Beautiful terminal report for Management Quality Agent
# ============================================================

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box

console = Console()


def print_management_report(result: dict):
    """Print the full management quality report to terminal."""

    ticker       = result["ticker"]
    company_name = result["company_name"]
    sector       = result["sector"]
    total_score  = result["total_score"]
    grade        = result["grade"]
    verdict      = result["verdict"]
    summary      = result["summary"]
    scores       = result["scores"]
    checks       = result["buffett_checks"]
    buffett_pass = result["buffett_pass"]
    all_flags    = result["all_flags"]
    all_positives = result["all_positives"]

    console.print()
    console.rule(f"[bold yellow]👔 MANAGEMENT QUALITY REPORT — {ticker}[/bold yellow]")
    console.print(f"  [bold]{company_name}[/bold]  |  {sector}")
    console.print()

    # ── Verdict Panel ────────────────────────────────────────
    score_bar = _make_score_bar(total_score, 100)
    verdict_color = (
        "green" if total_score >= 65 else
        "yellow" if total_score >= 45 else
        "red"
    )

    panel_content = (
        f"  {verdict} — {summary}\n\n"
        f"  Management Score : [bold]{total_score}/100[/bold]  Grade: [bold]{grade}[/bold]\n"
        f"  {score_bar}\n\n"
        f"  Buffett Checklist: {buffett_pass}/6 passed"
    )
    console.print(Panel(
        panel_content,
        title="[bold]Management Verdict[/bold]",
        border_style=verdict_color,
        padding=(0, 2),
    ))
    console.print()

    # ── Dimension Scores ─────────────────────────────────────
    dim_table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title="📊 Score Breakdown by Dimension",
        title_style="bold",
    )
    dim_table.add_column("Dimension",         style="bold", width=28)
    dim_table.add_column("Score",             justify="center", width=12)
    dim_table.add_column("Max",               justify="center", width=8)
    dim_table.add_column("Rating",            justify="center", width=14)
    dim_table.add_column("Top Signal",        width=42)

    dim_config = [
        ("promoter",   "👥 Promoter Commitment"),
        ("capital",    "💼 Capital Allocation"),
        ("integrity",  "📋 Earnings Integrity"),
        ("governance", "🏛️  Governance Quality"),
    ]

    for key, label in dim_config:
        dim      = scores[key]
        sc       = dim["score"]
        mx       = dim["max"]
        pct      = sc / mx * 100
        rating   = _rating_label(pct)
        top_flag = (dim["positives"][0] if dim["positives"] else
                    dim["flags"][0] if dim["flags"] else "—")
        # Truncate to fit
        top_flag = top_flag[:42] if len(top_flag) > 42 else top_flag
        color    = "green" if pct >= 65 else "yellow" if pct >= 45 else "red"
        dim_table.add_row(label, f"[{color}]{sc}/{mx}[/{color}]", str(mx), rating, top_flag)

    console.print(dim_table)
    console.print()

    # ── Buffett's 6 Questions ────────────────────────────────
    bq_table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold magenta",
        title="🎯 Buffett's 6 Management Questions",
        title_style="bold",
    )
    bq_table.add_column("#",        width=4,  justify="center")
    bq_table.add_column("Question", width=48)
    bq_table.add_column("Result",   width=36)

    for i, check in enumerate(checks, 1):
        status = "✅ PASS" if check["pass"] else "❌ FAIL"
        color  = "green" if check["pass"] else "red"
        bq_table.add_row(
            str(i),
            check["question"],
            f"[{color}]{status}[/{color}]  {check['detail']}"
        )

    console.print(bq_table)
    console.print()

    # ── Green Flags ──────────────────────────────────────────
    if all_positives:
        console.print("[bold green]✅ GREEN FLAGS[/bold green]")
        for p in all_positives[:6]:   # top 6
            console.print(f"   {p}")
        console.print()

    # ── Red Flags ────────────────────────────────────────────
    if all_flags:
        console.print("[bold red]⚠️  RED FLAGS & WARNINGS[/bold red]")
        for f in all_flags:
            console.print(f"   {f}")
        console.print()

    # ── Key Details ─────────────────────────────────────────
    details_table = Table(
        box=box.SIMPLE,
        show_header=False,
        title="📌 Key Management Metrics",
        title_style="bold",
    )
    details_table.add_column("Metric", style="dim", width=32)
    details_table.add_column("Value",  width=20)
    details_table.add_column("Metric", style="dim", width=32)
    details_table.add_column("Value",  width=20)

    p  = scores["promoter"]["details"]
    ca = scores["capital"]["details"]
    ei = scores["integrity"]["details"]
    gv = scores["governance"]["details"]

    rows = [
        ("Promoter Holding",    f"{p.get('promoter_holding_pct', 0):.1f}%",
         "ROCE",                f"{ca.get('roce', 0):.1f}%"),
        ("Institutional Holding", f"{p.get('institutional_holding_pct', 0):.1f}%",
         "ROE",                 f"{ca.get('roe', 0):.1f}%"),
        ("Promoter Pledge",     f"{p.get('promoter_pledge_pct', 0):.1f}%",
         "CFO/PAT Ratio",       f"{ei.get('cfo_to_net_income', 0):.2f}x"),
        ("Dividend Yield",      f"{ca.get('dividend_yield', 0):.1f}%",
         "Net Margin",          f"{ei.get('net_margin', 0):.1f}%"),
        ("Payout Ratio",        f"{ca.get('payout_ratio', 0):.0f}%",
         "Debt/Equity",         f"{gv.get('de_ratio', 0):.2f}x"),
        ("Years of Data",       f"{gv.get('years_of_data', 0)} yrs",
         "Interest Coverage",   f"{gv.get('int_coverage', 0):.1f}x"),
    ]

    for r in rows:
        details_table.add_row(*[str(x) for x in r])

    console.print(details_table)
    console.print()
    console.rule("[dim]⚠️  Based on publicly available data. Not SEBI-registered investment advice.[/dim]")
    console.print()


def _make_score_bar(score: int, max_score: int, width: int = 30) -> str:
    filled = int(score / max_score * width)
    color  = "green" if score >= 65 else "yellow" if score >= 45 else "red"
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/{color}] {score}/{max_score}"


def _rating_label(pct: float) -> str:
    if pct >= 85:   return "[green]Exceptional[/green]"
    elif pct >= 70: return "[green]Excellent[/green]"
    elif pct >= 55: return "[cyan]Good[/cyan]"
    elif pct >= 40: return "[yellow]Average[/yellow]"
    elif pct >= 25: return "[orange3]Weak[/orange3]"
    else:           return "[red]Poor[/red]"
