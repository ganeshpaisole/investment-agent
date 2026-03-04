# ============================================================
# utils/specialized_printer.py — VERSION 1.0
# Report printer for Banking and Insurance agents
# ============================================================

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


# ════════════════════════════════════════════════════════════
# BANKING REPORT
# ════════════════════════════════════════════════════════════

def print_banking_report(result: dict):
    ticker       = result["ticker"]
    company_name = result["company_name"]
    total_score  = result["total_score"]
    grade        = result["grade"]
    verdict      = result["verdict"]
    scores       = result["scores"]
    checks       = result["buffett_checks"]
    buffett_pass = result["buffett_pass"]
    all_flags    = result["all_flags"]
    all_positives = result["all_positives"]
    banking      = result["banking_ratios"]
    iv           = result["intrinsic_value"]
    upside       = result["upside_pct"]
    cmp          = result["current_price"]

    console.print()
    console.rule(f"[bold cyan]🏦 BANKING ANALYSIS REPORT — {ticker}[/bold cyan]")
    console.print(f"  [bold]{company_name}[/bold]  |  {result['industry']}")
    console.print()

    # Verdict panel
    color = "green" if total_score >= 65 else "yellow" if total_score >= 45 else "red"
    upside_str = f"+{upside:.1f}%" if upside >= 0 else f"{upside:.1f}%"
    panel_content = (
        f"  {verdict}\n\n"
        f"  Banking Score   : [bold]{total_score}/100[/bold]  Grade: [bold]{grade}[/bold]\n"
        f"  {_make_bar(total_score, 100)}\n\n"
        f"  Current Price   : ₹{cmp:,.2f}\n"
        f"  P/B Intrinsic   : ₹{iv:,.2f}   ({upside_str} vs CMP)\n"
        f"  Buffett Checks  : {buffett_pass}/6 passed"
    )
    console.print(Panel(panel_content, title="[bold]Banking Verdict[/bold]",
                        border_style=color, padding=(0, 2)))
    console.print()

    # Dimension scores
    dim_table = Table(box=box.ROUNDED, header_style="bold cyan",
                      title="📊 Score Breakdown", title_style="bold")
    dim_table.add_column("Dimension",    style="bold", width=26)
    dim_table.add_column("Score",        justify="center", width=12)
    dim_table.add_column("Rating",       justify="center", width=14)
    dim_table.add_column("Top Signal",   width=44)

    dims = [
        ("asset_quality", "🏦 Asset Quality (NPA/ROA)"),
        ("profitability",  "💹 Profitability (ROA/ROE)"),
        ("liability",      "💰 Liability Franchise"),
        ("valuation",      "📊 Valuation (P/B / DDM)"),
    ]
    for key, label in dims:
        d   = scores[key]
        sc  = d["score"]; mx = d["max"]
        pct = sc / mx * 100
        top = (d["positives"][0] if d["positives"] else d["flags"][0] if d["flags"] else "—")[:44]
        c   = "green" if pct >= 65 else "yellow" if pct >= 45 else "red"
        dim_table.add_row(label, f"[{c}]{sc}/{mx}[/{c}]", _rating(pct), top)
    console.print(dim_table)
    console.print()

    # Buffett checklist
    bq = Table(box=box.SIMPLE, header_style="bold magenta",
               title="🎯 Buffett's 6 Banking Questions", title_style="bold")
    bq.add_column("#",        width=4, justify="center")
    bq.add_column("Question", width=50)
    bq.add_column("Result",   width=36)
    for i, c in enumerate(checks, 1):
        status = "✅ PASS" if c["pass"] else "❌ FAIL"
        col    = "green" if c["pass"] else "red"
        bq.add_row(str(i), c["question"], f"[{col}]{status}[/{col}]  {c['detail']}")
    console.print(bq)
    console.print()

    # Key metrics
    _print_banking_metrics(banking, scores)

    # Flags
    if all_positives:
        console.print("[bold green]✅ GREEN FLAGS[/bold green]")
        for p in all_positives[:6]: console.print(f"   {p}")
        console.print()
    if all_flags:
        console.print("[bold red]⚠️  RED FLAGS[/bold red]")
        for f in all_flags: console.print(f"   {f}")
        console.print()

    console.rule("[dim]⚠️  Banking analysis based on public data. Not SEBI-registered advice.[/dim]")
    console.print()


def _print_banking_metrics(banking: dict, scores: dict):
    t = Table(box=box.SIMPLE, show_header=False,
              title="📌 Key Banking Metrics", title_style="bold")
    t.add_column("Metric", style="dim", width=28)
    t.add_column("Value",  width=16)
    t.add_column("Metric", style="dim", width=28)
    t.add_column("Value",  width=16)

    vd = scores["valuation"].get("details", {})
    rows = [
        ("ROA",               f"{banking['roa']:.2f}%",
         "P/B Ratio",         f"{banking['pb_ratio']:.2f}x"),
        ("ROE",               f"{banking['roe']:.1f}%",
         "Justified P/B",     f"{vd.get('justified_pb',0):.2f}x"),
        ("GNPA",              f"{banking['gnpa_pct']:.1f}%" if banking['gnpa_pct'] > 0 else "N/A",
         "Book Value/Share",  f"₹{banking['bv_ps']:.1f}"),
        ("NNPA",              f"{banking['nnpa_pct']:.1f}%" if banking['nnpa_pct'] > 0 else "N/A",
         "DDM Value",         f"₹{vd.get('ddm_value',0):.0f}" if vd.get('ddm_value',0) > 0 else "N/A"),
        ("CAR",               f"{banking['car']:.1f}%" if banking['car'] > 0 else "N/A",
         "Dividend Yield",    f"{banking['dividend_yield']:.1f}%"),
        ("CASA Ratio",        f"{banking['casa_ratio']:.1f}%" if banking['casa_ratio'] > 0 else "N/A",
         "Revenue CAGR",      f"{banking['revenue_cagr_3yr']:.1f}%"),
    ]
    for r in rows:
        t.add_row(*[str(x) for x in r])
    console.print(t)
    console.print()


# ════════════════════════════════════════════════════════════
# INSURANCE REPORT
# ════════════════════════════════════════════════════════════

def print_insurance_report(result: dict):
    ticker        = result["ticker"]
    company_name  = result["company_name"]
    total_score   = result["total_score"]
    grade         = result["grade"]
    verdict       = result["verdict"]
    scores        = result["scores"]
    checks        = result["buffett_checks"]
    buffett_pass  = result["buffett_pass"]
    all_flags     = result["all_flags"]
    all_positives = result["all_positives"]
    ins           = result["insurance_ratios"]
    iv            = result["intrinsic_value"]
    upside        = result["upside_pct"]
    cmp           = result["current_price"]

    console.print()
    console.rule(f"[bold magenta]🛡️  INSURANCE ANALYSIS REPORT — {ticker}[/bold magenta]")
    console.print(f"  [bold]{company_name}[/bold]  |  {result['industry']}")
    console.print()

    color = "green" if total_score >= 65 else "yellow" if total_score >= 45 else "red"
    upside_str = f"+{upside:.1f}%" if upside >= 0 else f"{upside:.1f}%"
    panel_content = (
        f"  {verdict}\n\n"
        f"  Insurance Score : [bold]{total_score}/100[/bold]  Grade: [bold]{grade}[/bold]\n"
        f"  {_make_bar(total_score, 100)}\n\n"
        f"  Current Price   : ₹{cmp:,.2f}\n"
        f"  Fair Value (P/B): ₹{iv:,.2f}   ({upside_str} vs CMP)\n"
        f"  Buffett Checks  : {buffett_pass}/6 passed"
    )
    console.print(Panel(panel_content, title="[bold]Insurance Verdict[/bold]",
                        border_style=color, padding=(0, 2)))
    console.print()

    dim_table = Table(box=box.ROUNDED, header_style="bold magenta",
                      title="📊 Score Breakdown", title_style="bold")
    dim_table.add_column("Dimension",    style="bold", width=28)
    dim_table.add_column("Score",        justify="center", width=12)
    dim_table.add_column("Rating",       justify="center", width=14)
    dim_table.add_column("Top Signal",   width=44)

    dims = [
        ("underwriting", "📋 Underwriting Quality"),
        ("growth",        "📈 Growth & Franchise"),
        ("financial",     "💪 Financial Strength"),
        ("valuation",     "📊 Valuation (P/B / P/EV)"),
    ]
    for key, label in dims:
        d   = scores[key]
        sc  = d["score"]; mx = d["max"]
        pct = sc / mx * 100
        top = (d["positives"][0] if d["positives"] else d["flags"][0] if d["flags"] else "—")[:44]
        c   = "green" if pct >= 65 else "yellow" if pct >= 45 else "red"
        dim_table.add_row(label, f"[{c}]{sc}/{mx}[/{c}]", _rating(pct), top)
    console.print(dim_table)
    console.print()

    bq = Table(box=box.SIMPLE, header_style="bold magenta",
               title="🎯 Buffett's 6 Insurance Questions", title_style="bold")
    bq.add_column("#",        width=4, justify="center")
    bq.add_column("Question", width=50)
    bq.add_column("Result",   width=36)
    for i, c in enumerate(checks, 1):
        status = "✅ PASS" if c["pass"] else "❌ FAIL"
        col    = "green" if c["pass"] else "red"
        bq.add_row(str(i), c["question"], f"[{col}]{status}[/{col}]  {c['detail']}")
    console.print(bq)
    console.print()

    _print_insurance_metrics(ins, scores)

    if all_positives:
        console.print("[bold green]✅ GREEN FLAGS[/bold green]")
        for p in all_positives[:6]: console.print(f"   {p}")
        console.print()
    if all_flags:
        console.print("[bold red]⚠️  RED FLAGS[/bold red]")
        for f in all_flags: console.print(f"   {f}")
        console.print()

    console.rule("[dim]⚠️  Insurance analysis based on public data. Not SEBI-registered advice.[/dim]")
    console.print()


def _print_insurance_metrics(ins: dict, scores: dict):
    t = Table(box=box.SIMPLE, show_header=False,
              title="📌 Key Insurance Metrics", title_style="bold")
    t.add_column("Metric", style="dim", width=28)
    t.add_column("Value",  width=16)
    t.add_column("Metric", style="dim", width=28)
    t.add_column("Value",  width=16)

    vd = scores["valuation"].get("details", {})
    rows = [
        ("Claims Ratio",
         f"{ins['claims_ratio']:.1f}%" if ins['claims_ratio'] > 0 else "N/A",
         "P/B Ratio",       f"{ins['pb_ratio']:.2f}x"),
        ("Combined Ratio",
         f"{ins['combined_ratio']:.1f}%" if ins['combined_ratio'] > 0 else "N/A",
         "Fair P/B",        f"{vd.get('fair_pb',0):.1f}x"),
        ("VNB Margin",
         f"{ins['vnb_margin']:.1f}%" if ins['vnb_margin'] > 0 else "N/A",
         "Fair Value",      f"₹{vd.get('fair_value',0):,.0f}"),
        ("Solvency Ratio",
         f"{ins['solvency_ratio']:.0f}%" if ins['solvency_ratio'] > 0 else "N/A",
         "ROE",             f"{ins['roe']:.1f}%"),
        ("Revenue CAGR",    f"{ins['revenue_cagr']:.1f}%",
         "Dividend Yield",  f"{ins['dividend_yield']:.1f}%"),
        ("P/E Ratio",       f"{ins['pe_ratio']:.1f}x",
         "Net Margin",      f"{ins['net_margin']:.1f}%"),
    ]
    for r in rows:
        t.add_row(*[str(x) for x in r])
    console.print(t)
    console.print()


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def _make_bar(score: int, max_score: int, width: int = 30) -> str:
    filled = int(score / max_score * width)
    color  = "green" if score >= 65 else "yellow" if score >= 45 else "red"
    return f"[{color}]{'█' * filled}{'░' * (width-filled)}[/{color}] {score}/{max_score}"


def _rating(pct: float) -> str:
    if pct >= 85:   return "[green]Exceptional[/green]"
    elif pct >= 70: return "[green]Excellent[/green]"
    elif pct >= 55: return "[cyan]Good[/cyan]"
    elif pct >= 40: return "[yellow]Average[/yellow]"
    elif pct >= 25: return "[orange3]Weak[/orange3]"
    else:           return "[red]Poor[/red]"
