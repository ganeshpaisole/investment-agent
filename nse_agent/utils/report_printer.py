# ============================================================
# utils/report_printer.py — VERSION 1.1
# Fixed: Dividend yield now reads from corrected ratios["dividend"]
# Fixed: Asset-light valuation signal displayed properly
# ============================================================

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.rule import Rule

console = Console()


def print_fundamental_report(result: dict, buffett_eval: str, narrative: str):
    """
    Print the full fundamental analysis report in the terminal.
    """
    ticker   = result["ticker"]
    score    = result["total_score"]
    grade    = result["grade"]
    rec      = result["recommendation"]
    val_sum  = result["valuation_summary"]
    ratios   = result["ratios"]
    cats     = result["categories"]
    info     = result.get("raw_info", {})

    # Grade color
    grade_color = {"A": "bright_green", "B": "green", "C": "yellow", "D": "orange1", "F": "red"}.get(grade, "white")

    # ---- HEADER ----
    console.print()
    console.rule(f"[bold cyan]NSE FUNDAMENTAL ANALYSIS REPORT — {ticker}[/bold cyan]")
    console.print()

    # Company name and sector
    company_name = info.get("longName", ticker)
    sector       = info.get("sector", "N/A")
    industry     = info.get("industry", "N/A")
    mktcap_cr    = (info.get("marketCap", 0) or 0) / 1e7
    console.print(f"  [bold white]{company_name}[/bold white]  |  {sector} → {industry}  |  Market Cap: ₹{mktcap_cr:,.0f} Cr")
    console.print()

    # ---- SCORE PANEL ----
    score_text = Text()
    score_text.append(f"  FUNDAMENTAL SCORE: ", style="bold white")
    score_text.append(f"{score}/100  ", style=f"bold {grade_color}")
    score_text.append(f"Grade: {grade}  ", style=f"bold {grade_color}")
    score_text.append(f"| {rec}", style="italic white")
    console.print(Panel(score_text, title="[bold]Overall Rating[/bold]", border_style="cyan"))
    console.print()

    # ---- CATEGORY SCORECARD ----
    cat_table = Table(
        title="📊 Category Scorecard", box=box.ROUNDED,
        border_style="cyan", header_style="bold cyan"
    )
    cat_table.add_column("Category", style="white", min_width=30)
    cat_table.add_column("Score", justify="center", min_width=8)
    cat_table.add_column("Max",   justify="center", min_width=5)
    cat_table.add_column("Status", justify="center", min_width=15)

    for cat_name, cat_data in cats.items():
        s   = cat_data["score"]
        m   = cat_data["max_score"]
        pct = (s / m) * 100
        status = "🟢 Strong" if pct >= 75 else ("🟡 Average" if pct >= 50 else "🔴 Weak")
        cat_table.add_row(cat_data["category"], str(s), str(m), status)

    cat_table.add_section()
    cat_table.add_row("[bold]TOTAL[/bold]", f"[bold]{score}[/bold]", "[bold]100[/bold]", "")
    console.print(cat_table)
    console.print()

    # ---- DETAILED BREAKDOWN ----
    console.print("[bold cyan]📋 Detailed Scoring Breakdown[/bold cyan]")
    console.print()
    for cat_name, cat_data in cats.items():
        console.print(f"[bold yellow]▶ {cat_data['category']}[/bold yellow]")
        for line in cat_data["breakdown"]:
            console.print(f"   {line}")
        console.print()

    # ---- KEY RATIOS ----
    console.rule("[bold cyan]Key Financial Ratios[/bold cyan]")
    console.print()

    val  = ratios["valuation"]
    prof = ratios["profitability"]
    lev  = ratios["leverage"]
    div  = ratios["dividend"]   # ✅ Use corrected dividend data

    val_table = Table(title="Valuation", box=box.SIMPLE_HEAVY, border_style="blue")
    val_table.add_column("Metric", style="white")
    val_table.add_column("Value",  justify="right", style="cyan")
    val_table.add_row("P/E Ratio",      str(val["pe_ratio"]))
    val_table.add_row("P/B Ratio",      str(val["pb_ratio"]))
    val_table.add_row("EV/EBITDA",      str(val["ev_ebitda"]))
    val_table.add_row("PEG Ratio",      str(val["peg_ratio"]))
    val_table.add_row("Earnings Yield", f"{val['earnings_yield']:.2f}%")

    prof_table = Table(title="Profitability", box=box.SIMPLE_HEAVY, border_style="green")
    prof_table.add_column("Metric", style="white")
    prof_table.add_column("Value",  justify="right", style="green")
    prof_table.add_row("ROE",           f"{prof['roe']:.2f}%")
    prof_table.add_row("ROCE",          f"{prof['roce']:.2f}%")
    prof_table.add_row("Net Margin",    f"{prof['net_margin']:.2f}%")
    prof_table.add_row("EBITDA Margin", f"{prof['ebitda_margin']:.2f}%")

    lev_table = Table(title="Leverage & Safety", box=box.SIMPLE_HEAVY, border_style="red")
    lev_table.add_column("Metric", style="white")
    lev_table.add_column("Value",  justify="right", style="red")
    lev_table.add_row("D/E Ratio",      str(lev["debt_to_equity"]))
    lev_table.add_row("Current Ratio",  str(lev["current_ratio"]))
    lev_table.add_row("Interest Cover", f"{lev['interest_coverage']:.1f}x")
    lev_table.add_row("Net Debt (₹Cr)", str(lev["net_debt_cr"]))

    # ✅ FIXED: Dividend table uses corrected values
    div_table = Table(title="Dividend & Cash Flow", box=box.SIMPLE_HEAVY, border_style="magenta")
    div_table.add_column("Metric", style="white")
    div_table.add_column("Value",  justify="right", style="magenta")
    div_table.add_row("Dividend Rate",  f"₹{div['dividend_rate']:.1f}")
    div_table.add_row("Dividend Yield", f"{div['dividend_yield']:.2f}%")   # ✅ Corrected
    div_table.add_row("Payout Ratio",   f"{div['payout_ratio']:.1f}%")
    div_table.add_row("FCF (₹Cr)",      str(ratios["cashflow"]["fcf_cr"]))
    div_table.add_row("FCF Yield",      f"{ratios['cashflow']['fcf_yield']:.2f}%")

    console.print(Columns([val_table, prof_table, lev_table, div_table]))
    console.print()

    # ---- VALUATION SUMMARY ----
    console.rule("[bold cyan]Valuation Assessment[/bold cyan]")
    console.print()
    v = val_sum
    console.print(f"  Current Market Price  : [bold white]₹{v['current_price']:,.2f}[/bold white]")
    console.print(f"  Graham Number         : [bold cyan]₹{v['graham_number']:,.2f}[/bold cyan]")

    # ✅ FIXED: Show appropriate message for asset-light companies
    if v.get("is_asset_light"):
        console.print(f"  Valuation Method      : [bold blue]🔵 Asset-Light Business — DCF Recommended[/bold blue]")
        console.print(f"  Graham Number Note    : [dim]Graham Number undervalues IT/Services companies.[/dim]")
        console.print(f"                          [dim]DCF Valuation Engine (coming next) will give true IV.[/dim]")
    else:
        mos = v["margin_of_safety"]
        mos_color = "green" if mos >= 30 else "red"
        console.print(f"  Margin of Safety      : [bold {mos_color}]{mos:.1f}%[/bold {mos_color}]")
        console.print(f"  Valuation Signal      : {v['valuation_signal']}")

    console.print()

    # ---- GPT-4 BUFFETT EVALUATION ----
    console.rule("[bold magenta]🤖 GPT-4: Buffett's 10 Questions Evaluation[/bold magenta]")
    console.print()
    console.print(Panel(buffett_eval, border_style="magenta", padding=(1, 2)))
    console.print()

    # ---- GPT-4 NARRATIVE ----
    console.rule("[bold green]📝 GPT-4: Analyst Report[/bold green]")
    console.print()
    console.print(Panel(narrative, border_style="green", padding=(1, 2)))
    console.print()

    # ---- WHAT'S NEXT ----
    console.rule("[bold yellow]📌 Agent Modules Status[/bold yellow]")
    console.print()
    console.print("  ✅ [green]Fundamental Analysis Engine[/green]   — Active")
    console.print("  🔄 [yellow]DCF Valuation Engine[/yellow]         — Coming Next")
    console.print("  🔄 [yellow]Market Sentiment Agent[/yellow]        — Planned")
    console.print("  🔄 [yellow]Management Quality Agent[/yellow]      — Planned")
    console.print("  🔄 [yellow]Geopolitical Risk Agent[/yellow]       — Planned")
    console.print("  🔄 [yellow]Streamlit Dashboard[/yellow]           — Planned")
    console.print()

    # ---- FOOTER ----
    console.rule()
    console.print(
        "[dim]⚠️  DISCLAIMER: AI-generated research tool. "
        "Not SEBI-registered investment advice. "
        "Always do your own due diligence before investing.[/dim]"
    )
    console.print()
