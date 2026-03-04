# ============================================================
# utils/valuation_printer.py
# Beautiful terminal report for the DCF Valuation Engine
# ============================================================

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.rule import Rule

console = Console()


def print_valuation_report(result: dict):
    """
    Print a beautiful DCF valuation report in the terminal.

    Args:
        result: Output from ValuationAgent.analyze()
    """
    ticker    = result["ticker"]
    company   = result["company_name"]
    sector    = result["sector"]
    price     = result["current_price"]
    models    = result["models"]
    composite = result["composite"]
    rec       = result["recommendation"]
    score     = result["valuation_score"]

    dcf = models["dcf"]
    epv = models["epv"]
    ddm = models["ddm"]
    rel = models["relative"]

    iv    = composite["composite_iv"]
    upside = rec["upside_pct"]

    # Color scheme based on upside
    if upside >= 20:
        signal_color = "bright_green"
    elif upside >= 0:
        signal_color = "yellow"
    elif upside >= -15:
        signal_color = "orange1"
    else:
        signal_color = "red"

    # ---- HEADER ----
    console.print()
    console.rule(f"[bold cyan]💰 DCF VALUATION REPORT — {ticker}[/bold cyan]")
    console.print(f"  [bold white]{company}[/bold white]  |  {sector}")
    console.print()

    # ---- MAIN VERDICT PANEL ----
    verdict = Text()
    verdict.append(f"  {rec['signal']}\n\n", style=f"bold {signal_color}")
    verdict.append(f"  Current Market Price  : ", style="white")
    verdict.append(f"₹{price:,.2f}\n", style="bold white")
    verdict.append(f"  Intrinsic Value (IV)  : ", style="white")
    verdict.append(f"₹{iv:,.2f}\n", style=f"bold {signal_color}")
    verdict.append(f"  Upside / Downside     : ", style="white")
    verdict.append(f"{upside:+.1f}%\n", style=f"bold {signal_color}")
    verdict.append(f"  Margin of Safety      : ", style="white")
    verdict.append(f"{rec['mos_pct']:.1f}%\n\n", style="bold cyan")
    verdict.append(f"  📌 Entry Zone (30% MoS) : ", style="dim white")
    verdict.append(f"₹{rec['entry_price']:,.2f}  ", style="bold green")
    verdict.append(f"  🎯 Target Price : ", style="dim white")
    verdict.append(f"₹{rec['target_price']:,.2f}", style="bold cyan")

    console.print(Panel(verdict, title="[bold]Valuation Verdict[/bold]", border_style=signal_color))
    console.print()

    # ---- MODEL COMPARISON TABLE ----
    model_table = Table(
        title="📊 Intrinsic Value — Multi-Model Comparison",
        box=box.ROUNDED, border_style="cyan", header_style="bold cyan"
    )
    model_table.add_column("Valuation Model",    style="white",   min_width=25)
    model_table.add_column("Intrinsic Value",    justify="right", min_width=15, style="cyan")
    model_table.add_column("Weight",             justify="center", min_width=8)
    model_table.add_column("vs CMP",             justify="right", min_width=12)
    model_table.add_column("Key Assumption",     style="dim",     min_width=30)

    weights = composite["weights"]

    def vs_cmp(iv_val):
        if iv_val == 0 or iv_val == "N/A":
            return "N/A"
        pct = ((iv_val - price) / price) * 100
        color = "green" if pct >= 0 else "red"
        return f"[{color}]{pct:+.1f}%[/{color}]"

    # DCF row
    model_table.add_row(
        "1. DCF (Discounted Cash Flow)",
        f"₹{dcf['intrinsic_value']:,.2f}",
        f"{weights['dcf']*100:.0f}%",
        vs_cmp(dcf["intrinsic_value"]),
        f"FCF ₹{dcf['base_fcf_cr']:,.0f}Cr | g1={dcf['stage1_growth']:.1f}% | WACC={dcf['wacc']:.1f}%"
    )

    # EPV row
    model_table.add_row(
        "2. EPV (Earnings Power Value)",
        f"₹{epv['intrinsic_value']:,.2f}",
        f"{weights['epv']*100:.0f}%",
        vs_cmp(epv["intrinsic_value"]),
        f"NOPAT ₹{epv['nopat_cr']:,.0f}Cr | WACC={epv['wacc']:.1f}% | No-growth floor"
    )

    # Relative row
    model_table.add_row(
        "3. Relative (Sector Multiples)",
        f"₹{rel['intrinsic_value']:,.2f}",
        f"{weights['relative']*100:.0f}%",
        vs_cmp(rel["intrinsic_value"]),
        f"Fair P/E={rel['fair_pe_used']}x | Fair EV/EBITDA={rel['fair_ev_used']}x"
    )

    # DDM row
    if ddm.get("applicable"):
        model_table.add_row(
            "4. DDM (Dividend Discount)",
            f"₹{ddm['intrinsic_value']:,.2f}",
            f"{weights['ddm']*100:.0f}%",
            vs_cmp(ddm["intrinsic_value"]),
            f"D1=₹{ddm['dividend_d1']} | g={ddm['growth_rate_g']:.1f}% | Ke={ddm['cost_of_equity']:.1f}%"
        )
    else:
        model_table.add_row(
            "4. DDM (Dividend Discount)",
            "N/A", "0%", "N/A",
            "Not applicable (no dividend / growth ≥ Ke)"
        )

    model_table.add_section()
    model_table.add_row(
        "[bold]COMPOSITE IV (Weighted Avg)[/bold]",
        f"[bold {signal_color}]₹{iv:,.2f}[/bold {signal_color}]",
        "[bold]100%[/bold]",
        f"[bold {signal_color}]{upside:+.1f}%[/bold {signal_color}]",
        "[bold]Final intrinsic value estimate[/bold]"
    )

    console.print(model_table)
    console.print()

    # ---- DCF DETAIL BREAKDOWN ----
    console.print("[bold cyan]🔍 DCF Model — Detailed Assumptions[/bold cyan]")
    console.print()

    dcf_detail = Table(box=box.SIMPLE, border_style="blue", show_header=False)
    dcf_detail.add_column("Parameter", style="white",  min_width=30)
    dcf_detail.add_column("Value",     style="cyan",   min_width=20)
    dcf_detail.add_column("Note",      style="dim",    min_width=35)

    dcf_detail.add_row("Base Free Cash Flow",   f"₹{dcf['base_fcf_cr']:,.0f} Cr",       "Latest year FCF (CFO - Capex)")
    dcf_detail.add_row("Stage 1 Growth (Yr 1-5)", f"{dcf['stage1_growth']:.1f}% p.a.",  "Based on revenue CAGR + quality adj.")
    dcf_detail.add_row("Stage 2 Growth (Yr 6-10)", f"{dcf['stage2_growth']:.1f}% p.a.", "Normalizing to industry growth")
    dcf_detail.add_row("Terminal Growth Rate",  f"{dcf['terminal_growth']}% p.a.",       "India long-run GDP growth")
    dcf_detail.add_row("WACC (Discount Rate)",  f"{dcf['wacc']:.2f}%",                  "Risk-free 6.8% + Beta × ERP 5.5%")
    dcf_detail.add_row("PV of FCFs (10 yrs)",   f"₹{dcf['pv_fcf_cr']:,.0f} Cr",         "Discounted future cash flows")
    dcf_detail.add_row("PV of Terminal Value",  f"₹{dcf['pv_terminal_cr']:,.0f} Cr",    f"TV = {dcf['tv_pct_of_total']:.1f}% of total value")

    console.print(dcf_detail)
    console.print()

    # ---- SENSITIVITY TABLE ----
    console.print("[bold cyan]📉 DCF Sensitivity Analysis — Intrinsic Value at Different WACC & Growth Rates[/bold cyan]")
    console.print()

    # Build a mini sensitivity grid
    base_wacc   = dcf["wacc"]
    base_growth = dcf["stage1_growth"]
    base_iv     = dcf["intrinsic_value"]

    sens_table = Table(box=box.ROUNDED, border_style="dim", header_style="bold")
    sens_table.add_column("WACC \\ Growth", justify="center", style="bold", min_width=14)

    growth_scenarios = [base_growth - 3, base_growth, base_growth + 3]
    wacc_scenarios   = [base_wacc - 1, base_wacc, base_wacc + 1]

    for g in growth_scenarios:
        sens_table.add_column(f"g = {g:.1f}%", justify="center", min_width=14)

    for w in wacc_scenarios:
        row_vals = [f"WACC={w:.1f}%"]
        for g in growth_scenarios:
            # Simplified IV approximation for sensitivity
            scale   = (base_growth / max(g, 1)) * (base_wacc / max(w, 1))
            approx_iv = base_iv * (0.7 + 0.3 * scale)
            color = "green" if approx_iv > price else "red"
            row_vals.append(f"[{color}]₹{approx_iv:,.0f}[/{color}]")
        sens_table.add_row(*row_vals)

    console.print(sens_table)
    console.print(f"  [dim]CMP = ₹{price:,.2f} | Green = above CMP | Red = below CMP[/dim]")
    console.print()

    # ---- VALUATION SCORE ----
    score_color = "bright_green" if score >= 70 else ("yellow" if score >= 50 else "red")
    console.print(f"  [bold]Valuation Attractiveness Score:[/bold] [{score_color}]{score}/100[/{score_color}]")
    console.print()

    console.rule()
    console.print(
        "[dim]⚠️  DCF is based on estimates. Actual results may vary. "
        "Not SEBI-registered investment advice.[/dim]"
    )
    console.print()
