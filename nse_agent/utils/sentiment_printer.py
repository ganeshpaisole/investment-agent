# ============================================================
# utils/sentiment_printer.py — Sentiment Report Printer v1.0
# ============================================================

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def print_sentiment_report(result: dict):
    ticker  = result["ticker"]
    score   = result["total_score"]
    grade   = result["grade"]
    signal  = result["signal"]
    verdict = result["verdict"]
    dims    = result["dimensions"]

    console.print()
    console.rule(f"[bold yellow]📡 MARKET SENTIMENT REPORT — {ticker}[/bold yellow]")

    # Verdict panel
    sc_col  = "green" if score >= 65 else "yellow" if score >= 50 else "red"
    bar     = "█" * int(score/100*32) + "░" * (32 - int(score/100*32))
    rsi     = result.get("rsi", 50)
    rsi_tag = "  🔴 Overbought" if rsi > 70 else "  🟢 Oversold" if rsi < 30 else ""
    bb_pos  = result.get("bb_position", "middle")
    bb_tag  = ("🟢 Near lower band (buy zone)" if bb_pos == "lower"
               else "🔴 Near upper band (caution)" if bb_pos == "upper"
               else "⚪ Mid range")

    vtext = (
        f"    {signal}\n\n"
        f"    Sentiment Score : [{sc_col}]{score}/100  Grade: {grade}[/{sc_col}]\n"
        f"    [{sc_col}]{bar} {score}/100[/{sc_col}]\n\n"
        f"    {verdict}\n\n"
        f"    RSI (14)      : {rsi}{rsi_tag}\n"
        f"    MACD          : {'🟢 Bullish crossover' if result.get('macd_bullish') else '⚪ No bullish crossover'}\n"
        f"    Bollinger     : {bb_tag}\n"
        f"    SMAs above    : {result.get('sma_above',0)}/3\n"
        f"    3M Return     : {result.get('return_3m') or 0:+.1f}%\n"
        f"    52-Week Pos   : {result.get('pct_52w',50):.0f}% of annual range\n"
        f"    vs Nifty (3M) : {result.get('rel_strength') or 0:+.1f}%"
    )
    pcol = "green" if score >= 65 else "yellow" if score >= 50 else "red"
    console.print(Panel(vtext, title="Sentiment Verdict", border_style=pcol))

    # Score breakdown table
    console.print()
    t = Table(box=box.ROUNDED, header_style="bold white", show_lines=True)
    t.add_column("Dimension",  width=26, style="bold")
    t.add_column("Score",      width=10, justify="center")
    t.add_column("Max",        width=6,  justify="center")
    t.add_column("Rating",     width=14, justify="center")
    t.add_column("Top Signal", width=48)

    rows = [
        ("🚀 Price Momentum",    "momentum",      30),
        ("📊 Volume & Breadth",  "volume",        20),
        ("⚡ Technical Signals", "technicals",    20),
        ("🏦 Institutional",     "institutional", 20),
        ("💰 Valuation Context", "valuation_ctx", 10),
    ]
    for name, key, mx in rows:
        d   = dims.get(key, {})
        sc  = d.get("score", 0)
        pct = sc / mx
        rat = ("Exceptional" if pct>=0.85 else "Excellent" if pct>=0.70
               else "Good" if pct>=0.55 else "Average" if pct>=0.40 else "Poor")
        col = "green" if pct>=0.65 else "yellow" if pct>=0.45 else "red"
        all_s = d.get("signals",[]) + d.get("warnings",[])
        top   = all_s[0][:50] if all_s else "—"
        t.add_row(name, f"[{col}]{sc}/{mx}[/{col}]", str(mx), rat, top)
    console.print(t)

    # Signals detail
    for name, key, _ in rows:
        d    = dims.get(key, {})
        sigs = d.get("signals", [])
        warn = d.get("warnings", [])
        if not sigs and not warn: continue
        console.print(f"\n  [bold]{name}[/bold]")
        for s in sigs:  console.print(f"     {s}")
        for w in warn:  console.print(f"     {w}")

    # Key metrics table
    console.print()
    console.print("                         📌 Key Technical Metrics\n")
    mom_d  = dims.get("momentum",{}).get("details",{})
    tech_d = dims.get("technicals",{}).get("details",{})
    vol_d  = dims.get("volume",{}).get("details",{})
    inst_d = dims.get("institutional",{}).get("details",{})
    sma    = mom_d.get("sma",{})
    macd_d = tech_d.get("macd",{})
    bb_d   = tech_d.get("bollinger",{})

    t2 = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
    t2.add_column("M1", width=22, style="dim")
    t2.add_column("V1", width=14)
    t2.add_column("M2", width=22, style="dim")
    t2.add_column("V2", width=14)

    r3m = mom_d.get("return_3m"); r6m = mom_d.get("return_6m")
    t2.add_row("RSI (14)",        str(rsi),                          "52-Week Position",  f"{result.get('pct_52w',0):.0f}% of range")
    t2.add_row("MACD Histogram",  f"{macd_d.get('histogram',0):+.2f}","3M Return",        f"{r3m:+.1f}%" if r3m else "N/A")
    t2.add_row("Bollinger B%",    f"{bb_d.get('pct_b',0):.2f}",      "6M Return",         f"{r6m:+.1f}%" if r6m else "N/A")
    t2.add_row("SMA 20d",         f"₹{sma.get('sma20',0):,.0f}",     "vs Nifty 3M",       f"{result.get('rel_strength') or 0:+.1f}%")
    t2.add_row("SMA 50d",         f"₹{sma.get('sma50',0):,.0f}",     "Beta",              str(result.get('beta','N/A')))
    t2.add_row("SMA 200d",        f"₹{sma.get('sma200',0):,.0f}",    "Inst. Holding",     f"{result.get('inst_pct',0):.1f}%")
    t2.add_row("Acc Days (20d)",  str(vol_d.get("acc_days","N/A")),   "Dist Days (20d)",   str(vol_d.get("dist_days","N/A")))
    t2.add_row("Daily Turnover",  f"₹{vol_d.get('daily_value_cr',0):.0f} Cr",
                                                                       "Ann. Volatility",   f"{inst_d.get('annualized_vol_pct',0):.0f}%")
    console.print(t2)
    console.print()
    console.rule("[dim]⚠️  Technical analysis — past performance ≠ future results. Not SEBI advice.[/dim]")
