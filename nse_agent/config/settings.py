# ============================================================
# config/settings.py
# Central configuration for all thresholds and scoring weights
# Think of this as the "brain settings" — change values here
# to tune the agent's behavior without touching core code
# ============================================================

# ---- GRAHAM'S FUNDAMENTAL THRESHOLDS ----
# These are the minimum acceptable values for a quality company
GRAHAM_THRESHOLDS = {
    # Earnings Quality
    "min_eps_cagr_10yr": 10.0,        # EPS must grow at least 10% per year over 10 years
    "min_cfo_to_net_income": 0.9,     # Cash from operations / Net Income (quality check)
    "max_loss_years_in_10": 0,        # Zero loss years allowed in 10 years (strict)

    # Balance Sheet
    "min_current_ratio": 2.0,         # Current Assets / Current Liabilities > 2 (safety)
    "max_debt_to_equity": 0.5,        # Low debt is key — Graham preferred <0.5
    "min_interest_coverage": 5.0,     # EBIT / Interest Expense > 5x (can pay interest easily)

    # Profitability (Buffett additions)
    "min_roe_5yr_avg": 15.0,          # Return on Equity > 15% (Buffett's benchmark)
    "min_roce_5yr_avg": 15.0,         # Return on Capital Employed > 15%
    "min_net_margin": 8.0,            # Net Profit Margin > 8%

    # Dividend
    "min_dividend_years": 5,          # Must have paid dividend for 5+ years
    "min_payout_ratio": 20.0,         # Minimum 20% dividend payout
    "max_payout_ratio": 60.0,         # Maximum 60% (not paying out too much)

    # Valuation Safety
    "min_margin_of_safety": 30.0,     # Intrinsic Value must be 30% above market price
}

# ---- SCORING WEIGHTS ----
# How much each category contributes to the final Fundamental Score (total = 100)
FUNDAMENTAL_SCORING_WEIGHTS = {
    "earnings_quality":       25,   # EPS growth, consistency, cash quality
    "balance_sheet_strength": 25,   # Debt, liquidity, solvency
    "profitability_returns":  25,   # ROE, ROCE, margins
    "dividend_capital":       25,   # Dividend history, payout ratio
}

# ---- OVERALL COMPOSITE WEIGHTS ----
# How each sub-agent contributes to the master composite score
COMPOSITE_WEIGHTS = {
    "fundamental":   0.25,
    "valuation":     0.20,
    "moat":          0.20,
    "management":    0.15,
    "sentiment":     0.08,
    "geo_risk":      0.07,
    "industry":      0.05,
}

# ---- GRADE THRESHOLDS ----
GRADE_MAP = {
    (80, 100): ("A", "Strong Buy Candidate"),
    (65,  79): ("B", "Good — Monitor Closely"),
    (50,  64): ("C", "Average — Proceed with Caution"),
    (35,  49): ("D", "Weak — Avoid"),
    (0,   34): ("F", "Poor — Definite Avoid"),
}

# ---- NSE STOCK SYMBOL FORMAT ----
# Yahoo Finance requires ".NS" suffix for NSE stocks
# Example: "RELIANCE" becomes "RELIANCE.NS"
NSE_SUFFIX = ".NS"

# ---- DATA PERIODS ----
HISTORICAL_YEARS = 10     # Years of data to analyze
SHORT_TERM_YEARS = 5      # For shorter-term ratios
PRICE_HISTORY = "10y"     # yfinance period string
