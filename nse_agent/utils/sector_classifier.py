# ============================================================
# utils/sector_classifier.py — VERSION 1.0
# Detects stock sector and determines which agents to run.
# Routes to specialized agents for Banking, Insurance, REIT.
# Warns users when generic DCF is inappropriate.
# ============================================================

from loguru import logger
from rich.console import Console
from rich.panel import Panel

console = Console()

# ── Sector routing map ───────────────────────────────────────
# Maps Yahoo Finance sector strings → our internal sector type

SECTOR_MAP = {
    # Generic agents work well
    "technology":             "IT",
    "information technology": "IT",
    "healthcare":             "PHARMA",
    "consumer defensive":     "FMCG",
    "consumer cyclical":      "CYCLICAL",
    "industrials":            "CYCLICAL",
    "basic materials":        "CYCLICAL",
    "energy":                 "CYCLICAL",
    "communication services": "IT",

    # Specialized agents needed
    "financial services":     "BANKING",
    "real estate":            "REIT",
    "utilities":              "UTILITY",
}

# ── Agent compatibility matrix ───────────────────────────────
AGENT_SUPPORT = {
    "IT": {
        "fundamental": True,
        "valuation":   True,
        "management":  True,
        "banking":     False,
        "insurance":   False,
        "notes":       None,
    },
    "PHARMA": {
        "fundamental": True,
        "valuation":   True,
        "management":  True,
        "banking":     False,
        "insurance":   False,
        "notes":       "⚠️  Pharma: Patent cliff and R&D pipeline not captured in DCF.",
    },
    "FMCG": {
        "fundamental": True,
        "valuation":   True,
        "management":  True,
        "banking":     False,
        "insurance":   False,
        "notes":       None,
    },
    "CYCLICAL": {
        "fundamental": True,
        "valuation":   True,   # with cyclical adjustments
        "management":  True,
        "banking":     False,
        "insurance":   False,
        "notes":       "ℹ️  Cyclical sector: WACC floor 10.5% and EBITDA×0.35 FCF proxy applied.",
    },
    "BANKING": {
        "fundamental": False,  # Standard D/E, current ratio meaningless for banks
        "valuation":   False,  # DCF/FCF not applicable — use Banking Agent
        "management":  True,   # Partially applicable
        "banking":     True,   # Use Banking Agent instead
        "insurance":   False,
        "notes":       None,
    },
    "INSURANCE": {
        "fundamental": False,
        "valuation":   False,
        "management":  True,
        "banking":     False,
        "insurance":   True,   # Use Insurance Agent
        "notes":       None,
    },
    "REIT": {
        "fundamental": False,
        "valuation":   False,  # Need NAV-based valuation
        "management":  True,
        "banking":     False,
        "insurance":   False,
        "notes":       "⚠️  Real Estate: NAV-based valuation required. DCF not reliable.",
    },
    "UTILITY": {
        "fundamental": True,
        "valuation":   True,
        "management":  True,
        "banking":     False,
        "insurance":   False,
        "notes":       "ℹ️  Utility: Regulated returns — WACC floor applied. DCF reliable.",
    },
    "NBFC": {
        "fundamental": False,  # D/E, current ratio misleading
        "valuation":   False,  # DCF unreliable — negative FCF by design
        "management":  True,
        "banking":     True,   # Use banking agent
        "insurance":   False,
        "notes":       "ℹ️  NBFC: Banking Agent used. ROA, P/B, AUM growth are key metrics.",
    },
    "UNKNOWN": {
        "fundamental": True,
        "valuation":   True,
        "management":  True,
        "banking":     False,
        "insurance":   False,
        "notes":       "⚠️  Unknown sector — results may be less accurate.",
    },
}

# ── Insurance company name keywords ─────────────────────────
INSURANCE_KEYWORDS = [
    "insurance", "life insurance", "general insurance",
    "reinsurance", "lic", "bajaj allianz", "hdfc life",
    "sbi life", "icici prudential", "star health",
    "new india", "united india", "oriental insurance",
]

# ── NBFC / Microfinance keywords ─────────────────────────────
NBFC_KEYWORDS = [
    "finance", "financial", "capital", "leasing", "lending",
    "microfinance", "housing finance", "bajaj finance",
    "muthoot", "manappuram", "chola", "shriram",
    "poonawalla", "mahindra finance",
]


class SectorClassifier:
    """
    Classifies a stock into a sector type and determines
    which agents are appropriate to run.
    """

    def __init__(self, ticker: str, info: dict):
        self.ticker      = ticker
        self.info        = info
        self.sector_raw  = (info.get("sector", "") or "").lower()
        self.industry    = (info.get("industry", "") or "").lower()
        self.name        = (info.get("longName", "") or "").lower()
        self.sector_type = self._classify()

    def _classify(self) -> str:
        """Determine internal sector type."""

        # Check for insurance first (often classified under financial services)
        if any(k in self.name for k in INSURANCE_KEYWORDS):
            return "INSURANCE"
        if any(k in self.industry for k in INSURANCE_KEYWORDS):
            return "INSURANCE"

        # Standard sector mapping
        sector_type = SECTOR_MAP.get(self.sector_raw, None)
        if sector_type:
            # Sub-classify financial services → banking vs NBFC vs insurance
            if sector_type == "BANKING":
                if any(k in self.industry for k in INSURANCE_KEYWORDS):
                    return "INSURANCE"
                # NBFCs are financial services but not banks
                # They can partially use standard agents
                if any(k in self.name for k in NBFC_KEYWORDS):
                    return "NBFC"
            return sector_type

        return "UNKNOWN"

    def get_support(self) -> dict:
        """Return agent support matrix for this sector."""
        return AGENT_SUPPORT.get(self.sector_type, AGENT_SUPPORT["UNKNOWN"])

    def print_sector_banner(self):
        """Print sector classification and agent compatibility."""
        support = self.get_support()
        notes   = support.get("notes")

        # Build compatibility string
        compat = []
        if support["fundamental"]: compat.append("✅ Fundamental")
        else:                       compat.append("⛔ Fundamental")
        if support["valuation"]:   compat.append("✅ DCF Valuation")
        else:                       compat.append("⛔ DCF Valuation")
        if support["management"]:  compat.append("✅ Management")
        else:                       compat.append("⛔ Management")
        if support["banking"]:     compat.append("🏦 Banking Agent")
        if support["insurance"]:   compat.append("🛡️  Insurance Agent")

        color = (
            "green"  if self.sector_type in ("IT", "FMCG", "PHARMA") else
            "yellow" if self.sector_type in ("CYCLICAL", "UTILITY", "NBFC", "UNKNOWN") else
            "cyan"   if self.sector_type in ("BANKING", "INSURANCE", "REIT") else
            "white"
        )

        content = f"  Sector Type : [bold]{self.sector_type}[/bold]  ({self.sector_raw.title()})\n"
        content += f"  Industry    : {self.info.get('industry', 'N/A')}\n\n"
        content += "  Agent Compatibility:\n"
        for c in compat:
            content += f"    {c}\n"
        if notes:
            content += f"\n  {notes}"

        console.print(Panel(
            content,
            title=f"[bold {color}]🏷️  SECTOR CLASSIFICATION — {self.ticker}[/bold {color}]",
            border_style=color,
            padding=(0, 2),
        ))
        console.print()

    def should_run(self, agent: str) -> bool:
        """Check if a specific agent should run for this sector."""
        support = self.get_support()
        return support.get(agent, False)

    def get_warnings(self) -> list:
        """Return list of sector-specific warnings."""
        warnings = []
        support  = self.get_support()

        if not support["fundamental"]:
            warnings.append(
                f"⛔ Standard Fundamental Analysis not suitable for {self.sector_type} sector. "
                f"Key ratios (D/E, Current Ratio) are misleading for this business model."
            )
        if not support["valuation"]:
            warnings.append(
                f"⛔ DCF Valuation not suitable for {self.sector_type} sector. "
                f"{'Use Banking Agent for NIM/NPA analysis.' if self.sector_type == 'BANKING' else ''}"
                f"{'Use Insurance Agent for embedded value analysis.' if self.sector_type == 'INSURANCE' else ''}"
                f"{'Use NAV-based valuation for Real Estate.' if self.sector_type == 'REIT' else ''}"
            )
        if support.get("notes"):
            warnings.append(support["notes"])

        return warnings
