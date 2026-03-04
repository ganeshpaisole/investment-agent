"""
╔══════════════════════════════════════════════════════════╗
║     PSU ADJUSTMENT MODULE — NSE Agent                    ║
║     Government Ownership Scoring Corrections             ║
║     Policy Risk | Divestment | Capex Mandates            ║
╚══════════════════════════════════════════════════════════╝

Why PSUs need special treatment:
  - Government can override commercial decisions
  - Dividend payout mandated (often >30% or ₹X minimum)
  - Capex driven by policy, not just IRR
  - Valuation discounts vs private peers (PSU discount ~20-40%)
  - Divestment overhang suppresses price
  - But: cheap valuations, monopoly positions, state backing
"""

from loguru import logger

# ─────────────────────────────────────────────────────────
# PSU REGISTRY — Nifty50 PSUs with ownership & risk profile
# ─────────────────────────────────────────────────────────
PSU_REGISTRY = {
    # Banking
    "SBIN": {
        "full_name":       "State Bank of India",
        "sector":          "BANKING",
        "govt_stake_pct":  57.5,
        "divestment_risk": "LOW",      # Core PSU, unlikely to divest
        "policy_capex":    True,
        "dividend_mandate": True,
        "psu_discount_pct": 25,        # vs private bank peers
        "moat":            "State backing, largest bank network",
        "key_risks":       ["NPA cycles", "Political lending", "Lower NIMs vs private peers"],
        "key_positives":   ["Largest bank in India", "State guarantee implied", "Digital transformation"],
    },

    # Energy / Oil
    "ONGC": {
        "full_name":       "Oil and Natural Gas Corporation",
        "sector":          "ENERGY",
        "govt_stake_pct":  58.9,
        "divestment_risk": "LOW",
        "policy_capex":    True,
        "dividend_mandate": True,
        "psu_discount_pct": 35,
        "moat":            "Largest upstream oil/gas company in India",
        "key_risks":       ["Crude oil price volatility", "Subsidy burden risk", "Exploration capex"],
        "key_positives":   ["Deep upstream reserves", "High dividend yield", "Energy security play"],
    },
    "BPCL": {
        "full_name":       "Bharat Petroleum Corporation",
        "sector":          "ENERGY",
        "govt_stake_pct":  52.9,
        "divestment_risk": "MEDIUM",   # Was in divestment list, then removed
        "policy_capex":    True,
        "dividend_mandate": True,
        "psu_discount_pct": 30,
        "moat":            "Refining + marketing network",
        "key_risks":       ["Divestment uncertainty", "Refining margin volatility", "GRM compression"],
        "key_positives":   ["Large retail fuel network", "Petrochemical expansion"],
    },
    "COALINDIA": {
        "full_name":       "Coal India Limited",
        "sector":          "ENERGY",
        "govt_stake_pct":  63.1,
        "divestment_risk": "LOW",
        "policy_capex":    True,
        "dividend_mandate": True,
        "psu_discount_pct": 20,
        "moat":            "Near-monopoly in Indian coal production",
        "key_risks":       ["Energy transition risk", "Labor issues", "Captive coal competition"],
        "key_positives":   ["High dividend yield (6-8%)", "Monopoly position", "India coal demand structural"],
    },

    # Power / Utilities
    "NTPC": {
        "full_name":       "NTPC Limited",
        "sector":          "UTILITY",
        "govt_stake_pct":  51.1,
        "divestment_risk": "LOW",
        "policy_capex":    True,      # Mandated renewable capacity
        "dividend_mandate": True,
        "psu_discount_pct": 20,
        "moat":            "Largest power generator in India",
        "key_risks":       ["Tariff regulation", "Fuel supply security", "Renewable transition capex"],
        "key_positives":   ["Regulated returns (15.5% ROE guaranteed)", "Renewable pivot", "Dividend consistency"],
    },
    "POWERGRID": {
        "full_name":       "Power Grid Corporation of India",
        "sector":          "UTILITY",
        "govt_stake_pct":  51.3,
        "divestment_risk": "LOW",
        "policy_capex":    True,
        "dividend_mandate": True,
        "psu_discount_pct": 15,       # Lower discount — stable regulated returns
        "moat":            "Monopoly in interstate power transmission",
        "key_risks":       ["Tariff revision risk", "Stranded asset risk"],
        "key_positives":   ["Monopoly transmission", "Guaranteed 15.5% ROE", "Stable cash flows"],
    },

    # Infrastructure
    "ADANIENT": {
        "full_name":       "Adani Enterprises",
        "sector":          "INFRA",
        "govt_stake_pct":  0,          # Private, not PSU
        "divestment_risk": "N/A",
        "policy_capex":    False,
        "dividend_mandate": False,
        "psu_discount_pct": 0,
        "moat":            "Ports, airports, energy conglomerate",
        "key_risks":       ["Concentrated promoter holding", "High leverage", "Regulatory scrutiny"],
        "key_positives":   ["Government project wins", "Infrastructure monopoly positions"],
    },
}

# Which Nifty50 tickers are PSUs
PSU_TICKERS = {k for k, v in PSU_REGISTRY.items() if v.get("govt_stake_pct", 0) > 50}


def is_psu(ticker: str) -> bool:
    """Returns True if ticker is a government-majority-owned company."""
    return ticker.upper() in PSU_TICKERS


def get_psu_info(ticker: str) -> dict:
    """Returns full PSU profile for a ticker."""
    return PSU_REGISTRY.get(ticker.upper(), {})


class PSUAdjustmentModule:
    """
    Applies PSU-specific scoring corrections to the standard agent outputs.

    Key adjustments:
      1. Valuation score: Apply PSU discount to fair value
      2. Management score: Penalize for policy-driven decisions
      3. Growth score: Adjust for capex mandates
      4. Dividend score: Boost for mandatory high payouts
      5. Composite score: Apply PSU governance penalty
    """

    PSU_GOVERNANCE_PENALTY = 8      # pts deducted from composite for PSU governance
    PSU_STABILITY_BONUS    = 5      # pts added for state backing / no default risk
    PSU_DIVIDEND_BONUS     = 5      # pts for typically high dividends

    def __init__(self, ticker: str):
        self.ticker   = ticker.upper()
        self.is_psu   = is_psu(self.ticker)
        self.psu_info = get_psu_info(self.ticker)

    def adjust_scores(self, scores: dict, raw_data: dict = None) -> dict:
        """
        Takes existing score dict from sector agents and applies PSU corrections.

        Args:
            scores: {"fundamental": 70, "valuation": 65, "management": 72, ...}
            raw_data: Optional raw financial data for deeper adjustments

        Returns:
            Adjusted scores dict + PSU context
        """
        if not self.is_psu:
            return {**scores, "psu_adjusted": False, "psu_context": None}

        info = self.psu_info
        adjusted = scores.copy()
        adjustments = []

        # ── 1. Valuation Adjustment ─────────────────────────────────
        # PSUs trade at a discount — fair value should reflect this
        psu_discount = info.get("psu_discount_pct", 25)
        val_score    = adjusted.get("valuation", 50)

        # If stock trades at PSU discount, it may actually be undervalued
        # Moderate boost to valuation score since the discount IS the value
        val_adjustment = min(psu_discount * 0.3, 10)  # Up to +10 pts
        adjusted["valuation"] = min(100, val_score + val_adjustment)
        adjustments.append(f"✅ Valuation boosted +{val_adjustment:.0f}pts (PSU discount creates value opportunity)")

        # ── 2. Management / Governance Penalty ─────────────────────
        # PSU management faces political interference
        mgmt_score = adjusted.get("management", 50)
        adjusted["management"] = max(0, mgmt_score - self.PSU_GOVERNANCE_PENALTY)
        adjustments.append(f"⚠️ Management penalized -{self.PSU_GOVERNANCE_PENALTY}pts (government interference risk)")

        # ── 3. Divestment Risk Adjustment ──────────────────────────
        div_risk = info.get("divestment_risk", "LOW")
        if div_risk == "HIGH":
            # Uncertainty suppresses all scores
            for key in ["fundamental", "valuation"]:
                adjusted[key] = max(0, adjusted.get(key, 50) - 10)
            adjustments.append("🔴 High divestment risk — significant uncertainty")
        elif div_risk == "MEDIUM":
            adjustments.append("🟡 Medium divestment risk — monitor government announcements")
        else:
            adjusted["fundamental"] = min(100, adjusted.get("fundamental", 50) + 3)
            adjustments.append("✅ Low divestment risk — core strategic asset")

        # ── 4. Dividend Mandate Bonus ───────────────────────────────
        if info.get("dividend_mandate", False):
            adjusted["fundamental"] = min(100, adjusted.get("fundamental", 50) + self.PSU_DIVIDEND_BONUS)
            adjustments.append(f"✅ Dividend mandate +{self.PSU_DIVIDEND_BONUS}pts (consistent income)")

        # ── 5. State Backing Stability Bonus ───────────────────────
        adjusted["fundamental"] = min(100, adjusted.get("fundamental", 50) + self.PSU_STABILITY_BONUS)
        adjustments.append(f"✅ State backing stability +{self.PSU_STABILITY_BONUS}pts (zero default risk)")

        # ── 6. Policy Capex Drag ────────────────────────────────────
        if info.get("policy_capex", False):
            adjusted["fundamental"] = max(0, adjusted.get("fundamental", 50) - 5)
            adjustments.append("⚠️ Policy-driven capex may suppress FCF (−5pts)")

        # ── 7. Peer Comparison Caveat ───────────────────────────────
        # PSUs should only be compared vs other PSUs, not private peers
        adjustments.append(f"ℹ️  Compare only vs PSU peers | Govt stake: {info.get('govt_stake_pct')}%")

        return {
            **adjusted,
            "psu_adjusted":     True,
            "psu_context": {
                "company":          info.get("full_name", self.ticker),
                "govt_stake_pct":   info.get("govt_stake_pct", 0),
                "divestment_risk":  info.get("divestment_risk", "UNKNOWN"),
                "psu_discount_pct": info.get("psu_discount_pct", 0),
                "moat":             info.get("moat", ""),
                "key_risks":        info.get("key_risks", []),
                "key_positives":    info.get("key_positives", []),
                "adjustments_applied": adjustments,
            }
        }

    def get_fair_value_with_psu_adjustment(self, base_fair_value: float, current_price: float) -> dict:
        """
        Adjusts intrinsic value for PSU discount.
        PSUs typically deserve a 20-35% discount to DCF because:
          - Management not fully commercially incentivized
          - Government may extract value (dividends, cheap gas supply, etc.)
        """
        if not self.is_psu or base_fair_value <= 0:
            return {"adjusted_iv": base_fair_value, "psu_discount_applied": 0}

        discount  = self.psu_info.get("psu_discount_pct", 25) / 100
        adj_iv    = base_fair_value * (1 - discount)
        mos       = round((adj_iv - current_price) / current_price * 100, 1) if current_price else 0

        return {
            "base_iv":              round(base_fair_value, 2),
            "adjusted_iv":          round(adj_iv, 2),
            "psu_discount_applied": round(discount * 100, 0),
            "adjusted_mos_pct":     mos,
            "note": f"PSU discount of {discount*100:.0f}% applied. "
                    f"Compare CMP ₹{current_price} vs adj. IV ₹{round(adj_iv,0)}"
        }

    def get_psu_report(self) -> str:
        """Returns a formatted PSU context string for reports."""
        if not self.is_psu:
            return ""

        info = self.psu_info
        lines = [
            f"\n{'─'*50}",
            f"🏛️  PSU ANALYSIS: {info.get('full_name', self.ticker)}",
            f"{'─'*50}",
            f"Government Stake : {info.get('govt_stake_pct', 0)}%",
            f"Divestment Risk  : {info.get('divestment_risk', 'UNKNOWN')}",
            f"PSU Discount     : ~{info.get('psu_discount_pct', 0)}% vs private peers",
            f"Moat             : {info.get('moat', 'N/A')}",
            f"\n✅ Key Positives:",
        ]
        for pos in info.get("key_positives", []):
            lines.append(f"   • {pos}")
        lines.append(f"\n⚠️  Key Risks:")
        for risk in info.get("key_risks", []):
            lines.append(f"   • {risk}")
        lines.append(f"{'─'*50}")
        return "\n".join(lines)


# Convenience function for integration with master agent
def apply_psu_adjustments(ticker: str, scores: dict, base_fair_value: float = 0,
                           current_price: float = 0) -> dict:
    """One-call interface for master orchestrator."""
    module = PSUAdjustmentModule(ticker)
    adjusted_scores = module.adjust_scores(scores)
    iv_adjustment   = module.get_fair_value_with_psu_adjustment(base_fair_value, current_price)
    psu_report_str  = module.get_psu_report()

    return {
        "scores":        adjusted_scores,
        "iv_adjustment": iv_adjustment,
        "psu_report":    psu_report_str,
        "is_psu":        module.is_psu,
    }


if __name__ == "__main__":
    # Test
    result = apply_psu_adjustments(
        "SBIN",
        scores={"fundamental": 65, "valuation": 58, "management": 70},
        base_fair_value=900,
        current_price=750
    )
    import json
    print(json.dumps(result, indent=2, default=str))
