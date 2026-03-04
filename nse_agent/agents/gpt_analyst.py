# ============================================================
# agents/gpt_analyst.py
# GPT-4 Powered Narrative Analyst
#
# After our scoring engine calculates numbers, this module
# sends those numbers to GPT-4 which then writes a professional
# analyst narrative — just like a CFA analyst would write.
#
# GPT-4 also checks all 10 of Buffett's key questions.
# ============================================================

import os
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

# Load API keys from .env file
load_dotenv()


class GPTAnalyst:
    """
    Uses GPT-4 to generate professional analyst commentary
    on fundamental analysis results.

    Usage:
        analyst = GPTAnalyst()
        narrative = analyst.generate_narrative(fundamental_result)
    """

    def __init__(self):
        """Initialize the OpenAI client with API key from .env"""

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ OPENAI_API_KEY not found!\n"
                "Please add your key to the .env file:\n"
                "OPENAI_API_KEY=sk-your-key-here"
            )

        # Ensure API key is available to the OpenAI library via environment
        os.environ["OPENAI_API_KEY"] = api_key
        # Initialize the client without passing runtime kwargs (some OpenAI
        # client versions don't accept extra constructor args like 'proxies').
        self.client = OpenAI()
        self.model  = os.getenv("OPENAI_MODEL", "gpt-4o")
        logger.info(f"🤖 GPT Analyst initialized — using model: {self.model}")

    # ----------------------------------------------------------
    # BUFFETT'S 10 QUESTIONS EVALUATOR
    # ----------------------------------------------------------

    def evaluate_buffett_questions(self, result: dict) -> str:
        """
        Ask GPT-4 to evaluate the company against Buffett's
        famous 10 investment criteria.

        Args:
            result: Output from FundamentalAnalysisAgent.analyze()

        Returns:
            String with GPT-4's evaluation of all 10 questions
        """
        ticker   = result["ticker"]
        ratios   = result["ratios"]
        info     = result.get("raw_info", {})
        score    = result["total_score"]
        val_sum  = result["valuation_summary"]

        # Build a rich prompt with all the financial data
        prompt = f"""
You are a senior investment analyst trained in Warren Buffett's investment philosophy.

Evaluate {ticker} (NSE-listed Indian company) against Buffett's 10 investment criteria.

## FINANCIAL DATA:
- Sector: {info.get('sector', 'N/A')}
- Industry: {info.get('industry', 'N/A')}
- Market Cap: ₹{info.get('marketCap', 0)/1e7:.0f} Crores
- Current Price: ₹{val_sum['current_price']:,.2f}
- Graham Number (Intrinsic Value): ₹{val_sum['graham_number']:,.2f}
- Margin of Safety: {val_sum['margin_of_safety']:.1f}%

## PROFITABILITY:
- ROE: {ratios['profitability']['roe']:.1f}%
- ROCE: {ratios['profitability']['roce']:.1f}%
- Net Profit Margin: {ratios['profitability']['net_margin']:.1f}%
- EBITDA Margin: {ratios['profitability']['ebitda_margin']:.1f}%

## FINANCIAL HEALTH:
- Debt/Equity: {ratios['leverage']['debt_to_equity']:.2f}
- Current Ratio: {ratios['leverage']['current_ratio']:.2f}
- Interest Coverage: {ratios['leverage']['interest_coverage']:.1f}x
- Free Cash Flow: ₹{ratios['cashflow']['fcf_cr']:.0f} Crores

## GROWTH:
- Revenue Growth (YoY): {ratios['growth']['revenue_growth_yoy']:.1f}%
- EPS Growth (YoY): {ratios['growth']['earnings_growth_yoy']:.1f}%

## VALUATION:
- P/E Ratio: {ratios['valuation']['pe_ratio']:.1f}
- P/B Ratio: {ratios['valuation']['pb_ratio']:.2f}
- Earnings Yield: {ratios['valuation']['earnings_yield']:.2f}%

## FUNDAMENTAL SCORE: {score}/100

---

Now evaluate this company against Buffett's 10 questions:

1. Is the business simple and understandable?
2. Does it have a consistent 10+ year operating history?
3. Does it have favorable long-term prospects / competitive moat?
4. Is management rational in capital allocation?
5. Is management honest with shareholders?
6. Does management resist institutional imperative (following competitors blindly)?
7. What is the return on equity WITHOUT heavy leverage?
8. What are the "owner earnings" (FCF)?
9. Does the company require heavy reinvestment / is it capital-light?
10. Can we buy it at a significant discount to intrinsic value?

For each question, give a YES/PARTIAL/NO rating with a 1-2 sentence explanation using the financial data provided.
End with an OVERALL BUFFETT VERDICT: PASS / PARTIAL PASS / FAIL with 2-3 sentence summary.

Format your response clearly with numbered questions.
"""

        logger.info("🤖 Asking GPT-4 to evaluate Buffett's 10 questions...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a world-class investment analyst specialized in "
                        "Indian equity markets and value investing. You follow Graham "
                        "and Buffett principles rigorously. Be specific, data-driven, "
                        "and concise. Always reference the actual numbers provided."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,   # Lower = more consistent, factual responses
            max_tokens=1500
        )

        return response.choices[0].message.content

    # ----------------------------------------------------------
    # NARRATIVE REPORT GENERATOR
    # ----------------------------------------------------------

    def generate_narrative(self, result: dict) -> str:
        """
        Generate a full professional analyst narrative for the stock.

        Args:
            result: Output from FundamentalAnalysisAgent.analyze()

        Returns:
            Full narrative report as a string
        """
        ticker    = result["ticker"]
        score     = result["total_score"]
        grade     = result["grade"]
        rec       = result["recommendation"]
        ratios    = result["ratios"]
        info      = result.get("raw_info", {})
        val_sum   = result["valuation_summary"]
        cats      = result["categories"]

        # Build category summaries for the prompt
        cat_summary = ""
        for cat_name, cat_data in cats.items():
            cat_summary += f"\n{cat_data['category']}: {cat_data['score']}/{cat_data['max_score']}\n"

        prompt = f"""
You are a Senior Equity Research Analyst at a top Indian investment firm.
Write a professional fundamental analysis report for {ticker} (NSE Listed).

## KEY DATA:
- Company: {info.get('longName', ticker)}
- Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}
- Market Cap: ₹{info.get('marketCap', 0)/1e7:.0f} Crores
- Business Description: {info.get('longBusinessSummary', 'N/A')[:300]}...

## SCORING SUMMARY:
- FUNDAMENTAL SCORE: {score}/100 | GRADE: {grade} | {rec}
{cat_summary}

## VALUATION:
- Current Price: ₹{val_sum['current_price']:,.2f}
- Graham Number: ₹{val_sum['graham_number']:,.2f}
- Margin of Safety: {val_sum['margin_of_safety']:.1f}%
- Signal: {val_sum['valuation_signal']}
- P/E: {ratios['valuation']['pe_ratio']} | P/B: {ratios['valuation']['pb_ratio']}

## KEY FINANCIALS:
- ROE: {ratios['profitability']['roe']:.1f}% | ROCE: {ratios['profitability']['roce']:.1f}%
- Net Margin: {ratios['profitability']['net_margin']:.1f}%
- D/E Ratio: {ratios['leverage']['debt_to_equity']:.2f}
- Interest Coverage: {ratios['leverage']['interest_coverage']:.1f}x
- FCF: ₹{ratios['cashflow']['fcf_cr']:.0f} Crores
- Revenue Growth: {ratios['growth']['revenue_growth_yoy']:.1f}%

---

Write a concise analyst report with EXACTLY these 5 sections:

**1. INVESTMENT THESIS** (3-4 sentences: what this company does and why it matters)

**2. KEY STRENGTHS** (3 bullet points with data references)

**3. KEY RISKS & RED FLAGS** (3 bullet points with data references)

**4. VALUATION ASSESSMENT** (2-3 sentences: is it cheap or expensive vs intrinsic value)

**5. ANALYST RECOMMENDATION** (2-3 sentences: buy/hold/avoid with entry price suggestion)

Be specific. Use actual numbers. Write like a CFA-level analyst for Indian markets.
Mention Graham and Buffett principles where relevant.
"""

        logger.info("🤖 Generating GPT-4 analyst narrative...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior CFA-level equity analyst covering Indian markets. "
                        "Your reports are used by institutional investors. Be data-driven, "
                        "specific, and follow Graham-Buffett value investing principles. "
                        "Never be vague — always reference the numbers provided."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=1200
        )

        return response.choices[0].message.content
