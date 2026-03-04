import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment and force cached/Screener flow
load_dotenv()
os.environ.setdefault("USE_YFINANCE", "false")

# Ensure imports match how main.py loads modules (add nse_agent to sys.path)
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
NS_DIR = PROJECT_ROOT / "nse_agent"
if str(NS_DIR) not in sys.path:
    sys.path.insert(0, str(NS_DIR))

from agents.fundamental_agent import FundamentalAnalysisAgent
from agents.gpt_analyst import GPTAnalyst
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment and force cached/Screener flow
load_dotenv()
os.environ.setdefault("USE_YFINANCE", "false")

from nse_agent.agents.fundamental_agent import FundamentalAnalysisAgent
from nse_agent.agents.gpt_analyst import GPTAnalyst


def main(ticker: str = "TCS"):
    ticker = ticker.upper().strip()

    # Run fundamental analysis (will use cache / Screener when USE_YFINANCE=false)
    fa = FundamentalAnalysisAgent(ticker)
    result = fa.analyze()

    # Run GPT analyst
    analyst = GPTAnalyst()

    narrative = ""
    buffett_eval = ""
    try:
        narrative = analyst.generate_narrative(result)
    except Exception as e:
        narrative = f"[ERROR] generate_narrative failed: {e}\n"

    try:
        buffett_eval = analyst.evaluate_buffett_questions(result)
    except Exception as e:
        buffett_eval = f"[ERROR] evaluate_buffett_questions failed: {e}\n"

    # Save outputs
    out_dir = Path("nse_agent") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}_gpt_report.txt"

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("=== GPT ANALYST NARRATIVE ===\n\n")
        fh.write(narrative + "\n\n")
        fh.write("=== BUFFETT 10 QUESTIONS EVALUATION ===\n\n")
        fh.write(buffett_eval + "\n")

    print(f"Saved GPT report to: {out_path}")


if __name__ == "__main__":
    main()
