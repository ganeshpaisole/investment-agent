# 🏦 NSE Stock Forecast AI Agent
### Fundamental Analysis Engine — Graham + Buffett Framework

---

## 📁 Project Structure
```
nse_agent/
├── main.py                    ← RUN THIS to start analysis
├── requirements.txt           ← Install dependencies
├── .env.example               ← Copy to .env and add your API key
│
├── config/
│   └── settings.py            ← All thresholds and weights
│
├── agents/
│   ├── fundamental_agent.py   ← Core scoring engine (Graham/Buffett)
│   └── gpt_analyst.py         ← GPT-4 narrative analyst
│
└── utils/
    ├── data_fetcher.py        ← Gets data from Yahoo Finance + Screener.in
    ├── ratio_calculator.py    ← Computes all financial ratios
    └── report_printer.py      ← Beautiful terminal output
```

---

## ⚡ Quick Start (Step by Step)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set Up Your API Key
```bash
# Copy the example env file
cp .env.example .env

# Open .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

If you used the repository template, there's a `.env.template` at the project root — copy it to `.env` and paste your `OPENAI_API_KEY` value.

Example:
```bash
copy ..\.env.template .env
# then open .env and add your key
```

### Step 3: Run the Agent
```bash
# Interactive mode (it will ask you for ticker)
python main.py

# Direct mode (pass ticker as argument)
python main.py --ticker TCS
python main.py --ticker RELIANCE
python main.py --ticker INFY

# Without GPT-4 (for testing, no API key needed)
python main.py --ticker TCS --no-gpt
```

---

## 📊 What the Agent Analyzes

### Fundamental Score (100 points)
| Category | Points | What it checks |
|---|---|---|
| Earnings Quality | 25 | EPS growth, cash flow quality, profit consistency |
| Balance Sheet | 25 | Current ratio, debt/equity, interest coverage |
| Profitability | 25 | ROE, ROCE, net margins |
| Dividend & Capital | 25 | FCF, dividend history, payout ratio |

### Graham Thresholds Applied
- EPS Growth ≥ 10% per year
- Current Ratio ≥ 2.0
- Debt/Equity ≤ 0.5
- Interest Coverage ≥ 5x
- ROE ≥ 15% (Buffett's benchmark)
- Margin of Safety ≥ 30%

### GPT-4 Analysis
- Buffett's 10 Investment Questions evaluated
- Professional analyst narrative
- Key strengths and red flags
- Entry price suggestion

---

## 🧪 Test Tickers to Try
```bash
python main.py --ticker TCS          # IT giant
python main.py --ticker ITC          # FMCG conglomerate
python main.py --ticker HDFC         # Banking/Finance
python main.py --ticker BAJFINANCE   # NBFC
python main.py --ticker INFY         # IT services
```

---

## ⚠️ Disclaimer
This tool is for educational and research purposes only.
Not SEBI-registered investment advice.
Always do your own due diligence before investing.
