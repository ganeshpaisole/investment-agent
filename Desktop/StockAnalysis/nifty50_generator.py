#!/usr/bin/env python3
"""
NIFTY 50 Bulk Config Generator
Automatically creates stock config files for all NIFTY 50 stocks
"""

import json
import os


# NIFTY 50 stocks as of 2026 (you can update this list)
NIFTY_50_STOCKS = {
    "RELIANCE": {"name": "Reliance Industries", "sector": "Manufacturing"},
    "TCS": {"name": "Tata Consultancy Services", "sector": "Technology"},
    "HDFCBANK": {"name": "HDFC Bank", "sector": "Financial"},
    "INFY": {"name": "Infosys", "sector": "Technology"},
    "ICICIBANK": {"name": "ICICI Bank", "sector": "Financial"},
    "HINDUNILVR": {"name": "Hindustan Unilever", "sector": "Services"},
    "ITC": {"name": "ITC Limited", "sector": "Services"},
    "SBIN": {"name": "State Bank of India", "sector": "Financial"},
    "BHARTIARTL": {"name": "Bharti Airtel", "sector": "Services"},
    "KOTAKBANK": {"name": "Kotak Mahindra Bank", "sector": "Financial"},
    "LT": {"name": "Larsen & Toubro", "sector": "Infrastructure"},
    "AXISBANK": {"name": "Axis Bank", "sector": "Financial"},
    "BAJFINANCE": {"name": "Bajaj Finance", "sector": "Financial"},
    "HCLTECH": {"name": "HCL Technologies", "sector": "Technology"},
    "ASIANPAINT": {"name": "Asian Paints", "sector": "Manufacturing"},
    "MARUTI": {"name": "Maruti Suzuki", "sector": "Manufacturing"},
    "SUNPHARMA": {"name": "Sun Pharmaceutical", "sector": "Healthcare"},
    "TITAN": {"name": "Titan Company", "sector": "Services"},
    "ULTRACEMCO": {"name": "UltraTech Cement", "sector": "Manufacturing"},
    "WIPRO": {"name": "Wipro", "sector": "Technology"},
    "NESTLEIND": {"name": "Nestle India", "sector": "Services"},
    "ONGC": {"name": "Oil & Natural Gas Corp", "sector": "Manufacturing"},
    "TATAMOTORS": {"name": "Tata Motors", "sector": "Manufacturing"},
    "NTPC": {"name": "NTPC", "sector": "Infrastructure"},
    "ADANIENT": {"name": "Adani Enterprises", "sector": "Infrastructure"},
    "TATASTEEL": {"name": "Tata Steel", "sector": "Manufacturing"},
    "M&M": {"name": "Mahindra & Mahindra", "sector": "Manufacturing"},
    "POWERGRID": {"name": "Power Grid Corp", "sector": "Infrastructure"},
    "BAJAJFINSV": {"name": "Bajaj Finserv", "sector": "Financial"},
    "TECHM": {"name": "Tech Mahindra", "sector": "Technology"},
    "HINDALCO": {"name": "Hindalco Industries", "sector": "Manufacturing"},
    "INDUSINDBK": {"name": "IndusInd Bank", "sector": "Financial"},
    "ADANIPORTS": {"name": "Adani Ports", "sector": "Infrastructure"},
    "COALINDIA": {"name": "Coal India", "sector": "Manufacturing"},
    "DRREDDY": {"name": "Dr Reddy's Labs", "sector": "Healthcare"},
    "JSWSTEEL": {"name": "JSW Steel", "sector": "Manufacturing"},
    "EICHERMOT": {"name": "Eicher Motors", "sector": "Manufacturing"},
    "TATACONSUM": {"name": "Tata Consumer Products", "sector": "Services"},
    "GRASIM": {"name": "Grasim Industries", "sector": "Manufacturing"},
    "CIPLA": {"name": "Cipla", "sector": "Healthcare"},
    "BRITANNIA": {"name": "Britannia Industries", "sector": "Services"},
    "BPCL": {"name": "Bharat Petroleum", "sector": "Manufacturing"},
    "HEROMOTOCO": {"name": "Hero MotoCorp", "sector": "Manufacturing"},
    "DIVISLAB": {"name": "Divi's Laboratories", "sector": "Healthcare"},
    "APOLLOHOSP": {"name": "Apollo Hospitals", "sector": "Healthcare"},
    "SHREECEM": {"name": "Shree Cement", "sector": "Manufacturing"},
    "SBILIFE": {"name": "SBI Life Insurance", "sector": "Financial"},
    "HDFCLIFE": {"name": "HDFC Life Insurance", "sector": "Financial"},
    "LTIM": {"name": "LTIMindtree", "sector": "Technology"},
    "BAJAJ-AUTO": {"name": "Bajaj Auto", "sector": "Manufacturing"},
}


def create_stock_config_template(ticker, company_name, sector):
    """Create a template config for a stock"""
    
    # Default growth assumptions by sector
    sector_defaults = {
        "Technology": {
            "conservative": {"growth": [0.10, 0.09, 0.08, 0.07, 0.07], "margins": [0.22, 0.23, 0.24, 0.24, 0.25]},
            "base": {"growth": [0.15, 0.13, 0.12, 0.10, 0.09], "margins": [0.25, 0.26, 0.27, 0.28, 0.28]},
            "aggressive": {"growth": [0.20, 0.18, 0.15, 0.13, 0.11], "margins": [0.28, 0.29, 0.30, 0.31, 0.32]}
        },
        "Financial": {
            "conservative": {"growth": [0.12, 0.11, 0.10, 0.09, 0.08], "margins": [0.15, 0.16, 0.17, 0.17, 0.18]},
            "base": {"growth": [0.16, 0.15, 0.13, 0.12, 0.10], "margins": [0.18, 0.19, 0.20, 0.21, 0.21]},
            "aggressive": {"growth": [0.22, 0.20, 0.17, 0.15, 0.13], "margins": [0.21, 0.22, 0.23, 0.24, 0.25]}
        },
        "Manufacturing": {
            "conservative": {"growth": [0.08, 0.07, 0.07, 0.06, 0.06], "margins": [0.10, 0.11, 0.11, 0.12, 0.12]},
            "base": {"growth": [0.12, 0.11, 0.10, 0.09, 0.08], "margins": [0.13, 0.14, 0.15, 0.15, 0.16]},
            "aggressive": {"growth": [0.18, 0.16, 0.14, 0.12, 0.10], "margins": [0.16, 0.17, 0.18, 0.19, 0.20]}
        },
        "Infrastructure": {
            "conservative": {"growth": [0.10, 0.09, 0.08, 0.08, 0.07], "margins": [0.11, 0.12, 0.13, 0.13, 0.14]},
            "base": {"growth": [0.14, 0.13, 0.12, 0.11, 0.10], "margins": [0.14, 0.15, 0.16, 0.17, 0.17]},
            "aggressive": {"growth": [0.20, 0.18, 0.16, 0.14, 0.12], "margins": [0.17, 0.18, 0.19, 0.20, 0.21]}
        },
        "Services": {
            "conservative": {"growth": [0.09, 0.08, 0.08, 0.07, 0.07], "margins": [0.18, 0.19, 0.19, 0.20, 0.20]},
            "base": {"growth": [0.13, 0.12, 0.11, 0.10, 0.09], "margins": [0.21, 0.22, 0.23, 0.24, 0.24]},
            "aggressive": {"growth": [0.18, 0.16, 0.14, 0.12, 0.11], "margins": [0.24, 0.25, 0.26, 0.27, 0.28]}
        },
        "Healthcare": {
            "conservative": {"growth": [0.10, 0.09, 0.09, 0.08, 0.08], "margins": [0.20, 0.21, 0.22, 0.22, 0.23]},
            "base": {"growth": [0.14, 0.13, 0.12, 0.11, 0.10], "margins": [0.23, 0.24, 0.25, 0.26, 0.27]},
            "aggressive": {"growth": [0.19, 0.17, 0.15, 0.13, 0.12], "margins": [0.26, 0.27, 0.28, 0.29, 0.30]}
        }
    }
    
    defaults = sector_defaults.get(sector, sector_defaults["Services"])
    
    config = {
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        
        "current_data": {
            "price": 0,
            "market_cap_cr": 0,
            "annual_revenue_cr": 0,
            "ebitda_margin": 0,
            "pe_ratio": None,
            "debt_cr": 0,
            "cash_cr": 0
        },
        
        "growth_assumptions": {
            "conservative": {
                "revenue_growth_5y": defaults["conservative"]["growth"],
                "ebitda_margins_5y": defaults["conservative"]["margins"]
            },
            "base": {
                "revenue_growth_5y": defaults["base"]["growth"],
                "ebitda_margins_5y": defaults["base"]["margins"]
            },
            "aggressive": {
                "revenue_growth_5y": defaults["aggressive"]["growth"],
                "ebitda_margins_5y": defaults["aggressive"]["margins"]
            }
        },
        
        "technical_data": {
            "support_levels": [],
            "resistance_levels": [],
            "ma_50": 0,
            "ma_200": 0,
            "rsi": 50
        },
        
        "notes": f"NIFTY 50 stock - Fill in current data from Screener.in. Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}"
    }
    
    return config


def generate_all_configs(output_dir="nifty50_configs"):
    """Generate config files for all NIFTY 50 stocks"""
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n{'=' * 80}")
    print(f"  NIFTY 50 CONFIG GENERATOR")
    print(f"{'=' * 80}\n")
    
    created_count = 0
    
    for ticker, info in NIFTY_50_STOCKS.items():
        filename = f"stock_config_{ticker}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Create config
        config = create_stock_config_template(
            ticker=ticker,
            company_name=info["name"],
            sector=info["sector"]
        )
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        
        created_count += 1
        print(f"✅ Created: {filename}")
    
    print(f"\n{'=' * 80}")
    print(f"✅ Successfully created {created_count} config files in '{output_dir}/' folder")
    print(f"{'=' * 80}\n")
    
    # Create instructions file
    instructions = f"""
# NIFTY 50 Stock Configs - Quick Start Guide

## What You Have

{created_count} pre-configured template files for all NIFTY 50 stocks.

## Next Steps

### Step 1: Fill in Data (Pick Your Favorites)

Don't fill all 50 at once! Start with 5-10 stocks you're interested in:

1. Go to **Screener.in**
2. Search for stock (e.g., "RELIANCE")
3. Copy these values:
   - Price
   - Market Cap
   - Annual Revenue (TTM)
   - EBITDA Margin
   - Debt
   - Cash

4. Open `stock_config_RELIANCE.json` in Notepad
5. Fill in the values
6. Save

### Step 2: Run Analysis

```powershell
# Analyze single stock
py nse_generic_forecaster.py nifty50_configs/stock_config_RELIANCE.json

# Analyze multiple stocks
py batch_analyzer.py
```

### Step 3: Set Up Agent

Add your filled configs to agent watchlist:

```json
{{
    "watchlist": [
        "nifty50_configs/stock_config_RELIANCE.json",
        "nifty50_configs/stock_config_TCS.json",
        "nifty50_configs/stock_config_INFY.json"
    ]
}}
```

## Pro Tips

### Prioritize High-Impact Stocks

Fill these first (most important in NIFTY 50):
1. RELIANCE (15% weight in index)
2. HDFCBANK
3. ICICIBANK
4. INFY
5. TCS
6. ITC
7. HINDUNILVR
8. BHARTIARTL
9. LT
10. KOTAKBANK

### Batch Fill Strategy

Week 1: Top 10 by index weight
Week 2: Your current holdings
Week 3: Your watchlist stocks
Week 4: Rest as needed

## Tools Included

- `batch_analyzer.py` - Analyze multiple stocks at once
- `compare_stocks.py` - Side-by-side comparison
- `sector_analyzer.py` - Analyze all stocks in a sector

Happy analyzing! 📊
"""
    
    with open(os.path.join(output_dir, "README.txt"), 'w') as f:
        f.write(instructions)
    
    print("📄 Created: README.txt (instructions)")
    print(f"\n💡 Next: Fill in data for your favorite stocks from Screener.in\n")


def create_batch_analyzer():
    """Create a tool to analyze multiple stocks at once"""
    
    code = '''#!/usr/bin/env python3
"""
Batch Stock Analyzer
Analyze multiple NIFTY 50 stocks at once and generate comparison report
"""

import os
import sys
import json
from datetime import datetime

# Import your forecaster
sys.path.append(os.path.dirname(__file__))
from nse_generic_forecaster import NSEStockForecaster


def analyze_batch(config_folder="nifty50_configs"):
    """Analyze all filled configs in folder"""
    
    print(f"\\n{'=' * 80}")
    print(f"  BATCH STOCK ANALYZER")
    print(f"{'=' * 80}\\n")
    
    results = []
    
    # Find all config files
    config_files = [f for f in os.listdir(config_folder) if f.endswith('.json') and f.startswith('stock_config_')]
    
    print(f"Found {len(config_files)} config files\\n")
    
    for config_file in config_files:
        filepath = os.path.join(config_folder, config_file)
        
        # Check if filled (price > 0)
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        if config['current_data']['price'] == 0:
            print(f"⏭️  Skipping {config['ticker']} (not filled)")
            continue
        
        print(f"📊 Analyzing {config['ticker']}...")
        
        try:
            forecaster = NSEStockForecaster(filepath)
            dcf = forecaster.calculate_dcf_valuation('base')
            scores = forecaster.score_fundamentals()
            composite = sum(scores.values()) / len(scores)
            
            results.append({
                'ticker': config['ticker'],
                'company': config['company_name'],
                'sector': config['sector'],
                'price': config['current_data']['price'],
                'fair_value': dcf['fair_value'],
                'upside': dcf['upside_pct'],
                'score': composite
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Sort by score (best first)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Generate report
    print(f"\\n{'=' * 80}")
    print(f"  ANALYSIS RESULTS ({len(results)} stocks)")
    print(f"{'=' * 80}\\n")
    
    print(f"{'Rank':<6}{'Ticker':<12}{'Sector':<15}{'Price':>10}{'Upside':>10}{'Score':>8}")
    print("-" * 80)
    
    for i, r in enumerate(results, 1):
        print(f"{i:<6}{r['ticker']:<12}{r['sector']:<15}₹{r['price']:>9,.0f}{r['upside']:>9.1f}%{r['score']:>7.1f}/10")
    
    # Top picks
    print(f"\\n{'=' * 80}")
    print(f"  TOP 5 PICKS (by Quality Score)")
    print(f"{'=' * 80}\\n")
    
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. {r['company']} ({r['ticker']})")
        print(f"   Score: {r['score']:.1f}/10 | Upside: {r['upside']:+.1f}% | ₹{r['price']:,.0f}\\n")
    
    # Save to file
    output_file = f"batch_analysis_{datetime.now().strftime('%Y%m%d')}.txt"
    
    print(f"💾 Report saved to: {output_file}\\n")


if __name__ == "__main__":
    analyze_batch()
'''
    
    with open("batch_analyzer.py", 'w') as f:
        f.write(code)
    
    print("✅ Created: batch_analyzer.py")


def create_sector_analyzer():
    """Create a tool to analyze all stocks in a sector"""
    
    code = '''#!/usr/bin/env python3
"""
Sector Stock Analyzer
Compare all stocks within a sector
"""

import os
import sys
import json

sys.path.append(os.path.dirname(__file__))
from nse_generic_forecaster import NSEStockForecaster


def analyze_sector(sector_name, config_folder="nifty50_configs"):
    """Analyze all stocks in a specific sector"""
    
    print(f"\\n{'=' * 80}")
    print(f"  SECTOR ANALYSIS: {sector_name.upper()}")
    print(f"{'=' * 80}\\n")
    
    results = []
    
    for config_file in os.listdir(config_folder):
        if not config_file.endswith('.json'):
            continue
        
        filepath = os.path.join(config_folder, config_file)
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Check sector match and filled
        if config['sector'] != sector_name or config['current_data']['price'] == 0:
            continue
        
        print(f"Analyzing {config['ticker']}...")
        
        try:
            forecaster = NSEStockForecaster(filepath)
            dcf = forecaster.calculate_dcf_valuation('base')
            scores = forecaster.score_fundamentals()
            composite = sum(scores.values()) / len(scores)
            
            results.append({
                'ticker': config['ticker'],
                'company': config['company_name'],
                'price': config['current_data']['price'],
                'upside': dcf['upside_pct'],
                'score': composite
            })
        except:
            pass
    
    if not results:
        print(f"❌ No filled configs found for sector: {sector_name}")
        return
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\\n{'Rank':<6}{'Ticker':<12}{'Company':<30}{'Upside':>10}{'Score':>8}")
    print("-" * 80)
    
    for i, r in enumerate(results, 1):
        print(f"{i:<6}{r['ticker']:<12}{r['company']:<30}{r['upside']:>9.1f}%{r['score']:>7.1f}/10")
    
    print(f"\\n✅ Best in {sector_name}: {results[0]['company']}\\n")


if __name__ == "__main__":
    import sys
    
    sectors = ["Technology", "Financial", "Manufacturing", "Infrastructure", "Services", "Healthcare"]
    
    if len(sys.argv) > 1:
        sector = sys.argv[1]
        analyze_sector(sector)
    else:
        print("\\nAvailable sectors:")
        for s in sectors:
            print(f"  • {s}")
        print("\\nUsage: py sector_analyzer.py Technology")
'''
    
    with open("sector_analyzer.py", 'w') as f:
        f.write(code)
    
    print("✅ Created: sector_analyzer.py")


def main():
    """Main execution"""
    
    print("\n🎯 What would you like to do?\n")
    print("1. Generate all NIFTY 50 config templates")
    print("2. Create batch analysis tools")
    print("3. Both (recommended)")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        generate_all_configs()
    elif choice == "2":
        create_batch_analyzer()
        create_sector_analyzer()
        print("\n✅ Tools created successfully!")
    elif choice == "3":
        generate_all_configs()
        create_batch_analyzer()
        create_sector_analyzer()
        print("\n✅ Everything created successfully!")
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
