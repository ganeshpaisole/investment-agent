#!/usr/bin/env python3
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
    
    print(f"\n{'=' * 80}")
    print(f"  SECTOR ANALYSIS: {sector_name.upper()}")
    print(f"{'=' * 80}\n")
    
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
    
    print(f"\n{'Rank':<6}{'Ticker':<12}{'Company':<30}{'Upside':>10}{'Score':>8}")
    print("-" * 80)
    
    for i, r in enumerate(results, 1):
        print(f"{i:<6}{r['ticker']:<12}{r['company']:<30}{r['upside']:>9.1f}%{r['score']:>7.1f}/10")
    
    print(f"\n✅ Best in {sector_name}: {results[0]['company']}\n")


if __name__ == "__main__":
    import sys
    
    sectors = ["Technology", "Financial", "Manufacturing", "Infrastructure", "Services", "Healthcare"]
    
    if len(sys.argv) > 1:
        sector = sys.argv[1]
        analyze_sector(sector)
    else:
        print("\nAvailable sectors:")
        for s in sectors:
            print(f"  • {s}")
        print("\nUsage: py sector_analyzer.py Technology")
