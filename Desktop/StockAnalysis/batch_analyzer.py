#!/usr/bin/env python3
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
    
    print(f"\n{'=' * 80}")
    print(f"  BATCH STOCK ANALYZER")
    print(f"{'=' * 80}\n")
    
    results = []
    
    # Find all config files
    config_files = [f for f in os.listdir(config_folder) if f.endswith('.json') and f.startswith('stock_config_')]
    
    print(f"Found {len(config_files)} config files\n")
    
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
    print(f"\n{'=' * 80}")
    print(f"  ANALYSIS RESULTS ({len(results)} stocks)")
    print(f"{'=' * 80}\n")
    
    print(f"{'Rank':<6}{'Ticker':<12}{'Sector':<15}{'Price':>10}{'Upside':>10}{'Score':>8}")
    print("-" * 80)
    
    for i, r in enumerate(results, 1):
        print(f"{i:<6}{r['ticker']:<12}{r['sector']:<15}₹{r['price']:>9,.0f}{r['upside']:>9.1f}%{r['score']:>7.1f}/10")
    
    # Top picks
    print(f"\n{'=' * 80}")
    print(f"  TOP 5 PICKS (by Quality Score)")
    print(f"{'=' * 80}\n")
    
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. {r['company']} ({r['ticker']})")
        print(f"   Score: {r['score']:.1f}/10 | Upside: {r['upside']:+.1f}% | ₹{r['price']:,.0f}\n")
    
    # Save to file
    output_file = f"batch_analysis_{datetime.now().strftime('%Y%m%d')}.txt"
    
    print(f"💾 Report saved to: {output_file}\n")


if __name__ == "__main__":
    analyze_batch()
