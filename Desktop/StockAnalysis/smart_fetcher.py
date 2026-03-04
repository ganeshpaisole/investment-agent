#!/usr/bin/env python3
"""
No-API Stock Data Fetcher
Works without any API keys using Yahoo Finance
"""

import json
import os
import urllib.request
import urllib.error
import time
from datetime import datetime


class NoAPIFetcher:
    """
    Fetch stock data without any API keys
    Uses Yahoo Finance public data
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def fetch_yahoo_complete(self, ticker):
        """
        Fetch complete data from Yahoo Finance
        No API key needed - uses public endpoints
        """
        try:
            # Yahoo Finance uses .NS suffix for NSE stocks
            yahoo_symbol = f"{ticker}.NS"
            
            # Get quote data
            quote_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval=1d"
            
            req = urllib.request.Request(quote_url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                quote_data = json.loads(response.read().decode())
            
            result = quote_data['chart']['result'][0]
            meta = result['meta']
            
            # Get financial data
            stats_url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{yahoo_symbol}?modules=financialData,defaultKeyStatistics,summaryDetail"
            
            req = urllib.request.Request(stats_url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                stats_data = json.loads(response.read().decode())
            
            summary = stats_data['quoteSummary']['result'][0]
            
            # Extract data
            price = meta.get('regularMarketPrice', 0)
            
            # Get market cap (in USD)
            market_cap_usd = summary.get('summaryDetail', {}).get('marketCap', {}).get('raw', 0)
            
            # Get revenue (in local currency)
            revenue = summary.get('financialData', {}).get('totalRevenue', {}).get('raw', 0)
            
            # Get margins
            ebitda_margins = summary.get('financialData', {}).get('ebitdaMargins', {}).get('raw', 0)
            
            # Get debt and cash
            total_debt = summary.get('financialData', {}).get('totalDebt', {}).get('raw', 0)
            total_cash = summary.get('financialData', {}).get('totalCash', {}).get('raw', 0)
            
            # PE ratio
            pe_ratio = summary.get('summaryDetail', {}).get('trailingPE', {}).get('raw', None)
            
            # Convert to INR Crores
            # Yahoo returns revenue/debt/cash in INR for Indian stocks
            to_crores = 1 / 10000000  # Convert to Crores
            
            return {
                'price': round(price, 2),
                'market_cap_cr': round(market_cap_usd * 83 / 10000000, 2),  # USD to INR Cr
                'annual_revenue_cr': round(revenue * to_crores, 2) if revenue else 0,
                'ebitda_margin': round(ebitda_margins, 4) if ebitda_margins else 0,
                'debt_cr': round(total_debt * to_crores, 2) if total_debt else 0,
                'cash_cr': round(total_cash * to_crores, 2) if total_cash else 0,
                'pe_ratio': round(pe_ratio, 2) if pe_ratio else None,
                'source': 'Yahoo Finance',
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'price': 0
            }
    
    def update_config(self, ticker, config_folder="nifty50_configs"):
        """Update config file with fetched data"""
        
        config_path = os.path.join(config_folder, f"stock_config_{ticker}.json")
        
        if not os.path.exists(config_path):
            print(f"❌ Config not found: {config_path}")
            return False
        
        print(f"\nFetching {ticker}...", end=" ")
        
        # Fetch data
        data = self.fetch_yahoo_complete(ticker)
        
        if not data['success']:
            print(f"❌ Failed: {data.get('error', 'Unknown error')}")
            return False
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update all fields
        updated_fields = []
        
        if data['price'] > 0:
            config['current_data']['price'] = data['price']
            updated_fields.append('price')
        
        if data['market_cap_cr'] > 0:
            config['current_data']['market_cap_cr'] = data['market_cap_cr']
            updated_fields.append('market_cap')
        
        if data['annual_revenue_cr'] > 0:
            config['current_data']['annual_revenue_cr'] = data['annual_revenue_cr']
            updated_fields.append('revenue')
        
        if data['ebitda_margin'] > 0:
            config['current_data']['ebitda_margin'] = data['ebitda_margin']
            updated_fields.append('margins')
        
        if data['debt_cr'] > 0:
            config['current_data']['debt_cr'] = data['debt_cr']
            updated_fields.append('debt')
        
        if data['cash_cr'] > 0:
            config['current_data']['cash_cr'] = data['cash_cr']
            updated_fields.append('cash')
        
        if data['pe_ratio']:
            config['current_data']['pe_ratio'] = data['pe_ratio']
            updated_fields.append('pe')
        
        # Add metadata
        config['last_updated'] = data['timestamp']
        config['data_source'] = data['source']
        
        # Save
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ ₹{data['price']:,.2f}")
        print(f"   Updated: {', '.join(updated_fields)}")
        
        if len(updated_fields) < 7:
            print(f"   ⚠️ Some fields missing - Yahoo doesn't have complete data")
        
        return True
    
    def update_all(self, config_folder="nifty50_configs", delay=3):
        """Update all configs"""
        
        print("\n" + "=" * 80)
        print("  AUTO DATA FETCHER (No API Keys Needed)")
        print("=" * 80 + "\n")
        
        config_files = [f for f in os.listdir(config_folder) 
                       if f.startswith('stock_config_') and f.endswith('.json')]
        
        print(f"Found {len(config_files)} configs\n")
        
        success_count = 0
        fail_count = 0
        
        for config_file in config_files:
            ticker = config_file.replace('stock_config_', '').replace('.json', '')
            
            if self.update_config(ticker, config_folder):
                success_count += 1
            else:
                fail_count += 1
            
            # Rate limiting
            time.sleep(delay)
        
        print("\n" + "=" * 80)
        print(f"✅ Successfully updated: {success_count}")
        print(f"❌ Failed: {fail_count}")
        print("=" * 80 + "\n")


def main():
    """Main execution"""
    
    import sys
    
    fetcher = NoAPIFetcher()
    
    if len(sys.argv) > 1:
        # Update specific stocks
        for ticker in sys.argv[1:]:
            fetcher.update_config(ticker.upper())
    else:
        # Update all
        fetcher.update_all()


if __name__ == "__main__":
    main()
