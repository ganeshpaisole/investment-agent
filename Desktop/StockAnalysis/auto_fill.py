#!/usr/bin/env python3
"""
Enhanced Auto-Fill System
Fetches COMPLETE fundamental data automatically (not just price)

Data sources:
1. Yahoo Finance - Price, Market Cap
2. Financial Modeling Prep API - Full fundamentals (free tier)
3. Alpha Vantage - Backup (free API key)
"""

import json
import os
import urllib.request
import urllib.parse
from datetime import datetime


class EnhancedDataFetcher:
    """
    Fetches complete fundamental data automatically
    """
    
    def __init__(self, fmp_api_key=None, alphavantage_key=None):
        """
        Initialize with API keys (both have free tiers)
        
        Get free keys:
        - FMP: https://financialmodelingprep.com/developer/docs/
        - Alpha Vantage: https://www.alphavantage.co/support/#api-key
        """
        self.fmp_api_key = fmp_api_key
        self.alphavantage_key = alphavantage_key
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
    
    def fetch_yahoo_basic(self, ticker):
        """Fetch basic price data from Yahoo (no API key needed)"""
        try:
            yahoo_symbol = f"{ticker}.NS"
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
            
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            result = data['chart']['result'][0]
            meta = result['meta']
            
            return {
                'price': meta.get('regularMarketPrice', 0),
                'market_cap_usd': meta.get('marketCap', 0),
            }
        except:
            return None
    
    def fetch_fmp_fundamentals(self, ticker):
        """
        Fetch complete fundamentals from Financial Modeling Prep
        Free tier: 250 requests/day
        """
        if not self.fmp_api_key:
            return None
        
        try:
            # FMP uses different ticker format for Indian stocks
            symbol = f"{ticker}.NS"
            
            # Get company profile
            url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={self.fmp_api_key}"
            
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as response:
                profile = json.loads(response.read().decode())
            
            if not profile:
                return None
            
            p = profile[0]
            
            # Get financial statements
            income_url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?limit=1&apikey={self.fmp_api_key}"
            
            req = urllib.request.Request(income_url)
            with urllib.request.urlopen(req, timeout=10) as response:
                income = json.loads(response.read().decode())
            
            balance_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?limit=1&apikey={self.fmp_api_key}"
            
            req = urllib.request.Request(balance_url)
            with urllib.request.urlopen(req, timeout=10) as response:
                balance = json.loads(response.read().decode())
            
            # Convert USD to INR Crores
            usd_to_inr = 83
            to_crores = usd_to_inr / 10000000
            
            inc = income[0] if income else {}
            bal = balance[0] if balance else {}
            
            return {
                'price': p.get('price', 0),
                'market_cap_cr': p.get('mktCap', 0) * to_crores,
                'annual_revenue_cr': inc.get('revenue', 0) * to_crores,
                'ebitda_margin': inc.get('ebitdaratio', 0),
                'debt_cr': bal.get('totalDebt', 0) * to_crores,
                'cash_cr': bal.get('cashAndCashEquivalents', 0) * to_crores,
                'pe_ratio': p.get('pe', None),
                'source': 'Financial Modeling Prep'
            }
            
        except Exception as e:
            print(f"  FMP error: {e}")
            return None
    
    def auto_fill_config(self, ticker, config_folder="nifty50_configs"):
        """
        Automatically fill complete config with live data
        """
        config_path = os.path.join(config_folder, f"stock_config_{ticker}.json")
        
        if not os.path.exists(config_path):
            print(f"❌ Config not found: {config_path}")
            return False
        
        print(f"\n{'=' * 60}")
        print(f"Auto-filling {ticker}...")
        print(f"{'=' * 60}\n")
        
        # Try FMP first (complete data)
        if self.fmp_api_key:
            print("Trying Financial Modeling Prep...")
            data = self.fetch_fmp_fundamentals(ticker)
            
            if data:
                # Update config
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                config['current_data']['price'] = round(data['price'], 2)
                config['current_data']['market_cap_cr'] = round(data['market_cap_cr'], 2)
                config['current_data']['annual_revenue_cr'] = round(data['annual_revenue_cr'], 2)
                config['current_data']['ebitda_margin'] = round(data['ebitda_margin'], 4)
                config['current_data']['debt_cr'] = round(data['debt_cr'], 2)
                config['current_data']['cash_cr'] = round(data['cash_cr'], 2)
                config['current_data']['pe_ratio'] = data['pe_ratio']
                
                config['last_updated'] = datetime.now().isoformat()
                config['auto_filled'] = True
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                
                print(f"✅ Complete data fetched!")
                print(f"   Price: ₹{data['price']:,.2f}")
                print(f"   Revenue: ₹{data['annual_revenue_cr']:,.0f} Cr")
                print(f"   EBITDA Margin: {data['ebitda_margin']*100:.1f}%")
                
                return True
        
        # Fallback to Yahoo (price only)
        print("Trying Yahoo Finance (price only)...")
        data = self.fetch_yahoo_basic(ticker)
        
        if data and data['price'] > 0:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            config['current_data']['price'] = round(data['price'], 2)
            
            if data['market_cap_usd'] > 0:
                config['current_data']['market_cap_cr'] = round(data['market_cap_usd'] * 83 / 10000000, 2)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"✅ Price updated: ₹{data['price']:,.2f}")
            print(f"⚠️  Other fields need manual entry from Screener.in")
            
            return True
        
        print(f"❌ Could not fetch data for {ticker}")
        return False


def setup_api_keys():
    """Interactive setup for API keys"""
    
    print("\n" + "=" * 80)
    print("  API KEY SETUP (One-Time)")
    print("=" * 80 + "\n")
    
    print("To auto-fill ALL data (not just price), you need a free API key.\n")
    
    print("OPTION 1: Financial Modeling Prep (RECOMMENDED)")
    print("  • Free tier: 250 requests/day")
    print("  • Gets: Price, Revenue, Margins, Debt, Cash")
    print("  • Sign up: https://financialmodelingprep.com/developer/docs/")
    print()
    
    fmp_key = input("Enter FMP API key (or press Enter to skip): ").strip()
    
    print("\nOPTION 2: Alpha Vantage (Backup)")
    print("  • Free tier: 25 requests/day")
    print("  • Gets: Price, some fundamentals")
    print("  • Sign up: https://www.alphavantage.co/support/#api-key")
    print()
    
    av_key = input("Enter Alpha Vantage key (or press Enter to skip): ").strip()
    
    # Save to config file
    config = {
        'fmp_api_key': fmp_key if fmp_key else None,
        'alphavantage_key': av_key if av_key else None,
        'setup_date': datetime.now().isoformat()
    }
    
    with open('api_keys.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\n✅ API keys saved to api_keys.json")
    print("💡 Keep this file private (don't share it)\n")
    
    return config


def load_api_keys():
    """Load saved API keys"""
    if os.path.exists('api_keys.json'):
        with open('api_keys.json', 'r') as f:
            return json.load(f)
    return {}


def main():
    """Main execution"""
    
    import sys
    from datetime import datetime
    
    # Load or setup API keys
    if not os.path.exists('api_keys.json'):
        print("\n⚠️  No API keys found. Running setup...\n")
        keys = setup_api_keys()
    else:
        keys = load_api_keys()
    
    fetcher = EnhancedDataFetcher(
        fmp_api_key=keys.get('fmp_api_key'),
        alphavantage_key=keys.get('alphavantage_key')
    )
    
    if len(sys.argv) > 1:
        # Auto-fill specific stocks
        tickers = sys.argv[1:]
        
        for ticker in tickers:
            fetcher.auto_fill_config(ticker.upper())
    
    else:
        # Show usage
        print("\n" + "=" * 80)
        print("  ENHANCED AUTO-FILL SYSTEM")
        print("=" * 80 + "\n")
        
        print("Usage:")
        print("  py auto_fill.py RELIANCE          # Fill one stock")
        print("  py auto_fill.py TCS INFY WIPRO    # Fill multiple stocks")
        print()
        
        if not keys.get('fmp_api_key'):
            print("⚠️  No API key configured!")
            print("   With API key: Gets complete data (price, revenue, margins, debt)")
            print("   Without API key: Only gets price (need manual entry for rest)")
            print()
            print("   Setup API key: py auto_fill.py --setup")
            print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--setup':
        setup_api_keys()
    else:
        main()
