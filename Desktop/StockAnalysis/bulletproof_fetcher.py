#!/usr/bin/env python3
"""
Bulletproof Stock Data Fetcher
Uses yfinance library - most reliable way to get Yahoo Finance data
"""

import json
import os
import sys
from datetime import datetime

# Check if yfinance is installed
try:
    import yfinance as yf
except ImportError:
    print("\n❌ yfinance library not installed")
    print("\n📦 Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "--quiet"])
    import yfinance as yf
    print("✅ yfinance installed successfully!\n")


class BulletproofFetcher:
    """
    Uses yfinance library - handles all the API complexity for us
    """
    
    def fetch_stock_data(self, ticker):
        """
        Fetch complete stock data using yfinance
        """
        try:
            # Create ticker object
            stock = yf.Ticker(f"{ticker}.NS")
            
            # Get current info
            info = stock.info
            
            # Get historical data to confirm stock exists
            hist = stock.history(period="1d")
            
            if hist.empty:
                return {'success': False, 'error': 'No data available'}
            
            # Extract data
            price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)
            
            # Fallback to latest close price if currentPrice not available
            if price == 0 and not hist.empty:
                price = hist['Close'].iloc[-1]
            
            # Market cap (in local currency - INR)
            market_cap = info.get('marketCap', 0)
            market_cap_cr = market_cap / 10000000  # Convert to Crores
            
            # Revenue (total revenue in local currency)
            revenue = info.get('totalRevenue', 0)
            revenue_cr = revenue / 10000000
            
            # EBITDA Margin
            ebitda_margins = info.get('ebitdaMargins', 0)
            
            # Debt and Cash
            total_debt = info.get('totalDebt', 0)
            debt_cr = total_debt / 10000000
            
            total_cash = info.get('totalCash', 0)
            cash_cr = total_cash / 10000000
            
            # PE Ratio
            pe_ratio = info.get('trailingPE', None)
            
            return {
                'success': True,
                'price': round(price, 2),
                'market_cap_cr': round(market_cap_cr, 2),
                'annual_revenue_cr': round(revenue_cr, 2),
                'ebitda_margin': round(ebitda_margins, 4),
                'debt_cr': round(debt_cr, 2),
                'cash_cr': round(cash_cr, 2),
                'pe_ratio': round(pe_ratio, 2) if pe_ratio else None,
                'timestamp': datetime.now().isoformat(),
                'source': 'Yahoo Finance (yfinance)'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_config(self, ticker, config_folder="nifty50_configs"):
        """Update config file with fetched data"""
        
        # Try both possible locations
        possible_paths = [
            os.path.join(config_folder, f"stock_config_{ticker}.json"),
            f"stock_config_{ticker}.json"
        ]
        
        config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if not config_path:
            print(f"❌ Config not found for {ticker}")
            print(f"   Expected: {possible_paths[0]}")
            return False
        
        print(f"\n{'─' * 60}")
        print(f"Fetching {ticker}...")
        print(f"{'─' * 60}")
        
        # Fetch data
        data = self.fetch_stock_data(ticker)
        
        if not data['success']:
            print(f"❌ Failed: {data.get('error', 'Unknown error')}")
            return False
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Count what we got
        fields_updated = []
        
        # Update fields
        if data['price'] > 0:
            config['current_data']['price'] = data['price']
            fields_updated.append('Price')
        
        if data['market_cap_cr'] > 0:
            config['current_data']['market_cap_cr'] = data['market_cap_cr']
            fields_updated.append('Market Cap')
        
        if data['annual_revenue_cr'] > 0:
            config['current_data']['annual_revenue_cr'] = data['annual_revenue_cr']
            fields_updated.append('Revenue')
        
        if data['ebitda_margin'] > 0:
            config['current_data']['ebitda_margin'] = data['ebitda_margin']
            fields_updated.append('EBITDA Margin')
        
        if data['debt_cr'] > 0:
            config['current_data']['debt_cr'] = data['debt_cr']
            fields_updated.append('Debt')
        
        if data['cash_cr'] > 0:
            config['current_data']['cash_cr'] = data['cash_cr']
            fields_updated.append('Cash')
        
        if data['pe_ratio']:
            config['current_data']['pe_ratio'] = data['pe_ratio']
            fields_updated.append('P/E')
        
        # Add metadata
        config['last_updated'] = data['timestamp']
        config['data_source'] = data['source']
        
        # Save
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Print results
        print(f"\n✅ SUCCESS!")
        print(f"   Price: ₹{data['price']:,.2f}")
        
        if data['market_cap_cr'] > 0:
            print(f"   Market Cap: ₹{data['market_cap_cr']:,.0f} Cr")
        
        if data['annual_revenue_cr'] > 0:
            print(f"   Revenue: ₹{data['annual_revenue_cr']:,.0f} Cr")
        
        if data['ebitda_margin'] > 0:
            print(f"   EBITDA Margin: {data['ebitda_margin']*100:.2f}%")
        
        if data['debt_cr'] > 0:
            print(f"   Debt: ₹{data['debt_cr']:,.0f} Cr")
        
        if data['cash_cr'] > 0:
            print(f"   Cash: ₹{data['cash_cr']:,.0f} Cr")
        
        if data['pe_ratio']:
            print(f"   P/E Ratio: {data['pe_ratio']:.2f}")
        
        print(f"\n📊 Updated {len(fields_updated)}/7 fields: {', '.join(fields_updated)}")
        
        if len(fields_updated) < 7:
            print(f"⚠️  Missing fields need manual entry from Screener.in")
        
        return True
    
    def update_multiple(self, tickers, config_folder="nifty50_configs"):
        """Update multiple stocks"""
        
        print("\n" + "=" * 80)
        print("  BULLETPROOF STOCK DATA FETCHER")
        print("=" * 80)
        
        success = 0
        failed = 0
        
        for ticker in tickers:
            if self.update_config(ticker, config_folder):
                success += 1
            else:
                failed += 1
            
            # Small delay to be nice to servers
            import time
            time.sleep(1)
        
        print("\n" + "=" * 80)
        print(f"✅ Successfully updated: {success}")
        print(f"❌ Failed: {failed}")
        print("=" * 80 + "\n")


def main():
    """Main execution"""
    
    fetcher = BulletproofFetcher()
    
    if len(sys.argv) > 1:
        # Update specific stock(s)
        tickers = [t.upper() for t in sys.argv[1:]]
        fetcher.update_multiple(tickers)
    else:
        # Show usage
        print("\n" + "=" * 80)
        print("  BULLETPROOF STOCK DATA FETCHER")
        print("=" * 80 + "\n")
        
        print("Usage:")
        print("  py bulletproof_fetcher.py RELIANCE          # Fetch one stock")
        print("  py bulletproof_fetcher.py TCS INFY WIPRO    # Fetch multiple stocks")
        print()
        print("To fetch all NIFTY 50 stocks, use:")
        print("  py fetch_all_nifty50.py")
        print()


if __name__ == "__main__":
    main()
