#!/usr/bin/env python3
"""
Live NSE Data Fetcher
Automatically fetches current prices and updates config files

Uses multiple data sources:
1. NSE India API (primary)
2. Yahoo Finance (backup)
3. Google Finance (backup)
"""

import json
import os
import time
from datetime import datetime
import urllib.request
import urllib.error


class NSEDataFetcher:
    """Fetch live data from NSE and other sources"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    
    def fetch_from_yahoo(self, ticker):
        """
        Fetch data from Yahoo Finance
        Works without API key
        """
        try:
            # Yahoo Finance uses .NS suffix for NSE stocks
            yahoo_symbol = f"{ticker}.NS"
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
            
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            # Extract data
            result = data['chart']['result'][0]
            meta = result['meta']
            
            price = meta.get('regularMarketPrice', 0)
            prev_close = meta.get('previousClose', 0)
            
            # Calculate market cap (if available)
            shares = meta.get('sharesOutstanding', 0)
            market_cap_usd = shares * price if shares else 0
            market_cap_inr_cr = (market_cap_usd * 83) / 10000000  # Convert to INR Crores
            
            return {
                'price': round(price, 2),
                'market_cap_cr': round(market_cap_inr_cr, 2),
                'change': round(((price - prev_close) / prev_close * 100), 2) if prev_close else 0,
                'source': 'Yahoo Finance',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"  ⚠️ Yahoo Finance failed: {e}")
            return None
    
    def fetch_from_screener_web(self, ticker):
        """
        Fetch from Screener.in (web scraping - backup method)
        """
        try:
            url = f"https://www.screener.in/company/{ticker}/consolidated/"
            
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode()
            
            # Simple parsing (this is fragile and may break)
            # In production, use BeautifulSoup
            import re
            
            # Extract price
            price_match = re.search(r'Current Price.*?₹\s*([\d,]+\.?\d*)', html)
            price = float(price_match.group(1).replace(',', '')) if price_match else 0
            
            # Extract market cap
            mcap_match = re.search(r'Market Cap.*?₹\s*([\d,]+\.?\d*)\s*Cr', html)
            market_cap = float(mcap_match.group(1).replace(',', '')) if mcap_match else 0
            
            return {
                'price': round(price, 2),
                'market_cap_cr': round(market_cap, 2),
                'source': 'Screener.in',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"  ⚠️ Screener.in failed: {e}")
            return None
    
    def fetch_stock_data(self, ticker):
        """
        Try multiple sources in order of reliability
        """
        print(f"Fetching {ticker}...", end=" ")
        
        # Try Yahoo Finance first (most reliable)
        data = self.fetch_from_yahoo(ticker)
        
        if data and data['price'] > 0:
            print(f"✅ ₹{data['price']:,.2f} ({data['source']})")
            return data
        
        # Try Screener.in as backup
        data = self.fetch_from_screener_web(ticker)
        
        if data and data['price'] > 0:
            print(f"✅ ₹{data['price']:,.2f} ({data['source']})")
            return data
        
        print("❌ Failed")
        return None
    
    def update_config_file(self, config_path, live_data):
        """Update config file with live data"""
        
        # Load existing config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update with live data
        config['current_data']['price'] = live_data['price']
        
        if live_data.get('market_cap_cr', 0) > 0:
            config['current_data']['market_cap_cr'] = live_data['market_cap_cr']
        
        # Add metadata
        config['last_updated'] = live_data['timestamp']
        config['data_source'] = live_data['source']
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        return True
    
    def update_all_configs(self, config_folder="nifty50_configs", delay=2):
        """
        Update all config files with live prices
        
        Args:
            config_folder: Folder containing config files
            delay: Seconds to wait between requests (to avoid rate limiting)
        """
        
        print("\n" + "=" * 80)
        print("  LIVE DATA UPDATER - NSE Stock Prices")
        print("=" * 80 + "\n")
        
        config_files = [f for f in os.listdir(config_folder) 
                       if f.startswith('stock_config_') and f.endswith('.json')]
        
        print(f"Found {len(config_files)} config files\n")
        
        updated_count = 0
        failed_count = 0
        
        for config_file in config_files:
            config_path = os.path.join(config_folder, config_file)
            
            # Load config to get ticker
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            ticker = config['ticker']
            
            # Fetch live data
            live_data = self.fetch_stock_data(ticker)
            
            if live_data:
                # Update config file
                self.update_config_file(config_path, live_data)
                updated_count += 1
            else:
                failed_count += 1
            
            # Rate limiting - be nice to servers
            time.sleep(delay)
        
        print("\n" + "=" * 80)
        print(f"✅ Updated: {updated_count} stocks")
        print(f"❌ Failed: {failed_count} stocks")
        print("=" * 80 + "\n")
        
        if failed_count > 0:
            print("💡 Tip: Failed stocks may need manual data entry from Screener.in\n")


def update_single_stock(ticker, config_folder="nifty50_configs"):
    """Update a single stock config"""
    
    fetcher = NSEDataFetcher()
    
    config_file = f"stock_config_{ticker}.json"
    config_path = os.path.join(config_folder, config_file)
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    print(f"\nUpdating {ticker}...")
    live_data = fetcher.fetch_stock_data(ticker)
    
    if live_data:
        fetcher.update_config_file(config_path, live_data)
        print(f"✅ {ticker} updated successfully")
        return True
    else:
        print(f"❌ Failed to fetch data for {ticker}")
        return False


def main():
    """Main execution"""
    
    import sys
    
    fetcher = NSEDataFetcher()
    
    if len(sys.argv) > 1:
        # Update specific stock(s)
        tickers = sys.argv[1:]
        
        for ticker in tickers:
            update_single_stock(ticker.upper())
    
    else:
        # Update all stocks
        config_folder = "nifty50_configs"
        
        if not os.path.exists(config_folder):
            config_folder = "."  # Try current directory
        
        fetcher.update_all_configs(config_folder, delay=2)
        
        print("💾 All configs updated with live prices!")
        print("\n🔄 To update again: py live_data_fetcher.py")
        print("🎯 To update one stock: py live_data_fetcher.py RELIANCE\n")


if __name__ == "__main__":
    main()
