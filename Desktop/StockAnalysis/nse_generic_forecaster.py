#!/usr/bin/env python3
"""
Generic NSE Stock Forecasting Model
Works for any NSE-listed stock - just edit the config file

Author: Quantitative Analysis Framework
Version: 1.0
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
import os


class NSEStockForecaster:
    """
    Universal stock forecasting model for NSE stocks
    Automatically adapts to different company characteristics
    """
    
    def __init__(self, config_file='stock_config.json'):
        """
        Initialize with stock configuration
        
        Args:
            config_file: Path to JSON config file with stock parameters
        """
        self.config = self.load_config(config_file)
        self.validate_config()
        
        # Extract core parameters
        self.ticker = self.config['ticker']
        self.company_name = self.config['company_name']
        self.sector = self.config['sector']
        
        # Financial data
        self.current_price = self.config['current_data']['price']
        self.market_cap = self.config['current_data']['market_cap_cr']
        self.annual_revenue = self.config['current_data']['annual_revenue_cr']
        self.ebitda_margin = self.config['current_data']['ebitda_margin']
        self.pe_ratio = self.config['current_data'].get('pe_ratio', None)
        self.debt = self.config['current_data'].get('debt_cr', 0)
        self.cash = self.config['current_data'].get('cash_cr', 0)
        
        # Growth parameters
        self.growth_scenarios = self.config['growth_assumptions']
        
        # Technical data
        self.technical = self.config['technical_data']
        
        # Shares outstanding
        self.shares_outstanding = self.market_cap / self.current_price if self.current_price > 0 else 1
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Config file '{config_file}' not found. "
                f"Please create it using create_stock_config() function."
            )
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def validate_config(self):
        """Validate that all required fields are present"""
        required_fields = [
            'ticker', 'company_name', 'sector', 
            'current_data', 'growth_assumptions', 'technical_data'
        ]
        
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")
    
    def calculate_dcf_valuation(self, scenario='base'):
        """
        Calculate DCF valuation for any stock
        Automatically adjusts for profitability stage
        """
        
        # Get growth rates and margins for scenario
        growth = self.growth_scenarios[scenario]['revenue_growth_5y']
        margin_trajectory = self.growth_scenarios[scenario]['ebitda_margins_5y']
        
        # Capex rate (adjust based on sector)
        capex_rates = {
            'Technology': 0.15,
            'Manufacturing': 0.25,
            'Infrastructure': 0.35,
            'Services': 0.10,
            'Healthcare': 0.20,
            'Financial': 0.05
        }
        capex_rate = capex_rates.get(self.sector, 0.20)  # Default 20%
        
        # Tax rate
        tax_rate = 0.25
        
        # WACC (adjust based on sector risk)
        wacc_ranges = {
            'Technology': (0.12, 0.16),
            'Manufacturing': (0.11, 0.14),
            'Infrastructure': (0.10, 0.13),
            'Services': (0.13, 0.17),
            'Healthcare': (0.11, 0.15),
            'Financial': (0.10, 0.12)
        }
        wacc_range = wacc_ranges.get(self.sector, (0.12, 0.15))
        wacc = np.mean(wacc_range)
        
        # Terminal growth
        terminal_growth = 0.06 if self.sector == 'Financial' else 0.08
        
        # Project 5 years
        projections = []
        revenue = self.annual_revenue
        
        for year, (g, margin) in enumerate(zip(growth, margin_trajectory), start=1):
            revenue = revenue * (1 + g)
            ebitda = revenue * margin
            nopat = ebitda * (1 - tax_rate)
            capex = revenue * capex_rate
            fcf = nopat - capex
            pv_fcf = fcf / ((1 + wacc) ** year)
            
            projections.append({
                'Year': datetime.now().year + year,
                'Revenue': revenue,
                'EBITDA': ebitda,
                'FCF': fcf,
                'PV_FCF': pv_fcf
            })
        
        df = pd.DataFrame(projections)
        
        # Terminal value
        final_fcf = projections[-1]['FCF']
        terminal_value = final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_terminal_value = terminal_value / ((1 + wacc) ** 5)
        
        # Enterprise and equity value
        enterprise_value = df['PV_FCF'].sum() + pv_terminal_value
        equity_value = enterprise_value - self.debt + self.cash
        fair_value = equity_value / self.shares_outstanding
        
        return {
            'scenario': scenario,
            'fair_value': fair_value,
            'enterprise_value': enterprise_value,
            'upside_pct': ((fair_value - self.current_price) / self.current_price) * 100,
            'projections': df
        }
    
    def calculate_relative_valuation(self):
        """
        Calculate fair value using industry multiples
        Better for growth/unprofitable companies
        """
        
        # Industry P/S multiples (approximate)
        industry_ps_multiples = {
            'Technology': {'low': 3, 'mid': 8, 'high': 15},
            'Manufacturing': {'low': 0.5, 'mid': 1.5, 'high': 3},
            'Infrastructure': {'low': 1, 'mid': 2.5, 'high': 5},
            'Services': {'low': 1, 'mid': 3, 'high': 6},
            'Healthcare': {'low': 2, 'mid': 5, 'high': 10},
            'Financial': {'low': 1, 'mid': 2, 'high': 4}
        }
        
        multiples = industry_ps_multiples.get(
            self.sector, 
            {'low': 1, 'mid': 3, 'high': 6}
        )
        
        # Adjust multiples based on growth rate
        base_growth = self.growth_scenarios['base']['revenue_growth_5y'][0]
        
        if base_growth > 0.40:  # High growth
            ps_multiple = multiples['high']
        elif base_growth > 0.20:  # Medium growth
            ps_multiple = multiples['mid']
        else:  # Low growth
            ps_multiple = multiples['low']
        
        # Calculate valuations
        valuations = {}
        for scenario in ['conservative', 'base', 'aggressive']:
            # Project 5-year revenue
            revenue = self.annual_revenue
            growth_rates = self.growth_scenarios[scenario]['revenue_growth_5y']
            
            for g in growth_rates:
                revenue = revenue * (1 + g)
            
            # Apply P/S multiple
            market_cap_5y = revenue * ps_multiple
            fair_value_5y = market_cap_5y / self.shares_outstanding
            
            # Discount back to present (using 15% discount rate)
            fair_value_today = fair_value_5y / ((1.15) ** 5)
            
            valuations[scenario] = {
                'fair_value_today': fair_value_today,
                'target_5y': fair_value_5y,
                'implied_cagr': ((fair_value_5y / self.current_price) ** (1/5) - 1)
            }
        
        return valuations
    
    def score_fundamentals(self):
        """Score fundamental strength (0-10 scale)"""
        
        scores = {}
        
        # Revenue growth score
        base_growth = self.growth_scenarios['base']['revenue_growth_5y'][0]
        if base_growth > 0.40:
            scores['growth'] = 10
        elif base_growth > 0.25:
            scores['growth'] = 8
        elif base_growth > 0.15:
            scores['growth'] = 6
        else:
            scores['growth'] = 4
        
        # Margin quality
        if self.ebitda_margin > 0.40:
            scores['margins'] = 10
        elif self.ebitda_margin > 0.25:
            scores['margins'] = 8
        elif self.ebitda_margin > 0.15:
            scores['margins'] = 6
        elif self.ebitda_margin > 0:
            scores['margins'] = 4
        else:
            scores['margins'] = 2
        
        # Balance sheet strength
        if self.debt == 0:
            scores['balance_sheet'] = 10
        elif self.debt < self.annual_revenue * 0.5:
            scores['balance_sheet'] = 8
        elif self.debt < self.annual_revenue * 1.5:
            scores['balance_sheet'] = 6
        else:
            scores['balance_sheet'] = 3
        
        # Valuation (P/S ratio)
        ps_ratio = self.market_cap / self.annual_revenue if self.annual_revenue > 0 else 999
        if ps_ratio < 2:
            scores['valuation'] = 10
        elif ps_ratio < 5:
            scores['valuation'] = 7
        elif ps_ratio < 10:
            scores['valuation'] = 5
        else:
            scores['valuation'] = 3
        
        return scores
    
    def generate_price_targets(self):
        """Generate year-by-year price targets"""
        
        targets = {}
        
        for scenario in ['conservative', 'base', 'aggressive']:
            scenario_targets = []
            
            # Get relative valuation for 5 years out
            rel_val = self.calculate_relative_valuation()
            target_5y = rel_val[scenario]['target_5y']
            
            # Interpolate yearly targets
            cagr = ((target_5y / self.current_price) ** (1/5) - 1)
            
            price = self.current_price
            for year in range(1, 6):
                price = price * (1 + cagr)
                scenario_targets.append({
                    'Year': datetime.now().year + year,
                    'Price_Target': price
                })
            
            targets[scenario] = pd.DataFrame(scenario_targets)
        
        return targets
    
    def generate_report(self, output_file=None):
        """Generate comprehensive analysis report"""
        
        print("\n" + "=" * 80)
        print(f"  {self.company_name} ({self.ticker}) - STOCK ANALYSIS REPORT")
        print("=" * 80 + "\n")
        
        print(f"Sector: {self.sector}")
        print(f"Current Price: ₹{self.current_price:,.2f}")
        print(f"Market Cap: ₹{self.market_cap:,.0f} Cr")
        print(f"Annual Revenue: ₹{self.annual_revenue:,.0f} Cr")
        print(f"EBITDA Margin: {self.ebitda_margin:.1%}")
        
        # Section 1: DCF Valuation
        print("\n" + "=" * 80)
        print("  DCF VALUATION")
        print("=" * 80 + "\n")
        
        for scenario in ['conservative', 'base', 'aggressive']:
            dcf = self.calculate_dcf_valuation(scenario)
            print(f"{scenario.upper()}: ₹{dcf['fair_value']:,.0f} ({dcf['upside_pct']:+.1f}%)")
        
        # Section 2: Relative Valuation
        print("\n" + "=" * 80)
        print("  RELATIVE VALUATION (P/S Multiple Method)")
        print("=" * 80 + "\n")
        
        rel_val = self.calculate_relative_valuation()
        
        print(f"{'Scenario':<15} {'Fair Value':<15} {'5Y Target':<15} {'Implied CAGR':<15}")
        print("-" * 60)
        for scenario, vals in rel_val.items():
            print(f"{scenario.title():<15} ₹{vals['fair_value_today']:<13,.0f} "
                  f"₹{vals['target_5y']:<13,.0f} {vals['implied_cagr']:<14.1%}")
        
        # Section 3: Fundamental Scores
        print("\n" + "=" * 80)
        print("  FUNDAMENTAL QUALITY SCORES (0-10)")
        print("=" * 80 + "\n")
        
        scores = self.score_fundamentals()
        composite = np.mean(list(scores.values()))
        
        for factor, score in scores.items():
            print(f"{factor.title():<20}: {score}/10")
        
        print(f"\nComposite Score: {composite:.1f}/10")
        
        if composite >= 8:
            rating = "🟢 STRONG BUY"
        elif composite >= 6:
            rating = "🟢 BUY"
        elif composite >= 5:
            rating = "🟡 HOLD"
        else:
            rating = "🔴 AVOID"
        
        print(f"Rating: {rating}")
        
        # Section 4: Price Targets
        print("\n" + "=" * 80)
        print("  YEAR-BY-YEAR PRICE TARGETS")
        print("=" * 80 + "\n")
        
        targets = self.generate_price_targets()
        
        # Combine into single table
        years = targets['base']['Year'].tolist()
        print(f"{'Year':<8}", end="")
        for scenario in ['conservative', 'base', 'aggressive']:
            print(f"{scenario.title():<15}", end="")
        print()
        print("-" * 53)
        
        for i, year in enumerate(years):
            print(f"{year:<8}", end="")
            for scenario in ['conservative', 'base', 'aggressive']:
                target = targets[scenario].iloc[i]['Price_Target']
                print(f"₹{target:<13,.0f}", end="")
            print()
        
        # Section 5: Technical Levels
        print("\n" + "=" * 80)
        print("  TECHNICAL LEVELS")
        print("=" * 80 + "\n")
        
        support = self.technical.get('support_levels', [])
        resistance = self.technical.get('resistance_levels', [])
        
        if support:
            print(f"Support Levels: {', '.join([f'₹{s:,.0f}' for s in support])}")
        if resistance:
            print(f"Resistance Levels: {', '.join([f'₹{r:,.0f}' for r in resistance])}")
        
        print(f"\nRecommended Entry Zones:")
        if support:
            print(f"  • Aggressive: ₹{self.current_price * 1.05:,.0f} (current + 5%)")
            print(f"  • Moderate: ₹{support[0]:,.0f} (first support)")
            if len(support) > 1:
                print(f"  • Conservative: ₹{support[1]:,.0f} (second support)")
        
        # Final recommendation
        print("\n" + "=" * 80)
        print("  INVESTMENT RECOMMENDATION")
        print("=" * 80 + "\n")
        
        dcf_base = self.calculate_dcf_valuation('base')
        rel_base = rel_val['base']
        
        avg_upside = (dcf_base['upside_pct'] + 
                     ((rel_base['fair_value_today'] - self.current_price) / self.current_price * 100)) / 2
        
        print(f"Average Fair Value Upside: {avg_upside:+.1f}%")
        print(f"5-Year Target (Base Case): ₹{rel_base['target_5y']:,.0f}")
        print(f"Expected CAGR: {rel_base['implied_cagr']:.1%}")
        
        if avg_upside > 30 and composite >= 6:
            action = "🟢 BUY - Initiate position with staggered entry"
        elif avg_upside > 0 and composite >= 5:
            action = "🟡 HOLD - Wait for better entry or accumulate slowly"
        else:
            action = "🔴 AVOID - Risk/reward not favorable"
        
        print(f"\nAction: {action}")
        
        print("\n" + "=" * 80)
        print("  END OF REPORT")
        print("=" * 80 + "\n")
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                # Redirect print to file (simplified)
                f.write(f"Analysis Report for {self.company_name}\n")
                f.write(f"Generated: {datetime.now()}\n")
        
        return {
            'dcf_valuation': dcf_base,
            'relative_valuation': rel_base,
            'scores': scores,
            'composite_score': composite,
            'targets': targets
        }


def create_stock_config_template(ticker, company_name, output_file='stock_config.json'):
    """
    Create a template configuration file for a new stock
    User fills in the values and saves
    """
    
    template = {
        "ticker": ticker,
        "company_name": company_name,
        "sector": "Technology",  # Change: Technology/Manufacturing/Infrastructure/Services/Healthcare/Financial
        
        "current_data": {
            "price": 0,  # Fill in current stock price
            "market_cap_cr": 0,  # Fill in market cap in Crores
            "annual_revenue_cr": 0,  # Fill in annual revenue in Crores
            "ebitda_margin": 0,  # Fill in as decimal (e.g., 0.25 for 25%)
            "pe_ratio": None,  # Optional: P/E ratio
            "debt_cr": 0,  # Optional: Total debt in Crores
            "cash_cr": 0  # Optional: Cash & equivalents in Crores
        },
        
        "growth_assumptions": {
            "conservative": {
                "revenue_growth_5y": [0.15, 0.12, 0.10, 0.08, 0.08],  # Yearly growth rates
                "ebitda_margins_5y": [0.20, 0.21, 0.22, 0.23, 0.24]  # Yearly margins
            },
            "base": {
                "revenue_growth_5y": [0.25, 0.22, 0.18, 0.15, 0.12],
                "ebitda_margins_5y": [0.25, 0.27, 0.29, 0.30, 0.31]
            },
            "aggressive": {
                "revenue_growth_5y": [0.40, 0.35, 0.30, 0.25, 0.20],
                "ebitda_margins_5y": [0.30, 0.33, 0.36, 0.38, 0.40]
            }
        },
        
        "technical_data": {
            "support_levels": [],  # List of support levels, e.g., [2200, 2000, 1800]
            "resistance_levels": [],  # List of resistance levels, e.g., [2800, 3000, 3500]
            "ma_50": 0,  # 50-day moving average
            "ma_200": 0,  # 200-day moving average
            "rsi": 50  # Current RSI (0-100)
        },
        
        "notes": "Template created. Fill in all values before running analysis."
    }
    
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=4)
    
    print(f"\n✅ Configuration template created: {output_file}")
    print(f"\n📝 Next steps:")
    print(f"1. Open {output_file} in a text editor")
    print(f"2. Fill in all the values (price, revenue, growth rates, etc.)")
    print(f"3. Save the file")
    print(f"4. Run: python3 nse_generic_forecaster.py")
    print(f"\n💡 Tip: You can create multiple config files for different stocks")
    print(f"   Example: stock_config_TCS.json, stock_config_INFY.json, etc.\n")


def main():
    """Main execution - analyze stock from config file"""
    
    import sys
    
    # Check if config file specified
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'stock_config.json'
    
    # Check if config exists
    if not os.path.exists(config_file):
        print(f"\n❌ Config file '{config_file}' not found!")
        print(f"\n🔧 Creating a template for you...\n")
        
        ticker = input("Enter NSE ticker symbol (e.g., E2E, TCS, INFY): ").strip().upper()
        company = input("Enter company name (e.g., E2E Networks): ").strip()
        
        if ticker and company:
            create_stock_config_template(ticker, company, config_file)
        else:
            print("\n❌ Ticker and company name required!")
        
        return
    
    # Load and analyze
    try:
        forecaster = NSEStockForecaster(config_file)
        results = forecaster.generate_report()
        
        # Save results
        output_file = f"analysis_{forecaster.ticker}_{datetime.now().strftime('%Y%m%d')}.txt"
        print(f"\n💾 Report saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\n💡 Make sure your config file is properly formatted!")


if __name__ == "__main__":
    main()
