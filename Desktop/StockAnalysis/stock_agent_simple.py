#!/usr/bin/env python3
"""
Simple Stock Analysis Agent
Runs automatically and sends alerts via email/SMS
"""

import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import time

# Import your existing forecaster
import sys
sys.path.append(os.path.dirname(__file__))
from nse_generic_forecaster import NSEStockForecaster


class StockAgent:
    """
    Autonomous agent that monitors stocks and sends alerts
    """
    
    def __init__(self, config_file='agent_config.json'):
        """Initialize agent with configuration"""
        self.config = self.load_config(config_file)
        self.email_config = self.config.get('email', {})
        self.watchlist = self.config.get('watchlist', [])
        self.alert_rules = self.config.get('alert_rules', {})
        
    def load_config(self, config_file):
        """Load agent configuration"""
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def analyze_stock(self, stock_config_file):
        """Run analysis on a single stock"""
        try:
            forecaster = NSEStockForecaster(stock_config_file)
            
            # Get current data
            ticker = forecaster.ticker
            current_price = forecaster.current_price
            
            # Run analysis
            dcf = forecaster.calculate_dcf_valuation('base')
            scores = forecaster.score_fundamentals()
            composite = sum(scores.values()) / len(scores)
            
            return {
                'ticker': ticker,
                'company': forecaster.company_name,
                'current_price': current_price,
                'fair_value': dcf['fair_value'],
                'upside': dcf['upside_pct'],
                'quality_score': composite,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error analyzing {stock_config_file}: {e}")
            return None
    
    def check_alerts(self, analysis):
        """Check if any alert conditions are met"""
        alerts = []
        
        ticker = analysis['ticker']
        price = analysis['current_price']
        upside = analysis['upside']
        score = analysis['quality_score']
        
        # Alert rules
        rules = self.alert_rules.get(ticker, {})
        
        # Price drop alert
        if 'max_price' in rules and price < rules['max_price']:
            alerts.append({
                'type': 'PRICE_DROP',
                'message': f"{ticker} dropped to ₹{price:,.0f} (below ₹{rules['max_price']:,.0f})"
            })
        
        # Buy opportunity alert
        if 'buy_below' in rules and price <= rules['buy_below']:
            alerts.append({
                'type': 'BUY_OPPORTUNITY',
                'message': f"🟢 BUY ALERT: {ticker} at ₹{price:,.0f} (target: ₹{rules['buy_below']:,.0f})"
            })
        
        # Upside threshold alert
        if upside > 30 and score >= 6:
            alerts.append({
                'type': 'HIGH_UPSIDE',
                'message': f"📈 {ticker}: {upside:+.1f}% upside with {score:.1f}/10 quality"
            })
        
        # Quality improvement alert
        if 'min_score' in rules and score >= rules['min_score']:
            alerts.append({
                'type': 'QUALITY_IMPROVED',
                'message': f"⭐ {ticker} quality score: {score:.1f}/10"
            })
        
        return alerts
    
    def send_email_alert(self, subject, body):
        """Send email notification"""
        
        if not self.email_config.get('enabled', False):
            print("Email notifications disabled")
            return
        
        try:
            # Email configuration
            sender = self.email_config['sender_email']
            password = self.email_config['sender_password']
            recipient = self.email_config['recipient_email']
            smtp_server = self.email_config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = self.email_config.get('smtp_port', 587)
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Email sent: {subject}")
            
        except Exception as e:
            print(f"❌ Email failed: {e}")
    
    def generate_daily_report(self, analyses):
        """Generate daily summary report"""
        
        report = f"""
DAILY STOCK ANALYSIS REPORT
{datetime.now().strftime('%B %d, %Y')}
{'=' * 60}

"""
        
        for analysis in analyses:
            if analysis is None:
                continue
            
            report += f"""
{analysis['company']} ({analysis['ticker']})
Current Price: ₹{analysis['current_price']:,.0f}
Fair Value: ₹{analysis['fair_value']:,.0f} ({analysis['upside']:+.1f}%)
Quality Score: {analysis['quality_score']:.1f}/10

"""
        
        return report
    
    def run_daily_analysis(self):
        """Run analysis on all watchlist stocks"""
        
        print(f"\n{'=' * 60}")
        print(f"🤖 Stock Agent Running - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}\n")
        
        analyses = []
        all_alerts = []
        
        # Analyze each stock in watchlist
        for stock_config in self.watchlist:
            print(f"Analyzing {stock_config}...")
            
            analysis = self.analyze_stock(stock_config)
            if analysis:
                analyses.append(analysis)
                
                # Check for alerts
                alerts = self.check_alerts(analysis)
                if alerts:
                    all_alerts.extend(alerts)
                    for alert in alerts:
                        print(f"  🚨 {alert['type']}: {alert['message']}")
        
        # Generate daily report
        daily_report = self.generate_daily_report(analyses)
        print(daily_report)
        
        # Send email if there are alerts
        if all_alerts:
            alert_body = "\n".join([f"• {a['message']}" for a in all_alerts])
            full_report = f"ALERTS:\n{alert_body}\n\n{daily_report}"
            
            self.send_email_alert(
                subject=f"Stock Alerts - {len(all_alerts)} notifications",
                body=full_report
            )
        elif self.config.get('send_daily_summary', False):
            # Send daily summary even without alerts
            self.send_email_alert(
                subject=f"Daily Stock Report - {datetime.now().strftime('%b %d')}",
                body=daily_report
            )
        
        print(f"\n✅ Analysis complete. {len(all_alerts)} alerts generated.\n")
        
        return analyses, all_alerts
    
    def run_continuous(self, interval_hours=24):
        """Run agent continuously"""
        
        print(f"🤖 Agent started in continuous mode (every {interval_hours} hours)")
        
        while True:
            try:
                self.run_daily_analysis()
                
                # Wait for next run
                next_run = datetime.now().timestamp() + (interval_hours * 3600)
                print(f"⏰ Next run: {datetime.fromtimestamp(next_run).strftime('%Y-%m-%d %H:%M:%S')}")
                
                time.sleep(interval_hours * 3600)
                
            except KeyboardInterrupt:
                print("\n👋 Agent stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in continuous run: {e}")
                time.sleep(300)  # Wait 5 minutes before retry


def create_agent_config_template():
    """Create template configuration file for the agent"""
    
    template = {
        "email": {
            "enabled": False,
            "sender_email": "your_email@gmail.com",
            "sender_password": "your_app_password",
            "recipient_email": "your_email@gmail.com",
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587
        },
        
        "watchlist": [
            "stock_config_E2E.json",
            "stock_config_TCS.json"
        ],
        
        "alert_rules": {
            "E2E": {
                "buy_below": 2200,
                "max_price": 2500,
                "min_score": 7.0
            },
            "TCS": {
                "buy_below": 3500,
                "max_price": 4000,
                "min_score": 6.5
            }
        },
        
        "send_daily_summary": False,
        "run_time": "09:00",
        
        "notes": "Configure email and alert rules above"
    }
    
    with open('agent_config.json', 'w') as f:
        json.dump(template, f, indent=4)
    
    print("✅ Created agent_config.json template")
    print("\n📝 Next steps:")
    print("1. Edit agent_config.json with your email and alert rules")
    print("2. Run: python stock_agent_simple.py")


def main():
    """Main execution"""
    
    import sys
    
    # Check if config exists
    if not os.path.exists('agent_config.json'):
        print("❌ agent_config.json not found")
        print("\n🔧 Creating template...")
        create_agent_config_template()
        return
    
    # Create and run agent
    agent = StockAgent('agent_config.json')
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        # Run continuously
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 24
        agent.run_continuous(interval_hours=interval)
    else:
        # Run once
        agent.run_daily_analysis()


if __name__ == "__main__":
    main()
