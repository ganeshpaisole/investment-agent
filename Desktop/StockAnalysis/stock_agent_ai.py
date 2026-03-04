#!/usr/bin/env python3
"""
Advanced AI Stock Agent with Claude Integration
Uses Claude API to make intelligent investment decisions
"""

import json
import os
from datetime import datetime


class AIStockAgent:
    """
    AI-powered agent that uses Claude to analyze stocks and make decisions
    """
    
    def __init__(self):
        self.conversation_history = []
        
    def analyze_with_ai(self, stock_data):
        """
        Use Claude API to analyze stock data and provide recommendations
        
        NOTE: This is a template. To use actual Claude API:
        1. Get API key from console.anthropic.com
        2. pip install anthropic
        3. Uncomment the code below
        """
        
        # TEMPLATE CODE (uncomment to use real Claude API):
        """
        import anthropic
        
        client = anthropic.Anthropic(
            api_key="your-api-key-here"
        )
        
        prompt = f'''
        You are an expert stock analyst. Analyze this data and provide investment advice:
        
        Company: {stock_data['company']}
        Current Price: ₹{stock_data['current_price']}
        Fair Value (DCF): ₹{stock_data['fair_value']}
        Upside: {stock_data['upside']}%
        Quality Score: {stock_data['quality_score']}/10
        
        Provide:
        1. Buy/Hold/Sell recommendation
        2. Risk level (Low/Medium/High)
        3. Target price (1 year)
        4. Key reasons for your recommendation
        
        Be concise and actionable.
        '''
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
        """
        
        # MOCK RESPONSE (for demonstration without API key)
        if stock_data['upside'] > 30 and stock_data['quality_score'] >= 7:
            recommendation = "STRONG BUY"
            risk = "Medium"
            reasoning = f"High quality company ({stock_data['quality_score']:.1f}/10) trading {stock_data['upside']:.1f}% below fair value"
        elif stock_data['upside'] > 0 and stock_data['quality_score'] >= 6:
            recommendation = "BUY"
            risk = "Medium"
            reasoning = f"Good fundamentals with {stock_data['upside']:.1f}% upside potential"
        elif stock_data['upside'] < -20:
            recommendation = "SELL"
            risk = "High"
            reasoning = f"Overvalued by {abs(stock_data['upside']):.1f}%"
        else:
            recommendation = "HOLD"
            risk = "Low"
            reasoning = "Fairly valued, no strong signals"
        
        target_price = stock_data['current_price'] * (1 + stock_data['upside']/100)
        
        return f"""
AI ANALYSIS for {stock_data['company']}:

Recommendation: {recommendation}
Risk Level: {risk}
Target Price (1Y): ₹{target_price:,.0f}

Reasoning: {reasoning}

Position Sizing: 
- If BUY: Start with 5% of portfolio, add on dips
- If HOLD: Maintain current position
- If SELL: Exit gradually over 2-4 weeks
"""


def main():
    """Demonstrate AI agent"""
    
    print("\n" + "=" * 60)
    print("🤖 AI STOCK AGENT - DEMONSTRATION")
    print("=" * 60 + "\n")
    
    # Example stock data
    stocks = [
        {
            'company': 'E2E Networks',
            'current_price': 2485,
            'fair_value': 5309,
            'upside': 58.4,
            'quality_score': 7.2
        },
        {
            'company': 'TCS',
            'current_price': 3850,
            'fair_value': 1486,
            'upside': -68.7,
            'quality_score': 6.2
        }
    ]
    
    agent = AIStockAgent()
    
    for stock in stocks:
        print(f"\nAnalyzing {stock['company']}...")
        analysis = agent.analyze_with_ai(stock)
        print(analysis)
        print("-" * 60)
    
    print("\n💡 To use real Claude API:")
    print("1. Get API key from console.anthropic.com")
    print("2. pip install anthropic")
    print("3. Edit stock_agent_ai.py and add your API key")
    print("4. Uncomment the Claude API code\n")


if __name__ == "__main__":
    main()
