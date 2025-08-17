#!/usr/bin/env python3
"""
Minimal working arbitrage bot
This actually runs and finds opportunities
"""

import os
import sys
import time
import json
from datetime import datetime

# Check for dependencies
try:
    import ccxt
    print("✅ ccxt found")
except ImportError:
    print("Installing ccxt...")
    os.system(f"{sys.executable} -m pip install -q ccxt")
    import ccxt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Installing python-dotenv...")
    os.system(f"{sys.executable} -m pip install -q python-dotenv")
    from dotenv import load_dotenv
    load_dotenv()

class SimpleArbitrageBot:
    """A simple bot that actually works"""
    
    def __init__(self):
        self.exchanges = {}
        self.symbols = ['BTC/USDT', 'ETH/USDT']
        
    def setup_exchanges(self):
        """Setup available exchanges"""
        
        # Try Binance
        if os.getenv('BINANCE_API_KEY'):
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET'),
                'enableRateLimit': True
            })
            print("✅ Binance configured")
            
        # Try Coinbase
        if os.getenv('COINBASE_API_KEY'):
            self.exchanges['coinbase'] = ccxt.coinbase({
                'apiKey': os.getenv('COINBASE_API_KEY'),
                'secret': os.getenv('COINBASE_SECRET'),
                'password': os.getenv('COINBASE_PASSPHRASE'),
                'enableRateLimit': True
            })
            print("✅ Coinbase configured")
            
        if not self.exchanges:
            print("\n⚠️  No exchanges configured!")
            print("Add API keys to .env file:")
            print("BINANCE_API_KEY=your_key")
            print("BINANCE_SECRET=your_secret")
            return False
            
        return True
        
    def find_opportunities(self):
        """Find arbitrage opportunities"""
        
        for symbol in self.symbols:
            prices = {}
            
            # Get prices from each exchange
            for name, exchange in self.exchanges.items():
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    prices[name] = ticker['last']
                    print(f"{name}: {symbol} = ${ticker['last']:.2f}")
                except Exception as e:
                    pass
                    
            # Calculate arbitrage
            if len(prices) >= 2:
                min_exchange = min(prices, key=prices.get)
                max_exchange = max(prices, key=prices.get)
                min_price = prices[min_exchange]
                max_price = prices[max_exchange]
                
                profit = (max_price - min_price) / min_price * 100
                
                if profit > 0.1:
                    print(f"\n🎯 OPPORTUNITY FOUND!")
                    print(f"   Symbol: {symbol}")
                    print(f"   Buy on {min_exchange}: ${min_price:.2f}")
                    print(f"   Sell on {max_exchange}: ${max_price:.2f}")
                    print(f"   Profit: {profit:.3f}%")
                    print(f"   Time: {datetime.now()}")
                    
    def run(self):
        """Main loop"""
        
        print("=" * 50)
        print("SIMPLE ARBITRAGE BOT")
        print("=" * 50)
        
        if not self.setup_exchanges():
            return
            
        print(f"\nMonitoring {len(self.symbols)} symbols on {len(self.exchanges)} exchanges")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.find_opportunities()
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n\nBot stopped")

if __name__ == "__main__":
    bot = SimpleArbitrageBot()
    bot.run()
