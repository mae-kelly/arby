#!/usr/bin/env python3
"""
Debug Arbitrage Bot - Shows exactly what's happening
"""

import os
import sys
import time
import traceback
from datetime import datetime

# Install dependencies if needed
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
    print("✅ .env loaded")
except ImportError:
    print("Installing python-dotenv...")
    os.system(f"{sys.executable} -m pip install -q python-dotenv")
    from dotenv import load_dotenv
    load_dotenv()

class DebugArbitrageBot:
    """A bot that shows exactly what it's doing"""
    
    def __init__(self):
        self.exchanges = {}
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
        self.scan_count = 0
        
    def setup_exchanges(self):
        """Setup exchanges with detailed logging"""
        
        print("\n🔧 Setting up exchanges...")
        
        # Check .env file
        if not os.path.exists('.env'):
            print("❌ No .env file found!")
            print("Create .env file with:")
            print("BINANCE_API_KEY=your_key")
            print("BINANCE_SECRET=your_secret")
            return False
            
        print("✅ .env file found")
        
        # Try Binance
        binance_key = os.getenv('BINANCE_API_KEY')
        binance_secret = os.getenv('BINANCE_SECRET')
        
        if binance_key and binance_secret:
            try:
                print(f"🔑 Binance API key: {binance_key[:8]}...")
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': binance_key,
                    'secret': binance_secret,
                    'enableRateLimit': True,
                    'sandbox': False  # Use real exchange
                })
                print("✅ Binance configured")
            except Exception as e:
                print(f"❌ Binance error: {e}")
        else:
            print("⚠️  Binance API keys not found in .env")
            
        # Try Kraken (no API key needed for public data)
        try:
            self.exchanges['kraken'] = ccxt.kraken({
                'enableRateLimit': True
            })
            print("✅ Kraken configured (public data)")
        except Exception as e:
            print(f"❌ Kraken error: {e}")
            
        # Try KuCoin (no API key needed for public data)
        try:
            self.exchanges['kucoin'] = ccxt.kucoin({
                'enableRateLimit': True
            })
            print("✅ KuCoin configured (public data)")
        except Exception as e:
            print(f"❌ KuCoin error: {e}")
            
        if not self.exchanges:
            print("❌ No exchanges configured!")
            return False
            
        print(f"✅ Total exchanges: {len(self.exchanges)}")
        return True
        
    def test_exchange_connection(self):
        """Test if exchanges are working"""
        
        print("\n🔌 Testing exchange connections...")
        
        for name, exchange in self.exchanges.items():
            try:
                print(f"Testing {name}...")
                
                # Try to fetch a ticker
                ticker = exchange.fetch_ticker('BTC/USDT')
                price = ticker['last']
                
                print(f"✅ {name}: BTC/USDT = ${price:,.2f}")
                
            except Exception as e:
                print(f"❌ {name} failed: {str(e)[:100]}")
                
    def get_prices_with_debug(self):
        """Get prices with detailed logging"""
        
        self.scan_count += 1
        print(f"\n📊 Scan #{self.scan_count} at {datetime.now().strftime('%H:%M:%S')}")
        
        all_prices = {}
        
        for symbol in self.symbols:
            print(f"\n  💰 Fetching {symbol} prices...")
            symbol_prices = {}
            
            for name, exchange in self.exchanges.items():
                try:
                    print(f"    📡 {name}...", end="")
                    
                    # Check if symbol exists
                    if not hasattr(exchange, 'markets') or not exchange.markets:
                        exchange.load_markets()
                        
                    if symbol not in exchange.markets:
                        print(" ❌ Symbol not available")
                        continue
                        
                    ticker = exchange.fetch_ticker(symbol)
                    
                    bid = ticker.get('bid')
                    ask = ticker.get('ask')
                    last = ticker.get('last')
                    
                    if bid and ask:
                        symbol_prices[name] = {
                            'bid': bid,
                            'ask': ask,
                            'last': last,
                            'spread': ask - bid
                        }
                        print(f" ✅ ${last:.2f} (spread: ${ask-bid:.2f})")
                    else:
                        print(" ⚠️  No bid/ask data")
                        
                except Exception as e:
                    print(f" ❌ Error: {str(e)[:50]}")
                    
            if len(symbol_prices) >= 2:
                all_prices[symbol] = symbol_prices
                print(f"    ✅ Got {len(symbol_prices)} prices for {symbol}")
            else:
                print(f"    ⚠️  Only {len(symbol_prices)} prices for {symbol}")
                
        return all_prices
        
    def find_opportunities_with_debug(self, all_prices):
        """Find opportunities with detailed output"""
        
        print(f"\n🔍 Looking for arbitrage opportunities...")
        opportunities_found = 0
        
        for symbol, prices in all_prices.items():
            print(f"\n  🎯 Analyzing {symbol}:")
            
            if len(prices) < 2:
                print(f"    ⚠️  Need at least 2 exchanges, have {len(prices)}")
                continue
                
            # Show all prices
            print("    📋 Current prices:")
            for exchange, data in prices.items():
                print(f"      {exchange}: ${data['last']:.2f} (bid: ${data['bid']:.2f}, ask: ${data['ask']:.2f})")
                
            # Find min/max
            exchanges = list(prices.keys())
            best_opportunities = []
            
            for i, ex1 in enumerate(exchanges):
                for ex2 in exchanges[i+1:]:
                    # Direction 1: Buy on ex1, sell on ex2
                    buy_price = prices[ex1]['ask']
                    sell_price = prices[ex2]['bid']
                    
                    if sell_price > buy_price:
                        profit = sell_price - buy_price
                        profit_pct = (profit / buy_price) * 100
                        
                        best_opportunities.append({
                            'symbol': symbol,
                            'buy_exchange': ex1,
                            'sell_exchange': ex2,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'profit': profit,
                            'profit_pct': profit_pct
                        })
                        
                    # Direction 2: Buy on ex2, sell on ex1
                    buy_price = prices[ex2]['ask']
                    sell_price = prices[ex1]['bid']
                    
                    if sell_price > buy_price:
                        profit = sell_price - buy_price
                        profit_pct = (profit / buy_price) * 100
                        
                        best_opportunities.append({
                            'symbol': symbol,
                            'buy_exchange': ex2,
                            'sell_exchange': ex1,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'profit': profit,
                            'profit_pct': profit_pct
                        })
                        
            # Show opportunities
            if best_opportunities:
                print("    🎉 OPPORTUNITIES FOUND:")
                for opp in sorted(best_opportunities, key=lambda x: x['profit_pct'], reverse=True):
                    opportunities_found += 1
                    print(f"      🔥 Buy {opp['buy_exchange']} @ ${opp['buy_price']:.2f}")
                    print(f"         Sell {opp['sell_exchange']} @ ${opp['sell_price']:.2f}")
                    print(f"         Profit: ${opp['profit']:.2f} ({opp['profit_pct']:.3f}%)")
                    print()
            else:
                print("    😴 No profitable opportunities")
                
        if opportunities_found == 0:
            print("\n💤 No arbitrage opportunities found this scan")
            print("   This is normal - profitable arbitrage is rare!")
            print("   The bot is working correctly.")
        else:
            print(f"\n🎊 Total opportunities found: {opportunities_found}")
            
    def run_debug(self):
        """Run with full debugging"""
        
        print("=" * 60)
        print("🐛 DEBUG ARBITRAGE BOT")
        print("=" * 60)
        
        # Setup
        if not self.setup_exchanges():
            print("\n❌ Setup failed. Please check your configuration.")
            return
            
        # Test connections
        self.test_exchange_connection()
        
        print(f"\n🚀 Starting monitoring loop...")
        print("   Press Ctrl+C to stop")
        
        try:
            while True:
                print("\n" + "="*50)
                
                try:
                    # Get prices
                    all_prices = self.get_prices_with_debug()
                    
                    # Find opportunities
                    if all_prices:
                        self.find_opportunities_with_debug(all_prices)
                    else:
                        print("⚠️  No price data retrieved")
                        
                except Exception as e:
                    print(f"❌ Error in scan: {e}")
                    traceback.print_exc()
                    
                print(f"\n⏳ Waiting 15 seconds before next scan...")
                time.sleep(15)
                
        except KeyboardInterrupt:
            print(f"\n\n👋 Bot stopped after {self.scan_count} scans")

if __name__ == "__main__":
    bot = DebugArbitrageBot()
    bot.run_debug()