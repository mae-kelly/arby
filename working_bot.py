#!/usr/bin/env python3
"""
WORKING Arbitrage Bot - Standalone Version
No external imports needed, shows real data
Fixed for location restrictions and API issues
"""

import json
import time
import urllib.request
import urllib.parse
from datetime import datetime

class WorkingArbitrageBot:
    def __init__(self):
        # Using more accessible APIs
        self.exchanges = {
            'coingecko': 'https://api.coingecko.com/api/v3/simple/price',
            'coinbase': 'https://api.coinbase.com/v2/exchange-rates',
            'kraken': 'https://api.kraken.com/0/public/Ticker',
            'kucoin': 'https://api.kucoin.com/api/v1/market/orderbook/level1'
        }
        self.symbols = ['bitcoin', 'ethereum', 'solana', 'binancecoin']
        self.opportunities_found = 0
        
    def fetch_coingecko_prices(self):
        """Fetch prices from CoinGecko (most reliable)"""
        try:
            symbols_str = ','.join(self.symbols)
            url = f"{self.exchanges['coingecko']}?ids={symbols_str}&vs_currencies=usd"
            
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            
            prices = {}
            symbol_map = {
                'bitcoin': 'BTCUSD',
                'ethereum': 'ETHUSD', 
                'solana': 'SOLUSD',
                'binancecoin': 'BNBUSD'
            }
            
            for coin_id, price_data in data.items():
                if coin_id in symbol_map:
                    prices[symbol_map[coin_id]] = price_data['usd']
            
            return prices
        except Exception as e:
            print(f"CoinGecko error: {e}")
            return {}
    
    def fetch_coinbase_prices(self):
        """Fetch prices from Coinbase"""
        try:
            url = "https://api.coinbase.com/v2/exchange-rates?currency=USD"
            
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            
            prices = {}
            if 'data' in data and 'rates' in data['data']:
                rates = data['data']['rates']
                
                # Convert from USD rates to token prices
                symbol_map = {
                    'BTC': 'BTCUSD',
                    'ETH': 'ETHUSD',
                    'SOL': 'SOLUSD'
                }
                
                for coin, symbol in symbol_map.items():
                    if coin in rates and float(rates[coin]) > 0:
                        prices[symbol] = 1.0 / float(rates[coin])
            
            return prices
        except Exception as e:
            print(f"Coinbase error: {e}")
            return {}
    
    def fetch_kucoin_prices(self):
        """Fetch prices from KuCoin"""
        try:
            prices = {}
            symbols = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
            
            for symbol in symbols:
                try:
                    url = f"{self.exchanges['kucoin']}?symbol={symbol}"
                    
                    with urllib.request.urlopen(url) as response:
                        data = json.loads(response.read().decode())
                    
                    if 'data' in data and 'price' in data['data']:
                        clean_symbol = symbol.replace('-', '').replace('USDT', 'USD')
                        prices[clean_symbol] = float(data['data']['price'])
                    
                    time.sleep(0.1)  # Rate limiting
                except:
                    continue
            
            return prices
        except Exception as e:
            print(f"KuCoin error: {e}")
            return {}
    
    def fetch_kraken_prices(self):
        """Fetch prices from Kraken"""
        try:
            symbols = 'XBTUSD,ETHUSD,SOLUSD'
            url = f"{self.exchanges['kraken']}?pair={symbols}"
            
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            
            prices = {}
            if 'result' in data:
                symbol_map = {
                    'XXBTZUSD': 'BTCUSD',
                    'XETHZUSD': 'ETHUSD', 
                    'SOLUSD': 'SOLUSD'
                }
                
                for kraken_symbol, clean_symbol in symbol_map.items():
                    if kraken_symbol in data['result']:
                        price_data = data['result'][kraken_symbol]
                        prices[clean_symbol] = float(price_data['c'][0])  # Last price
            
            return prices
        except Exception as e:
            print(f"Kraken error: {e}")
            return {}
    
    def simulate_exchange_data(self):
        """Generate simulated exchange data for demonstration"""
        import random
        
        base_prices = {
            'BTCUSD': 43000 + random.uniform(-500, 500),
            'ETHUSD': 2500 + random.uniform(-50, 50),
            'SOLUSD': 95 + random.uniform(-5, 5),
            'BNBUSD': 310 + random.uniform(-10, 10)
        }
        
        # Create price variations for different "exchanges"
        exchanges_data = {}
        
        for i, exchange_name in enumerate(['Exchange_A', 'Exchange_B', 'Exchange_C']):
            prices = {}
            for symbol, base_price in base_prices.items():
                # Add random variation (-0.5% to +0.5%)
                variation = random.uniform(-0.005, 0.005)
                prices[symbol] = base_price * (1 + variation)
            exchanges_data[exchange_name] = prices
        
        return exchanges_data
    
    def find_arbitrage_opportunities(self, all_prices):
        """Find arbitrage opportunities between exchanges"""
        opportunities = []
        
        # Get all available symbols across exchanges
        all_symbols = set()
        for exchange_prices in all_prices.values():
            all_symbols.update(exchange_prices.keys())
        
        for symbol in all_symbols:
            # Collect prices for this symbol from all exchanges
            symbol_prices = {}
            for exchange, prices in all_prices.items():
                if symbol in prices and prices[symbol] > 0:
                    symbol_prices[exchange] = prices[symbol]
            
            # Need at least 2 exchanges for arbitrage
            if len(symbol_prices) < 2:
                continue
            
            # Find min and max prices
            min_exchange = min(symbol_prices, key=symbol_prices.get)
            max_exchange = max(symbol_prices, key=symbol_prices.get)
            min_price = symbol_prices[min_exchange]
            max_price = symbol_prices[max_exchange]
            
            # Calculate profit percentage
            if min_price > 0:
                profit_pct = ((max_price - min_price) / min_price) * 100
                
                # Show opportunities > 0.05% (lowered threshold)
                if profit_pct > 0.05:
                    opportunities.append({
                        'symbol': symbol,
                        'buy_exchange': min_exchange,
                        'sell_exchange': max_exchange,
                        'buy_price': min_price,
                        'sell_price': max_price,
                        'profit_pct': profit_pct,
                        'profit_usd': (max_price - min_price) * 1000  # For 1000 units
                    })
        
        return sorted(opportunities, key=lambda x: x['profit_pct'], reverse=True)
    
    def display_prices(self, all_prices):
        """Display current prices from all exchanges"""
        print(f"\n📊 CURRENT PRICES ({datetime.now().strftime('%H:%M:%S')})")
        print("=" * 70)
        
        # Get all symbols
        all_symbols = set()
        for exchange_prices in all_prices.values():
            all_symbols.update(exchange_prices.keys())
        
        for symbol in sorted(all_symbols):
            print(f"\n{symbol}:")
            for exchange, prices in all_prices.items():
                if symbol in prices:
                    print(f"  {exchange.ljust(12)}: ${prices[symbol]:,.2f}")
                else:
                    print(f"  {exchange.ljust(12)}: N/A")
    
    def display_opportunities(self, opportunities):
        """Display arbitrage opportunities"""
        if opportunities:
            print(f"\n🎯 ARBITRAGE OPPORTUNITIES FOUND!")
            print("=" * 60)
            
            for i, opp in enumerate(opportunities, 1):
                self.opportunities_found += 1
                print(f"\n{i}. {opp['symbol']} Arbitrage:")
                print(f"   Buy on {opp['buy_exchange'].capitalize()}: ${opp['buy_price']:,.2f}")
                print(f"   Sell on {opp['sell_exchange'].capitalize()}: ${opp['sell_price']:,.2f}")
                print(f"   Profit: {opp['profit_pct']:.3f}% (${opp['profit_usd']:,.2f} on 1000 units)")
                print(f"   Timestamp: {datetime.now()}")
        else:
            print(f"\n😴 No arbitrage opportunities found (this is normal)")
            print("   Profitable arbitrage is rare in efficient markets")
    
    def run(self):
        """Main bot loop"""
        print("🚀 WORKING ARBITRAGE BOT v2.0")
        print("=" * 70)
        print("This bot fetches REAL prices and shows actual arbitrage opportunities")
        print("Fixed for location restrictions and API issues")
        print("Press Ctrl+C to stop\n")
        
        scan_count = 0
        
        try:
            while True:
                scan_count += 1
                print(f"\n🔄 Scan #{scan_count}")
                
                # Try to fetch real prices first
                all_prices = {
                    'coingecko': self.fetch_coingecko_prices(),
                    'coinbase': self.fetch_coinbase_prices(),
                    'kraken': self.fetch_kraken_prices(),
                    'kucoin': self.fetch_kucoin_prices()
                }
                
                # Check if we got any real data
                has_real_data = any(len(prices) > 0 for prices in all_prices.values())
                
                if not has_real_data:
                    print("⚠️  Real APIs not accessible, using simulated data for demonstration")
                    all_prices = self.simulate_exchange_data()
                
                # Display current prices
                self.display_prices(all_prices)
                
                # Find and display opportunities
                opportunities = self.find_arbitrage_opportunities(all_prices)
                self.display_opportunities(opportunities)
                
                # Summary
                print(f"\n📈 SUMMARY:")
                print(f"   Total scans: {scan_count}")
                print(f"   Opportunities found: {self.opportunities_found}")
                if scan_count > 0:
                    print(f"   Success rate: {(self.opportunities_found/scan_count)*100:.1f}%")
                
                if has_real_data:
                    print(f"   Data source: Real exchange APIs ✅")
                else:
                    print(f"   Data source: Simulated (APIs blocked) ⚠️")
                
                # Wait before next scan
                print(f"\n⏳ Waiting 15 seconds before next scan...")
                time.sleep(15)
                
        except KeyboardInterrupt:
            print(f"\n\n👋 Bot stopped after {scan_count} scans")
            print(f"Total opportunities found: {self.opportunities_found}")
            print("Thanks for using the Working Arbitrage Bot!")

def main():
    """Entry point"""
    bot = WorkingArbitrageBot()
    bot.run()

if __name__ == "__main__":
    main()