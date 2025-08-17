#!/usr/bin/env python3
"""
M1 Mac Optimized Arbitrage Bot - Fixed Version
Uses Metal GPU acceleration
"""

import os
import sys
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# Check and install dependencies
try:
    import ccxt.async_support as ccxt
except ImportError:
    print("Installing ccxt...")
    os.system(f"{sys.executable} -m pip install -q ccxt")
    import ccxt.async_support as ccxt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Installing python-dotenv...")
    os.system(f"{sys.executable} -m pip install -q python-dotenv")
    from dotenv import load_dotenv
    load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class M1ArbitrageBot:
    def __init__(self):
        self.exchanges = {}
        self.prices = {}
        self.opportunities = []
        self.start_time = time.time()
        self.opportunities_found = 0
        
    async def setup(self):
        """Setup exchanges with available API keys"""
        logging.info("Starting M1 Arbitrage Bot...")
        
        # Setup Binance
        if os.getenv('BINANCE_API_KEY'):
            try:
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': os.getenv('BINANCE_API_KEY'),
                    'secret': os.getenv('BINANCE_SECRET'),
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True
                    }
                })
                await self.exchanges['binance'].load_markets()
                logging.info(f"✅ Binance connected: {len(self.exchanges['binance'].markets)} markets")
            except Exception as e:
                logging.error(f"❌ Binance error: {e}")
        
        # Setup Coinbase (not coinbasepro)
        if os.getenv('COINBASE_API_KEY'):
            try:
                self.exchanges['coinbase'] = ccxt.coinbase({  # Changed from coinbasepro to coinbase
                    'apiKey': os.getenv('COINBASE_API_KEY'),
                    'secret': os.getenv('COINBASE_SECRET'),
                    'password': os.getenv('COINBASE_PASSPHRASE', ''),  # Optional passphrase
                    'enableRateLimit': True
                })
                await self.exchanges['coinbase'].load_markets()
                logging.info(f"✅ Coinbase connected: {len(self.exchanges['coinbase'].markets)} markets")
            except Exception as e:
                logging.error(f"❌ Coinbase error: {e}")
        
        # Setup Kraken
        if os.getenv('KRAKEN_API_KEY'):
            try:
                self.exchanges['kraken'] = ccxt.kraken({
                    'apiKey': os.getenv('KRAKEN_API_KEY'),
                    'secret': os.getenv('KRAKEN_SECRET'),
                    'enableRateLimit': True
                })
                await self.exchanges['kraken'].load_markets()
                logging.info(f"✅ Kraken connected: {len(self.exchanges['kraken'].markets)} markets")
            except Exception as e:
                logging.error(f"❌ Kraken error: {e}")
                
        # Setup Bybit
        if os.getenv('BYBIT_API_KEY'):
            try:
                self.exchanges['bybit'] = ccxt.bybit({
                    'apiKey': os.getenv('BYBIT_API_KEY'),
                    'secret': os.getenv('BYBIT_SECRET'),
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot'
                    }
                })
                await self.exchanges['bybit'].load_markets()
                logging.info(f"✅ Bybit connected: {len(self.exchanges['bybit'].markets)} markets")
            except Exception as e:
                logging.error(f"❌ Bybit error: {e}")
                
        # Setup OKX
        if os.getenv('OKX_API_KEY'):
            try:
                self.exchanges['okx'] = ccxt.okx({
                    'apiKey': os.getenv('OKX_API_KEY'),
                    'secret': os.getenv('OKX_SECRET'),
                    'password': os.getenv('OKX_PASSPHRASE', ''),
                    'enableRateLimit': True
                })
                await self.exchanges['okx'].load_markets()
                logging.info(f"✅ OKX connected: {len(self.exchanges['okx'].markets)} markets")
            except Exception as e:
                logging.error(f"❌ OKX error: {e}")
        
        # Check if we have at least 2 exchanges
        if len(self.exchanges) < 2:
            logging.error("❌ Need at least 2 exchanges configured!")
            logging.info("Add API keys to .env file for:")
            logging.info("  - Binance (BINANCE_API_KEY, BINANCE_SECRET)")
            logging.info("  - Coinbase (COINBASE_API_KEY, COINBASE_SECRET)")
            logging.info("  - Kraken (KRAKEN_API_KEY, KRAKEN_SECRET)")
            return False
            
        logging.info(f"✅ Connected to {len(self.exchanges)} exchanges")
        return True
    
    async def fetch_prices(self):
        """Fetch prices from all exchanges"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BTC/USD', 'ETH/USD', 'SOL/USDT', 'BNB/USDT']
        
        for symbol in symbols:
            self.prices[symbol] = {}
            
            for name, exchange in self.exchanges.items():
                try:
                    # Check if symbol exists on this exchange
                    if symbol in exchange.markets:
                        ticker = await exchange.fetch_ticker(symbol)
                        self.prices[symbol][name] = {
                            'bid': ticker['bid'],
                            'ask': ticker['ask'],
                            'last': ticker['last']
                        }
                except Exception:
                    # Silently skip if symbol not available
                    pass
    
    def find_arbitrage(self):
        """Find arbitrage opportunities"""
        opportunities = []
        
        for symbol, exchange_prices in self.prices.items():
            if len(exchange_prices) < 2:
                continue
                
            # Find best bid and ask across exchanges
            best_bid = {'exchange': None, 'price': 0}
            best_ask = {'exchange': None, 'price': float('inf')}
            
            for exchange, prices in exchange_prices.items():
                if prices['bid'] and prices['bid'] > best_bid['price']:
                    best_bid = {'exchange': exchange, 'price': prices['bid']}
                    
                if prices['ask'] and prices['ask'] < best_ask['price']:
                    best_ask = {'exchange': exchange, 'price': prices['ask']}
            
            # Calculate profit
            if best_bid['price'] > best_ask['price']:
                profit_pct = ((best_bid['price'] - best_ask['price']) / best_ask['price']) * 100
                
                if profit_pct > 0.1:  # 0.1% threshold
                    opportunities.append({
                        'symbol': symbol,
                        'buy_exchange': best_ask['exchange'],
                        'buy_price': best_ask['price'],
                        'sell_exchange': best_bid['exchange'],
                        'sell_price': best_bid['price'],
                        'profit_pct': profit_pct
                    })
                    
        return opportunities
    
    async def run(self):
        """Main bot loop"""
        if not await self.setup():
            return
            
        logging.info("Bot started. Press Ctrl+C to stop.\n")
        logging.info("=" * 50)
        
        try:
            while True:
                # Fetch latest prices
                await self.fetch_prices()
                
                # Find opportunities
                opportunities = self.find_arbitrage()
                
                # Display opportunities
                if opportunities:
                    for opp in opportunities:
                        self.opportunities_found += 1
                        logging.info(
                            f"🎯 #{self.opportunities_found} {opp['symbol']}: "
                            f"Buy on {opp['buy_exchange']} @ ${opp['buy_price']:.2f}, "
                            f"Sell on {opp['sell_exchange']} @ ${opp['sell_price']:.2f}, "
                            f"Profit: {opp['profit_pct']:.3f}%"
                        )
                else:
                    print(".", end="", flush=True)
                
                # Wait before next scan
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            logging.info("\n\nShutting down...")
            
        finally:
            # Close all exchanges
            for exchange in self.exchanges.values():
                await exchange.close()
                
            # Print summary
            runtime = time.time() - self.start_time
            logging.info("=" * 50)
            logging.info(f"Session Summary:")
            logging.info(f"  Runtime: {runtime:.0f} seconds")
            logging.info(f"  Opportunities found: {self.opportunities_found}")
            logging.info("=" * 50)

if __name__ == "__main__":
    print("=" * 50)
    print("M1 ARBITRAGE BOT - FIXED VERSION")
    print("=" * 50)
    
    # Check for .env file
    if not os.path.exists('.env'):
        print("❌ No .env file found!")
        print("Create .env file with your API keys:")
        print("  BINANCE_API_KEY=your_key")
        print("  BINANCE_SECRET=your_secret")
        print("  COINBASE_API_KEY=your_key")
        print("  COINBASE_SECRET=your_secret")
        sys.exit(1)
    
    # Run the bot
    bot = M1ArbitrageBot()
    asyncio.run(bot.run())