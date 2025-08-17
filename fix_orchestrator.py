#!/usr/bin/env python3
"""
Fixed Orchestrator for M1 Mac - Handles threading properly
"""

import os
import sys
import platform
import asyncio
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging
from datetime import datetime

# Set up multiprocessing for M1
if platform.system() == 'Darwin':
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)

# Detect platform
IS_M1 = platform.processor() == 'arm' and platform.system() == 'Darwin'
IS_COLAB = 'COLAB_GPU' in os.environ or os.path.exists('/content/sample_data')
GPU_TYPE = 'A100' if IS_COLAB else ('M1' if IS_M1 else 'CPU')

print(f"🖥️  Platform: {GPU_TYPE}")
print(f"📍 Environment: {'Google Colab' if IS_COLAB else 'Local'}")

# Platform-specific imports with proper error handling
GPU_AVAILABLE = False

if IS_M1:
    try:
        # For M1, we'll use numpy with accelerate framework
        os.environ['ACCELERATE_USE_METAL'] = '1'
        GPU_AVAILABLE = True
        print("✅ M1 acceleration enabled via Accelerate framework")
    except Exception as e:
        print(f"⚠️  M1 acceleration setup warning: {e}")
        GPU_AVAILABLE = False

# Import exchange library
try:
    import ccxt.async_support as ccxt
except ImportError:
    print("Installing ccxt...")
    os.system(f"{sys.executable} -m pip install -q ccxt")
    import ccxt.async_support as ccxt

# Import other dependencies
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Installing python-dotenv...")
    os.system(f"{sys.executable} -m pip install -q python-dotenv")
    from dotenv import load_dotenv
    load_dotenv()

try:
    import aiohttp
except ImportError:
    print("Installing aiohttp...")
    os.system(f"{sys.executable} -m pip install -q aiohttp")
    import aiohttp

# Setup logging with proper file handling
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/arbitrage.log', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_pct: float
    volume: float
    timestamp: float

class SimpleArbitrageBot:
    """Simplified bot that works on M1 without threading issues"""
    
    def __init__(self):
        self.exchanges = {}
        self.running = True
        self.opportunities_found = 0
        self.total_profit = 0.0
        self.start_time = time.time()
        
    async def initialize_exchanges(self):
        """Initialize exchanges from environment"""
        
        exchange_configs = {
            'binance': {
                'name': 'binance',
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET')
            },
            'coinbase': {
                'name': 'coinbasepro',  # Use coinbasepro for ccxt
                'apiKey': os.getenv('COINBASE_API_KEY'),
                'secret': os.getenv('COINBASE_SECRET'),
                'password': os.getenv('COINBASE_PASSPHRASE')
            },
            'kraken': {
                'name': 'kraken',
                'apiKey': os.getenv('KRAKEN_API_KEY'),
                'secret': os.getenv('KRAKEN_SECRET')
            }
        }
        
        for key, config in exchange_configs.items():
            if config.get('apiKey') and config.get('secret'):
                try:
                    exchange_class = getattr(ccxt, config['name'])
                    exchange_config = {
                        'apiKey': config['apiKey'],
                        'secret': config['secret'],
                        'enableRateLimit': True,
                        'options': {'defaultType': 'spot'}
                    }
                    
                    if 'password' in config and config['password']:
                        exchange_config['password'] = config['password']
                    
                    self.exchanges[key] = exchange_class(exchange_config)
                    
                    # Load markets
                    await self.exchanges[key].load_markets()
                    logger.info(f"✅ {key}: {len(self.exchanges[key].symbols)} markets loaded")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to initialize {key}: {e}")
        
        if not self.exchanges:
            logger.warning("⚠️  No exchanges configured! Please add API keys to .env file")
            logger.info("Example .env content:")
            logger.info("BINANCE_API_KEY=your_api_key_here")
            logger.info("BINANCE_SECRET=your_secret_here")
            return False
            
        return True
    
    async def fetch_prices(self, symbol: str) -> Dict:
        """Fetch prices for a symbol from all exchanges"""
        prices = {}
        
        for name, exchange in self.exchanges.items():
            try:
                if symbol in exchange.symbols:
                    ticker = await exchange.fetch_ticker(symbol)
                    prices[name] = {
                        'bid': ticker.get('bid', 0),
                        'ask': ticker.get('ask', 0),
                        'last': ticker.get('last', 0),
                        'volume': ticker.get('quoteVolume', 0)
                    }
            except Exception as e:
                logger.debug(f"Error fetching {symbol} from {name}: {e}")
                
        return prices
    
    def find_arbitrage(self, symbol: str, prices: Dict) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities for a symbol"""
        opportunities = []
        
        exchanges = list(prices.keys())
        
        for i, buy_exchange in enumerate(exchanges):
            for sell_exchange in exchanges[i+1:]:
                buy_ask = prices[buy_exchange]['ask']
                sell_bid = prices[sell_exchange]['bid']
                
                if buy_ask > 0 and sell_bid > 0:
                    # Check buy from exchange1, sell to exchange2
                    if sell_bid > buy_ask:
                        profit = sell_bid - buy_ask
                        profit_pct = (profit / buy_ask) * 100
                        
                        if profit_pct > 0.1:  # 0.1% threshold
                            opportunities.append(ArbitrageOpportunity(
                                symbol=symbol,
                                buy_exchange=buy_exchange,
                                sell_exchange=sell_exchange,
                                buy_price=buy_ask,
                                sell_price=sell_bid,
                                profit_pct=profit_pct,
                                volume=min(prices[buy_exchange]['volume'], prices[sell_exchange]['volume']),
                                timestamp=time.time()
                            ))
                    
                    # Check reverse direction
                    buy_ask_rev = prices[sell_exchange]['ask']
                    sell_bid_rev = prices[buy_exchange]['bid']
                    
                    if buy_ask_rev > 0 and sell_bid_rev > 0 and sell_bid_rev > buy_ask_rev:
                        profit = sell_bid_rev - buy_ask_rev
                        profit_pct = (profit / buy_ask_rev) * 100
                        
                        if profit_pct > 0.1:
                            opportunities.append(ArbitrageOpportunity(
                                symbol=symbol,
                                buy_exchange=sell_exchange,
                                sell_exchange=buy_exchange,
                                buy_price=buy_ask_rev,
                                sell_price=sell_bid_rev,
                                profit_pct=profit_pct,
                                volume=min(prices[sell_exchange]['volume'], prices[buy_exchange]['volume']),
                                timestamp=time.time()
                            ))
        
        return opportunities
    
    async def scan_markets(self):
        """Scan markets for arbitrage opportunities"""
        
        # Get common symbols across exchanges
        common_symbols = set()
        for exchange in self.exchanges.values():
            if not common_symbols:
                common_symbols = set(exchange.symbols)
            else:
                common_symbols &= set(exchange.symbols)
        
        # Focus on major pairs
        target_symbols = [s for s in common_symbols if 'USDT' in s or 'USD' in s][:20]
        
        logger.info(f"Scanning {len(target_symbols)} symbols across {len(self.exchanges)} exchanges")
        
        all_opportunities = []
        
        for symbol in target_symbols:
            prices = await self.fetch_prices(symbol)
            
            if len(prices) >= 2:
                opportunities = self.find_arbitrage(symbol, prices)
                all_opportunities.extend(opportunities)
        
        # Sort by profit
        all_opportunities.sort(key=lambda x: x.profit_pct, reverse=True)
        
        # Display top opportunities
        for opp in all_opportunities[:5]:
            self.opportunities_found += 1
            self.total_profit += opp.profit_pct
            
            logger.info(
                f"🎯 {opp.symbol}: Buy {opp.buy_exchange} @ ${opp.buy_price:.2f}, "
                f"Sell {opp.sell_exchange} @ ${opp.sell_price:.2f}, "
                f"Profit: {opp.profit_pct:.3f}%"
            )
    
    async def run(self):
        """Main bot loop"""
        
        logger.info("=" * 60)
        logger.info("ARBITRAGE BOT - M1 OPTIMIZED")
        logger.info(f"Platform: {GPU_TYPE}")
        logger.info("=" * 60)
        
        # Initialize exchanges
        if not await self.initialize_exchanges():
            logger.error("Failed to initialize exchanges. Exiting.")
            return
        
        logger.info("Bot started. Press Ctrl+C to stop.")
        logger.info("")
        
        try:
            while self.running:
                scan_start = time.time()
                
                # Scan for opportunities
                await self.scan_markets()
                
                # Show statistics
                elapsed = time.time() - self.start_time
                scan_time = time.time() - scan_start
                
                logger.info(
                    f"📊 Stats: {self.opportunities_found} opportunities found | "
                    f"Scan time: {scan_time:.1f}s | "
                    f"Runtime: {elapsed:.0f}s"
                )
                logger.info("-" * 60)
                
                # Wait before next scan
                await asyncio.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
            self.running = False
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.running = False
        finally:
            # Cleanup
            logger.info("Closing exchange connections...")
            for exchange in self.exchanges.values():
                try:
                    await exchange.close()
                except:
                    pass
            
            logger.info("Bot stopped.")
            logger.info(f"Total opportunities found: {self.opportunities_found}")
            logger.info(f"Total potential profit: {self.total_profit:.2f}%")

async def main():
    """Entry point"""
    bot = SimpleArbitrageBot()
    await bot.run()

if __name__ == "__main__":
    # For M1 Mac, use asyncio with proper event loop policy
    if IS_M1:
        # Avoid threading issues on M1
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("Using uvloop for better performance")
        except ImportError:
            logger.info("Using default asyncio event loop")
    
    # Run the bot
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)