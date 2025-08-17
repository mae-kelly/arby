#!/usr/bin/env python3
"""
Fixed Arbitrage Bot with Better Logging and Error Handling
"""

import os
import sys
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import ccxt
try:
    import ccxt.async_support as ccxt
except ImportError:
    print("Installing ccxt...")
    os.system("pip install -q ccxt")
    import ccxt.async_support as ccxt

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Installing python-dotenv...")
    os.system("pip install -q python-dotenv")
    from dotenv import load_dotenv
    load_dotenv()

@dataclass
class ArbitrageOpportunity:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_pct: float
    profit_usd: float
    volume: float

class FixedArbitrageBot:
    def __init__(self):
        self.exchanges = {}
        self.tickers = {}
        self.common_symbols = set()
        self.opportunities_found = 0
        self.total_potential_profit = 0.0
        
    async def initialize_exchanges(self):
        """Initialize exchanges with better error handling"""
        
        # Try to connect to exchanges that are available
        exchange_configs = {
            'kraken': {
                'apiKey': os.getenv('KRAKEN_API_KEY'),
                'secret': os.getenv('KRAKEN_SECRET')
            },
            'okx': {
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET'),
                'password': os.getenv('OKX_PASSPHRASE')
            },
            'kucoin': {
                'apiKey': os.getenv('KUCOIN_API_KEY'),
                'secret': os.getenv('KUCOIN_SECRET'),
                'password': os.getenv('KUCOIN_PASSPHRASE')
            },
            'gateio': {
                'apiKey': os.getenv('GATEIO_API_KEY'),
                'secret': os.getenv('GATEIO_SECRET')
            },
            'bitget': {
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_SECRET'),
                'password': os.getenv('BITGET_PASSPHRASE')
            },
            'mexc': {
                'apiKey': os.getenv('MEXC_API_KEY'),
                'secret': os.getenv('MEXC_SECRET')
            }
        }
        
        # For exchanges without API keys, use public data only
        public_exchanges = ['kraken', 'okx', 'kucoin', 'gateio', 'mexc']
        
        for name in public_exchanges:
            try:
                logger.info(f"Connecting to {name}...")
                
                exchange_class = getattr(ccxt, name)
                config = {
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True
                    }
                }
                
                # Add API keys if available
                if name in exchange_configs and exchange_configs[name].get('apiKey'):
                    config.update(exchange_configs[name])
                
                self.exchanges[name] = exchange_class(config)
                
                # Load markets
                markets = await self.exchanges[name].load_markets()
                logger.info(f"✅ {name}: {len(markets)} markets loaded")
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ {name}: {str(e)[:100]}")
                continue
        
        if len(self.exchanges) < 2:
            logger.error("Need at least 2 exchanges for arbitrage!")
            return False
            
        logger.info(f"✅ Connected to {len(self.exchanges)} exchanges")
        return True
    
    async def fetch_common_symbols(self):
        """Find symbols that exist on multiple exchanges"""
        
        logger.info("Finding common trading pairs...")
        
        all_symbols = {}
        for exchange_name, exchange in self.exchanges.items():
            symbols = set(exchange.symbols)
            all_symbols[exchange_name] = symbols
            logger.info(f"  {exchange_name}: {len(symbols)} symbols")
        
        # Find symbols that exist on at least 2 exchanges
        symbol_counts = {}
        for exchange_name, symbols in all_symbols.items():
            for symbol in symbols:
                if symbol not in symbol_counts:
                    symbol_counts[symbol] = []
                symbol_counts[symbol].append(exchange_name)
        
        self.common_symbols = {
            symbol for symbol, exchanges in symbol_counts.items() 
            if len(exchanges) >= 2
        }
        
        logger.info(f"Found {len(self.common_symbols)} common symbols")
        
        # Show some examples
        examples = list(self.common_symbols)[:5]
        logger.info(f"Example symbols: {examples}")
        
        return len(self.common_symbols) > 0
    
    async def fetch_ticker_safe(self, exchange_name: str, exchange, symbol: str):
        """Safely fetch a ticker with error handling"""
        try:
            ticker = await exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            # Silently skip errors for individual tickers
            return None
    
    async def fetch_all_tickers(self):
        """Fetch tickers for common symbols"""
        
        logger.info(f"Fetching prices for {len(self.common_symbols)} symbols...")
        
        # Focus on major pairs to reduce load
        priority_symbols = [
            s for s in self.common_symbols 
            if 'BTC' in s or 'ETH' in s or 'USDT' in s
        ][:50]  # Limit to top 50 pairs
        
        if not priority_symbols:
            priority_symbols = list(self.common_symbols)[:50]
        
        self.tickers = {}
        
        for symbol in priority_symbols:
            symbol_tickers = {}
            
            # Fetch from each exchange
            tasks = []
            exchange_names = []
            
            for exchange_name, exchange in self.exchanges.items():
                if symbol in exchange.symbols:
                    tasks.append(self.fetch_ticker_safe(exchange_name, exchange, symbol))
                    exchange_names.append(exchange_name)
            
            if len(tasks) >= 2:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if result and not isinstance(result, Exception):
                        symbol_tickers[exchange_names[i]] = result
                
                if len(symbol_tickers) >= 2:
                    self.tickers[symbol] = symbol_tickers
        
        logger.info(f"Fetched prices for {len(self.tickers)} symbols with 2+ exchanges")
        
        return len(self.tickers) > 0
    
    def find_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities in fetched tickers"""
        
        opportunities = []
        
        for symbol, exchange_tickers in self.tickers.items():
            # Get all exchanges with this symbol
            exchanges = list(exchange_tickers.keys())
            
            if len(exchanges) < 2:
                continue
            
            # Compare all exchange pairs
            for i in range(len(exchanges)):
                for j in range(i + 1, len(exchanges)):
                    ex1, ex2 = exchanges[i], exchanges[j]
                    ticker1 = exchange_tickers[ex1]
                    ticker2 = exchange_tickers[ex2]
                    
                    # Check both directions
                    opportunities.extend(self.check_arbitrage(
                        symbol, ex1, ticker1, ex2, ticker2
                    ))
        
        # Sort by profit
        opportunities.sort(key=lambda x: x.profit_pct, reverse=True)
        
        return opportunities
    
    def check_arbitrage(self, symbol: str, ex1: str, t1: dict, ex2: str, t2: dict) -> List[ArbitrageOpportunity]:
        """Check for arbitrage between two exchanges"""
        
        opportunities = []
        
        # Need valid bid/ask prices
        if not all([
            t1.get('bid'), t1.get('ask'),
            t2.get('bid'), t2.get('ask')
        ]):
            return opportunities
        
        # Direction 1: Buy on ex1, sell on ex2
        if t1['ask'] and t2['bid'] and t2['bid'] > t1['ask']:
            profit = t2['bid'] - t1['ask']
            profit_pct = (profit / t1['ask']) * 100
            
            # Estimate volume (min of both sides)
            volume = min(
                t1.get('askVolume', 0) or 0,
                t2.get('bidVolume', 0) or 0
            )
            
            if profit_pct > 0.01:  # 0.01% minimum
                opportunities.append(ArbitrageOpportunity(
                    symbol=symbol,
                    buy_exchange=ex1,
                    sell_exchange=ex2,
                    buy_price=t1['ask'],
                    sell_price=t2['bid'],
                    profit_pct=profit_pct,
                    profit_usd=profit * volume if volume else profit,
                    volume=volume
                ))
        
        # Direction 2: Buy on ex2, sell on ex1
        if t2['ask'] and t1['bid'] and t1['bid'] > t2['ask']:
            profit = t1['bid'] - t2['ask']
            profit_pct = (profit / t2['ask']) * 100
            
            volume = min(
                t2.get('askVolume', 0) or 0,
                t1.get('bidVolume', 0) or 0
            )
            
            if profit_pct > 0.01:  # 0.01% minimum
                opportunities.append(ArbitrageOpportunity(
                    symbol=symbol,
                    buy_exchange=ex2,
                    sell_exchange=ex1,
                    buy_price=t2['ask'],
                    sell_price=t1['bid'],
                    profit_pct=profit_pct,
                    profit_usd=profit * volume if volume else profit,
                    volume=volume
                ))
        
        return opportunities
    
    async def run(self):
        """Main bot loop"""
        
        logger.info("=" * 50)
        logger.info("FIXED ARBITRAGE BOT")
        logger.info("=" * 50)
        
        # Initialize exchanges
        if not await self.initialize_exchanges():
            logger.error("Failed to initialize exchanges")
            return
        
        # Find common symbols
        if not await self.fetch_common_symbols():
            logger.error("No common symbols found")
            return
        
        logger.info("Bot started. Press Ctrl+C to stop.")
        logger.info("=" * 50)
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                logger.info(f"\n--- Iteration {iteration} ---")
                
                # Fetch latest prices
                if await self.fetch_all_tickers():
                    
                    # Find opportunities
                    opportunities = self.find_arbitrage_opportunities()
                    
                    if opportunities:
                        logger.info(f"Found {len(opportunities)} opportunities!")
                        
                        # Show top 5
                        for i, opp in enumerate(opportunities[:5], 1):
                            self.opportunities_found += 1
                            self.total_potential_profit += opp.profit_pct
                            
                            logger.info(
                                f"  {i}. {opp.symbol}: "
                                f"Buy {opp.buy_exchange} @ ${opp.buy_price:.4f}, "
                                f"Sell {opp.sell_exchange} @ ${opp.sell_price:.4f}, "
                                f"Profit: {opp.profit_pct:.3f}%"
                            )
                    else:
                        logger.info("No profitable opportunities found this iteration")
                    
                    # Summary stats
                    logger.info(
                        f"📊 Total opportunities found: {self.opportunities_found}, "
                        f"Potential profit: {self.total_potential_profit:.2f}%"
                    )
                else:
                    logger.warning("Failed to fetch tickers")
                
                # Wait before next iteration
                logger.info(f"Waiting 30 seconds before next scan...")
                await asyncio.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            # Cleanup
            for exchange in self.exchanges.values():
                await exchange.close()
            logger.info("Bot stopped.")

async def main():
    bot = FixedArbitrageBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())