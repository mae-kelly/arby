#!/usr/bin/env python3
"""
Unified Orchestrator - Works on both M1 Mac and Google Colab A100
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

# Detect platform
IS_COLAB = 'COLAB_GPU' in os.environ or os.path.exists('/content/sample_data')
IS_M1 = platform.processor() == 'arm' and platform.system() == 'Darwin'
GPU_TYPE = 'A100' if IS_COLAB else ('M1' if IS_M1 else 'CPU')

print(f"🖥️  Platform: {GPU_TYPE}")
print(f"📍 Environment: {'Google Colab' if IS_COLAB else 'Local'}")

# Platform-specific GPU imports
if IS_COLAB:
    try:
        import cupy as cp
        import numba
        from numba import cuda
        GPU_AVAILABLE = True
        print("✅ CUDA GPU acceleration available (A100)")
    except ImportError:
        print("Installing CUDA dependencies...")
        os.system("pip install -q cupy-cuda11x numba")
        import cupy as cp
        from numba import cuda
        GPU_AVAILABLE = True
elif IS_M1:
    try:
        import tensorflow as tf
        # Configure TensorFlow for M1
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        GPU_AVAILABLE = bool(gpus)
        print(f"✅ Metal GPU acceleration available: {gpus}")
    except ImportError:
        print("Installing M1 dependencies...")
        os.system("pip install -q tensorflow-macos tensorflow-metal")
        import tensorflow as tf
        GPU_AVAILABLE = True
else:
    GPU_AVAILABLE = False
    print("⚠️  No GPU acceleration available")

# Common imports
try:
    import ccxt.async_support as ccxt
except ImportError:
    print("Installing ccxt...")
    os.system("pip install -q ccxt")
    import ccxt.async_support as ccxt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Installing python-dotenv...")
    os.system("pip install -q python-dotenv")
    from dotenv import load_dotenv
    load_dotenv()

try:
    import aiohttp
    import websockets
except ImportError:
    print("Installing async dependencies...")
    os.system("pip install -q aiohttp websockets")
    import aiohttp
    import websockets

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/arbitrage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ArbitrageOpportunity:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_pct: float
    volume: float
    confidence: float
    chain: str = "ethereum"
    
class GPUAccelerator:
    """Unified GPU accelerator for both M1 and A100"""
    
    def __init__(self):
        self.gpu_type = GPU_TYPE
        self.setup_gpu()
        
    def setup_gpu(self):
        """Setup GPU based on platform"""
        if self.gpu_type == 'A100':
            self.setup_cuda()
        elif self.gpu_type == 'M1':
            self.setup_metal()
        else:
            logger.warning("No GPU acceleration available")
            
    def setup_cuda(self):
        """Setup CUDA for A100"""
        if IS_COLAB:
            # Verify CUDA
            self.device = cuda.get_current_device()
            logger.info(f"CUDA Device: {self.device.name}")
            logger.info(f"Compute Capability: {self.device.compute_capability}")
            
    def setup_metal(self):
        """Setup Metal for M1"""
        if IS_M1:
            logger.info("Metal Performance Shaders configured for M1")
            
    def find_arbitrage_gpu(self, prices: np.ndarray) -> List[Dict]:
        """GPU-accelerated arbitrage finding"""
        
        if self.gpu_type == 'A100':
            return self.find_arbitrage_cuda(prices)
        elif self.gpu_type == 'M1':
            return self.find_arbitrage_metal(prices)
        else:
            return self.find_arbitrage_cpu(prices)
            
    def find_arbitrage_cuda(self, prices: np.ndarray) -> List[Dict]:
        """CUDA implementation for A100"""
        if not IS_COLAB:
            return []
            
        # Transfer to GPU
        gpu_prices = cp.asarray(prices)
        
        # Calculate price differences
        n = len(prices)
        opportunities = []
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = gpu_prices[j, 1] - gpu_prices[i, 0]  # sell - buy
                if diff > 0:
                    profit_pct = (diff / gpu_prices[i, 0]) * 100
                    if profit_pct > 0.1:  # 0.1% threshold
                        opportunities.append({
                            'indices': (int(i), int(j)),
                            'profit_pct': float(profit_pct)
                        })
                        
        return opportunities
        
    def find_arbitrage_metal(self, prices: np.ndarray) -> List[Dict]:
        """Metal implementation for M1"""
        if not IS_M1:
            return []
            
        # Use TensorFlow for M1 GPU acceleration
        with tf.device('/GPU:0'):
            prices_tf = tf.constant(prices, dtype=tf.float32)
            
            # Vectorized price comparison
            buy_prices = tf.expand_dims(prices_tf[:, 0], 1)
            sell_prices = tf.expand_dims(prices_tf[:, 1], 0)
            
            # Calculate profit matrix
            profit_matrix = (sell_prices - buy_prices) / buy_prices * 100
            
            # Find profitable pairs
            profitable = tf.where(profit_matrix > 0.1)
            
            opportunities = []
            for idx in profitable.numpy():
                if idx[0] > idx[1]:  # Avoid duplicates
                    opportunities.append({
                        'indices': (int(idx[1]), int(idx[0])),
                        'profit_pct': float(profit_matrix[idx[0], idx[1]])
                    })
                    
        return opportunities
        
    def find_arbitrage_cpu(self, prices: np.ndarray) -> List[Dict]:
        """CPU fallback implementation"""
        opportunities = []
        n = len(prices)
        
        for i in range(n):
            for j in range(i + 1, n):
                buy_price = prices[i, 0]
                sell_price = prices[j, 1]
                
                if sell_price > buy_price:
                    profit_pct = ((sell_price - buy_price) / buy_price) * 100
                    if profit_pct > 0.1:
                        opportunities.append({
                            'indices': (i, j),
                            'profit_pct': profit_pct
                        })
                        
        return opportunities

class UnifiedExchangeManager:
    """Exchange manager that works on both platforms"""
    
    def __init__(self):
        self.exchanges = {}
        self.markets = {}
        self.tickers = defaultdict(dict)
        self.gpu = GPUAccelerator()
        
    async def initialize(self):
        """Initialize exchanges based on environment"""
        
        # List of exchanges to try
        exchange_configs = {
            'binance': {
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET')
            },
            'coinbase': {
                'apiKey': os.getenv('COINBASE_API_KEY'),
                'secret': os.getenv('COINBASE_SECRET'),
                'password': os.getenv('COINBASE_PASSPHRASE')
            },
            'kraken': {
                'apiKey': os.getenv('KRAKEN_API_KEY'),
                'secret': os.getenv('KRAKEN_SECRET')
            },
            'bybit': {
                'apiKey': os.getenv('BYBIT_API_KEY'),
                'secret': os.getenv('BYBIT_SECRET')
            }
        }
        
        # Initialize exchanges with available credentials
        for name, config in exchange_configs.items():
            if config.get('apiKey') and config.get('secret'):
                try:
                    exchange_class = getattr(ccxt, name)
                    self.exchanges[name] = exchange_class({
                        **config,
                        'enableRateLimit': True,
                        'options': {'defaultType': 'spot'}
                    })
                    
                    # Load markets
                    await self.exchanges[name].load_markets()
                    self.markets[name] = self.exchanges[name].markets
                    
                    logger.info(f"✅ {name}: {len(self.markets[name])} markets loaded")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to initialize {name}: {e}")
                    
        if not self.exchanges:
            logger.warning("⚠️  No exchanges configured! Add API keys to .env file")
            return False
            
        return True
        
    async def fetch_all_tickers(self):
        """Fetch tickers from all exchanges"""
        
        tasks = []
        for name, exchange in self.exchanges.items():
            tasks.append(self.fetch_exchange_tickers(name, exchange))
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def fetch_exchange_tickers(self, name: str, exchange):
        """Fetch tickers for one exchange"""
        try:
            tickers = await exchange.fetch_tickers()
            self.tickers[name] = tickers
            logger.debug(f"Updated {len(tickers)} tickers from {name}")
        except Exception as e:
            logger.error(f"Error fetching tickers from {name}: {e}")
            
    def find_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities using GPU acceleration"""
        
        opportunities = []
        
        # Get common symbols across exchanges
        common_symbols = self.get_common_symbols()
        
        for symbol in common_symbols:
            prices = []
            exchanges_for_symbol = []
            
            # Collect prices from all exchanges
            for exchange_name, tickers in self.tickers.items():
                if symbol in tickers and tickers[symbol].get('bid') and tickers[symbol].get('ask'):
                    prices.append([
                        tickers[symbol]['bid'],
                        tickers[symbol]['ask'],
                        tickers[symbol].get('quoteVolume', 0)
                    ])
                    exchanges_for_symbol.append(exchange_name)
                    
            if len(prices) >= 2:
                # Convert to numpy array for GPU processing
                price_array = np.array(prices, dtype=np.float32)
                
                # Find opportunities using GPU
                gpu_opportunities = self.gpu.find_arbitrage_gpu(price_array)
                
                # Convert GPU results to ArbitrageOpportunity objects
                for opp in gpu_opportunities:
                    i, j = opp['indices']
                    
                    opportunity = ArbitrageOpportunity(
                        symbol=symbol,
                        buy_exchange=exchanges_for_symbol[i],
                        sell_exchange=exchanges_for_symbol[j],
                        buy_price=float(prices[i][0]),
                        sell_price=float(prices[j][1]),
                        profit_pct=opp['profit_pct'],
                        volume=min(prices[i][2], prices[j][2]),
                        confidence=0.8
                    )
                    
                    opportunities.append(opportunity)
                    
        # Sort by profit
        opportunities.sort(key=lambda x: x.profit_pct, reverse=True)
        
        return opportunities[:100]  # Top 100 opportunities
        
    def get_common_symbols(self) -> set:
        """Get symbols that exist on multiple exchanges"""
        
        all_symbols = defaultdict(int)
        
        for exchange_name, tickers in self.tickers.items():
            for symbol in tickers.keys():
                all_symbols[symbol] += 1
                
        # Return symbols that exist on at least 2 exchanges
        return {symbol for symbol, count in all_symbols.items() if count >= 2}

class UnifiedArbitrageBot:
    """Main bot that works on both M1 and A100"""
    
    def __init__(self):
        self.exchange_manager = UnifiedExchangeManager()
        self.running = True
        self.stats = {
            'opportunities_found': 0,
            'total_profit_potential': 0.0,
            'start_time': time.time()
        }
        
    async def initialize(self):
        """Initialize the bot"""
        
        logger.info("=" * 60)
        logger.info("UNIFIED ARBITRAGE BOT")
        logger.info(f"Platform: {GPU_TYPE}")
        logger.info(f"GPU Acceleration: {GPU_AVAILABLE}")
        logger.info("=" * 60)
        
        # Initialize exchanges
        success = await self.exchange_manager.initialize()
        
        if not success:
            logger.error("Failed to initialize exchanges")
            return False
            
        return True
        
    async def run(self):
        """Main execution loop"""
        
        if not await self.initialize():
            return
            
        logger.info("Bot started. Press Ctrl+C to stop.")
        
        try:
            while self.running:
                # Fetch latest prices
                await self.exchange_manager.fetch_all_tickers()
                
                # Find opportunities
                opportunities = self.exchange_manager.find_arbitrage_opportunities()
                
                # Log opportunities
                for opp in opportunities[:10]:  # Top 10
                    self.stats['opportunities_found'] += 1
                    self.stats['total_profit_potential'] += opp.profit_pct
                    
                    logger.info(
                        f"🎯 {opp.symbol}: Buy {opp.buy_exchange} @ ${opp.buy_price:.2f}, "
                        f"Sell {opp.sell_exchange} @ ${opp.sell_price:.2f}, "
                        f"Profit: {opp.profit_pct:.3f}%"
                    )
                    
                # Show stats
                elapsed = time.time() - self.stats['start_time']
                logger.info(
                    f"📊 Stats: {self.stats['opportunities_found']} opportunities, "
                    f"Potential: {self.stats['total_profit_potential']:.2f}%, "
                    f"Runtime: {elapsed:.0f}s"
                )
                
                # Wait before next iteration
                await asyncio.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.running = False
            
        # Cleanup
        for exchange in self.exchange_manager.exchanges.values():
            await exchange.close()
            
        logger.info("Bot stopped.")

async def main():
    """Entry point"""
    bot = UnifiedArbitrageBot()
    await bot.run()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run the bot
    asyncio.run(main())
