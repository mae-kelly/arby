#!/bin/bash

echo "🚀 Starting Arbitrage Bot on M1 Mac (Fixed)"
echo "==========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "Python version: $PYTHON_VERSION"

# Set environment for M1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# For M1 Mac threading issues
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✅ Using existing virtual environment"
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Install/upgrade dependencies
echo "Checking dependencies..."
pip install -q --upgrade pip

# Install required packages
pip install -q ccxt python-dotenv aiohttp numpy

# Optional: Install uvloop for better performance
pip install -q uvloop 2>/dev/null || echo "uvloop not available, using default event loop"

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found!"
    echo "Creating template .env file..."
    
    cat > .env << 'ENV_EOF'
# Add your API keys here
BINANCE_API_KEY=
BINANCE_SECRET=

COINBASE_API_KEY=
COINBASE_SECRET=
COINBASE_PASSPHRASE=

KRAKEN_API_KEY=
KRAKEN_SECRET=
ENV_EOF
    
    echo "Please edit .env file and add your API keys"
    exit 1
fi

# Check if at least one API key is configured
if ! grep -q "API_KEY=." .env; then
    echo "⚠️  No API keys configured in .env file!"
    echo "Please add at least 2 exchange API keys to .env"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Save the fixed orchestrator
echo "Creating fixed orchestrator..."
cat > src/orchestrator_m1.py << 'ORCHESTRATOR_EOF'
#!/usr/bin/env python3
"""
M1-Optimized Arbitrage Bot - No threading issues
"""

import os
import sys
import asyncio
import time
import logging
from typing import Dict, List
from dataclasses import dataclass

# Simple setup without complex threading
import ccxt.async_support as ccxt
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Opportunity:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    profit_pct: float

class SimpleBot:
    def __init__(self):
        self.exchanges = {}
        
    async def setup(self):
        """Setup exchanges"""
        
        # Binance
        if os.getenv('BINANCE_API_KEY'):
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET'),
                'enableRateLimit': True
            })
            
        # Coinbase
        if os.getenv('COINBASE_API_KEY'):
            self.exchanges['coinbase'] = ccxt.coinbasepro({
                'apiKey': os.getenv('COINBASE_API_KEY'),
                'secret': os.getenv('COINBASE_SECRET'),
                'password': os.getenv('COINBASE_PASSPHRASE'),
                'enableRateLimit': True
            })
            
        # Kraken
        if os.getenv('KRAKEN_API_KEY'):
            self.exchanges['kraken'] = ccxt.kraken({
                'apiKey': os.getenv('KRAKEN_API_KEY'),
                'secret': os.getenv('KRAKEN_SECRET'),
                'enableRateLimit': True
            })
            
        if not self.exchanges:
            logger.error("No exchanges configured!")
            return False
            
        # Load markets
        for name, exchange in self.exchanges.items():
            try:
                await exchange.load_markets()
                logger.info(f"✅ {name}: {len(exchange.symbols)} markets")
            except Exception as e:
                logger.error(f"❌ {name}: {e}")
                
        return True
        
    async def find_opportunities(self):
        """Find arbitrage opportunities"""
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        for symbol in symbols:
            prices = {}
            
            # Get prices
            for name, exchange in self.exchanges.items():
                try:
                    if symbol in exchange.symbols:
                        ticker = await exchange.fetch_ticker(symbol)
                        prices[name] = ticker
                except:
                    pass
                    
            # Find arbitrage
            if len(prices) >= 2:
                exchanges = list(prices.keys())
                for i in range(len(exchanges)):
                    for j in range(i+1, len(exchanges)):
                        ex1, ex2 = exchanges[i], exchanges[j]
                        
                        if prices[ex1]['bid'] and prices[ex2]['ask']:
                            profit = (prices[ex1]['bid'] - prices[ex2]['ask']) / prices[ex2]['ask'] * 100
                            if profit > 0.1:
                                logger.info(f"🎯 {symbol}: Buy {ex2} @ ${prices[ex2]['ask']:.2f}, Sell {ex1} @ ${prices[ex1]['bid']:.2f}, Profit: {profit:.3f}%")
                                
                        if prices[ex2]['bid'] and prices[ex1]['ask']:
                            profit = (prices[ex2]['bid'] - prices[ex1]['ask']) / prices[ex1]['ask'] * 100
                            if profit > 0.1:
                                logger.info(f"🎯 {symbol}: Buy {ex1} @ ${prices[ex1]['ask']:.2f}, Sell {ex2} @ ${prices[ex2]['bid']:.2f}, Profit: {profit:.3f}%")
    
    async def run(self):
        """Main loop"""
        
        logger.info("Starting M1 Arbitrage Bot...")
        
        if not await self.setup():
            return
            
        logger.info("Bot running. Press Ctrl+C to stop.\n")
        
        try:
            while True:
                await self.find_opportunities()
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            logger.info("\nStopping...")
            
        # Cleanup
        for exchange in self.exchanges.values():
            await exchange.close()

if __name__ == "__main__":
    bot = SimpleBot()
    asyncio.run(bot.run())
ORCHESTRATOR_EOF

# Run the bot
echo ""
echo "Starting bot..."
echo "--------------"
python3 src/orchestrator_m1.py

# Deactivate virtual environment on exit
deactivate 2>/dev/null || true