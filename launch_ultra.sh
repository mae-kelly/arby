#!/bin/bash
# Launch Ultra-Profitable Arbitrage System

echo "🚀 LAUNCHING ULTRA-PROFITABLE ARBITRAGE SYSTEM"
echo "==============================================="

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Creating .env template..."
    cat > .env << 'ENV_TEMPLATE'
# Exchange API Keys
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET=your_binance_secret

COINBASE_API_KEY=your_coinbase_api_key
COINBASE_SECRET=your_coinbase_secret

KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_secret

# Performance Settings
MIN_PROFIT_PCT=0.1
MAX_SLIPPAGE=0.2
MAX_GAS_PRICE_GWEI=500
ENV_TEMPLATE
    
    echo "📝 Please edit .env file with your API keys"
    echo "Then run: ./launch_ultra.sh"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Set Python path
export PYTHONPATH="$PYTHONPATH:./src/python"

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import ccxt, numpy, asyncio; print('✅ All dependencies available')" || {
    echo "Installing missing dependencies..."
    pip install -q ccxt numpy python-dotenv aiohttp
}

# Launch the system
echo "🎯 Launching ultra orchestrator..."
python3 src/python/ultra_orchestrator.py
