#!/bin/bash

echo "🚀 CRYPTO ARBITRAGE BOT LAUNCHER"
echo "================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
echo "Installing dependencies..."
pip install -q ccxt python-dotenv aiohttp websockets

# Create logs directory
mkdir -p logs

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  No .env file found. Creating one..."
    cat > .env << 'EOF'
# Exchange API Keys (Optional for demo mode)
KRAKEN_API_KEY=
KRAKEN_SECRET=

OKX_API_KEY=
OKX_SECRET=
OKX_PASSPHRASE=

KUCOIN_API_KEY=
KUCOIN_SECRET=
KUCOIN_PASSPHRASE=
EOF
    echo "✅ Created .env file (API keys optional for demo mode)"
fi

# Menu
echo ""
echo "SELECT MODE:"
echo "------------"
echo "1) Demo Mode (no API keys needed)"
echo "2) Real Mode (requires API keys in .env)"
echo "3) Quick Demo (5 seconds)"
echo "4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Starting in DEMO MODE..."
        echo "========================"
        python3 final_bot.py --demo
        ;;
    2)
        echo ""
        echo "Starting in REAL MODE..."
        echo "========================"
        python3 src/orchestrator_m1_fixed.py
        ;;
    3)
        echo ""
        echo "Running quick demo..."
        echo "===================="
        python3 -c "
import asyncio
import random
import time

async def quick_demo():
    print('🎮 QUICK DEMO - Simulated Arbitrage')
    print('=' * 40)
    
    exchanges = ['Kraken', 'OKX', 'KuCoin']
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    for i in range(5):
        symbol = random.choice(symbols)
        ex1, ex2 = random.sample(exchanges, 2)
        price1 = random.uniform(40000, 50000) if 'BTC' in symbol else random.uniform(2000, 3000)
        price2 = price1 * random.uniform(1.001, 1.005)
        profit = ((price2 - price1) / price1) * 100
        
        print(f'\\n🎯 Opportunity #{i+1}:')
        print(f'   {symbol}')
        print(f'   Buy on {ex1}: \${price1:.2f}')
        print(f'   Sell on {ex2}: \${price2:.2f}')
        print(f'   Profit: {profit:.3f}%')
        
        await asyncio.sleep(1)
    
    print('\\n✅ Demo complete!')

asyncio.run(quick_demo())
"
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac