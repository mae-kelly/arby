#!/bin/bash

# QUICK START - Get the bot running in 5 minutes
echo "⚡ QUICK START - Arbitrage Bot"
echo "=============================="

# Install dependencies quickly
echo "Installing dependencies..."
pip3 install ccxt python-dotenv aiohttp websockets

# Create minimal .env for testing
if [ ! -f .env ]; then
    cat > .env << 'EOF'
# Leave these empty for price monitoring mode
BINANCE_API_KEY=
BINANCE_SECRET=
COINBASE_API_KEY=
COINBASE_SECRET=
KRAKEN_API_KEY=
KRAKEN_SECRET=

# For mainnet flash loans (ADVANCED ONLY)
PRIVATE_KEY=
ETH_RPC_URL=https://mainnet.infura.io/v3/YOUR_KEY
FLASH_CONTRACT_ADDRESS=

# Settings
MIN_PROFIT_USD=100
DEPLOYMENT=local
EOF
    echo "✅ Created basic .env file"
fi

echo ""
echo "🎯 CHOOSE YOUR STARTING POINT:"
echo ""
echo "1. 🔰 DEMO MODE (No API keys needed)"
echo "   - Shows how the bot works with simulated data"
echo "   - Safe to run, no real trading"
echo ""
echo "2. 📊 REAL PRICE MONITORING (No API keys needed)"  
echo "   - Fetches real prices from public APIs"
echo "   - Shows actual arbitrage opportunities"
echo "   - No trading, just monitoring"
echo ""
echo "3. 🚀 FULL ARBITRAGE BOT (API keys required)"
echo "   - Executes real trades"
echo "   - Requires exchange API keys"
echo "   - Can make/lose real money"
echo ""

read -p "Enter choice (1/2/3): " choice

case $choice in
    1)
        echo "🔰 Starting DEMO mode..."
        python3 working_bot.py
        ;;
    2)
        echo "📊 Starting REAL PRICE monitoring..."
        python3 src/python/simple_bot.py
        ;;
    3)
        echo "🚀 Starting FULL arbitrage bot..."
        echo "⚠️  Make sure you've added API keys to .env file!"
        echo "⚠️  This can execute real trades with real money!"
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            python3 runner.py
        else
            echo "Cancelled."
        fi
        ;;
    *)
        echo "Invalid choice. Starting demo mode..."
        python3 working_bot.py
        ;;
esac