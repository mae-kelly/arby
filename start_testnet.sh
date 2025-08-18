#!/bin/bash

echo "🧪 STARTING SAFE TESTNET TRADING SYSTEM"
echo "========================================"
echo "✅ No real money at risk!"
echo "✅ Perfect for learning and testing"
echo ""

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if .env.testnet exists
if [ ! -f .env.testnet ]; then
    echo -e "${RED}❌ .env.testnet file not found!${NC}"
    echo "Creating template .env.testnet file..."
    
    cat > .env.testnet << 'ENVEOF'
# TESTNET CONFIGURATION - EDIT THESE VALUES
GOERLI_RPC=https://goerli.infura.io/v3/YOUR_INFURA_KEY
TESTNET_PRIVATE_KEY=your_testnet_private_key_without_0x
TESTNET_WALLET=your_testnet_wallet_address
DISCORD_WEBHOOK=your_discord_webhook_url
ENVEOF
    
    echo -e "${YELLOW}⚠️ Please edit .env.testnet with your testnet settings${NC}"
    echo "Then run this script again."
    exit 1
fi

echo -e "${GREEN}✅ Loading testnet configuration...${NC}"
source .env.testnet

# Check Python dependencies
echo -e "${BLUE}📦 Checking Python dependencies...${NC}"
python3 -c "import web3, aiohttp, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required Python packages..."
    pip3 install web3 aiohttp requests python-dotenv
fi

echo -e "${GREEN}✅ Dependencies ready${NC}"

# Test testnet configuration
echo -e "${BLUE}🧪 Testing testnet configuration...${NC}"
python3 testnet_config.py

echo ""
echo -e "${YELLOW}🚰 IMPORTANT: Get free testnet ETH first!${NC}"
echo "Visit these faucets to get free testnet ETH:"
echo "• Goerli: https://goerlifaucet.com"
echo "• Sepolia: https://sepoliafaucet.com"
echo "• Mumbai: https://faucet.polygon.technology"
echo ""

read -p "Have you gotten testnet ETH? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please get testnet ETH first, then run this script again."
    exit 1
fi

echo -e "${GREEN}🚀 Starting testnet arbitrage system...${NC}"

# Start testnet monitoring
echo "📊 Starting testnet monitor..."
python3 testnet_monitor.py &
MONITOR_PID=$!

# Start main testnet arbitrage (if available)
if [ -f "testnet_comprehensive_arbitrage.py" ]; then
    echo "💰 Starting testnet arbitrage scanner..."
    python3 testnet_comprehensive_arbitrage.py &
    ARBITRAGE_PID=$!
else
    echo "⚠️ testnet_comprehensive_arbitrage.py not found, using fallback..."
    python3 testnet_realistic_arbitrage.py &
    ARBITRAGE_PID=$!
fi

echo ""
echo -e "${GREEN}✅ TESTNET SYSTEM STARTED!${NC}"
echo "================================"
echo "📊 Monitor at: https://goerli.etherscan.io"
echo "💰 All trades are SIMULATED with fake money"
echo "🔍 Watch the console for arbitrage opportunities"
echo ""
echo -e "${RED}🛑 To stop all processes: killall python3${NC}"
echo -e "${BLUE}📱 Discord alerts: Check your webhook${NC}"

# Keep script running and handle shutdown
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down testnet system...${NC}"
    kill $MONITOR_PID $ARBITRAGE_PID 2>/dev/null
    echo -e "${GREEN}✅ Testnet system stopped safely${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for user interrupt
echo "Press Ctrl+C to stop the system..."
wait
