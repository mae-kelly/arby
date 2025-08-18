#!/bin/bash

# ==========================================
# OKX CEX ARBITRAGE QUICK START
# ==========================================
# This script sets up and starts OKX-based arbitrage
# using your existing API credentials

echo "🏦 STARTING OKX CEX ARBITRAGE SYSTEM"
echo "====================================="
echo "✅ Using your existing OKX API credentials"
echo "✅ Safe sandbox mode - no real money at risk"
echo "✅ Real market data and opportunities"
echo ""

# Create simplified .env file for OKX CEX trading
cat > .env.okx << 'EOF'
# OKX CEX ARBITRAGE CONFIGURATION
# Your existing credentials - already configured!

OKX_API_KEY=8a760df1-4a2d-471b-ba42-d16893614dab
OKX_SECRET_KEY=C9F3FC89A6A30226E11DFFD098C7CF3D
OKX_PASSPHRASE=Shamrock1!
OKX_SANDBOX=true
OKX_BASE_URL=https://www.okx.com

# DISCORD ALERTS
DISCORD_WEBHOOK=https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3

# CEX ARBITRAGE SETTINGS
MIN_SPREAD_THRESHOLD=0.05  # 0.05% minimum spread
MAX_TRADE_SIZE=1000        # USD maximum per trade
ENABLE_TRADING=false       # Start with monitoring only

# TRADING PAIRS TO MONITOR
TRADING_PAIRS=BTC-USDT,ETH-USDT,SOL-USDT,DOGE-USDT
EOF

echo "✅ Created .env.okx configuration"

# Install Python dependencies
echo "📦 Installing required Python packages..."
pip install aiohttp requests python-dotenv

echo "✅ Dependencies installed"

# Test OKX connection
echo "🔌 Testing OKX API connection..."

python3 << 'PYEOF'
import asyncio
import aiohttp
import hmac
import hashlib
import base64
import time
import json
import os

# Load OKX credentials
OKX_API_KEY = "8a760df1-4a2d-471b-ba42-d16893614dab"
OKX_SECRET_KEY = "C9F3FC89A6A30226E11DFFD098C7CF3D"
OKX_PASSPHRASE = "Shamrock1!"
OKX_BASE_URL = "https://www.okx.com"

def sign_request(timestamp, method, request_path, body=""):
    message = timestamp + method + request_path + body
    mac = hmac.new(
        bytes(OKX_SECRET_KEY, encoding='utf8'),
        bytes(message, encoding='utf-8'),
        digestmod=hashlib.sha256
    )
    return base64.b64encode(mac.digest()).decode()

async def test_okx_connection():
    print("🔍 Testing OKX API connection...")
    
    try:
        # Test with BTC-USDT ticker
        timestamp = str(int(time.time() * 1000))
        method = "GET"
        request_path = "/api/v5/market/ticker?instId=BTC-USDT"
        
        headers = {
            'OK-ACCESS-KEY': OKX_API_KEY,
            'OK-ACCESS-SIGN': sign_request(timestamp, method, request_path),
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': OKX_PASSPHRASE,
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OKX_BASE_URL}{request_path}", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("code") == "0":
                        ticker = data["data"][0]
                        price = ticker.get("last")
                        bid = ticker.get("bidPx")
                        ask = ticker.get("askPx")
                        spread = ((float(ask) - float(bid)) / float(bid)) * 100
                        
                        print("✅ OKX API connection successful!")
                        print(f"📊 BTC-USDT: ${price}")
                        print(f"💰 Bid: ${bid} | Ask: ${ask}")
                        print(f"📈 Spread: {spread:.4f}%")
                        return True
                    else:
                        print(f"❌ OKX API error: {data.get('msg')}")
                        return False
                else:
                    print(f"❌ HTTP error: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

# Run the test
if asyncio.run(test_okx_connection()):
    print("\n🎉 OKX API is working perfectly!")
    print("Ready to start arbitrage monitoring...")
else:
    print("\n❌ OKX API connection failed")
    print("Check your credentials or try again later")
PYEOF

echo ""
echo "🚀 STARTING OKX CEX ARBITRAGE MONITOR..."
echo "======================================="

# Start the OKX arbitrage system
python3 main_orchestrator.py &
MAIN_PID=$!

echo "✅ OKX arbitrage system started!"
echo ""
echo "📊 WHAT'S HAPPENING:"
echo "• Monitoring BTC, ETH, SOL, DOGE spreads on OKX"
echo "• Looking for arbitrage opportunities"
echo "• Alerts will be sent to Discord"
echo "• Running in SAFE SANDBOX mode"
echo ""
echo "🔍 WATCH FOR:"
echo "• Spread alerts in console"
echo "• Discord notifications"
echo "• Arbitrage opportunity detection"
echo ""
echo "🛑 TO STOP: Press Ctrl+C or run: killall python3"

# Function to handle cleanup
cleanup() {
    echo ""
    echo "🛑 Stopping OKX arbitrage system..."
    kill $MAIN_PID 2>/dev/null
    echo "✅ System stopped cleanly"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "💡 TIP: Open another terminal to run additional commands"
echo "📱 Check your Discord for arbitrage alerts!"
echo ""
echo "Press Ctrl+C to stop the system..."

# Wait for the main process
wait $MAIN_PID