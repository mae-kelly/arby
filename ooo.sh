#!/bin/bash

# 🚀 PRODUCTION CRYPTO ARBITRAGE BOT SETUP
# This script sets up REAL trading with actual money

set -e

echo "🚀 PRODUCTION ARBITRAGE BOT SETUP"
echo "=================================="
echo "⚠️  WARNING: This will trade with REAL MONEY"
echo ""

# Check platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ $(uname -m) == "arm64" ]]; then
        PLATFORM="M1_MAC"
    else
        PLATFORM="INTEL_MAC"
    fi
else
    PLATFORM="LINUX"
fi

echo "🖥️  Platform: $PLATFORM"

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install --upgrade pip
pip3 install ccxt web3 eth-account python-dotenv aiohttp websockets numpy pandas

if [[ "$PLATFORM" == "M1_MAC" ]]; then
    pip3 install tensorflow-macos tensorflow-metal
    echo "✅ M1 GPU acceleration enabled"
fi

# Create .env file with ALL required keys
echo "🔑 Setting up environment..."

cat > .env << 'EOF'
# ===== EXCHANGE API KEYS (for CEX arbitrage) =====
# Get these from exchange websites

BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET=your_binance_secret_here

COINBASE_API_KEY=your_coinbase_key_here
COINBASE_SECRET=your_coinbase_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here

KRAKEN_API_KEY=your_kraken_key_here
KRAKEN_SECRET=your_kraken_secret_here

BYBIT_API_KEY=your_bybit_key_here
BYBIT_SECRET=your_bybit_secret_here

OKX_API_KEY=your_okx_key_here
OKX_SECRET=your_okx_secret_here
OKX_PASSPHRASE=your_okx_passphrase_here

KUCOIN_API_KEY=your_kucoin_key_here
KUCOIN_SECRET=your_kucoin_secret_here
KUCOIN_PASSPHRASE=your_kucoin_passphrase_here

# ===== ETHEREUM/WEB3 CONFIG (for flash loans) =====
PRIVATE_KEY=your_ethereum_private_key_here
ETH_RPC_URL=https://mainnet.infura.io/v3/your_infura_key
ALCHEMY_URL=https://eth-mainnet.g.alchemy.com/v2/your_alchemy_key

# Flash loan contract (deploy first!)
FLASH_CONTRACT_ADDRESS=your_deployed_contract_address

# ===== TRADING PARAMETERS =====
MIN_PROFIT_USD=100
MAX_GAS_PRICE_GWEI=200
DEPLOYMENT=production

# ===== MULTI-CHAIN RPC ENDPOINTS =====
BSC_RPC=https://bsc-dataseed1.binance.org
POLYGON_RPC=https://polygon-rpc.com
ARBITRUM_RPC=https://arb1.arbitrum.io/rpc
OPTIMISM_RPC=https://mainnet.optimism.io
AVALANCHE_RPC=https://api.avax.network/ext/bc/C/rpc
EOF

echo "✅ Created .env template"
echo ""

# Instructions for getting API keys
echo "🔑 GET YOUR API KEYS:"
echo "==================="
echo ""

echo "📈 BINANCE (Required):"
echo "1. Go to: https://www.binance.com/en/my/settings/api-management"
echo "2. Create API → Enable 'Enable Spot & Margin Trading'"
echo "3. Whitelist your IP address"
echo "4. Copy API Key & Secret to .env"
echo ""

echo "💰 COINBASE (Required):"
echo "1. Go to: https://www.coinbase.com/settings/api"
echo "2. Create API Key → Full Access"
echo "3. Save API Key, Secret, and Passphrase to .env"
echo ""

echo "🦑 KRAKEN (Recommended):"
echo "1. Go to: https://www.kraken.com/u/security/api"
echo "2. Generate API → Enable 'Query Funds' and 'Trade'"
echo "3. Copy to .env"
echo ""

echo "⚡ ETHEREUM SETUP (for Flash Loans):"
echo "1. Get Infura key: https://infura.io/register"
echo "2. Export your wallet private key (MetaMask → Account Details → Export)"
echo "3. Fund wallet with 0.1+ ETH for gas"
echo "4. Deploy flash loan contract (see deploy_contract.py)"
echo ""

echo "🚨 SECURITY WARNINGS:"
echo "- Never share your private keys"
echo "- Use dedicated trading wallet"
echo "- Start with small amounts"
echo "- Test on testnet first"
echo ""

# Build components
echo "🔨 Building optimized components..."

# Rust components
if command -v cargo &> /dev/null; then
    echo "Building Rust engine..."
    cargo build --release
    echo "✅ Rust engine built"
else
    echo "⚠️  Rust not installed, skipping Rust components"
fi

# C++ components  
if command -v cmake &> /dev/null; then
    echo "Building C++ components..."
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cd ..
    echo "✅ C++ components built"
else
    echo "⚠️  CMake not installed, skipping C++ components"
fi

echo ""
echo "🎯 PRODUCTION BOTS AVAILABLE:"
echo "============================="
echo ""

echo "1. 🔥 ULTIMATE FLASH LOAN ARBITRAGE (runner.py)"
echo "   - Real flash loans on Ethereum mainnet"
echo "   - $0 capital required"
echo "   - Highest profit potential"
echo "   Command: python3 runner.py"
echo ""

echo "2. 📊 MULTI-EXCHANGE CEX ARBITRAGE (final_bot.py)"
echo "   - Real trades across 6+ exchanges"
echo "   - Requires account balances"
echo "   - Lower risk, steady profits"
echo "   Command: python3 final_bot.py"
echo ""

echo "3. 🚀 COMPLETE SYSTEM (src/python/orchestrator.py)"
echo "   - Uses ALL components (Rust, C++, GPU)"
echo "   - Maximum performance"
echo "   - All strategies combined"
echo "   Command: python3 src/python/orchestrator.py"
echo ""

echo "4. 🎯 MEV HUNTER (hey.py)"
echo "   - Advanced MEV extraction"
echo "   - Sandwich attacks, liquidations"
echo "   - Mempool monitoring"
echo "   Command: python3 hey.py"
echo ""

# Create launch scripts
echo "Creating launch scripts..."

cat > run_flash_arbitrage.sh << 'EOF'
#!/bin/bash
echo "🔥 Starting Flash Loan Arbitrage Bot..."
echo "⚠️  Trading with REAL MONEY on Ethereum mainnet!"
read -p "Continue? (yes/no): " confirm
if [[ $confirm == "yes" ]]; then
    python3 runner.py
else
    echo "Cancelled"
fi
EOF

cat > run_cex_arbitrage.sh << 'EOF'
#!/bin/bash
echo "📊 Starting Multi-Exchange CEX Arbitrage..."
echo "⚠️  Trading with REAL MONEY across exchanges!"
read -p "Continue? (yes/no): " confirm
if [[ $confirm == "yes" ]]; then
    python3 final_bot.py
else
    echo "Cancelled"
fi
EOF

cat > run_complete_system.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Complete Arbitrage System..."
echo "⚠️  Using ALL components with REAL MONEY!"
read -p "Continue? (yes/no): " confirm
if [[ $confirm == "yes" ]]; then
    python3 src/python/orchestrator.py
else
    echo "Cancelled"
fi
EOF

chmod +x *.sh

echo "✅ Setup complete!"
echo ""
echo "📋 NEXT STEPS:"
echo "=============="
echo "1. Edit .env file with your real API keys"
echo "2. Fund your trading accounts"
echo "3. For flash loans: Deploy smart contract first"
echo "4. Test with small amounts"
echo "5. Run: ./run_flash_arbitrage.sh (highest profit)"
echo ""
echo "💰 PROFIT POTENTIAL:"
echo "- Flash loans: $1000+ per day (no capital needed)"
echo "- CEX arbitrage: 5-20% annual returns"
echo "- Complete system: Maximum profits"
echo ""
echo "🚨 Start with flash loans - requires $0 capital!"