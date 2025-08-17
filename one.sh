#!/bin/bash

# Crypto Arbitrage Bot Setup Script
# This script will help you set up the arbitrage bot with all required API keys and dependencies

set -e  # Exit on any error

echo "🚀 CRYPTO ARBITRAGE BOT SETUP"
echo "=============================="
echo ""

# Check if we're on macOS (M1/Intel) or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macOS"
    if [[ $(uname -m) == "arm64" ]]; then
        ARCH="M1"
    else
        ARCH="Intel"
    fi
else
    PLATFORM="Linux"
    ARCH="x86_64"
fi

echo "🖥️  Platform: $PLATFORM $ARCH"
echo ""

# Step 1: Install Python dependencies
echo "📦 Step 1: Installing Python dependencies..."
echo "============================================"

# Install required Python packages
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "✅ Python dependencies installed"
echo ""

# Step 2: Set up environment file
echo "🔑 Step 2: Setting up API keys (.env file)"
echo "=========================================="

if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
# Exchange API Keys (for CEX arbitrage)
BINANCE_API_KEY=
BINANCE_SECRET=
COINBASE_API_KEY=
COINBASE_SECRET=
COINBASE_PASSPHRASE=
KRAKEN_API_KEY=
KRAKEN_SECRET=
BYBIT_API_KEY=
BYBIT_SECRET=
OKX_API_KEY=
OKX_SECRET=
OKX_PASSPHRASE=

# Ethereum/Web3 Configuration (for flash loans)
PRIVATE_KEY=
ETH_RPC_URL=https://mainnet.infura.io/v3/YOUR_INFURA_KEY
INFURA_URL=https://mainnet.infura.io/v3/YOUR_INFURA_KEY
ALCHEMY_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_KEY

# Flash Loan Contract (deploy this first!)
FLASH_CONTRACT_ADDRESS=

# Trading Parameters
MIN_PROFIT_USD=100
MAX_GAS_PRICE_GWEI=200

# Safety Settings
DEPLOYMENT=mainnet
EOF
    echo "✅ Created .env file"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🔑 YOU NEED TO ADD YOUR API KEYS TO THE .env FILE"
echo ""
echo "Here's how to get each API key:"
echo ""

# Binance API Keys
echo "📈 BINANCE API KEYS:"
echo "1. Go to https://www.binance.com/en/my/settings/api-management"
echo "2. Create new API key"
echo "3. Enable 'Enable Spot & Margin Trading'"
echo "4. Add your IP address to IP whitelist"
echo "5. Copy API Key and Secret to .env file"
echo ""

# Coinbase API Keys
echo "💰 COINBASE API KEYS:"
echo "1. Go to https://www.coinbase.com/settings/api"
echo "2. Create new API key"
echo "3. Enable trading permissions"
echo "4. Save the Key, Secret, and Passphrase to .env file"
echo ""

# Ethereum RPC URLs
echo "⛓️  ETHEREUM RPC PROVIDERS:"
echo "INFURA:"
echo "1. Go to https://infura.io/register"
echo "2. Create new project"
echo "3. Copy the Mainnet URL and replace YOUR_INFURA_KEY"
echo ""
echo "ALCHEMY:"
echo "1. Go to https://dashboard.alchemy.com/"
echo "2. Create new app on Ethereum Mainnet"
echo "3. Copy the HTTPS URL and replace YOUR_ALCHEMY_KEY"
echo ""

# Private Key Warning
echo "🔐 ETHEREUM PRIVATE KEY:"
echo "⚠️  WARNING: This is for MAINNET trading with REAL money!"
echo "1. Create a NEW wallet specifically for this bot"
echo "2. Fund it with enough ETH for gas fees (~0.1 ETH)"
echo "3. Export the private key (64 characters, no 0x prefix)"
echo "4. Add it to .env file"
echo "⚠️  NEVER share your private key or commit it to git!"
echo ""

# Step 3: Check for Rust (optional but recommended)
echo "🦀 Step 3: Checking for Rust (optional performance boost)"
echo "======================================================"

if command -v cargo &> /dev/null; then
    echo "✅ Rust is installed"
    echo "Building high-performance Rust engine..."
    cargo build --release
    echo "✅ Rust engine built"
else
    echo "⚠️  Rust not installed (optional)"
    echo "To install Rust for better performance:"
    echo "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi

echo ""

# Step 4: Deploy Flash Loan Contract (for advanced users)
echo "📜 Step 4: Flash Loan Contract Deployment"
echo "========================================"
echo "⚠️  ADVANCED USERS ONLY: To use flash loan arbitrage, you need to deploy the smart contract"
echo ""
echo "1. Install Hardhat/Truffle:"
echo "   npm install -g @nomicfoundation/hardhat-toolbox"
echo ""
echo "2. Deploy the FlashLoanReceiver.sol contract:"
echo "   - Use Remix IDE: https://remix.ethereum.org/"
echo "   - Upload src/solidity/FlashLoanReceiver.sol"
echo "   - Deploy to Ethereum mainnet"
echo "   - Add deployed address to FLASH_CONTRACT_ADDRESS in .env"
echo ""
echo "3. Or use a testing contract address (limited functionality):"
echo "   FLASH_CONTRACT_ADDRESS=0x1234567890123456789012345678901234567890"
echo ""

# Step 5: Test the setup
echo "🧪 Step 5: Testing the setup"
echo "=========================="

echo "Testing basic bot functionality..."

# Test simple bot first
echo "Testing simple arbitrage scanner..."
if python3 simple_bot.py --test 2>/dev/null; then
    echo "✅ Simple bot test passed"
else
    echo "⚠️  Simple bot test failed, but continuing..."
fi

echo ""

# Step 6: Choose your bot mode
echo "🎯 Step 6: Choose Your Bot Mode"
echo "=============================="
echo ""
echo "Available bot modes:"
echo ""
echo "1. 🔰 BEGINNER: Simple price monitoring (safe)"
echo "   python3 simple_bot.py"
echo ""
echo "2. 📊 INTERMEDIATE: CEX arbitrage (requires API keys)"
echo "   python3 src/orchestrator_m1_fixed.py"
echo ""
echo "3. ⚡ ADVANCED: Flash loan arbitrage (requires contract + mainnet)"
echo "   python3 runner.py"
echo ""
echo "4. 🚀 FULL SYSTEM: All components (requires everything)"
echo "   python3 full_system.py"
echo ""

# Create launch scripts
echo "📝 Creating launch scripts..."

# Simple launcher
cat > run_simple.sh << 'EOF'
#!/bin/bash
echo "🔰 Starting Simple Arbitrage Monitor"
python3 simple_bot.py
EOF

# Intermediate launcher  
cat > run_intermediate.sh << 'EOF'
#!/bin/bash
echo "📊 Starting CEX Arbitrage Bot"
echo "Make sure your .env file has exchange API keys!"
python3 src/orchestrator_m1_fixed.py
EOF

# Advanced launcher
cat > run_advanced.sh << 'EOF'
#!/bin/bash
echo "⚡ Starting Flash Loan Arbitrage Bot"
echo "⚠️  WARNING: This trades on MAINNET with REAL money!"
echo "Make sure you have:"
echo "- Funded wallet with ETH for gas"
echo "- Deployed flash loan contract"
echo "- All RPC URLs configured"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm
if [ "$confirm" = "yes" ]; then
    python3 runner.py
else
    echo "Cancelled."
fi
EOF

# Make scripts executable
chmod +x run_simple.sh run_intermediate.sh run_advanced.sh

echo "✅ Launch scripts created"
echo ""

# Final instructions
echo "🎉 SETUP COMPLETE!"
echo "=================="
echo ""
echo "Next steps:"
echo ""
echo "1. 📝 Edit the .env file with your API keys:"
echo "   nano .env"
echo ""
echo "2. 🚀 Choose your bot mode:"
echo "   ./run_simple.sh      - Safe price monitoring"
echo "   ./run_intermediate.sh - CEX arbitrage (needs API keys)"
echo "   ./run_advanced.sh     - Flash loans (needs everything)"
echo ""
echo "3. 📚 Read the documentation:"
echo "   - Check README.md for detailed instructions"
echo "   - Review src/solidity/ for smart contracts"
echo "   - Look at strategies/ for different approaches"
echo ""
echo "⚠️  IMPORTANT REMINDERS:"
echo "- Start with simple mode first"
echo "- Never commit your .env file to git"
echo "- Test on testnet before mainnet"
echo "- Only invest what you can afford to lose"
echo ""
echo "🆘 Need help? Check the logs in logs/ directory"
echo ""
echo "Happy arbitraging! 🚀💰"