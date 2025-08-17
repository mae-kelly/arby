#!/bin/bash

# 🚀 PRODUCTION ARBITRAGE BOT SETUP SCRIPT
# This script sets up the complete production arbitrage system
# with ALL exchanges, ALL tokens, ALL layers

set -e  # Exit on any error

echo "🚀 PRODUCTION ARBITRAGE SYSTEM SETUP"
echo "=================================================="
echo "⚠️  WARNING: This sets up REAL TRADING with REAL MONEY"
echo "⚠️  Only proceed if you understand the risks"
echo "=================================================="

# Confirm user wants to proceed
read -p "Do you want to proceed with PRODUCTION setup? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Setup cancelled"
    exit 1
fi

# Platform detection
OS=$(uname -s)
ARCH=$(uname -m)
IS_M1=false
IS_COLAB=false

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    IS_M1=true
    echo "✅ Detected: M1 Mac"
elif [[ -n "$COLAB_GPU" ]] || [[ -d "/content/sample_data" ]]; then
    IS_COLAB=true
    echo "✅ Detected: Google Colab"
else
    echo "✅ Detected: Standard Linux/Intel system"
fi

# Create production directory structure
echo "📁 Creating production directory structure..."
mkdir -p {config,logs,data,keys,backups,monitoring}
mkdir -p config/{exchanges,chains,strategies,risk}
mkdir -p data/{prices,orderbooks,transactions,analytics}
mkdir -p logs/{trading,system,errors,audit}

# Install system dependencies
echo "📦 Installing system dependencies..."
if [[ "$OS" == "Darwin" ]]; then
    # macOS
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install rust cmake boost rapidjson tbb
    
    if [[ "$IS_M1" == true ]]; then
        echo "Installing M1-specific packages..."
        brew install llvm
        export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
    fi
else
    # Linux
    sudo apt-get update
    sudo apt-get install -y build-essential cmake libboost-all-dev rapidjson-dev libtbb-dev
    
    # Install Rust
    if ! command -v rustc &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
        source ~/.cargo/env
    fi
fi

# Install Python dependencies with specific versions
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip

# Core trading dependencies
pip install ccxt==4.2.25 websockets==12.0 aiohttp==3.9.1
pip install python-dotenv==1.0.0 pandas==2.1.4 numpy==1.24.3
pip install web3==6.12.0 eth-account==0.10.0

# Security and encryption
pip install cryptography==41.0.8 pycryptodome==3.19.0
pip install bcrypt==4.1.2 keyring==24.3.0

# Database and caching
pip install redis==5.0.1 psycopg2-binary==2.9.9
pip install sqlalchemy==2.0.23 alembic==1.13.1

# Monitoring and observability
pip install prometheus-client==0.19.0 grafana-api==1.0.3
pip install sentry-sdk==1.38.0 structlog==23.2.0

# ML and analytics
pip install scikit-learn==1.3.2 lightgbm==4.1.0
pip install matplotlib==3.8.2 plotly==5.17.0

# Platform-specific GPU packages
if [[ "$IS_M1" == true ]]; then
    echo "Installing M1 GPU acceleration..."
    pip install tensorflow-macos==2.15.0 tensorflow-metal==1.1.0
    pip install accelerate==0.25.0
elif [[ "$IS_COLAB" == true ]]; then
    echo "Installing CUDA acceleration..."
    pip install cupy-cuda11x==12.3.0 numba==0.58.1
    pip install tensorflow-gpu==2.15.0
fi

# Build Rust components
echo "🦀 Building Rust arbitrage engine..."
if [[ "$IS_M1" == true ]]; then
    # M1-specific build flags
    export RUSTFLAGS="-C target-cpu=native"
    cargo build --release --features "metal-gpu"
else
    cargo build --release --features "cuda-gpu"
fi

# Build C++ components
echo "⚡ Building C++ high-performance components..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

# Compile GPU kernels if available
if [[ "$IS_COLAB" == true ]] || command -v nvcc &> /dev/null; then
    echo "🎮 Compiling CUDA kernels..."
    nvcc -O3 -arch=sm_80 --use_fast_math src/gpu_kernel.cu -o build/gpu_kernel.so -shared -Xcompiler -fPIC
elif [[ "$IS_M1" == true ]]; then
    echo "🍎 Compiling Metal kernels..."
    xcrun -sdk macosx metal -O3 src/metal/gpu_kernel.metal -o build/gpu_kernel.metallib
fi

# Setup secure environment configuration
echo "🔐 Setting up secure production environment..."
cat > .env.production << 'EOF'
# 🚀 PRODUCTION ARBITRAGE BOT CONFIGURATION
# ================================================

# SECURITY SETTINGS
BOT_MODE=production
ENABLE_TRADING=false  # SET TO true WHEN READY
MAX_POSITION_USD=50000
MAX_DAILY_LOSS_USD=10000
EMERGENCY_STOP_LOSS_PCT=5.0

# EXCHANGE API KEYS (REPLACE WITH YOUR KEYS)
# ================================================

# BINANCE (Largest volume)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET=your_binance_secret_here
BINANCE_TESTNET=true  # SET TO false FOR MAINNET

# COINBASE (US regulation friendly)
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_SECRET=your_coinbase_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here

# KRAKEN (European focus)
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_SECRET=your_kraken_secret_here

# BYBIT (Asian markets)
BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_SECRET=your_bybit_secret_here

# OKX (Global derivatives)
OKX_API_KEY=your_okx_api_key_here
OKX_SECRET=your_okx_secret_here
OKX_PASSPHRASE=your_okx_passphrase_here

# KUCOIN (Altcoins)
KUCOIN_API_KEY=your_kucoin_api_key_here
KUCOIN_SECRET=your_kucoin_secret_here
KUCOIN_PASSPHRASE=your_kucoin_passphrase_here

# HUOBI (Asian focus)
HUOBI_API_KEY=your_huobi_api_key_here
HUOBI_SECRET=your_huobi_secret_here

# GATEIO (Wide selection)
GATEIO_API_KEY=your_gateio_api_key_here
GATEIO_SECRET=your_gateio_secret_here

# BLOCKCHAIN RPC ENDPOINTS
# ================================================

# ETHEREUM MAINNET
ETH_RPC_URL=https://mainnet.infura.io/v3/YOUR_INFURA_KEY
ETH_RPC_BACKUP=https://eth-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_KEY
ETH_PRIVATE_KEY=your_ethereum_private_key_here

# BINANCE SMART CHAIN
BSC_RPC_URL=https://bsc-dataseed1.binance.org
BSC_PRIVATE_KEY=your_bsc_private_key_here

# POLYGON
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_KEY
POLYGON_PRIVATE_KEY=your_polygon_private_key_here

# ARBITRUM
ARBITRUM_RPC_URL=https://arb-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_KEY
ARBITRUM_PRIVATE_KEY=your_arbitrum_private_key_here

# OPTIMISM
OPTIMISM_RPC_URL=https://opt-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_KEY
OPTIMISM_PRIVATE_KEY=your_optimism_private_key_here

# AVALANCHE
AVALANCHE_RPC_URL=https://api.avax.network/ext/bc/C/rpc
AVALANCHE_PRIVATE_KEY=your_avalanche_private_key_here

# FLASH LOAN CONTRACTS (DEPLOY THESE FIRST)
# ================================================
AAVE_FLASH_CONTRACT=your_deployed_aave_flash_contract
BALANCER_FLASH_CONTRACT=your_deployed_balancer_flash_contract
UNISWAP_FLASH_CONTRACT=your_deployed_uniswap_flash_contract

# MEV AND FLASHBOTS
# ================================================
FLASHBOTS_SIGNER_KEY=your_flashbots_signer_key
FLASHBOTS_RPC_URL=https://relay.flashbots.net
MEV_BLOCKER_ENABLED=true

# MONITORING AND ALERTING
# ================================================
DISCORD_WEBHOOK_URL=your_discord_webhook_for_alerts
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# DATABASE CONFIGURATION
# ================================================
REDIS_URL=redis://localhost:6379/0
POSTGRES_URL=postgresql://user:pass@localhost:5432/arbitrage

# RISK MANAGEMENT
# ================================================
MIN_PROFIT_USD=25.0
MAX_SLIPPAGE_PCT=2.0
MAX_GAS_PRICE_GWEI=300
POSITION_SIZE_PCT=10.0
STOP_LOSS_PCT=3.0

# STRATEGY CONFIGURATION
# ================================================
ENABLE_CEX_ARBITRAGE=true
ENABLE_DEX_ARBITRAGE=true
ENABLE_CROSS_CHAIN=true
ENABLE_FLASH_LOANS=true
ENABLE_MEV_STRATEGIES=true
ENABLE_LIQUIDATIONS=true

# TOKENS TO MONITOR (ADD MORE AS NEEDED)
# ================================================
MONITORED_TOKENS=BTC,ETH,BNB,SOL,MATIC,AVAX,LINK,UNI,AAVE,COMP,MKR,SUSHI,CRV,YFI,1INCH
STABLE_COINS=USDT,USDC,DAI,BUSD,FRAX,TUSD

# PERFORMANCE TUNING
# ================================================
WORKER_THREADS=8
WS_CONNECTIONS_PER_EXCHANGE=5
ORDER_BOOK_DEPTH=20
PRICE_UPDATE_INTERVAL_MS=100
OPPORTUNITY_SCAN_INTERVAL_MS=50

EOF

# Create exchange-specific configurations
echo "📊 Creating exchange configurations..."

# Binance configuration
cat > config/exchanges/binance.json << 'EOF'
{
  "name": "binance",
  "type": "cex",
  "base_url": "https://api.binance.com",
  "websocket_url": "wss://stream.binance.com:9443/ws",
  "features": {
    "spot_trading": true,
    "futures_trading": true,
    "margin_trading": true,
    "lending": false
  },
  "fees": {
    "maker": 0.001,
    "taker": 0.001,
    "withdrawal_fees": {
      "BTC": 0.0005,
      "ETH": 0.005,
      "USDT": 1.0
    }
  },
  "limits": {
    "min_trade_usd": 10,
    "max_trade_usd": 1000000,
    "rate_limit_per_minute": 1200
  },
  "supported_symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "MATIC/USDT"],
  "priority": 1
}
EOF

# Create strategy configurations
echo "🎯 Creating strategy configurations..."

cat > config/strategies/cross_exchange.json << 'EOF'
{
  "name": "cross_exchange_arbitrage",
  "enabled": true,
  "min_profit_usd": 25.0,
  "max_position_usd": 50000,
  "max_slippage_pct": 1.0,
  "exchanges": ["binance", "coinbase", "kraken", "bybit", "okx"],
  "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"],
  "execution_delay_ms": 100,
  "risk_score_threshold": 0.8
}
EOF

cat > config/strategies/flash_loans.json << 'EOF'
{
  "name": "flash_loan_arbitrage",
  "enabled": true,
  "min_profit_usd": 100.0,
  "max_loan_amount_usd": 1000000,
  "providers": [
    {
      "name": "aave",
      "fee_pct": 0.09,
      "max_amount_usd": 10000000
    },
    {
      "name": "balancer",
      "fee_pct": 0.0,
      "max_amount_usd": 5000000
    },
    {
      "name": "dydx",
      "fee_pct": 0.0,
      "max_amount_usd": 1000000
    }
  ],
  "supported_tokens": ["WETH", "USDC", "USDT", "DAI", "WBTC"]
}
EOF

# Setup monitoring and alerting
echo "📈 Setting up monitoring and alerting..."

cat > config/monitoring.json << 'EOF'
{
  "prometheus": {
    "enabled": true,
    "port": 9090,
    "metrics_interval_seconds": 30
  },
  "grafana": {
    "enabled": true,
    "port": 3000,
    "dashboards": ["trading", "performance", "risk"]
  },
  "alerts": {
    "discord_enabled": true,
    "telegram_enabled": true,
    "email_enabled": false,
    "alert_conditions": {
      "large_profit": {
        "threshold_usd": 1000,
        "enabled": true
      },
      "large_loss": {
        "threshold_usd": 500,
        "enabled": true
      },
      "system_error": {
        "enabled": true
      },
      "api_failures": {
        "threshold_per_hour": 10,
        "enabled": true
      }
    }
  }
}
EOF

# Create smart contract deployment script
echo "📝 Creating smart contract deployment script..."

cat > deploy_contracts.js << 'EOF'
// Smart Contract Deployment Script for Flash Loan Arbitrage
const { ethers } = require('hardhat');

async function main() {
    console.log('🚀 Deploying flash loan arbitrage contracts...');
    
    // Deploy Multi-DEX Arbitrage Contract
    const MultiDexArbitrage = await ethers.getContractFactory('MultiDexArbitrage');
    const multiDex = await MultiDexArbitrage.deploy();
    await multiDex.deployed();
    console.log('✅ MultiDexArbitrage deployed to:', multiDex.address);
    
    // Deploy MEV Executor Contract
    const MEVExecutor = await ethers.getContractFactory('MEVExecutor');
    const mevExecutor = await MEVExecutor.deploy();
    await mevExecutor.deployed();
    console.log('✅ MEVExecutor deployed to:', mevExecutor.address);
    
    // Deploy Flash Loan Receiver
    const FlashLoanReceiver = await ethers.getContractFactory('FlashLoanReceiver');
    const flashReceiver = await FlashLoanReceiver.deploy();
    await flashReceiver.deployed();
    console.log('✅ FlashLoanReceiver deployed to:', flashReceiver.address);
    
    // Save contract addresses
    const contracts = {
        MultiDexArbitrage: multiDex.address,
        MEVExecutor: mevExecutor.address,
        FlashLoanReceiver: flashReceiver.address
    };
    
    require('fs').writeFileSync('config/contracts.json', JSON.stringify(contracts, null, 2));
    console.log('✅ Contract addresses saved to config/contracts.json');
}

main().catch(console.error);
EOF

# Create production startup script
echo "🚀 Creating production startup script..."

cat > start_production.sh << 'EOF'
#!/bin/bash

# 🚀 PRODUCTION ARBITRAGE BOT STARTUP SCRIPT

set -e

echo "🚀 Starting Production Arbitrage Bot"
echo "=================================================="

# Check if running as root (not recommended)
if [[ $EUID -eq 0 ]]; then
   echo "⚠️  WARNING: Running as root is not recommended"
   echo "Consider creating a dedicated user for the bot"
fi

# Load production environment
if [[ ! -f .env.production ]]; then
    echo "❌ .env.production file not found!"
    echo "Run ./setup_production.sh first"
    exit 1
fi

export $(cat .env.production | grep -v '^#' | xargs)

# Verify critical environment variables
required_vars=("BINANCE_API_KEY" "ETH_RPC_URL" "MIN_PROFIT_USD")
for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        echo "❌ Required environment variable $var is not set"
        exit 1
    fi
done

# Check if trading is explicitly enabled
if [[ "$ENABLE_TRADING" != "true" ]]; then
    echo "⚠️  Trading is DISABLED. Set ENABLE_TRADING=true in .env.production to enable"
    echo "Running in simulation mode..."
fi

# Start supporting services
echo "🔧 Starting supporting services..."

# Start Redis (for caching)
if ! pgrep redis-server > /dev/null; then
    echo "Starting Redis..."
    redis-server --daemonize yes
fi

# Start PostgreSQL (for data storage)
if ! pgrep postgres > /dev/null; then
    echo "Starting PostgreSQL..."
    sudo service postgresql start 2>/dev/null || true
fi

# Create log rotation
cat > /etc/logrotate.d/arbitrage-bot << 'LOGROTATE'
/path/to/arbitrage-bot/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 $USER $USER
    postrotate
        killall -SIGUSR1 arbitrage-bot 2>/dev/null || true
    endscript
}
LOGROTATE

# Set up monitoring
echo "📊 Setting up monitoring..."
python3 -c "
import json
import os
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# Start Prometheus metrics server
start_http_server(9090)
print('✅ Prometheus metrics server started on port 9090')
"

# Check system resources
echo "💻 Checking system resources..."
python3 -c "
import psutil
import sys

# Check available RAM
ram_gb = psutil.virtual_memory().total / (1024**3)
if ram_gb < 4:
    print(f'⚠️  Warning: Only {ram_gb:.1f}GB RAM available. Recommend 8GB+')

# Check available disk space
disk_gb = psutil.disk_usage('/').free / (1024**3)
if disk_gb < 10:
    print(f'⚠️  Warning: Only {disk_gb:.1f}GB disk space free. Recommend 50GB+')

# Check CPU cores
cpu_count = psutil.cpu_count()
print(f'✅ System resources: {ram_gb:.1f}GB RAM, {disk_gb:.1f}GB disk, {cpu_count} CPU cores')
"

# Test exchange connections
echo "🔌 Testing exchange connections..."
python3 -c "
import ccxt
import asyncio
import os

async def test_exchanges():
    exchanges = {
        'binance': ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY', ''),
            'secret': os.getenv('BINANCE_SECRET', ''),
            'testnet': True  # Start with testnet
        }),
        'coinbase': ccxt.coinbase({
            'apiKey': os.getenv('COINBASE_API_KEY', ''),
            'secret': os.getenv('COINBASE_SECRET', ''),
            'password': os.getenv('COINBASE_PASSPHRASE', ''),
        })
    }
    
    for name, exchange in exchanges.items():
        try:
            if exchange.apiKey:
                balance = await exchange.fetch_balance()
                print(f'✅ {name}: Connected successfully')
            else:
                print(f'⚠️  {name}: No API key configured')
        except Exception as e:
            print(f'❌ {name}: Connection failed - {str(e)[:100]}')
        finally:
            await exchange.close()

asyncio.run(test_exchanges())
"

# Final safety check
echo "🛡️  Final safety check..."
if [[ "$ENABLE_TRADING" == "true" ]]; then
    echo "=================================================="
    echo "⚠️  DANGER: REAL TRADING IS ENABLED"
    echo "⚠️  This bot will trade with REAL MONEY"
    echo "⚠️  You could lose significant amounts"
    echo "=================================================="
    
    read -p "Are you ABSOLUTELY SURE you want to continue? (type 'I UNDERSTAND THE RISKS'): " -r
    if [[ $REPLY != "I UNDERSTAND THE RISKS" ]]; then
        echo "Startup cancelled for safety"
        exit 1
    fi
fi

# Start the main arbitrage engine
echo "🚀 Starting main arbitrage engine..."

# Choose the appropriate orchestrator based on platform
if [[ "$IS_M1" == true ]]; then
    echo "Using M1-optimized orchestrator..."
    python3 src/orchestrator_m1_fixed.py
elif [[ "$IS_COLAB" == true ]]; then
    echo "Using Colab A100-optimized orchestrator..."
    python3 src/orchestrator_unified.py
else
    echo "Using full-featured orchestrator..."
    python3 src/python/orchestrator_full.py
fi

EOF

chmod +x start_production.sh

# Create safety and monitoring scripts
echo "🛡️  Creating safety and monitoring scripts..."

cat > emergency_stop.sh << 'EOF'
#!/bin/bash

echo "🚨 EMERGENCY STOP ACTIVATED"
echo "Stopping all trading immediately..."

# Kill all arbitrage processes
pkill -f "orchestrator"
pkill -f "arbitrage"

# Cancel all open orders (implement per exchange)
python3 -c "
import ccxt
import asyncio
import os

async def cancel_all_orders():
    exchanges = {
        'binance': ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET'),
        }),
        'coinbase': ccxt.coinbase({
            'apiKey': os.getenv('COINBASE_API_KEY'),
            'secret': os.getenv('COINBASE_SECRET'),
            'password': os.getenv('COINBASE_PASSPHRASE'),
        })
    }
    
    for name, exchange in exchanges.items():
        try:
            if exchange.apiKey:
                orders = await exchange.fetch_open_orders()
                for order in orders:
                    await exchange.cancel_order(order['id'])
                print(f'✅ Cancelled {len(orders)} orders on {name}')
        except Exception as e:
            print(f'Error cancelling orders on {name}: {e}')
        finally:
            await exchange.close()

asyncio.run(cancel_all_orders())
"

echo "✅ Emergency stop completed"
EOF

chmod +x emergency_stop.sh

# Create monitoring dashboard
cat > monitor.py << 'EOF'
#!/usr/bin/env python3
"""Real-time monitoring dashboard"""

import asyncio
import time
import psutil
import json
from datetime import datetime

class ArbitrageMonitor:
    def __init__(self):
        self.start_time = time.time()
        
    async def run_dashboard(self):
        while True:
            self.clear_screen()
            self.show_header()
            self.show_system_stats()
            self.show_trading_stats()
            self.show_recent_opportunities()
            await asyncio.sleep(5)
    
    def clear_screen(self):
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def show_header(self):
        runtime = time.time() - self.start_time
        print("🚀 ARBITRAGE BOT PRODUCTION MONITOR")
        print("=" * 50)
        print(f"Runtime: {runtime//3600:.0f}h {(runtime%3600)//60:.0f}m {runtime%60:.0f}s")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def show_system_stats(self):
        cpu_pct = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        
        print("💻 SYSTEM RESOURCES")
        print(f"CPU: {cpu_pct}%")
        print(f"RAM: {ram.percent}% ({ram.used//1024//1024//1024}GB used)")
        print()
    
    def show_trading_stats(self):
        print("💰 TRADING PERFORMANCE")
        print("Opportunities Found: 0")
        print("Trades Executed: 0")
        print("Total Profit: $0.00")
        print("Success Rate: 0%")
        print()
    
    def show_recent_opportunities(self):
        print("🎯 RECENT OPPORTUNITIES")
        print("No opportunities found yet...")
        print()

if __name__ == "__main__":
    monitor = ArbitrageMonitor()
    asyncio.run(monitor.run_dashboard())
EOF

# Final instructions
echo ""
echo "🎉 PRODUCTION SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "📋 NEXT STEPS:"
echo ""
echo "1. 🔐 ADD YOUR API KEYS:"
echo "   Edit .env.production and add your real API keys"
echo ""
echo "2. 💰 FUND YOUR ACCOUNTS:"
echo "   Add trading capital to your exchange accounts"
echo ""
echo "3. 🏗️  DEPLOY SMART CONTRACTS (for DeFi strategies):"
echo "   npm install && node deploy_contracts.js"
echo ""
echo "4. 🧪 TEST IN SIMULATION MODE:"
echo "   ./start_production.sh  # (trading disabled by default)"
echo ""
echo "5. 🚀 GO LIVE (when ready):"
echo "   Set ENABLE_TRADING=true in .env.production"
echo "   ./start_production.sh"
echo ""
echo "📊 MONITORING:"
echo "   python3 monitor.py                    # Real-time dashboard"
echo "   ./emergency_stop.sh                   # Emergency stop"
echo "   tail -f logs/trading/arbitrage.log    # Trading logs"
echo ""
echo "⚠️  SAFETY REMINDERS:"
echo "   - Start with small amounts"
echo "   - Test thoroughly in simulation mode"
echo "   - Monitor continuously when live"
echo "   - Have emergency stop procedures ready"
echo ""
echo "🎯 The bot will monitor ALL major exchanges and tokens:"
echo "   Exchanges: Binance, Coinbase, Kraken, Bybit, OKX, KuCoin, Huobi, Gate.io"
echo "   Chains: Ethereum, BSC, Polygon, Arbitrum, Optimism, Avalanche"
echo "   Strategies: CEX arbitrage, DEX arbitrage, Flash loans, MEV, Cross-chain"
echo ""
echo "Good luck with your arbitrage empire! 🚀💰"