# 🚀 COMPLETE PRODUCTION ARBITRAGE BOT ENVIRONMENT
# ====================================================
# This file contains ALL the configuration needed for 
# a fully functional arbitrage bot across ALL exchanges,
# ALL tokens, and ALL blockchain layers.

# 🔐 SECURITY & RISK MANAGEMENT
# ====================================================
BOT_MODE=production
ENABLE_TRADING=false  # ⚠️ SET TO true ONLY WHEN READY FOR REAL MONEY
MAX_POSITION_USD=100000          # Maximum position size per trade
MAX_DAILY_LOSS_USD=25000         # Daily loss limit before auto-stop
MAX_TOTAL_EXPOSURE_USD=500000    # Maximum total exposure across all positions
EMERGENCY_STOP_LOSS_PCT=8.0      # Emergency stop loss percentage
POSITION_SIZE_PCT=5.0            # Percentage of account to use per trade
MIN_PROFIT_USD=50.0              # Minimum profit threshold per trade
MAX_SLIPPAGE_PCT=2.5             # Maximum allowed slippage
STOP_LOSS_PCT=3.0                # Individual trade stop loss

# 📊 CENTRALIZED EXCHANGES (CEX) - TIER 1
# ====================================================

# BINANCE (World's largest crypto exchange)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET=your_binance_secret_here
BINANCE_TESTNET=true             # Set to false for mainnet
BINANCE_MAX_ORDERS_PER_SECOND=10
BINANCE_TRADING_FEE=0.001        # 0.1% trading fee

# COINBASE (US-regulated, institutional grade)
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_SECRET=your_coinbase_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here
COINBASE_SANDBOX=true            # Set to false for mainnet
COINBASE_TRADING_FEE=0.005       # 0.5% trading fee

# KRAKEN (European focus, high security)
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_SECRET=your_kraken_secret_here
KRAKEN_TRADING_FEE=0.0026        # 0.26% trading fee

# BYBIT (Derivatives leader)
BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_SECRET=your_bybit_secret_here
BYBIT_TESTNET=true
BYBIT_TRADING_FEE=0.001

# OKX (Global derivatives and spot)
OKX_API_KEY=your_okx_api_key_here
OKX_SECRET=your_okx_secret_here
OKX_PASSPHRASE=your_okx_passphrase_here
OKX_TRADING_FEE=0.001

# KUCOIN (Wide altcoin selection)
KUCOIN_API_KEY=your_kucoin_api_key_here
KUCOIN_SECRET=your_kucoin_secret_here
KUCOIN_PASSPHRASE=your_kucoin_passphrase_here
KUCOIN_TRADING_FEE=0.001

# HUOBI (Asian markets)
HUOBI_API_KEY=your_huobi_api_key_here
HUOBI_SECRET=your_huobi_secret_here
HUOBI_TRADING_FEE=0.002

# GATE.IO (Extensive token listings)
GATEIO_API_KEY=your_gateio_api_key_here
GATEIO_SECRET=your_gateio_secret_here
GATEIO_TRADING_FEE=0.002

# MEXC (Emerging markets)
MEXC_API_KEY=your_mexc_api_key_here
MEXC_SECRET=your_mexc_secret_here
MEXC_TRADING_FEE=0.002

# BITGET (Copy trading leader)
BITGET_API_KEY=your_bitget_api_key_here
BITGET_SECRET=your_bitget_secret_here
BITGET_PASSPHRASE=your_bitget_passphrase_here
BITGET_TRADING_FEE=0.001

# 🌐 BLOCKCHAIN RPC ENDPOINTS
# ====================================================

# ETHEREUM MAINNET (Primary L1)
ETH_RPC_URL=https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID
ETH_RPC_BACKUP_1=https://eth-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_KEY
ETH_RPC_BACKUP_2=https://rpc.ankr.com/eth
ETH_PRIVATE_KEY=your_ethereum_private_key_here
ETH_CHAIN_ID=1

# BINANCE SMART CHAIN (BSC)
BSC_RPC_URL=https://bsc-dataseed1.binance.org
BSC_RPC_BACKUP=https://bsc-dataseed2.binance.org
BSC_PRIVATE_KEY=your_bsc_private_key_here
BSC_CHAIN_ID=56

# POLYGON (Ethereum L2)
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_KEY
POLYGON_RPC_BACKUP=https://rpc.ankr.com/polygon
POLYGON_PRIVATE_KEY=your_polygon_private_key_here
POLYGON_CHAIN_ID=137

# ARBITRUM (Optimistic Rollup)
ARBITRUM_RPC_URL=https://arb-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_KEY
ARBITRUM_RPC_BACKUP=https://rpc.ankr.com/arbitrum
ARBITRUM_PRIVATE_KEY=your_arbitrum_private_key_here
ARBITRUM_CHAIN_ID=42161

# OPTIMISM (Optimistic Rollup)
OPTIMISM_RPC_URL=https://opt-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_KEY
OPTIMISM_RPC_BACKUP=https://rpc.ankr.com/optimism
OPTIMISM_PRIVATE_KEY=your_optimism_private_key_here
OPTIMISM_CHAIN_ID=10

# AVALANCHE (High throughput L1)
AVALANCHE_RPC_URL=https://api.avax.network/ext/bc/C/rpc
AVALANCHE_RPC_BACKUP=https://rpc.ankr.com/avalanche
AVALANCHE_PRIVATE_KEY=your_avalanche_private_key_here
AVALANCHE_CHAIN_ID=43114

# BASE (Coinbase L2)
BASE_RPC_URL=https://mainnet.base.org
BASE_PRIVATE_KEY=your_base_private_key_here
BASE_CHAIN_ID=8453

# FANTOM (Fast finality)
FANTOM_RPC_URL=https://rpc.ankr.com/fantom
FANTOM_PRIVATE_KEY=your_fantom_private_key_here
FANTOM_CHAIN_ID=250

# CRONOS (Crypto.com Chain)
CRONOS_RPC_URL=https://evm.cronos.org
CRONOS_PRIVATE_KEY=your_cronos_private_key_here
CRONOS_CHAIN_ID=25

# 🏦 DEFI PROTOCOLS & SMART CONTRACTS
# ====================================================

# FLASH LOAN PROVIDERS
AAVE_LENDING_POOL=0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9
BALANCER_VAULT=0xBA12222222228d8Ba445958a75a0704d566BF2C8
COMPOUND_COMPTROLLER=0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B
MAKER_FLASH_LENDER=0x60744434d6339a6B27d73d9Eda62b6F66a0a04FA

# DEX ROUTERS
UNISWAP_V2_ROUTER=0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
UNISWAP_V3_ROUTER=0xE592427A0AEce92De3Edee1F18E0157C05861564
SUSHISWAP_ROUTER=0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F
CURVE_REGISTRY=0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5
PANCAKESWAP_ROUTER=0x10ED43C718714eb63d5aA57B78B54704E256C495
QUICKSWAP_ROUTER=0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff

# YOUR DEPLOYED ARBITRAGE CONTRACTS (Deploy using deploy_contracts.js)
MULTIDEX_ARBITRAGE_CONTRACT=your_deployed_multidex_contract_address
MEV_EXECUTOR_CONTRACT=your_deployed_mev_executor_address
FLASH_LOAN_RECEIVER_CONTRACT=your_deployed_flash_receiver_address

# 🎯 MEV & FLASHBOTS CONFIGURATION
# ====================================================
FLASHBOTS_RELAY_URL=https://relay.flashbots.net
FLASHBOTS_SIGNER_PRIVATE_KEY=your_flashbots_signer_key
EDEN_RELAY_URL=https://api.edennetwork.io/v1
MEV_BLOCKER_ENABLED=true
MAX_MEV_GAS_PRICE_GWEI=500

# ⛽ GAS OPTIMIZATION
# ====================================================
MAX_GAS_PRICE_GWEI=300          # Maximum gas price to pay
PRIORITY_FEE_GWEI=2             # Priority fee for transactions
GAS_ESTIMATION_BUFFER=1.2       # 20% buffer on gas estimates
DYNAMIC_GAS_PRICING=true        # Adjust gas based on network conditions

# 🔗 CROSS-CHAIN BRIDGES
# ====================================================
STARGATE_ROUTER=0x8731d54E9D02c286767d56ac03e8037C07e01e98
HOP_BRIDGE=0xb8901acB165ed027E32754E0FFe830802919727f
SYNAPSE_BRIDGE=0x2796317b0fF8538F253012862c06787Adfb8cEb6
MULTICHAIN_ROUTER=0xC10Ef9F491C9B59f936957026020C321651ac078

# 📊 TOKENS TO MONITOR (Comprehensive List)
# ====================================================

# MAJOR CRYPTOCURRENCIES
MONITORED_TOKENS_TIER1=BTC,ETH,BNB,SOL,ADA,DOT,MATIC,AVAX,LINK,UNI

# DEFI TOKENS
MONITORED_TOKENS_DEFI=AAVE,COMP,MKR,SUSHI,CRV,YFI,1INCH,SNX,BAL,LIDO

# STABLECOINS
STABLECOINS=USDT,USDC,DAI,BUSD,FRAX,TUSD,LUSD,sUSD,USDD

# LAYER 1 TOKENS
L1_TOKENS=ETH,BNB,SOL,ADA,DOT,AVAX,NEAR,ATOM,FTM,ONE

# LAYER 2 TOKENS
L2_TOKENS=MATIC,OP,ARB,IMX,LRC,METIS

# EXCHANGE TOKENS
EXCHANGE_TOKENS=BNB,UNI,SUSHI,CRV,1INCH,DYDX,GMX

# MEME TOKENS (High volatility opportunities)
MEME_TOKENS=DOGE,SHIB,PEPE,FLOKI,BABYDOGE

# 🎛️ TRADING STRATEGY CONFIGURATION
# ====================================================

# CEX ARBITRAGE
ENABLE_CEX_ARBITRAGE=true
CEX_MIN_PROFIT_PCT=0.1          # 0.1% minimum profit
CEX_MAX_POSITION_USD=50000      # Maximum position size
CEX_EXECUTION_DELAY_MS=50       # Execution delay in milliseconds

# DEX ARBITRAGE
ENABLE_DEX_ARBITRAGE=true
DEX_MIN_PROFIT_PCT=0.3          # 0.3% minimum profit (higher due to gas)
DEX_MAX_GAS_USD=100             # Maximum gas cost per trade
DEX_SLIPPAGE_TOLERANCE=1.0      # 1% slippage tolerance

# CROSS-CHAIN ARBITRAGE
ENABLE_CROSS_CHAIN_ARBITRAGE=true
CROSS_CHAIN_MIN_PROFIT_PCT=0.5  # 0.5% minimum profit
CROSS_CHAIN_MAX_BRIDGE_TIME_MIN=30  # Maximum bridge time
CROSS_CHAIN_BRIDGE_FEE_TOLERANCE=0.1  # 0.1% bridge fee tolerance

# FLASH LOAN ARBITRAGE
ENABLE_FLASH_LOAN_ARBITRAGE=true
FLASH_LOAN_MIN_PROFIT_USD=200   # Minimum profit to justify gas costs
FLASH_LOAN_MAX_AMOUNT_USD=2000000  # Maximum flash loan amount
FLASH_LOAN_SAFETY_MARGIN=1.1    # 10% safety margin

# MEV STRATEGIES
ENABLE_MEV_STRATEGIES=true
ENABLE_SANDWICH_ATTACKS=true    # ⚠️ Use responsibly
ENABLE_FRONTRUNNING=true
ENABLE_BACKRUNNING=true
ENABLE_LIQUIDATION_MEV=true
MEV_MIN_PROFIT_USD=300          # Higher threshold for MEV

# TRIANGULAR ARBITRAGE
ENABLE_TRIANGULAR_ARBITRAGE=true
TRIANGULAR_MIN_PROFIT_PCT=0.2   # 0.2% minimum profit
TRIANGULAR_MAX_HOPS=4           # Maximum number of trading hops

# 📈 PERFORMANCE & MONITORING
# ====================================================

# EXECUTION SETTINGS
WORKER_THREADS=16               # Number of worker threads
WS_CONNECTIONS_PER_EXCHANGE=3   # WebSocket connections per exchange
ORDER_BOOK_DEPTH=50             # Order book depth to analyze
PRICE_UPDATE_INTERVAL_MS=50     # Price update frequency
OPPORTUNITY_SCAN_INTERVAL_MS=25 # How often to scan for opportunities
MAX_CONCURRENT_TRADES=10        # Maximum simultaneous trades

# MONITORING & ALERTING
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=true
GRAFANA_PORT=3000

# DISCORD ALERTS
DISCORD_WEBHOOK_URL=your_discord_webhook_url_here
DISCORD_ALERT_MIN_PROFIT=500    # Minimum profit to alert

# TELEGRAM ALERTS
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# EMAIL ALERTS
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password_here
ALERT_EMAIL=your_alerts@gmail.com

# 💾 DATABASE CONFIGURATION
# ====================================================
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=20

POSTGRES_URL=postgresql://arbitrage_user:secure_password@localhost:5432/arbitrage_db
POSTGRES_MAX_CONNECTIONS=20

MONGODB_URL=mongodb://localhost:27017/arbitrage
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_influxdb_token_here

# 🧠 MACHINE LEARNING & AI
# ====================================================
ENABLE_ML_PREDICTIONS=true
ML_MODEL_PATH=./models/arbitrage_predictor.pkl
ML_RETRAIN_INTERVAL_HOURS=24
ML_CONFIDENCE_THRESHOLD=0.7

# 🔒 SECURITY CONFIGURATIONS
# ====================================================
API_RATE_LIMIT_PER_MINUTE=3000
IP_WHITELIST_ENABLED=false
IP_WHITELIST=127.0.0.1,YOUR_VPS_IP
ENCRYPTION_KEY=your_32_character_encryption_key_here
SESSION_TIMEOUT_MINUTES=30

# WALLET SECURITY
HARDWARE_WALLET_ENABLED=false  # Set to true if using hardware wallet
MULTISIG_ENABLED=false          # Set to true for multisig wallets
SPENDING_LIMITS_ENABLED=true

# 🌍 GLOBAL SETTINGS
# ====================================================
TIMEZONE=UTC
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR
LOG_RETENTION_DAYS=30
DATA_RETENTION_DAYS=90

# API TIMEOUTS
HTTP_TIMEOUT_SECONDS=10
WEBSOCKET_TIMEOUT_SECONDS=30
BLOCKCHAIN_RPC_TIMEOUT_SECONDS=20

# BACKUP & RECOVERY
AUTO_BACKUP_ENABLED=true
BACKUP_INTERVAL_HOURS=6
BACKUP_LOCATION=/backups/arbitrage

# 🚨 EMERGENCY CONFIGURATIONS
# ====================================================
EMERGENCY_STOP_HOTKEY=CTRL+ALT+E
EMERGENCY_CONTACT_EMAIL=emergency@yourcompany.com
EMERGENCY_CONTACT_PHONE=+1234567890
KILL_SWITCH_URL=https://yourdomain.com/emergency-stop

# CIRCUIT BREAKERS
CIRCUIT_BREAKER_ENABLED=true
MAX_CONSECUTIVE_LOSSES=5        # Stop after 5 consecutive losses
MAX_HOURLY_LOSS_USD=5000       # Stop if hourly loss exceeds this
UNUSUAL_ACTIVITY_THRESHOLD=3    # Stop if 3 unusual events detected

# ⚡ ADVANCED FEATURES
# ====================================================

# HIGH-FREQUENCY TRADING
HFT_MODE_ENABLED=false          # ⚠️ Requires specialized hardware
ULTRA_LOW_LATENCY=false         # Requires colocation
FPGA_ACCELERATION=false         # Hardware acceleration

# INSTITUTIONAL FEATURES
PRIME_BROKERAGE_ENABLED=false
OTC_TRADING_ENABLED=false
INSTITUTIONAL_APIS_ENABLED=false

# REGULATORY COMPLIANCE
TRADE_REPORTING_ENABLED=false
KYC_VERIFICATION_REQUIRED=false
AML_MONITORING_ENABLED=false
TAX_REPORTING_ENABLED=false

# 🎮 GAMING & GAMIFICATION
# ====================================================
LEADERBOARD_ENABLED=false
ACHIEVEMENT_SYSTEM_ENABLED=false
SOCIAL_TRADING_ENABLED=false

# ===============================================
# END OF CONFIGURATION
# ===============================================

# 📝 NOTES:
# - Replace ALL "your_*_here" values with actual credentials
# - Start with ENABLE_TRADING=false for testing
# - Test with small amounts before going full scale
# - Monitor continuously when live
# - Keep backups of this configuration file
# - NEVER commit real API keys to version control
# - Use environment-specific files (.env.production, .env.staging, .env.development)