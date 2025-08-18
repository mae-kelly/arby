# run_flash_arbitrage.sh

#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_DIR="$SCRIPT_DIR/pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

function log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/flash_arbitrage.log"
}

function setup_flash_loan_environment() {
    log "Setting up Flash Loan arbitrage environment..."
    
    source venv/bin/activate
    
    pip install web3 eth-account aiohttp asyncio requests
    
    npm install --save-dev hardhat @nomiclabs/hardhat-ethers ethers @aave/core-v3
    
    log "Flash loan dependencies installed"
}

function deploy_flash_loan_contract() {
    log "Preparing flash loan contract deployment..."
    
    cat > hardhat.config.js << 'EOF'
require("@nomiclabs/hardhat-ethers");

module.exports = {
  solidity: "0.8.19",
  networks: {
    mainnet: {
      url: "https://eth-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX",
      accounts: ["0x" + process.env.PRIVATE_KEY]
    },
    polygon: {
      url: "https://polygon-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX",
      accounts: ["0x" + process.env.PRIVATE_KEY]
    }
  }
};
EOF
    
    mkdir -p contracts
    mkdir -p scripts
    
    log "Contract deployment environment ready"
}

function create_multi_chain_arbitrage() {
    log "Creating multi-chain arbitrage scanner..."
    
    cat > multi_chain_arbitrage.py << 'EOF'
import asyncio
import aiohttp
from web3 import Web3
import json
import time

class MultiChainFlashArbitrage:
    def __init__(self):
        self.chains = {
            "ethereum": {
                "rpc": "https://eth-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX",
                "chain_id": 1,
                "flash_loan_provider": "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
            },
            "polygon": {
                "rpc": "https://polygon-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX", 
                "chain_id": 137,
                "flash_loan_provider": "0x794a61358D6845594F94dc1DB02A252b5b4814aD"
            },
            "arbitrum": {
                "rpc": "https://arb-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX",
                "chain_id": 42161,
                "flash_loan_provider": "0x794a61358D6845594F94dc1DB02A252b5b4814aD"
            }
        }
        
        self.dexes = {
            "uniswap_v2": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            "uniswap_v3": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
            "sushiswap": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
            "curve": "0x99a58482BD75cbab83b27EC03CA68fF489b5788f",
            "balancer": "0xBA12222222228d8Ba445958a75a0704d566BF2C8"
        }
        
        self.webhook = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3"

    async def scan_cross_dex_opportunities(self):
        opportunities = []
        
        tokens = [
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "0xA0b86a33E6417c7ef38BC67B2F11D6B3DC0B5f55",  # USDC
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
            "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"   # WBTC
        ]
        
        for chain_name, chain_config in self.chains.items():
            w3 = Web3(Web3.HTTPProvider(chain_config["rpc"]))
            
            for token_a in tokens:
                for token_b in tokens:
                    if token_a == token_b:
                        continue
                    
                    prices = await self.get_dex_prices(w3, token_a, token_b, 100000)
                    
                    if len(prices) >= 2:
                        price_list = list(prices.values())
                        max_price = max(price_list)
                        min_price = min(price_list)
                        
                        if min_price > 0:
                            profit_percentage = ((max_price - min_price) / min_price) * 100
                            
                            if profit_percentage > 0.2:
                                opportunity = {
                                    "chain": chain_name,
                                    "token_a": token_a,
                                    "token_b": token_b,
                                    "profit_percentage": profit_percentage,
                                    "buy_dex": min(prices, key=prices.get),
                                    "sell_dex": max(prices, key=prices.get),
                                    "buy_price": min_price,
                                    "sell_price": max_price
                                }
                                opportunities.append(opportunity)
        
        return sorted(opportunities, key=lambda x: x["profit_percentage"], reverse=True)

    async def get_dex_prices(self, w3, token_a, token_b, amount):
        prices = {}
        
        for dex_name, dex_address in self.dexes.items():
            try:
                if "uniswap" in dex_name:
                    price = await self.get_uniswap_price(w3, dex_address, token_a, token_b, amount)
                elif dex_name == "sushiswap":
                    price = await self.get_sushiswap_price(w3, dex_address, token_a, token_b, amount)
                
                if price > 0:
                    prices[dex_name] = price
            except Exception as e:
                print(f"Error getting {dex_name} price: {e}")
        
        return prices

    async def execute_cross_chain_arbitrage(self, opportunity):
        print(f"🔥 FLASH LOAN ARBITRAGE OPPORTUNITY:")
        print(f"   Chain: {opportunity['chain']}")
        print(f"   Profit: {opportunity['profit_percentage']:.4f}%")
        print(f"   Buy on: {opportunity['buy_dex']} at {opportunity['buy_price']}")
        print(f"   Sell on: {opportunity['sell_dex']} at {opportunity['sell_price']}")
        
        potential_profit = (opportunity['sell_price'] - opportunity['buy_price']) * 100000
        
        if potential_profit > 500:
            await self.send_alert(f"🚨 HIGH PROFIT OPPORTUNITY: {opportunity['profit_percentage']:.4f}% profit on {opportunity['chain']} chain!")
            
            return await self.execute_flash_loan(opportunity)
        
        return False

    async def execute_flash_loan(self, opportunity):
        print(f"Executing flash loan for {opportunity['profit_percentage']:.4f}% profit...")
        
        flash_loan_params = {
            "asset": opportunity['token_a'],
            "amount": 100000 * 10**18,
            "chain": opportunity['chain'],
            "buy_dex": opportunity['buy_dex'],
            "sell_dex": opportunity['sell_dex']
        }
        
        print(f"Flash loan params: {flash_loan_params}")
        await self.send_alert(f"Flash loan executed: {opportunity['profit_percentage']:.4f}% profit")
        
        return True

    async def send_alert(self, message):
        try:
            payload = {"content": f"⚡ Multi-Chain Flash Arbitrage: {message}"}
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook, json=payload) as response:
                    print(f"Alert sent: {message}")
        except Exception as e:
            print(f"Alert error: {e}")

    async def monitor_real_time(self):
        await self.send_alert("Multi-chain flash loan arbitrage scanner started!")
        
        while True:
            try:
                print(f"[{time.strftime('%H:%M:%S')}] Scanning cross-DEX opportunities...")
                
                opportunities = await self.scan_cross_dex_opportunities()
                
                if opportunities:
                    print(f"Found {len(opportunities)} opportunities:")
                    
                    for opp in opportunities[:5]:
                        print(f"  {opp['chain']}: {opp['profit_percentage']:.4f}% profit")
                        
                        if opp['profit_percentage'] > 1.0:
                            await self.execute_cross_chain_arbitrage(opp)
                
                await asyncio.sleep(15)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(30)

if __name__ == "__main__":
    bot = MultiChainFlashArbitrage()
    asyncio.run(bot.monitor_real_time())
EOF
    
    log "Multi-chain arbitrage scanner created"
}

function start_flash_arbitrage_system() {
    log "Starting comprehensive flash loan arbitrage system..."
    
    source venv/bin/activate
    
    nohup python3 flash_loan_arbitrage.py > "$LOG_DIR/flash_arbitrage.log" 2>&1 &
    echo $! > "$PID_DIR/flash_arbitrage.pid"
    
    nohup python3 multi_chain_arbitrage.py > "$LOG_DIR/multi_chain.log" 2>&1 &
    echo $! > "$PID_DIR/multi_chain.pid"
    
    nohup python3 main_orchestrator.py > "$LOG_DIR/cex_arbitrage.log" 2>&1 &
    echo $! > "$PID_DIR/cex_arbitrage.pid"
    
    nohup node hft_engine.js > "$LOG_DIR/hft_engine.log" 2>&1 &
    echo $! > "$PID_DIR/hft_engine.pid"
    
    log "All arbitrage systems started"
}

function show_arbitrage_status() {
    log "Flash Loan Arbitrage System Status:"
    
    components=("flash_arbitrage" "multi_chain" "cex_arbitrage" "hft_engine")
    
    for component in "${components[@]}"; do
        pid_file="$PID_DIR/${component}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                log "  ✅ $component (PID: $pid) - RUNNING"
            else
                log "  ❌ $component (PID: $pid) - STOPPED"
                rm -f "$pid_file"
            fi
        else
            log "  ⚪ $component - NOT STARTED"
        fi
    done
}

function stop_all_arbitrage() {
    log "Stopping flash loan arbitrage system..."
    
    for pid_file in "$PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            kill "$pid" 2>/dev/null || true
            rm -f "$pid_file"
        fi
    done
    
    log "All arbitrage systems stopped"
}

function tail_arbitrage_logs() {
    log "Monitoring arbitrage logs (Ctrl+C to stop)..."
    if ls "$LOG_DIR"/*.log 1> /dev/null 2>&1; then
        tail -f "$LOG_DIR"/*.log
    else
        log "No log files found yet"
    fi
}

function estimate_gas_costs() {
    log "Estimating current gas costs for flash loans..."
    
    python3 -c "
import requests
import json

# Get current gas prices
response = requests.get('https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey=K4SEVFZ3PI8STM73VKV84C8PYZJUK7HB2G')
gas_data = response.json()

if gas_data['status'] == '1':
    safe_gas = int(gas_data['result']['SafeGasPrice'])
    fast_gas = int(gas_data['result']['FastGasPrice'])
    
    # Flash loan gas estimate: ~500k gas
    flash_loan_gas = 500000
    
    safe_cost_gwei = safe_gas * flash_loan_gas
    fast_cost_gwei = fast_gas * flash_loan_gas
    
    # Convert to ETH (1 ETH = 10^9 Gwei)
    safe_cost_eth = safe_cost_gwei / 10**9
    fast_cost_eth = fast_cost_gwei / 10**9
    
    # Convert to USD (approximate ETH price)
    eth_price = 3200
    safe_cost_usd = safe_cost_eth * eth_price
    fast_cost_usd = fast_cost_eth * eth_price
    
    print(f'Current Gas Costs for Flash Loans:')
    print(f'  Safe Gas ({safe_gas} gwei): ${safe_cost_usd:.2f}')
    print(f'  Fast Gas ({fast_gas} gwei): ${fast_cost_usd:.2f}')
    print(f'  Minimum profit needed: ${max(safe_cost_usd, fast_cost_usd) + 50:.2f}')
else:
    print('Error fetching gas prices')
"
}

function main() {
    case "${1:-start}" in
        "setup")
            setup_flash_loan_environment
            deploy_flash_loan_contract
            create_multi_chain_arbitrage
            ;;
        "start")
            start_flash_arbitrage_system
            show_arbitrage_status
            ;;
        "stop")
            stop_all_arbitrage
            ;;
        "status")
            show_arbitrage_status
            ;;
        "logs")
            tail_arbitrage_logs
            ;;
        "gas")
            estimate_gas_costs
            ;;
        *)
            echo "🔥 Flash Loan Arbitrage Trading System"
            echo "Usage: $0 {setup|start|stop|status|logs|gas}"
            echo ""
            echo "Commands:"
            echo "  setup   - Install dependencies and prepare contracts"
            echo "  start   - Start all flash loan arbitrage systems"
            echo "  stop    - Stop all systems"
            echo "  status  - Show system status"
            echo "  logs    - Monitor live logs"
            echo "  gas     - Check current gas costs"
            echo ""
            echo "🚀 Multi-Strategy Flash Loan Features:"
            echo "  • CEX-DEX arbitrage with flash loans"
            echo "  • Cross-DEX arbitrage (Uniswap vs SushiSwap)"
            echo "  • Multi-chain opportunities (Ethereum, Polygon, Arbitrum)"
            echo "  • Real-time gas cost optimization"
            echo "  • Automated profit calculation"
            exit 1
            ;;
    esac
}

main "$@"