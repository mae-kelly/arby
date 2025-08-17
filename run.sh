#!/bin/bash

# Ultra-Fast Arbitrage Bot Runner
# Production-ready execution script with monitoring and recovery

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please copy .env.example to .env and configure it"
    exit 1
fi

# Set deployment mode
export DEPLOYMENT=${DEPLOYMENT:-local}
export NODE_ENV=production

# Performance settings
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-0}
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)
export VECLIB_MAXIMUM_THREADS=$(nproc)

# Memory settings
export MALLOC_ARENA_MAX=2
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_TOP_PAD_=131072

# Python settings
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Function to check dependencies
check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 not found${NC}"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Node.js not found${NC}"
        exit 1
    fi
    
    # Check Rust
    if ! command -v cargo &> /dev/null; then
        echo -e "${YELLOW}Rust not found, some components may not work${NC}"
    fi
    
    # Check CUDA
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}GPU detected:${NC}"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    else
        echo -e "${YELLOW}No NVIDIA GPU detected, using CPU${NC}"
    fi
    
    # Check Redis
    if ! command -v redis-cli &> /dev/null; then
        echo -e "${YELLOW}Redis not installed${NC}"
    fi
}

# Function to start Redis
start_redis() {
    if command -v redis-server &> /dev/null; then
        if ! pgrep -x redis-server > /dev/null; then
            echo -e "${BLUE}Starting Redis server...${NC}"
            redis-server --daemonize yes \
                        --port ${REDIS_PORT:-6379} \
                        --maxmemory 2gb \
                        --maxmemory-policy allkeys-lru \
                        --save "" \
                        --appendonly no
        else
            echo -e "${GREEN}Redis already running${NC}"
        fi
    fi
}

# Function to start monitoring
start_monitoring() {
    if [ "$1" = "dashboard" ]; then
        # Start Prometheus metrics exporter
        python3 -c "
from prometheus_client import start_http_server
import time
start_http_server(8000)
print('Metrics available at http://localhost:8000')
while True: time.sleep(1)
" &
        METRICS_PID=$!
        echo $METRICS_PID > .metrics.pid
    fi
}

# Function to compile contracts
compile_contracts() {
    if [ "$1" = "deploy" ]; then
        echo -e "${BLUE}Compiling smart contracts...${NC}"
        cd src/solidity
        npx truffle compile
        
        if [ "$2" = "mainnet" ]; then
            echo -e "${YELLOW}Deploying to mainnet - this will cost gas!${NC}"
            read -p "Are you sure? (yes/no): " confirm
            if [ "$confirm" = "yes" ]; then
                npx truffle migrate --network mainnet
            fi
        else
            npx truffle migrate --network goerli
        fi
        cd ../..
    fi
}

# Function to run the bot
run_bot() {
    MODE=$1
    
    case $MODE in
        "notebook")
            echo -e "${BLUE}Starting Jupyter notebook...${NC}"
            jupyter notebook main.ipynb --port=8888 --no-browser
            ;;
            
        "debug")
            echo -e "${BLUE}Starting in debug mode...${NC}"
            export DEBUG=true
            python3 -m pdb src/python/orchestrator.py
            ;;
            
        "test")
            echo -e "${BLUE}Running tests...${NC}"
            python3 -m pytest tests/ -v --cov=src/python --cov-report=html
            ;;
            
        "backtest")
            echo -e "${BLUE}Running backtest...${NC}"
            python3 src/python/backtest.py --start-date=$2 --end-date=$3
            ;;
            
        "profile")
            echo -e "${BLUE}Running with profiler...${NC}"
            python3 -m cProfile -o profile.stats src/python/orchestrator.py
            echo "Profile saved to profile.stats"
            echo "View with: python3 -m pstats profile.stats"
            ;;
            
        "production")
            echo -e "${GREEN}Starting arbitrage bot in production mode...${NC}"
            
            # Create log directory
            mkdir -p logs
            
            # Start with process manager
            if command -v pm2 &> /dev/null; then
                pm2 start src/python/orchestrator.py \
                    --name crypto-arb-bot \
                    --interpreter python3 \
                    --max-memory-restart 4G \
                    --log logs/bot.log \
                    --error logs/error.log \
                    --merge-logs
                    
                echo -e "${GREEN}Bot started with PM2${NC}"
                echo "View logs: pm2 logs crypto-arb-bot"
                echo "Stop bot: pm2 stop crypto-arb-bot"
            else
                # Run with nohup as fallback
                nohup python3 src/python/orchestrator.py \
                    > logs/bot.log 2> logs/error.log &
                    
                BOT_PID=$!
                echo $BOT_PID > .bot.pid
                echo -e "${GREEN}Bot started with PID: $BOT_PID${NC}"
            fi
            ;;
            
        *)
            echo -e "${BLUE}Starting arbitrage bot...${NC}"
            python3 src/python/orchestrator.py
            ;;
    esac
}

# Function to stop the bot
stop_bot() {
    echo -e "${YELLOW}Stopping arbitrage bot...${NC}"
    
    # Stop PM2 if running
    if command -v pm2 &> /dev/null; then
        pm2 stop crypto-arb-bot 2>/dev/null || true
    fi
    
    # Stop process by PID
    if [ -f .bot.pid ]; then
        kill $(cat .bot.pid) 2>/dev/null || true
        rm .bot.pid
    fi
    
    # Stop metrics server
    if [ -f .metrics.pid ]; then
        kill $(cat .metrics.pid) 2>/dev/null || true
        rm .metrics.pid
    fi
    
    # Stop Redis
    if command -v redis-cli &> /dev/null; then
        redis-cli shutdown 2>/dev/null || true
    fi
    
    echo -e "${GREEN}Bot stopped${NC}"
}

# Function to show status
show_status() {
    echo -e "${BLUE}=== Arbitrage Bot Status ===${NC}"
    
    # Check if bot is running
    if [ -f .bot.pid ] && kill -0 $(cat .bot.pid) 2>/dev/null; then
        echo -e "${GREEN}Bot: Running (PID: $(cat .bot.pid))${NC}"
    else
        echo -e "${RED}Bot: Not running${NC}"
    fi
    
    # Check Redis
    if pgrep -x redis-server > /dev/null; then
        echo -e "${GREEN}Redis: Running${NC}"
    else
        echo -e "${RED}Redis: Not running${NC}"
    fi
    
    # Check metrics
    if [ -f .metrics.pid ] && kill -0 $(cat .metrics.pid) 2>/dev/null; then
        echo -e "${GREEN}Metrics: Running (http://localhost:8000)${NC}"
    else
        echo -e "${YELLOW}Metrics: Not running${NC}"
    fi
    
    # Show recent logs
    if [ -f logs/bot.log ]; then
        echo -e "\n${BLUE}Recent logs:${NC}"
        tail -n 10 logs/bot.log
    fi
    
    # Show performance metrics
    if [ -f logs/performance.json ]; then
        echo -e "\n${BLUE}Performance:${NC}"
        python3 -c "
import json
with open('logs/performance.json') as f:
    data = json.load(f)
    print(f'Total Profit: \${data.get(\"total_profit\", 0):,.2f}')
    print(f'Success Rate: {data.get(\"success_rate\", 0):.1f}%')
    print(f'Total Trades: {data.get(\"total_trades\", 0)}')
"
    fi
}

# Function to update the bot
update_bot() {
    echo -e "${BLUE}Updating arbitrage bot...${NC}"
    
    # Pull latest code
    if [ -d .git ]; then
        git pull origin main
    fi
    
    # Update dependencies
    pip install -q --upgrade -r requirements.txt
    npm install
    
    # Rebuild components
    ./build.sh
    
    echo -e "${GREEN}Update complete${NC}"
}

# Main execution
case "$1" in
    start)
        check_dependencies
        start_redis
        start_monitoring ${2:-""}
        compile_contracts ${2:-""} ${3:-""}
        run_bot ${2:-"production"}
        ;;
        
    stop)
        stop_bot
        ;;
        
    restart)
        stop_bot
        sleep 2
        check_dependencies
        start_redis
        run_bot "production"
        ;;
        
    status)
        show_status
        ;;
        
    update)
        update_bot
        ;;
        
    notebook)
        check_dependencies
        start_redis
        run_bot "notebook"
        ;;
        
    debug)
        check_dependencies
        start_redis
        run_bot "debug"
        ;;
        
    test)
        run_bot "test"
        ;;
        
    backtest)
        run_bot "backtest" ${2:-"2024-01-01"} ${3:-"2024-12-31"}
        ;;
        
    profile)
        check_dependencies
        start_redis
        run_bot "profile"
        ;;
        
    logs)
        if [ -f logs/bot.log ]; then
            tail -f logs/bot.log
        else
            echo "No logs found"
        fi
        ;;
        
    monitor)
        watch -n 1 "$0 status"
        ;;
        
    help|*)
        echo "Ultra-Fast Arbitrage Bot Runner"
        echo ""
        echo "Usage: $0 {command} [options]"
        echo ""
        echo "Commands:"
        echo "  start [mode]      Start the bot (modes: production, dashboard, deploy)"
        echo "  stop              Stop the bot"
        echo "  restart           Restart the bot"
        echo "  status            Show bot status"
        echo "  update            Update bot and dependencies"
        echo "  notebook          Start Jupyter notebook"
        echo "  debug             Start in debug mode"
        echo "  test              Run tests"
        echo "  backtest [start] [end]  Run backtest"
        echo "  profile           Run with profiler"
        echo "  logs              Show logs"
        echo "  monitor           Monitor bot status"
        echo "  help              Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 start                    # Start in production mode"
        echo "  $0 start dashboard          # Start with monitoring dashboard"
        echo "  $0 start deploy mainnet     # Deploy contracts and start"
        echo "  $0 backtest 2024-01-01 2024-12-31  # Run backtest for 2024"
        echo ""
        ;;
esac

exit 0