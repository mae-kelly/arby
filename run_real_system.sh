#!/bin/bash

echo "🚀 REAL Trading System Launcher"
echo "==============================="

function start_real_system() {
    echo "Starting REAL components..."
    
    # Start realistic arbitrage detector
    echo "📊 Starting realistic arbitrage detector..."
    source venv/bin/activate
    nohup python3 realistic_arbitrage.py > logs/realistic_arbitrage.log 2>&1 &
    echo $! > pids/realistic_arbitrage.pid
    
    # Start real CEX monitor
    echo "💱 Starting CEX spread monitor..."
    nohup python3 main_orchestrator.py > logs/cex_monitor.log 2>&1 &
    echo $! > pids/cex_monitor.pid
    
    # Start real HFT engine
    echo "⚡ Starting HFT order book monitor..."
    nohup node hft_engine.js > logs/hft_monitor.log 2>&1 &
    echo $! > pids/hft_monitor.pid
    
    sleep 3
    echo "✅ All REAL components started"
}

function stop_real_system() {
    echo "Stopping all components..."
    
    if [ -d "pids" ]; then
        for pid_file in pids/*.pid; do
            if [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                kill "$pid" 2>/dev/null || true
                rm -f "$pid_file"
            fi
        done
    fi
    
    echo "✅ All components stopped"
}

function show_status() {
    echo "REAL Trading System Status:"
    
    components=("realistic_arbitrage" "cex_monitor" "hft_monitor")
    
    for component in "${components[@]}"; do
        pid_file="pids/${component}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo "  ✅ $component (PID: $pid) - RUNNING"
            else
                echo "  ❌ $component (PID: $pid) - STOPPED"
                rm -f "$pid_file"
            fi
        else
            echo "  ⚪ $component - NOT STARTED"
        fi
    done
}

function tail_logs() {
    echo "📊 Monitoring REAL system logs..."
    if ls logs/*.log 1> /dev/null 2>&1; then
        tail -f logs/*.log
    else
        echo "No log files found yet"
    fi
}

case "${1:-start}" in
    "start")
        start_real_system
        show_status
        ;;
    "stop")
        stop_real_system
        ;;
    "status")
        show_status
        ;;
    "logs")
        tail_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs}"
        echo ""
        echo "🎯 REAL System Features:"
        echo "  • Actual DEX price comparisons"
        echo "  • Real gas cost calculations" 
        echo "  • Realistic profit estimates ($2-20)"
        echo "  • Live CEX spread monitoring"
        echo "  • Real WebSocket order book feeds"
        ;;
esac
