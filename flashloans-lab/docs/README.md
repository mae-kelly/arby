# Flash Loans Lab - Production MEV Bot

Professional-grade MEV bot with 100+ strategies for flash loans, arbitrage, and liquidations.

## Features

- **100+ Trading Strategies**: DEX arbitrage, flash loans, liquidations, basis trades
- **Multi-Chain Support**: Base, Arbitrum, Ethereum mainnet
- **Self-Healing**: Automatic parameter tuning and strategy optimization
- **Paper Trading**: Full simulation with real data before going live
- **Risk Management**: Global caps, circuit breakers, token safety checks
- **Real-Time Dashboard**: Web UI for monitoring and control

## Quick Start

1. Initialize the repository:
```bash
chmod +x scripts/init_repo.sh
./scripts/init_repo.sh
```

2. Configure your RPC endpoints in `configs/chains.yaml`

3. Start paper trading:
```bash
./scripts/run_paper.sh
```

4. Open dashboard at http://localhost:3000

## Architecture

- **Python**: Core orchestrator, strategies, simulation
- **Rust**: High-performance AMM math and routing
- **TypeScript**: Web dashboard
- **Docker**: Containerized deployment

## Legal & Ethical

This bot only implements legal, non-predatory strategies:
- ✅ Arbitrage (improves market efficiency)
- ✅ Liquidations (maintains protocol health)
- ❌ No sandwich attacks
- ❌ No front-running
- ❌ No oracle manipulation

## Performance

Target metrics in optimal conditions:
- 95%+ simulation accuracy
- <100ms opportunity detection
- 20+ profitable trades per day
- 5-30 bps profit per trade

## Risk Warning

Crypto trading carries significant risks. This software is for educational purposes.
Always test thoroughly in paper mode before risking real funds.
