#!/usr/bin/env python3
"""CLI for Flash Loans Lab MEV Bot"""
import asyncio
import click
import yaml
from pathlib import Path
from typing import Optional
from .config import Config
from .datasources import EVMRPCClient, DEXIndex, CEXWebsocket, GasTracker
from .strategies.registry import StrategyRegistry
from .execsim.simulator import Simulator
from .optimize.bandit import ThompsonBandit
from .observability.metrics import MetricsServer

@click.group()
def cli():
    """Flash Loans Lab - MEV Bot Control Center"""
    pass

@cli.command()
@click.option('--chains', default='base,arbitrum', help='Chains to scan')
@click.option('--families', default='1,2,3', help='Strategy families to enable')
@click.option('--min-edge-bps', default=6, help='Minimum edge in basis points')
@click.option('--mode', default='paper', help='Execution mode: paper/live')
async def scan(chains: str, families: str, min_edge_bps: int, mode: str):
    """Run continuous arbitrage scanning"""
    config = Config.load()
    registry = StrategyRegistry(config)
    simulator = Simulator(config)
    allocator = ThompsonBandit()
    metrics = MetricsServer()
    
    print(f"Starting scan on {chains} with min edge {min_edge_bps} bps")
    
    # Initialize data sources
    rpcs = {}
    for chain in chains.split(','):
        rpcs[chain] = EVMRPCClient(config.get_rpc(chain))
    
    dex_index = DEXIndex(rpcs)
    await dex_index.initialize()
    
    gas_tracker = GasTracker(rpcs)
    cex_ws = CEXWebsocket(['binance', 'okx'])
    await cex_ws.connect()
    
    # Main loop
    while True:
        try:
            # Snapshot current state
            state = {
                'pools': await dex_index.get_active_pools(),
                'gas': await gas_tracker.get_current_gas(),
                'cex_prices': cex_ws.get_latest_prices(),
                'block': await rpcs[chains.split(',')[0]].get_block_number()
            }
            
            # Discover opportunities
            candidates = []
            for strategy in registry.get_enabled(families):
                try:
                    opps = await strategy.discover(state)
                    candidates.extend(opps)
                except Exception as e:
                    print(f"Strategy {strategy.id} error: {e}")
            
            # Simulate all candidates
            simulations = []
            for candidate in candidates:
                sim = await simulator.simulate(candidate, state)
                if sim.ok and sim.pnl_native >= config.min_ev_native:
                    simulations.append(sim)
                    print(f"✅ Profitable: {sim.pnl_native:.4f} ETH from {candidate.strategy_id}")
            
            # Update allocator with results
            allocator.record(simulations)
            
            # Push to metrics
            metrics.record_scan(len(candidates), len(simulations))
            
            await asyncio.sleep(config.loop_seconds)
            
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"Loop error: {e}")
            await asyncio.sleep(5)

@cli.command()
def paper():
    """Start paper trading mode"""
    asyncio.run(scan(chains='base,arbitrum', families='1,2,3', min_edge_bps=6, mode='paper'))

@cli.command()
def optimize():
    """Run parameter optimization"""
    from .optimize.bayes_opt import BayesianOptimizer
    optimizer = BayesianOptimizer()
    optimizer.run_optimization()

@cli.command()
def dashboard():
    """Launch web dashboard"""
    import subprocess
    subprocess.run(['npm', 'run', 'dev'], cwd='webui')

if __name__ == '__main__':
    cli()
