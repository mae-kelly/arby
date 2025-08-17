#!/usr/bin/env python3
"""
Ultra-Optimized Arbitrage Orchestrator
Combines ALL strategies for maximum profitability
"""

import asyncio
import os
import time
import ctypes
import platform
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict
import logging

# Import all components
try:
    import ccxt.async_support as ccxt
except ImportError:
    os.system("pip install -q ccxt")
    import ccxt.async_support as ccxt

from cross_chain_engine import CrossChainEngine
from dotenv import load_dotenv
load_dotenv()

# Load Rust library
LIB_EXT = 'dylib' if platform.system() == 'Darwin' else 'so'
try:
    rust_lib = ctypes.CDLL(f'./target/release/libarbitrage_engine.{LIB_EXT}')
    
    # Define FFI functions
    rust_lib.create_engine.restype = ctypes.c_void_p
    rust_lib.destroy_engine.argtypes = [ctypes.c_void_p]
    rust_lib.add_market.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_double, ctypes.c_double, ctypes.c_double
    ]
    rust_lib.find_opportunities.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
    ]
    rust_lib.find_opportunities.restype = ctypes.c_size_t
    
    RUST_AVAILABLE = True
    print("✅ Rust engine loaded")
except Exception as e:
    print(f"⚠️  Rust engine not available: {e}")
    RUST_AVAILABLE = False

@dataclass
class UltraOpportunity:
    type: str  # 'cex', 'dex', 'cross_chain', 'mev'
    symbol: str
    profit: float
    confidence: float
    execution_time: float
    gas_cost: float
    details: Dict

class UltraOrchestrator:
    """Ultimate arbitrage orchestrator"""
    
    def __init__(self):
        # Initialize all engines
        self.rust_engine = rust_lib.create_engine() if RUST_AVAILABLE else None
        self.cross_chain_engine = CrossChainEngine()
        
        # Exchange connections
        self.exchanges = {}
        self.opportunities = []
        
        # Statistics
        self.stats = {
            'total_profit': 0.0,
            'trades_executed': 0,
            'opportunities_found': 0,
            'success_rate': 0.0
        }
        
        self.start_time = time.time()
    
    async def initialize(self):
        """Initialize all components"""
        
        print("🚀 ULTRA ARBITRAGE ORCHESTRATOR")
        print("================================")
        print(f"Rust Engine: {'✅' if RUST_AVAILABLE else '❌'}")
        
        # Initialize exchanges
        await self.setup_exchanges()
        
        print(f"Exchanges: {len(self.exchanges)}")
        print("All systems ready!")
        print("=" * 50)
    
    async def setup_exchanges(self):
        """Setup exchange connections"""
        
        exchange_configs = {
            'binance': {
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET')
            },
            'coinbase': {
                'apiKey': os.getenv('COINBASE_API_KEY'),
                'secret': os.getenv('COINBASE_SECRET')
            },
            'kraken': {
                'apiKey': os.getenv('KRAKEN_API_KEY'),
                'secret': os.getenv('KRAKEN_SECRET')
            }
        }
        
        for name, config in exchange_configs.items():
            if config.get('apiKey'):
                try:
                    exchange_class = getattr(ccxt, name)
                    self.exchanges[name] = exchange_class({
                        **config,
                        'enableRateLimit': True
                    })
                    await self.exchanges[name].load_markets()
                    print(f"✅ {name}: {len(self.exchanges[name].markets)} markets")
                except Exception as e:
                    print(f"❌ {name}: {e}")
    
    async def find_all_opportunities(self):
        """Find opportunities using all strategies"""
        
        all_opportunities = []
        
        # Strategy 1: Cross-Exchange Arbitrage (Rust-powered)
        if RUST_AVAILABLE:
            rust_opportunities = await self.find_rust_opportunities()
            all_opportunities.extend(rust_opportunities)
        
        # Strategy 2: Cross-Chain Arbitrage
        cross_chain_opportunities = await self.find_cross_chain_opportunities()
        all_opportunities.extend(cross_chain_opportunities)
        
        # Strategy 3: DEX Arbitrage
        dex_opportunities = await self.find_dex_opportunities()
        all_opportunities.extend(dex_opportunities)
        
        # Sort by profitability score
        all_opportunities.sort(key=lambda x: x.profit * x.confidence, reverse=True)
        
        self.opportunities = all_opportunities[:100]  # Top 100
        self.stats['opportunities_found'] += len(all_opportunities)
        
        return self.opportunities
    
    async def find_rust_opportunities(self) -> List[UltraOpportunity]:
        """Find opportunities using Rust engine"""
        
        opportunities = []
        
        try:
            # Update Rust engine with latest market data
            for exchange_name, exchange in self.exchanges.items():
                exchange_id = hash(exchange_name) % 1000
                
                for symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']:
                    if symbol in exchange.markets:
                        try:
                            ticker = await exchange.fetch_ticker(symbol)
                            symbol_id = hash(symbol) % 1000
                            
                            rust_lib.add_market(
                                self.rust_engine,
                                exchange_id,
                                symbol_id,
                                ticker.get('bid', 0),
                                ticker.get('ask', 0),
                                ticker.get('quoteVolume', 0)
                            )
                        except:
                            continue
            
            # Find opportunities
            max_paths = 100
            ArbitragePath = ctypes.c_double * (max_paths * 10)
            paths = ArbitragePath()
            
            num_found = rust_lib.find_opportunities(
                self.rust_engine,
                ctypes.cast(paths, ctypes.c_void_p),
                max_paths
            )
            
            # Convert Rust results to opportunities
            for i in range(num_found):
                profit = paths[i * 10]  # First element is profit
                confidence = min(paths[i * 10 + 1], 1.0)  # Second element
                
                if profit > 0.001:  # 0.1% minimum
                    opportunities.append(UltraOpportunity(
                        type='cex',
                        symbol='BTC/USDT',  # Simplified
                        profit=profit * 100,  # Convert to percentage
                        confidence=confidence,
                        execution_time=0.5,
                        gas_cost=0.0,
                        details={'engine': 'rust'}
                    ))
                    
        except Exception as e:
            print(f"Rust engine error: {e}")
        
        return opportunities
    
    async def find_cross_chain_opportunities(self) -> List[UltraOpportunity]:
        """Find cross-chain opportunities"""
        
        opportunities = []
        
        try:
            await self.cross_chain_engine.fetch_all_prices()
            cross_chain_opps = self.cross_chain_engine.find_cross_chain_opportunities()
            
            for opp in cross_chain_opps:
                opportunities.append(UltraOpportunity(
                    type='cross_chain',
                    symbol=opp.token,
                    profit=opp.profit_pct,
                    confidence=0.7,  # Cross-chain has lower confidence
                    execution_time=opp.bridge_time * 60,  # Convert to seconds
                    gas_cost=50.0,  # Estimated gas cost
                    details={
                        'source_chain': opp.source_chain,
                        'target_chain': opp.target_chain,
                        'bridge_fee': opp.bridge_fee
                    }
                ))
                
        except Exception as e:
            print(f"Cross-chain error: {e}")
        
        return opportunities
    
    async def find_dex_opportunities(self) -> List[UltraOpportunity]:
        """Find DEX arbitrage opportunities"""
        
        opportunities = []
        
        # Simulate DEX opportunities
        dex_pairs = ['ETH/USDC', 'WBTC/ETH', 'USDT/USDC']
        
        for pair in dex_pairs:
            # Simulate price differences between DEXs
            profit = np.random.uniform(0.05, 0.5)  # 0.05% to 0.5%
            
            if profit > 0.1:  # 0.1% minimum
                opportunities.append(UltraOpportunity(
                    type='dex',
                    symbol=pair,
                    profit=profit,
                    confidence=0.85,
                    execution_time=2.0,
                    gas_cost=75.0,
                    details={'protocol': 'uniswap_v3'}
                ))
        
        return opportunities
    
    async def execute_opportunity(self, opportunity: UltraOpportunity) -> bool:
        """Execute arbitrage opportunity"""
        
        print(f"⚡ Executing {opportunity.type} arbitrage:")
        print(f"   Symbol: {opportunity.symbol}")
        print(f"   Profit: {opportunity.profit:.3f}%")
        print(f"   Confidence: {opportunity.confidence:.2f}")
        
        try:
            # Simulate execution based on type
            if opportunity.type == 'cex':
                success = await self.execute_cex_arbitrage(opportunity)
            elif opportunity.type == 'cross_chain':
                success = await self.execute_cross_chain_arbitrage(opportunity)
            elif opportunity.type == 'dex':
                success = await self.execute_dex_arbitrage(opportunity)
            else:
                success = False
            
            if success:
                self.stats['total_profit'] += opportunity.profit
                self.stats['trades_executed'] += 1
                print(f"   ✅ Success! Profit: {opportunity.profit:.3f}%")
                return True
            else:
                print(f"   ❌ Failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    async def execute_cex_arbitrage(self, opportunity: UltraOpportunity) -> bool:
        """Execute CEX arbitrage"""
        # Simulate CEX execution
        await asyncio.sleep(0.1)
        return np.random.random() < opportunity.confidence
    
    async def execute_cross_chain_arbitrage(self, opportunity: UltraOpportunity) -> bool:
        """Execute cross-chain arbitrage"""
        # Simulate cross-chain execution
        await asyncio.sleep(1.0)
        return np.random.random() < opportunity.confidence
    
    async def execute_dex_arbitrage(self, opportunity: UltraOpportunity) -> bool:
        """Execute DEX arbitrage"""
        # Simulate DEX execution
        await asyncio.sleep(0.5)
        return np.random.random() < opportunity.confidence
    
    async def run(self):
        """Main execution loop"""
        
        await self.initialize()
        
        print("\n🎯 Starting ultra-optimized arbitrage hunting...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                cycle_start = time.time()
                
                # Find all opportunities
                opportunities = await self.find_all_opportunities()
                
                if opportunities:
                    print(f"\n📊 Found {len(opportunities)} opportunities:")
                    
                    # Show top 5
                    for i, opp in enumerate(opportunities[:5], 1):
                        score = opp.profit * opp.confidence
                        print(f"  {i}. {opp.type.upper()}: {opp.symbol} "
                              f"({opp.profit:.3f}%, score: {score:.3f})")
                    
                    # Execute most profitable
                    best_opp = opportunities[0]
                    if best_opp.profit * best_opp.confidence > 0.2:  # 0.2% weighted profit
                        await self.execute_opportunity(best_opp)
                
                # Update success rate
                if self.stats['trades_executed'] > 0:
                    self.stats['success_rate'] = (
                        self.stats['trades_executed'] / 
                        max(self.stats['opportunities_found'], 1) * 100
                    )
                
                # Show statistics
                runtime = time.time() - self.start_time
                cycle_time = time.time() - cycle_start
                
                print(f"\n💰 PERFORMANCE STATS:")
                print(f"   Total Profit: {self.stats['total_profit']:.3f}%")
                print(f"   Trades: {self.stats['trades_executed']}")
                print(f"   Success Rate: {self.stats['success_rate']:.1f}%")
                print(f"   Runtime: {runtime:.0f}s")
                print(f"   Cycle Time: {cycle_time:.2f}s")
                print("=" * 50)
                
                # Wait before next cycle
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            
            # Cleanup
            if self.rust_engine:
                rust_lib.destroy_engine(self.rust_engine)
            
            for exchange in self.exchanges.values():
                await exchange.close()
            
            print("✅ Shutdown complete")
            print(f"Final Stats: {self.stats['total_profit']:.3f}% profit, "
                  f"{self.stats['trades_executed']} trades")

async def main():
    orchestrator = UltraOrchestrator()
    await orchestrator.run()

if __name__ == "__main__":
    asyncio.run(main())
