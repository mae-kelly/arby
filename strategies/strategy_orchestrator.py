import asyncio
from typing import Dict, List
import importlib
import os

class StrategyOrchestrator:
    def __init__(self):
        self.strategies = {}
        self.active_strategies = set()
        self.strategy_performance = {}
        self.load_all_strategies()
    
    def load_all_strategies(self):
        """Load all strategy modules"""
        strategy_modules = [
            'flash.aave_flash',
            'flash.balancer_flash',
            'cross-exchange.cex_arbitrage',
            'dex.uniswap_v3_strategy',
            'mev.sandwich_attack',
            'cross-chain.bridge_arbitrage',
            'statistical.mean_reversion'
        ]
        
        for module_path in strategy_modules:
            try:
                module = importlib.import_module(f'strategies.{module_path}')
                strategy_name = module_path.split('.')[-1]
                self.strategies[strategy_name] = module
                self.active_strategies.add(strategy_name)
            except ImportError as e:
                print(f"Failed to load strategy {module_path}: {e}")
    
    async def scan_all_opportunities(self) -> List[Dict]:
        """Scan all active strategies for opportunities"""
        all_opportunities = []
        
        tasks = []
        for strategy_name in self.active_strategies:
            if strategy_name in self.strategies:
                tasks.append(self.scan_strategy(strategy_name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_opportunities.extend(result)
        
        # Sort by expected profit
        return sorted(all_opportunities, 
                     key=lambda x: x.get('expected_profit', 0), 
                     reverse=True)
    
    async def scan_strategy(self, strategy_name: str) -> List[Dict]:
        """Scan a specific strategy for opportunities"""
        try:
            strategy_module = self.strategies[strategy_name]
            
            # Different strategies have different interfaces
            if hasattr(strategy_module, 'scan_opportunities'):
                return await strategy_module.scan_opportunities()
            elif hasattr(strategy_module, 'scan_all_pairs'):
                return await strategy_module.scan_all_pairs()
            else:
                return []
                
        except Exception as e:
            print(f"Error scanning strategy {strategy_name}: {e}")
            return []
    
    def get_strategy_performance(self) -> Dict:
        """Get performance metrics for all strategies"""
        return self.strategy_performance.copy()
    
    def enable_strategy(self, strategy_name: str):
        """Enable a strategy"""
        if strategy_name in self.strategies:
            self.active_strategies.add(strategy_name)
    
    def disable_strategy(self, strategy_name: str):
        """Disable a strategy"""
        self.active_strategies.discard(strategy_name)
