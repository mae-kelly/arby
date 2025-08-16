"""Strategy registry and loader"""
from typing import List, Dict
import importlib
import os

class StrategyRegistry:
    def __init__(self, config):
        self.config = config
        self.strategies = {}
        self._load_strategies()
    
    def _load_strategies(self):
        """Load all strategy modules"""
        families_dir = 'engine/strategies/families'
        
        for filename in os.listdir(families_dir):
            if filename.startswith('family_') and filename.endswith('.py'):
                module_name = filename[:-3]
                module = importlib.import_module(f'engine.strategies.families.{module_name}')
                
                # Get all strategy classes from module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and hasattr(attr, 'id'):
                        strategy = attr(self.config)
                        self.strategies[strategy.id] = strategy
        
        print(f"Loaded {len(self.strategies)} strategies")
    
    def get_enabled(self, families: str) -> List['StrategySpec']:
        """Get enabled strategies from specified families"""
        enabled_families = [int(f) for f in families.split(',')]
        enabled = []
        
        for strategy in self.strategies.values():
            if strategy.family in enabled_families:
                if self.config.strategies.get(strategy.id, {}).get('enabled', True):
                    enabled.append(strategy)
        
        return enabled
