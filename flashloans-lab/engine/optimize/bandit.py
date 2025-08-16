"""Thompson Sampling for strategy allocation"""
import numpy as np
from typing import List, Dict

class ThompsonBandit:
    def __init__(self):
        self.successes = {}
        self.failures = {}
        
    def record(self, simulations: List['SimResult']):
        """Record results for each strategy"""
        for sim in simulations:
            strategy_id = sim.diagnostics.get('strategy_id')
            if not strategy_id:
                continue
            
            if strategy_id not in self.successes:
                self.successes[strategy_id] = 1
                self.failures[strategy_id] = 1
            
            if sim.pnl_native > 0:
                self.successes[strategy_id] += 1
            else:
                self.failures[strategy_id] += 1
    
    def sample_weights(self) -> Dict[str, float]:
        """Sample weights for each strategy"""
        weights = {}
        
        for strategy_id in self.successes:
            # Thompson sampling from Beta distribution
            alpha = self.successes[strategy_id]
            beta = self.failures[strategy_id]
            
            sample = np.random.beta(alpha, beta)
            weights[strategy_id] = sample
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total
        
        return weights
