"""Gas price tracking and prediction"""
import asyncio
from typing import Dict
import numpy as np
from collections import deque

class GasTracker:
    def __init__(self, rpcs: Dict[str, 'EVMRPCClient']):
        self.rpcs = rpcs
        self.gas_history = {chain: deque(maxlen=100) for chain in rpcs}
        self.current_gas = {}
        
    async def update(self):
        """Update gas prices from all chains"""
        for chain, rpc in self.rpcs.items():
            base_fee = await rpc.get_base_fee()
            priority_fee = await rpc.get_priority_fee_hint()
            
            total_fee = base_fee + priority_fee
            self.gas_history[chain].append(total_fee)
            
            # Exponential smoothing
            if len(self.gas_history[chain]) > 1:
                alpha = 0.3
                smoothed = alpha * total_fee + (1 - alpha) * self.current_gas.get(chain, total_fee)
                self.current_gas[chain] = smoothed
            else:
                self.current_gas[chain] = total_fee
    
    async def get_current_gas(self) -> Dict[str, float]:
        """Get current smoothed gas prices"""
        await self.update()
        return self.current_gas
    
    def predict_gas(self, chain: str, blocks_ahead: int = 1) -> float:
        """Predict future gas price"""
        if chain not in self.gas_history or len(self.gas_history[chain]) < 2:
            return self.current_gas.get(chain, 20e9)  # 20 gwei default
        
        # Simple linear extrapolation
        history = list(self.gas_history[chain])
        if len(history) >= 2:
            trend = history[-1] - history[-2]
            prediction = history[-1] + trend * blocks_ahead
            return max(1e9, prediction)  # Min 1 gwei
        
        return history[-1]
