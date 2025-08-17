import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from blockchain.web3_manager import Web3Manager

class GasOptimizer:
    def __init__(self):
        self.web3_manager = Web3Manager()
        self.base_gas_price = 20 * 1e9
        self.max_gas_price = 200 * 1e9
        
    async def initialize(self):
        await self.web3_manager.initialize()
        print("⛽ Gas optimizer initialized")

    async def get_optimal_gas_price(self, urgency: str = "normal") -> int:
        try:
            current_gas = await self.web3_manager.get_gas_price()
            
            if urgency == "low":
                multiplier = 0.9
            elif urgency == "high":
                multiplier = 1.5
            elif urgency == "urgent":
                multiplier = 2.0
            else:
                multiplier = 1.1
            
            optimal_gas = int(current_gas * multiplier)
            optimal_gas = max(optimal_gas, self.base_gas_price)
            optimal_gas = min(optimal_gas, self.max_gas_price)
            
            print(f"⛽ Optimal gas price: {optimal_gas / 1e9:.1f} gwei")
            return optimal_gas
            
        except Exception as e:
            print(f"Failed to get optimal gas price: {e}")
            return int(20 * 1e9)

    async def shutdown(self):
        print("Gas optimizer shutdown")