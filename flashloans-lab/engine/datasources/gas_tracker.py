import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .evm_rpc import EVMRPCClient
import statistics
from collections import deque
import time

@dataclass
class GasEstimate:
    base_fee: int
    priority_fee: int
    max_fee: int
    confidence: float  # 0-1, probability of inclusion in next block
    
@dataclass
class GasMetrics:
    chain_id: int
    timestamp: int
    base_fee_gwei: float
    priority_fee_p50_gwei: float
    priority_fee_p90_gwei: float
    block_utilization: float

class GasTracker:
    def __init__(self, rpc_client: EVMRPCClient, history_blocks: int = 20):
        self.rpc = rpc_client
        self.history_blocks = history_blocks
        self.gas_history: deque = deque(maxlen=history_blocks)
        self.last_update = 0
        
    async def update_gas_metrics(self) -> GasMetrics:
        """Update gas metrics from recent blocks"""
        current_time = time.time()
        
        # Don't update more than once per 10 seconds
        if current_time - self.last_update < 10:
            return self.get_latest_metrics()
            
        latest_block = await self.rpc.get_latest_block()
        
        # Get gas data from recent blocks
        gas_prices = []
        utilization_rates = []
        
        for i in range(min(self.history_blocks, latest_block.number)):
            try:
                block_num = latest_block.number - i
                block = await self.rpc.w3.eth.get_block(block_num, full_transactions=True)
                
                if hasattr(block, 'baseFeePerGas') and block.baseFeePerGas:
                    base_fee = block.baseFeePerGas
                    
                    # Calculate priority fees from transactions
                    priority_fees = []
                    for tx in block.transactions:
                        if hasattr(tx, 'maxPriorityFeePerGas') and tx.maxPriorityFeePerGas:
                            priority_fees.append(tx.maxPriorityFeePerGas)
                    
                    if priority_fees:
                        gas_prices.extend(priority_fees)
                        
                    # Calculate block utilization
                    utilization = block.gasUsed / block.gasLimit if block.gasLimit > 0 else 0
                    utilization_rates.append(utilization)
                    
            except Exception as e:
                # Skip problematic blocks
                continue
                
        if gas_prices and utilization_rates:
            metrics = GasMetrics(
                chain_id=self.rpc.chain_id,
                timestamp=current_time,
                base_fee_gwei=latest_block.base_fee_per_gas / 1e9,
                priority_fee_p50_gwei=statistics.median(gas_prices) / 1e9,
                priority_fee_p90_gwei=statistics.quantiles(gas_prices, n=10)[8] / 1e9 if len(gas_prices) >= 10 else statistics.median(gas_prices) / 1e9,
                block_utilization=statistics.mean(utilization_rates)
            )
            
            self.gas_history.append(metrics)
            self.last_update = current_time
            return metrics
            
        # Fallback to current block data
        current_gas = await self.rpc.get_gas_price()
        return GasMetrics(
            chain_id=self.rpc.chain_id,
            timestamp=current_time,
            base_fee_gwei=current_gas['base_fee'] / 1e9,
            priority_fee_p50_gwei=current_gas['max_priority_fee'] / 1e9,
            priority_fee_p90_gwei=current_gas['max_priority_fee'] * 1.5 / 1e9,
            block_utilization=0.5  # assume 50% when unknown
        )
    
    def get_latest_metrics(self) -> Optional[GasMetrics]:
        """Get most recent gas metrics"""
        return self.gas_history[-1] if self.gas_history else None
    
    def estimate_gas_for_confidence(self, confidence: float = 0.95) -> GasEstimate:
        """Estimate gas price for given confidence level"""
        latest = self.get_latest_metrics()
        if not latest:
            # Fallback estimates
            return GasEstimate(
                base_fee=20_000_000_000,  # 20 gwei
                priority_fee=2_000_000_000,  # 2 gwei
                max_fee=25_000_000_000,  # 25 gwei
                confidence=0.8
            )
            
        # Adjust priority fee based on confidence level
        if confidence >= 0.95:
            priority_fee = int(latest.priority_fee_p90_gwei * 1e9)
        elif confidence >= 0.8:
            priority_fee = int(latest.priority_fee_p50_gwei * 1.5 * 1e9)
        else:
            priority_fee = int(latest.priority_fee_p50_gwei * 1e9)
            
        base_fee = int(latest.base_fee_gwei * 1e9)
        
        # Add buffer for base fee volatility
        if latest.block_utilization > 0.9:
            base_fee = int(base_fee * 1.125)  # 12.5% buffer for high utilization
        elif latest.block_utilization > 0.7:
            base_fee = int(base_fee * 1.06)   # 6% buffer
            
        return GasEstimate(
            base_fee=base_fee,
            priority_fee=priority_fee,
            max_fee=base_fee + priority_fee,
            confidence=confidence
        )
    
    def get_gas_trend(self) -> str:
        """Get gas price trend: rising, falling, stable"""
        if len(self.gas_history) < 3:
            return "stable"
            
        recent = list(self.gas_history)[-3:]
        base_fees = [m.base_fee_gwei for m in recent]
        
        if base_fees[-1] > base_fees[0] * 1.1:
            return "rising"
        elif base_fees[-1] < base_fees[0] * 0.9:
            return "falling"
        else:
            return "stable"
