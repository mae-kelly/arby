import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
import time
import random
from ..pricing.route_search import Route
from ..datasources.gas_tracker import GasEstimate

@dataclass
class CandidateTrade:
    strategy_name: str
    route: Route
    flash_loan_amount: int
    flash_loan_token: str
    expected_profit: int
    gas_estimate: int
    confidence: float = 0.8
    timestamp: float = field(default_factory=time.time)
    
@dataclass
class SimulationResult:
    trade: CandidateTrade
    success: bool
    realized_profit: int
    gas_cost: int
    flash_loan_fee: int
    slippage_cost: int
    latency_penalty: int
    total_cost: int
    net_pnl: int
    execution_time_ms: float
    revert_probability: float
    fill_probability: float
    diagnostics: Dict[str, any] = field(default_factory=dict)

class ExecutionSimulator:
    def __init__(self, config):
        self.config = config
        self.slippage_model = SlippageModel()
        self.gas_model = GasModel()
        self.latency_model = LatencyModel()
        
    async def simulate_trade(self, 
                           trade: CandidateTrade, 
                           gas_estimate: GasEstimate,
                           current_block: int) -> SimulationResult:
        """Simulate complete trade execution"""
        
        start_time = time.time()
        
        # 1. Calculate gas costs
        gas_cost = self.gas_model.calculate_gas_cost(
            trade.gas_estimate, gas_estimate.max_fee
        )
        
        # 2. Calculate flash loan fees
        flash_loan_fee = self._calculate_flash_loan_fee(
            trade.flash_loan_amount, trade.flash_loan_token
        )
        
        # 3. Model slippage impact
        slippage_cost = self.slippage_model.calculate_slippage_cost(trade.route)
        
        # 4. Model latency/MEV impact
        latency_penalty = self.latency_model.calculate_latency_penalty(
            trade, current_block
        )
        
        # 5. Calculate revert probability
        revert_prob = self._calculate_revert_probability(trade, gas_estimate)
        
        # 6. Calculate fill probability
        fill_prob = self._calculate_fill_probability(trade, gas_estimate)
        
        # 7. Determine if trade would succeed
        total_cost = gas_cost + flash_loan_fee + slippage_cost + latency_penalty
        realized_profit = max(0, trade.expected_profit - slippage_cost - latency_penalty)
        net_pnl = realized_profit - total_cost
        
        # Trade succeeds if profitable after all costs and random factors
        success = (
            net_pnl > 0 and 
            random.random() > revert_prob and
            random.random() < fill_prob
        )
        
        execution_time = (time.time() - start_time) * 1000  # ms
        
        return SimulationResult(
            trade=trade,
            success=success,
            realized_profit=realized_profit,
            gas_cost=gas_cost,
            flash_loan_fee=flash_loan_fee,
            slippage_cost=slippage_cost,
            latency_penalty=latency_penalty,
            total_cost=total_cost,
            net_pnl=net_pnl,
            execution_time_ms=execution_time,
            revert_probability=revert_prob,
            fill_probability=fill_prob,
            diagnostics={
                'route_hops': len(trade.route.hops),
                'price_impact': trade.route.calculate_price_impact(),
                'gas_efficiency': realized_profit / gas_cost if gas_cost > 0 else 0
            }
        )
        
    def _calculate_flash_loan_fee(self, amount: int, token: str) -> int:
        """Calculate flash loan fees from different protocols"""
        # Aave v3: 0.05% (5 bps)
        aave_fee = amount * 5 // 10000
        
        # Balancer: 0% (but opportunity cost of gas)
        balancer_fee = 0
        
        # Return conservative estimate (Aave)
        return aave_fee
        
    def _calculate_revert_probability(self, 
                                    trade: CandidateTrade, 
                                    gas_estimate: GasEstimate) -> float:
        """Estimate probability of transaction reverting"""
        base_revert_rate = 0.02  # 2% base failure rate
        
        # Higher revert probability for:
        # - More complex routes
        complexity_factor = len(trade.route.hops) * 0.005
        
        # - Lower gas prices (less likely to be included)
        gas_factor = max(0, (1.0 - gas_estimate.confidence) * 0.1)
        
        # - Higher price impact
        impact_factor = trade.route.calculate_price_impact() * 0.05
        
        return min(0.5, base_revert_rate + complexity_factor + gas_factor + impact_factor)
        
    def _calculate_fill_probability(self,
                                  trade: CandidateTrade,
                                  gas_estimate: GasEstimate) -> float:
        """Estimate probability of trade being filled"""
        base_fill_rate = 0.85  # 85% base fill rate
        
        # Higher fill probability for:
        # - Higher gas prices
        gas_bonus = gas_estimate.confidence * 0.1
        
        # - Lower complexity
        complexity_penalty = (len(trade.route.hops) - 1) * 0.05
        
        # - Higher confidence
        confidence_bonus = trade.confidence * 0.05
        
        return max(0.1, min(0.98, 
            base_fill_rate + gas_bonus - complexity_penalty + confidence_bonus
        ))

class SlippageModel:
    def __init__(self):
        self.base_slippage_bps = 1  # 1 bps base slippage
        
    def calculate_slippage_cost(self, route: Route) -> int:
        """Calculate slippage cost for route"""
        price_impact = route.calculate_price_impact()
        
        # Slippage increases quadratically with price impact
        slippage_rate = self.base_slippage_bps + (price_impact * 100) ** 1.5
        slippage_rate = min(slippage_rate, 500)  # Cap at 5%
        
        return int(route.total_amount_out * slippage_rate / 10000)
        
class GasModel:
    def __init__(self):
        # Gas costs for different operations (rough estimates)
        self.gas_costs = {
            'flash_loan': 50000,
            'swap_v2': 100000,
            'swap_v3': 150000, 
            'transfer': 21000,
            'approval': 45000
        }
        
    def calculate_gas_cost(self, estimated_gas: int, gas_price_wei: int) -> int:
        """Calculate total gas cost in wei"""
        # Add 10% buffer to gas estimate
        actual_gas = int(estimated_gas * 1.1)
        return actual_gas * gas_price_wei

class LatencyModel:
    def __init__(self):
        self.block_time = 12.0  # seconds
        
    def calculate_latency_penalty(self, trade: CandidateTrade, current_block: int) -> int:
        """Calculate penalty due to block inclusion delays"""
        # Assume 1 block delay on average
        expected_delay_blocks = 1.0
        
        # Price decay rate: 0.1% per block for volatile pairs
        decay_rate_per_block = 0.001
        
        total_decay = expected_delay_blocks * decay_rate_per_block
        penalty = int(trade.expected_profit * total_decay)
        
        return min(penalty, trade.expected_profit // 2)  # Cap at 50% of profit
