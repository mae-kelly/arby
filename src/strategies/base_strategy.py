import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from dataclasses import dataclass
from core.opportunity_scanner import Opportunity
from core.risk_manager import RiskManager
from core.portfolio_manager import PortfolioManager
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class TradeResult:
    success: bool
    profit: float
    gas_cost: float
    execution_time: float
    error: Optional[str] = None

class BaseStrategy(ABC):
    def __init__(self, risk_manager: RiskManager, portfolio_manager: PortfolioManager):
        self.risk_manager = risk_manager
        self.portfolio_manager = portfolio_manager
        self.name = self.__class__.__name__
        self.enabled = True
        self.running = False
        self.stats = {
            "trades_executed": 0,
            "successful_trades": 0,
            "total_profit": 0.0,
            "total_gas_cost": 0.0
        }

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def execute_opportunity(self, opportunity: Opportunity) -> TradeResult:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass

    async def start(self):
        self.running = True
        logger.info(f"Strategy {self.name} started")

    async def shutdown(self):
        self.running = False
        logger.info(f"Strategy {self.name} shutdown")

    def is_ready(self) -> bool:
        return self.enabled and self.running

    async def _execute_trade(self, trade_params: Dict) -> TradeResult:
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await self._perform_trade(trade_params)
            execution_time = asyncio.get_event_loop().time() - start_time
            
            self._update_stats(result, execution_time)
            
            return TradeResult(
                success=True,
                profit=result.get('profit', 0),
                gas_cost=result.get('gas_cost', 0),
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Trade execution failed: {e}")
            
            return TradeResult(
                success=False,
                profit=0,
                gas_cost=0,
                execution_time=execution_time,
                error=str(e)
            )

    @abstractmethod
    async def _perform_trade(self, trade_params: Dict) -> Dict:
        pass

    def _update_stats(self, result: Dict, execution_time: float):
        self.stats["trades_executed"] += 1
        
        if result.get('success', False):
            self.stats["successful_trades"] += 1
            self.stats["total_profit"] += result.get('profit', 0)
        
        self.stats["total_gas_cost"] += result.get('gas_cost', 0)

    def get_stats(self) -> Dict:
        success_rate = 0
        if self.stats["trades_executed"] > 0:
            success_rate = self.stats["successful_trades"] / self.stats["trades_executed"]
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "net_profit": self.stats["total_profit"] - self.stats["total_gas_cost"]
        }

    async def _validate_opportunity(self, opportunity: Opportunity) -> bool:
        if not self.is_ready():
            return False
        
        if not self.risk_manager.check_opportunity(opportunity):
            return False
        
        return await self._strategy_specific_validation(opportunity)

    @abstractmethod
    async def _strategy_specific_validation(self, opportunity: Opportunity) -> bool:
        pass

    def _calculate_position_size(self, opportunity: Opportunity) -> float:
        return self.risk_manager.get_position_size(opportunity)

    async def _log_trade_attempt(self, opportunity: Opportunity, params: Dict):
        logger.info(f"{self.name} attempting trade: {opportunity.strategy_type} "
                   f"profit={opportunity.profit_estimate} confidence={opportunity.confidence}")

    async def _log_trade_result(self, result: TradeResult):
        if result.success:
            logger.info(f"{self.name} trade successful: profit={result.profit} "
                       f"gas={result.gas_cost} time={result.execution_time:.2f}s")
        else:
            logger.error(f"{self.name} trade failed: {result.error} "
                        f"time={result.execution_time:.2f}s")