import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from core.opportunity_scanner import Opportunity
from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

@dataclass
class RiskMetrics:
    total_exposure: float
    position_count: int
    daily_pnl: float
    max_drawdown: float
    var_95: float

class RiskManager:
    def __init__(self):
        self.positions = {}
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.exposure_limits = {
            "total": settings.max_position_size * 10,
            "per_strategy": settings.max_position_size * 2,
            "per_asset": settings.max_position_size
        }
        self.risk_metrics = RiskMetrics(0, 0, 0, 0, 0)

    async def initialize(self):
        await self._load_positions()
        await self._calculate_risk_metrics()

    async def _load_positions(self):
        pass

    async def _calculate_risk_metrics(self):
        total_exposure = sum(abs(pos['size'] * pos['price']) for pos in self.positions.values())
        position_count = len(self.positions)
        
        self.risk_metrics = RiskMetrics(
            total_exposure=total_exposure,
            position_count=position_count,
            daily_pnl=self.daily_pnl,
            max_drawdown=self.max_drawdown,
            var_95=total_exposure * 0.05
        )

    def check_opportunity(self, opportunity: Opportunity) -> bool:
        if not self._check_exposure_limits(opportunity):
            return False
        
        if not self._check_profit_threshold(opportunity):
            return False
        
        if not self._check_confidence_threshold(opportunity):
            return False
        
        if not self._check_correlation_limits(opportunity):
            return False
        
        return True

    def _check_exposure_limits(self, opportunity: Opportunity) -> bool:
        estimated_exposure = self._estimate_exposure(opportunity)
        
        if self.risk_metrics.total_exposure + estimated_exposure > self.exposure_limits["total"]:
            logger.warning(f"Total exposure limit exceeded: {estimated_exposure}")
            return False
        
        strategy_exposure = sum(
            abs(pos['size'] * pos['price']) for pos in self.positions.values()
            if pos.get('strategy') == opportunity.strategy_type
        )
        
        if strategy_exposure + estimated_exposure > self.exposure_limits["per_strategy"]:
            logger.warning(f"Strategy exposure limit exceeded: {opportunity.strategy_type}")
            return False
        
        return True

    def _check_profit_threshold(self, opportunity: Opportunity) -> bool:
        min_profit = settings.min_profit_threshold * 10000
        
        if opportunity.profit_estimate < min_profit:
            return False
        
        risk_adjusted_profit = opportunity.profit_estimate * opportunity.confidence
        
        return risk_adjusted_profit > min_profit * 0.8

    def _check_confidence_threshold(self, opportunity: Opportunity) -> bool:
        min_confidence = 0.5
        
        if opportunity.strategy_type == "flash_loan":
            min_confidence = 0.7
        elif opportunity.strategy_type == "liquidation":
            min_confidence = 0.8
        
        return opportunity.confidence >= min_confidence

    def _check_correlation_limits(self, opportunity: Opportunity) -> bool:
        similar_positions = [
            pos for pos in self.positions.values()
            if pos.get('strategy') == opportunity.strategy_type
        ]
        
        if len(similar_positions) > 5:
            return False
        
        return True

    def _estimate_exposure(self, opportunity: Opportunity) -> float:
        base_size = 1000.0
        
        if opportunity.strategy_type == "flash_loan":
            return opportunity.profit_estimate * 50
        elif opportunity.strategy_type == "cross_exchange":
            return base_size * 2
        elif opportunity.strategy_type == "triangular":
            return base_size
        elif opportunity.strategy_type == "liquidation":
            return opportunity.data.get('collateral_value', base_size)
        else:
            return base_size

    async def update_position(self, position_id: str, position_data: Dict):
        self.positions[position_id] = position_data
        await self._calculate_risk_metrics()

    async def close_position(self, position_id: str, pnl: float):
        if position_id in self.positions:
            del self.positions[position_id]
        
        self.daily_pnl += pnl
        
        if self.daily_pnl < self.max_drawdown:
            self.max_drawdown = self.daily_pnl
        
        await self._calculate_risk_metrics()

    async def check_exposure(self):
        await self._calculate_risk_metrics()
        
        if self.risk_metrics.total_exposure > self.exposure_limits["total"] * 0.9:
            logger.warning("Approaching total exposure limit")
        
        if self.daily_pnl < -self.exposure_limits["total"] * 0.1:
            logger.error("Daily loss limit exceeded")
            await self._emergency_shutdown()

    async def _emergency_shutdown(self):
        logger.critical("Emergency shutdown triggered")
        
        for position_id in list(self.positions.keys()):
            await self._close_emergency_position(position_id)

    async def _close_emergency_position(self, position_id: str):
        logger.warning(f"Force closing position: {position_id}")
        
        await self.close_position(position_id, 0)

    def get_position_size(self, opportunity: Opportunity) -> float:
        max_size = settings.max_position_size
        
        confidence_multiplier = opportunity.confidence
        profit_multiplier = min(opportunity.profit_estimate / 100, 2.0)
        
        size = max_size * confidence_multiplier * profit_multiplier * 0.1
        
        return min(size, max_size)

    def get_risk_metrics(self) -> RiskMetrics:
        return self.risk_metrics

    async def shutdown(self):
        logger.info("Risk manager shutting down")
        
        for position_id in list(self.positions.keys()):
            await self._close_emergency_position(position_id)