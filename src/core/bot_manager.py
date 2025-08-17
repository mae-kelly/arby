import asyncio
from typing import Dict, List
from core.opportunity_scanner import OpportunityScanner
from core.risk_manager import RiskManager
from core.portfolio_manager import PortfolioManager
from strategies.flash_loan_arbitrage import FlashLoanArbitrage
from strategies.cross_exchange_arbitrage import CrossExchangeArbitrage
from strategies.triangular_arbitrage import TriangularArbitrage
from strategies.liquidation_hunter import LiquidationHunter
from strategies.stablecoin_arbitrage import StablecoinArbitrage
from data.price_feed import PriceFeed
from utils.logger import get_logger
from config.settings import settings, STRATEGY_CONFIGS

logger = get_logger(__name__)

class BotManager:
    def __init__(self):
        self.scanner = OpportunityScanner()
        self.risk_manager = RiskManager()
        self.portfolio_manager = PortfolioManager()
        self.price_feed = PriceFeed()
        self.strategies = {}
        self.running = False
        self.tasks = []

    async def initialize(self):
        await self.price_feed.initialize()
        await self.portfolio_manager.initialize()
        await self.risk_manager.initialize()
        await self.scanner.initialize()
        
        self._load_strategies()
        
        for strategy in self.strategies.values():
            await strategy.initialize()

    def _load_strategies(self):
        if STRATEGY_CONFIGS["flash_loan"]["enabled"]:
            self.strategies["flash_loan"] = FlashLoanArbitrage(
                self.risk_manager, self.portfolio_manager
            )
        
        if STRATEGY_CONFIGS["cross_exchange"]["enabled"]:
            self.strategies["cross_exchange"] = CrossExchangeArbitrage(
                self.risk_manager, self.portfolio_manager
            )
        
        if STRATEGY_CONFIGS["triangular"]["enabled"]:
            self.strategies["triangular"] = TriangularArbitrage(
                self.risk_manager, self.portfolio_manager
            )
        
        if STRATEGY_CONFIGS["liquidation"]["enabled"]:
            self.strategies["liquidation"] = LiquidationHunter(
                self.risk_manager, self.portfolio_manager
            )
        
        self.strategies["stablecoin"] = StablecoinArbitrage(
            self.risk_manager, self.portfolio_manager
        )

    async def start(self):
        self.running = True
        
        self.tasks.append(asyncio.create_task(self.price_feed.start()))
        self.tasks.append(asyncio.create_task(self.scanner.start()))
        self.tasks.append(asyncio.create_task(self._opportunity_handler()))
        self.tasks.append(asyncio.create_task(self._health_monitor()))
        
        for strategy in self.strategies.values():
            self.tasks.append(asyncio.create_task(strategy.start()))

    async def _opportunity_handler(self):
        while self.running:
            try:
                opportunities = await self.scanner.get_opportunities()
                
                for opportunity in opportunities:
                    if not self.risk_manager.check_opportunity(opportunity):
                        continue
                    
                    strategy = self.strategies.get(opportunity.strategy_type)
                    if strategy and strategy.is_ready():
                        await strategy.execute_opportunity(opportunity)
                        
            except Exception as e:
                logger.error(f"Error in opportunity handler: {e}")
            
            await asyncio.sleep(0.1)

    async def _health_monitor(self):
        while self.running:
            try:
                await self.portfolio_manager.update_positions()
                await self.risk_manager.check_exposure()
                
                for strategy in self.strategies.values():
                    await strategy.health_check()
                    
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
            
            await asyncio.sleep(5)

    async def shutdown(self):
        self.running = False
        
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        for strategy in self.strategies.values():
            await strategy.shutdown()
        
        await self.scanner.shutdown()
        await self.price_feed.shutdown()
        await self.portfolio_manager.shutdown()
        await self.risk_manager.shutdown()