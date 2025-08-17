import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
from data.price_feed import PriceFeed
from data.market_data import MarketData
from exchanges.binance_client import BinanceClient
from exchanges.coinbase_client import CoinbaseClient
from exchanges.uniswap_client import UniswapClient
from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

@dataclass
class Opportunity:
    strategy_type: str
    profit_estimate: float
    confidence: float
    data: Dict
    timestamp: float
    expires_at: float

class OpportunityScanner:
    def __init__(self):
        self.price_feed = PriceFeed()
        self.market_data = MarketData()
        self.exchanges = {}
        self.opportunities = asyncio.Queue()
        self.running = False

    async def initialize(self):
        self.exchanges["binance"] = BinanceClient()
        self.exchanges["coinbase"] = CoinbaseClient()
        self.exchanges["uniswap"] = UniswapClient()
        
        for exchange in self.exchanges.values():
            await exchange.initialize()
        
        await self.market_data.initialize()

    async def start(self):
        self.running = True
        await asyncio.gather(
            self._scan_cross_exchange(),
            self._scan_triangular(),
            self._scan_flash_loan(),
            self._scan_liquidations(),
            self._scan_stablecoin_depegs()
        )

    async def _scan_cross_exchange(self):
        while self.running:
            try:
                symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
                
                for symbol in symbols:
                    prices = {}
                    for name, exchange in self.exchanges.items():
                        if hasattr(exchange, 'get_ticker'):
                            ticker = await exchange.get_ticker(symbol)
                            if ticker:
                                prices[name] = ticker['bid'], ticker['ask']
                    
                    opportunity = self._analyze_cross_exchange_spread(symbol, prices)
                    if opportunity:
                        await self.opportunities.put(opportunity)
                        
            except Exception as e:
                logger.error(f"Error scanning cross exchange: {e}")
            
            await asyncio.sleep(1)

    async def _scan_triangular(self):
        while self.running:
            try:
                for exchange_name, exchange in self.exchanges.items():
                    if hasattr(exchange, 'get_all_tickers'):
                        tickers = await exchange.get_all_tickers()
                        opportunity = self._analyze_triangular_arbitrage(exchange_name, tickers)
                        if opportunity:
                            await self.opportunities.put(opportunity)
                            
            except Exception as e:
                logger.error(f"Error scanning triangular: {e}")
            
            await asyncio.sleep(2)

    async def _scan_flash_loan(self):
        while self.running:
            try:
                dex_prices = await self.exchanges["uniswap"].get_pool_prices()
                cex_prices = await self.exchanges["binance"].get_all_tickers()
                
                opportunity = self._analyze_flash_loan_arbitrage(dex_prices, cex_prices)
                if opportunity:
                    await self.opportunities.put(opportunity)
                    
            except Exception as e:
                logger.error(f"Error scanning flash loan: {e}")
            
            await asyncio.sleep(0.5)

    async def _scan_liquidations(self):
        while self.running:
            try:
                positions = await self.market_data.get_lending_positions()
                
                for position in positions:
                    if position['health_factor'] < 1.05:
                        opportunity = Opportunity(
                            strategy_type="liquidation",
                            profit_estimate=position['liquidation_bonus'] * position['collateral_value'],
                            confidence=0.9,
                            data=position,
                            timestamp=asyncio.get_event_loop().time(),
                            expires_at=asyncio.get_event_loop().time() + 300
                        )
                        await self.opportunities.put(opportunity)
                        
            except Exception as e:
                logger.error(f"Error scanning liquidations: {e}")
            
            await asyncio.sleep(5)

    async def _scan_stablecoin_depegs(self):
        while self.running:
            try:
                stablecoins = ["USDC", "USDT", "DAI", "BUSD"]
                
                for coin in stablecoins:
                    price = await self.price_feed.get_price(f"{coin}/USD")
                    if price and abs(price - 1.0) > 0.01:
                        opportunity = Opportunity(
                            strategy_type="stablecoin",
                            profit_estimate=abs(price - 1.0) * 100000,
                            confidence=0.8,
                            data={"symbol": coin, "price": price, "deviation": abs(price - 1.0)},
                            timestamp=asyncio.get_event_loop().time(),
                            expires_at=asyncio.get_event_loop().time() + 600
                        )
                        await self.opportunities.put(opportunity)
                        
            except Exception as e:
                logger.error(f"Error scanning stablecoin depegs: {e}")
            
            await asyncio.sleep(10)

    def _analyze_cross_exchange_spread(self, symbol: str, prices: Dict) -> Optional[Opportunity]:
        if len(prices) < 2:
            return None
        
        best_bid = max(prices.values(), key=lambda x: x[0])
        best_ask = min(prices.values(), key=lambda x: x[1])
        
        spread = (best_bid[0] - best_ask[1]) / best_ask[1]
        
        if spread > settings.min_profit_threshold:
            return Opportunity(
                strategy_type="cross_exchange",
                profit_estimate=spread * 10000,
                confidence=0.7,
                data={"symbol": symbol, "spread": spread, "prices": prices},
                timestamp=asyncio.get_event_loop().time(),
                expires_at=asyncio.get_event_loop().time() + 60
            )
        
        return None

    def _analyze_triangular_arbitrage(self, exchange: str, tickers: Dict) -> Optional[Opportunity]:
        triangles = [
            ("BTC/USDT", "ETH/BTC", "ETH/USDT"),
            ("BNB/USDT", "ETH/BNB", "ETH/USDT")
        ]
        
        for triangle in triangles:
            if all(pair in tickers for pair in triangle):
                rate1 = tickers[triangle[0]]['bid']
                rate2 = tickers[triangle[1]]['bid'] 
                rate3 = 1 / tickers[triangle[2]]['ask']
                
                final_rate = rate1 * rate2 * rate3
                profit = (final_rate - 1.0)
                
                if profit > settings.min_profit_threshold:
                    return Opportunity(
                        strategy_type="triangular",
                        profit_estimate=profit * 10000,
                        confidence=0.6,
                        data={"triangle": triangle, "profit": profit, "exchange": exchange},
                        timestamp=asyncio.get_event_loop().time(),
                        expires_at=asyncio.get_event_loop().time() + 30
                    )
        
        return None

    def _analyze_flash_loan_arbitrage(self, dex_prices: Dict, cex_prices: Dict) -> Optional[Opportunity]:
        for symbol in ["WETH", "WBTC", "USDC"]:
            if symbol in dex_prices and f"{symbol}/USDT" in cex_prices:
                dex_price = dex_prices[symbol]
                cex_price = cex_prices[f"{symbol}/USDT"]['last']
                
                spread = abs(dex_price - cex_price) / cex_price
                
                if spread > settings.min_profit_threshold * 2:
                    return Opportunity(
                        strategy_type="flash_loan",
                        profit_estimate=spread * 50000,
                        confidence=0.8,
                        data={"symbol": symbol, "dex_price": dex_price, "cex_price": cex_price},
                        timestamp=asyncio.get_event_loop().time(),
                        expires_at=asyncio.get_event_loop().time() + 15
                    )
        
        return None

    async def get_opportunities(self) -> List[Opportunity]:
        opportunities = []
        
        while not self.opportunities.empty():
            try:
                opportunity = self.opportunities.get_nowait()
                if opportunity.expires_at > asyncio.get_event_loop().time():
                    opportunities.append(opportunity)
            except asyncio.QueueEmpty:
                break
        
        return opportunities

    async def shutdown(self):
        self.running = False
        for exchange in self.exchanges.values():
            await exchange.shutdown()