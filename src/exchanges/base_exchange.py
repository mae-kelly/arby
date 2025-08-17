import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class OrderBook:
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    timestamp: float

@dataclass
class Ticker:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: float

@dataclass
class Order:
    id: str
    symbol: str
    side: str
    amount: float
    price: float
    status: str
    timestamp: float

class BaseExchange(ABC):
    def __init__(self, name: str):
        self.name = name
        self.connected = False
        self.rate_limiter = None
        self.fees = {}
        self.markets = {}

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        pass

    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[OrderBook]:
        pass

    @abstractmethod
    async def get_all_tickers(self) -> Dict[str, Ticker]:
        pass

    @abstractmethod
    async def place_order(self, symbol: str, side: str, amount: float, 
                         price: Optional[float] = None, order_type: str = "market") -> Order:
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        pass

    @abstractmethod
    async def get_balance(self, currency: Optional[str] = None) -> Dict[str, float]:
        pass

    async def place_market_order(self, symbol: str, side: str, amount: float) -> Order:
        return await self.place_order(symbol, side, amount, None, "market")

    async def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Order:
        return await self.place_order(symbol, side, amount, price, "limit")

    def get_trading_fee(self, symbol: str, side: str) -> float:
        return self.fees.get(symbol, {}).get(side, 0.001)

    def calculate_fee(self, symbol: str, side: str, amount: float, price: float) -> float:
        fee_rate = self.get_trading_fee(symbol, side)
        return amount * price * fee_rate

    async def get_best_bid_ask(self, symbol: str) -> Optional[Tuple[float, float]]:
        ticker = await self.get_ticker(symbol)
        if ticker:
            return ticker.bid, ticker.ask
        return None

    async def get_mid_price(self, symbol: str) -> Optional[float]:
        bid_ask = await self.get_best_bid_ask(symbol)
        if bid_ask:
            return (bid_ask[0] + bid_ask[1]) / 2
        return None

    def is_connected(self) -> bool:
        return self.connected

    async def test_connection(self) -> bool:
        try:
            await self.get_balance()
            return True
        except:
            return False

    async def wait_for_rate_limit(self):
        if self.rate_limiter:
            await self.rate_limiter.wait()

    def _validate_symbol(self, symbol: str) -> bool:
        return symbol in self.markets

    def _validate_side(self, side: str) -> bool:
        return side.lower() in ['buy', 'sell']

    def _validate_amount(self, amount: float) -> bool:
        return amount > 0

    def _validate_price(self, price: Optional[float]) -> bool:
        return price is None or price > 0

    async def _handle_error(self, error: Exception, operation: str):
        logger.error(f"{self.name} {operation} error: {error}")
        
        if "rate limit" in str(error).lower():
            await asyncio.sleep(1)
        elif "insufficient" in str(error).lower():
            logger.warning(f"{self.name} insufficient balance")
        elif "invalid" in str(error).lower():
            logger.warning(f"{self.name} invalid request: {error}")

    async def shutdown(self):
        self.connected = False
        logger.info(f"{self.name} exchange connection closed")

    def __str__(self):
        return f"{self.name}Exchange(connected={self.connected})"