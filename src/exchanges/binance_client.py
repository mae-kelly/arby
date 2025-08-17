import asyncio
import ccxt.async_support as ccxt
from typing import Dict, List, Optional
from exchanges.base_exchange import BaseExchange, Ticker, OrderBook, Order
from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

class BinanceClient(BaseExchange):
    def __init__(self):
        super().__init__("Binance")
        self.client = None
        self.websocket = None
        self.subscriptions = {}

    async def initialize(self):
        self.client = ccxt.binance({
            'apiKey': settings.binance_api_key,
            'secret': settings.binance_secret_key,
            'sandbox': settings.environment != "production",
            'enableRateLimit': True,
            'rateLimit': 1200,
            'options': {
                'defaultType': 'spot'
            }
        })
        
        try:
            await self.client.load_markets()
            self.markets = self.client.markets
            self.fees = self.client.fees
            self.connected = True
            logger.info("Binance client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Binance: {e}")
            raise

    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        try:
            await self.wait_for_rate_limit()
            ticker_data = await self.client.fetch_ticker(symbol)
            
            return Ticker(
                symbol=symbol,
                bid=ticker_data['bid'],
                ask=ticker_data['ask'],
                last=ticker_data['last'],
                volume=ticker_data['baseVolume'],
                timestamp=ticker_data['timestamp']
            )
        except Exception as e:
            await self._handle_error(e, "get_ticker")
            return None

    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[OrderBook]:
        try:
            await self.wait_for_rate_limit()
            order_book_data = await self.client.fetch_order_book(symbol, limit)
            
            return OrderBook(
                bids=order_book_data['bids'],
                asks=order_book_data['asks'],
                timestamp=order_book_data['timestamp']
            )
        except Exception as e:
            await self._handle_error(e, "get_order_book")
            return None

    async def get_all_tickers(self) -> Dict[str, Ticker]:
        try:
            await self.wait_for_rate_limit()
            tickers_data = await self.client.fetch_tickers()
            
            tickers = {}
            for symbol, ticker_data in tickers_data.items():
                tickers[symbol] = Ticker(
                    symbol=symbol,
                    bid=ticker_data['bid'],
                    ask=ticker_data['ask'],
                    last=ticker_data['last'],
                    volume=ticker_data['baseVolume'],
                    timestamp=ticker_data['timestamp']
                )
            
            return tickers
        except Exception as e:
            await self._handle_error(e, "get_all_tickers")
            return {}

    async def place_order(self, symbol: str, side: str, amount: float, 
                         price: Optional[float] = None, order_type: str = "market") -> Order:
        try:
            await self.wait_for_rate_limit()
            
            order_data = await self.client.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price
            )
            
            return Order(
                id=order_data['id'],
                symbol=symbol,
                side=side,
                amount=amount,
                price=price or 0,
                status=order_data['status'],
                timestamp=order_data['timestamp']
            )
        except Exception as e:
            await self._handle_error(e, "place_order")
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            await self.wait_for_rate_limit()
            await self.client.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            await self._handle_error(e, "cancel_order")
            return False

    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        try:
            await self.wait_for_rate_limit()
            order_data = await self.client.fetch_order(order_id, symbol)
            
            return Order(
                id=order_data['id'],
                symbol=symbol,
                side=order_data['side'],
                amount=order_data['amount'],
                price=order_data['price'],
                status=order_data['status'],
                timestamp=order_data['timestamp']
            )
        except Exception as e:
            await self._handle_error(e, "get_order_status")
            return None

    async def get_balance(self, currency: Optional[str] = None) -> Dict[str, float]:
        try:
            await self.wait_for_rate_limit()
            balance_data = await self.client.fetch_balance()
            
            if currency:
                return {currency: balance_data.get(currency, {}).get('free', 0)}
            
            balances = {}
            for curr, data in balance_data.items():
                if isinstance(data, dict) and 'free' in data:
                    balances[curr] = data['free']
            
            return balances
        except Exception as e:
            await self._handle_error(e, "get_balance")
            return {}

    async def get_deposit_address(self, currency: str) -> Optional[str]:
        try:
            await self.wait_for_rate_limit()
            address_data = await self.client.fetch_deposit_address(currency)
            return address_data.get('address')
        except Exception as e:
            await self._handle_error(e, "get_deposit_address")
            return None

    async def withdraw(self, currency: str, amount: float, address: str, 
                      tag: Optional[str] = None) -> Optional[str]:
        try:
            await self.wait_for_rate_limit()
            
            params = {}
            if tag:
                params['tag'] = tag
            
            withdrawal_data = await self.client.withdraw(
                currency, amount, address, params=params
            )
            
            return withdrawal_data.get('id')
        except Exception as e:
            await self._handle_error(e, "withdraw")
            return None

    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        try:
            await self.wait_for_rate_limit()
            fees_data = await self.client.fetch_trading_fees()
            
            symbol_fees = fees_data.get(symbol, {})
            return {
                'maker': symbol_fees.get('maker', 0.001),
                'taker': symbol_fees.get('taker', 0.001)
            }
        except Exception as e:
            await self._handle_error(e, "get_trading_fees")
            return {'maker': 0.001, 'taker': 0.001}

    async def start_websocket(self, symbols: List[str]):
        try:
            for symbol in symbols:
                await self._subscribe_ticker(symbol)
                await self._subscribe_orderbook(symbol)
            
            logger.info(f"WebSocket subscriptions started for {len(symbols)} symbols")
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")

    async def _subscribe_ticker(self, symbol: str):
        pass

    async def _subscribe_orderbook(self, symbol: str):
        pass

    async def shutdown(self):
        if self.client:
            await self.client.close()
        
        if self.websocket:
            await self.websocket.close()
        
        await super().shutdown()
        logger.info("Binance client shutdown complete")