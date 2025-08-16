import asyncio
import websockets
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import logging

@dataclass
class OrderBookSnapshot:
    symbol: str
    bids: List[Tuple[float, float]]  # price, size
    asks: List[Tuple[float, float]]
    timestamp: int

@dataclass
class FundingRate:
    symbol: str
    rate: float
    timestamp: int
    next_funding: int

class CEXWebSocketClient:
    def __init__(self, exchange: str = "binance"):
        self.exchange = exchange
        self.ws = None
        self.subscriptions: Dict[str, Callable] = {}
        self.order_books: Dict[str, OrderBookSnapshot] = {}
        self.funding_rates: Dict[str, FundingRate] = {}
        self.logger = logging.getLogger(f"cex_ws_{exchange}")
        
    async def connect(self):
        """Connect to CEX websocket"""
        if self.exchange == "binance":
            uri = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
        elif self.exchange == "okx":
            uri = "wss://ws.okx.com:8443/ws/v5/public"
        else:
            raise ValueError(f"Unsupported exchange: {self.exchange}")
            
        self.ws = await websockets.connect(uri)
        self.logger.info(f"Connected to {self.exchange}")
        
    async def subscribe_orderbook(self, symbol: str):
        """Subscribe to order book updates"""
        if not self.ws:
            await self.connect()
            
        if self.exchange == "binance":
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{symbol.lower()}@bookTicker"],
                "id": 1
            }
        elif self.exchange == "okx":
            subscribe_msg = {
                "op": "subscribe",
                "args": [{"channel": "books", "instId": symbol.upper()}]
            }
            
        await self.ws.send(json.dumps(subscribe_msg))
        self.logger.info(f"Subscribed to {symbol} order book")
        
    async def subscribe_funding(self, symbol: str):
        """Subscribe to funding rate updates"""
        if not self.ws:
            await self.connect()
            
        if self.exchange == "binance":
            # Binance doesn't have real-time funding, poll REST API
            pass
        elif self.exchange == "okx":
            subscribe_msg = {
                "op": "subscribe", 
                "args": [{"channel": "funding-rate", "instId": symbol.upper()}]
            }
            await self.ws.send(json.dumps(subscribe_msg))
            
    async def listen(self):
        """Listen for incoming messages"""
        if not self.ws:
            await self.connect()
            
        try:
            async for message in self.ws:
                data = json.loads(message)
                await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("Connection closed, reconnecting...")
            await self.connect()
            await self.listen()
            
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming websocket messages"""
        if self.exchange == "binance":
            await self._handle_binance_message(data)
        elif self.exchange == "okx":
            await self._handle_okx_message(data)
            
    async def _handle_binance_message(self, data: Dict[str, Any]):
        """Handle Binance message format"""
        if "s" in data and "b" in data and "a" in data:  # Book ticker
            symbol = data["s"]
            self.order_books[symbol] = OrderBookSnapshot(
                symbol=symbol,
                bids=[(float(data["b"]), float(data["B"]))],
                asks=[(float(data["a"]), float(data["A"]))],
                timestamp=data.get("E", 0)
            )
            
    async def _handle_okx_message(self, data: Dict[str, Any]):
        """Handle OKX message format"""
        if data.get("arg", {}).get("channel") == "books":
            book_data = data["data"][0]
            symbol = data["arg"]["instId"]
            
            bids = [(float(bid[0]), float(bid[1])) for bid in book_data.get("bids", [])]
            asks = [(float(ask[0]), float(ask[1])) for ask in book_data.get("asks", [])]
            
            self.order_books[symbol] = OrderBookSnapshot(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=int(book_data["ts"])
            )
            
    async def get_best_bid_ask(self, symbol: str) -> Optional[Tuple[float, float]]:
        """Get best bid/ask for symbol"""
        if symbol in self.order_books:
            book = self.order_books[symbol]
            if book.bids and book.asks:
                return book.bids[0][0], book.asks[0][0]
        return None
        
    async def disconnect(self):
        """Disconnect websocket"""
        if self.ws:
            await self.ws.close()
            self.logger.info(f"Disconnected from {self.exchange}")
