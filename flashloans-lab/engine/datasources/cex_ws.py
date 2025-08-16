"""CEX websocket connections for real-time prices"""
import asyncio
import json
from typing import Dict, List
import websockets
from datetime import datetime

class CEXWebsocket:
    def __init__(self, exchanges: List[str]):
        self.exchanges = exchanges
        self.prices = {}
        self.connections = {}
        
    async def connect(self):
        """Connect to all exchanges"""
        for exchange in self.exchanges:
            if exchange == 'binance':
                asyncio.create_task(self._connect_binance())
            elif exchange == 'okx':
                asyncio.create_task(self._connect_okx())
    
    async def _connect_binance(self):
        """Connect to Binance websocket"""
        uri = "wss://stream.binance.com:9443/ws"
        symbols = ['btcusdt', 'ethusdt', 'usdcusdt']
        
        async with websockets.connect(uri) as ws:
            # Subscribe to ticker streams
            sub_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{s}@ticker" for s in symbols],
                "id": 1
            }
            await ws.send(json.dumps(sub_msg))
            
            async for message in ws:
                data = json.loads(message)
                if 'e' in data and data['e'] == '24hrTicker':
                    symbol = data['s'].upper()
                    self.prices[f'binance:{symbol}'] = {
                        'bid': float(data['b']),
                        'ask': float(data['a']),
                        'timestamp': datetime.now()
                    }
    
    async def _connect_okx(self):
        """Connect to OKX websocket"""
        uri = "wss://ws.okx.com:8443/ws/v5/public"
        
        async with websockets.connect(uri) as ws:
            # Subscribe to tickers
            sub_msg = {
                "op": "subscribe",
                "args": [
                    {"channel": "tickers", "instId": "BTC-USDT"},
                    {"channel": "tickers", "instId": "ETH-USDT"}
                ]
            }
            await ws.send(json.dumps(sub_msg))
            
            async for message in ws:
                data = json.loads(message)
                if 'data' in data:
                    for ticker in data['data']:
                        symbol = ticker['instId']
                        self.prices[f'okx:{symbol}'] = {
                            'bid': float(ticker['bidPx']),
                            'ask': float(ticker['askPx']),
                            'timestamp': datetime.now()
                        }
    
    def get_latest_prices(self) -> Dict:
        """Get latest prices from all exchanges"""
        return self.prices.copy()
