import asyncio
import os
import sys
import time
import json
import ctypes
import multiprocessing as mp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiohttp
import websockets
import uvloop
import cupy as cp
import numpy as np
from numba import cuda

# Import our compiled modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rust_core', 'target', 'release'))
try:
    import arbitrage_core
except ImportError:
    print("Warning: Rust core not available, using Python fallback")
    arbitrage_core = None

# Load C++ execution engine
try:
    cpp_lib = ctypes.CDLL('./cpp_engine/build/libarbitrage_engine.so')
    cpp_lib.init_execution_engine.argtypes = [ctypes.c_int]
    cpp_lib.update_market_data.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    cpp_lib.scan_opportunities.argtypes = [ctypes.c_double]
    cpp_lib.scan_opportunities.restype = ctypes.c_char_p
    cpp_lib.execute_trade.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
    cpp_lib.execute_trade.restype = ctypes.c_bool
    cpp_lib.get_performance_stats.restype = ctypes.c_char_p
except Exception as e:
    print(f"Warning: C++ engine not available: {e}")
    cpp_lib = None

@dataclass
class ExchangeConfig:
    name: str
    api_url: str
    websocket_url: str
    fee_rate: float
    rate_limit: int
    supports_flash_loans: bool = False
    chain: str = "ethereum"

@dataclass
class ArbitrageOpportunity:
    strategy: str
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread: float
    volume: float
    profit_estimate: float
    confidence: float
    gas_cost: float
    net_profit: float
    timestamp: float

class UltraHighFrequencyController:
    def __init__(self, gpu_manager, exchange_manager):
        self.gpu_manager = gpu_manager
        self.exchange_manager = exchange_manager
        self.running = False
        self.opportunities_found = 0
        self.total_profit = 0.0
        self.execution_times = []
        
        # Performance optimization
        self.max_workers = min(64, (os.cpu_count() or 1) * 4)
        self.batch_size = int(os.getenv('BATCH_SIZE', 10000))
        self.scan_interval = float(os.getenv('SCAN_INTERVAL_MS', 50)) / 1000
        
        # Exchange configurations for maximum coverage
        self.exchanges = self._initialize_exchanges()
        self.price_cache = {}
        self.cache_lock = asyncio.Lock()
        
        # GPU acceleration setup
        self.gpu_enabled = cp.cuda.is_available()
        if self.gpu_enabled:
            self.gpu_device = cp.cuda.Device(0)
            self.gpu_stream = cp.cuda.Stream()
            print(f"✅ GPU acceleration enabled: {self.gpu_device}")
        
        # Initialize compiled engines
        if cpp_lib:
            cpp_lib.init_execution_engine(50000)
            print("✅ C++ execution engine initialized")
        
        if arbitrage_core:
            self.rust_engine = arbitrage_core.PyArbitrageEngine(1.0)
            print("✅ Rust core engine initialized")
        else:
            self.rust_engine = None

    def _initialize_exchanges(self) -> Dict[str, ExchangeConfig]:
        return {
            # Tier 1 CEX
            "binance": ExchangeConfig("binance", "https://api.binance.com/api/v3", "wss://stream.binance.com:9443/ws", 0.001, 1200, False),
            "coinbase": ExchangeConfig("coinbase", "https://api.exchange.coinbase.com", "wss://ws-feed.exchange.coinbase.com", 0.005, 10, False),
            "okx": ExchangeConfig("okx", "https://www.okx.com/api/v5", "wss://ws.okx.com:8443/ws/v5/public", 0.001, 600, False),
            "bybit": ExchangeConfig("bybit", "https://api.bybit.com/v5", "wss://stream.bybit.com/v5/public/spot", 0.001, 120, False),
            "huobi": ExchangeConfig("huobi", "https://api.huobi.pro", "wss://api.huobi.pro/ws", 0.002, 100, False),
            "kucoin": ExchangeConfig("kucoin", "https://api.kucoin.com/api/v1", "wss://ws-api.kucoin.com/endpoint", 0.001, 45, False),
            "gate": ExchangeConfig("gate", "https://api.gateio.ws/api/v4", "wss://api.gateio.ws/ws/v4", 0.002, 900, False),
            "mexc": ExchangeConfig("mexc", "https://api.mexc.com/api/v3", "wss://wbs.mexc.com/ws", 0.002, 1000, False),
            "bitget": ExchangeConfig("bitget", "https://api.bitget.com/api/spot/v1", "wss://ws.bitget.com/spot/v1/stream", 0.001, 600, False),
            "kraken": ExchangeConfig("kraken", "https://api.kraken.com/0/public", "wss://ws.kraken.com", 0.0026, 15, False),
            
            # Ethereum DEX
            "uniswap_v3": ExchangeConfig("uniswap_v3", "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3", "", 0.003, 1000, True, "ethereum"),
            "uniswap_v2": ExchangeConfig("uniswap_v2", "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2", "", 0.003, 1000, True, "ethereum"),
            "sushiswap": ExchangeConfig("sushiswap", "https://api.thegraph.com/subgraphs/name/sushiswap/exchange", "", 0.003, 1000, True, "ethereum"),
            "curve": ExchangeConfig("curve", "https://api.curve.fi/api/getPools/all", "", 0.0004, 500, True, "ethereum"),
            "balancer": ExchangeConfig("balancer", "https://api.thegraph.com/subgraphs/name/balancer-labs/balancer-v2", "", 0.0005, 500, True, "ethereum"),
            "1inch": ExchangeConfig("1inch", "https://api.1inch.io/v5.0/1", "", 0.003, 100, True, "ethereum"),
            
            # Polygon DEX  
            "quickswap": ExchangeConfig("quickswap", "https://api.thegraph.com/subgraphs/name/sameepsi/quickswap06", "", 0.003, 1000, False, "polygon"),
            "sushiswap_polygon": ExchangeConfig("sushiswap_polygon", "https://api.thegraph.com/subgraphs/name/sushiswap/matic-exchange", "", 0.003, 1000, False, "polygon"),
            
            # Arbitrum DEX
            "uniswap_arbitrum": ExchangeConfig("uniswap_arbitrum", "https://api.thegraph.com/subgraphs/name/ianlapham/arbitrum-minimal", "", 0.003, 1000, False, "arbitrum"),
            "sushiswap_arbitrum": ExchangeConfig("sushiswap_arbitrum", "https://api.thegraph.com/subgraphs/name/sushiswap/arbitrum-exchange", "", 0.003, 1000, False, "arbitrum"),
            
            # Optimism DEX
            "uniswap_optimism": ExchangeConfig("uniswap_optimism", "https://api.thegraph.com/subgraphs/name/ianlapham/optimism-post-regenesis", "", 0.003, 1000, False, "optimism"),
            
            # Base DEX
            "uniswap_base": ExchangeConfig("uniswap_base", "https://api.studio.thegraph.com/query/base-uniswap", "", 0.003, 1000, False, "base"),
            
            # BSC DEX
            "pancakeswap": ExchangeConfig("pancakeswap", "https://api.thegraph.com/subgraphs/name/pancakeswap/exchange", "", 0.0025, 1000, False, "bsc"),
            "biswap": ExchangeConfig("biswap", "https://api.biswap.org/api/v1", "", 0.001, 500, False, "bsc"),
            
            # Avalanche DEX
            "trader_joe": ExchangeConfig("trader_joe", "https://api.thegraph.com/subgraphs/name/traderjoe-xyz/exchange", "", 0.003, 1000, False, "avalanche"),
            "pangolin": ExchangeConfig("pangolin", "https://api.thegraph.com/subgraphs/name/pangolindex/exchange", "", 0.003, 500, False, "avalanche"),
            
            # Solana DEX
            "jupiter": ExchangeConfig("jupiter", "https://quote-api.jup.ag/v6", "", 0.003, 600, True, "solana"),
            "raydium": ExchangeConfig("raydium", "https://api.raydium.io/v2", "", 0.0025, 300, False, "solana"),
            "orca": ExchangeConfig("orca", "https://api.orca.so", "", 0.003, 200, False, "solana"),
        }

    async def initialize(self):
        print("🚀 Initializing Ultra-High-Frequency Arbitrage Controller")
        
        # Initialize GPU memory pools
        if self.gpu_enabled:
            await self._initialize_gpu_memory()
        
        # Initialize exchange connections
        await self._initialize_exchange_connections()
        
        # Warm up compiled engines
        if self.rust_engine:
            self.rust_engine.update_price("test", "BTCUSDT", 50000.0, 50001.0, 1000000.0)
            self.rust_engine.clear_old_data()
        
        if cpp_lib:
            cpp_lib.update_market_data(b"test", b"ETHUSDT", 3000.0, 3001.0, 500000.0)
        
        print("✅ Initialization complete")

    async def _initialize_gpu_memory(self):
        """Pre-allocate GPU memory for maximum performance"""
        self.gpu_prices = cp.zeros((100000, 4), dtype=cp.float32)  # bid, ask, volume, timestamp
        self.gpu_spreads = cp.zeros(100000, dtype=cp.float32)
        self.gpu_profits = cp.zeros(100000, dtype=cp.float32)
        self.gpu_indices = cp.zeros(100000, dtype=cp.int32)
        print("✅ GPU memory pools allocated")

    async def _initialize_exchange_connections(self):
        """Initialize WebSocket connections to all exchanges"""
        self.websocket_connections = {}
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5),
            connector=aiohttp.TCPConnector(limit=1000, limit_per_host=50)
        )
        
        tasks = []
        for exchange_name, config in self.exchanges.items():
            if config.websocket_url:
                task = self._connect_websocket(exchange_name, config)
                tasks.append(task)
        
        # Connect to exchanges in parallel
        await asyncio.gather(*tasks, return_exceptions=True)
        print(f"✅ Connected to {len(self.websocket_connections)} exchange WebSockets")

    async def _connect_websocket(self, exchange_name: str, config: ExchangeConfig):
        """Connect to individual exchange WebSocket"""
        try:
            if exchange_name.startswith("binance"):
                await self._connect_binance_ws(config)
            elif exchange_name.startswith("coinbase"):
                await self._connect_coinbase_ws(config)
            elif exchange_name.startswith("okx"):
                await self._connect_okx_ws(config)
            # Add more exchange-specific connections as needed
            
        except Exception as e:
            print(f"⚠️  Failed to connect to {exchange_name}: {e}")

    async def _connect_binance_ws(self, config: ExchangeConfig):
        """Binance WebSocket connection with all symbols"""
        uri = f"{config.websocket_url}/!ticker@arr"
        
        async def binance_handler():
            try:
                async with websockets.connect(uri) as ws:
                    self.websocket_connections["binance"] = ws
                    async for message in ws:
                        data = json.loads(message)
                        await self._process_binance_data(data)
            except Exception as e:
                print(f"Binance WS error: {e}")
        
        asyncio.create_task(binance_handler())

    async def _connect_coinbase_ws(self, config: ExchangeConfig):
        """Coinbase WebSocket connection"""
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "LINK-USD", "DOT-USD", "AVAX-USD"],
            "channels": ["ticker", "level2"]
        }
        
        async def coinbase_handler():
            try:
                async with websockets.connect(config.websocket_url) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    self.websocket_connections["coinbase"] = ws
                    async for message in ws:
                        data = json.loads(message)
                        await self._process_coinbase_data(data)
            except Exception as e:
                print(f"Coinbase WS error: {e}")
        
        asyncio.create_task(coinbase_handler())

    async def _connect_okx_ws(self, config: ExchangeConfig):
        """OKX WebSocket connection"""
        subscribe_msg = {
            "op": "subscribe",
            "args": [
                {"channel": "tickers", "instType": "SPOT"},
                {"channel": "books", "instType": "SPOT"}
            ]
        }
        
        async def okx_handler():
            try:
                async with websockets.connect(config.websocket_url) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    self.websocket_connections["okx"] = ws
                    async for message in ws:
                        data = json.loads(message)
                        await self._process_okx_data(data)
            except Exception as e:
                print(f"OKX WS error: {e}")
        
        asyncio.create_task(okx_handler())

    async def _process_binance_data(self, data: List[Dict]):
        """Process Binance ticker data with maximum speed"""
        timestamp = time.time()
        
        for ticker in data:
            symbol = ticker['s']
            bid = float(ticker['b'])
            ask = float(ticker['a'])
            volume = float(ticker['v'])
            
            # Update caches simultaneously
            await self._update_price_cache("binance", symbol, bid, ask, volume, timestamp)

    async def _process_coinbase_data(self, data: Dict):
        """Process Coinbase data"""
        if data.get('type') == 'ticker':
            symbol = data['product_id'].replace('-', '')
            bid = float(data['best_bid'])
            ask = float(data['best_ask'])
            volume = float(data['volume_24h'])
            timestamp = time.time()
            
            await self._update_price_cache("coinbase", symbol, bid, ask, volume, timestamp)

    async def _process_okx_data(self, data: Dict):
        """Process OKX data"""
        if 'data' in data:
            for item in data['data']:
                if 'instId' in item:
                    symbol = item['instId'].replace('-', '')
                    bid = float(item.get('bidPx', 0))
                    ask = float(item.get('askPx', 0))
                    volume = float(item.get('vol24h', 0))
                    timestamp = time.time()
                    
                    await self._update_price_cache("okx", symbol, bid, ask, volume, timestamp)

    async def _update_price_cache(self, exchange: str, symbol: str, bid: 