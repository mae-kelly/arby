import os
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from web3 import Web3
from eth_account import Account
from decimal import Decimal
import json
import time
from typing import Dict, List, Tuple, Optional
import ccxt.async_support as ccxt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
from collections import defaultdict
import redis
import msgpack
import zmq.asyncio
import uvloop
from cytoolz import pipe, partial
import pyarrow as pa
import pyarrow.plasma as plasma

# GPU imports
try:
    import cupy as cp
    import cudf
    from numba import cuda, jit
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU libraries not available, falling back to CPU")

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Constants
MIN_PROFIT_USD = float(os.getenv('MIN_PROFIT_USD', '100'))
MAX_GAS_PRICE = int(os.getenv('MAX_GAS_PRICE_GWEI', '500'))
DEPLOYMENT = os.getenv('DEPLOYMENT', 'local')

@dataclass
class ArbitrageOpportunity:
    path: List[str]
    exchanges: List[str]
    profit_usd: float
    gas_cost: float
    confidence: float
    timestamp: float

class HyperOptimizedOrchestrator:
    def __init__(self):
        self.setup_gpu()
        self.setup_connections()
        self.setup_exchanges()
        self.setup_web3()
        self.opportunities = asyncio.Queue(maxsize=10000)
        self.executed = set()
        self.plasma_client = plasma.connect("/tmp/plasma")
        
    def setup_gpu(self):
        if GPU_AVAILABLE and DEPLOYMENT == 'colab':
            self.device = cuda.get_current_device()
            self.gpu_memory_pool = cp.get_default_memory_pool()
            self.pinned_memory_pool = cp.get_default_pinned_memory_pool()
        elif GPU_AVAILABLE:
            # M1 GPU setup
            os.environ['METAL_DEVICE_WRAPPER'] = '1'
            
    def setup_connections(self):
        self.redis = redis.Redis(decode_responses=False)
        self.zmq_context = zmq.asyncio.Context()
        self.publisher = self.zmq_context.socket(zmq.PUB)
        self.publisher.bind("tcp://127.0.0.1:5555")
        
    def setup_exchanges(self):
        self.exchanges = {}
        exchange_configs = {
            'binance': {'apiKey': os.getenv('BINANCE_API_KEY'), 
                       'secret': os.getenv('BINANCE_SECRET')},
            'coinbase': {'apiKey': os.getenv('COINBASE_API_KEY'),
                        'secret': os.getenv('COINBASE_SECRET')},
            'kraken': {'apiKey': os.getenv('KRAKEN_API_KEY'),
                      'secret': os.getenv('KRAKEN_SECRET')},
            'bybit': {'apiKey': os.getenv('BYBIT_API_KEY'),
                     'secret': os.getenv('BYBIT_SECRET')}
        }
        
        for name, config in exchange_configs.items():
            try:
                exchange_class = getattr(ccxt, name)
                self.exchanges[name] = exchange_class({
                    **config,
                    'enableRateLimit': False,
                    'options': {'defaultType': 'spot'}
                })
            except Exception as e:
                print(f"Failed to initialize {name}: {e}")
                
    def setup_web3(self):
        rpcs = os.getenv('RPC_ENDPOINTS_ETH', '').split(',')
        self.w3_providers = [Web3(Web3.HTTPProvider(rpc)) for rpc in rpcs if rpc]
        self.w3 = self.w3_providers[0] if self.w3_providers else None
        
        if self.w3:
            self.account = Account.from_key(os.getenv('PRIVATE_KEY', ''))
            self.contract_address = os.getenv('FLASHLOAN_CONTRACT', '')
            
    @cuda.jit
    def gpu_calculate_profits(prices, fees, amounts):
        """CUDA kernel for parallel profit calculation"""
        i = cuda.grid(1)
        if i < prices.shape[0]:
            profit = amounts[i]
            for j in range(prices.shape[1]):
                if prices[i, j] > 0:
                    profit = profit * (1 - fees[i, j]) * prices[i, j]
            amounts[i] = profit - amounts[i]
            
    async def fetch_all_orderbooks(self):
        """Fetch orderbooks from all exchanges in parallel"""
        tasks = []
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'MATIC/USDT']
        
        for exchange_name, exchange in self.exchanges.items():
            for symbol in symbols:
                if symbol in exchange.markets:
                    tasks.append(self.fetch_orderbook(exchange, symbol, exchange_name))
                    
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]
    
    async def fetch_orderbook(self, exchange, symbol, name):
        try:
            orderbook = await exchange.fetch_order_book(symbol, limit=20)
            return {
                'exchange': name,
                'symbol': symbol,
                'bids': orderbook['bids'][:10],
                'asks': orderbook['asks'][:10],
                'timestamp': time.time()
            }
        except Exception as e:
            return None
            
    def find_arbitrage_paths(self, orderbooks):
        """Find arbitrage opportunities using GPU acceleration"""
        if not orderbooks:
            return []
            
        opportunities = []
        symbols = defaultdict(list)
        
        for ob in orderbooks:
            if ob:
                symbols[ob['symbol']].append(ob)
                
        for symbol, obs in symbols.items():
            if len(obs) < 2:
                continue
                
            # Cross-exchange arbitrage
            for i in range(len(obs)):
                for j in range(i+1, len(obs)):
                    opp = self.check_cross_exchange(obs[i], obs[j])
                    if opp:
                        opportunities.append(opp)
                        
        # Triangular arbitrage within each exchange
        for exchange_name, exchange in self.exchanges.items():
            exchange_obs = [ob for ob in orderbooks if ob and ob['exchange'] == exchange_name]
            tri_opps = self.find_triangular_arbitrage(exchange_obs)
            opportunities.extend(tri_opps)
            
        return opportunities
    
    def check_cross_exchange(self, ob1, ob2):
        """Check for cross-exchange arbitrage"""
        if not ob1['asks'] or not ob2['bids']:
            return None
            
        buy_price = ob1['asks'][0][0]
        sell_price = ob2['bids'][0][0]
        
        if sell_price > buy_price * 1.002:  # 0.2% profit threshold
            quantity = min(ob1['asks'][0][1], ob2['bids'][0][1])
            profit = (sell_price - buy_price) * quantity
            
            if profit > MIN_PROFIT_USD:
                return ArbitrageOpportunity(
                    path=[ob1['symbol']],
                    exchanges=[ob1['exchange'], ob2['exchange']],
                    profit_usd=profit,
                    gas_cost=0,  # CEX no gas
                    confidence=0.95,
                    timestamp=time.time()
                )
        return None
        
    def find_triangular_arbitrage(self, orderbooks):
        """Find triangular arbitrage opportunities"""
        opportunities = []
        
        # Group by base currency
        by_base = defaultdict(list)
        for ob in orderbooks:
            if ob and '/' in ob['symbol']:
                base, quote = ob['symbol'].split('/')
                by_base[base].append(ob)
                
        # Check triangular paths
        for base in by_base:
            if len(by_base[base]) >= 2:
                # Simple triangular: BASE/USDT -> BASE/BTC -> BTC/USDT
                # This is simplified, real implementation would be more complex
                pass
                
        return opportunities
        
    async def execute_opportunity(self, opp: ArbitrageOpportunity):
        """Execute arbitrage opportunity"""
        if opp.profit_usd < MIN_PROFIT_USD:
            return False
            
        # Check if already executed
        opp_hash = hash((tuple(opp.path), tuple(opp.exchanges), opp.timestamp))
        if opp_hash in self.executed:
            return False
            
        self.executed.add(opp_hash)
        
        try:
            if len(opp.exchanges) == 2 and all(e in self.exchanges for e in opp.exchanges):
                # Cross-exchange arbitrage
                return await self.execute_cross_exchange(opp)
            elif self.w3 and opp.gas_cost > 0:
                # On-chain arbitrage
                return await self.execute_onchain(opp)
        except Exception as e:
            print(f"Execution failed: {e}")
            return False
            
    async def execute_cross_exchange(self, opp: ArbitrageOpportunity):
        """Execute cross-exchange arbitrage"""
        symbol = opp.path[0]
        buy_exchange = self.exchanges[opp.exchanges[0]]
        sell_exchange = self.exchanges[opp.exchanges[1]]
        
        # Calculate optimal amount
        amount = min(
            await self.get_balance(buy_exchange, 'USDT'),
            opp.profit_usd * 10  # Max 10x the profit as capital
        )
        
        # Place orders simultaneously
        buy_task = buy_exchange.create_market_buy_order(symbol, amount)
        sell_task = sell_exchange.create_market_sell_order(symbol, amount)
        
        results = await asyncio.gather(buy_task, sell_task, return_exceptions=True)
        
        if all(not isinstance(r, Exception) for r in results):
            print(f"Executed arbitrage: {opp.profit_usd} USD profit")
            return True
        return False
        
    async def execute_onchain(self, opp: ArbitrageOpportunity):
        """Execute on-chain arbitrage via smart contract"""
        if not self.w3 or not self.contract_address:
            return False
            
        # Build transaction
        tx = {
            'from': self.account.address,
            'to': self.contract_address,
            'gas': 3000000,
            'gasPrice': self.w3.toWei(MAX_GAS_PRICE, 'gwei'),
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'data': self.encode_arbitrage_data(opp)
        }
        
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        
        print(f"Submitted on-chain arbitrage: {tx_hash.hex()}")
        return True
        
    def encode_arbitrage_data(self, opp: ArbitrageOpportunity):
        """Encode arbitrage path for smart contract"""
        # This would encode the actual path data
        return '0x'
        
    async def get_balance(self, exchange, currency):
        """Get balance on exchange"""
        try:
            balance = await exchange.fetch_balance()
            return balance.get(currency, {}).get('free', 0)
        except:
            return 0
            
    async def monitor_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Fetch all orderbooks
                orderbooks = await self.fetch_all_orderbooks()
                
                # Find arbitrage opportunities
                opportunities = self.find_arbitrage_paths(orderbooks)
                
                # Queue opportunities for execution
                for opp in opportunities:
                    if opp.profit_usd > MIN_PROFIT_USD:
                        await self.opportunities.put(opp)
                        
                # Publish to subscribers
                if opportunities:
                    data = msgpack.packb([o.__dict__ for o in opportunities])
                    await self.publisher.send(data)
                    
            except Exception as e:
                print(f"Monitor error: {e}")
                
            await asyncio.sleep(0.01)  # 10ms interval
            
    async def execution_loop(self):
        """Execute queued opportunities"""
        while True:
            try:
                opp = await asyncio.wait_for(self.opportunities.get(), timeout=0.1)
                asyncio.create_task(self.execute_opportunity(opp))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Execution error: {e}")
                
    async def run(self):
        """Run the orchestrator"""
        print(f"Starting orchestrator in {DEPLOYMENT} mode")
        print(f"GPU Available: {GPU_AVAILABLE}")
        
        # Start monitoring and execution loops
        await asyncio.gather(
            self.monitor_loop(),
            self.execution_loop(),
            return_exceptions=True
        )

def main():
    # Use uvloop for better performance
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    orchestrator = HyperOptimizedOrchestrator()
    asyncio.run(orchestrator.run())

if __name__ == "__main__":
    main()
# Add MEV Hunter integration
from mev_hunter import AdvancedMEVHunter

class EnhancedOrchestrator(HyperOptimizedOrchestrator):
    def __init__(self):
        super().__init__()
        self.mev_hunter = AdvancedMEVHunter()
        
    async def run_with_mev(self):
        """Run orchestrator with MEV hunting"""
        
        # Start MEV hunter in parallel
        mev_task = asyncio.create_task(self.mev_hunter.start_hunting())
        
        # Start original orchestrator
        main_task = asyncio.create_task(super().run())
        
        # Run both together
        await asyncio.gather(mev_task, main_task, return_exceptions=True)

# Replace main function
def main():
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    orchestrator = EnhancedOrchestrator()
    asyncio.run(orchestrator.run_with_mev())
