#!/usr/bin/env python3
"""
Full Orchestrator - Uses ALL components (Rust, C++, GPU)
"""

import os
import sys
import ctypes
import platform
import asyncio
import numpy as np
import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# Platform detection
IS_COLAB = 'COLAB_GPU' in os.environ or os.path.exists('/content/sample_data')
IS_M1 = platform.processor() == 'arm' and platform.system() == 'Darwin'
GPU_TYPE = 'A100' if IS_COLAB else ('M1' if IS_M1 else 'CPU')
LIB_EXT = 'so' if IS_COLAB or platform.system() == 'Linux' else 'dylib'

print(f"🚀 FULL SYSTEM ORCHESTRATOR")
print(f"Platform: {GPU_TYPE}")
print(f"Loading ALL components...")

# Load Rust library
try:
    rust_lib = ctypes.CDLL(f'./target/release/libarbitrage_engine.{LIB_EXT}')
    print("✅ Rust engine loaded")
    
    # Define Rust FFI functions
    rust_lib.create_engine.restype = ctypes.c_void_p
    rust_lib.destroy_engine.argtypes = [ctypes.c_void_p]
    rust_lib.update_market.argtypes = [
        ctypes.c_void_p,  # engine
        ctypes.c_uint32,  # exchange_id
        ctypes.c_uint32,  # symbol_id
        ctypes.c_double,  # bid
        ctypes.c_double,  # ask
        ctypes.c_double,  # bid_volume
        ctypes.c_double,  # ask_volume
        ctypes.c_double,  # fee
    ]
    rust_lib.find_arbitrage.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t
    ]
    rust_lib.find_arbitrage.restype = ctypes.c_size_t
    
    RUST_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Rust engine not available: {e}")
    RUST_AVAILABLE = False

# Load C++ orderbook library
try:
    cpp_orderbook = ctypes.CDLL(f'./build/liborderbook.{LIB_EXT}')
    print("✅ C++ orderbook scanner loaded")
    
    # Define C++ FFI functions
    cpp_orderbook.create_orderbook.restype = ctypes.c_void_p
    cpp_orderbook.destroy_orderbook.argtypes = [ctypes.c_void_p]
    cpp_orderbook.update_orderbook.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int
    ]
    cpp_orderbook.get_best_bid.argtypes = [ctypes.c_void_p]
    cpp_orderbook.get_best_bid.restype = ctypes.c_double
    cpp_orderbook.get_best_ask.argtypes = [ctypes.c_void_p]
    cpp_orderbook.get_best_ask.restype = ctypes.c_double
    
    CPP_ORDERBOOK_AVAILABLE = True
except Exception as e:
    print(f"⚠️  C++ orderbook not available: {e}")
    CPP_ORDERBOOK_AVAILABLE = False

# Load C++ mempool monitor
try:
    cpp_mempool = ctypes.CDLL(f'./build/libmempool.{LIB_EXT}')
    print("✅ C++ mempool monitor loaded")
    
    cpp_mempool.create_mempool.restype = ctypes.c_void_p
    cpp_mempool.destroy_mempool.argtypes = [ctypes.c_void_p]
    cpp_mempool.add_transaction.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_double
    ]
    cpp_mempool.get_mempool_size.argtypes = [ctypes.c_void_p]
    cpp_mempool.get_mempool_size.restype = ctypes.c_int
    
    CPP_MEMPOOL_AVAILABLE = True
except Exception as e:
    print(f"⚠️  C++ mempool not available: {e}")
    CPP_MEMPOOL_AVAILABLE = False

# Load GPU kernel
GPU_KERNEL_AVAILABLE = False
if IS_COLAB:
    try:
        gpu_kernel = ctypes.CDLL('./build/gpu_kernel.so')
        gpu_kernel.find_arbitrage_gpu.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_float
        ]
        GPU_KERNEL_AVAILABLE = True
        print("✅ CUDA GPU kernel loaded")
    except:
        print("⚠️  CUDA kernel not available")
elif IS_M1:
    try:
        import Metal
        import MetalPerformanceShaders as mps
        GPU_KERNEL_AVAILABLE = True
        print("✅ Metal GPU acceleration available")
    except:
        print("⚠️  Metal not available")

# Import exchange libraries
try:
    import ccxt.async_support as ccxt
except ImportError:
    os.system("pip install -q ccxt")
    import ccxt.async_support as ccxt

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FullSystemOrchestrator:
    """Orchestrator that uses ALL components"""
    
    def __init__(self):
        # Initialize all components
        self.rust_engine = rust_lib.create_engine() if RUST_AVAILABLE else None
        self.cpp_orderbooks = {}
        self.cpp_mempool = cpp_mempool.create_mempool() if CPP_MEMPOOL_AVAILABLE else None
        
        # Exchange management
        self.exchanges = {}
        self.symbol_to_id = {}
        self.exchange_to_id = {}
        
        # Statistics
        self.stats = {
            'rust_calls': 0,
            'cpp_calls': 0,
            'gpu_calls': 0,
            'opportunities_found': 0
        }
        
    async def initialize(self):
        """Initialize all systems"""
        
        print("\n" + "="*60)
        print("INITIALIZING FULL SYSTEM")
        print("="*60)
        
        # Initialize exchanges
        await self.initialize_exchanges()
        
        # Initialize orderbooks
        if CPP_ORDERBOOK_AVAILABLE:
            for exchange_name in self.exchanges:
                self.cpp_orderbooks[exchange_name] = cpp_orderbook.create_orderbook()
                print(f"✅ C++ orderbook created for {exchange_name}")
                
        print(f"\nComponents Status:")
        print(f"  Rust Engine: {'✅' if RUST_AVAILABLE else '❌'}")
        print(f"  C++ Orderbook: {'✅' if CPP_ORDERBOOK_AVAILABLE else '❌'}")
        print(f"  C++ Mempool: {'✅' if CPP_MEMPOOL_AVAILABLE else '❌'}")
        print(f"  GPU Kernel: {'✅' if GPU_KERNEL_AVAILABLE else '❌'}")
        print("="*60 + "\n")
        
    async def initialize_exchanges(self):
        """Initialize exchanges"""
        
        exchange_configs = {
            'binance': {
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET')
            },
            'coinbase': {
                'apiKey': os.getenv('COINBASE_API_KEY'),
                'secret': os.getenv('COINBASE_SECRET')
            }
        }
        
        exchange_id = 0
        for name, config in exchange_configs.items():
            if config.get('apiKey'):
                try:
                    exchange_class = getattr(ccxt, name)
                    self.exchanges[name] = exchange_class({
                        **config,
                        'enableRateLimit': True
                    })
                    await self.exchanges[name].load_markets()
                    self.exchange_to_id[name] = exchange_id
                    exchange_id += 1
                    
                    print(f"✅ {name}: {len(self.exchanges[name].markets)} markets")
                except Exception as e:
                    print(f"❌ {name}: {e}")
                    
    async def update_orderbooks(self):
        """Update orderbooks using C++"""
        
        for exchange_name, exchange in self.exchanges.items():
            # Fetch orderbook for major pairs
            symbols = ['BTC/USDT', 'ETH/USDT']
            
            for symbol in symbols:
                try:
                    orderbook = await exchange.fetch_order_book(symbol)
                    
                    if CPP_ORDERBOOK_AVAILABLE and exchange_name in self.cpp_orderbooks:
                        # Update C++ orderbook
                        cpp_ob = self.cpp_orderbooks[exchange_name]
                        
                        # Update bids
                        for bid in orderbook['bids'][:10]:
                            cpp_orderbook.update_orderbook(cpp_ob, bid[0], bid[1], 1)
                            
                        # Update asks
                        for ask in orderbook['asks'][:10]:
                            cpp_orderbook.update_orderbook(cpp_ob, ask[0], ask[1], 0)
                            
                        self.stats['cpp_calls'] += 1
                        
                        # Get best prices from C++
                        best_bid = cpp_orderbook.get_best_bid(cpp_ob)
                        best_ask = cpp_orderbook.get_best_ask(cpp_ob)
                        
                        # Update Rust engine
                        if RUST_AVAILABLE and self.rust_engine:
                            symbol_id = self.get_symbol_id(symbol)
                            exchange_id = self.exchange_to_id[exchange_name]
                            
                            rust_lib.update_market(
                                self.rust_engine,
                                exchange_id,
                                symbol_id,
                                best_bid,
                                best_ask,
                                orderbook['bids'][0][1] if orderbook['bids'] else 0,
                                orderbook['asks'][0][1] if orderbook['asks'] else 0,
                                0.001  # fee
                            )
                            self.stats['rust_calls'] += 1
                            
                except Exception as e:
                    logger.error(f"Error updating {symbol} on {exchange_name}: {e}")
                    
    def get_symbol_id(self, symbol: str) -> int:
        """Get or create symbol ID"""
        if symbol not in self.symbol_to_id:
            self.symbol_to_id[symbol] = len(self.symbol_to_id)
        return self.symbol_to_id[symbol]
        
    async def find_opportunities(self):
        """Find arbitrage using all components"""
        
        opportunities = []
        
        # Method 1: Use Rust engine
        if RUST_AVAILABLE and self.rust_engine:
            # Allocate space for results
            max_paths = 100
            ArbitragePath = ctypes.c_double * (max_paths * 10)
            paths = ArbitragePath()
            
            # Find arbitrage paths
            num_paths = rust_lib.find_arbitrage(
                self.rust_engine,
                ctypes.cast(paths, ctypes.c_void_p),
                max_paths
            )
            
            if num_paths > 0:
                self.stats['opportunities_found'] += num_paths
                print(f"🦀 Rust found {num_paths} opportunities")
                
        # Method 2: Use GPU kernel
        if GPU_KERNEL_AVAILABLE and IS_COLAB:
            # Prepare price data
            prices = []
            for exchange_name in self.exchanges:
                if CPP_ORDERBOOK_AVAILABLE and exchange_name in self.cpp_orderbooks:
                    cpp_ob = self.cpp_orderbooks[exchange_name]
                    bid = cpp_orderbook.get_best_bid(cpp_ob)
                    ask = cpp_orderbook.get_best_ask(cpp_ob)
                    prices.extend([bid, ask])
                    
            if prices:
                # Call GPU kernel
                prices_array = (ctypes.c_float * len(prices))(*prices)
                opportunities_array = (ctypes.c_int * (len(prices) * len(prices)))()
                
                gpu_kernel.find_arbitrage_gpu(
                    prices_array,
                    opportunities_array,
                    len(prices) // 2,
                    0.001  # threshold
                )
                
                self.stats['gpu_calls'] += 1
                print(f"🎮 GPU processed {len(prices)//2} markets")
                
        # Method 3: Check mempool
        if CPP_MEMPOOL_AVAILABLE and self.cpp_mempool:
            mempool_size = cpp_mempool.get_mempool_size(self.cpp_mempool)
            if mempool_size > 0:
                print(f"📊 Mempool: {mempool_size} pending transactions")
                
        return opportunities
        
    async def run(self):
        """Main execution loop using all components"""
        
        await self.initialize()
        
        print("\n🚀 Full system running - Using ALL components")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Update orderbooks (C++)
                await self.update_orderbooks()
                
                # Find opportunities (Rust + GPU)
                await self.find_opportunities()
                
                # Show statistics
                print(f"\n📊 Component Usage Stats:")
                print(f"  Rust calls: {self.stats['rust_calls']}")
                print(f"  C++ calls: {self.stats['cpp_calls']}")
                print(f"  GPU calls: {self.stats['gpu_calls']}")
                print(f"  Opportunities: {self.stats['opportunities_found']}")
                
                await asyncio.sleep(10)
                
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            
            # Cleanup
            if self.rust_engine:
                rust_lib.destroy_engine(self.rust_engine)
                
            for cpp_ob in self.cpp_orderbooks.values():
                cpp_orderbook.destroy_orderbook(cpp_ob)
                
            if self.cpp_mempool:
                cpp_mempool.destroy_mempool(self.cpp_mempool)
                
            for exchange in self.exchanges.values():
                await exchange.close()
                
            print("✅ Cleanup complete")

async def main():
    orchestrator = FullSystemOrchestrator()
    await orchestrator.run()

if __name__ == "__main__":
    asyncio.run(main())
