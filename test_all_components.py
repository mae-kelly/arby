#!/usr/bin/env python3
"""Test ALL components"""

import os
import ctypes
import platform

print("Testing ALL Components")
print("="*40)

# Check platform
IS_M1 = platform.processor() == 'arm' and platform.system() == 'Darwin'
LIB_EXT = 'dylib' if IS_M1 else 'so'

components = {
    'Rust Engine': f'./target/release/libarbitrage_engine.{LIB_EXT}',
    'C++ Orderbook': f'./build/liborderbook.{LIB_EXT}',
    'C++ Mempool': f'./build/libmempool.{LIB_EXT}',
    'GPU Kernel': f'./build/gpu_kernel.{LIB_EXT}'
}

for name, path in components.items():
    if os.path.exists(path):
        try:
            lib = ctypes.CDLL(path)
            print(f"✅ {name}: Loaded successfully")
        except Exception as e:
            print(f"⚠️  {name}: Found but failed to load - {e}")
    else:
        print(f"❌ {name}: Not found at {path}")

print("\nTo run with ALL components:")
print("python3 src/python/orchestrator_full.py")
