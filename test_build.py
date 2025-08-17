#!/usr/bin/env python3
import sys
import os
import platform
import ctypes

print("🧪 Testing M1 Build")
print("=" * 40)

# Platform info
print(f"Platform: {platform.system()}")
print(f"Processor: {platform.processor()}")
print(f"Architecture: {platform.machine()}")

# Test Rust library
try:
    rust_lib = ctypes.CDLL('./target/release/libarbitrage_engine.dylib')
    print("✅ Rust library loaded")
except Exception as e:
    print(f"❌ Rust library failed: {e}")

# Test C++ libraries
try:
    cpp_orderbook = ctypes.CDLL('./build/liborderbook.dylib')
    print("✅ C++ orderbook loaded")
except Exception as e:
    print(f"⚠️  C++ orderbook failed: {e}")

try:
    cpp_mempool = ctypes.CDLL('./build/libmempool.dylib')
    print("✅ C++ mempool loaded")
except Exception as e:
    print(f"⚠️  C++ mempool failed: {e}")

# Test Python imports
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ M1 GPU available: {gpus[0]}")
    else:
        print("⚠️  No M1 GPU detected")
except ImportError:
    print("❌ TensorFlow not installed")

try:
    import ccxt
    print("✅ CCXT installed")
except ImportError:
    print("❌ CCXT not installed")

print("\n📊 Build Summary:")
if os.path.exists('./target/release/libarbitrage_engine.dylib'):
    size = os.path.getsize('./target/release/libarbitrage_engine.dylib') / 1024 / 1024
    print(f"  Rust library: {size:.2f} MB")
if os.path.exists('./build/liborderbook.dylib'):
    size = os.path.getsize('./build/liborderbook.dylib') / 1024
    print(f"  C++ orderbook: {size:.2f} KB")

print("\n✅ Build test complete!")
