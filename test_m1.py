#!/usr/bin/env python3
"""
Test script for M1 Mac
"""

import os
import sys
import platform

print("Testing M1 Mac Setup")
print("=" * 40)

# Check platform
print(f"Platform: {platform.system()}")
print(f"Processor: {platform.processor()}")
print(f"Python: {sys.version}")

# Check GPU
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Metal GPU detected: {gpus}")
    else:
        print("⚠️  No Metal GPU detected")
except ImportError:
    print("❌ TensorFlow not installed")
    print("Run: pip install tensorflow-macos tensorflow-metal")

# Check dependencies
deps = ['ccxt', 'numpy', 'pandas', 'aiohttp', 'websockets', 'python-dotenv']
for dep in deps:
    try:
        __import__(dep)
        print(f"✅ {dep} installed")
    except ImportError:
        print(f"❌ {dep} not installed")

# Check .env
if os.path.exists('.env'):
    print("✅ .env file exists")
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for API keys
    if os.getenv('BINANCE_API_KEY'):
        print("✅ Binance API configured")
    if os.getenv('COINBASE_API_KEY'):
        print("✅ Coinbase API configured")
else:
    print("❌ No .env file found")

print("\nTo run the bot:")
print("python3 src/orchestrator_unified.py")
