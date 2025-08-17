#!/usr/bin/env python3
"""Test if components are working"""

import os
import sys

print("Testing build components...\n")

# Test Python
print("✅ Python:", sys.version.split()[0])

# Test if ccxt can be imported
try:
    import ccxt
    print("✅ ccxt installed")
except:
    print("⚠️  ccxt not installed - run: pip3 install ccxt")

# Check for .env
if os.path.exists('.env'):
    print("✅ .env file exists")
else:
    print("⚠️  No .env file - copy from .env.example")

# Check for build directory
if os.path.exists('build'):
    print("✅ build directory exists")
    
    # Check for compiled libraries
    libs = ['libscanner.dylib', 'libarbitrage_engine.dylib', 'scanner']
    for lib in libs:
        if os.path.exists(f'build/{lib}'):
            print(f"   ✅ {lib} found")

# Check for Rust target
if os.path.exists('target/release'):
    print("✅ Rust build directory exists")

print("\n" + "="*50)
print("To run the bot:")
print("1. Add API keys to .env file")
print("2. Run: python3 src/python/simple_bot.py")
print("="*50)
