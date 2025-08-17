#!/usr/bin/env python3
"""
Full system runner - uses ALL components
"""

import subprocess
import sys
import os
import ctypes
import platform

def check_components():
    """Check which components are available"""
    components = {}
    
    # Check platform
    is_m1 = platform.processor() == 'arm' and platform.system() == 'Darwin'
    lib_ext = 'dylib' if is_m1 else 'so'
    
    # Check Rust engine
    rust_path = f'./target/release/libarbitrage_engine.{lib_ext}'
    try:
        ctypes.CDLL(rust_path)
        components['rust'] = True
        print("✅ Rust engine available")
    except:
        components['rust'] = False
        print("⚠️  Rust engine not available")
    
    # Check C++ components
    cpp_orderbook_path = f'./build/liborderbook.{lib_ext}'
    try:
        ctypes.CDLL(cpp_orderbook_path)
        components['cpp_orderbook'] = True
        print("✅ C++ orderbook available")
    except:
        components['cpp_orderbook'] = False
        print("⚠️  C++ orderbook not available")
    
    cpp_mempool_path = f'./build/libmempool.{lib_ext}'
    try:
        ctypes.CDLL(cpp_mempool_path)
        components['cpp_mempool'] = True
        print("✅ C++ mempool available")
    except:
        components['cpp_mempool'] = False
        print("⚠️  C++ mempool not available")
    
    # Check GPU
    if is_m1:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            components['gpu'] = bool(gpus)
            print(f"✅ M1 GPU available: {gpus}" if gpus else "⚠️  M1 GPU not available")
        except:
            components['gpu'] = False
            print("⚠️  M1 GPU not available")
    else:
        try:
            import cupy
            components['gpu'] = True
            print("✅ CUDA GPU available")
        except:
            components['gpu'] = False
            print("⚠️  CUDA GPU not available")
    
    return components

def run_full_system():
    """Run the full system with all components"""
    
    print("🔥 FULL SYSTEM MODE - Maximum Performance")
    print("=" * 60)
    
    # Check components
    components = check_components()
    
    total_components = sum(components.values())
    print(f"\n📊 Available components: {total_components}/4")
    
    if total_components == 0:
        print("❌ No optimized components available")
        print("Run: python3 setup.py install  # to build components")
        print("Or use: python3 run_simple_bot.py  # for basic version")
        return
    
    # Run full orchestrator
    script = 'src/python/orchestrator_full.py'
    
    print(f"\n🚀 Starting full system...")
    print("This will use ALL available components for maximum performance")
    
    try:
        subprocess.run([sys.executable, script], check=True)
    except KeyboardInterrupt:
        print("\n👋 Full system stopped by user")
    except FileNotFoundError:
        print(f"❌ Script not found: {script}")
    except Exception as e:
        print(f"❌ Error running full system: {e}")

if __name__ == "__main__":
    run_full_system()