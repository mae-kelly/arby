#!/usr/bin/env python3
"""
Platform-optimized bot runner
"""

import subprocess
import sys
import platform
import os

def detect_platform():
    """Detect the current platform"""
    if 'COLAB_GPU' in os.environ or os.path.exists('/content/sample_data'):
        return 'COLAB'
    elif platform.processor() == 'arm' and platform.system() == 'Darwin':
        return 'M1'
    else:
        return 'STANDARD'

def run_optimized_bot():
    """Run the platform-optimized bot"""
    
    platform_type = detect_platform()
    
    print(f"🚀 Starting Optimized Bot for {platform_type}")
    print("=" * 50)
    
    # Select appropriate script
    if platform_type == 'M1':
        script = 'src/orchestrator_m1_fixed.py'
        print("🍎 Using M1 Metal GPU acceleration")
    elif platform_type == 'COLAB':
        script = 'src/orchestrator_unified.py'
        print("🔥 Using A100 CUDA acceleration")
    else:
        script = 'src/orchestrator_unified.py'
        print("💻 Using CPU optimization")
    
    # Run the bot
    try:
        subprocess.run([sys.executable, script], check=True)
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except FileNotFoundError:
        print(f"❌ Script not found: {script}")
        print("Make sure you're in the project directory")
    except Exception as e:
        print(f"❌ Error running bot: {e}")

if __name__ == "__main__":
    run_optimized_bot()