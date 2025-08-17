#!/usr/bin/env python3
"""
Simple bot runner - easiest way to get started
"""

import subprocess
import sys

def run_simple_bot():
    """Run the simple arbitrage bot"""
    
    print("🤖 Starting Simple Arbitrage Bot")
    print("=" * 40)
    
    # Check if .env exists
    import os
    if not os.path.exists('.env'):
        print("❌ No .env file found!")
        print("Create .env file with your API keys first")
        print("See .env template for required keys")
        return
    
    # Run the simple bot
    try:
        subprocess.run([sys.executable, 'src/python/simple_bot.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except FileNotFoundError:
        print("❌ Bot file not found. Make sure you're in the project directory")
    except Exception as e:
        print(f"❌ Error running bot: {e}")

if __name__ == "__main__":
    run_simple_bot()