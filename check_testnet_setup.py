#!/usr/bin/env python3
"""Quick testnet setup verification"""

import os
import sys

def check_setup():
    print("🧪 TESTNET SETUP VERIFICATION")
    print("=" * 40)
    
    checks = []
    
    # Check .env.testnet
    if os.path.exists('.env.testnet'):
        checks.append("✅ .env.testnet file exists")
        
        with open('.env.testnet', 'r') as f:
            content = f.read()
            if 'YOUR_' in content:
                checks.append("⚠️  .env.testnet needs configuration")
            else:
                checks.append("✅ .env.testnet appears configured")
    else:
        checks.append("❌ .env.testnet file missing")
    
    # Check testnet files
    testnet_files = [
        'testnet_config.py',
        'testnet_monitor.py', 
        'start_testnet.sh',
        'TESTNET_GUIDE.md'
    ]
    
    for file in testnet_files:
        if os.path.exists(file):
            checks.append(f"✅ {file} created")
        else:
            checks.append(f"❌ {file} missing")
    
    # Check Python dependencies
    try:
        import web3
        checks.append("✅ web3 library available")
    except ImportError:
        checks.append("❌ web3 library missing (run: pip install web3)")
    
    try:
        import aiohttp
        checks.append("✅ aiohttp library available")
    except ImportError:
        checks.append("❌ aiohttp library missing (run: pip install aiohttp)")
    
    # Print results
    for check in checks:
        print(check)
    
    print("\n🚀 NEXT STEPS:")
    print("1. Configure .env.testnet with your settings")
    print("2. Get free testnet ETH from faucets")
    print("3. Run: ./start_testnet.sh")
    print("\n📖 Read TESTNET_GUIDE.md for detailed instructions")

if __name__ == "__main__":
    check_setup()
