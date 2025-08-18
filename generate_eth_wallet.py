#!/usr/bin/env python3
"""
Generate Ethereum Testnet Wallet
This creates a proper Ethereum wallet for testnet use
"""

def generate_ethereum_wallet():
    try:
        from eth_account import Account
        import secrets
        
        # Generate secure random private key
        private_key = secrets.token_hex(32)
        account = Account.from_key(private_key)
        
        print("🔐 NEW ETHEREUM TESTNET WALLET GENERATED")
        print("=" * 50)
        print(f"Private Key: {private_key}")
        print(f"Wallet Address: {account.address}")
        print()
        print("✅ This is a proper Ethereum wallet format!")
        print("⚠️ TESTNET ONLY - Save these credentials safely")
        
        return private_key, account.address
        
    except ImportError:
        print("❌ Missing eth_account library")
        print("Install with: pip install eth-account")
        return None, None

def create_complete_env_file():
    """Create complete .env.testnet with everything configured"""
    
    print("Generating Ethereum wallet...")
    private_key, wallet_address = generate_ethereum_wallet()
    
    if not private_key:
        return
    
    env_content = f"""# COMPLETE TESTNET TRADING CONFIGURATION
# Everything you need for safe testnet trading!

# OKX DEMO/SANDBOX API
OKX_API_KEY=8a760df1-4a2d-471b-ba42-d16893614dab
OKX_SECRET_KEY=C9F3FC89A6A30226E11DFFD098C7CF3D
OKX_PASSPHRASE=Shamrock1!
OKX_SANDBOX=true
OKX_BASE_URL=https://www.okx.com

# TESTNET RPC ENDPOINTS (Working endpoints)
GOERLI_RPC=https://goerli.infura.io/v3/2e1c7909e5e4488e99010fabd3590a79
SEPOLIA_RPC=https://sepolia.infura.io/v3/2e1c7909e5e4488e99010fabd3590a79
MUMBAI_RPC=https://polygon-mumbai.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX

# ETHEREUM TESTNET WALLET (GENERATED AUTOMATICALLY)
TESTNET_PRIVATE_KEY={private_key}
TESTNET_WALLET={wallet_address}

# TESTNET CONTRACT ADDRESSES
AAVE_TESTNET_POOL=0x6Ae43d3271ff6888e7Fc43Fd7321a503ff738951
UNISWAP_TESTNET_ROUTER=0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D

# SAFE LIMITS FOR TESTING
MAX_TRADE_SIZE=0.1  # ETH
MIN_PROFIT_THRESHOLD=0.001  # ETH
ENABLE_REAL_TRANSACTIONS=false  # Start with simulation

# NOTIFICATIONS
DISCORD_WEBHOOK=https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3
ENABLE_ALERTS=true

# ADDITIONAL SAFETY SETTINGS
MAX_DAILY_LOSS=50  # USD
STOP_LOSS_PERCENTAGE=5  # %
"""
    
    with open('.env.testnet', 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ Created complete .env.testnet file!")
    print(f"📝 Your wallet address: {wallet_address}")
    print(f"🚰 Get free testnet ETH from:")
    print(f"   • Goerli: https://goerlifaucet.com")
    print(f"   • Sepolia: https://sepoliafaucet.com")
    print(f"   • Mumbai: https://faucet.polygon.technology")

def alternative_demo_wallet():
    """Show how to use a well-known demo wallet"""
    print("\n🔄 ALTERNATIVE: Use Demo Wallet")
    print("=" * 40)
    print("For quick testing, you can use this well-known demo wallet:")
    print("Private Key: ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80")
    print("Address: 0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266")
    print()
    print("⚠️ WARNING: This is a PUBLIC demo wallet!")
    print("• Everyone knows this private key")
    print("• Only use for initial testing")
    print("• Generate your own wallet for real testing")

if __name__ == "__main__":
    print("🧪 ETHEREUM TESTNET WALLET GENERATOR")
    print("=" * 45)
    
    choice = input("Choose option:\n1. Generate NEW wallet (recommended)\n2. Show demo wallet info\n3. Exit\nChoice (1-3): ")
    
    if choice == "1":
        create_complete_env_file()
    elif choice == "2":
        alternative_demo_wallet()
    else:
        print("👋 Goodbye!")