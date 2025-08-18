# TESTNET VERSION - Safe for testing with fake money
import os
from web3 import Web3
from dotenv import load_dotenv

load_dotenv('.env.testnet')

# Enhanced Testnet Configuration
TESTNET_CONFIG = {
    "goerli": {
        "rpc": "https://goerli.infura.io/v3/YOUR_KEY",
        "chain_id": 5,
        "explorer": "https://goerli.etherscan.io",
        "faucet": "https://goerlifaucet.com",
        "name": "Goerli Testnet"
    },
    "sepolia": {
        "rpc": "https://sepolia.infura.io/v3/YOUR_KEY", 
        "chain_id": 11155111,
        "explorer": "https://sepolia.etherscan.io",
        "faucet": "https://sepoliafaucet.com",
        "name": "Sepolia Testnet"
    },
    "mumbai": {
        "rpc": "https://polygon-mumbai.g.alchemy.com/v2/YOUR_KEY",
        "chain_id": 80001,
        "explorer": "https://mumbai.polygonscan.com",
        "faucet": "https://faucet.polygon.technology",
        "name": "Mumbai Testnet"
    }
}

# Verified Testnet Token Addresses
TESTNET_TOKENS = {
    "goerli": {
        "WETH": "0xB4FBF271143F4FBf85CF6d983c700c3d4d515E1f",
        "USDC": "0x07865c6E87B9F70255377e024ace6630C1Eaa37F",
        "DAI": "0x73967c6a0904aA032C103b4104747E88c566B1A2",
        "LINK": "0x326C977E6efc84E512bB9C30f76E30c160eD06FB"
    },
    "sepolia": {
        "WETH": "0x7b79995e5f793A07Bc00c21412e50Ecae098E7f9",
        "USDC": "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238"
    }
}

class SafeTestnetTrader:
    def __init__(self, network="goerli"):
        self.network = network
        self.config = TESTNET_CONFIG[network]
        
        # Initialize Web3 connection
        rpc_url = os.getenv('GOERLI_RPC', self.config["rpc"])
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        self.tokens = TESTNET_TOKENS.get(network, {})
        self.max_trade_size = 0.01  # 0.01 ETH maximum for safety
        
    def check_connection(self):
        """Check if connected to testnet"""
        try:
            if self.w3.is_connected():
                chain_id = self.w3.eth.chain_id
                latest_block = self.w3.eth.block_number
                print(f"✅ Connected to {self.config['name']}")
                print(f"   Chain ID: {chain_id}")
                print(f"   Latest block: {latest_block:,}")
                return True
            else:
                print(f"❌ Failed to connect to {self.config['name']}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
        
    def get_testnet_eth_instructions(self):
        """Instructions to get free testnet ETH"""
        print(f"\n🚰 GET FREE TESTNET ETH:")
        print(f"   Network: {self.config['name']}")
        print(f"   Faucet: {self.config['faucet']}")
        print(f"   Explorer: {self.config['explorer']}")
        print(f"   Need wallet address: {os.getenv('TESTNET_WALLET', 'Configure in .env.testnet')}")
        
    def check_balance(self):
        """Check testnet ETH balance"""
        wallet = os.getenv('TESTNET_WALLET')
        if not wallet:
            print("❌ No wallet address configured in .env.testnet")
            return 0
            
        try:
            balance_wei = self.w3.eth.get_balance(wallet)
            balance_eth = self.w3.from_wei(balance_wei, 'ether')
            print(f"💰 Testnet balance: {balance_eth:.4f} ETH")
            return float(balance_eth)
        except Exception as e:
            print(f"❌ Balance check failed: {e}")
            return 0
            
    def safe_testnet_trade(self, amount_eth, token_pair="WETH/USDC"):
        """Execute a safe testnet trade simulation"""
        # Safety limits
        if amount_eth > self.max_trade_size:
            print(f"⚠️ Limiting trade to {self.max_trade_size} ETH for safety")
            amount_eth = self.max_trade_size
            
        balance = self.check_balance()
        if balance < amount_eth:
            print(f"❌ Insufficient balance: {balance:.4f} ETH < {amount_eth} ETH")
            return False
            
        print(f"\n🧪 TESTNET TRADE SIMULATION:")
        print(f"   Network: {self.config['name']}")
        print(f"   Pair: {token_pair}")
        print(f"   Amount: {amount_eth} ETH")
        print(f"   Status: SIMULATION ONLY (no real execution)")
        
        # Simulate gas estimation
        estimated_gas = 150000  # Typical swap gas
        gas_price = self.w3.eth.gas_price if self.w3.is_connected() else 20000000000
        gas_cost_eth = (estimated_gas * gas_price) / 10**18
        
        print(f"   Estimated gas: {gas_cost_eth:.6f} ETH")
        print(f"   Net trade size: {amount_eth - gas_cost_eth:.6f} ETH")
        
        return {
            "success": True, 
            "testnet": True,
            "simulated": True,
            "network": self.network,
            "amount": amount_eth,
            "gas_cost": gas_cost_eth
        }

# Usage example and testing
if __name__ == "__main__":
    print("🧪 TESTNET CONFIGURATION TEST")
    print("=" * 40)
    
    trader = SafeTestnetTrader("goerli")
    
    # Test connection
    trader.check_connection()
    
    # Show faucet instructions
    trader.get_testnet_eth_instructions()
    
    # Check balance
    trader.check_balance()
    
    # Simulate a safe trade
    result = trader.safe_testnet_trade(0.001, "WETH/USDC")
    print(f"\n📊 Trade result: {result}")
    
    print("\n✅ Testnet configuration test complete!")
    print("💡 Configure .env.testnet with real keys to enable full testing")
