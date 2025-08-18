import os
from web3 import Web3

# Testnet Configuration
TESTNET_CONFIG = {
    "goerli": {
        "rpc": "https://goerli.infura.io/v3/YOUR_KEY",
        "chain_id": 5,
        "explorer": "https://goerli.etherscan.io",
        "faucet": "https://goerlifaucet.com"
    },
    "sepolia": {
        "rpc": "https://sepolia.infura.io/v3/YOUR_KEY", 
        "chain_id": 11155111,
        "explorer": "https://sepolia.etherscan.io",
        "faucet": "https://sepoliafaucet.com"
    },
    "mumbai": {
        "rpc": "https://polygon-mumbai.g.alchemy.com/v2/YOUR_KEY",
        "chain_id": 80001,
        "explorer": "https://mumbai.polygonscan.com",
        "faucet": "https://faucet.polygon.technology"
    }
}

# Testnet Token Addresses (these change, verify before use)
TESTNET_TOKENS = {
    "goerli": {
        "WETH": "0xB4FBF271143F4FBf85CF6d983c700c3d4d515E1f",
        "USDC": "0x07865c6E87B9F70255377e024ace6630C1Eaa37F",
        "DAI": "0x73967c6a0904aA032C103b4104747E88c566B1A2"
    },
    "sepolia": {
        "WETH": "0x7b79995e5f793A07Bc00c21412e50Ecae098E7f9",
        "USDC": "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238"
    }
}

class TestnetTrader:
    def __init__(self, network="goerli"):
        self.network = network
        self.config = TESTNET_CONFIG[network]
        self.w3 = Web3(Web3.HTTPProvider(self.config["rpc"]))
        self.tokens = TESTNET_TOKENS.get(network, {})
        
    def get_testnet_eth(self):
        """Instructions to get free testnet ETH"""
        print(f"🚰 Get free testnet ETH from: {self.config['faucet']}")
        print(f"📊 Check transactions at: {self.config['explorer']}")
        
    def safe_trade(self, amount_eth):
        """Execute a safe testnet trade"""
        if amount_eth > 0.1:
            print("⚠️ Limiting trade to 0.1 ETH for safety")
            amount_eth = 0.1
            
        print(f"🧪 Executing testnet trade: {amount_eth} ETH")
        # Simulation logic here
        return {"success": True, "testnet": True}

# Usage example
if __name__ == "__main__":
    trader = TestnetTrader("goerli")
    trader.get_testnet_eth()
    trader.safe_trade(0.01)
