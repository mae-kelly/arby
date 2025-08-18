# TESTNET VERSION - Safe for testing with fake money
import asyncio
import json
import time
from datetime import datetime
from web3 import Web3
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv('.env.testnet')

class TestnetMonitor:
    def __init__(self):
        # Initialize testnet connection
        rpc_url = os.getenv('GOERLI_RPC', 'https://goerli.infura.io/v3/YOUR_KEY')
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        self.wallet_address = os.getenv('TESTNET_WALLET')
        self.webhook_url = os.getenv('DISCORD_WEBHOOK')
        
        self.trades = []
        self.total_gas_used = 0
        self.start_time = datetime.now()
        
    async def send_alert(self, message):
        """Send Discord alert"""
        if not self.webhook_url:
            print(f"📱 Alert: {message}")
            return
            
        try:
            payload = {"content": f"🧪 Testnet Monitor: {message}"}
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        print(f"📱 Alert sent: {message}")
        except Exception as e:
            print(f"❌ Alert error: {e}")
    
    def check_connection(self):
        """Check testnet connection"""
        try:
            if self.w3.is_connected():
                chain_id = self.w3.eth.chain_id
                latest_block = self.w3.eth.block_number
                print(f"✅ Connected to Goerli testnet (Chain ID: {chain_id})")
                print(f"📊 Latest block: {latest_block:,}")
                return True
            else:
                print("❌ Failed to connect to Goerli testnet")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    async def monitor_testnet_activity(self):
        """Monitor testnet activity"""
        print("🧪 Starting testnet monitoring...")
        
        if not self.check_connection():
            print("❌ Cannot start monitoring - no testnet connection")
            return
        
        await self.send_alert("Testnet monitoring started - safe testing environment!")
        
        cycle = 0
        while True:
            try:
                print(f"\n{'='*50}")
                print(f"🧪 TESTNET MONITOR - Cycle {cycle}")
                print(f"{'='*50}")
                
                # Check latest block
                latest_block = self.w3.eth.block_number
                print(f"📊 Latest Goerli block: {latest_block:,}")
                
                # Monitor gas prices
                gas_price = self.w3.eth.gas_price
                gas_gwei = self.w3.from_wei(gas_price, 'gwei')
                print(f"⛽ Current gas price: {gas_gwei:.2f} gwei")
                
                # Check wallet balance
                if self.wallet_address:
                    balance = self.w3.eth.get_balance(self.wallet_address)
                    balance_eth = self.w3.from_wei(balance, 'ether')
                    print(f"💰 Testnet ETH balance: {balance_eth:.4f} ETH")
                    
                    # Alert if low balance
                    if balance_eth < 0.01:
                        await self.send_alert(f"⚠️ Low testnet balance: {balance_eth:.4f} ETH - visit faucet!")
                else:
                    print("⚠️ No wallet address configured")
                
                # Show session statistics
                uptime = datetime.now() - self.start_time
                print(f"⏱️ Uptime: {uptime}")
                print(f"📈 Monitoring cycles: {cycle + 1}")
                print(f"📊 Total trades recorded: {len(self.trades)}")
                
                # Check for faucet recommendations
                if cycle % 10 == 0 and cycle > 0:
                    print("\n🚰 REMINDER: Need more testnet ETH?")
                    print("   • Goerli: https://goerlifaucet.com")
                    print("   • Sepolia: https://sepoliafaucet.com")
                
                cycle += 1
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"❌ Monitor error: {e}")
                await asyncio.sleep(60)
    
    def record_trade(self, trade_data):
        """Record a simulated trade"""
        trade_data['timestamp'] = datetime.now().isoformat()
        trade_data['testnet'] = True
        self.trades.append(trade_data)
        
        print(f"📝 Trade recorded: {trade_data}")

if __name__ == "__main__":
    print("🧪 TESTNET MONITOR")
    print("=" * 30)
    print("✅ Safe testing environment")
    print("💰 No real money at risk")
    print("🔍 Perfect for learning")
    print("")
    
    monitor = TestnetMonitor()
    
    try:
        asyncio.run(monitor.monitor_testnet_activity())
    except KeyboardInterrupt:
        print("\n🛑 Testnet monitor stopped by user")
    except Exception as e:
        print(f"❌ Monitor error: {e}")
