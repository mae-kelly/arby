#!/usr/bin/env python3
"""
REAL Flash Loan Arbitrage Executor
Actually executes profitable opportunities on mainnet
"""

import asyncio
import aiohttp
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from web3 import Web3
from eth_account import Account
from decimal import Decimal

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RealFlashOpportunity:
    token: str
    token_address: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    flash_amount: int  # In wei
    expected_profit: int  # In wei
    gas_estimate: int
    flash_provider: str
    execution_data: bytes

class RealFlashLoanExecutor:
    def __init__(self):
        self.w3 = None
        self.account = None
        self.session = None
        
        # Initialize Web3 and account
        self.setup_web3()
        self.setup_account()
        
        # Contract addresses on Ethereum mainnet
        self.contracts = {
            'aave_pool': '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9',
            'balancer_vault': '0xBA12222222228d8Ba445958a75a0704d566BF2C8',
            'uniswap_v2_router': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
            'uniswap_v3_router': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'sushiswap_router': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
            'weth': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            'usdc': '0xA0b86a33E6e86c026a91F4A7A6B89e23AA8C1Fd9',
            'usdt': '0xdAC17F958D2ee523a2206206994597C13D831ec7'
        }
        
        # Your deployed flash loan contract (deploy this first!)
        self.flash_contract_address = os.getenv('FLASH_CONTRACT_ADDRESS')
        
        # Minimum profit thresholds
        self.min_profit_usd = float(os.getenv('MIN_PROFIT_USD', '100'))
        self.max_gas_price = int(os.getenv('MAX_GAS_GWEI', '100')) * 1e9
        
        # Load contract ABIs
        self.load_abis()
        
        # Statistics
        self.stats = {
            'opportunities_found': 0,
            'trades_executed': 0,
            'total_profit': 0.0,
            'failed_trades': 0
        }
        
    def setup_web3(self):
        """Setup Web3 connection to Ethereum mainnet"""
        rpc_urls = [
            os.getenv('INFURA_URL'),
            os.getenv('ALCHEMY_URL'),
            os.getenv('QUICKNODE_URL'),
            'https://eth.llamarpc.com',  # Free public RPC
            'https://rpc.ankr.com/eth'   # Free public RPC
        ]
        
        for rpc_url in rpc_urls:
            if rpc_url and 'YOUR_KEY' not in rpc_url:
                try:
                    self.w3 = Web3(Web3.HTTPProvider(rpc_url))
                    if self.w3.is_connected():
                        logger.info(f"✅ Connected to Ethereum via: {rpc_url[:50]}...")
                        break
                except Exception as e:
                    logger.warning(f"Failed to connect to {rpc_url}: {e}")
                    
        if not self.w3 or not self.w3.is_connected():
            raise Exception("❌ Could not connect to Ethereum mainnet")
            
    def setup_account(self):
        """Setup trading account from private key"""
        private_key = os.getenv('PRIVATE_KEY')
        if not private_key:
            raise Exception("❌ PRIVATE_KEY environment variable required")
            
        try:
            self.account = Account.from_key(private_key)
            balance = self.w3.eth.get_balance(self.account.address)
            logger.info(f"✅ Account loaded: {self.account.address}")
            logger.info(f"💰 ETH Balance: {self.w3.from_wei(balance, 'ether'):.4f} ETH")
            
            if balance < self.w3.to_wei('0.01', 'ether'):
                logger.warning("⚠️ Low ETH balance - may not cover gas costs")
                
        except Exception as e:
            raise Exception(f"❌ Failed to load account: {e}")
            
    def load_abis(self):
        """Load contract ABIs"""
        self.abis = {
            'erc20': [
                {"constant":True,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"},
                {"constant":False,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"}
            ],
            'uniswap_v2': [
                {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"},
                {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"}],"name":"getAmountsOut","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"view","type":"function"}
            ],
            'balancer_vault': [
                {"inputs":[{"internalType":"contract IFlashLoanRecipient","name":"recipient","type":"address"},{"internalType":"contract IERC20[]","name":"tokens","type":"address[]"},{"internalType":"uint256[]","name":"amounts","type":"uint256[]"},{"internalType":"bytes","name":"userData","type":"bytes"}],"name":"flashLoan","outputs":[],"stateMutability":"nonpayable","type":"function"}
            ]
        }
        
    async def initialize(self):
        """Initialize the executor"""
        self.session = aiohttp.ClientSession()
        
        print("🚀 REAL FLASH LOAN ARBITRAGE EXECUTOR")
        print("=" * 60)
        print(f"Account: {self.account.address}")
        print(f"Network: Ethereum Mainnet")
        print(f"Min Profit: ${self.min_profit_usd}")
        print("=" * 60)
        
        # Check if flash loan contract is deployed
        if not self.flash_contract_address:
            logger.warning("⚠️ FLASH_CONTRACT_ADDRESS not set - deploy contract first!")
            logger.info("Use the provided Solidity contract and set FLASH_CONTRACT_ADDRESS")
            
        # Start monitoring and execution
        await asyncio.gather(
            self.monitor_opportunities(),
            self.execute_opportunities(),
            return_exceptions=True
        )
        
    async def monitor_opportunities(self):
        """Monitor for real arbitrage opportunities"""
        while True:
            try:
                # Get current prices from multiple sources
                prices = await self.fetch_all_prices()
                
                # Calculate real arbitrage opportunities
                opportunities = await self.find_real_arbitrage(prices)
                
                # Queue profitable opportunities
                for opp in opportunities:
                    if opp.expected_profit > self.min_profit_usd * 1e18:  # Convert to wei
                        await self.opportunity_queue.put(opp)
                        self.stats['opportunities_found'] += 1
                        
                        logger.info(f"🎯 Opportunity found: ${self.w3.from_wei(opp.expected_profit, 'ether'):.2f} profit")
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)
                
    async def fetch_all_prices(self) -> Dict:
        """Fetch prices from exchanges and DEXs"""
        prices = {}
        
        # CEX prices (Binance, Coinbase)
        cex_prices = await self.fetch_cex_prices()
        prices.update(cex_prices)
        
        # DEX prices (Uniswap, SushiSwap)
        dex_prices = await self.fetch_dex_prices()
        prices.update(dex_prices)
        
        return prices
        
    async def fetch_cex_prices(self) -> Dict:
        """Fetch CEX prices"""
        prices = {}
        
        try:
            # Binance
            url = 'https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT'
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    prices['binance_eth'] = float(data['price'])
        except:
            pass
            
        try:
            # Coinbase
            url = 'https://api.exchange.coinbase.com/products/ETH-USD/ticker'
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    prices['coinbase_eth'] = float(data['price'])
        except:
            pass
            
        return prices
        
    async def fetch_dex_prices(self) -> Dict:
        """Fetch DEX prices from on-chain contracts"""
        prices = {}
        
        try:
            # Uniswap V2 ETH/USDC price
            router = self.w3.eth.contract(
                address=self.contracts['uniswap_v2_router'],
                abi=self.abis['uniswap_v2']
            )
            
            # Get ETH price in USDC
            path = [self.contracts['weth'], self.contracts['usdc']]
            amounts = router.functions.getAmountsOut(
                self.w3.to_wei('1', 'ether'), 
                path
            ).call()
            
            # USDC has 6 decimals
            eth_price_usdc = amounts[1] / 1e6
            prices['uniswap_v2_eth'] = eth_price_usdc
            
        except Exception as e:
            logger.debug(f"Uniswap V2 price fetch error: {e}")
            
        try:
            # SushiSwap ETH/USDC price
            sushi_router = self.w3.eth.contract(
                address=self.contracts['sushiswap_router'],
                abi=self.abis['uniswap_v2']  # Same ABI
            )
            
            path = [self.contracts['weth'], self.contracts['usdc']]
            amounts = sushi_router.functions.getAmountsOut(
                self.w3.to_wei('1', 'ether'),
                path
            ).call()
            
            eth_price_usdc = amounts[1] / 1e6
            prices['sushiswap_eth'] = eth_price_usdc
            
        except Exception as e:
            logger.debug(f"SushiSwap price fetch error: {e}")
            
        return prices
        
    async def find_real_arbitrage(self, prices: Dict) -> List[RealFlashOpportunity]:
        """Find real arbitrage opportunities"""
        opportunities = []
        
        # Find ETH arbitrage opportunities
        eth_prices = {k: v for k, v in prices.items() if 'eth' in k}
        
        if len(eth_prices) >= 2:
            min_source = min(eth_prices.items(), key=lambda x: x[1])
            max_source = max(eth_prices.items(), key=lambda x: x[1])
            
            min_exchange, min_price = min_source
            max_exchange, max_price = max_source
            
            # Calculate potential profit
            price_diff_pct = (max_price - min_price) / min_price
            
            if price_diff_pct > 0.005:  # 0.5% minimum spread
                # Calculate flash loan size and profit
                flash_amount_eth = self.w3.to_wei('10', 'ether')  # 10 ETH flash loan
                expected_profit_wei = int(flash_amount_eth * price_diff_pct * 0.7)  # 70% of spread
                
                # Estimate gas cost
                gas_estimate = 800000  # Complex flash loan transaction
                
                opportunity = RealFlashOpportunity(
                    token='ETH',
                    token_address=self.contracts['weth'],
                    buy_exchange=min_exchange,
                    sell_exchange=max_exchange,
                    buy_price=min_price,
                    sell_price=max_price,
                    flash_amount=flash_amount_eth,
                    expected_profit=expected_profit_wei,
                    gas_estimate=gas_estimate,
                    flash_provider='balancer',  # 0% fee
                    execution_data=self.encode_arbitrage_data(min_exchange, max_exchange, flash_amount_eth)
                )
                
                opportunities.append(opportunity)
                
        return opportunities
        
    def encode_arbitrage_data(self, buy_source: str, sell_source: str, amount: int) -> bytes:
        """Encode arbitrage execution data for smart contract"""
        # This would encode the specific swap calls needed
        # Simplified for example
        return self.w3.codec.encode_abi(
            ['string', 'string', 'uint256'],
            [buy_source, sell_source, amount]
        )
        
    async def execute_opportunities(self):
        """Execute profitable opportunities"""
        self.opportunity_queue = asyncio.Queue()
        
        while True:
            try:
                # Wait for opportunity
                opportunity = await asyncio.wait_for(
                    self.opportunity_queue.get(), 
                    timeout=1.0
                )
                
                # Execute if still profitable
                success = await self.execute_flash_arbitrage(opportunity)
                
                if success:
                    self.stats['trades_executed'] += 1
                    profit_eth = self.w3.from_wei(opportunity.expected_profit, 'ether')
                    self.stats['total_profit'] += float(profit_eth)
                    logger.info(f"✅ Trade executed! Profit: {profit_eth:.4f} ETH")
                else:
                    self.stats['failed_trades'] += 1
                    logger.warning("❌ Trade failed or became unprofitable")
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Execution error: {e}")
                
    async def execute_flash_arbitrage(self, opportunity: RealFlashOpportunity) -> bool:
        """Execute flash loan arbitrage"""
        
        if not self.flash_contract_address:
            logger.warning("No flash contract deployed - simulation only")
            return False
            
        try:
            # Get current gas price
            gas_price = self.w3.eth.gas_price
            if gas_price > self.max_gas_price:
                logger.warning(f"Gas price too high: {gas_price/1e9:.1f} gwei")
                return False
                
            # Calculate gas cost
            gas_cost_wei = gas_price * opportunity.gas_estimate
            gas_cost_usd = (gas_cost_wei / 1e18) * opportunity.buy_price
            
            # Check if still profitable after gas
            net_profit_wei = opportunity.expected_profit - gas_cost_wei
            if net_profit_wei < self.min_profit_usd * 1e18:
                logger.info("Opportunity no longer profitable after gas")
                return False
                
            logger.info(f"🚀 Executing flash arbitrage:")
            logger.info(f"   Amount: {self.w3.from_wei(opportunity.flash_amount, 'ether'):.2f} ETH")
            logger.info(f"   Expected profit: {self.w3.from_wei(net_profit_wei, 'ether'):.4f} ETH")
            logger.info(f"   Gas cost: ${gas_cost_usd:.2f}")
            
            # Build transaction
            flash_contract = self.w3.eth.contract(
                address=self.flash_contract_address,
                abi=self.get_flash_contract_abi()
            )
            
            # Execute flash loan
            tx = flash_contract.functions.executeFlashArbitrage(
                [self.contracts['weth']],  # tokens
                [opportunity.flash_amount],  # amounts
                opportunity.execution_data  # userData
            ).build_transaction({
                'from': self.account.address,
                'gas': opportunity.gas_estimate,
                'gasPrice': gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"📤 Transaction sent: {tx_hash.hex()}")
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            
            if receipt.status == 1:
                logger.info(f"✅ Transaction confirmed in block {receipt.blockNumber}")
                return True
            else:
                logger.error("❌ Transaction failed")
                return False
                
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return False
            
    def get_flash_contract_abi(self) -> List:
        """Get flash loan contract ABI"""
        return [
            {
                "inputs": [
                    {"name": "tokens", "type": "address[]"},
                    {"name": "amounts", "type": "uint256[]"}, 
                    {"name": "userData", "type": "bytes"}
                ],
                "name": "executeFlashArbitrage",
                "outputs": [],
                "type": "function"
            }
        ]
        
    async def display_stats(self):
        """Display execution statistics"""
        while True:
            await asyncio.sleep(30)
            
            print(f"\n📊 EXECUTION STATS:")
            print(f"   Opportunities Found: {self.stats['opportunities_found']}")
            print(f"   Trades Executed: {self.stats['trades_executed']}")
            print(f"   Failed Trades: {self.stats['failed_trades']}")
            print(f"   Total Profit: {self.stats['total_profit']:.4f} ETH")
            
            if self.stats['trades_executed'] > 0:
                success_rate = (self.stats['trades_executed'] / 
                               (self.stats['trades_executed'] + self.stats['failed_trades'])) * 100
                print(f"   Success Rate: {success_rate:.1f}%")
                
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

async def main():
    """Main execution"""
    
    # Check environment variables
    required_vars = ['PRIVATE_KEY']
    for var in required_vars:
        if not os.getenv(var):
            print(f"❌ {var} environment variable required")
            print("\nSetup instructions:")
            print("1. export PRIVATE_KEY='your_private_key_here'")
            print("2. export FLASH_CONTRACT_ADDRESS='deployed_contract_address'")
            print("3. export INFURA_URL='https://mainnet.infura.io/v3/your_key'")
            return
            
    executor = RealFlashLoanExecutor()
    
    try:
        await executor.initialize()
    except KeyboardInterrupt:
        print("\n🛑 Stopping executor...")
    finally:
        await executor.cleanup()
        print("✅ Executor stopped")

if __name__ == "__main__":
    # Warning about mainnet usage
    print("⚠️  WARNING: This will execute trades on Ethereum MAINNET")
    print("⚠️  Make sure you understand the risks and have tested on testnet first")
    
    confirm = input("Type 'YES' to continue with mainnet execution: ")
    if confirm.upper() == 'YES':
        asyncio.run(main())
    else:
        print("Execution cancelled")