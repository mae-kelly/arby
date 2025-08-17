import asyncio
from web3 import Web3, AsyncWeb3
from web3.providers import WebsocketProvider, HTTPProvider
from eth_account import Account
from eth_abi import encode_abi
import json
import os
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import aiohttp
from hexbytes import HexBytes
import time

class Web3Manager:
    """High-performance Web3 interface for multi-chain operations"""
    
    def __init__(self):
        self.providers = {}
        self.w3_instances = {}
        self.contracts = {}
        self.account = None
        self.nonces = {}
        self.gas_trackers = {}
        
        self.setup_providers()
        self.load_contracts()
        
    def setup_providers(self):
        """Setup Web3 providers for all chains"""
        
        chains = {
            'ethereum': os.getenv('RPC_ENDPOINTS_ETH', '').split(','),
            'bsc': os.getenv('RPC_ENDPOINTS_BSC', '').split(','),
            'polygon': os.getenv('RPC_ENDPOINTS_POLYGON', '').split(','),
            'arbitrum': os.getenv('RPC_ENDPOINTS_ARBITRUM', '').split(','),
            'optimism': os.getenv('RPC_ENDPOINTS_OPTIMISM', '').split(','),
            'avalanche': os.getenv('RPC_ENDPOINTS_AVALANCHE', '').split(','),
        }
        
        for chain, rpcs in chains.items():
            if rpcs and rpcs[0]:
                # Use WebSocket if available, else HTTP
                for rpc in rpcs:
                    if rpc.startswith('wss://'):
                        provider = WebsocketProvider(rpc)
                        break
                    elif rpc.startswith('http'):
                        provider = HTTPProvider(rpc)
                        break
                else:
                    continue
                    
                self.providers[chain] = provider
                self.w3_instances[chain] = Web3(provider)
                
        # Setup account
        private_key = os.getenv('PRIVATE_KEY')
        if private_key:
            self.account = Account.from_key(private_key)
            
    def load_contracts(self):
        """Load contract ABIs and addresses"""
        
        # Common contract ABIs
        self.abis = {
            'ERC20': json.loads('[{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transfer","outputs":[{"name":"","type":"bool"}],"type":"function"}]'),
            
            'UniswapV2Router': json.loads('[{"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"type":"function"}]'),
            
            'FlashLoan': json.loads('[{"inputs":[{"internalType":"address","name":"receiverAddress","type":"address"},{"internalType":"address[]","name":"assets","type":"address[]"},{"internalType":"uint256[]","name":"amounts","type":"uint256[]"},{"internalType":"uint256[]","name":"modes","type":"uint256[]"},{"internalType":"address","name":"onBehalfOf","type":"address"},{"internalType":"bytes","name":"params","type":"bytes"},{"internalType":"uint16","name":"referralCode","type":"uint16"}],"name":"flashLoan","outputs":[],"type":"function"}]')
        }
        
        # Contract addresses by chain
        self.addresses = {
            'ethereum': {
                'AAVE_POOL': '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9',
                'UNISWAP_V2': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
                'UNISWAP_V3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'SUSHISWAP': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
                'BALANCER': '0xBA12222222228d8Ba445958a75a0704d566BF2C8',
            },
            'bsc': {
                'PANCAKESWAP': '0x10ED43C718714eb63d5aA57B78B54704E256C495',
                'VENUS': '0xfD36E2c2a6789Db23113685031d7F16329158384',
            },
            'polygon': {
                'QUICKSWAP': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff',
                'AAVE_POOL': '0x8dFf5E27EA6b7AC08EbFdf9eB090F32ee9a30fcf',
            }
        }
        
    async def get_gas_price(self, chain: str = 'ethereum') -> int:
        """Get optimal gas price for chain"""
        
        w3 = self.w3_instances.get(chain)
        if not w3:
            return 50 * 10**9  # Default 50 gwei
            
        try:
            # Get base fee from latest block
            latest = w3.eth.get_block('latest')
            base_fee = latest.get('baseFeePerGas', 0)
            
            # Calculate priority fee (tip)
            if chain == 'ethereum':
                # Use Flashbots for MEV protection
                priority_fee = await self.get_flashbots_priority_fee()
            else:
                priority_fee = 2 * 10**9  # 2 gwei default tip
                
            return base_fee + priority_fee
            
        except Exception as e:
            print(f"Gas price error: {e}")
            return 50 * 10**9
            
    async def get_flashbots_priority_fee(self) -> int:
        """Get Flashbots recommended priority fee"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.flashbots.net/v1/blocks') as resp:
                    data = await resp.json()
                    # Extract median priority fee from recent blocks
                    fees = [b.get('priority_fee', 2e9) for b in data.get('blocks', [])]
                    return int(sorted(fees)[len(fees)//2]) if fees else 2 * 10**9
        except:
            return 2 * 10**9
            
    async def send_flashbots_bundle(self, transactions: List[Dict]) -> str:
        """Send transaction bundle via Flashbots"""
        
        flashbots_url = os.getenv('FLASHBOTS_RPC', 'https://relay.flashbots.net')
        
        bundle = {
            'jsonrpc': '2.0',
            'method': 'eth_sendBundle',
            'params': [{
                'txs': [self.sign_transaction(tx) for tx in transactions],
                'blockNumber': hex(self.w3_instances['ethereum'].eth.block_number + 1)
            }],
            'id': 1
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(flashbots_url, json=bundle) as resp:
                result = await resp.json()
                return result.get('result', {}).get('bundleHash', '')
                
    def sign_transaction(self, tx: Dict) -> str:
        """Sign transaction and return raw tx"""
        
        if not self.account:
            raise ValueError("No account configured")
            
        # Add nonce if not present
        if 'nonce' not in tx:
            chain = tx.get('chain', 'ethereum')
            tx['nonce'] = self.get_nonce(chain)
            
        signed = self.account.sign_transaction(tx)
        return signed.rawTransaction.hex()
        
    def get_nonce(self, chain: str) -> int:
        """Get and track nonce for chain"""
        
        if chain not in self.nonces:
            w3 = self.w3_instances.get(chain)
            if w3 and self.account:
                self.nonces[chain] = w3.eth.get_transaction_count(self.account.address)
            else:
                self.nonces[chain] = 0
                
        nonce = self.nonces[chain]
        self.nonces[chain] += 1
        return nonce
        
    async def execute_swap(
        self,
        chain: str,
        router: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        min_amount_out: int,
        deadline: Optional[int] = None
    ) -> str:
        """Execute swap on DEX"""
        
        w3 = self.w3_instances.get(chain)
        if not w3:
            raise ValueError(f"Chain {chain} not configured")
            
        router_address = self.addresses.get(chain, {}).get(router)
        if not router_address:
            raise ValueError(f"Router {router} not found on {chain}")
            
        # Build transaction
        contract = w3.eth.contract(
            address=router_address,
            abi=self.abis['UniswapV2Router']
        )
        
        if not deadline:
            deadline = int(time.time()) + 300  # 5 minutes
            
        tx = contract.functions.swapExactTokensForTokens(
            amount_in,
            min_amount_out,
            [token_in, token_out],
            self.account.address,
            deadline
        ).build_transaction({
            'from': self.account.address,
            'gas': 300000,
            'gasPrice': await self.get_gas_price(chain),
            'nonce': self.get_nonce(chain)
        })
        
        # Sign and send
        signed = self.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed)
        
        return tx_hash.hex()
        
    async def execute_flash_loan(
        self,
        chain: str,
        assets: List[str],
        amounts: List[int],
        params: bytes
    ) -> str:
        """Execute flash loan"""
        
        w3 = self.w3_instances.get(chain)
        if not w3:
            raise ValueError(f"Chain {chain} not configured")
            
        pool_address = self.addresses.get(chain, {}).get('AAVE_POOL')
        if not pool_address:
            raise ValueError(f"AAVE pool not found on {chain}")
            
        contract = w3.eth.contract(
            address=pool_address,
            abi=self.abis['FlashLoan']
        )
        
        tx = contract.functions.flashLoan(
            self.account.address,
            assets,
            amounts,
            [0] * len(assets),  # No debt
            self.account.address,
            params,
            0  # No referral
        ).build_transaction({
            'from': self.account.address,
            'gas': 3000000,
            'gasPrice': await self.get_gas_price(chain),
            'nonce': self.get_nonce(chain)
        })
        
        signed = self.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed)
        
        return tx_hash.hex()
        
    async def get_token_balance(
        self,
        chain: str,
        token: str,
        address: Optional[str] = None
    ) -> int:
        """Get token balance"""
        
        w3 = self.w3_instances.get(chain)
        if not w3:
            return 0
            
        if not address:
            address = self.account.address
            
        if token == 'ETH':
            return w3.eth.get_balance(address)
        else:
            contract = w3.eth.contract(
                address=Web3.toChecksumAddress(token),
                abi=self.abis['ERC20']
            )
            return contract.functions.balanceOf(address).call()
            
    async def monitor_mempool(self, chain: str, callback):
        """Monitor mempool for opportunities"""
        
        w3 = self.w3_instances.get(chain)
        if not w3:
            return
            
        # Subscribe to pending transactions
        pending_filter = w3.eth.filter('pending')
        
        while True:
            try:
                for tx_hash in pending_filter.get_new_entries():
                    tx = w3.eth.get_transaction(tx_hash)
                    if tx:
                        await callback(tx)
            except Exception as e:
                print(f"Mempool error: {e}")
                
            await asyncio.sleep(0.01)
            
    def decode_swap_data(self, input_data: str) -> Optional[Dict]:
        """Decode swap transaction data"""
        
        try:
            # Uniswap V2 swap signature
            if input_data.startswith('0x38ed1739'):
                # swapExactTokensForTokens
                decoded = decode_abi(
                    ['uint256', 'uint256', 'address[]', 'address', 'uint256'],
                    HexBytes(input_data[10:])
                )
                return {
                    'type': 'swap',
                    'amountIn': decoded[0],
                    'amountOutMin': decoded[1],
                    'path': decoded[2],
                    'to': decoded[3],
                    'deadline': decoded[4]
                }
        except:
            pass
            
        return None