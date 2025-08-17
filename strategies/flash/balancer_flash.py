import asyncio
from web3 import Web3
from typing import Dict, List, Tuple

class BalancerFlashLoanStrategy:
    def __init__(self, w3: Web3, vault_address: str = '0xBA12222222228d8Ba445958a75a0704d566BF2C8'):
        self.w3 = w3
        self.vault_address = vault_address
        self.vault_abi = self.load_balancer_abi()
        self.vault = w3.eth.contract(address=vault_address, abi=self.vault_abi)
    
    async def execute_zero_fee_flash_loan(self, tokens: List[str], amounts: List[int], strategy_data: bytes) -> str:
        """Execute Balancer flash loan (0% fee)"""
        
        tx = self.vault.functions.flashLoan(
            self.w3.eth.default_account,  # recipient
            tokens,  # tokens
            amounts,  # amounts
            strategy_data  # userData
        ).build_transaction({
            'gas': 3000000,
            'gasPrice': await self.get_gas_price(),
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.default_account)
        })
        
        signed = self.w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
        return self.w3.eth.send_raw_transaction(signed.rawTransaction).hex()
    
    def load_balancer_abi(self) -> List[Dict]:
        return [{"inputs":[{"internalType":"contract IFlashLoanRecipient","name":"recipient","type":"address"},{"internalType":"contract IERC20[]","name":"tokens","type":"address[]"},{"internalType":"uint256[]","name":"amounts","type":"uint256[]"},{"internalType":"bytes","name":"userData","type":"bytes"}],"name":"flashLoan","outputs":[],"stateMutability":"nonpayable","type":"function"}]
