import asyncio
from typing import Dict, Optional, Any, List
from web3 import Web3, AsyncWeb3
from eth_account import Account

class Web3Manager:
    def __init__(self):
        self.w3 = None
        self.account = None
        self.nonce = 0
        self.chain_id = 1
        self.connected = False

    async def initialize(self):
        try:
            rpc_url = "https://eth-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX"
            self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
            
            if not await self.w3.is_connected():
                raise Exception("Failed to connect to Ethereum node")
            
            private_key = "HjFs1U5F7mbWJiDKs7izTP96MEHytvm1yiSvKLT4mEvz"
            if private_key:
                try:
                    self.account = Account.from_key(private_key)
                    self.nonce = await self.w3.eth.get_transaction_count(self.account.address)
                except Exception as e:
                    print(f"Warning: Could not load account: {e}")
            
            self.chain_id = await self.w3.eth.chain_id
            self.connected = True
            
            print(f"✅ Web3 connected to chain {self.chain_id}")
            
        except Exception as e:
            print(f"❌ Web3 initialization failed: {e}")
            raise

    async def get_balance(self, address: Optional[str] = None) -> float:
        try:
            if self.account:
                target_address = address or self.account.address
            else:
                target_address = "0x0000000000000000000000000000000000000000"
            
            balance_wei = await self.w3.eth.get_balance(target_address)
            return balance_wei / 1e18
        except Exception as e:
            print(f"Failed to get balance: {e}")
            return 0.0

    async def get_gas_price(self) -> int:
        try:
            return await self.w3.eth.gas_price
        except Exception as e:
            print(f"Failed to get gas price: {e}")
            return 20 * 1e9

    async def build_transaction(self, to: Optional[str] = None, value: int = 0, 
                              data: Optional[bytes] = None, gas_price: Optional[int] = None,
                              gas_limit: Optional[int] = None) -> Dict:
        if not self.account:
            raise Exception("No account configured")
        
        if gas_price is None:
            gas_price = await self.get_gas_price()
        
        transaction = {
            'from': self.account.address,
            'nonce': self.nonce,
            'gasPrice': gas_price,
            'chainId': self.chain_id,
            'value': value
        }
        
        if to:
            transaction['to'] = to
        
        if data:
            transaction['data'] = data
        
        if gas_limit:
            transaction['gas'] = gas_limit
        else:
            transaction['gas'] = 200000
        
        return transaction

    async def send_transaction(self, transaction: Dict) -> str:
        try:
            signed_txn = self.account.sign_transaction(transaction)
            tx_hash = await self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            self.nonce += 1
            return tx_hash.hex()
        except Exception as e:
            print(f"Failed to send transaction: {e}")
            raise

    async def wait_for_transaction(self, tx_hash: str, timeout: int = 300) -> Dict:
        try:
            receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout)
            return dict(receipt)
        except Exception as e:
            print(f"Transaction wait failed: {e}")
            raise

    def get_address(self) -> str:
        if self.account:
            return self.account.address
        return ""

    async def get_code(self, address: str) -> str:
        try:
            code = await self.w3.eth.get_code(address)
            return code.hex()
        except Exception as e:
            print(f"Failed to get code: {e}")
            return "0x"

    def is_connected(self) -> bool:
        return self.connected

    async def shutdown(self):
        self.connected = False
        print("Web3 manager shutdown")