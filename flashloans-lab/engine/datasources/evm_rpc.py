import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from web3 import AsyncWeb3
from web3.eth import AsyncEth
import json
from dataclasses import dataclass

@dataclass
class BlockInfo:
    number: int
    timestamp: int
    base_fee_per_gas: int
    gas_limit: int
    gas_used: int

class EVMRPCClient:
    def __init__(self, rpc_url: str, chain_id: int):
        self.rpc_url = rpc_url
        self.chain_id = chain_id
        self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        
    async def get_latest_block(self) -> BlockInfo:
        """Get latest block information"""
        block = await self.w3.eth.get_block('latest')
        return BlockInfo(
            number=block.number,
            timestamp=block.timestamp,
            base_fee_per_gas=block.baseFeePerGas or 0,
            gas_limit=block.gasLimit,
            gas_used=block.gasUsed
        )
    
    async def call_contract(self, 
                          to_address: str, 
                          data: str, 
                          block: str = 'latest') -> str:
        """Make eth_call to contract"""
        try:
            result = await self.w3.eth.call({
                'to': to_address,
                'data': data
            }, block)
            return result.hex()
        except Exception as e:
            raise RuntimeError(f"Contract call failed: {e}")
    
    async def get_balance(self, address: str, block: str = 'latest') -> int:
        """Get ETH balance of address"""
        return await self.w3.eth.get_balance(address, block)
    
    async def estimate_gas(self, transaction: Dict[str, Any]) -> int:
        """Estimate gas for transaction"""
        return await self.w3.eth.estimate_gas(transaction)
    
    async def get_gas_price(self) -> Dict[str, int]:
        """Get current gas pricing"""
        block = await self.get_latest_block()
        gas_price = await self.w3.eth.gas_price
        return {
            'base_fee': block.base_fee_per_gas,
            'gas_price': gas_price,
            'max_priority_fee': max(1_000_000_000, gas_price - block.base_fee_per_gas)
        }
