"""EVM RPC client for on-chain data"""
import asyncio
from typing import Dict, List, Optional, Any
from web3 import AsyncWeb3, AsyncHTTPProvider
from web3.types import BlockData, TxReceipt
from eth_typing import Address
import aiohttp

class EVMRPCClient:
    def __init__(self, rpc_url: str):
        self.w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url))
        self.rpc_url = rpc_url
        
    async def get_pool_state(self, address: str, pool_type: str) -> Dict:
        """Get current state of a liquidity pool"""
        if pool_type == 'univ2':
            # UniV2 pair contract
            pair_abi = [
                {"constant":True,"inputs":[],"name":"getReserves","outputs":[
                    {"name":"reserve0","type":"uint112"},
                    {"name":"reserve1","type":"uint112"},
                    {"name":"blockTimestampLast","type":"uint32"}
                ],"type":"function"},
                {"constant":True,"inputs":[],"name":"token0","outputs":[{"name":"","type":"address"}],"type":"function"},
                {"constant":True,"inputs":[],"name":"token1","outputs":[{"name":"","type":"address"}],"type":"function"},
            ]
            
            contract = self.w3.eth.contract(address=address, abi=pair_abi)
            reserves = await contract.functions.getReserves().call()
            token0 = await contract.functions.token0().call()
            token1 = await contract.functions.token1().call()
            
            return {
                'type': 'univ2',
                'address': address,
                'token0': token0,
                'token1': token1,
                'reserve0': reserves[0],
                'reserve1': reserves[1],
                'timestamp': reserves[2]
            }
            
        elif pool_type == 'univ3':
            # UniV3 pool state
            pool_abi = [
                {"inputs":[],"name":"slot0","outputs":[
                    {"name":"sqrtPriceX96","type":"uint160"},
                    {"name":"tick","type":"int24"},
                    {"name":"observationIndex","type":"uint16"},
                    {"name":"observationCardinality","type":"uint16"},
                    {"name":"observationCardinalityNext","type":"uint16"},
                    {"name":"feeProtocol","type":"uint8"},
                    {"name":"unlocked","type":"bool"}
                ],"type":"function"},
                {"inputs":[],"name":"liquidity","outputs":[{"name":"","type":"uint128"}],"type":"function"},
                {"inputs":[],"name":"fee","outputs":[{"name":"","type":"uint24"}],"type":"function"}
            ]
            
            contract = self.w3.eth.contract(address=address, abi=pool_abi)
            slot0 = await contract.functions.slot0().call()
            liquidity = await contract.functions.liquidity().call()
            fee = await contract.functions.fee().call()
            
            return {
                'type': 'univ3',
                'address': address,
                'sqrtPriceX96': slot0[0],
                'tick': slot0[1],
                'liquidity': liquidity,
                'fee': fee
            }
    
    async def get_base_fee(self) -> int:
        """Get current base fee"""
        block = await self.w3.eth.get_block('latest')
        return block.get('baseFeePerGas', 0)
    
    async def get_priority_fee_hint(self) -> int:
        """Get suggested priority fee"""
        try:
            fee_history = await self.w3.eth.fee_history(10, 'latest', [50])
            return int(fee_history['reward'][0][0] if fee_history['reward'] else 2e9)
        except:
            return int(2e9)  # 2 gwei default
    
    async def get_block_number(self) -> int:
        """Get current block number"""
        return await self.w3.eth.block_number
    
    async def subscribe_mempool(self) -> AsyncIterator[dict]:
        """Subscribe to mempool transactions (if supported)"""
        # Note: Requires websocket provider and node with mempool access
        async def poll_pending():
            while True:
                try:
                    pending = await self.w3.eth.get_filter_changes('pending')
                    for tx_hash in pending:
                        tx = await self.w3.eth.get_transaction(tx_hash)
                        yield tx
                except Exception as e:
                    print(f"Mempool poll error: {e}")
                await asyncio.sleep(0.1)
        
        return poll_pending()
