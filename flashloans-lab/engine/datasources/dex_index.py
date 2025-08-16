import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
import json
from .evm_rpc import EVMRPCClient

@dataclass
class PoolInfo:
    address: str
    token0: str
    token1: str
    fee: int
    pool_type: str  # 'uniswap_v2', 'uniswap_v3', 'solidly'
    reserve0: Optional[int] = None
    reserve1: Optional[int] = None
    sqrt_price_x96: Optional[int] = None
    liquidity: Optional[int] = None
    tick: Optional[int] = None

class DEXIndexer:
    def __init__(self, rpc_client: EVMRPCClient):
        self.rpc = rpc_client
        self.pools: Dict[str, PoolInfo] = {}
        
    async def index_uniswap_v2_pool(self, pool_address: str) -> PoolInfo:
        """Index a Uniswap V2 style pool"""
        # Get reserves
        reserves_data = await self.rpc.call_contract(
            pool_address, 
            "0x0902f1ac"  # getReserves()
        )
        
        reserve0 = int(reserves_data[2:66], 16)
        reserve1 = int(reserves_data[66:130], 16)
        
        # Get token addresses
        token0_data = await self.rpc.call_contract(pool_address, "0x0dfe1681")
        token1_data = await self.rpc.call_contract(pool_address, "0xd21220a7")
        
        token0 = "0x" + token0_data[-40:]
        token1 = "0x" + token1_data[-40:]
        
        pool = PoolInfo(
            address=pool_address,
            token0=token0,
            token1=token1,
            fee=300,  # 0.3% standard
            pool_type="uniswap_v2",
            reserve0=reserve0,
            reserve1=reserve1
        )
        
        self.pools[pool_address] = pool
        return pool
    
    async def index_uniswap_v3_pool(self, pool_address: str) -> PoolInfo:
        """Index a Uniswap V3 pool"""
        # Get slot0
        slot0_data = await self.rpc.call_contract(
            pool_address,
            "0x3850c7bd"  # slot0()
        )
        
        sqrt_price_x96 = int(slot0_data[2:66], 16)
        tick = int.from_bytes(bytes.fromhex(slot0_data[66:130]), 'big', signed=True)
        
        # Get liquidity
        liquidity_data = await self.rpc.call_contract(pool_address, "0x1a686502")
        liquidity = int(liquidity_data[2:66], 16)
        
        # Get fee
        fee_data = await self.rpc.call_contract(pool_address, "0xddca3f43")
        fee = int(fee_data[2:66], 16)
        
        # Get tokens
        token0_data = await self.rpc.call_contract(pool_address, "0x0dfe1681")
        token1_data = await self.rpc.call_contract(pool_address, "0xd21220a7")
        
        token0 = "0x" + token0_data[-40:]
        token1 = "0x" + token1_data[-40:]
        
        pool = PoolInfo(
            address=pool_address,
            token0=token0,
            token1=token1,
            fee=fee,
            pool_type="uniswap_v3",
            sqrt_price_x96=sqrt_price_x96,
            liquidity=liquidity,
            tick=tick
        )
        
        self.pools[pool_address] = pool
        return pool
    
    async def refresh_pool_state(self, pool_address: str) -> PoolInfo:
        """Refresh pool state data"""
        if pool_address not in self.pools:
            raise ValueError(f"Pool {pool_address} not indexed")
            
        pool = self.pools[pool_address]
        
        if pool.pool_type == "uniswap_v2":
            return await self.index_uniswap_v2_pool(pool_address)
        elif pool.pool_type == "uniswap_v3":
            return await self.index_uniswap_v3_pool(pool_address)
        else:
            raise NotImplementedError(f"Pool type {pool.pool_type} not implemented")
    
    async def get_pools_for_tokens(self, token0: str, token1: str) -> List[PoolInfo]:
        """Get all pools for a token pair"""
        return [
            pool for pool in self.pools.values()
            if (pool.token0.lower() == token0.lower() and pool.token1.lower() == token1.lower()) or
               (pool.token0.lower() == token1.lower() and pool.token1.lower() == token0.lower())
        ]
