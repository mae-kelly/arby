"""DEX pool discovery and indexing"""
import asyncio
from typing import Dict, List, Set
import pandas as pd
from datetime import datetime, timedelta

class DEXIndex:
    def __init__(self, rpcs: Dict[str, 'EVMRPCClient']):
        self.rpcs = rpcs
        self.pools = pd.DataFrame()
        self.hot_pools = set()
        
    async def initialize(self):
        """Discover all pools from allowlisted factories"""
        print("Initializing DEX index...")
        
        # Factory addresses by chain
        factories = {
            'base': {
                'univ2': ['0x8909Dc15e40173Ff4699343b6eB8132c65e18eC6'],  # BaseSwap
                'univ3': ['0x33128a8fC17869897dcE68Ed026d694621f6FDfD'],  # UniV3
                'curve': [],  # Add Curve factories
            },
            'arbitrum': {
                'univ2': ['0xf1D7CC64Fb4452F05c498126312eBE29f30Fbcf9'],  # Camelot
                'univ3': ['0x1F98431c8aD98523631AE4a59f267346ea31F984'],  # UniV3
                'curve': ['0xF18056Bbd320E96A48e3Fbf8bC061322531aac99'],  # Curve
            }
        }
        
        all_pools = []
        
        for chain, chain_factories in factories.items():
            if chain not in self.rpcs:
                continue
                
            rpc = self.rpcs[chain]
            
            # Discover UniV2 pairs
            for factory in chain_factories.get('univ2', []):
                pools = await self._discover_univ2_pairs(rpc, factory)
                for pool in pools:
                    pool['chain'] = chain
                    all_pools.append(pool)
            
            # Discover UniV3 pools
            for factory in chain_factories.get('univ3', []):
                pools = await self._discover_univ3_pools(rpc, factory)
                for pool in pools:
                    pool['chain'] = chain
                    all_pools.append(pool)
        
        self.pools = pd.DataFrame(all_pools)
        print(f"Indexed {len(self.pools)} pools across {len(factories)} chains")
        
        # Mark hot pools (high volume/TVL)
        if len(self.pools) > 0:
            self.hot_pools = set(self.pools.nlargest(100, 'tvl_usd')['address'].values)
    
    async def _discover_univ2_pairs(self, rpc, factory_address: str) -> List[dict]:
        """Discover UniV2-style pairs from factory"""
        pools = []
        
        # Get pair count from factory
        factory_abi = [
            {"constant":True,"inputs":[],"name":"allPairsLength","outputs":[{"name":"","type":"uint256"}],"type":"function"},
            {"constant":True,"inputs":[{"name":"","type":"uint256"}],"name":"allPairs","outputs":[{"name":"","type":"address"}],"type":"function"}
        ]
        
        try:
            contract = rpc.w3.eth.contract(address=factory_address, abi=factory_abi)
            pair_count = await contract.functions.allPairsLength().call()
            
            # Get last 100 pairs (most recent/active)
            start = max(0, pair_count - 100)
            for i in range(start, min(pair_count, start + 100)):
                pair_address = await contract.functions.allPairs(i).call()
                pool_state = await rpc.get_pool_state(pair_address, 'univ2')
                
                pools.append({
                    'address': pair_address,
                    'type': 'univ2',
                    'factory': factory_address,
                    'token0': pool_state['token0'],
                    'token1': pool_state['token1'],
                    'reserve0': pool_state['reserve0'],
                    'reserve1': pool_state['reserve1'],
                    'tvl_usd': 0,  # Would calculate from reserves * prices
                    'discovered_at': datetime.now()
                })
        except Exception as e:
            print(f"Error discovering UniV2 pairs: {e}")
        
        return pools
    
    async def _discover_univ3_pools(self, rpc, factory_address: str) -> List[dict]:
        """Discover UniV3 pools from factory events"""
        pools = []
        
        # Query PoolCreated events
        factory_abi = [
            {"anonymous":False,"inputs":[
                {"indexed":True,"name":"token0","type":"address"},
                {"indexed":True,"name":"token1","type":"address"},
                {"indexed":True,"name":"fee","type":"uint24"},
                {"indexed":False,"name":"tickSpacing","type":"int24"},
                {"indexed":False,"name":"pool","type":"address"}
            ],"name":"PoolCreated","type":"event"}
        ]
        
        try:
            contract = rpc.w3.eth.contract(address=factory_address, abi=factory_abi)
            
            # Get recent pool creation events
            from_block = await rpc.get_block_number() - 10000  # Last ~10k blocks
            events = await contract.events.PoolCreated.get_logs(fromBlock=from_block)
            
            for event in events[-50:]:  # Last 50 pools
                pool_address = event['args']['pool']
                pool_state = await rpc.get_pool_state(pool_address, 'univ3')
                
                pools.append({
                    'address': pool_address,
                    'type': 'univ3',
                    'factory': factory_address,
                    'token0': event['args']['token0'],
                    'token1': event['args']['token1'],
                    'fee': event['args']['fee'],
                    'tick': pool_state['tick'],
                    'liquidity': pool_state['liquidity'],
                    'tvl_usd': 0,
                    'discovered_at': datetime.now()
                })
        except Exception as e:
            print(f"Error discovering UniV3 pools: {e}")
        
        return pools
    
    async def get_active_pools(self) -> pd.DataFrame:
        """Get currently active pools with fresh state"""
        # Refresh hot pools every N seconds
        for pool_address in self.hot_pools:
            # Would refresh state here
            pass
        return self.pools[self.pools['address'].isin(self.hot_pools)]
