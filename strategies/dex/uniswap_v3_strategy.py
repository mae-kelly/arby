import asyncio
from web3 import Web3
from typing import Dict, List, Tuple
import json

class UniswapV3Strategy:
    def __init__(self, w3: Web3):
        self.w3 = w3
        self.router_address = '0xE592427A0AEce92De3Edee1F18E0157C05861564'
        self.quoter_address = '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6'
        self.factory_address = '0x1F98431c8aD98523631AE4a59f267346ea31F984'
        self.pool_fees = [100, 500, 3000, 10000]  # 0.01%, 0.05%, 0.3%, 1%
        
    async def scan_v3_opportunities(self, token_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Scan Uniswap V3 pools for arbitrage opportunities"""
        opportunities = []
        
        for token_a, token_b in token_pairs:
            for fee in self.pool_fees:
                try:
                    pool_address = await self.get_pool_address(token_a, token_b, fee)
                    if pool_address != '0x0000000000000000000000000000000000000000':
                        opportunity = await self.analyze_pool(token_a, token_b, fee, pool_address)
                        if opportunity:
                            opportunities.append(opportunity)
                except Exception:
                    continue
        
        return opportunities
    
    async def get_pool_address(self, token_a: str, token_b: str, fee: int) -> str:
        """Get Uniswap V3 pool address"""
        factory_abi = [{"inputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"address","name":"","type":"address"},{"internalType":"uint24","name":"","type":"uint24"}],"name":"getPool","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"}]
        factory = self.w3.eth.contract(address=self.factory_address, abi=factory_abi)
        return factory.functions.getPool(token_a, token_b, fee).call()
    
    async def analyze_pool(self, token_a: str, token_b: str, fee: int, pool_address: str) -> Dict:
        """Analyze pool for arbitrage opportunities"""
        # Get pool state
        pool_abi = [{"inputs":[],"name":"slot0","outputs":[{"internalType":"uint160","name":"sqrtPriceX96","type":"uint160"},{"internalType":"int24","name":"tick","type":"int24"},{"internalType":"uint16","name":"observationIndex","type":"uint16"},{"internalType":"uint16","name":"observationCardinality","type":"uint16"},{"internalType":"uint16","name":"observationCardinalityNext","type":"uint16"},{"internalType":"uint8","name":"feeProtocol","type":"uint8"},{"internalType":"bool","name":"unlocked","type":"bool"}],"stateMutability":"view","type":"function"}]
        pool = self.w3.eth.contract(address=pool_address, abi=pool_abi)
        
        slot0 = pool.functions.slot0().call()
        sqrt_price_x96 = slot0[0]
        
        # Calculate current price
        price = (sqrt_price_x96 / (2**96)) ** 2
        
        # Compare with other DEXs (simplified)
        external_price = await self.get_external_price(token_a, token_b)
        
        if external_price and abs(price - external_price) / external_price > 0.005:  # 0.5% threshold
            return {
                'type': 'uniswap_v3',
                'token_a': token_a,
                'token_b': token_b,
                'fee': fee,
                'pool_address': pool_address,
                'internal_price': price,
                'external_price': external_price,
                'profit_pct': abs(price - external_price) / external_price * 100
            }
        
        return None
    
    async def get_external_price(self, token_a: str, token_b: str) -> float:
        """Get price from external sources (Coingecko, other DEXs)"""
        # Simplified implementation
        return 1.0  # Would integrate with price feeds
