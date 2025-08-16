import asyncio
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import heapq
from ..datasources.dex_index import PoolInfo
from .amm_math import AMMCalculator

@dataclass
class RouteHop:
    pool: PoolInfo
    token_in: str
    token_out: str
    amount_in: int
    amount_out: int
    
@dataclass
class Route:
    hops: List[RouteHop]
    total_amount_in: int
    total_amount_out: int
    gas_estimate: int
    
    @property
    def tokens(self) -> List[str]:
        if not self.hops:
            return []
        tokens = [self.hops[0].token_in]
        for hop in self.hops:
            tokens.append(hop.token_out)
        return tokens
        
    def calculate_price_impact(self) -> float:
        """Calculate total price impact across route"""
        if not self.hops or self.total_amount_in == 0:
            return 0.0
            
        # Simple approximation: sum of individual hop impacts
        total_impact = 0.0
        for hop in self.hops:
            if hop.pool.pool_type == "uniswap_v2" and hop.pool.reserve0 and hop.pool.reserve1:
                if hop.token_in.lower() == hop.pool.token0.lower():
                    reserve_in = hop.pool.reserve0
                else:
                    reserve_in = hop.pool.reserve1
                    
                # Price impact ≈ amount_in / (2 * reserve_in)
                impact = hop.amount_in / (2 * reserve_in) if reserve_in > 0 else 0
                total_impact += impact
                
        return min(total_impact, 1.0)  # Cap at 100%

class RouteSearcher:
    def __init__(self, dex_indexer):
        self.dex_indexer = dex_indexer
        self.calculator = AMMCalculator()
        
    async def find_routes(self, 
                         token_in: str, 
                         token_out: str, 
                         amount_in: int,
                         max_hops: int = 3,
                         min_liquidity: int = 10000) -> List[Route]:
        """Find optimal routes between tokens"""
        
        routes = []
        
        # Direct routes (1 hop)
        direct_pools = await self.dex_indexer.get_pools_for_tokens(token_in, token_out)
        for pool in direct_pools:
            route = await self._calculate_single_hop(pool, token_in, token_out, amount_in)
            if route and self._meets_liquidity_threshold(route, min_liquidity):
                routes.append(route)
                
        # Multi-hop routes
        if max_hops > 1:
            multi_hop_routes = await self._find_multi_hop_routes(
                token_in, token_out, amount_in, max_hops, min_liquidity
            )
            routes.extend(multi_hop_routes)
            
        # Sort by output amount (best first)
        routes.sort(key=lambda r: r.total_amount_out, reverse=True)
        return routes[:10]  # Return top 10 routes
        
    async def _calculate_single_hop(self, 
                                   pool: PoolInfo, 
                                   token_in: str, 
                                   token_out: str, 
                                   amount_in: int) -> Optional[Route]:
        """Calculate single hop route"""
        try:
            if pool.pool_type == "uniswap_v2":
                amount_out = self._calculate_v2_output(pool, token_in, amount_in)
            elif pool.pool_type == "uniswap_v3":
                amount_out = self._calculate_v3_output(pool, token_in, amount_in)
            elif pool.pool_type == "solidly":
                amount_out = self._calculate_solidly_output(pool, token_in, amount_in)
            else:
                return None
                
            if amount_out <= 0:
                return None
                
            hop = RouteHop(
                pool=pool,
                token_in=token_in,
                token_out=token_out,
                amount_in=amount_in,
                amount_out=amount_out
            )
            
            return Route(
                hops=[hop],
                total_amount_in=amount_in,
                total_amount_out=amount_out,
                gas_estimate=150000  # Base gas estimate
            )
            
        except Exception:
            return None
            
    def _calculate_v2_output(self, pool: PoolInfo, token_in: str, amount_in: int) -> int:
        """Calculate Uniswap V2 output"""
        if not pool.reserve0 or not pool.reserve1:
            return 0
            
        if token_in.lower() == pool.token0.lower():
            return self.calculator.uniswap_v2_get_amount_out(
                amount_in, pool.reserve0, pool.reserve1
            )
        else:
            return self.calculator.uniswap_v2_get_amount_out(
                amount_in, pool.reserve1, pool.reserve0
            )
            
    def _calculate_v3_output(self, pool: PoolInfo, token_in: str, amount_in: int) -> int:
        """Calculate Uniswap V3 output"""
        if not pool.sqrt_price_x96 or not pool.liquidity:
            return 0
            
        zero_for_one = token_in.lower() == pool.token0.lower()
        amount_out, _ = self.calculator.uniswap_v3_get_amount_out(
            amount_in, pool.sqrt_price_x96, pool.liquidity, pool.fee, zero_for_one
        )
        return amount_out
        
    def _calculate_solidly_output(self, pool: PoolInfo, token_in: str, amount_in: int) -> int:
        """Calculate Solidly output"""
        if not pool.reserve0 or not pool.reserve1:
            return 0
            
        # Assume stable if fee < 25 bps
        stable = pool.fee < 25
        
        if token_in.lower() == pool.token0.lower():
            return self.calculator.solidly_get_amount_out(
                amount_in, pool.reserve0, pool.reserve1, stable, pool.fee
            )
        else:
            return self.calculator.solidly_get_amount_out(
                amount_in, pool.reserve1, pool.reserve0, stable, pool.fee
            )
            
    async def _find_multi_hop_routes(self, 
                                   token_in: str, 
                                   token_out: str, 
                                   amount_in: int,
                                   max_hops: int, 
                                   min_liquidity: int) -> List[Route]:
        """Find multi-hop routes using BFS"""
        routes = []
        
        # Common bridge tokens for routing
        bridge_tokens = {
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "0xA0b86a33E6441b920e5E6C4F27ea9c0c2f2f8F69",  # USDC
            "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
        }
        
        # Try 2-hop routes through bridge tokens
        if max_hops >= 2:
            for bridge_token in bridge_tokens:
                if bridge_token == token_in or bridge_token == token_out:
                    continue
                    
                # First hop: token_in -> bridge_token
                first_hop_pools = await self.dex_indexer.get_pools_for_tokens(token_in, bridge_token)
                for pool1 in first_hop_pools:
                    intermediate_amount = self._calculate_output_for_pool(pool1, token_in, amount_in)
                    if intermediate_amount <= 0:
                        continue
                        
                    # Second hop: bridge_token -> token_out
                    second_hop_pools = await self.dex_indexer.get_pools_for_tokens(bridge_token, token_out)
                    for pool2 in second_hop_pools:
                        final_amount = self._calculate_output_for_pool(pool2, bridge_token, intermediate_amount)
                        if final_amount <= 0:
                            continue
                            
                        route = Route(
                            hops=[
                                RouteHop(pool1, token_in, bridge_token, amount_in, intermediate_amount),
                                RouteHop(pool2, bridge_token, token_out, intermediate_amount, final_amount)
                            ],
                            total_amount_in=amount_in,
                            total_amount_out=final_amount,
                            gas_estimate=250000  # Higher gas for multi-hop
                        )
                        
                        if self._meets_liquidity_threshold(route, min_liquidity):
                            routes.append(route)
                            
        return routes
        
    def _calculate_output_for_pool(self, pool: PoolInfo, token_in: str, amount_in: int) -> int:
        """Generic output calculation for any pool type"""
        if pool.pool_type == "uniswap_v2":
            return self._calculate_v2_output(pool, token_in, amount_in)
        elif pool.pool_type == "uniswap_v3":
            return self._calculate_v3_output(pool, token_in, amount_in)
        elif pool.pool_type == "solidly":
            return self._calculate_solidly_output(pool, token_in, amount_in)
        return 0
        
    def _meets_liquidity_threshold(self, route: Route, min_liquidity: int) -> bool:
        """Check if route meets minimum liquidity requirements"""
        for hop in route.hops:
            if hop.pool.pool_type == "uniswap_v2":
                if not hop.pool.reserve0 or not hop.pool.reserve1:
                    return False
                min_reserve = min(hop.pool.reserve0, hop.pool.reserve1)
                if min_reserve < min_liquidity:
                    return False
            elif hop.pool.pool_type == "uniswap_v3":
                if not hop.pool.liquidity or hop.pool.liquidity < min_liquidity:
                    return False
                    
        return True
