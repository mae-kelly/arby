"""Family A: DEX Arbitrage Strategies"""
from ..base import StrategySpec
from ...execsim.simulator import Candidate
from typing import List

class UniV2UniV3Arb(StrategySpec):
    """UniV2 ↔ UniV3 two-hop arbitrage"""
    
    def __init__(self, config):
        super().__init__(
            id='a1_univ2_univ3',
            name='UniV2-UniV3 Arbitrage',
            family=1,
            params=config.strategies.get('a1_univ2_univ3', {
                'min_edge_bps': 6,
                'max_hops': 2,
                'min_liquidity_usd': 50000
            })
        )
    
    async def discover(self, state: dict) -> List[Candidate]:
        """Find UniV2/V3 price discrepancies"""
        candidates = []
        pools = state.get('pools', pd.DataFrame())
        
        if pools.empty:
            return candidates
        
        # Group pools by token pair
        v2_pools = pools[pools['type'] == 'univ2']
        v3_pools = pools[pools['type'] == 'univ3']
        
        for _, v2_pool in v2_pools.iterrows():
            # Find matching V3 pool
            matching_v3 = v3_pools[
                ((v3_pools['token0'] == v2_pool['token0']) & 
                 (v3_pools['token1'] == v2_pool['token1'])) |
                ((v3_pools['token0'] == v2_pool['token1']) & 
                 (v3_pools['token1'] == v2_pool['token0']))
            ]
            
            for _, v3_pool in matching_v3.iterrows():
                # Calculate price difference
                v2_price = v2_pool['reserve1'] / v2_pool['reserve0']
                v3_price = self._calculate_v3_price(v3_pool)
                
                price_diff_bps = abs(v2_price - v3_price) / v2_price * 10000
                
                if price_diff_bps >= self.params['min_edge_bps']:
                    # Determine direction
                    if v2_price > v3_price:
                        # Buy on V3, sell on V2
                        route = [v3_pool['address'], v2_pool['address']]
                        token_in = v2_pool['token0']
                        token_out = v2_pool['token1']
                    else:
                        # Buy on V2, sell on V3
                        route = [v2_pool['address'], v3_pool['address']]
                        token_in = v2_pool['token1']
                        token_out = v2_pool['token0']
                    
                    candidates.append(Candidate(
                        strategy_id=self.id,
                        route=route,
                        token_in=token_in,
                        token_out=token_out,
                        amount_in=int(1e18),  # 1 token base unit
                        chain=v2_pool['chain'],
                        meta={
                            'use_flash_loan': True,
                            'flash_provider': 'aave',
                            'expected_profit_bps': price_diff_bps
                        }
                    ))
        
        return candidates[:10]  # Limit to top opportunities
    
    def _calculate_v3_price(self, pool: dict) -> float:
        """Calculate UniV3 price from sqrtPriceX96"""
        sqrt_price = pool['sqrtPriceX96'] / (2**96)
        return sqrt_price ** 2
    
    async def simulate(self, candidate: Candidate, state: dict) -> 'SimResult':
        """Use common simulator"""
        from ...execsim.simulator import Simulator
        simulator = Simulator(self.config)
        return await simulator.simulate(candidate, state)
    
    async def build_tx(self, sim: 'SimResult') -> dict:
        """Build transaction"""
        return {
            'to': '0x0',  # Router address
            'data': sim.call_data,
            'value': 0,
            'gas': int(sim.diagnostics.get('gas_units', 500000))
        }

class TriangularArb(StrategySpec):
    """Triangular arbitrage within single DEX"""
    
    def __init__(self, config):
        super().__init__(
            id='a10_triangular',
            name='Triangular Arbitrage',
            family=1,
            params=config.strategies.get('a10_triangular', {
                'min_edge_bps': 8,
                'tokens': ['WETH', 'USDC', 'USDT', 'DAI', 'WBTC']
            })
        )
    
    async def discover(self, state: dict) -> List[Candidate]:
        """Find triangular arbitrage paths"""
        candidates = []
        pools = state.get('pools', pd.DataFrame())
        
        if pools.empty:
            return candidates
        
        # Build graph of token connections
        tokens = self.params['tokens']
        
        for token_a in tokens:
            for token_b in tokens:
                if token_b <= token_a:
                    continue
                    
                for token_c in tokens:
                    if token_c <= token_b or token_c == token_a:
                        continue
                    
                    # Find pools for triangle A->B->C->A
                    pool_ab = self._find_pool(pools, token_a, token_b)
                    pool_bc = self._find_pool(pools, token_b, token_c)
                    pool_ca = self._find_pool(pools, token_c, token_a)
                    
                    if pool_ab is not None and pool_bc is not None and pool_ca is not None:
                        # Calculate profitability
                        profit_ratio = self._calculate_triangle_ratio(
                            pool_ab, pool_bc, pool_ca
                        )
                        
                        profit_bps = (profit_ratio - 1) * 10000
                        
                        if profit_bps >= self.params['min_edge_bps']:
                            candidates.append(Candidate(
                                strategy_id=self.id,
                                route=[
                                    pool_ab['address'],
                                    pool_bc['address'],
                                    pool_ca['address']
                                ],
                                token_in=token_a,
                                token_out=token_a,
                                amount_in=int(1e18),
                                chain=pool_ab['chain'],
                                meta={
                                    'use_flash_loan': True,
                                    'flash_provider': 'balancer',  # 0% fee
                                    'triangle': f'{token_a}->{token_b}->{token_c}->{token_a}',
                                    'expected_profit_bps': profit_bps
                                }
                            ))
        
        return candidates[:5]
    
    def _find_pool(self, pools: pd.DataFrame, token0: str, token1: str) -> dict:
        """Find pool for token pair"""
        matches = pools[
            ((pools['token0'] == token0) & (pools['token1'] == token1)) |
            ((pools['token0'] == token1) & (pools['token1'] == token0))
        ]
        
        if not matches.empty:
            return matches.iloc[0].to_dict()
        return None
    
    def _calculate_triangle_ratio(self, pool_ab, pool_bc, pool_ca) -> float:
        """Calculate profitability of triangle"""
        # Simplified calculation
        # Would implement proper math considering fees
        
        price_ab = pool_ab['reserve1'] / pool_ab['reserve0']
        price_bc = pool_bc['reserve1'] / pool_bc['reserve0']
        price_ca = pool_ca['reserve1'] / pool_ca['reserve0']
        
        # Product of exchange rates around triangle
        ratio = price_ab * price_bc * price_ca
        
        # Account for fees (0.3% per swap for UniV2)
        ratio *= (0.997 ** 3)
        
        return ratio
    
    async def simulate(self, candidate: Candidate, state: dict) -> 'SimResult':
        from ...execsim.simulator import Simulator
        simulator = Simulator(self.config)
        return await simulator.simulate(candidate, state)
    
    async def build_tx(self, sim: 'SimResult') -> dict:
        return {
            'to': '0x0',
            'data': sim.call_data,
            'value': 0,
            'gas': int(sim.diagnostics.get('gas_units', 500000))
        }

# More strategies would be defined here...
