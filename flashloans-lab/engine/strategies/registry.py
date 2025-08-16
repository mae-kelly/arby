import asyncio
import importlib
from typing import Dict, List, Optional, Type
import yaml
from .base import StrategyBase, StrategyConfig
from .family_dex_arb import *
from .family_cross_dex import *
from .family_flash_loan import *
from .family_cex_dex import *
from .family_oracle import *
from .family_liquidity import *
from .family_liquidation import *
from .family_stablecoin import *
from .family_gas import *
from .family_cross_chain import *

class StrategyRegistry:
    def __init__(self):
        self.strategies: Dict[str, StrategyBase] = {}
        self.strategy_classes: Dict[str, Type[StrategyBase]] = {}
        self._register_built_in_strategies()
        
    def _register_built_in_strategies(self):
        """Register all built-in strategy classes"""
        
        # Family 1: Single-DEX Flash Arbitrage (10 strategies)
        self.strategy_classes.update({
            'uniswap_v2_fee_tier': UniswapV2FeeTierArb,
            'uniswap_v3_tick_crossing': UniswapV3TickCrossing,
            'uniswap_v3_fee_tier': UniswapV3FeeTierArb,
            'solidly_stable_volatile': SolidlyStableVolatile,
            'camelot_algebra': CamelotAlgebraArb,
            'sushi_trident': SushiTridentArb,
            'curve_imbalance': CurveImbalanceArb,
            'balancer_weighted': BalancerWeightedArb,
            'dodo_proactive': DodoProactiveArb,
            'bancor_v3': BancorV3Arb
        })
        
        # Family 2: Multi-DEX Same-Chain (15 strategies)
        self.strategy_classes.update({
            'uni_sushi_arb': UniSushiArb,
            'uni_curve_arb': UniCurveArb,
            'sushi_camelot_arb': SushiCamelotArb,
            'aerodrome_velodrome': AerodromeVelodromeArb,
            'triangular_weth_usdc_wbtc': TriangularEthUsdcBtc,
            'triangular_stable_loop': TriangularStableLoop,
            'quad_arb_path': QuadArbPath,
            'penta_arb_path': PentaArbPath,
            'cross_amm_curve_diff': CrossAmmCurveDiff,
            'multi_hop_split_route': MultiHopSplitRoute,
            'pool_aggregator_arb': PoolAggregatorArb,
            'router_vs_direct': RouterVsDirectArb,
            'fee_tier_cascade': FeeTierCascadeArb,
            'liquidity_fragmentation': LiquidityFragmentationArb,
            'bridge_token_arb': BridgeTokenArb
        })
        
        # Family 3: Flash Loan Loops (10 strategies)
        self.strategy_classes.update({
            'aave_flash_multi_hop': AaveFlashMultiHop,
            'balancer_flash_stable': BalancerFlashStable,
            'flash_loan_sandwich': FlashLoanSandwich,
            'compound_flash_arb': CompoundFlashArb,
            'maker_flash_arb': MakerFlashArb,
            'euler_flash_arb': EulerFlashArb,
            'iron_bank_flash': IronBankFlash,
            'yearn_vault_flash': YearnVaultFlash,
            'convex_flash_arb': ConvexFlashArb,
            'lido_flash_arb': LidoFlashArb
        })
        
        # Family 4: CEX/DEX Dislocations (10 strategies)
        self.strategy_classes.update({
            'binance_uniswap_spread': BinanceUniswapSpread,
            'coinbase_curve_spread': CoinbaseCurveSpread,
            'okx_sushi_spread': OkxSushiSpread,
            'funding_spot_basis': FundingSpotBasis,
            'perp_dex_spread': PerpDexSpread,
            'cex_depth_arb': CexDepthArb,
            'funding_rate_arb': FundingRateArb,
            'basis_trade_detector': BasisTradeDetector,
            'cross_exchange_momentum': CrossExchangeMomentum,
            'cex_dex_latency': CexDexLatencyArb
        })
        
        # Family 5: Oracle Lag/Heartbeat (10 strategies)
        self.strategy_classes.update({
            'chainlink_lag_detector': ChainlinkLagDetector,
            'band_oracle_lag': BandOracleLib,
            'twap_spot_drift': TwapSpotDrift,
            'price_feed_stale': PriceFeedStale,
            'oracle_sandwich': OracleSandwich,
            'heartbeat_edge': HeartbeatEdge,
            'oracle_frontrun': OracleFrontrun,
            'median_oracle_arb': MedianOracleArb,
            'volatility_oracle_lag': VolatilityOracleLag,
            'cross_chain_oracle': CrossChainOracleArb
        })
        
        # Family 6: Liquidity Mirages & JIT LP (10 strategies)
        self.strategy_classes.update({
            'jit_lp_detector': JitLpDetector,
            'fake_liquidity_detector': FakeLiquidityDetector,
            'lp_withdrawal_arb': LpWithdrawalArb,
            'concentrated_liquidity': ConcentratedLiquidityArb,
            'range_order_arb': RangeOrderArb,
            'lp_fee_harvesting': LpFeeHarvesting,
            'impermanent_loss_hedge': ImpermanentLossHedge,
            'liquidity_mining_arb': LiquidityMiningArb,
            'yield_farming_arb': YieldFarmingArb,
            'lp_token_arb': LpTokenArb
        })
        
        # Family 7: Liquidation Adjacent (10 strategies)
        self.strategy_classes.update({
            'aave_liquidation_detector': AaveLiquidationDetector,
            'compound_liquidation': CompoundLiquidation,
            'maker_vault_liquidation': MakerVaultLiquidation,
            'euler_liquidation': EulerLiquidation,
            'iron_bank_liquidation': IronBankLiquidation,
            'reflexer_liquidation': ReflexerLiquidation,
            'liquity_liquidation': LiquityLiquidation,
            'abracadabra_liquidation': AbracadabraLiquidation,
            'benqi_liquidation': BenqiLiquidation,
            'cream_liquidation': CreamLiquidation
        })
        
        # Family 8: Stablecoin Depeg (10 strategies)
        self.strategy_classes.update({
            'usdc_dai_depeg': UsdcDaiDepeg,
            'usdt_usdc_depeg': UsdtUsdcDepeg,
            'frax_usdc_depeg': FraxUsdcDepeg,
            'lusd_usdc_depeg': LusdUsdcDepeg,
            'mim_usdc_depeg': MimUsdcDepeg,
            'ust_depeg_detector': UstDepegDetector,
            'iron_titan_depeg': IronTitanDepeg,
            'fei_depeg_detector': FeiDepegDetector,
            'tribal_fei_depeg': TribalFeiDepeg,
            'curve_stable_depeg': CurveStableDepeg
        })
        
        # Family 9: Gas-Aware Micro-Ops (10 strategies)
        self.strategy_classes.update({
            'gas_price_arb': GasPriceArb,
            'priority_fee_arb': PriorityFeeArb,
            'base_fee_prediction': BaseFeePredicition,
            'eip1559_arb': Eip1559Arb,
            'gas_auction_arb': GasAuctionArb,
            'flashbots_bundle': FlashbotsBundleArb,
            'gas_token_arb': GasTokenArb,
            'chi_gas_arb': ChiGasArb,
            'gas_station_arb': GasStationArb,
            'mev_boost_arb': MevBoostArb
        })
        
        # Family 10: Cross-Chain & Backrun (15 strategies)  
        self.strategy_classes.update({
            'base_arbitrum_arb': BaseArbitrumArb,
            'polygon_ethereum_arb': PolygonEthereumArb,
            'optimism_arbitrum_arb': OptimismArbitrumArb,
            'avalanche_ethereum_arb': AvalancheEthereumArb,
            'bsc_ethereum_arb': BscEthereumArb,
            'backrun_large_trades': BackrunLargeTrades,
            'sandwich_protection': SandwichProtection,
            'mev_protection_arb': MevProtectionArb,
            'atomic_arb_detector': AtomicArbDetector,
            'cross_chain_latency': CrossChainLatencyArb,
            'relay_delay_arb': RelayDelayArb,
            'bridge_arb_detector': BridgeArbDetector,
            'rollup_batch_arb': RollupBatchArb,
            'state_sync_arb': StateSyncArb,
            'finality_arb': FinalityArb
        })
        
    def load_strategies_from_config(self, config_path: str = "configs/strategies.yaml") -> Dict[str, StrategyBase]:
        """Load and instantiate strategies from configuration"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        strategies = {}
        
        for strategy_name, strategy_config in config_data.get('strategies', {}).items():
            if strategy_name in self.strategy_classes:
                config = StrategyConfig(
                    name=strategy_name,
                    **strategy_config
                )
                
                strategy_class = self.strategy_classes[strategy_name]
                strategies[strategy_name] = strategy_class(config)
                
        self.strategies = strategies
        return strategies
        
    def get_strategy(self, name: str) -> Optional[StrategyBase]:
        """Get strategy by name"""
        return self.strategies.get(name)
        
    def get_enabled_strategies(self) -> Dict[str, StrategyBase]:
        """Get all enabled strategies"""
        return {name: strategy for name, strategy in self.strategies.items() 
                if strategy.enabled}
                
    def enable_strategy(self, name: str):
        """Enable a strategy"""
        if name in self.strategies:
            self.strategies[name].enabled = True
            
    def disable_strategy(self, name: str):
        """Disable a strategy"""
        if name in self.strategies:
            self.strategies[name].enabled = False
            
    def get_strategies_by_family(self, family: str) -> Dict[str, StrategyBase]:
        """Get strategies by family name"""
        family_strategies = {}
        for name, strategy in self.strategies.items():
            if name.startswith(family) or family in name:
                family_strategies[name] = strategy
        return family_strategies
        
    def list_all_strategies(self) -> List[str]:
        """List all available strategy names"""
        return list(self.strategy_classes.keys())
