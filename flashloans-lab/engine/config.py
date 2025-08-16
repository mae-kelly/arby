import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml

@dataclass
class ChainConfig:
    name: str
    rpc_url: str
    chain_id: int
    gas_multiplier: float = 1.0
    block_time: float = 2.0
    
@dataclass
class StrategyConfig:
    enabled: bool = True
    min_edge_bps: int = 5
    max_hops: int = 3
    max_gas: int = 500000
    pool_depth_floor: int = 10000
    
@dataclass
class Config:
    chains: Dict[str, ChainConfig] = field(default_factory=dict)
    strategies: Dict[str, StrategyConfig] = field(default_factory=dict)
    risk_limits: Dict[str, float] = field(default_factory=lambda: {
        'max_drawdown_pct': 5.0,
        'kelly_fraction': 0.25,
        'var_confidence': 0.95
    })
    
    @classmethod
    def load(cls, config_dir: str = "configs") -> "Config":
        config = cls()
        
        # Load chain configs
        with open(f"{config_dir}/chains.yaml", "r") as f:
            chain_data = yaml.safe_load(f)
            for name, data in chain_data["chains"].items():
                config.chains[name] = ChainConfig(**data)
        
        # Load strategy configs
        with open(f"{config_dir}/strategies.yaml", "r") as f:
            strategy_data = yaml.safe_load(f)
            for name, data in strategy_data["strategies"].items():
                config.strategies[name] = StrategyConfig(**data)
                
        return config
