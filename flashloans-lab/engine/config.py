"""Configuration management for the bot"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
import os
from pathlib import Path

@dataclass
class Config:
    # Chain settings
    chains: Dict[str, dict]
    
    # Strategy settings
    strategies: Dict[str, dict]
    
    # Risk settings
    min_ev_native: float = 0.005
    max_price_impact_bps: int = 30
    max_gas_native: float = 0.002
    reject_transfer_tax: bool = True
    token_blacklist: List[str] = None
    
    # Execution settings
    loop_seconds: int = 2
    simulation_threads: int = 4
    
    @classmethod
    def load(cls, path: str = 'configs/config.yaml') -> 'Config':
        """Load configuration from YAML files"""
        config_data = {}
        
        # Load all config files
        config_dir = Path('configs')
        for config_file in ['chains.yaml', 'strategies.yaml', 'risk.yaml']:
            file_path = config_dir / config_file
            if file_path.exists():
                with open(file_path) as f:
                    data = yaml.safe_load(f)
                    config_data.update(data)
        
        # Override with environment variables
        if os.getenv('MIN_EV_NATIVE'):
            config_data['min_ev_native'] = float(os.getenv('MIN_EV_NATIVE'))
        
        return cls(**config_data)
    
    def get_rpc(self, chain: str) -> str:
        """Get RPC endpoint for chain"""
        return self.chains.get(chain, {}).get('rpc')
