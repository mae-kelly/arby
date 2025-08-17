import os
from typing import Dict, Any
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    environment: str = "production"
    log_level: str = "INFO"
    
    redis_url: str = "redis://localhost:6379"
    database_url: str = "postgresql://user:pass@localhost:5432/arbitrage"
    
    ethereum_rpc_url: str = "https://eth-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX"
    ethereum_private_key: str = "HjFs1U5F7mbWJiDKs7izTP96MEHytvm1yiSvKLT4mEvz"
    ethereum_gas_limit: int = 2000000
    ethereum_gas_price_gwei: int = 20
    
    etherscan_api_key: str = "K4SEVFZ3PI8STM73VKV84C8PYZJUK7HB2G"
    alchemy_api_key: str = "alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX"
    
    okx_api_key: str = "8a760df1-4a2d-471b-ba42-d16893614dab"
    okx_secret_key: str = "C9F3FC89A6A30226E11DFFD098C7CF3D"
    okx_passphrase: str = ""
    
    wallet_address: str = "HjFs1U5F7mbWJiDKs7izTP96MEHytvm1yiSvKLT4mEvz"
    
    discord_webhook: str = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3"
    
    max_position_size: float = 10000.0
    min_profit_threshold: float = 0.005
    max_slippage: float = 0.01
    
    flashbots_relay_url: str = "https://relay.flashbots.net"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

EXCHANGE_CONFIGS = {
    "okx": {
        "fees": 0.001,
        "min_order_size": 0.001,
        "rate_limit": 600
    },
    "uniswap": {
        "fees": 0.003,
        "gas_estimate": 150000,
        "slippage_tolerance": 0.005
    }
}

STRATEGY_CONFIGS = {
    "flash_loan": {
        "enabled": True,
        "min_profit": 100,
        "max_gas_price": 100,
        "platforms": ["aave", "balancer"]
    },
    "cross_exchange": {
        "enabled": True,
        "min_profit": 50,
        "max_exposure": 50000,
        "exchanges": ["okx", "uniswap"]
    },
    "triangular": {
        "enabled": True,
        "min_profit": 25,
        "max_hops": 3,
        "scan_interval": 1
    },
    "liquidation": {
        "enabled": True,
        "min_profit": 100,
        "health_factor_threshold": 1.05,
        "platforms": ["aave", "compound"]
    }
}