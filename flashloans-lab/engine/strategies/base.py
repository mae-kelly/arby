import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ..execsim.simulator import CandidateTrade
from ..pricing.route_search import Route

@dataclass
class StrategyConfig:
    name: str
    enabled: bool = True
    min_edge_bps: int = 5
    max_hops: int = 3
    max_gas: int = 500_000
    pool_depth_floor: int = 10_000
    max_position_size: int = 1_000_000
    confidence_threshold: float = 0.7
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}

@dataclass
class StrategyDetectionResult:
    opportunity_found: bool
    candidate_trade: Optional[CandidateTrade] = None
    confidence: float = 0.0
    edge_bps: float = 0.0
    risk_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class StrategyBase(ABC):
    """Base class for all arbitrage strategies"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.detection_count = 0
        self.execution_count = 0
        self.last_detection_time = 0
        
    @abstractmethod
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        """
        Detect arbitrage opportunities
        
        Args:
            market_data: Current market state including:
                - pools: Dict of pool states
                - prices: Current token prices
                - gas: Current gas prices
                - block: Current block info
                
        Returns:
            StrategyDetectionResult with opportunity details
        """
        pass
        
    @abstractmethod
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        """
        Simulate trade execution and return expected PnL
        
        Args:
            candidate: Trade to simulate
            market_data: Current market state
            
        Returns:
            Expected profit in wei (negative for loss)
        """
        pass
        
    @abstractmethod
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        """
        Calculate risk score for trade (0.0 = low risk, 1.0 = high risk)
        
        Args:
            candidate: Trade to evaluate
            
        Returns:
            Risk score between 0.0 and 1.0
        """
        pass
        
    @abstractmethod
    def get_required_tokens(self) -> List[str]:
        """
        Get list of tokens this strategy monitors
        
        Returns:
            List of token addresses
        """
        pass
        
    def should_execute(self, result: StrategyDetectionResult) -> bool:
        """
        Decide if opportunity should be executed
        
        Args:
            result: Detection result
            
        Returns:
            True if trade should be executed
        """
        if not self.enabled or not result.opportunity_found:
            return False
            
        # Check minimum edge requirement
        if result.edge_bps < self.config.min_edge_bps:
            return False
            
        # Check confidence threshold
        if result.confidence < self.config.confidence_threshold:
            return False
            
        # Check risk limits
        if result.risk_score > 0.8:  # High risk threshold
            return False
            
        return True
        
    def update_metrics(self, detected: bool, executed: bool = False):
        """Update strategy performance metrics"""
        if detected:
            self.detection_count += 1
            
        if executed:
            self.execution_count += 1
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'detections': self.detection_count,
            'executions': self.execution_count,
            'execution_rate': self.execution_count / max(self.detection_count, 1),
            'config': self.config.__dict__
        }

class FlashLoanStrategy(StrategyBase):
    """Base class for flash loan arbitrage strategies"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.flash_loan_protocols = ['aave_v3', 'balancer', 'uniswap_v3']
        
    def get_optimal_flash_loan_protocol(self, token: str, amount: int) -> str:
        """Determine best flash loan protocol for given token/amount"""
        # Aave v3: 0.05% fee but wide token support
        # Balancer: 0% fee but limited tokens
        # Uniswap v3: 0% fee but only for pool tokens
        
        # For demo, prefer Balancer (0% fee) when available
        if token.lower() in self._get_balancer_vault_tokens():
            return 'balancer'
        elif token.lower() in self._get_uniswap_v3_tokens():
            return 'uniswap_v3'
        else:
            return 'aave_v3'  # Fallback with fee
            
    def _get_balancer_vault_tokens(self) -> List[str]:
        """Get tokens available for Balancer flash loans"""
        return [
            '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # WETH
            '0xA0b86a33E6441b920e5E6C4F27ea9c0c2f2f8F69',  # USDC
            '0x6B175474E89094C44Da98b954EedeAC495271d0F',   # DAI
        ]
        
    def _get_uniswap_v3_tokens(self) -> List[str]:
        """Get tokens available for Uniswap v3 flash loans"""
        return [
            '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # WETH
            '0xA0b86a33E6441b920e5E6C4F27ea9c0c2f2f8F69',  # USDC
            '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',  # WBTC
        ]
        
    def calculate_flash_loan_fee(self, protocol: str, amount: int) -> int:
        """Calculate flash loan fee for protocol"""
        if protocol == 'balancer' or protocol == 'uniswap_v3':
            return 0  # Free flash loans
        elif protocol == 'aave_v3':
            return amount * 5 // 10000  # 0.05% = 5 bps
        else:
            return amount * 30 // 10000  # 0.3% fallback
        if result.edge_bps < self.config.min_edge_bps:
            return False
            
        # Check confidence threshold
        if result.confidence < self.config.confidence_threshold:
            return False
            
        # Check risk score
        if result.risk_score > 0.8:  # High risk threshold
            return False
            
        return True
        
    def update_config(self, new_config: Dict[str, Any]):
        """Update strategy configuration"""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.params[key] = value
                
    def get_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'detection_count': self.detection_count,
            'execution_count': self.execution_count,
            'last_detection_time': self.last_detection_time,
            'config': {
                'min_edge_bps': self.config.min_edge_bps,
                'max_hops': self.config.max_hops,
                'confidence_threshold': self.config.confidence_threshold
            }
        }

class FlashLoanStrategy(StrategyBase):
    """Base class for flash loan arbitrage strategies"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.flash_loan_protocols = ['aave_v3', 'balancer', 'uniswap_v3']
        
    def get_optimal_flash_loan_protocol(self, token: str, amount: int) -> str:
        """Determine best flash loan protocol for given token/amount"""
        # Aave v3: 0.05% fee, high liquidity
        # Balancer: 0% fee, medium liquidity  
        # Uniswap v3: Variable fee, depends on pool
        
        # For demo, prefer Balancer (0% fee) if available
        return 'balancer'
        
    def calculate_flash_loan_fee(self, protocol: str, amount: int) -> int:
        """Calculate flash loan fee for protocol"""
        fees = {
            'aave_v3': 5,      # 5 bps = 0.05%
            'balancer': 0,     # 0%
            'uniswap_v3': 0    # Variable, assume 0 for now
        }
        
        fee_bps = fees.get(protocol, 5)
        return amount * fee_bps // 10000
        if result.edge_bps < self.config.min_edge_bps:
            return False
            
        # Check confidence threshold
        if result.confidence < self.config.confidence_threshold:
            return False
            
        # Check risk score
        if result.risk_score > 0.8:  # High risk threshold
            return False
            
        return True
        
    def update_config(self, new_config: Dict[str, Any]):
        """Update strategy configuration"""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.params[key] = value
                
    def get_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'detection_count': self.detection_count,
            'execution_count': self.execution_count,
            'last_detection_time': self.last_detection_time,
            'config': {
                'min_edge_bps': self.config.min_edge_bps,
                'max_hops': self.config.max_hops,
                'confidence_threshold': self.config.confidence_threshold
            }
        }

class FlashLoanStrategy(StrategyBase):
    """Base class for flash loan arbitrage strategies"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.flash_loan_protocols = ['aave_v3', 'balancer', 'uniswap_v3']
        
    def get_optimal_flash_loan_protocol(self, token: str, amount: int) -> str:
        """Determine best flash loan protocol for given token/amount"""
        # Aave v3: 0.05% fee, high liquidity
        # Balancer: 0% fee, medium liquidity  
        # Uniswap v3: Variable fee, depends on pool
        
        # For demo, prefer Balancer (0% fee) if available
        return 'balancer'
        
    def calculate_flash_loan_fee(self, protocol: str, amount: int) -> int:
        """Calculate flash loan fee for protocol"""
        fees = {
            'aave_v3': 5,      # 5 bps = 0.05%
            'balancer': 0,     # 0%
            'uniswap_v3': 0    # Variable, assume 0 for now
        }
        
        fee_bps = fees.get(protocol, 5)
        return amount * fee_bps // 10000
