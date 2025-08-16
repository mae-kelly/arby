"""Base strategy interface"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class StrategySpec(ABC):
    id: str
    name: str
    family: int
    params: dict
    
    @abstractmethod
    async def discover(self, state: dict) -> List['Candidate']:
        """Discover opportunities from current state"""
        pass
    
    @abstractmethod
    async def simulate(self, candidate: 'Candidate', state: dict) -> 'SimResult':
        """Simulate execution of candidate"""
        pass
    
    @abstractmethod
    async def build_tx(self, sim: 'SimResult') -> dict:
        """Build transaction from simulation"""
        pass
