import asyncio
import numpy as np
from typing import Dict, List, Tuple
from collections import deque

class MeanReversionStrategy:
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.price_history = {}
        self.z_score_threshold = 2.0
        self.positions = {}
    
    async def scan_mean_reversion_opportunities(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Scan for mean reversion opportunities"""
        opportunities = []
        
        # Update price history
        for symbol, price in current_prices.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.lookback_period)
            self.price_history[symbol].append(price)
        
        # Calculate opportunities
        for symbol in current_prices:
            if len(self.price_history[symbol]) >= 20:  # Minimum history
                opportunity = self.analyze_mean_reversion(symbol, current_prices[symbol])
                if opportunity:
                    opportunities.append(opportunity)
        
        return opportunities
    
    def analyze_mean_reversion(self, symbol: str, current_price: float) -> Dict:
        """Analyze mean reversion for a symbol"""
        prices = np.array(list(self.price_history[symbol]))
        
        if len(prices) < 20:
            return None
        
        # Calculate rolling statistics
        returns = np.diff(np.log(prices))
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Calculate current z-score
        if len(returns) > 0 and std_return > 0:
            current_return = np.log(current_price / prices[-2])
            z_score = (current_return - mean_return) / std_return
            
            # Check for significant deviation
            if abs(z_score) > self.z_score_threshold:
                return {
                    'type': 'mean_reversion',
                    'symbol': symbol,
                    'current_price': current_price,
                    'mean_price': np.exp(np.mean(np.log(prices))),
                    'z_score': z_score,
                    'direction': 'short' if z_score > 0 else 'long',
                    'confidence': min(abs(z_score) / 3.0, 1.0),
                    'expected_return': -z_score * std_return
                }
        
        return None
    
    def calculate_position_size(self, opportunity: Dict, max_position_value: float = 10000) -> float:
        """Calculate optimal position size using Kelly criterion"""
        confidence = opportunity['confidence']
        expected_return = abs(opportunity['expected_return'])
        
        # Simplified Kelly fraction
        kelly_fraction = confidence * expected_return / 0.1  # Assume 10% volatility
        kelly_fraction = min(kelly_fraction, 0.25)  # Cap at 25%
        
        return max_position_value * kelly_fraction
