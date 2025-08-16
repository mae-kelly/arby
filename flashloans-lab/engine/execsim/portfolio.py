import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import statistics
from .simulator import SimulationResult

@dataclass
class StrategyMetrics:
    name: str
    total_trades: int = 0
    successful_trades: int = 0
    total_pnl: int = 0
    total_volume: int = 0
    max_drawdown: int = 0
    current_drawdown: int = 0
    peak_pnl: int = 0
    win_rate: float = 0.0
    avg_profit_per_trade: float = 0.0
    sharpe_ratio: float = 0.0
    recent_pnl: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, result: SimulationResult):
        """Update metrics with new trade result"""
        self.total_trades += 1
        self.total_volume += result.trade.flash_loan_amount
        
        if result.success:
            self.successful_trades += 1
            
        # Update PnL tracking
        self.total_pnl += result.net_pnl
        self.recent_pnl.append(result.net_pnl)
        
        # Update drawdown
        if self.total_pnl > self.peak_pnl:
            self.peak_pnl = self.total_pnl
            self.current_drawdown = 0
        else:
            self.current_drawdown = self.peak_pnl - self.total_pnl
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
        # Update derived metrics
        self.win_rate = self.successful_trades / self.total_trades if self.total_trades > 0 else 0
        self.avg_profit_per_trade = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        if len(self.recent_pnl) > 10:
            returns = list(self.recent_pnl)
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 1
            self.sharpe_ratio = mean_return / std_return if std_return > 0 else 0

@dataclass
class RiskLimits:
    max_drawdown_pct: float = 5.0
    max_position_size: int = 1_000_000  # 1M USD equivalent
    max_daily_trades: int = 1000
    kelly_fraction: float = 0.25
    var_confidence: float = 0.95
    circuit_breaker_loss: int = 100_000  # Stop if lose 100k
    
class PortfolioManager:
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        self.global_pnl = 0
        self.global_volume = 0
        self.daily_trades = 0
        self.last_reset = time.time()
        self.emergency_stop = False
        self.token_exposures: Dict[str, int] = defaultdict(int)
        
    def register_strategy(self, strategy_name: str):
        """Register a new strategy for tracking"""
        if strategy_name not in self.strategy_metrics:
            self.strategy_metrics[strategy_name] = StrategyMetrics(name=strategy_name)
            
    def record_trade(self, result: SimulationResult):
        """Record a completed trade"""
        strategy_name = result.trade.strategy_name
        
        # Ensure strategy is registered
        self.register_strategy(strategy_name)
        
        # Update strategy metrics
        self.strategy_metrics[strategy_name].update(result)
        
        # Update global metrics
        self.global_pnl += result.net_pnl
        self.global_volume += result.trade.flash_loan_amount
        self.daily_trades += 1
        
        # Update token exposures (simplified)
        for hop in result.trade.route.hops:
            self.token_exposures[hop.token_in] += hop.amount_in
            self.token_exposures[hop.token_out] -= hop.amount_out
            
        # Check risk limits
        self._check_risk_limits()
        
        # Reset daily counters if needed
        current_time = time.time()
        if current_time - self.last_reset > 86400:  # 24 hours
            self.daily_trades = 0
            self.last_reset = current_time
            
    def can_trade(self, strategy_name: str, trade_size: int) -> Tuple[bool, str]:
        """Check if a trade is allowed under risk limits"""
        
        if self.emergency_stop:
            return False, "Emergency stop activated"
            
        # Check daily trade limit
        if self.daily_trades >= self.risk_limits.max_daily_trades:
            return False, "Daily trade limit exceeded"
            
        # Check position size limit
        if trade_size > self.risk_limits.max_position_size:
            return False, f"Trade size {trade_size} exceeds limit {self.risk_limits.max_position_size}"
            
        # Check strategy-specific limits
        if strategy_name in self.strategy_metrics:
            metrics = self.strategy_metrics[strategy_name]
            
            # Check drawdown limit
            drawdown_pct = (metrics.current_drawdown / max(metrics.peak_pnl, 1)) * 100
            if drawdown_pct > self.risk_limits.max_drawdown_pct:
                return False, f"Strategy drawdown {drawdown_pct:.1f}% exceeds limit"
                
            # Check win rate (disable if too low)
            if metrics.total_trades > 50 and metrics.win_rate < 0.2:
                return False, f"Strategy win rate {metrics.win_rate:.1f} too low"
                
        return True, "OK"
        
    def get_kelly_fraction(self, strategy_name: str) -> float:
        """Calculate Kelly-optimal position sizing"""
        if strategy_name not in self.strategy_metrics:
            return 0.1  # Conservative default
            
        metrics = self.strategy_metrics[strategy_name]
        
        if metrics.total_trades < 20:
            return 0.1  # Not enough data
            
        # Simplified Kelly: f = (bp - q) / b
        # where b = odds, p = win prob, q = lose prob
        win_rate = metrics.win_rate
        lose_rate = 1 - win_rate
        
        if win_rate <= 0 or lose_rate <= 0:
            return 0.05
            
        # Estimate average win/loss ratio
        if len(metrics.recent_pnl) < 10:
            return 0.1
            
        wins = [pnl for pnl in metrics.recent_pnl if pnl > 0]
        losses = [abs(pnl) for pnl in metrics.recent_pnl if pnl < 0]
        
        if not wins or not losses:
            return 0.1
            
        avg_win = statistics.mean(wins)
        avg_loss = statistics.mean(losses)
        
        if avg_loss == 0:
            return 0.1
            
        odds = avg_win / avg_loss
        kelly = (odds * win_rate - lose_rate) / odds
        
        # Apply Kelly fraction limit and conservative scaling
        return max(0.01, min(kelly * self.risk_limits.kelly_fraction, 0.25))
        
    def _check_risk_limits(self):
        """Check global risk limits and trigger emergency stop if needed"""
        
        # Check global loss limit
        if self.global_pnl < -self.risk_limits.circuit_breaker_loss:
            self.emergency_stop = True
            print(f"⚠️ EMERGENCY STOP: Global PnL {self.global_pnl} below limit")
            
        # Check individual strategy limits
        for strategy_name, metrics in self.strategy_metrics.items():
            if metrics.current_drawdown > self.risk_limits.circuit_breaker_loss // 2:
                print(f"⚠️ WARNING: Strategy {strategy_name} high drawdown {metrics.current_drawdown}")
                
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary statistics"""
        total_trades = sum(m.total_trades for m in self.strategy_metrics.values())
        successful_trades = sum(m.successful_trades for m in self.strategy_metrics.values())
        
        return {
            'global_pnl': self.global_pnl,
            'global_volume': self.global_volume, 
            'total_trades': total_trades,
            'global_win_rate': successful_trades / total_trades if total_trades > 0 else 0,
            'daily_trades': self.daily_trades,
            'emergency_stop': self.emergency_stop,
            'active_strategies': len(self.strategy_metrics),
            'top_strategies': sorted(
                [(name, m.total_pnl) for name, m in self.strategy_metrics.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
        }
        
    def reset_emergency_stop(self):
        """Reset emergency stop (manual intervention)"""
        self.emergency_stop = False
        print("🔄 Emergency stop reset")
