import numpy as np
import pandas as pd
from gym import Env, spaces
from datetime import datetime, timedelta
import random

class TradingEnvironment(Env):
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001, 
                 max_position=1.0, lookback_window=20):
        super(TradingEnvironment, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.lookback_window = lookback_window
        
        self.action_space = spaces.Discrete(3)  # 0: Sell, 1: Hold, 2: Buy
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback_window * 8,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self):
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_value = self.initial_balance
        self.trades = []
        self.portfolio_history = []
        
        return self._get_observation()
        
    def _get_observation(self):
        if self.current_step < self.lookback_window:
            return np.zeros(self.lookback_window * 8)
            
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        features = []
        
        prices = window_data['price'].values
        volumes = window_data['volume'].values
        
        price_returns = np.diff(prices) / prices[:-1]
        price_returns = np.append([0], price_returns)
        
        volume_changes = np.diff(volumes) / volumes[:-1]
        volume_changes = np.append([0], volume_changes)
        
        normalized_prices = (prices - prices.mean()) / (prices.std() + 1e-8)
        normalized_volumes = (volumes - volumes.mean()) / (volumes.std() + 1e-8)
        
        rsi_values = window_data.get('rsi', np.full(len(window_data), 50)).values
        macd_values = window_data.get('macd', np.zeros(len(window_data))).values
        
        sma_10 = window_data.get('sma_10', prices).values
        sma_50 = window_data.get('sma_50', prices).values
        
        features.extend(normalized_prices)
        features.extend(normalized_volumes)
        features.extend(price_returns)
        features.extend(volume_changes)
        features.extend((rsi_values - 50) / 50)
        features.extend(macd_values / (np.abs(macd_values).max() + 1e-8))
        features.extend((prices - sma_10) / (sma_10 + 1e-8))
        features.extend((prices - sma_50) / (sma_50 + 1e-8))
        
        return np.array(features, dtype=np.float32)
        
    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
            
        current_price = self.data.iloc[self.current_step]['price']
        next_price = self.data.iloc[self.current_step + 1]['price']
        
        old_total_value = self.total_value
        
        if action == 0:  # Sell
            if self.position > 0:
                sell_amount = min(self.position, self.max_position)
                self.balance += sell_amount * current_price * (1 - self.transaction_cost)
                self.position -= sell_amount
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'amount': sell_amount,
                    'price': current_price,
                    'balance': self.balance,
                    'position': self.position
                })
                
        elif action == 2:  # Buy
            max_buy = self.balance / (current_price * (1 + self.transaction_cost))
            buy_amount = min(max_buy, self.max_position - self.position)
            
            if buy_amount > 0.001:  # Minimum trade size
                cost = buy_amount * current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.position += buy_amount
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'amount': buy_amount,
                    'price': current_price,
                    'balance': self.balance,
                    'position': self.position
                })
        
        self.current_step += 1
        
        self.total_value = self.balance + self.position * next_price
        
        reward = self._calculate_reward(old_total_value, self.total_value, action)
        
        self.portfolio_history.append({
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'total_value': self.total_value,
            'price': next_price
        })
        
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'total_value': self.total_value,
            'balance': self.balance,
            'position': self.position,
            'trades': len(self.trades),
            'current_price': next_price
        }
        
        return self._get_observation(), reward, done, info
        
    def _calculate_reward(self, old_value, new_value, action):
        portfolio_return = (new_value - old_value) / old_value if old_value > 0 else 0
        
        if self.current_step > self.lookback_window:
            market_return = (self.data.iloc[self.current_step]['price'] - 
                           self.data.iloc[self.current_step - 1]['price']) / \
                           self.data.iloc[self.current_step - 1]['price']
        else:
            market_return = 0
        
        alpha = portfolio_return - market_return
        
        if len(self.portfolio_history) >= 2:
            values = [p['total_value'] for p in self.portfolio_history[-20:]]
            if len(values) > 1:
                volatility = np.std(np.diff(values) / values[:-1])
                sharpe_ratio = np.mean(np.diff(values) / values[:-1]) / (volatility + 1e-8)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        drawdown_penalty = self._calculate_drawdown_penalty()
        
        transaction_penalty = 0
        if len(self.trades) > 0 and self.trades[-1]['step'] == self.current_step - 1:
            transaction_penalty = -0.001
        
        reward = alpha * 10 + sharpe_ratio * 0.1 + drawdown_penalty + transaction_penalty
        
        return reward
        
    def _calculate_drawdown_penalty(self):
        if len(self.portfolio_history) < 2:
            return 0
            
        values = [p['total_value'] for p in self.portfolio_history]
        peak = np.maximum.accumulate(values)
        drawdown = (values[-1] - peak[-1]) / peak[-1]
        
        return min(drawdown * 10, 0)
        
    def get_portfolio_stats(self):
        if len(self.portfolio_history) < 2:
            return {}
            
        values = [p['total_value'] for p in self.portfolio_history]
        returns = np.diff(values) / values[:-1]
        
        total_return = (values[-1] - self.initial_balance) / self.initial_balance
        
        if len(returns) > 0:
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = np.mean(returns) * 252 / (volatility + 1e-8)
        else:
            volatility = 0
            sharpe_ratio = 0
        
        peak = np.maximum.accumulate(values)
        drawdown = (np.array(values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'final_value': values[-1]
        }

class ArbitrageEnvironment(Env):
    def __init__(self, pool_data, gas_costs, initial_balance=1000, 
                 min_profit_threshold=0.001, max_trade_size=100):
        super(ArbitrageEnvironment, self).__init__()
        
        self.pool_data = pool_data
        self.gas_costs = gas_costs
        self.initial_balance = initial_balance
        self.min_profit_threshold = min_profit_threshold
        self.max_trade_size = max_trade_size
        
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.total_profit = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.gas_spent = 0
        
        return self._get_observation()
        
    def _get_observation(self):
        if self.current_step >= len(self.pool_data):
            return np.zeros(15)
            
        current_data = self.pool_data.iloc[self.current_step]
        
        features = [
            current_data.get('pool1_price', 0),
            current_data.get('pool2_price', 0),
            current_data.get('price_diff', 0),
            current_data.get('pool1_liquidity', 0),
            current_data.get('pool2_liquidity', 0),
            current_data.get('pool1_volume', 0),
            current_data.get('pool2_volume', 0),
            current_data.get('gas_price', 0),
            current_data.get('network_congestion', 0),
            self.balance / self.initial_balance,
            self.total_profit / self.initial_balance,
            self.successful_trades,
            self.failed_trades,
            self.gas_spent / self.initial_balance,
            current_data.get('market_volatility', 0)
        ]
        
        return np.array(features, dtype=np.float32)
        
    def step(self, action):
        if self.current_step >= len(self.pool_data) - 1:
            return self._get_observation(), 0, True, {}
            
        current_data = self.pool_data.iloc[self.current_step]
        
        trade_size_ratio = action[0]
        route_choice = action[1]
        
        trade_size = trade_size_ratio * min(self.max_trade_size, self.balance)
        
        if trade_size < 0.01:
            reward = -0.001
        else:
            reward = self._execute_arbitrage(current_data, trade_size, route_choice)
        
        self.current_step += 1
        
        done = self.current_step >= len(self.pool_data) - 1
        
        info = {
            'balance': self.balance,
            'total_profit': self.total_profit,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'success_rate': self.successful_trades / max(1, self.successful_trades + self.failed_trades)
        }
        
        return self._get_observation(), reward, done, info
        
    def _execute_arbitrage(self, data, trade_size, route_choice):
        pool1_price = data.get('pool1_price', 0)
        pool2_price = data.get('pool2_price', 0)
        gas_price = data.get('gas_price', 20)
        
        if pool1_price == 0 or pool2_price == 0:
            return -0.01
        
        price_diff = abs(pool1_price - pool2_price) / min(pool1_price, pool2_price)
        
        if price_diff < self.min_profit_threshold:
            return -0.001
        
        gas_cost = self._calculate_gas_cost(gas_price, route_choice)
        slippage = self._calculate_slippage(data, trade_size)
        
        gross_profit = trade_size * price_diff
        net_profit = gross_profit - gas_cost - slippage
        
        success_probability = self._calculate_success_probability(data, trade_size)
        
        if random.random() < success_probability and net_profit > 0:
            self.balance += net_profit
            self.total_profit += net_profit
            self.successful_trades += 1
            self.gas_spent += gas_cost
            
            reward = net_profit / trade_size
        else:
            self.balance -= gas_cost
            self.failed_trades += 1
            self.gas_spent += gas_cost
            
            reward = -gas_cost / trade_size
        
        return reward
        
    def _calculate_gas_cost(self, gas_price, route_choice):
        base_gas = 150000 if route_choice < 0.5 else 250000
        return base_gas * gas_price / 1e9 * 0.001
        
    def _calculate_slippage(self, data, trade_size):
        pool1_liquidity = data.get('pool1_liquidity', 1000000)
        pool2_liquidity = data.get('pool2_liquidity', 1000000)
        
        min_liquidity = min(pool1_liquidity, pool2_liquidity)
        slippage_rate = (trade_size / min_liquidity) ** 0.5 * 0.01
        
        return trade_size * slippage_rate
        
    def _calculate_success_probability(self, data, trade_size):
        base_prob = 0.95
        
        network_congestion = data.get('network_congestion', 0.5)
        prob_adjustment = network_congestion * 0.3
        
        liquidity_factor = min(1.0, data.get('pool1_liquidity', 0) / 1000000)
        prob_adjustment += (1 - liquidity_factor) * 0.2
        
        return max(0.1, base_prob - prob_adjustment)

class MultiAssetEnvironment(Env):
    def __init__(self, data_dict, initial_balance=10000, max_positions=3):
        super(MultiAssetEnvironment, self).__init__()
        
        self.data_dict = data_dict
        self.assets = list(data_dict.keys())
        self.initial_balance = initial_balance
        self.max_positions = max_positions
        
        num_assets = len(self.assets)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(num_assets,), dtype=np.float32
        )
        
        obs_size = num_assets * 10 + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self):
        self.current_step = 20
        self.balance = self.initial_balance
        self.positions = {asset: 0.0 for asset in self.assets}
        self.total_value = self.initial_balance
        self.portfolio_history = []
        
        return self._get_observation()
        
    def _get_observation(self):
        features = []
        
        for asset in self.assets:
            asset_data = self.data_dict[asset]
            
            if self.current_step < len(asset_data):
                current_price = asset_data.iloc[self.current_step]['price']
                
                recent_prices = asset_data.iloc[max(0, self.current_step-10):self.current_step]['price'].values
                if len(recent_prices) > 1:
                    returns = np.diff(recent_prices) / recent_prices[:-1]
                    volatility = np.std(returns)
                    momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                else:
                    volatility = 0
                    momentum = 0
                
                rsi = asset_data.iloc[self.current_step].get('rsi', 50)
                macd = asset_data.iloc[self.current_step].get('macd', 0)
                
                features.extend([
                    current_price / 1000,
                    volatility,
                    momentum,
                    (rsi - 50) / 50,
                    macd / 100,
                    self.positions[asset],
                    recent_prices[-1] / recent_prices[0] - 1 if len(recent_prices) > 1 else 0,
                    asset_data.iloc[self.current_step].get('volume', 0) / 1000000,
                    asset_data.iloc[self.current_step].get('sma_10', current_price) / current_price - 1,
                    asset_data.iloc[self.current_step].get('sma_50', current_price) / current_price - 1
                ])
            else:
                features.extend([0] * 10)
        
        features.extend([
            self.balance / self.initial_balance,
            self.total_value / self.initial_balance,
            sum(1 for pos in self.positions.values() if abs(pos) > 0.01),
            len(self.portfolio_history),
            self._calculate_portfolio_sharpe()
        ])
        
        return np.array(features, dtype=np.float32)
        
    def step(self, actions):
        if self.current_step >= min(len(data) for data in self.data_dict.values()) - 1:
            return self._get_observation(), 0, True, {}
        
        old_total_value = self.total_value
        
        for i, asset in enumerate(self.assets):
            action = actions[i]
            self._execute_trade(asset, action)
        
        self.current_step += 1
        self._update_portfolio_value()
        
        reward = self._calculate_reward(old_total_value)
        
        self.portfolio_history.append({
            'step': self.current_step,
            'total_value': self.total_value,
            'balance': self.balance,
            'positions': self.positions.copy()
        })
        
        done = self.current_step >= min(len(data) for data in self.data_dict.values()) - 1
        
        info = {
            'total_value': self.total_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'num_positions': sum(1 for pos in self.positions.values() if abs(pos) > 0.01)
        }
        
        return self._get_observation(), reward, done, info
        
    def _execute_trade(self, asset, action):
        if self.current_step >= len(self.data_dict[asset]):
            return
            
        current_price = self.data_dict[asset].iloc[self.current_step]['price']
        
        if action > 0.1:  # Buy
            max_buy = self.balance / (current_price * 1.001)
            buy_amount = action * max_buy * 0.1
            
            if buy_amount * current_price > 1.0:
                cost = buy_amount * current_price * 1.001
                self.balance -= cost
                self.positions[asset] += buy_amount
                
        elif action < -0.1:  # Sell
            sell_amount = abs(action) * self.positions[asset] * 0.1
            
            if sell_amount > 0.001:
                proceeds = sell_amount * current_price * 0.999
                self.balance += proceeds
                self.positions[asset] -= sell_amount
        
        self.positions[asset] = max(0, self.positions[asset])
        
    def _update_portfolio_value(self):
        asset_value = 0
        
        for asset, position in self.positions.items():
            if self.current_step < len(self.data_dict[asset]):
                current_price = self.data_dict[asset].iloc[self.current_step]['price']
                asset_value += position * current_price
        
        self.total_value = self.balance + asset_value
        
    def _calculate_reward(self, old_total_value):
        portfolio_return = (self.total_value - old_total_value) / old_total_value if old_total_value > 0 else 0
        
        market_returns = []
        for asset in self.assets:
            if self.current_step < len(self.data_dict[asset]) and self.current_step > 0:
                current_price = self.data_dict[asset].iloc[self.current_step]['price']
                prev_price = self.data_dict[asset].iloc[self.current_step - 1]['price']
                market_return = (current_price - prev_price) / prev_price
                market_returns.append(market_return)
        
        avg_market_return = np.mean(market_returns) if market_returns else 0
        alpha = portfolio_return - avg_market_return
        
        num_positions = sum(1 for pos in self.positions.values() if abs(pos) > 0.01)
        diversification_bonus = 0.001 if 1 < num_positions < len(self.assets) else 0
        
        concentration_penalty = 0
        if num_positions > 0:
            position_values = []
            for asset, position in self.positions.items():
                if self.current_step < len(self.data_dict[asset]):
                    price = self.data_dict[asset].iloc[self.current_step]['price']
                    position_values.append(position * price)
            
            if position_values:
                total_position_value = sum(position_values)
                if total_position_value > 0:
                    weights = [pv / total_position_value for pv in position_values]
                    concentration = sum(w ** 2 for w in weights)
                    if concentration > 0.8:
                        concentration_penalty = -0.001