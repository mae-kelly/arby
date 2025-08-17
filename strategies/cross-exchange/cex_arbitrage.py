import asyncio
import ccxt.async_support as ccxt
from typing import Dict, List, Tuple
import numpy as np

class CEXArbitrageStrategy:
    def __init__(self, exchanges: Dict[str, ccxt.Exchange]):
        self.exchanges = exchanges
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'MATIC/USDT']
        self.min_profit_threshold = 0.002  # 0.2%
    
    async def scan_all_pairs(self) -> List[Dict]:
        """Scan all exchange pairs for arbitrage"""
        opportunities = []
        
        for symbol in self.symbols:
            prices = await self.fetch_all_prices(symbol)
            if len(prices) >= 2:
                arb_ops = self.find_arbitrage_opportunities(symbol, prices)
                opportunities.extend(arb_ops)
        
        return sorted(opportunities, key=lambda x: x['profit_pct'], reverse=True)
    
    async def fetch_all_prices(self, symbol: str) -> Dict[str, Dict]:
        """Fetch prices from all exchanges simultaneously"""
        tasks = []
        for name, exchange in self.exchanges.items():
            if symbol in exchange.markets:
                tasks.append(self.fetch_ticker_safe(name, exchange, symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        prices = {}
        
        for result in results:
            if isinstance(result, dict) and 'exchange' in result:
                prices[result['exchange']] = result
        
        return prices
    
    async def fetch_ticker_safe(self, name: str, exchange: ccxt.Exchange, symbol: str) -> Dict:
        """Safely fetch ticker with error handling"""
        try:
            ticker = await exchange.fetch_ticker(symbol)
            return {
                'exchange': name,
                'symbol': symbol,
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['quoteVolume']
            }
        except Exception:
            return {}
    
    def find_arbitrage_opportunities(self, symbol: str, prices: Dict[str, Dict]) -> List[Dict]:
        """Find arbitrage between exchange pairs"""
        opportunities = []
        exchanges = list(prices.keys())
        
        for i, buy_ex in enumerate(exchanges):
            for sell_ex in exchanges[i+1:]:
                buy_data = prices[buy_ex]
                sell_data = prices[sell_ex]
                
                # Direction 1: Buy on buy_ex, sell on sell_ex
                if buy_data.get('ask') and sell_data.get('bid'):
                    profit_pct = (sell_data['bid'] - buy_data['ask']) / buy_data['ask']
                    if profit_pct > self.min_profit_threshold:
                        opportunities.append({
                            'symbol': symbol,
                            'buy_exchange': buy_ex,
                            'sell_exchange': sell_ex,
                            'buy_price': buy_data['ask'],
                            'sell_price': sell_data['bid'],
                            'profit_pct': profit_pct * 100,
                            'volume': min(buy_data.get('volume', 0), sell_data.get('volume', 0))
                        })
                
                # Direction 2: Buy on sell_ex, sell on buy_ex
                if sell_data.get('ask') and buy_data.get('bid'):
                    profit_pct = (buy_data['bid'] - sell_data['ask']) / sell_data['ask']
                    if profit_pct > self.min_profit_threshold:
                        opportunities.append({
                            'symbol': symbol,
                            'buy_exchange': sell_ex,
                            'sell_exchange': buy_ex,
                            'buy_price': sell_data['ask'],
                            'sell_price': buy_data['bid'],
                            'profit_pct': profit_pct * 100,
                            'volume': min(buy_data.get('volume', 0), sell_data.get('volume', 0))
                        })
        
        return opportunities
    
    async def execute_arbitrage(self, opportunity: Dict) -> bool:
        """Execute cross-exchange arbitrage"""
        try:
            buy_exchange = self.exchanges[opportunity['buy_exchange']]
            sell_exchange = self.exchanges[opportunity['sell_exchange']]
            
            # Calculate optimal trade size
            trade_size = self.calculate_trade_size(opportunity)
            
            # Execute trades simultaneously
            buy_task = buy_exchange.create_market_buy_order(
                opportunity['symbol'], 
                trade_size / opportunity['buy_price']
            )
            sell_task = sell_exchange.create_market_sell_order(
                opportunity['symbol'],
                trade_size / opportunity['buy_price']
            )
            
            results = await asyncio.gather(buy_task, sell_task, return_exceptions=True)
            
            return all(not isinstance(r, Exception) for r in results)
            
        except Exception as e:
            print(f"Execution error: {e}")
            return False
    
    def calculate_trade_size(self, opportunity: Dict) -> float:
        """Calculate optimal trade size based on volume and balance"""
        max_volume = opportunity['volume'] * 0.1  # 10% of volume
        max_balance = 10000  # $10k max per trade
        return min(max_volume, max_balance)
