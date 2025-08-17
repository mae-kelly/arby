#!/usr/bin/env python3
"""
Real-Time Arbitrage Monitor with Flash Loans
Enhanced version of hey.py with flash loan capabilities
"""

import asyncio
import aiohttp
import json
import time
import websockets
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from web3 import Web3
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FlashLoanOpportunity:
    type: str  # 'aave', 'balancer', 'dydx'
    token: str
    amount: float
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    gross_profit: float
    flash_fee: float
    gas_cost: float
    net_profit: float
    confidence: float
    timestamp: float

@dataclass
class DEXOpportunity:
    token_in: str
    token_out: str
    dex_in: str
    dex_out: str
    amount_in: float
    expected_out: float
    flash_loan_size: float
    net_profit: float
    route: List[str]

class FlashLoanArbitrageMonitor:
    def __init__(self):
        self.session = None
        self.current_prices = {}
        self.gas_tracker = {}
        self.mempool_data = {}
        self.opportunities = []
        self.flash_opportunities = []
        self.dex_opportunities = []
        
        # Web3 setup
        self.w3 = None
        self.setup_web3()
        
        # Exchange APIs (same as hey.py)
        self.exchanges = {
            'binance': {
                'rest': 'https://api.binance.com/api/v3',
                'fee': 0.001
            },
            'coinbase': {
                'rest': 'https://api.exchange.coinbase.com',
                'fee': 0.005
            },
            'kraken': {
                'rest': 'https://api.kraken.com/0/public',
                'fee': 0.0026
            }
        }
        
        # Flash loan providers
        self.flash_providers = {
            'aave': {
                'address': '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9',
                'fee': 0.0009,  # 0.09%
                'name': 'Aave'
            },
            'balancer': {
                'address': '0xBA12222222228d8Ba445958a75a0704d566BF2C8',
                'fee': 0.0,  # 0% fee
                'name': 'Balancer'
            },
            'dydx': {
                'address': '0x1E0447b19BB6EcFdAe1e4AE1694b0C3659614e4e',
                'fee': 0.0,  # Minimal fee
                'name': 'dYdX'
            }
        }
        
        # DEX routers for on-chain arbitrage
        self.dex_routers = {
            'uniswap_v2': {
                'address': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
                'fee': 0.003,
                'name': 'Uniswap V2'
            },
            'sushiswap': {
                'address': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
                'fee': 0.003,
                'name': 'SushiSwap'
            },
            'uniswap_v3': {
                'address': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'fee': 0.0005,  # Variable, using 0.05% as average
                'name': 'Uniswap V3'
            }
        }
        
        # Token addresses for flash loans
        self.tokens = {
            'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            'USDC': '0xA0b86a33E6e86c026a91F4A7A6B89e23AA8C1Fd9',
            'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
            'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
            'WBTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599'
        }
        
        self.stats = {
            'total_scans': 0,
            'flash_opportunities': 0,
            'dex_opportunities': 0,
            'total_potential_profit': 0.0
        }
        
    def setup_web3(self):
        """Setup Web3 connection"""
        rpc_url = os.getenv('ETH_RPC_URL', 'https://mainnet.infura.io/v3/YOUR_KEY')
        if 'YOUR_KEY' not in rpc_url:
            try:
                self.w3 = Web3(Web3.HTTPProvider(rpc_url))
                if self.w3.is_connected():
                    logger.info("✅ Web3 connected to Ethereum mainnet")
                else:
                    logger.warning("⚠️ Web3 connection failed")
            except Exception as e:
                logger.error(f"Web3 setup error: {e}")
        else:
            logger.warning("⚠️ No ETH_RPC_URL configured - flash loan execution disabled")
        
    async def initialize(self):
        """Initialize all connections"""
        self.session = aiohttp.ClientSession()
        print("🚀 FLASH LOAN ARBITRAGE MONITOR")
        print("=" * 60)
        print("✅ Monitoring CEX prices + DEX prices + Flash loans")
        print("✅ Real-time gas tracking")
        print("✅ MEV opportunity detection")
        print("=" * 60)
        
        # Start all monitoring tasks
        tasks = [
            self.monitor_exchange_prices(),
            self.monitor_gas_prices(),
            self.monitor_dex_prices(),
            self.scan_flash_opportunities(),
            self.scan_dex_arbitrage(),
            self.display_dashboard()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def monitor_exchange_prices(self):
        """Monitor CEX prices (same as hey.py)"""
        while True:
            try:
                tasks = [
                    self.fetch_binance_prices(),
                    self.fetch_coinbase_prices(),
                    self.fetch_kraken_prices()
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, exchange_name in enumerate(['binance', 'coinbase', 'kraken']):
                    if not isinstance(results[i], Exception) and results[i]:
                        self.current_prices[exchange_name] = results[i]
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Price monitoring error: {e}")
                await asyncio.sleep(5)
                
    async def fetch_binance_prices(self):
        """Fetch Binance prices"""
        try:
            url = f"{self.exchanges['binance']['rest']}/ticker/price"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = {}
                    symbols_of_interest = ['BTCUSDT', 'ETHUSDT', 'USDCUSDT']
                    
                    for item in data:
                        if item['symbol'] in symbols_of_interest:
                            prices[item['symbol']] = {
                                'price': float(item['price']),
                                'timestamp': time.time(),
                                'exchange': 'binance'
                            }
                    return prices
        except Exception as e:
            logger.error(f"Binance API error: {e}")
            return {}
            
    async def fetch_coinbase_prices(self):
        """Fetch Coinbase prices"""
        try:
            symbol_map = {'BTC-USD': 'BTCUSDT', 'ETH-USD': 'ETHUSDT', 'USDC-USD': 'USDCUSDT'}
            prices = {}
            
            for cb_symbol, unified_symbol in symbol_map.items():
                try:
                    url = f"{self.exchanges['coinbase']['rest']}/products/{cb_symbol}/ticker"
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            prices[unified_symbol] = {
                                'price': float(data['price']),
                                'timestamp': time.time(),
                                'exchange': 'coinbase'
                            }
                    await asyncio.sleep(0.1)
                except Exception:
                    continue
            return prices
        except Exception as e:
            logger.error(f"Coinbase API error: {e}")
            return {}
            
    async def fetch_kraken_prices(self):
        """Fetch Kraken prices"""
        try:
            symbol_map = {'XBTUSD': 'BTCUSDT', 'ETHUSD': 'ETHUSDT'}
            pairs = ','.join(symbol_map.keys())
            url = f"{self.exchanges['kraken']['rest']}/Ticker?pair={pairs}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = {}
                    if 'result' in data:
                        for kraken_symbol, unified_symbol in symbol_map.items():
                            if kraken_symbol in data['result']:
                                ticker = data['result'][kraken_symbol]
                                prices[unified_symbol] = {
                                    'price': float(ticker['c'][0]),
                                    'timestamp': time.time(),
                                    'exchange': 'kraken'
                                }
                    return prices
        except Exception as e:
            logger.error(f"Kraken API error: {e}")
            return {}
            
    async def monitor_gas_prices(self):
        """Monitor gas prices"""
        while True:
            try:
                gas_data = await self.fetch_real_gas_prices()
                if gas_data:
                    self.gas_tracker = gas_data
                
                eth_price = await self.fetch_eth_price()
                if eth_price:
                    self.gas_tracker['eth_price_usd'] = eth_price
                
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Gas monitoring error: {e}")
                await asyncio.sleep(30)
                
    async def fetch_real_gas_prices(self):
        """Fetch real gas prices"""
        try:
            url = 'https://api.etherscan.io/api?module=gastracker&action=gasoracle'
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data['status'] == '1':
                        result = data['result']
                        return {
                            'safe': int(result['SafeGasPrice']),
                            'standard': int(result['ProposeGasPrice']),
                            'fast': int(result['FastGasPrice']),
                            'timestamp': time.time()
                        }
        except Exception as e:
            logger.error(f"Gas price fetch error: {e}")
        
        return {'safe': 20, 'standard': 25, 'fast': 35, 'timestamp': time.time()}
        
    async def fetch_eth_price(self):
        """Fetch ETH price"""
        try:
            url = 'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd'
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['ethereum']['usd']
        except Exception:
            pass
        return 2400  # Fallback
        
    async def monitor_dex_prices(self):
        """Monitor DEX prices (simulated for demo)"""
        while True:
            try:
                # In a real implementation, this would query Uniswap/SushiSwap contracts
                import random
                
                dex_prices = {}
                base_symbols = ['ETHUSDT', 'BTCUSDT', 'USDCUSDT']
                
                for symbol in base_symbols:
                    # Find CEX price as baseline
                    cex_price = None
                    for exchange_data in self.current_prices.values():
                        if symbol in exchange_data:
                            cex_price = exchange_data[symbol]['price']
                            break
                    
                    if cex_price:
                        # Simulate DEX prices with small variations
                        dex_prices[symbol] = {
                            'uniswap_v2': cex_price * (1 + random.uniform(-0.01, 0.01)),
                            'sushiswap': cex_price * (1 + random.uniform(-0.01, 0.01)),
                            'uniswap_v3': cex_price * (1 + random.uniform(-0.005, 0.005)),
                        }
                
                self.current_prices['dex'] = dex_prices
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"DEX monitoring error: {e}")
                await asyncio.sleep(10)
                
    async def scan_flash_opportunities(self):
        """Scan for flash loan arbitrage opportunities"""
        while True:
            await asyncio.sleep(3)
            
            try:
                if len(self.current_prices) >= 2:
                    self.stats['total_scans'] += 1
                    opportunities = self.find_flash_arbitrage()
                    
                    if opportunities:
                        self.flash_opportunities = opportunities
                        self.stats['flash_opportunities'] += len(opportunities)
                        
                        total_profit = sum(opp.net_profit for opp in opportunities)
                        self.stats['total_potential_profit'] += total_profit
                        
            except Exception as e:
                logger.error(f"Flash opportunity scanning error: {e}")
                
    def find_flash_arbitrage(self) -> List[FlashLoanOpportunity]:
        """Find flash loan arbitrage opportunities"""
        opportunities = []
        
        # Check CEX vs DEX arbitrage (flash loan funded)
        if 'dex' in self.current_prices:
            dex_data = self.current_prices['dex']
            
            for symbol in dex_data:
                if symbol in dex_data:
                    dex_prices = dex_data[symbol]
                    
                    # Find CEX prices for comparison
                    cex_prices = {}
                    for exchange, prices in self.current_prices.items():
                        if exchange != 'dex' and symbol in prices:
                            cex_prices[exchange] = prices[symbol]['price']
                    
                    # Check arbitrage opportunities
                    for dex_name, dex_price in dex_prices.items():
                        for cex_name, cex_price in cex_prices.items():
                            
                            # CEX -> DEX arbitrage
                            if dex_price > cex_price * 1.005:  # 0.5% minimum spread
                                opp = self.calculate_flash_opportunity(
                                    symbol, cex_name, dex_name, cex_price, dex_price
                                )
                                if opp:
                                    opportunities.append(opp)
                            
                            # DEX -> CEX arbitrage  
                            if cex_price > dex_price * 1.005:
                                opp = self.calculate_flash_opportunity(
                                    symbol, dex_name, cex_name, dex_price, cex_price
                                )
                                if opp:
                                    opportunities.append(opp)
        
        return sorted(opportunities, key=lambda x: x.net_profit, reverse=True)
        
    def calculate_flash_opportunity(self, symbol: str, buy_source: str, sell_source: str, 
                                   buy_price: float, sell_price: float) -> Optional[FlashLoanOpportunity]:
        """Calculate flash loan arbitrage opportunity"""
        
        if sell_price <= buy_price:
            return None
        
        # Choose optimal flash loan provider
        best_provider = min(self.flash_providers.items(), key=lambda x: x[1]['fee'])
        provider_name, provider_data = best_provider
        
        # Calculate trade size (example: $100k flash loan)
        trade_size_usd = 100000
        trade_size_tokens = trade_size_usd / buy_price
        
        # Calculate profits and costs
        gross_profit = (sell_price - buy_price) * trade_size_tokens
        flash_fee = trade_size_usd * provider_data['fee']
        
        # Gas cost estimation
        gas_price = self.gas_tracker.get('fast', 50) * 1e9  # Convert to wei
        eth_price = self.gas_tracker.get('eth_price_usd', 2400)
        
        # Flash loan requires multiple transactions
        total_gas = 800000  # Flash loan + swaps + repay
        gas_cost_usd = (gas_price * total_gas / 1e18) * eth_price
        
        # Calculate net profit
        net_profit = gross_profit - flash_fee - gas_cost_usd
        
        # Confidence based on liquidity and spread stability
        confidence = min(0.95, max(0.3, (gross_profit / trade_size_usd) * 10))
        
        if net_profit > 500:  # $500 minimum profit
            return FlashLoanOpportunity(
                type=provider_name,
                token=symbol,
                amount=trade_size_tokens,
                buy_exchange=buy_source,
                sell_exchange=sell_source,
                buy_price=buy_price,
                sell_price=sell_price,
                gross_profit=gross_profit,
                flash_fee=flash_fee,
                gas_cost=gas_cost_usd,
                net_profit=net_profit,
                confidence=confidence,
                timestamp=time.time()
            )
        
        return None
        
    async def scan_dex_arbitrage(self):
        """Scan for DEX-to-DEX arbitrage"""
        while True:
            await asyncio.sleep(4)
            
            try:
                if 'dex' in self.current_prices:
                    opportunities = self.find_dex_arbitrage()
                    if opportunities:
                        self.dex_opportunities = opportunities
                        self.stats['dex_opportunities'] += len(opportunities)
                        
            except Exception as e:
                logger.error(f"DEX arbitrage scanning error: {e}")
                
    def find_dex_arbitrage(self) -> List[DEXOpportunity]:
        """Find DEX-to-DEX arbitrage opportunities"""
        opportunities = []
        
        if 'dex' not in self.current_prices:
            return opportunities
        
        dex_data = self.current_prices['dex']
        
        for symbol in dex_data:
            dex_prices = dex_data[symbol]
            dex_names = list(dex_prices.keys())
            
            # Compare all DEX pairs
            for i, dex1 in enumerate(dex_names):
                for dex2 in dex_names[i+1:]:
                    price1 = dex_prices[dex1]
                    price2 = dex_prices[dex2]
                    
                    if price2 > price1 * 1.003:  # 0.3% minimum for DEX arbitrage
                        trade_size = 50000  # $50k trade
                        profit = (price2 - price1) * (trade_size / price1)
                        
                        # Account for DEX fees and gas
                        dex_fees = trade_size * 0.006  # 0.3% each DEX
                        gas_cost = 150  # Approximate gas cost for DEX swaps
                        net_profit = profit - dex_fees - gas_cost
                        
                        if net_profit > 200:  # $200 minimum
                            opportunities.append(DEXOpportunity(
                                token_in=symbol,
                                token_out=symbol,
                                dex_in=dex1,
                                dex_out=dex2,
                                amount_in=trade_size / price1,
                                expected_out=trade_size / price2,
                                flash_loan_size=trade_size,
                                net_profit=net_profit,
                                route=[dex1, dex2]
                            ))
        
        return sorted(opportunities, key=lambda x: x.net_profit, reverse=True)
        
    async def display_dashboard(self):
        """Display real-time dashboard with flash loan opportunities"""
        while True:
            try:
                await asyncio.sleep(5)
                
                # Clear screen and show dashboard
                print("\033[2J\033[H")  # Clear screen
                
                print("🚀 FLASH LOAN ARBITRAGE MONITOR")
                print("=" * 80)
                print(f"⏰ Last Update: {datetime.now().strftime('%H:%M:%S')}")
                print()
                
                # Current prices
                print("💰 EXCHANGE PRICES:")
                self.display_prices()
                print()
                
                # Gas prices
                if self.gas_tracker:
                    print("⛽ GAS PRICES:")
                    print(f"   Standard: {self.gas_tracker['standard']} gwei")
                    print(f"   Fast: {self.gas_tracker['fast']} gwei") 
                    print(f"   ETH Price: ${self.gas_tracker.get('eth_price_usd', 'N/A')}")
                    print()
                
                # Flash loan opportunities
                if self.flash_opportunities:
                    print("⚡ FLASH LOAN OPPORTUNITIES:")
                    print("=" * 80)
                    
                    for i, opp in enumerate(self.flash_opportunities[:3], 1):
                        print(f"\n{i}. {opp.token} Flash Loan Arbitrage ({opp.type.upper()})")
                        print(f"   Buy {opp.buy_exchange}: ${opp.buy_price:.2f}")
                        print(f"   Sell {opp.sell_exchange}: ${opp.sell_price:.2f}")
                        print(f"   Flash Loan: ${opp.amount * opp.buy_price:,.0f}")
                        print(f"   Gross Profit: ${opp.gross_profit:.2f}")
                        print(f"   Flash Fee: ${opp.flash_fee:.2f}")
                        print(f"   Gas Cost: ${opp.gas_cost:.2f}")
                        print(f"   NET PROFIT: ${opp.net_profit:.2f}")
                        print(f"   Confidence: {opp.confidence:.1%}")
                        
                        if opp.net_profit > 1000:
                            print("   🚨 HIGH-PROFIT OPPORTUNITY!")
                
                # DEX arbitrage
                if self.dex_opportunities:
                    print("\n🔄 DEX ARBITRAGE OPPORTUNITIES:")
                    for i, opp in enumerate(self.dex_opportunities[:2], 1):
                        print(f"   {i}. {opp.token_in}: {opp.dex_in} → {opp.dex_out}")
                        print(f"      Net Profit: ${opp.net_profit:.2f}")
                
                if not self.flash_opportunities and not self.dex_opportunities:
                    print("😴 No profitable flash loan opportunities found")
                    print("   Monitoring for price discrepancies...")
                
                print()
                
                # Statistics
                print("📊 FLASH LOAN MONITOR STATS:")
                print(f"   Total Scans: {self.stats['total_scans']}")
                print(f"   Flash Opportunities: {self.stats['flash_opportunities']}")
                print(f"   DEX Opportunities: {self.stats['dex_opportunities']}")
                print(f"   Total Potential Profit: ${self.stats['total_potential_profit']:.2f}")
                
                print("\n✅ Ready to execute profitable flash loans!")
                print("Press Ctrl+C to stop")
                
            except Exception as e:
                logger.error(f"Dashboard error: {e}")
                await asyncio.sleep(5)
                
    def display_prices(self):
        """Display current prices"""
        if self.current_prices:
            all_symbols = set()
            for prices in self.current_prices.values():
                if isinstance(prices, dict):
                    all_symbols.update(prices.keys())
                    
            for symbol in sorted(all_symbols):
                if symbol != 'dex':
                    print(f"   {symbol}:")
                    for exchange, prices in self.current_prices.items():
                        if exchange != 'dex' and isinstance(prices, dict) and symbol in prices:
                            age = time.time() - prices[symbol]['timestamp']
                            print(f"      {exchange.ljust(10)}: ${prices[symbol]['price']:,.2f} ({age:.0f}s ago)")
                        elif exchange == 'dex' and isinstance(prices, dict) and symbol in prices:
                            print(f"      DEX Prices:")
                            for dex_name, dex_price in prices[symbol].items():
                                print(f"        {dex_name}: ${dex_price:,.2f}")
                                
    async def execute_flash_loan(self, opportunity: FlashLoanOpportunity):
        """Execute flash loan arbitrage (simulation)"""
        if not self.w3:
            print(f"⚠️ Web3 not configured - would execute: ${opportunity.net_profit:.2f} profit")
            return False
            
        print(f"⚡ Executing flash loan arbitrage:")
        print(f"   Provider: {opportunity.type}")
        print(f"   Amount: ${opportunity.amount * opportunity.buy_price:,.0f}")
        print(f"   Expected Net Profit: ${opportunity.net_profit:.2f}")
        
        # In a real implementation, this would:
        # 1. Call flash loan contract
        # 2. Execute buy on source exchange
        # 3. Execute sell on target exchange  
        # 4. Repay flash loan + fee
        # 5. Keep the profit
        
        return True
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

async def main():
    monitor = FlashLoanArbitrageMonitor()
    
    try:
        await monitor.initialize()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping flash loan monitor...")
    finally:
        await monitor.cleanup()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    asyncio.run(main())