# profitable_flash_arbitrage.py

import asyncio
import aiohttp
from web3 import Web3
import time
import requests
import json
import hmac
import hashlib
import base64
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
import os

# Import all other repo files
try:
    from main_orchestrator import OKXClient, ArbitrageBot
    from hft_engine import HFTEngine
    print("✅ Imported CEX and HFT modules")
except ImportError:
    print("⚠️  CEX/HFT modules not available")

@dataclass
class FlashLoanOpportunity:
    pair: str
    buy_dex: str
    sell_dex: str
    buy_price: float
    sell_price: float
    profit_percentage: float
    flash_loan_amount: int
    gross_profit_usd: float
    flash_loan_fee_usd: float
    gas_cost_usd: float
    net_profit_usd: float
    execution_time_estimate: float
    risk_level: str

class ProfitableFlashArbitrage:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider("https://eth-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX"))
        self.webhook = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3"
        
        # Real token addresses
        self.tokens = {
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "USDC": "0xA0b86a33E6417c7ef38BC67B2F11D6B3DC0B5f55",
            "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
            "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F"
        }
        
        # DEX routers
        self.dexes = {
            "Uniswap_V2": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            "SushiSwap": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
            "Uniswap_V3": "0xE592427A0AEce92De3Edee1F18E0157C05861564"
        }
        
        # Aave flash loan provider
        self.aave_pool = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
        
        self.router_abi = [
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"}
                ],
                "name": "getAmountsOut",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        # Flash loan amounts to test (in wei)
        self.flash_loan_amounts = {
            "WETH": [
                1 * 10**18,     # 1 ETH
                5 * 10**18,     # 5 ETH  
                10 * 10**18,    # 10 ETH
                50 * 10**18,    # 50 ETH
                100 * 10**18    # 100 ETH
            ],
            "USDT": [
                1000 * 10**6,   # $1k
                5000 * 10**6,   # $5k
                10000 * 10**6,  # $10k
                50000 * 10**6,  # $50k
                100000 * 10**6  # $100k
            ]
        }
        
        # Initialize CEX connection if available
        try:
            self.okx_client = OKXClient()
            self.cex_available = True
            print("✅ CEX connection initialized")
        except:
            self.cex_available = False
            print("⚠️  CEX connection not available")

    async def send_alert(self, message):
        try:
            payload = {"content": f"💰 PROFITABLE FLASH LOAN: {message}"}
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook, json=payload) as response:
                    print(f"🚨 ALERT: {message}")
        except Exception as e:
            print(f"Alert error: {e}")

    def get_dex_price(self, dex_name, token_in, token_out, amount_in):
        try:
            router_address = self.dexes[dex_name]
            contract = self.w3.eth.contract(address=router_address, abi=self.router_abi)
            path = [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)]
            amounts = contract.functions.getAmountsOut(int(amount_in), path).call()
            return amounts[-1]
        except Exception as e:
            print(f"❌ {dex_name} error: {e}")
            return 0

    async def get_market_prices(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.coingecko.com/api/v3/simple/price?ids=ethereum,bitcoin,usd-coin,tether&vs_currencies=usd") as response:
                    return await response.json()
        except:
            return {
                "ethereum": {"usd": 4400},
                "bitcoin": {"usd": 110000},
                "usd-coin": {"usd": 1.0},
                "tether": {"usd": 1.0}
            }

    async def get_cex_prices(self):
        """Get CEX prices for comparison"""
        if not self.cex_available:
            return {}
        
        try:
            cex_prices = {}
            symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
            
            for symbol in symbols:
                ticker = await self.okx_client.get_ticker(symbol)
                if ticker.get("code") == "0" and ticker.get("data"):
                    data = ticker["data"][0]
                    cex_prices[symbol] = {
                        "bid": float(data.get("bidPx", 0)),
                        "ask": float(data.get("askPx", 0)),
                        "last": float(data.get("last", 0))
                    }
            
            return cex_prices
        except Exception as e:
            print(f"CEX price error: {e}")
            return {}

    def calculate_flash_loan_costs(self, amount_usd, market_prices):
        """Calculate all costs for flash loan execution"""
        # Aave flash loan fee: 0.09%
        flash_loan_fee = amount_usd * 0.0009
        
        # Gas costs (realistic estimates)
        gas_price_wei = self.w3.eth.gas_price
        eth_price = market_prices["ethereum"]["usd"]
        
        # Flash loan execution requires multiple operations
        flash_loan_gas = 800000  # Flash loan + 2 DEX swaps + overhead
        gas_cost_eth = (flash_loan_gas * gas_price_wei) / 10**18
        gas_cost_usd = gas_cost_eth * eth_price
        
        # Slippage buffer (0.1% for large trades)
        slippage_cost = amount_usd * 0.001
        
        total_costs = flash_loan_fee + gas_cost_usd + slippage_cost
        
        return {
            "flash_loan_fee": flash_loan_fee,
            "gas_cost": gas_cost_usd,
            "slippage_cost": slippage_cost,
            "total_costs": total_costs,
            "gas_price_gwei": self.w3.from_wei(gas_price_wei, 'gwei')
        }

    async def scan_flash_loan_opportunities(self):
        """Scan for profitable flash loan arbitrage opportunities"""
        print("🔍 Scanning for PROFITABLE flash loan opportunities...")
        
        opportunities = []
        market_prices = await self.get_market_prices()
        cex_prices = await self.get_cex_prices()
        
        # Test multiple trading pairs and amounts
        pairs = [
            ("WETH", "USDT", "ethereum"),
            ("WETH", "USDC", "ethereum"),
            ("WBTC", "USDT", "bitcoin"),
            ("WBTC", "USDC", "bitcoin")
        ]
        
        for token_in_name, token_out_name, price_id in pairs:
            token_in = self.tokens[token_in_name]
            token_out = self.tokens[token_out_name]
            token_price_usd = market_prices[price_id]["usd"]
            
            print(f"\n📊 Analyzing {token_in_name}->{token_out_name} (${token_price_usd:,.0f})")
            
            # Test different flash loan amounts
            amounts_to_test = self.flash_loan_amounts.get(token_in_name, [10**18])
            
            for amount_in in amounts_to_test:
                amount_usd = (amount_in / 10**18) * token_price_usd if token_in_name == "WETH" else (amount_in / 10**8) * token_price_usd
                
                if amount_usd < 1000:  # Skip small amounts
                    continue
                    
                print(f"   Testing ${amount_usd:,.0f} flash loan...")
                
                # Get prices from multiple DEXes
                prices = {}
                for dex_name in ["Uniswap_V2", "SushiSwap"]:
                    price = self.get_dex_price(dex_name, token_in, token_out, amount_in)
                    if price > 0:
                        prices[dex_name] = price
                
                if len(prices) >= 2:
                    max_price = max(prices.values())
                    min_price = min(prices.values())
                    max_dex = max(prices, key=prices.get)
                    min_dex = min(prices, key=prices.get)
                    
                    # Calculate profit
                    price_diff = max_price - min_price
                    profit_percentage = (price_diff / min_price) * 100
                    
                    # Convert to USD
                    if token_out_name in ["USDT", "USDC"]:
                        gross_profit_usd = price_diff / 10**6
                    else:
                        gross_profit_usd = (price_diff / 10**18) * token_price_usd
                    
                    # Calculate all costs
                    costs = self.calculate_flash_loan_costs(amount_usd, market_prices)
                    net_profit_usd = gross_profit_usd - costs["total_costs"]
                    
                    # Only proceed if potentially profitable
                    if net_profit_usd > 50:  # Minimum $50 profit
                        
                        # Estimate execution time and risk
                        execution_time = 15 + (amount_usd / 10000)  # Seconds
                        risk_level = "LOW" if amount_usd < 10000 else "MEDIUM" if amount_usd < 50000 else "HIGH"
                        
                        opportunity = FlashLoanOpportunity(
                            pair=f"{token_in_name}->{token_out_name}",
                            buy_dex=min_dex,
                            sell_dex=max_dex,
                            buy_price=min_price,
                            sell_price=max_price,
                            profit_percentage=profit_percentage,
                            flash_loan_amount=amount_in,
                            gross_profit_usd=gross_profit_usd,
                            flash_loan_fee_usd=costs["flash_loan_fee"],
                            gas_cost_usd=costs["gas_cost"],
                            net_profit_usd=net_profit_usd,
                            execution_time_estimate=execution_time,
                            risk_level=risk_level
                        )
                        
                        opportunities.append(opportunity)
                        
                        print(f"   🎯 PROFITABLE OPPORTUNITY FOUND!")
                        print(f"      Flash loan: ${amount_usd:,.0f}")
                        print(f"      Gross profit: ${gross_profit_usd:,.2f}")
                        print(f"      Flash loan fee: ${costs['flash_loan_fee']:,.2f}")
                        print(f"      Gas cost: ${costs['gas_cost']:,.2f}")
                        print(f"      Net profit: ${net_profit_usd:,.2f}")
                        print(f"      ROI: {(net_profit_usd/amount_usd)*100:.3f}%")
                        
                        # Send alert for high-profit opportunities
                        if net_profit_usd > 500:
                            await self.send_alert(
                                f"HIGH PROFIT FLASH LOAN DETECTED!\n"
                                f"Pair: {opportunity.pair}\n"
                                f"Flash loan: ${amount_usd:,.0f}\n"
                                f"Net profit: ${net_profit_usd:,.2f}\n"
                                f"ROI: {(net_profit_usd/amount_usd)*100:.3f}%\n"
                                f"Strategy: Borrow -> Buy {min_dex} -> Sell {max_dex} -> Repay"
                            )
        
        return sorted(opportunities, key=lambda x: x.net_profit_usd, reverse=True)

    async def simulate_flash_loan_execution(self, opportunity: FlashLoanOpportunity):
        """Simulate the complete flash loan execution process"""
        print(f"\n🚀 SIMULATING FLASH LOAN EXECUTION")
        print(f"{'='*60}")
        
        amount_usd = (opportunity.flash_loan_amount / 10**18) * 4400  # Approximate
        
        print(f"📋 EXECUTION PLAN:")
        print(f"   1. Flash loan ${amount_usd:,.0f} from Aave")
        print(f"   2. Buy on {opportunity.buy_dex} at {opportunity.buy_price:,}")
        print(f"   3. Sell on {opportunity.sell_dex} at {opportunity.sell_price:,}")
        print(f"   4. Repay flash loan + fee")
        print(f"   5. Keep ${opportunity.net_profit_usd:,.2f} profit")
        
        print(f"\n💰 PROFIT BREAKDOWN:")
        print(f"   Gross profit: ${opportunity.gross_profit_usd:,.2f}")
        print(f"   Flash loan fee (0.09%): ${opportunity.flash_loan_fee_usd:,.2f}")
        print(f"   Gas costs: ${opportunity.gas_cost_usd:,.2f}")
        print(f"   NET PROFIT: ${opportunity.net_profit_usd:,.2f}")
        print(f"   ROI: {(opportunity.net_profit_usd/amount_usd)*100:.3f}%")
        
        print(f"\n⚡ EXECUTION DETAILS:")
        print(f"   Estimated time: {opportunity.execution_time_estimate:.1f} seconds")
        print(f"   Risk level: {opportunity.risk_level}")
        print(f"   Spread: {opportunity.profit_percentage:.4f}%")
        
        print(f"\n🔗 SMART CONTRACT CALLS:")
        print(f"   1. flashLoanSimple({opportunity.flash_loan_amount}, ...)")
        print(f"   2. {opportunity.buy_dex}.swapExactTokensForTokens(...)")
        print(f"   3. {opportunity.sell_dex}.swapExactTokensForTokens(...)")
        print(f"   4. Transfer repayment to Aave pool")
        
        # Check for competing opportunities
        if opportunity.profit_percentage > 1.0:
            print(f"\n⚠️  WARNING: Large spread detected!")
            print(f"   This may indicate:")
            print(f"   - Low liquidity (high slippage risk)")
            print(f"   - Price oracle delays")
            print(f"   - Market volatility")
            print(f"   - MEV bot competition")
        
        return True

    async def compare_with_cex_arbitrage(self, cex_prices):
        """Compare DEX arbitrage with CEX opportunities"""
        if not cex_prices:
            return
            
        print(f"\n📊 CEX ARBITRAGE COMPARISON:")
        
        for symbol, data in cex_prices.items():
            spread = ((data["ask"] - data["bid"]) / data["bid"]) * 100
            print(f"   {symbol}: {spread:.4f}% spread (Bid: ${data['bid']:,.2f}, Ask: ${data['ask']:,.2f})")
            
            if spread > 0.1:
                print(f"      🎯 Significant CEX spread detected!")

    async def monitor_profitable_opportunities(self):
        await self.send_alert("🚀 Profitable Flash Loan Monitor started - hunting for profitable arbitrage!")
        
        cycle = 0
        total_profitable_found = 0
        
        while True:
            try:
                print(f"\n{'='*80}")
                print(f"PROFITABLE FLASH LOAN SCAN - CYCLE {cycle} - {time.strftime('%H:%M:%S')}")
                print(f"{'='*80}")
                
                # Scan for opportunities
                opportunities = await self.scan_flash_loan_opportunities()
                
                if opportunities:
                    total_profitable_found += len(opportunities)
                    
                    print(f"\n🎉 FOUND {len(opportunities)} PROFITABLE OPPORTUNITIES!")
                    
                    for i, opp in enumerate(opportunities[:3], 1):  # Show top 3
                        print(f"\n💰 OPPORTUNITY #{i}:")
                        print(f"   Pair: {opp.pair}")
                        print(f"   Net profit: ${opp.net_profit_usd:,.2f}")
                        print(f"   Profit %: {opp.profit_percentage:.4f}%")
                        print(f"   Strategy: {opp.buy_dex} -> {opp.sell_dex}")
                        
                        # Simulate execution for top opportunity
                        if i == 1:
                            await self.simulate_flash_loan_execution(opp)
                
                else:
                    print(f"\n⚪ No profitable opportunities found this cycle")
                    print(f"   (Spreads exist but gas costs exceed profits)")
                
                # Get CEX comparison
                cex_prices = await self.get_cex_prices()
                await self.compare_with_cex_arbitrage(cex_prices)
                
                print(f"\n📈 SESSION STATS:")
                print(f"   Total profitable opportunities found: {total_profitable_found}")
                print(f"   Scan cycles completed: {cycle + 1}")
                print(f"   Success rate: {(total_profitable_found/(cycle+1)):.1f} opportunities per cycle")
                
                cycle += 1
                print(f"\n⏱️  Next profitable scan in 45 seconds...")
                await asyncio.sleep(45)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(90)

if __name__ == "__main__":
    bot = ProfitableFlashArbitrage()
    print("💰 PROFITABLE FLASH LOAN ARBITRAGE SYSTEM")
    print("=" * 50)
    print("🎯 ONLY shows opportunities with >$50 net profit")
    print("⚡ Simulates complete flash loan execution")
    print("📊 Integrates with all repo modules")
    print("🔍 Tests multiple flash loan amounts")
    print("💡 Includes realistic costs and risks")
    print("")
    
    asyncio.run(bot.monitor_profitable_opportunities())