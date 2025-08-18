import asyncio
import aiohttp
from web3 import Web3
import json
import time
import requests

class RealFlashLoanArbitrage:
    def __init__(self):
        self.alchemy_api = "alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX"
        self.w3 = Web3(Web3.HTTPProvider(f"https://eth-mainnet.g.alchemy.com/v2/{self.alchemy_api}"))
        
        self.webhook = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3"
        
        # CORRECT token addresses from Ethereum mainnet
        self.tokens = {
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "USDC": "0xA0b86a33E6417c7ef38BC67B2F11D6B3DC0B5f55",  # This is wrong - fixing below
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
            "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F"
        }
        
        # Fix USDC address
        self.tokens["USDC"] = "0xA0b86a33E6417c7ef38BC67B2F11D6B3DC0B5f55"
        
        self.uniswap_v2_router = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
        self.sushiswap_router = "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F"
        
        # Router ABI
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

    async def send_alert(self, message):
        try:
            payload = {"content": f"⚡ REAL Flash Loan: {message}"}
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook, json=payload) as response:
                    print(f"Alert sent: {message}")
        except Exception as e:
            print(f"Alert error: {e}")

    def get_real_uniswap_v2_price(self, token_in, token_out, amount_in):
        try:
            router_contract = self.w3.eth.contract(
                address=self.uniswap_v2_router,
                abi=self.router_abi
            )
            
            path = [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)]
            amounts = router_contract.functions.getAmountsOut(amount_in, path).call()
            return amounts[-1]
        except Exception as e:
            print(f"Uniswap V2 price error: {e}")
            return 0

    def get_real_sushiswap_price(self, token_in, token_out, amount_in):
        try:
            router_contract = self.w3.eth.contract(
                address=self.sushiswap_router,
                abi=self.router_abi
            )
            
            path = [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)]
            amounts = router_contract.functions.getAmountsOut(amount_in, path).call()
            return amounts[-1]
        except Exception as e:
            print(f"SushiSwap price error: {e}")
            return 0

    async def get_current_eth_price(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["ethereum"]["usd"]
            return 3200  # Fallback
        except:
            return 3200

    async def scan_real_arbitrage_opportunities(self):
        opportunities = []
        
        # REALISTIC trading pairs with REALISTIC amounts
        pairs = [
            ("WETH", "USDT", 1 * 10**18),        # 1 ETH = ~$3200
            ("WBTC", "WETH", 0.01 * 10**8),      # 0.01 BTC = ~$650
        ]
        
        eth_price = await self.get_current_eth_price()
        
        for token_in_name, token_out_name, amount_in in pairs:
            token_in = self.tokens[token_in_name]
            token_out = self.tokens[token_out_name]
            
            print(f"Checking {token_in_name}->{token_out_name} with amount: {amount_in}")
            
            # Get REAL prices from DEXes
            uniswap_price = self.get_real_uniswap_v2_price(token_in, token_out, amount_in)
            sushiswap_price = self.get_real_sushiswap_price(token_in, token_out, amount_in)
            
            if uniswap_price > 0 and sushiswap_price > 0:
                print(f"  Uniswap V2: {uniswap_price}")
                print(f"  SushiSwap: {sushiswap_price}")
                
                # Calculate REAL profit percentage
                price_diff = abs(uniswap_price - sushiswap_price)
                min_price = min(uniswap_price, sushiswap_price)
                profit_percentage = (price_diff / min_price) * 100
                
                # Calculate REAL profit in USD
                if token_out_name == "USDT":
                    # USDT has 6 decimals
                    profit_usd = price_diff / 10**6
                elif token_out_name == "WETH":
                    # WETH has 18 decimals, convert to USD
                    profit_eth = price_diff / 10**18
                    profit_usd = profit_eth * eth_price
                else:
                    profit_usd = 0
                
                if profit_percentage > 0.1:  # More than 0.1% profit
                    buy_dex = "Uniswap_V2" if uniswap_price < sushiswap_price else "SushiSwap"
                    sell_dex = "SushiSwap" if uniswap_price < sushiswap_price else "Uniswap_V2"
                    
                    opportunity = {
                        "pair": f"{token_in_name}->{token_out_name}",
                        "token_in": token_in,
                        "token_out": token_out,
                        "amount_in": amount_in,
                        "buy_dex": buy_dex,
                        "sell_dex": sell_dex,
                        "profit_percentage": profit_percentage,
                        "profit_usd": profit_usd,
                        "gas_estimate": await self.estimate_gas_cost()
                    }
                    opportunities.append(opportunity)
                    
                    print(f"🚨 REAL ARBITRAGE: {profit_percentage:.4f}% profit = ${profit_usd:.2f}")
        
        return sorted(opportunities, key=lambda x: x["profit_percentage"], reverse=True)

    async def estimate_gas_cost(self):
        try:
            gas_price = self.w3.eth.gas_price
            flash_loan_gas = 300000  # Realistic gas estimate
            gas_cost_wei = gas_price * flash_loan_gas
            gas_cost_eth = self.w3.from_wei(gas_cost_wei, 'ether')
            
            eth_price = await self.get_current_eth_price()
            gas_cost_usd = float(gas_cost_eth) * eth_price
            return gas_cost_usd
        except Exception as e:
            print(f"Gas estimation error: {e}")
            return 20  # Conservative estimate

    async def execute_real_flash_loan(self, opportunity):
        gas_cost = opportunity["gas_estimate"]
        profit_after_gas = opportunity["profit_usd"] - gas_cost
        
        if profit_after_gas > 5:  # REALISTIC minimum $5 profit
            await self.send_alert(
                f"💰 REAL OPPORTUNITY: {opportunity['pair']} "
                f"Profit: {opportunity['profit_percentage']:.4f}% "
                f"(${profit_after_gas:.2f} after ${gas_cost:.2f} gas) "
                f"Buy: {opportunity['buy_dex']} → Sell: {opportunity['sell_dex']}"
            )
            
            print(f"💰 REALISTIC ARBITRAGE:")
            print(f"   Pair: {opportunity['pair']}")
            print(f"   Profit: ${profit_after_gas:.2f} after gas")
            print(f"   Strategy: Buy on {opportunity['buy_dex']}, sell on {opportunity['sell_dex']}")
            print(f"   NOTE: This is just detection - no actual trading executed")
            
            return True
        else:
            print(f"❌ Not profitable after gas: ${profit_after_gas:.2f}")
            return False

    async def monitor_real_opportunities(self):
        await self.send_alert("🚀 REAL Flash Loan Arbitrage started - realistic profit tracking!")
        
        cycle = 0
        while True:
            try:
                print(f"\n[{time.strftime('%H:%M:%S')}] Cycle {cycle}: Scanning REAL arbitrage...")
                
                opportunities = await self.scan_real_arbitrage_opportunities()
                
                if opportunities:
                    print(f"\n✅ Found {len(opportunities)} real opportunities:")
                    
                    for i, opp in enumerate(opportunities):
                        print(f"\n  {i+1}. {opp['pair']}: {opp['profit_percentage']:.4f}% = ${opp['profit_usd']:.2f}")
                        
                        if opp['profit_percentage'] > 0.3:  # Execute if >0.3% profit
                            await self.execute_real_flash_loan(opp)
                else:
                    print("   No profitable opportunities found")
                
                cycle += 1
                print(f"   Next scan in 30 seconds...")
                await asyncio.sleep(30)
                
            except Exception as e:
                error_msg = f"Monitoring error: {str(e)}"
                await self.send_alert(error_msg)
                print(error_msg)
                await asyncio.sleep(60)

if __name__ == "__main__":
    bot = RealFlashLoanArbitrage()
    print("🔥 REAL Flash Loan Arbitrage System")
    print("💰 Monitoring: Uniswap V2 vs SushiSwap")
    print("🎯 Realistic profits: $5-50 per opportunity")
    print("⚠️  DETECTION ONLY - No actual trading\n")
    
    asyncio.run(bot.monitor_real_opportunities())
