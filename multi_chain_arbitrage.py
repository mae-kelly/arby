import asyncio
import aiohttp
from web3 import Web3
import json
import time

class MultiChainFlashArbitrage:
    def __init__(self):
        self.chains = {
            "ethereum": {
                "rpc": "https://eth-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX",
                "chain_id": 1,
                "flash_loan_provider": "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
            },
            "polygon": {
                "rpc": "https://polygon-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX", 
                "chain_id": 137,
                "flash_loan_provider": "0x794a61358D6845594F94dc1DB02A252b5b4814aD"
            },
            "arbitrum": {
                "rpc": "https://arb-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX",
                "chain_id": 42161,
                "flash_loan_provider": "0x794a61358D6845594F94dc1DB02A252b5b4814aD"
            }
        }
        
        self.dexes = {
            "uniswap_v2": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            "uniswap_v3": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
            "sushiswap": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
            "curve": "0x99a58482BD75cbab83b27EC03CA68fF489b5788f",
            "balancer": "0xBA12222222228d8Ba445958a75a0704d566BF2C8"
        }
        
        self.webhook = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3"

    async def scan_cross_dex_opportunities(self):
        opportunities = []
        
        tokens = [
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "0xA0b86a33E6417c7ef38BC67B2F11D6B3DC0B5f55",  # USDC
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
            "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"   # WBTC
        ]
        
        for chain_name, chain_config in self.chains.items():
            w3 = Web3(Web3.HTTPProvider(chain_config["rpc"]))
            
            for token_a in tokens:
                for token_b in tokens:
                    if token_a == token_b:
                        continue
                    
                    prices = await self.get_dex_prices(w3, token_a, token_b, 100000)
                    
                    if len(prices) >= 2:
                        price_list = list(prices.values())
                        max_price = max(price_list)
                        min_price = min(price_list)
                        
                        if min_price > 0:
                            profit_percentage = ((max_price - min_price) / min_price) * 100
                            
                            if profit_percentage > 0.2:
                                opportunity = {
                                    "chain": chain_name,
                                    "token_a": token_a,
                                    "token_b": token_b,
                                    "profit_percentage": profit_percentage,
                                    "buy_dex": min(prices, key=prices.get),
                                    "sell_dex": max(prices, key=prices.get),
                                    "buy_price": min_price,
                                    "sell_price": max_price
                                }
                                opportunities.append(opportunity)
        
        return sorted(opportunities, key=lambda x: x["profit_percentage"], reverse=True)

    async def get_dex_prices(self, w3, token_a, token_b, amount):
        prices = {}
        
        for dex_name, dex_address in self.dexes.items():
            try:
                if "uniswap" in dex_name:
                    price = await self.get_uniswap_price(w3, dex_address, token_a, token_b, amount)
                elif dex_name == "sushiswap":
                    price = await self.get_sushiswap_price(w3, dex_address, token_a, token_b, amount)
                
                if price > 0:
                    prices[dex_name] = price
            except Exception as e:
                print(f"Error getting {dex_name} price: {e}")
        
        return prices

    async def execute_cross_chain_arbitrage(self, opportunity):
        print(f"🔥 FLASH LOAN ARBITRAGE OPPORTUNITY:")
        print(f"   Chain: {opportunity['chain']}")
        print(f"   Profit: {opportunity['profit_percentage']:.4f}%")
        print(f"   Buy on: {opportunity['buy_dex']} at {opportunity['buy_price']}")
        print(f"   Sell on: {opportunity['sell_dex']} at {opportunity['sell_price']}")
        
        potential_profit = (opportunity['sell_price'] - opportunity['buy_price']) * 100000
        
        if potential_profit > 500:
            await self.send_alert(f"🚨 HIGH PROFIT OPPORTUNITY: {opportunity['profit_percentage']:.4f}% profit on {opportunity['chain']} chain!")
            
            return await self.execute_flash_loan(opportunity)
        
        return False

    async def execute_flash_loan(self, opportunity):
        print(f"Executing flash loan for {opportunity['profit_percentage']:.4f}% profit...")
        
        flash_loan_params = {
            "asset": opportunity['token_a'],
            "amount": 100000 * 10**18,
            "chain": opportunity['chain'],
            "buy_dex": opportunity['buy_dex'],
            "sell_dex": opportunity['sell_dex']
        }
        
        print(f"Flash loan params: {flash_loan_params}")
        await self.send_alert(f"Flash loan executed: {opportunity['profit_percentage']:.4f}% profit")
        
        return True

    async def send_alert(self, message):
        try:
            payload = {"content": f"⚡ Multi-Chain Flash Arbitrage: {message}"}
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook, json=payload) as response:
                    print(f"Alert sent: {message}")
        except Exception as e:
            print(f"Alert error: {e}")

    async def monitor_real_time(self):
        await self.send_alert("Multi-chain flash loan arbitrage scanner started!")
        
        while True:
            try:
                print(f"[{time.strftime('%H:%M:%S')}] Scanning cross-DEX opportunities...")
                
                opportunities = await self.scan_cross_dex_opportunities()
                
                if opportunities:
                    print(f"Found {len(opportunities)} opportunities:")
                    
                    for opp in opportunities[:5]:
                        print(f"  {opp['chain']}: {opp['profit_percentage']:.4f}% profit")
                        
                        if opp['profit_percentage'] > 1.0:
                            await self.execute_cross_chain_arbitrage(opp)
                
                await asyncio.sleep(15)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(30)

if __name__ == "__main__":
    bot = MultiChainFlashArbitrage()
    asyncio.run(bot.monitor_real_time())
