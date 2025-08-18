# TESTNET VERSION - Safe for testing with fake money
import asyncio
import aiohttp
from web3 import Web3
import time
import requests

class RealisticArbitrageDetector:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider("https://eth-testnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX"))
        self.webhook = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3"
        
        # REAL token addresses from Etherscan
        self.tokens = {
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "USDC": "0xA0b86a33E6417c7ef38BC67B2F11D6B3DC0B5f55",  # Real USDC
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
        }
        
        # Real DEX routers
        self.uniswap_router = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
        self.sushiswap_router = "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F"
        
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
            payload = {"content": f"💰 Realistic Arbitrage: {message}"}
            requests.post(self.webhook, json=payload, timeout=5)
            print(f"Alert: {message}")
        except Exception as e:
            print(f"Alert error: {e}")

    def get_uniswap_price(self, token_in, token_out, amount_in):
        try:
            contract = self.w3.eth.contract(address=self.uniswap_router, abi=self.router_abi)
            path = [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)]
            amounts = contract.functions.getAmountsOut(amount_in, path).call()
            return amounts[-1]
        except Exception as e:
            print(f"Uniswap error: {e}")
            return 0

    def get_sushiswap_price(self, token_in, token_out, amount_in):
        try:
            contract = self.w3.eth.contract(address=self.sushiswap_router, abi=self.router_abi)
            path = [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)]
            amounts = contract.functions.getAmountsOut(amount_in, path).call()
            return amounts[-1]
        except Exception as e:
            print(f"SushiSwap error: {e}")
            return 0

    async def get_eth_price(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd") as response:
                    data = await response.json()
                    return data["ethereum"]["usd"]
        except:
            return 3200

    async def scan_realistic_opportunities(self):
        print("🔍 Scanning for REALISTIC arbitrage opportunities...")
        
        # Small, realistic trading amounts
        pairs = [
            ("WETH", "USDT", 0.1 * 10**18),    # 0.1 ETH = ~$320
            ("WBTC", "USDC", 0.001 * 10**8),   # 0.001 BTC = ~$65
        ]
        
        eth_price = await self.get_eth_price()
        opportunities = []
        
        for token_in_name, token_out_name, amount_in in pairs:
            token_in = self.tokens[token_in_name]
            token_out = self.tokens[token_out_name]
            
            print(f"\n📊 Checking {token_in_name} -> {token_out_name} (${amount_in/(10**18)*eth_price:.0f} trade)")
            
            uniswap_out = self.get_uniswap_price(token_in, token_out, amount_in)
            sushiswap_out = self.get_sushiswap_price(token_in, token_out, amount_in)
            
            if uniswap_out > 0 and sushiswap_out > 0:
                print(f"   Uniswap: {uniswap_out}")
                print(f"   SushiSwap: {sushiswap_out}")
                
                # Calculate real price difference
                price_diff = abs(uniswap_out - sushiswap_out)
                min_price = min(uniswap_out, sushiswap_out)
                profit_percentage = (price_diff / min_price) * 100
                
                # Calculate profit in USD (realistic)
                if token_out_name == "USDT":
                    profit_usd = price_diff / 10**6  # USDT has 6 decimals
                elif token_out_name == "USDC":
                    profit_usd = price_diff / 10**6  # USDC has 6 decimals
                else:
                    profit_usd = (price_diff / 10**18) * eth_price  # Convert ETH to USD
                
                # Estimate gas cost
                gas_price_gwei = self.w3.from_wei(self.w3.eth.gas_price, 'gwei')
                gas_cost_usd = float(gas_price_gwei) * 200000 * eth_price / 10**9  # ~$10-50 typical
                
                net_profit = profit_usd - gas_cost_usd
                
                print(f"   Profit: {profit_percentage:.4f}% = ${profit_usd:.2f}")
                print(f"   Gas cost: ${gas_cost_usd:.2f}")
                print(f"   Net profit: ${net_profit:.2f}")
                
                if profit_percentage > 0.1:  # More than 0.1% profit
                    buy_dex = "Uniswap" if uniswap_out > sushiswap_out else "SushiSwap"
                    sell_dex = "SushiSwap" if uniswap_out > sushiswap_out else "Uniswap"
                    
                    opportunity = {
                        "pair": f"{token_in_name}->{token_out_name}",
                        "profit_percentage": profit_percentage,
                        "profit_usd": profit_usd,
                        "gas_cost_usd": gas_cost_usd,
                        "net_profit": net_profit,
                        "buy_dex": buy_dex,
                        "sell_dex": sell_dex,
                        "trade_value": amount_in/(10**18)*eth_price if token_in_name == "WETH" else 65
                    }
                    opportunities.append(opportunity)
                    
                    if net_profit > 2:  # Alert if net profit > $2
                        await self.send_alert(
                            f"Real opportunity: {opportunity['pair']} "
                            f"{profit_percentage:.4f}% profit = ${net_profit:.2f} net "
                            f"(Buy {buy_dex}, Sell {sell_dex})"
                        )
                        print(f"   🚨 REAL OPPORTUNITY DETECTED!")
        
        return opportunities

    async def monitor_continuously(self):
        await self.send_alert("🚀 Realistic Arbitrage Detector started!")
        
        cycle = 0
        while True:
            try:
                print(f"\n{'='*50}")
                print(f"CYCLE {cycle} - {time.strftime('%H:%M:%S')}")
                print(f"{'='*50}")
                
                opportunities = await self.scan_realistic_opportunities()
                
                if opportunities:
                    print(f"\n✅ Found {len(opportunities)} realistic opportunities:")
                    for opp in opportunities:
                        print(f"   {opp['pair']}: ${opp['net_profit']:.2f} net profit")
                else:
                    print("\n⚪ No profitable opportunities found")
                
                cycle += 1
                print(f"\n⏱️  Next scan in 60 seconds...")
                await asyncio.sleep(60)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(120)

if __name__ == "__main__":
    detector = RealisticArbitrageDetector()
    print("💰 REALISTIC Arbitrage Detector")
    print("🎯 Target: $2-20 profit per opportunity")
    print("⛽ Includes real gas costs")
    print("📊 Small trade sizes for testing\n")
    
    asyncio.run(detector.monitor_continuously())
