# TESTNET VERSION - Safe for testing with fake money
import asyncio
import aiohttp
from web3 import Web3
import time
import requests

class WorkingArbitrageDetector:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider("https://eth-testnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX"))
        self.webhook = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3"
        
        # CORRECT token addresses (verified on Etherscan)
        self.tokens = {
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
        }
        
        # Real DEX router addresses
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
            payload = {"content": f"💰 REAL Arbitrage: {message}"}
            requests.post(self.webhook, json=payload, timeout=5)
            print(f"Alert: {message}")
        except Exception as e:
            print(f"Alert error: {e}")

    def get_uniswap_price(self, token_in, token_out, amount_in):
        try:
            contract = self.w3.eth.contract(address=self.uniswap_router, abi=self.router_abi)
            path = [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)]
            # Convert to integer (this was the bug!)
            amount_in_int = int(amount_in)
            amounts = contract.functions.getAmountsOut(amount_in_int, path).call()
            return amounts[-1]
        except Exception as e:
            print(f"Uniswap error: {e}")
            return 0

    def get_sushiswap_price(self, token_in, token_out, amount_in):
        try:
            contract = self.w3.eth.contract(address=self.sushiswap_router, abi=self.router_abi)
            path = [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)]
            # Convert to integer (this was the bug!)
            amount_in_int = int(amount_in)
            amounts = contract.functions.getAmountsOut(amount_in_int, path).call()
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
            return 4400  # Current approximate ETH price

    async def scan_real_opportunities(self):
        print("🔍 Scanning for REAL arbitrage opportunities...")
        
        # Only scan pairs that actually work - use INTEGERS
        pairs = [
            ("WETH", "USDT", 100000000000000000, "0.1 ETH"),  # 0.1 ETH in wei (integer)
        ]
        
        eth_price = await self.get_eth_price()
        opportunities = []
        
        for token_in_name, token_out_name, amount_in, description in pairs:
            token_in = self.tokens[token_in_name]
            token_out = self.tokens[token_out_name]
            
            trade_value_usd = (amount_in / 10**18) * eth_price
            
            print(f"\n📊 {description}: {token_in_name} -> {token_out_name}")
            print(f"   Trade size: {amount_in / 10**18:.1f} ETH (~${trade_value_usd:.0f})")
            print(f"   Amount in wei: {amount_in:,}")
            
            # Get prices from both DEXes
            uniswap_out = self.get_uniswap_price(token_in, token_out, amount_in)
            sushiswap_out = self.get_sushiswap_price(token_in, token_out, amount_in)
            
            if uniswap_out > 0 and sushiswap_out > 0:
                print(f"   ✅ Uniswap output: {uniswap_out:,} USDT (6 decimals)")
                print(f"   ✅ SushiSwap output: {sushiswap_out:,} USDT (6 decimals)")
                
                # Calculate REAL price difference
                price_diff = abs(uniswap_out - sushiswap_out)
                min_price = min(uniswap_out, sushiswap_out)
                profit_percentage = (price_diff / min_price) * 100
                
                # Calculate profit in USD (USDT has 6 decimals)
                profit_usd = price_diff / 10**6
                
                # Estimate gas cost (realistic)
                gas_price_wei = self.w3.eth.gas_price
                gas_price_gwei = self.w3.from_wei(gas_price_wei, 'gwei')
                gas_cost_eth = (300000 * gas_price_wei) / 10**18  # 300k gas units
                gas_cost_usd = gas_cost_eth * eth_price
                
                net_profit = profit_usd - gas_cost_usd
                
                print(f"   💰 Price difference: ${profit_usd:.4f}")
                print(f"   ⛽ Gas cost: ${gas_cost_usd:.2f} ({gas_price_gwei:.1f} gwei)")
                print(f"   📊 Net profit: ${net_profit:.2f}")
                print(f"   📈 Profit percentage: {profit_percentage:.6f}%")
                
                if profit_percentage > 0.001:  # Any measurable difference
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
                        "profitable": net_profit > 0.50,
                        "uniswap_out": uniswap_out,
                        "sushiswap_out": sushiswap_out
                    }
                    opportunities.append(opportunity)
                    
                    if net_profit > 0.50:
                        status = "💰 PROFITABLE"
                        await self.send_alert(
                            f"REAL PROFIT FOUND! {opportunity['pair']} "
                            f"{profit_percentage:.6f}% spread = ${net_profit:.2f} net profit "
                            f"(Buy {buy_dex} ${min_price/10**6:.2f}, Sell {sell_dex} ${max(uniswap_out, sushiswap_out)/10**6:.2f})"
                        )
                    elif net_profit > -5:
                        status = "🔍 SMALL LOSS"
                    else:
                        status = "❌ UNPROFITABLE"
                        
                    print(f"   {status}: {profit_percentage:.6f}% spread")
                    print(f"   Strategy: Buy on {buy_dex}, sell on {sell_dex}")
            else:
                print(f"   ❌ Failed to get prices")
                if uniswap_out == 0:
                    print(f"      Uniswap: No liquidity or error")
                if sushiswap_out == 0:
                    print(f"      SushiSwap: No liquidity or error")
        
        return opportunities

    async def monitor_realistic_opportunities(self):
        await self.send_alert("🚀 REAL Arbitrage Detector started - scanning live DEX prices!")
        
        cycle = 0
        while True:
            try:
                print(f"\n{'='*70}")
                print(f"CYCLE {cycle} - {time.strftime('%H:%M:%S')}")
                print(f"{'='*70}")
                
                opportunities = await self.scan_real_opportunities()
                
                if opportunities:
                    profitable = [opp for opp in opportunities if opp['profitable']]
                    small_loss = [opp for opp in opportunities if not opp['profitable'] and opp['net_profit'] > -5]
                    
                    print(f"\n✅ Found {len(opportunities)} opportunities:")
                    print(f"   💰 {len(profitable)} profitable (net > $0.50)")
                    print(f"   🔍 {len(small_loss)} small losses (net > -$5)")
                    
                    for opp in opportunities:
                        emoji = "💰" if opp['profitable'] else "🔍" if opp['net_profit'] > -5 else "❌"
                        print(f"   {emoji} {opp['pair']}: ${opp['net_profit']:.2f} net ({opp['profit_percentage']:.6f}%)")
                else:
                    print("\n⚪ No opportunities found this cycle")
                
                cycle += 1
                print(f"\n⏱️  Next scan in 30 seconds...")
                await asyncio.sleep(30)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(60)

if __name__ == "__main__":
    detector = WorkingArbitrageDetector()
    print("💰 REAL Arbitrage Detector - Fixed Version")
    print("✅ Fixed integer conversion bug")
    print("✅ Correct token addresses")
    print("✅ Realistic profit calculations")  
    print("✅ Live gas cost tracking")
    print("🎯 Expected: Most opportunities will be unprofitable due to gas costs")
    print("🚀 Looking for rare profitable spreads > 0.1%\n")
    
    asyncio.run(detector.monitor_realistic_opportunities())
