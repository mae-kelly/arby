import asyncio
import aiohttp
import json
import time
import hmac
import hashlib
import base64
import requests

class OKXClient:
    def __init__(self):
        self.api_key = "8a760df1-4a2d-471b-ba42-d16893614dab"
        self.secret_key = "C9F3FC89A6A30226E11DFFD098C7CF3D"
        self.passphrase = "trading_bot_2024"
        self.base_url = "https://www.okx.com"
        
    def sign(self, timestamp, method, request_path, body=""):
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()
    
    async def get_ticker(self, symbol):
        timestamp = str(int(time.time() * 1000))
        method = "GET"
        request_path = f"/api/v5/market/ticker?instId={symbol}"
        
        headers = {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': self.sign(timestamp, method, request_path),
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}{request_path}", headers=headers) as response:
                    return await response.json()
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return {"code": "1", "msg": str(e)}

    async def get_account_balance(self):
        timestamp = str(int(time.time() * 1000))
        method = "GET"
        request_path = "/api/v5/account/balance"
        
        headers = {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': self.sign(timestamp, method, request_path),
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}{request_path}", headers=headers) as response:
                    return await response.json()
        except Exception as e:
            print(f"Balance error: {e}")
            return {"code": "1", "msg": str(e)}

class RealArbitrageBot:
    def __init__(self):
        self.okx_client = OKXClient()
        self.webhook = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3"
        self.min_spread_threshold = 0.05  # 0.05% minimum spread
        
    def send_alert(self, message):
        try:
            payload = {"content": f"🤖 REAL CEX Bot: {message}"}
            requests.post(self.webhook, json=payload, timeout=5)
            print(f"Alert: {message}")
        except Exception as e:
            print(f"Discord error: {e}")

    async def check_account_status(self):
        balance = await self.okx_client.get_account_balance()
        
        if balance.get("code") == "0":
            print("✅ OKX API connection successful")
            
            if balance.get("data"):
                for account in balance["data"]:
                    for detail in account.get("details", []):
                        if float(detail.get("bal", 0)) > 0:
                            currency = detail.get("ccy")
                            balance_amount = detail.get("bal")
                            print(f"   {currency}: {balance_amount}")
            else:
                print("   No balance data (likely empty account)")
        else:
            print(f"❌ OKX API Error: {balance.get('msg')}")
            self.send_alert(f"OKX API Issue: {balance.get('msg')}")

    def calculate_real_arbitrage(self, prices):
        opportunities = []
        
        for symbol, data in prices.items():
            if data.get("code") == "0" and data.get("data"):
                ticker = data["data"][0]
                
                bid_price = float(ticker.get("bidPx", 0))
                ask_price = float(ticker.get("askPx", 0))
                last_price = float(ticker.get("last", 0))
                
                if bid_price > 0 and ask_price > 0:
                    # Calculate spread percentage
                    spread = ((ask_price - bid_price) / bid_price) * 100
                    
                    if spread > self.min_spread_threshold:
                        # Calculate potential profit on small trade
                        trade_amount = 0.001  # Very small amount for testing
                        potential_profit = (ask_price - bid_price) * trade_amount
                        
                        opportunity = {
                            "symbol": symbol,
                            "bid": bid_price,
                            "ask": ask_price,
                            "last": last_price,
                            "spread_percentage": spread,
                            "potential_profit": potential_profit,
                            "trade_amount": trade_amount
                        }
                        opportunities.append(opportunity)
        
        return sorted(opportunities, key=lambda x: x["spread_percentage"], reverse=True)
    
    async def run(self):
        self.send_alert("REAL CEX Arbitrage Bot started - checking actual spreads!")
        
        # Check account status first
        await self.check_account_status()
        
        symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
        cycle = 0
        
        while True:
            try:
                print(f"\n[{time.strftime('%H:%M:%S')}] Cycle {cycle}: Checking real spreads...")
                
                # Get all prices
                prices = {}
                for symbol in symbols:
                    ticker = await self.okx_client.get_ticker(symbol)
                    prices[symbol] = ticker
                    
                    if ticker.get("code") == "0" and ticker.get("data"):
                        data = ticker["data"][0]
                        bid = data.get("bidPx")
                        ask = data.get("askPx") 
                        last = data.get("last")
                        spread = ((float(ask) - float(bid)) / float(bid)) * 100
                        print(f"  {symbol}: ${last} | Spread: {spread:.4f}% | Bid: ${bid} | Ask: ${ask}")
                
                # Find arbitrage opportunities
                opportunities = self.calculate_real_arbitrage(prices)
                
                if opportunities:
                    print(f"\n💰 Found {len(opportunities)} spread opportunities:")
                    
                    for opp in opportunities:
                        print(f"   {opp['symbol']}: {opp['spread_percentage']:.4f}% spread")
                        print(f"      Potential profit on {opp['trade_amount']} trade: ${opp['potential_profit']:.6f}")
                        
                        if opp['spread_percentage'] > 0.1:  # Alert on >0.1% spreads
                            self.send_alert(
                                f"Spread Alert: {opp['symbol']} has {opp['spread_percentage']:.4f}% spread "
                                f"(${opp['potential_profit']:.6f} profit on tiny trade)"
                            )
                else:
                    print("   No significant spreads found")
                
                cycle += 1
                await asyncio.sleep(10)
                
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(30)

if __name__ == "__main__":
    bot = RealArbitrageBot()
    print("🔥 REAL CEX Arbitrage Scanner")
    print("📊 Monitoring: BTC, ETH, SOL spreads on OKX")
    print("💰 Realistic profit tracking")
    print("⚠️  MONITORING ONLY - No actual trading\n")
    
    asyncio.run(bot.run())
