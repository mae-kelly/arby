#!/usr/bin/env python3
"""
FIXED OKX ALL TOKENS ARBITRAGE SCANNER
Complete standalone version - monitors EVERY token on OKX
"""

import asyncio
import aiohttp
import hmac
import hashlib
import base64
import time
import json
from datetime import datetime

class FixedOKXAllTokensScanner:
    def __init__(self):
        self.api_key = "8a760df1-4a2d-471b-ba42-d16893614dab"
        self.secret_key = "C9F3FC89A6A30226E11DFFD098C7CF3D"
        self.passphrase = "Shamrock1!"
        self.base_url = "https://www.okx.com"
        self.webhook = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3"
        
        # Settings
        self.min_spread_threshold = 0.01  # 0.01% minimum
        self.min_volume_24h = 1000  # $1000 minimum 24h volume
        
        # Priority token lists
        self.tier_1_tokens = [
            "BTC-USDT", "ETH-USDT", "BNB-USDT", "XRP-USDT", "ADA-USDT",
            "DOGE-USDT", "SOL-USDT", "TRX-USDT", "LTC-USDT", "MATIC-USDT",
            "DOT-USDT", "AVAX-USDT", "SHIB-USDT", "UNI-USDT", "ATOM-USDT",
            "LINK-USDT", "AAVE-USDT", "SUSHI-USDT", "CRV-USDT", "COMP-USDT"
        ]
        
        # Statistics
        self.total_scanned = 0
        self.opportunities_found = 0
        self.best_opportunity = None
        self.scan_count = 0
        
    def sign(self, timestamp, method, request_path, body=""):
        """Sign OKX API request"""
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()
    
    async def send_alert(self, message):
        """Send Discord alert"""
        try:
            payload = {"content": f"🌍 ALL OKX TOKENS: {message}"}
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook, json=payload, timeout=10) as response:
                    if response.status == 204:
                        print(f"📱 Alert sent: {message[:100]}...")
                    else:
                        print(f"⚠️ Alert failed: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Alert error: {e}")
    
    async def get_all_tickers(self):
        """Get ALL ticker data from OKX"""
        print("📊 Fetching ALL token data from OKX...")
        
        timestamp = str(int(time.time() * 1000))
        method = "GET"
        request_path = "/api/v5/market/tickers?instType=SPOT"
        
        headers = {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': self.sign(timestamp, method, request_path),
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}{request_path}", headers=headers, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("code") == "0":
                            tickers = data.get("data", [])
                            print(f"✅ Retrieved data for {len(tickers)} tokens!")
                            return tickers
                        else:
                            print(f"❌ API error: {data.get('msg')}")
                            return []
                    else:
                        print(f"❌ HTTP error: {response.status}")
                        return []
        except Exception as e:
            print(f"❌ Error fetching tickers: {e}")
            return []
    
    def categorize_token(self, symbol):
        """Categorize token by priority"""
        if symbol in self.tier_1_tokens:
            return 1, "🥇"  # Tier 1: Major coins
        elif any(keyword in symbol.upper() for keyword in ["PEPE", "SHIB", "FLOKI", "DOGE"]):
            return 2, "🚀"  # Tier 2: Meme coins (high volatility)
        elif any(keyword in symbol.upper() for keyword in ["AI", "GPU", "BOT", "META", "GAME"]):
            return 3, "🤖"  # Tier 3: Tech/Gaming tokens
        else:
            return 4, "💎"  # Tier 4: Other tokens
    
    def analyze_arbitrage(self, ticker):
        """Analyze single ticker for arbitrage opportunity"""
        try:
            symbol = ticker.get("instId")
            if not symbol or "-" not in symbol:
                return None
                
            last = float(ticker.get("last", 0))
            bid = float(ticker.get("bidPx", 0))
            ask = float(ticker.get("askPx", 0))
            vol24h = float(ticker.get("vol24h", 0))
            volCcy24h = float(ticker.get("volCcy24h", 0))
            
            if bid <= 0 or ask <= 0 or last <= 0:
                return None
            
            # Calculate spread
            spread = ((ask - bid) / bid) * 100
            
            # Estimate volume in USD
            if "USDT" in symbol or "USDC" in symbol:
                volume_24h_usd = volCcy24h
            else:
                volume_24h_usd = vol24h * last
            
            # Check minimum criteria
            if spread < self.min_spread_threshold or volume_24h_usd < self.min_volume_24h:
                return None
            
            # Categorize token
            tier, emoji = self.categorize_token(symbol)
            
            return {
                "symbol": symbol,
                "tier": tier,
                "emoji": emoji,
                "price": last,
                "bid": bid,
                "ask": ask,
                "spread_percent": spread,
                "volume_24h_usd": volume_24h_usd,
                "potential_profit_1k": spread * 10,  # Profit on $1000 trade
                "priority_score": spread * 100 + min(volume_24h_usd / 10000, 50),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return None
    
    async def scan_all_tokens(self):
        """Main scanning function"""
        print(f"\n{'='*80}")
        print(f"🌍 SCANNING ALL OKX TOKENS - Scan #{self.scan_count}")
        print(f"{'='*80}")
        
        # Get all tickers
        all_tickers = await self.get_all_tickers()
        if not all_tickers:
            return []
        
        self.total_scanned = len(all_tickers)
        
        # Analyze each ticker
        opportunities = []
        processed = 0
        
        print(f"🔍 Analyzing {len(all_tickers)} tokens...")
        
        for ticker in all_tickers:
            opportunity = self.analyze_arbitrage(ticker)
            if opportunity:
                opportunities.append(opportunity)
            
            processed += 1
            if processed % 100 == 0:
                print(f"   Processed {processed}/{len(all_tickers)}...")
        
        # Sort by spread (best first)
        opportunities.sort(key=lambda x: x["spread_percent"], reverse=True)
        
        self.opportunities_found = len(opportunities)
        if opportunities:
            self.best_opportunity = opportunities[0]
        
        return opportunities
    
    def display_results(self, opportunities):
        """Display scan results"""
        print(f"\n📊 SCAN RESULTS:")
        print(f"   Tokens scanned: {self.total_scanned:,}")
        print(f"   Opportunities found: {len(opportunities):,}")
        if self.total_scanned > 0:
            success_rate = (len(opportunities) / self.total_scanned) * 100
            print(f"   Success rate: {success_rate:.2f}%")
        
        if opportunities:
            print(f"\n🏆 TOP ARBITRAGE OPPORTUNITIES:")
            print(f"{'Rank':<5} {'Token':<15} {'Tier':<5} {'Price':<12} {'Spread':<8} {'Volume':<12} {'Profit/1k':<10}")
            print("-" * 80)
            
            for i, opp in enumerate(opportunities[:20], 1):  # Show top 20
                print(f"{i:<5} {opp['symbol']:<15} {opp['emoji']:<5} "
                      f"${opp['price']:<11.6f} {opp['spread_percent']:<7.4f}% "
                      f"${opp['volume_24h_usd']:<11,.0f} ${opp['potential_profit_1k']:<9.2f}")
            
            # Show tier breakdown
            tier_counts = {}
            for opp in opportunities:
                tier = opp['tier']
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            print(f"\n📈 OPPORTUNITIES BY TIER:")
            tier_names = {1: "Major Coins", 2: "Meme Coins", 3: "Tech/Gaming", 4: "Other"}
            for tier in sorted(tier_counts.keys()):
                print(f"   Tier {tier} ({tier_names.get(tier, 'Unknown')}): {tier_counts[tier]} opportunities")
        else:
            print(f"\n⚪ No opportunities found meeting criteria:")
            print(f"   • Minimum spread: {self.min_spread_threshold}%")
            print(f"   • Minimum volume: ${self.min_volume_24h:,}")
    
    async def continuous_monitoring(self):
        """Main monitoring loop"""
        # Send startup alert
        await self.send_alert("🚀 Started monitoring ALL OKX tokens for arbitrage opportunities!")
        
        total_opportunities_found = 0
        
        while True:
            try:
                start_time = time.time()
                
                # Scan all tokens
                opportunities = await self.scan_all_tokens()
                
                # Display results
                self.display_results(opportunities)
                
                # Update statistics
                total_opportunities_found += len(opportunities)
                scan_time = time.time() - start_time
                
                print(f"\n📈 SESSION STATISTICS:")
                print(f"   Scan #{self.scan_count + 1} completed in {scan_time:.1f} seconds")
                print(f"   Total opportunities found: {total_opportunities_found}")
                if self.scan_count > 0:
                    avg_opps = total_opportunities_found / (self.scan_count + 1)
                    print(f"   Average per scan: {avg_opps:.1f}")
                
                if self.best_opportunity:
                    best = self.best_opportunity
                    print(f"   Best opportunity: {best['symbol']} ({best['spread_percent']:.4f}%)")
                
                # Send alerts for exceptional opportunities
                if opportunities:
                    best = opportunities[0]
                    if best["spread_percent"] > 0.1:  # > 0.1% spread
                        await self.send_alert(
                            f"🚨 HIGH SPREAD DETECTED!\n"
                            f"{best['emoji']} {best['symbol']}\n"
                            f"Spread: {best['spread_percent']:.4f}%\n"
                            f"Price: ${best['price']:,.6f}\n"
                            f"Volume: ${best['volume_24h_usd']:,.0f}\n"
                            f"Profit on $1000: ${best['potential_profit_1k']:.2f}"
                        )
                
                # Send periodic summary
                if self.scan_count > 0 and (self.scan_count + 1) % 10 == 0:
                    await self.send_alert(
                        f"📊 10-Scan Summary:\n"
                        f"Tokens monitored: {self.total_scanned:,}\n"
                        f"Total opportunities: {total_opportunities_found}\n"
                        f"Best spread: {self.best_opportunity['spread_percent']:.4f}% "
                        f"({self.best_opportunity['symbol']})" if self.best_opportunity else "No major opportunities"
                    )
                
                self.scan_count += 1
                
                # Wait before next scan
                wait_time = 60  # 1 minute between scans
                print(f"\n⏱️  Next ALL-TOKENS scan in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                
            except KeyboardInterrupt:
                print(f"\n🛑 Stopping scanner...")
                break
            except Exception as e:
                print(f"❌ Scan error: {e}")
                await self.send_alert(f"⚠️ Scanner error: {str(e)[:100]}")
                await asyncio.sleep(120)  # Wait 2 minutes on error

if __name__ == "__main__":
    print("🌍 FIXED OKX ALL TOKENS ARBITRAGE SCANNER")
    print("=" * 60)
    print("🎯 Monitors EVERY token on OKX for arbitrage opportunities")
    print("⚡ Real-time spread detection across all trading pairs")
    print("📊 Smart categorization by token type")
    print("🔍 Filters for realistic opportunities only")
    print("📱 Discord alerts for best opportunities")
    print("")
    
    scanner = FixedOKXAllTokensScanner()
    
    try:
        asyncio.run(scanner.continuous_monitoring())
    except KeyboardInterrupt:
        print(f"\n🛑 Scanner stopped by user")
        print(f"📊 Final Statistics:")
        print(f"   Total scans completed: {scanner.scan_count}")
        print(f"   Total tokens scanned: {scanner.total_scanned:,}")
        print(f"   Total opportunities found: {scanner.opportunities_found:,}")
        if scanner.best_opportunity:
            best = scanner.best_opportunity
            print(f"   Best opportunity: {best['symbol']} ({best['spread_percent']:.4f}%)")
        print(f"✅ Scanner shutdown complete")
    except Exception as e:
        print(f"❌ Scanner error: {e}")