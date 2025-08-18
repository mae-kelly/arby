#!/usr/bin/env python3
"""
OKX ALL TOKENS ARBITRAGE SCANNER
Monitors EVERY single trading pair on OKX for arbitrage opportunities
"""

import asyncio
import aiohttp
import hmac
import hashlib
import base64
import time
import json
from datetime import datetime
import requests

class OKXAllTokensScanner:
    def __init__(self):
        self.api_key = "8a760df1-4a2d-471b-ba42-d16893614dab"
        self.secret_key = "C9F3FC89A6A30226E11DFFD098C7CF3D"
        self.passphrase = "Shamrock1!"
        self.base_url = "https://www.okx.com"
        self.webhook = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3"
        
        # Trading pairs cache
        self.all_pairs = []
        self.last_update = 0
        self.update_interval = 3600  # Update pairs list every hour
        
        # Arbitrage settings
        self.min_spread_threshold = 0.01  # 0.01% minimum
        self.min_volume_24h = 1000  # $1000 minimum 24h volume
        self.max_pairs_per_batch = 50  # Process in batches to avoid rate limits
        
        # Statistics
        self.total_scanned = 0
        self.opportunities_found = 0
        self.best_opportunity = None
        
    def sign(self, timestamp, method, request_path, body=""):
        """Sign OKX API request"""
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()
    
    async def get_all_trading_pairs(self):
        """Get ALL trading pairs available on OKX"""
        print("🔍 Fetching ALL trading pairs from OKX...")
        
        timestamp = str(int(time.time() * 1000))
        method = "GET"
        request_path = "/api/v5/public/instruments?instType=SPOT"
        
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
                    if response.status == 200:
                        data = await response.json()
                        if data.get("code") == "0":
                            instruments = data.get("data", [])
                            
                            # Filter for active spot trading pairs
                            active_pairs = []
                            for inst in instruments:
                                if inst.get("state") == "live":
                                    pair_id = inst.get("instId")
                                    if pair_id and "-" in pair_id:
                                        active_pairs.append({
                                            "symbol": pair_id,
                                            "baseCcy": inst.get("baseCcy"),
                                            "quoteCcy": inst.get("quoteCcy"),
                                            "minSz": inst.get("minSz"),
                                            "lotSz": inst.get("lotSz")
                                        })
                            
                            print(f"✅ Found {len(active_pairs)} active trading pairs!")
                            return active_pairs
                        else:
                            print(f"❌ API error: {data.get('msg')}")
                            return []
                    else:
                        print(f"❌ HTTP error: {response.status}")
                        return []
        except Exception as e:
            print(f"❌ Error fetching pairs: {e}")
            return []
    
    async def get_ticker_data(self, session, symbol):
        """Get ticker data for a specific symbol"""
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
            async with session.get(f"{self.base_url}{request_path}", headers=headers, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("code") == "0" and data.get("data"):
                        return data["data"][0]
                return None
        except:
            return None
    
    async def get_all_tickers_batch(self):
        """Get ticker data for ALL symbols using batch endpoint"""
        print("📊 Fetching market data for ALL tokens...")
        
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
                            print(f"✅ Retrieved market data for {len(tickers)} tokens!")
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
    
    def analyze_arbitrage_opportunity(self, ticker):
        """Analyze a single ticker for arbitrage opportunity"""
        try:
            symbol = ticker.get("instId")
            last = float(ticker.get("last", 0))
            bid = float(ticker.get("bidPx", 0))
            ask = float(ticker.get("askPx", 0))
            vol24h = float(ticker.get("vol24h", 0))
            volCcy24h = float(ticker.get("volCcy24h", 0))  # Volume in quote currency
            
            if bid > 0 and ask > 0 and last > 0:
                # Calculate spread percentage
                spread = ((ask - bid) / bid) * 100
                
                # Calculate 24h volume in USD (approximate)
                volume_24h_usd = volCcy24h if "USDT" in symbol or "USDC" in symbol else vol24h * last
                
                # Check if meets minimum criteria
                if spread >= self.min_spread_threshold and volume_24h_usd >= self.min_volume_24h:
                    return {
                        "symbol": symbol,
                        "price": last,
                        "bid": bid,
                        "ask": ask,
                        "spread_percent": spread,
                        "volume_24h_usd": volume_24h_usd,
                        "potential_profit_1k": spread * 10,  # Profit on $1000 trade
                        "timestamp": datetime.now().isoformat()
                    }
            return None
        except Exception as e:
            return None
    
    async def send_alert(self, message):
        """Send Discord alert"""
        try:
            payload = {"content": f"🌍 OKX ALL TOKENS: {message}"}
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook, json=payload, timeout=10) as response:
                    if response.status == 204:
                        print(f"📱 Alert sent: {message[:100]}...")
        except Exception as e:
            print(f"❌ Alert error: {e}")
    
    async def scan_all_tokens(self):
        """Scan ALL tokens for arbitrage opportunities"""
        print(f"\n{'='*80}")
        print(f"🌍 SCANNING ALL OKX TOKENS FOR ARBITRAGE")
        print(f"{'='*80}")
        
        # Get all ticker data at once (more efficient)
        all_tickers = await self.get_all_tickers_batch()
        
        if not all_tickers:
            print("❌ Failed to get ticker data")
            return []
        
        # Analyze each ticker for arbitrage opportunities
        opportunities = []
        processed = 0
        
        print(f"🔍 Analyzing {len(all_tickers)} tokens for arbitrage opportunities...")
        
        for ticker in all_tickers:
            opportunity = self.analyze_arbitrage_opportunity(ticker)
            if opportunity:
                opportunities.append(opportunity)
            
            processed += 1
            if processed % 100 == 0:
                print(f"   Processed {processed}/{len(all_tickers)} tokens...")
        
        self.total_scanned = len(all_tickers)
        self.opportunities_found = len(opportunities)
        
        # Sort by spread percentage (best first)
        opportunities.sort(key=lambda x: x["spread_percent"], reverse=True)
        
        return opportunities
    
    def display_results(self, opportunities):
        """Display scan results"""
        print(f"\n📊 SCAN RESULTS:")
        print(f"   Total tokens scanned: {self.total_scanned:,}")
        print(f"   Arbitrage opportunities found: {len(opportunities):,}")
        print(f"   Success rate: {(len(opportunities)/self.total_scanned)*100:.2f}%")
        
        if opportunities:
            print(f"\n🏆 TOP 10 ARBITRAGE OPPORTUNITIES:")
            print(f"{'Symbol':<15} {'Price':<12} {'Spread':<8} {'Volume 24h':<12} {'Profit/1k':<10}")
            print("-" * 70)
            
            for i, opp in enumerate(opportunities[:10], 1):
                print(f"{opp['symbol']:<15} ${opp['price']:<11.4f} {opp['spread_percent']:<7.4f}% "
                      f"${opp['volume_24h_usd']:<11,.0f} ${opp['potential_profit_1k']:<9.2f}")
            
            # Update best opportunity
            if opportunities:
                self.best_opportunity = opportunities[0]
                
                # Send alert for best opportunities
                best = opportunities[0]
                if best["spread_percent"] > 0.1:  # > 0.1% spread
                    await self.send_alert(
                        f"🚨 HIGH SPREAD DETECTED!\n"
                        f"Token: {best['symbol']}\n"
                        f"Spread: {best['spread_percent']:.4f}%\n"
                        f"Price: ${best['price']:,.4f}\n"
                        f"24h Volume: ${best['volume_24h_usd']:,.0f}\n"
                        f"Potential profit on $1000: ${best['potential_profit_1k']:.2f}"
                    )
        else:
            print(f"\n⚪ No arbitrage opportunities found meeting criteria:")
            print(f"   • Minimum spread: {self.min_spread_threshold}%")
            print(f"   • Minimum 24h volume: ${self.min_volume_24h:,}")
    
    async def continuous_monitoring(self):
        """Continuously monitor ALL tokens"""
        await self.send_alert("🚀 Started monitoring ALL OKX tokens for arbitrage opportunities!")
        
        cycle = 0
        total_opportunities = 0
        
        while True:
            try:
                start_time = time.time()
                
                print(f"\n{'='*100}")
                print(f"🌍 ALL TOKENS SCAN #{cycle} - {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*100}")
                
                # Scan all tokens
                opportunities = await self.scan_all_tokens()
                
                # Display results
                self.display_results(opportunities)
                
                # Update statistics
                total_opportunities += len(opportunities)
                scan_time = time.time() - start_time
                
                print(f"\n📈 SESSION STATISTICS:")
                print(f"   Scan #{cycle + 1} completed in {scan_time:.1f} seconds")
                print(f"   Total opportunities found today: {total_opportunities}")
                print(f"   Average opportunities per scan: {total_opportunities/(cycle+1):.1f}")
                
                if self.best_opportunity:
                    best = self.best_opportunity
                    print(f"   Best opportunity today: {best['symbol']} ({best['spread_percent']:.4f}%)")
                
                # Send periodic summary
                if cycle % 10 == 0 and cycle > 0:
                    await self.send_alert(
                        f"📊 Scan Summary (10 cycles):\n"
                        f"Total tokens monitored: {self.total_scanned:,}\n"
                        f"Opportunities found: {total_opportunities}\n"
                        f"Best spread: {self.best_opportunity['spread_percent']:.4f}% "
                        f"({self.best_opportunity['symbol']})" if self.best_opportunity else "No opportunities"
                    )
                
                cycle += 1
                
                # Wait before next scan (avoid rate limits)
                wait_time = 60  # 1 minute between full scans
                print(f"\n⏱️  Next ALL TOKENS scan in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                print(f"❌ Scan error: {e}")
                await self.send_alert(f"⚠️ Scanner error: {str(e)[:100]}")
                await asyncio.sleep(120)  # Wait 2 minutes on error

if __name__ == "__main__":
    print("🌍 OKX ALL TOKENS ARBITRAGE SCANNER")
    print("=" * 50)
    print("🎯 Monitoring EVERY token on OKX for arbitrage opportunities")
    print("⚡ Real-time spread detection across all trading pairs")
    print("📊 Comprehensive market analysis")
    print("🔍 Smart filtering for realistic opportunities")
    print("")
    
    scanner = OKXAllTokensScanner()
    
    try:
        asyncio.run(scanner.continuous_monitoring())
    except KeyboardInterrupt:
        print(f"\n🛑 Scanner stopped by user")
        print(f"📊 Final stats:")
        print(f"   Total tokens scanned: {scanner.total_scanned:,}")
        print(f"   Opportunities found: {scanner.opportunities_found:,}")
        if scanner.best_opportunity:
            best = scanner.best_opportunity
            print(f"   Best opportunity: {best['symbol']} ({best['spread_percent']:.4f}%)")
    except Exception as e:
        print(f"❌ Scanner error: {e}")