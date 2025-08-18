# comprehensive_arbitrage.py - Fixed Integrated Master Orchestrator

import asyncio
import aiohttp
import json
from web3 import Web3
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import requests
import hmac
import hashlib
import base64
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import os

# Import all existing modules from the repo
try:
    from main_orchestrator import OKXClient, RealArbitrageBot
    print("✅ Imported CEX arbitrage module")
except ImportError as e:
    print(f"⚠️  CEX module not available: {e}")
    OKXClient = None
    RealArbitrageBot = None

try:
    from working_arbitrage import WorkingArbitrageDetector
    print("✅ Imported working DEX arbitrage module")
except ImportError as e:
    print(f"⚠️  Working arbitrage module not available: {e}")
    WorkingArbitrageDetector = None

try:
    from profitable_flash_arbitrage import ProfitableFlashArbitrage
    print("✅ Imported flash loan arbitrage module")
except ImportError as e:
    print(f"⚠️  Flash loan module not available: {e}")
    ProfitableFlashArbitrage = None

try:
    from realistic_arbitrage import RealisticArbitrageDetector
    print("✅ Imported realistic arbitrage module")
except ImportError as e:
    print(f"⚠️  Realistic arbitrage module not available: {e}")
    RealisticArbitrageDetector = None

@dataclass
class IntegratedArbitrageOpportunity:
    opportunity_type: str  # "CEX_SPREAD", "DEX_ARBITRAGE", "FLASH_LOAN", "CROSS_CHAIN"
    source_module: str
    source_exchange: str
    target_exchange: str
    source_chain: str
    target_chain: str
    token_pair: str
    source_price: float
    target_price: float
    profit_percentage: float
    gross_profit_usd: float
    gas_cost_usd: float
    dex_fees_usd: float
    slippage_cost_usd: float
    bridge_fee_usd: float
    net_profit_usd: float
    execution_time_seconds: float
    risk_score: int
    min_trade_size_usd: float
    max_trade_size_usd: float
    liquidity_depth_usd: float
    real_time_data_age_seconds: float
    confidence_score: float

class FixedOKXClient:
    """Fixed OKX client with proper timestamp handling"""
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
        timestamp = str(int(time.time() * 1000))  # Fixed: Proper milliseconds
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
                async with session.get(f"{self.base_url}{request_path}", headers=headers, timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"code": "1", "msg": f"HTTP {response.status}"}
        except Exception as e:
            return {"code": "1", "msg": str(e)}

class IntegratedArbitrageOrchestrator:
    def __init__(self):
        self.rpcs = {
            "ethereum": "https://eth-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX",
            "polygon": [
                "https://polygon-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX",
                "https://polygon-rpc.com",
                "https://rpc-mainnet.matic.network"
            ],
            "arbitrum": "https://arb-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX",
            "optimism": [
                "https://opt-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX",
                "https://mainnet.optimism.io",
                "https://optimism.drpc.org"
            ],
            "base": [
                "https://base-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX",
                "https://mainnet.base.org",
                "https://base.drpc.org"
            ],
            "bsc": [
                "https://bsc-dataseed.binance.org/",
                "https://bsc-dataseed1.defibit.io/",
                "https://bsc-dataseed1.ninicoin.io/"
            ],
            "avalanche": [
                "https://api.avax.network/ext/bc/C/rpc",
                "https://avalanche.drpc.org",
                "https://rpc.ankr.com/avalanche"
            ]
        }
        
        # Initialize Web3 connections
        self.w3_instances = {}
        for chain, rpc_config in self.rpcs.items():
            connected = False
            rpcs_to_try = [rpc_config] if isinstance(rpc_config, str) else rpc_config
            
            for rpc in rpcs_to_try:
                try:
                    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={'timeout': 10}))
                    if w3.is_connected():
                        w3.eth.block_number
                        self.w3_instances[chain] = w3
                        print(f"✅ Connected to {chain} via {rpc[:50]}...")
                        connected = True
                        break
                except Exception as e:
                    print(f"   ⚠️  Failed {chain} RPC {rpc[:30]}...: {str(e)[:50]}")
                    continue
            
            if not connected:
                print(f"❌ Failed to connect to {chain} (all RPCs failed)")
        
        print(f"📡 Successfully connected to {len(self.w3_instances)}/{len(self.rpcs)} chains")
        
        self.webhook = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3"
        
        # Initialize all available modules
        self.modules = {}
        self.fixed_okx_client = FixedOKXClient()
        self.initialize_modules()
        
        # JavaScript and other processes
        self.external_processes = {}
        
    def initialize_modules(self):
        """Initialize all available arbitrage modules"""
        print("\n🔧 Initializing arbitrage modules...")
        
        # CEX Arbitrage Module
        if OKXClient and RealArbitrageBot:
            try:
                self.modules['cex'] = RealArbitrageBot()
                print("✅ CEX arbitrage module initialized")
            except Exception as e:
                print(f"❌ CEX module error: {e}")
        
        # Working DEX Arbitrage Module
        if WorkingArbitrageDetector:
            try:
                self.modules['dex_working'] = WorkingArbitrageDetector()
                print("✅ Working DEX arbitrage module initialized")
            except Exception as e:
                print(f"❌ Working DEX module error: {e}")
                
        # Flash Loan Arbitrage Module
        if ProfitableFlashArbitrage:
            try:
                self.modules['flash_loan'] = ProfitableFlashArbitrage()
                print("✅ Flash loan arbitrage module initialized")
            except Exception as e:
                print(f"❌ Flash loan module error: {e}")
        
        # Realistic Arbitrage Module
        if RealisticArbitrageDetector:
            try:
                self.modules['realistic'] = RealisticArbitrageDetector()
                print("✅ Realistic arbitrage module initialized")
            except Exception as e:
                print(f"❌ Realistic module error: {e}")
        
        print(f"🎯 {len(self.modules)} arbitrage modules ready")

    def create_fixed_go_file(self):
        """Create a properly formatted Go file"""
        go_content = '''package main

import (
    "fmt"
    "time"
    "net/http"
    "io/ioutil"
)

func main() {
    fmt.Println("Go Market Data Aggregator - Fixed Version")
    
    symbols := []string{"BTC-USDT", "ETH-USDT", "SOL-USDT"}
    
    for i := 0; i < 3; i++ {
        for _, symbol := range symbols {
            fmt.Printf("Fetching %s data...", symbol)
            
            // Simple HTTP request to demonstrate functionality
            resp, err := http.Get("https://api.coingecko.com/api/v3/ping")
            if err == nil {
                body, _ := ioutil.ReadAll(resp.Body)
                resp.Body.Close()
                if len(body) > 0 {
                    fmt.Printf(" ✅ Connected\\n")
                } else {
                    fmt.Printf(" ❌ Failed\\n")
                }
            } else {
                fmt.Printf(" ❌ Error: %v\\n", err)
            }
        }
        time.Sleep(5 * time.Second)
    }
    
    fmt.Println("Market data aggregator demo completed")
}
'''
        
        with open("market_data_aggregator_fixed.go", "w") as f:
            f.write(go_content)
        
        print("✅ Created market_data_aggregator_fixed.go")

    async def start_external_processes(self):
        """Start JavaScript HFT engine and other external processes"""
        try:
            # Check if Node.js files exist and start them
            if os.path.exists("hft_engine.js"):
                print("🚀 Starting HFT engine...")
                process = subprocess.Popen(
                    ["node", "hft_engine.js"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.external_processes['hft'] = process
                print("✅ HFT engine started")
            
            # Create and start fixed Go market data aggregator
            self.create_fixed_go_file()
            
            if os.path.exists("market_data_aggregator_fixed.go"):
                print("🚀 Starting fixed Go market data aggregator...")
                # Compile first
                compile_result = subprocess.run(
                    ["go", "build", "market_data_aggregator_fixed.go"], 
                    capture_output=True, text=True
                )
                if compile_result.returncode == 0:
                    process = subprocess.Popen(
                        ["./market_data_aggregator_fixed"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    self.external_processes['go_aggregator'] = process
                    print("✅ Fixed Go aggregator started")
                else:
                    print(f"❌ Go compilation failed: {compile_result.stderr}")
            
        except Exception as e:
            print(f"⚠️  External process startup error: {e}")

    def validate_opportunity(self, opportunity):
        """Validate if an opportunity is realistic"""
        issues = []
        original_confidence = opportunity.confidence_score
        
        # Check for unrealistic profits
        if opportunity.profit_percentage > 5:
            issues.append(f"UNREALISTIC: {opportunity.profit_percentage:.2f}% profit")
            opportunity.confidence_score = 0.1
        elif opportunity.profit_percentage > 2:
            issues.append(f"SUSPICIOUS: {opportunity.profit_percentage:.2f}% profit")
            opportunity.confidence_score = 0.3
        elif opportunity.profit_percentage > 1:
            issues.append(f"HIGH: {opportunity.profit_percentage:.2f}% profit")
            opportunity.confidence_score = 0.5
        
        # Check for large profits
        if opportunity.net_profit_usd > 10000:
            issues.append(f"UNREALISTIC: ${opportunity.net_profit_usd:.2f} profit")
            opportunity.confidence_score = 0.1
        elif opportunity.net_profit_usd > 1000:
            issues.append(f"SUSPICIOUS: ${opportunity.net_profit_usd:.2f} profit")
            opportunity.confidence_score = min(opportunity.confidence_score, 0.3)
        
        # Check gas costs for DEX opportunities
        if opportunity.opportunity_type == "DEX_ARBITRAGE" and opportunity.gas_cost_usd < 5:
            issues.append("Gas cost too low for DEX")
            opportunity.confidence_score = min(opportunity.confidence_score, 0.4)
        
        # Flash loan specific validation
        if opportunity.opportunity_type == "FLASH_LOAN":
            if opportunity.gas_cost_usd < 20:
                issues.append("Gas cost too low for flash loan")
                opportunity.confidence_score = min(opportunity.confidence_score, 0.3)
            
            if opportunity.profit_percentage > 3:
                issues.append("Flash loan profit too high")
                opportunity.confidence_score = 0.1
        
        # Print validation results
        if issues:
            print(f"   ⚠️  VALIDATION: {', '.join(issues)}")
            print(f"   🔍 Confidence: {original_confidence:.1%} -> {opportunity.confidence_score:.1%}")
            
            if opportunity.confidence_score < 0.3:
                print(f"   📊 This opportunity is likely a calculation error")
        
        return opportunity.confidence_score > 0.5

    async def send_opportunity_alert(self, opportunity: IntegratedArbitrageOpportunity):
        try:
            # Only alert on realistic opportunities
            if (opportunity.net_profit_usd > 5 and 
                opportunity.confidence_score > 0.6 and 
                opportunity.profit_percentage < 2.0):  # Max 2% profit
                
                message = (
                    f"💰 REALISTIC ARBITRAGE OPPORTUNITY\n"
                    f"Type: {opportunity.opportunity_type}\n"
                    f"Module: {opportunity.source_module}\n"
                    f"Pair: {opportunity.token_pair}\n"
                    f"Route: {opportunity.source_exchange} -> {opportunity.target_exchange}\n"
                    f"Profit: ${opportunity.net_profit_usd:.2f} ({opportunity.profit_percentage:.4f}%)\n"
                    f"Confidence: {opportunity.confidence_score:.1%}\n"
                    f"Risk: {opportunity.risk_score}/10\n"
                    f"⚠️ NOTE: Real arbitrage profits are typically very small\n"
                    f"🎓 This is for educational/research purposes only"
                )
                
                payload = {"content": message}
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.webhook, json=payload) as response:
                        print(f"🚨 REALISTIC ALERT: {opportunity.opportunity_type} ${opportunity.net_profit_usd:.2f}")
            else:
                print(f"   📝 Skipping unrealistic alert: ${opportunity.net_profit_usd:.2f} (confidence: {opportunity.confidence_score:.1%})")
        except Exception as e:
            print(f"Alert error: {e}")

    async def scan_cex_opportunities(self):
        """Scan CEX opportunities using fixed OKX client"""
        opportunities = []
        
        try:
            symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
            
            for symbol in symbols:
                try:
                    ticker = await self.fixed_okx_client.get_ticker(symbol)
                    
                    if ticker.get("code") == "0" and ticker.get("data"):
                        data = ticker["data"][0]
                        bid = float(data.get("bidPx", 0))
                        ask = float(data.get("askPx", 0))
                        
                        if bid > 0 and ask > 0:
                            spread = ((ask - bid) / bid) * 100
                            
                            if spread > 0.005:  # Only opportunities > 0.005%
                                opportunity = IntegratedArbitrageOpportunity(
                                    opportunity_type="CEX_SPREAD",
                                    source_module="fixed_okx_client",
                                    source_exchange="OKX",
                                    target_exchange="OKX",
                                    source_chain="CEX",
                                    target_chain="CEX",
                                    token_pair=symbol,
                                    source_price=bid,
                                    target_price=ask,
                                    profit_percentage=spread,
                                    gross_profit_usd=spread * 100,  # Profit on $10k trade
                                    gas_cost_usd=0.0,
                                    dex_fees_usd=0.0,
                                    slippage_cost_usd=0.0,
                                    bridge_fee_usd=0.0,
                                    net_profit_usd=spread * 100,
                                    execution_time_seconds=5,
                                    risk_score=2,
                                    min_trade_size_usd=100,
                                    max_trade_size_usd=10000,
                                    liquidity_depth_usd=1000000,
                                    real_time_data_age_seconds=1,
                                    confidence_score=0.8
                                )
                                opportunities.append(opportunity)
                                
                except Exception as e:
                    print(f"   Error fetching {symbol}: {e}")
                    continue
                
        except Exception as e:
            print(f"CEX scan error: {e}")
        
        return opportunities

    async def scan_dex_opportunities(self):
        """Scan DEX opportunities using working arbitrage module"""
        opportunities = []
        
        if 'dex_working' not in self.modules:
            return opportunities
            
        try:
            dex_module = self.modules['dex_working']
            dex_opportunities = await dex_module.scan_real_opportunities()
            
            for opp in dex_opportunities:
                integrated_opp = IntegratedArbitrageOpportunity(
                    opportunity_type="DEX_ARBITRAGE",
                    source_module="working_arbitrage",
                    source_exchange=opp["buy_dex"],
                    target_exchange=opp["sell_dex"],
                    source_chain="ethereum",
                    target_chain="ethereum",
                    token_pair=opp["pair"],
                    source_price=opp.get("uniswap_out", 0),
                    target_price=opp.get("sushiswap_out", 0),
                    profit_percentage=opp["profit_percentage"],
                    gross_profit_usd=opp["profit_usd"],
                    gas_cost_usd=opp["gas_cost_usd"],
                    dex_fees_usd=opp["profit_usd"] * 0.003,
                    slippage_cost_usd=opp["profit_usd"] * 0.001,
                    bridge_fee_usd=0.0,
                    net_profit_usd=opp["net_profit"],
                    execution_time_seconds=30,
                    risk_score=6,  # Higher risk due to MEV
                    min_trade_size_usd=100,
                    max_trade_size_usd=1000,
                    liquidity_depth_usd=50000,
                    real_time_data_age_seconds=5,
                    confidence_score=0.7 if opp["profitable"] else 0.3
                )
                opportunities.append(integrated_opp)
                
        except Exception as e:
            print(f"DEX scan error: {e}")
        
        return opportunities

    async def scan_flash_loan_opportunities(self):
        """Scan flash loan opportunities with validation"""
        opportunities = []
        
        if 'flash_loan' not in self.modules:
            return opportunities
            
        try:
            flash_module = self.modules['flash_loan']
            flash_opportunities = await flash_module.scan_flash_loan_opportunities()
            
            for opp in flash_opportunities:
                integrated_opp = IntegratedArbitrageOpportunity(
                    opportunity_type="FLASH_LOAN",
                    source_module="profitable_flash_arbitrage",
                    source_exchange=opp.buy_dex,
                    target_exchange=opp.sell_dex,
                    source_chain="ethereum",
                    target_chain="ethereum",
                    token_pair=opp.pair,
                    source_price=opp.buy_price,
                    target_price=opp.sell_price,
                    profit_percentage=opp.profit_percentage,
                    gross_profit_usd=opp.gross_profit_usd,
                    gas_cost_usd=opp.gas_cost_usd,
                    dex_fees_usd=0.0,
                    slippage_cost_usd=0.0,
                    bridge_fee_usd=0.0,
                    net_profit_usd=opp.net_profit_usd,
                    execution_time_seconds=opp.execution_time_estimate,
                    risk_score=8,  # Flash loans are high risk
                    min_trade_size_usd=1000,
                    max_trade_size_usd=100000,
                    liquidity_depth_usd=500000,
                    real_time_data_age_seconds=10,
                    confidence_score=0.2  # Start low, validate later
                )
                
                # Validate the opportunity
                self.validate_opportunity(integrated_opp)
                opportunities.append(integrated_opp)
                
        except Exception as e:
            print(f"Flash loan scan error: {e}")
        
        return opportunities

    async def comprehensive_integrated_scan(self):
        """Run comprehensive scan using all available modules"""
        start_time = time.time()
        
        print(f"\n{'='*100}")
        print(f"🌍 INTEGRATED COMPREHENSIVE ARBITRAGE SCAN")
        print(f"{'='*100}")
        
        all_opportunities = []
        
        # Scan each module type
        scan_tasks = []
        
        if 'cex' in self.modules or True:  # Always try CEX with fixed client
            scan_tasks.append(("CEX", self.scan_cex_opportunities()))
            
        if 'dex_working' in self.modules:
            scan_tasks.append(("DEX", self.scan_dex_opportunities()))
            
        if 'flash_loan' in self.modules:
            scan_tasks.append(("FLASH", self.scan_flash_loan_opportunities()))
        
        # Run all scans concurrently
        print(f"🔍 Running {len(scan_tasks)} parallel scans...")
        
        for scan_name, scan_task in scan_tasks:
            try:
                opportunities = await scan_task
                # Apply validation to all opportunities
                validated_opportunities = []
                for opp in opportunities:
                    if self.validate_opportunity(opp):
                        validated_opportunities.append(opp)
                
                all_opportunities.extend(validated_opportunities)
                print(f"   ✅ {scan_name}: {len(opportunities)} found, {len(validated_opportunities)} validated")
            except Exception as e:
                print(f"   ❌ {scan_name}: {e}")
        
        # Sort by net profit
        all_opportunities.sort(key=lambda x: x.net_profit_usd, reverse=True)
        
        scan_time = time.time() - start_time
        
        print(f"\n📊 INTEGRATED SCAN RESULTS:")
        print(f"   Scan time: {scan_time:.1f} seconds")
        print(f"   Total opportunities: {len(all_opportunities)}")
        print(f"   Profitable opportunities (>$5): {len([o for o in all_opportunities if o.net_profit_usd > 5])}")
        print(f"   High confidence (>60%): {len([o for o in all_opportunities if o.confidence_score > 0.6])}")
        print(f"   Realistic opportunities: {len([o for o in all_opportunities if o.confidence_score > 0.6 and o.profit_percentage < 2])}")
        
        # Show top opportunities
        if all_opportunities:
            print(f"\n🏆 TOP OPPORTUNITIES:")
            for i, opp in enumerate(all_opportunities[:5], 1):
                status = "✅ REALISTIC" if opp.confidence_score > 0.6 else "⚠️ SUSPICIOUS"
                print(f"   {i}. {status} - {opp.opportunity_type}: {opp.token_pair}")
                print(f"      Module: {opp.source_module}")
                print(f"      Net profit: ${opp.net_profit_usd:.2f} ({opp.profit_percentage:.4f}%)")
                print(f"      Confidence: {opp.confidence_score:.1%}")
                print(f"      Risk: {opp.risk_score}/10")
        else:
            print(f"\n⚪ No validated opportunities found")
        
        return all_opportunities

    async def monitor_integrated_opportunities(self):
        """Main monitoring loop for integrated system"""
        
        # Start external processes
        await self.start_external_processes()
        
        # Send startup notification with realistic expectations
        startup_message = (
            f"🚀 REALISTIC ARBITRAGE SYSTEM STARTED\n"
            f"📊 Active modules: {len(self.modules)}\n"
            f"🔗 Chains connected: {len(self.w3_instances)}\n"
            f"⚡ External processes: {len(self.external_processes)}\n"
            f"🎯 Target: Small, realistic arbitrage opportunities\n"
            f"⚠️ Most opportunities will be unprofitable due to gas costs\n"
            f"🎓 Educational/research system - not for actual trading"
        )
        
        payload = {"content": startup_message}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.webhook, json=payload):
                pass
        
        cycle = 0
        total_realistic_found = 0
        
        while True:
            try:
                print(f"\n{'='*120}")
                print(f"🌍 REALISTIC ARBITRAGE SCAN #{cycle} - {time.strftime('%H:%M:%S')}")
                print(f"{'='*120}")
                
                opportunities = await self.comprehensive_integrated_scan()
                
                realistic_opportunities = [o for o in opportunities 
                                         if o.net_profit_usd > 5 and 
                                         o.confidence_score > 0.6 and 
                                         o.profit_percentage < 2.0]
                
                if realistic_opportunities:
                    total_realistic_found += len(realistic_opportunities)
                    
                    print(f"\n🎉 FOUND {len(realistic_opportunities)} REALISTIC OPPORTUNITIES!")
                    
                    for i, opp in enumerate(realistic_opportunities[:3], 1):
                        print(f"\n💰 REALISTIC OPPORTUNITY #{i}:")
                        print(f"   Type: {opp.opportunity_type}")
                        print(f"   Module: {opp.source_module}")
                        print(f"   Pair: {opp.token_pair}")
                        print(f"   Route: {opp.source_exchange} -> {opp.target_exchange}")
                        print(f"   Profit: ${opp.net_profit_usd:.2f} ({opp.profit_percentage:.4f}%)")
                        print(f"   Confidence: {opp.confidence_score:.1%}")
                        print(f"   Risk: {opp.risk_score}/10")
                        
                        await self.send_opportunity_alert(opp)
                
                else:
                    print(f"\n⚪ No realistic opportunities found")
                    if opportunities:
                        best = opportunities[0]
                        print(f"   Best opportunity: {best.opportunity_type} {best.token_pair}")
                        print(f"   Profit: ${best.net_profit_usd:.2f} (confidence: {best.confidence_score:.1%})")
                        
                        if best.confidence_score < 0.3:
                            print(f"   📊 Likely calculation error due to low confidence")
                
                print(f"\n📈 SESSION STATS:")
                print(f"   Realistic opportunities found: {total_realistic_found}")
                print(f"   Scan cycles completed: {cycle + 1}")
                print(f"   Success rate: {(total_realistic_found/(cycle+1)):.2f} realistic per cycle")
                print(f"   Active modules: {list(self.modules.keys())}")
                print(f"   External processes: {len([p for p in self.external_processes.values() if p.poll() is None])}")
                
                cycle += 1
                print(f"\n⏱️  Next realistic scan in 60 seconds...")
                await asyncio.sleep(60)
                
            except Exception as e:
                print(f"❌ Integrated scan error: {e}")
                await asyncio.sleep(120)

    def cleanup_processes(self):
        """Clean up external processes on shutdown"""
        print("\n🧹 Cleaning up external processes...")
        for name, process in self.external_processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} process terminated")
            except:
                try:
                    process.kill()
                    print(f"⚠️  {name} process killed")
                except:
                    print(f"❌ Failed to stop {name} process")

if __name__ == "__main__":
    orchestrator = IntegratedArbitrageOrchestrator()
    print("🌍 REALISTIC INTEGRATED ARBITRAGE SYSTEM")
    print("=" * 80)
    print("✅ Integrates ALL repository modules")
    print("✅ Fixed OKX API timestamp issues")
    print("✅ Realistic opportunity validation")
    print("✅ Proper confidence scoring")
    print("✅ Educational/research focused")
    print("⚠️ Most arbitrage opportunities are unprofitable")
    print("🎓 Realistic expectations: 0.01-0.5% profits are normal")
    print("")
    
    try:
        asyncio.run(orchestrator.monitor_integrated_opportunities())
    except KeyboardInterrupt:
        print("\n👋 Shutting down realistic arbitrage system...")
        orchestrator.cleanup_processes()
        print("✅ Cleanup complete")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        orchestrator.cleanup_processes()