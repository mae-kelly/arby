import asyncio
import json
from typing import Dict, Optional
import aiohttp
from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

class DiscordNotifier:
    def __init__(self):
        self.webhook_url = settings.discord_webhook
        self.session = None

    async def initialize(self):
        self.session = aiohttp.ClientSession()

    async def send_message(self, content: str, embeds: Optional[list] = None):
        if not self.webhook_url:
            return False
        
        try:
            payload = {"content": content}
            
            if embeds:
                payload["embeds"] = embeds
            
            async with self.session.post(self.webhook_url, json=payload) as response:
                return response.status == 204
                
        except Exception as e:
            logger.error(f"Discord notification failed: {e}")
            return False

    async def send_trade_alert(self, strategy: str, symbol: str, profit: float, 
                              success: bool, details: Dict):
        color = 0x00ff00 if success else 0xff0000
        status = "✅ SUCCESS" if success else "❌ FAILED"
        
        embed = {
            "title": f"{status} - {strategy.upper()} Arbitrage",
            "color": color,
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Profit", "value": f"${profit:.2f}", "inline": True},
                {"name": "Strategy", "value": strategy, "inline": True}
            ],
            "timestamp": details.get("timestamp", ""),
            "footer": {"text": "Arbitrage Bot"}
        }
        
        if details.get("gas_cost"):
            embed["fields"].append({
                "name": "Gas Cost", 
                "value": f"${details['gas_cost']:.2f}", 
                "inline": True
            })
        
        if details.get("execution_time"):
            embed["fields"].append({
                "name": "Execution Time", 
                "value": f"{details['execution_time']:.2f}s", 
                "inline": True
            })
        
        await self.send_message("", [embed])

    async def send_risk_alert(self, alert_type: str, message: str, severity: str = "warning"):
        colors = {
            "info": 0x3498db,
            "warning": 0xf39c12,
            "error": 0xe74c3c,
            "critical": 0x8b0000
        }
        
        embed = {
            "title": f"🚨 Risk Alert - {alert_type.upper()}",
            "description": message,
            "color": colors.get(severity, 0xf39c12),
            "timestamp": "",
            "footer": {"text": "Risk Management System"}
        }
        
        await self.send_message("", [embed])

    async def send_system_alert(self, component: str, status: str, details: str = ""):
        color = 0x00ff00 if status == "healthy" else 0xff0000
        
        embed = {
            "title": f"⚙️ System Status - {component.upper()}",
            "description": f"Status: {status.upper()}",
            "color": color,
            "fields": [
                {"name": "Details", "value": details or "No additional details", "inline": False}
            ],
            "timestamp": "",
            "footer": {"text": "System Monitor"}
        }
        
        await self.send_message("", [embed])

    async def send_daily_summary(self, summary: Dict):
        total_profit = summary.get("total_profit", 0)
        total_trades = summary.get("total_trades", 0)
        success_rate = summary.get("success_rate", 0)
        
        color = 0x00ff00 if total_profit > 0 else 0xff0000
        
        embed = {
            "title": "📊 Daily Trading Summary",
            "color": color,
            "fields": [
                {"name": "Total Profit", "value": f"${total_profit:.2f}", "inline": True},
                {"name": "Total Trades", "value": str(total_trades), "inline": True},
                {"name": "Success Rate", "value": f"{success_rate:.1f}%", "inline": True},
                {"name": "Best Strategy", "value": summary.get("best_strategy", "N/A"), "inline": True},
                {"name": "Total Gas Cost", "value": f"${summary.get('total_gas_cost', 0):.2f}", "inline": True},
                {"name": "Net Profit", "value": f"${summary.get('net_profit', 0):.2f}", "inline": True}
            ],
            "timestamp": "",
            "footer": {"text": "Daily Report"}
        }
        
        await self.send_message("", [embed])

    async def send_opportunity_alert(self, opportunity: Dict):
        embed = {
            "title": "🎯 New Arbitrage Opportunity",
            "color": 0x3498db,
            "fields": [
                {"name": "Strategy", "value": opportunity.get("strategy", "Unknown"), "inline": True},
                {"name": "Estimated Profit", "value": f"${opportunity.get('profit', 0):.2f}", "inline": True},
                {"name": "Confidence", "value": f"{opportunity.get('confidence', 0):.1%}", "inline": True},
                {"name": "Symbol", "value": opportunity.get("symbol", "N/A"), "inline": True},
                {"name": "Exchange", "value": opportunity.get("exchange", "N/A"), "inline": True}
            ],
            "timestamp": "",
            "footer": {"text": "Opportunity Scanner"}
        }
        
        await self.send_message("", [embed])

    async def shutdown(self):
        if self.session:
            await self.session.close()

class NotificationManager:
    def __init__(self):
        self.discord = DiscordNotifier()
        self.enabled = bool(settings.discord_webhook)

    async def initialize(self):
        if self.enabled:
            await self.discord.initialize()

    async def send_trade_notification(self, strategy: str, symbol: str, 
                                    profit: float, success: bool, details: Dict):
        if self.enabled:
            await self.discord.send_trade_alert(strategy, symbol, profit, success, details)

    async def send_risk_notification(self, alert_type: str, message: str, severity: str = "warning"):
        if self.enabled:
            await self.discord.send_risk_alert(alert_type, message, severity)

    async def send_system_notification(self, component: str, status: str, details: str = ""):
        if self.enabled:
            await self.discord.send_system_alert(component, status, details)

    async def send_daily_summary(self, summary: Dict):
        if self.enabled:
            await self.discord.send_daily_summary(summary)

    async def send_opportunity_notification(self, opportunity: Dict):
        if self.enabled:
            await self.discord.send_opportunity_alert(opportunity)

    async def shutdown(self):
        if self.enabled:
            await self.discord.shutdown()