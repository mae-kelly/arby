import asyncio
import signal
import sys
from core.bot_manager import BotManager
from utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)

class ArbitrageBotMain:
    def __init__(self):
        self.bot_manager = BotManager()
        self.running = False

    async def start(self):
        self.running = True
        logger.info("Starting Arbitrage Bot")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            await self.bot_manager.initialize()
            await self.bot_manager.start()
            
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Critical error: {e}")
            raise
        finally:
            await self.shutdown()

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    async def shutdown(self):
        logger.info("Shutting down bot manager")
        await self.bot_manager.shutdown()
        logger.info("Bot shutdown complete")

async def main():
    bot = ArbitrageBotMain()
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())