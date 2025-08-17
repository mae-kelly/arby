import asyncio
from typing import Dict, List
from strategies.base_strategy import BaseStrategy, TradeResult
from core.opportunity_scanner import Opportunity
from blockchain.web3_manager import Web3Manager
from blockchain.gas_optimizer import GasOptimizer
from exchanges.uniswap_client import UniswapClient
from exchanges.binance_client import BinanceClient
from utils.logger import get_logger
from config.settings import STRATEGY_CONFIGS

logger = get_logger(__name__)

class FlashLoanArbitrage(BaseStrategy):
    def __init__(self, risk_manager, portfolio_manager):
        super().__init__(risk_manager, portfolio_manager)
        self.web3_manager = Web3Manager()
        self.gas_optimizer = GasOptimizer()
        self.uniswap = UniswapClient()
        self.binance = BinanceClient()
        self.contract_address = None
        self.min_profit = STRATEGY_CONFIGS["flash_loan"]["min_profit"]

    async def initialize(self):
        await self.web3_manager.initialize()
        await self.gas_optimizer.initialize()
        await self.uniswap.initialize()
        await self.binance.initialize()
        
        self.contract_address = await self._deploy_arbitrage_contract()
        logger.info(f"Flash loan contract deployed at: {self.contract_address}")

    async def execute_opportunity(self, opportunity: Opportunity) -> TradeResult:
        if not await self._validate_opportunity(opportunity):
            return TradeResult(False, 0, 0, 0, "Validation failed")

        await self._log_trade_attempt(opportunity, {})
        
        trade_params = await self._prepare_trade_params(opportunity)
        result = await self._execute_trade(trade_params)
        
        await self._log_trade_result(result)
        return result

    async def _prepare_trade_params(self, opportunity: Opportunity) -> Dict:
        symbol = opportunity.data["symbol"]
        dex_price = opportunity.data["dex_price"]
        cex_price = opportunity.data["cex_price"]
        
        amount = self._calculate_optimal_amount(opportunity)
        
        if dex_price < cex_price:
            buy_exchange = "uniswap"
            sell_exchange = "binance"
        else:
            buy_exchange = "binance"
            sell_exchange = "uniswap"
        
        return {
            "symbol": symbol,
            "amount": amount,
            "buy_exchange": buy_exchange,
            "sell_exchange": sell_exchange,
            "dex_price": dex_price,
            "cex_price": cex_price
        }

    def _calculate_optimal_amount(self, opportunity: Opportunity) -> float:
        base_amount = 10000.0
        profit_multiplier = min(opportunity.profit_estimate / 1000, 5.0)
        confidence_multiplier = opportunity.confidence
        
        return base_amount * profit_multiplier * confidence_multiplier

    async def _perform_trade(self, trade_params: Dict) -> Dict:
        symbol = trade_params["symbol"]
        amount = trade_params["amount"]
        
        gas_price = await self.gas_optimizer.get_optimal_gas_price()
        
        if trade_params["buy_exchange"] == "uniswap":
            tx_hash = await self._execute_dex_to_cex_arbitrage(trade_params, gas_price)
        else:
            tx_hash = await self._execute_cex_to_dex_arbitrage(trade_params, gas_price)
        
        receipt = await self.web3_manager.wait_for_transaction(tx_hash)
        
        profit = self._calculate_actual_profit(receipt, trade_params)
        gas_cost = receipt['gasUsed'] * gas_price / 1e18
        
        return {
            "success": True,
            "profit": profit,
            "gas_cost": gas_cost,
            "tx_hash": tx_hash
        }

    async def _execute_dex_to_cex_arbitrage(self, params: Dict, gas_price: int) -> str:
        symbol = params["symbol"]
        amount = params["amount"]
        
        flash_loan_data = await self._encode_flash_loan_data({
            "token": symbol,
            "amount": amount,
            "buy_dex": True,
            "uniswap_pool": await self.uniswap.get_pool_address(symbol),
            "target_price": params["cex_price"]
        })
        
        tx = await self.web3_manager.build_transaction(
            to=self.contract_address,
            data=flash_loan_data,
            gas_price=gas_price
        )
        
        return await self.web3_manager.send_transaction(tx)

    async def _execute_cex_to_dex_arbitrage(self, params: Dict, gas_price: int) -> str:
        symbol = params["symbol"]
        amount = params["amount"]
        
        await self.binance.place_market_order(symbol, "buy", amount)
        
        await asyncio.sleep(1)
        
        balance = await self.binance.get_balance(symbol.split('/')[0])
        
        if balance >= amount * 0.99:
            await self.binance.withdraw(symbol.split('/')[0], amount, 
                                      self.web3_manager.get_address())
            
            await asyncio.sleep(30)
            
            swap_data = await self._encode_swap_data({
                "token": symbol,
                "amount": amount,
                "min_out": amount * params["dex_price"] * 0.99
            })
            
            tx = await self.web3_manager.build_transaction(
                to=self.uniswap.router_address,
                data=swap_data,
                gas_price=gas_price
            )
            
            return await self.web3_manager.send_transaction(tx)
        
        raise Exception("CEX order execution failed")

    async def _encode_flash_loan_data(self, params: Dict) -> bytes:
        function_selector = "0x1234abcd"
        
        encoded_params = self.web3_manager.w3.codec.encode_abi(
            ['address', 'uint256', 'bool', 'address', 'uint256'],
            [
                params["token"],
                int(params["amount"] * 1e18),
                params["buy_dex"],
                params["uniswap_pool"],
                int(params["target_price"] * 1e18)
            ]
        )
        
        return function_selector.encode() + encoded_params

    async def _encode_swap_data(self, params: Dict) -> bytes:
        function_selector = "0xabcd1234"
        
        encoded_params = self.web3_manager.w3.codec.encode_abi(
            ['address', 'uint256', 'uint256'],
            [
                params["token"],
                int(params["amount"] * 1e18),
                int(params["min_out"] * 1e18)
            ]
        )
        
        return function_selector.encode() + encoded_params

    def _calculate_actual_profit(self, receipt: Dict, params: Dict) -> float:
        logs = receipt.get('logs', [])
        
        for log in logs:
            if log.get('topics', []) and log['topics'][0].hex() == "0x1234":
                profit_wei = int(log['data'], 16)
                return profit_wei / 1e18
        
        return 0.0

    async def _deploy_arbitrage_contract(self) -> str:
        bytecode = "0x608060405234801561001057600080fd5b50..."
        
        tx = await self.web3_manager.build_transaction(
            data=bytecode,
            gas_price=await self.gas_optimizer.get_optimal_gas_price()
        )
        
        tx_hash = await self.web3_manager.send_transaction(tx)
        receipt = await self.web3_manager.wait_for_transaction(tx_hash)
        
        return receipt['contractAddress']

    async def health_check(self) -> bool:
        try:
            balance = await self.web3_manager.get_balance()
            if balance < 0.1:
                logger.warning("Low ETH balance for gas")
                return False
            
            contract_code = await self.web3_manager.get_code(self.contract_address)
            if contract_code == "0x":
                logger.error("Arbitrage contract not found")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def _strategy_specific_validation(self, opportunity: Opportunity) -> bool:
        if opportunity.profit_estimate < self.min_profit:
            return False
        
        if opportunity.strategy_type != "flash_loan":
            return False
        
        symbol = opportunity.data.get("symbol")
        if not symbol:
            return False
        
        liquidity = await self.uniswap.get_pool_liquidity(symbol)
        if liquidity < 100000:
            return False
        
        return True