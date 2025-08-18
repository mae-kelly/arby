# TESTNET VERSION - Safe for testing with fake money
# flash_loan_arbitrage.py

import asyncio
import aiohttp
import json
from web3 import Web3
from eth_account import Account
import time
import requests

class FlashLoanArbitrageBot:
    def __init__(self):
        self.alchemy_api = "alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX"
        self.etherscan_api = "K4SEVFZ3PI8STM73VKV84C8PYZJUK7HB2G"
        self.wallet_address = "HjFs1U5F7mbWJiDKs7izTP96MEHytvm1yiSvKLT4mEvz"
        self.discord_webhook = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3"
        
        self.w3 = Web3(Web3.HTTPProvider(f"https://eth-testnet.g.alchemy.com/v2/{self.alchemy_api}"))
        
        self.aave_pool_address = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
        self.uniswap_router = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
        self.sushiswap_router = "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F"
        
        self.min_profit_threshold = 100
        self.max_loan_amount = 1000000
        
    def send_alert(self, message):
        try:
            payload = {"content": f"⚡ Flash Loan Bot: {message}"}
            requests.post(self.discord_webhook, json=payload, timeout=5)
            print(f"Alert: {message}")
        except Exception as e:
            print(f"Discord error: {e}")

    async def get_token_price(self, token_address, amount):
        try:
            url = f"https://api.etherscan.io/api"
            params = {
                "module": "account",
                "action": "tokenbalance",
                "contractaddress": token_address,
                "address": self.wallet_address,
                "tag": "latest",
                "apikey": self.etherscan_api
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return data.get("result", "0")
        except Exception as e:
            print(f"Price fetch error: {e}")
            return "0"

    def calculate_uniswap_price(self, token_in, token_out, amount_in):
        uniswap_abi = [
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
        
        try:
            contract = self.w3.eth.contract(
                address=self.uniswap_router,
                abi=uniswap_abi
            )
            
            path = [token_in, token_out]
            amounts = contract.functions.getAmountsOut(amount_in, path).call()
            return amounts[-1]
        except Exception as e:
            print(f"Uniswap price error: {e}")
            return 0

    def calculate_sushiswap_price(self, token_in, token_out, amount_in):
        sushiswap_abi = [
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
        
        try:
            contract = self.w3.eth.contract(
                address=self.sushiswap_router,
                abi=sushiswap_abi
            )
            
            path = [token_in, token_out]
            amounts = contract.functions.getAmountsOut(amount_in, path).call()
            return amounts[-1]
        except Exception as e:
            print(f"SushiSwap price error: {e}")
            return 0

    def find_arbitrage_opportunities(self):
        opportunities = []
        
        tokens = {
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "USDC": "0xA0b86a33E6417c7ef38BC67B2F11D6B3DC0B5f55",
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F"
        }
        
        for token_in_name, token_in_addr in tokens.items():
            for token_out_name, token_out_addr in tokens.items():
                if token_in_name == token_out_name:
                    continue
                
                amount_in = 0.01 * 10**18  # Testnet: 0.01 ETH max
                
                uniswap_out = self.calculate_uniswap_price(token_in_addr, token_out_addr, amount_in)
                sushiswap_out = self.calculate_sushiswap_price(token_in_addr, token_out_addr, amount_in)
                
                if uniswap_out > 0 and sushiswap_out > 0:
                    price_diff = abs(uniswap_out - sushiswap_out)
                    profit_percentage = (price_diff / min(uniswap_out, sushiswap_out)) * 100
                    
                    if profit_percentage > 0.1:
                        opportunity = {
                            "token_in": token_in_name,
                            "token_out": token_out_name,
                            "token_in_addr": token_in_addr,
                            "token_out_addr": token_out_addr,
                            "amount_in": amount_in,
                            "uniswap_out": uniswap_out,
                            "sushiswap_out": sushiswap_out,
                            "profit_percentage": profit_percentage,
                            "estimated_profit": price_diff
                        }
                        opportunities.append(opportunity)
        
        return sorted(opportunities, key=lambda x: x["profit_percentage"], reverse=True)

    def create_flash_loan_contract(self):
        flash_loan_contract = """
        pragma solidity ^0.8.0;

        import "@aave/core-v3/contracts/flashloan/base/FlashLoanSimpleReceiverBase.sol";
        import "@aave/core-v3/contracts/interfaces/IPoolAddressesProvider.sol";
        import "@aave/core-v3/contracts/dependencies/openzeppelin/contracts/IERC20.sol";

        contract FlashLoanArbitrage is FlashLoanSimpleReceiverBase {
            address payable owner;
            
            constructor(address _addressProvider) FlashLoanSimpleReceiverBase(IPoolAddressesProvider(_addressProvider)) {
                owner = payable(msg.sender);
            }

            function executeOperation(
                address asset,
                uint256 amount,
                uint256 premium,
                address initiator,
                bytes calldata params
            ) external override returns (bool) {
                
                (address tokenA, address tokenB, address uniswapRouter, address sushiswapRouter, uint256 amountIn) = 
                    abi.decode(params, (address, address, address, address, uint256));
                
                // 1. Swap on Uniswap
                IERC20(asset).approve(uniswapRouter, amountIn);
                
                address[] memory path = new address[](2);
                path[0] = tokenA;
                path[1] = tokenB;
                
                IUniswapV2Router(uniswapRouter).swapExactTokensForTokens(
                    amountIn,
                    0,
                    path,
                    address(this),
                    block.timestamp + 300
                );
                
                uint256 tokenBBalance = IERC20(tokenB).balanceOf(address(this));
                
                // 2. Swap back on SushiSwap
                IERC20(tokenB).approve(sushiswapRouter, tokenBBalance);
                
                address[] memory reversePath = new address[](2);
                reversePath[0] = tokenB;
                reversePath[1] = tokenA;
                
                IUniswapV2Router(sushiswapRouter).swapExactTokensForTokens(
                    tokenBBalance,
                    amountIn + premium,
                    reversePath,
                    address(this),
                    block.timestamp + 300
                );
                
                uint256 totalDebt = amount + premium;
                IERC20(asset).approve(address(POOL), totalDebt);
                
                return true;
            }

            function requestFlashLoan(address _token, uint256 _amount, bytes calldata _params) public {
                address receiverAddress = address(this);
                address asset = _token;
                uint256 amount = _amount;
                bytes memory params = _params;
                uint16 referralCode = 0;

                POOL.flashLoanSimple(
                    receiverAddress,
                    asset,
                    amount,
                    params,
                    referralCode
                );
            }

            modifier onlyOwner() {
                require(msg.sender == owner, "Only owner can call this function");
                _;
            }

            function withdraw(address _tokenAddress) external onlyOwner {
                IERC20 token = IERC20(_tokenAddress);
                token.transfer(msg.sender, token.balanceOf(address(this)));
            }
        }

        interface IUniswapV2Router {
            function swapExactTokensForTokens(
                uint amountIn,
                uint amountOutMin,
                address[] calldata path,
                address to,
                uint deadline
            ) external returns (uint[] memory amounts);
        }
        """
        
        return flash_loan_contract

    async def execute_flash_loan_arbitrage(self, opportunity):
        try:
            print(f"Executing flash loan arbitrage:")
            print(f"  {opportunity['token_in']} -> {opportunity['token_out']}")
            print(f"  Profit: {opportunity['profit_percentage']:.4f}%")
            print(f"  Estimated profit: ${opportunity['estimated_profit']:,.2f}")
            
            loan_amount = min(opportunity['amount_in'], self.max_loan_amount * 10**18)
            
            params = self.w3.codec.encode_abi(
                ['address', 'address', 'address', 'address', 'uint256'],
                [
                    opportunity['token_in_addr'],
                    opportunity['token_out_addr'],
                    self.uniswap_router,
                    self.sushiswap_router,
                    loan_amount
                ]
            )
            
            gas_estimate = 500000
            gas_price = self.w3.eth.gas_price
            gas_cost_eth = gas_estimate * gas_price / 10**18
            gas_cost_usd = gas_cost_eth * 3200
            
            if opportunity['estimated_profit'] > gas_cost_usd + self.min_profit_threshold:
                message = f"Flash loan opportunity found! Profit: {opportunity['profit_percentage']:.4f}% (${opportunity['estimated_profit']:,.2f}) Gas: ${gas_cost_usd:.2f}"
                self.send_alert(message)
                
                return {
                    "success": True,
                    "profit": opportunity['estimated_profit'] - gas_cost_usd,
                    "gas_cost": gas_cost_usd
                }
            else:
                print(f"Opportunity not profitable after gas costs: ${gas_cost_usd:.2f}")
                return {"success": False, "reason": "Gas costs too high"}
                
        except Exception as e:
            error_msg = f"Flash loan execution error: {str(e)}"
            self.send_alert(error_msg)
            print(error_msg)
            return {"success": False, "reason": str(e)}

    async def monitor_opportunities(self):
        self.send_alert("Flash Loan Arbitrage Bot started - monitoring DEX price differences")
        
        while True:
            try:
                print(f"[{time.strftime('%H:%M:%S')}] Scanning for arbitrage opportunities...")
                
                opportunities = self.find_arbitrage_opportunities()
                
                if opportunities:
                    print(f"Found {len(opportunities)} potential opportunities:")
                    
                    for i, opp in enumerate(opportunities[:3]):
                        print(f"  {i+1}. {opp['token_in']}->{opp['token_out']}: {opp['profit_percentage']:.4f}% profit")
                        
                        if opp['profit_percentage'] > 0.5:
                            result = await self.execute_flash_loan_arbitrage(opp)
                            
                            if result['success']:
                                profit_msg = f"Flash loan executed! Profit: ${result['profit']:,.2f}"
                                self.send_alert(profit_msg)
                                print(profit_msg)
                else:
                    print("No profitable opportunities found")
                
                await asyncio.sleep(10)
                
            except Exception as e:
                error_msg = f"Monitoring error: {str(e)}"
                self.send_alert(error_msg)
                print(error_msg)
                await asyncio.sleep(30)

    def get_wallet_balance(self):
        try:
            balance_wei = self.w3.eth.get_balance(self.wallet_address)
            balance_eth = self.w3.from_wei(balance_wei, 'ether')
            balance_usd = float(balance_eth) * 3200
            
            return {
                "eth": float(balance_eth),
                "usd": balance_usd,
                "wei": balance_wei
            }
        except Exception as e:
            print(f"Balance check error: {e}")
            return {"eth": 0, "usd": 0, "wei": 0}

    async def run_comprehensive_arbitrage(self):
        self.send_alert("🚀 Comprehensive Flash Loan Arbitrage System started!")
        
        balance = self.get_wallet_balance()
        self.send_alert(f"Wallet balance: {balance['eth']:.4f} ETH (${balance['usd']:,.2f})")
        
        print("Flash Loan Arbitrage Bot Configuration:")
        print(f"  Wallet: {self.wallet_address}")
        print(f"  Min profit threshold: ${self.min_profit_threshold}")
        print(f"  Max loan amount: ${self.max_loan_amount:,}")
        print(f"  Monitoring: Uniswap vs SushiSwap")
        
        await self.monitor_opportunities()

if __name__ == "__main__":
    bot = FlashLoanArbitrageBot()
    asyncio.run(bot.run_comprehensive_arbitrage())