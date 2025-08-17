// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
}

interface IUniswapV2Router {
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
    
    function getAmountsOut(uint amountIn, address[] calldata path)
        external view returns (uint[] memory amounts);
}

interface IUniswapV3Router {
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }
    
    function exactInputSingle(ExactInputSingleParams calldata params) 
        external payable returns (uint256 amountOut);
}

interface IFlashLoanReceiver {
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external returns (bool);
}

interface ILendingPool {
    function flashLoan(
        address receiverAddress,
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata modes,
        address onBehalfOf,
        bytes calldata params,
        uint16 referralCode
    ) external;
}

interface ICurvePool {
    function exchange(int128 i, int128 j, uint256 dx, uint256 min_dy) external returns (uint256);
    function get_dy(int128 i, int128 j, uint256 dx) external view returns (uint256);
}

interface IBalancerVault {
    function flashLoan(
        address recipient,
        address[] memory tokens,
        uint256[] memory amounts,
        bytes memory userData
    ) external;
}

contract MultiDexArbitrage is IFlashLoanReceiver {
    address private constant AAVE_LENDING_POOL = 0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9;
    address private constant BALANCER_VAULT = 0xBA12222222228d8Ba445958a75a0704d566BF2C8;
    address private constant UNISWAP_V2_ROUTER = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;
    address private constant UNISWAP_V3_ROUTER = 0xE592427A0AEce92De3Edee1F18E0157C05861564;
    address private constant SUSHISWAP_ROUTER = 0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F;
    
    address private immutable owner;
    uint256 private constant MAX_SLIPPAGE = 50; // 0.5%
    
    mapping(address => bool) private authorized;
    mapping(bytes32 => bool) private executedTrades;
    
    struct ArbitragePath {
        address[] tokens;
        address[] routers;
        uint24[] fees;
        uint256 amountIn;
        uint256 minAmountOut;
    }
    
    event ArbitrageExecuted(
        address indexed token,
        uint256 profit,
        uint256 gasUsed
    );
    
    modifier onlyAuthorized() {
        require(authorized[msg.sender] || msg.sender == owner, "Unauthorized");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        authorized[msg.sender] = true;
    }
    
    function executeArbitrage(ArbitragePath calldata path) external onlyAuthorized {
        uint256 gasStart = gasleft();
        bytes32 tradeHash = keccak256(abi.encode(path, block.number));
        require(!executedTrades[tradeHash], "Already executed");
        executedTrades[tradeHash] = true;
        
        // Execute flash loan
        address[] memory assets = new address[](1);
        assets[0] = path.tokens[0];
        
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = path.amountIn;
        
        uint256[] memory modes = new uint256[](1);
        modes[0] = 0; // No debt
        
        bytes memory params = abi.encode(path);
        
        // Try Balancer first (0% fee)
        try IBalancerVault(BALANCER_VAULT).flashLoan(
            address(this),
            assets,
            amounts,
            params
        ) {
            // Success
        } catch {
            // Fallback to Aave
            ILendingPool(AAVE_LENDING_POOL).flashLoan(
                address(this),
                assets,
                amounts,
                modes,
                address(this),
                params,
                0
            );
        }
        
        uint256 gasUsed = gasStart - gasleft();
        emit ArbitrageExecuted(path.tokens[0], 0, gasUsed);
    }
    
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        require(msg.sender == AAVE_LENDING_POOL || msg.sender == BALANCER_VAULT, "Invalid caller");
        require(initiator == address(this), "Invalid initiator");
        
        ArbitragePath memory path = abi.decode(params, (ArbitragePath));
        
        uint256 balanceBefore = IERC20(assets[0]).balanceOf(address(this));
        
        // Execute arbitrage trades
        uint256 currentAmount = amounts[0];
        
        for (uint256 i = 0; i < path.routers.length; i++) {
            currentAmount = _executeTrade(
                path.tokens[i],
                path.tokens[i + 1],
                currentAmount,
                path.routers[i],
                path.fees[i]
            );
        }
        
        // Ensure profit
        uint256 totalDebt = amounts[0] + premiums[0];
        require(currentAmount >= totalDebt, "No profit");
        
        // Repay flash loan
        IERC20(assets[0]).approve(msg.sender, totalDebt);
        
        // Transfer profit to owner
        uint256 profit = currentAmount - totalDebt;
        if (profit > 0) {
            IERC20(assets[0]).transfer(owner, profit);
        }
        
        return true;
    }
    
    function _executeTrade(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        address router,
        uint24 fee
    ) private returns (uint256) {
        IERC20(tokenIn).approve(router, amountIn);
        
        if (router == UNISWAP_V3_ROUTER) {
            IUniswapV3Router.ExactInputSingleParams memory params = 
                IUniswapV3Router.ExactInputSingleParams({
                    tokenIn: tokenIn,
                    tokenOut: tokenOut,
                    fee: fee,
                    recipient: address(this),
                    deadline: block.timestamp,
                    amountIn: amountIn,
                    amountOutMinimum: 0,
                    sqrtPriceLimitX96: 0
                });
            
            return IUniswapV3Router(router).exactInputSingle(params);
        } else {
            address[] memory path = new address[](2);
            path[0] = tokenIn;
            path[1] = tokenOut;
            
            uint[] memory amounts = IUniswapV2Router(router).swapExactTokensForTokens(
                amountIn,
                0,
                path,
                address(this),
                block.timestamp
            );
            
            return amounts[amounts.length - 1];
        }
    }
    
    function calculateProfit(ArbitragePath calldata path) external view returns (uint256) {
        uint256 amountOut = path.amountIn;
        
        for (uint256 i = 0; i < path.routers.length; i++) {
            amountOut = _getAmountOut(
                path.tokens[i],
                path.tokens[i + 1],
                amountOut,
                path.routers[i],
                path.fees[i]
            );
        }
        
        return amountOut > path.amountIn ? amountOut - path.amountIn : 0;
    }
    
    function _getAmountOut(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        address router,
        uint24 fee
    ) private view returns (uint256) {
        if (router == UNISWAP_V2_ROUTER || router == SUSHISWAP_ROUTER) {
            address[] memory path = new address[](2);
            path[0] = tokenIn;
            path[1] = tokenOut;
            
            uint[] memory amounts = IUniswapV2Router(router).getAmountsOut(amountIn, path);
            return amounts[1];
        }
        
        // For V3, approximate
        return amountIn * 997 / 1000;
    }
    
    function addAuthorized(address user) external {
        require(msg.sender == owner, "Only owner");
        authorized[user] = true;
    }
    
    function removeAuthorized(address user) external {
        require(msg.sender == owner, "Only owner");
        authorized[user] = false;
    }
    
    function emergencyWithdraw(address token) external {
        require(msg.sender == owner, "Only owner");
        uint256 balance = IERC20(token).balanceOf(address(this));
        IERC20(token).transfer(owner, balance);
    }
    
    receive() external payable {}
}