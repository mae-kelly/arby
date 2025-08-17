// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

interface IFlashLoanProvider {
    function flashLoan(
        address receiver,
        address[] calldata tokens,
        uint256[] calldata amounts,
        bytes calldata data
    ) external;
}

interface IDEXRouter {
    function swapExactTokensForTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external returns (uint256[] memory amounts);
    
    function getAmountsOut(uint256 amountIn, address[] calldata path)
        external view returns (uint256[] memory amounts);
}

contract MEVExecutor is ReentrancyGuard, Ownable {
    
    struct SandwichParams {
        address tokenIn;
        address tokenOut;
        uint256 frontrunAmount;
        uint256 backrunAmount;
        address[] frontrunPath;
        address[] backrunPath;
        address frontrunRouter;
        address backrunRouter;
        uint256 maxGasPrice;
    }
    
    struct LiquidationParams {
        address protocol;
        address user;
        address collateralAsset;
        address debtAsset;
        uint256 debtToCover;
        bool receiveAToken;
    }
    
    mapping(address => bool) public authorizedCallers;
    mapping(address => bool) public authorizedRouters;
    
    uint256 public constant MAX_SLIPPAGE = 300; // 3%
    uint256 public totalProfitExtracted;
    uint256 public successfulMEVs;
    
    event MEVExecuted(
        string indexed mevType,
        uint256 profit,
        uint256 gasUsed
    );
    
    event SandwichExecuted(
        address indexed token,
        uint256 frontrunAmount,
        uint256 backrunAmount,
        uint256 profit
    );
    
    event LiquidationExecuted(
        address indexed protocol,
        address indexed user,
        uint256 profit
    );
    
    modifier onlyAuthorized() {
        require(authorizedCallers[msg.sender] || msg.sender == owner(), "Unauthorized");
        _;
    }
    
    constructor() {
        authorizedCallers[msg.sender] = true;
        
        // Authorize major DEX routers
        authorizedRouters[0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D] = true; // Uniswap V2
        authorizedRouters[0xE592427A0AEce92De3Edee1F18E0157C05861564] = true; // Uniswap V3
        authorizedRouters[0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F] = true; // Sushiswap
    }
    
    function executeSandwich(
        SandwichParams calldata params
    ) external onlyAuthorized nonReentrant {
        require(tx.gasprice <= params.maxGasPrice, "Gas price too high");
        require(authorizedRouters[params.frontrunRouter], "Router not authorized");
        
        uint256 gasStart = gasleft();
        uint256 initialBalance = IERC20(params.tokenIn).balanceOf(address(this));
        
        // Execute frontrun
        IERC20(params.tokenIn).approve(params.frontrunRouter, params.frontrunAmount);
        
        IDEXRouter(params.frontrunRouter).swapExactTokensForTokens(
            params.frontrunAmount,
            0, // Accept any amount of tokens out
            params.frontrunPath,
            address(this),
            block.timestamp + 300
        );
        
        // Note: Victim transaction executes here (handled by mempool ordering)
        
        // Execute backrun  
        uint256 backrunBalance = IERC20(params.tokenOut).balanceOf(address(this));
        IERC20(params.tokenOut).approve(params.backrunRouter, backrunBalance);
        
        IDEXRouter(params.backrunRouter).swapExactTokensForTokens(
            backrunBalance,
            params.backrunAmount,
            params.backrunPath,
            address(this),
            block.timestamp + 300
        );
        
        uint256 finalBalance = IERC20(params.tokenIn).balanceOf(address(this));
        uint256 profit = finalBalance - initialBalance;
        
        require(profit > 0, "No profit");
        
        totalProfitExtracted += profit;
        successfulMEVs++;
        
        uint256 gasUsed = gasStart - gasleft();
        
        emit SandwichExecuted(
            params.tokenIn,
            params.frontrunAmount,
            params.backrunAmount,
            profit
        );
        
        emit MEVExecuted("sandwich", profit, gasUsed);
    }
    
    function executeLiquidation(
        LiquidationParams calldata params
    ) external onlyAuthorized nonReentrant {
        uint256 gasStart = gasleft();
        
        // Get flash loan for liquidation
        address[] memory assets = new address[](1);
        assets[0] = params.debtAsset;
        
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = params.debtToCover;
        
        bytes memory data = abi.encode(params);
        
        IFlashLoanProvider(params.protocol).flashLoan(
            address(this),
            assets,
            amounts,
            data
        );
        
        uint256 gasUsed = gasStart - gasleft();
        emit MEVExecuted("liquidation", 0, gasUsed);
    }
    
    function onFlashLoan(
        address,
        address[] calldata tokens,
        uint256[] calldata amounts,
        uint256[] calldata fees,
        bytes calldata data
    ) external returns (bool) {
        LiquidationParams memory params = abi.decode(data, (LiquidationParams));
        
        // Execute liquidation call
        uint256 balanceBefore = IERC20(params.collateralAsset).balanceOf(address(this));
        
        // Call liquidation function on lending protocol
        (bool success,) = params.protocol.call(
            abi.encodeWithSignature(
                "liquidationCall(address,address,address,uint256,bool)",
                params.collateralAsset,
                params.debtAsset,
                params.user,
                params.debtToCover,
                params.receiveAToken
            )
        );
        
        require(success, "Liquidation failed");
        
        uint256 balanceAfter = IERC20(params.collateralAsset).balanceOf(address(this));
        uint256 collateralReceived = balanceAfter - balanceBefore;
        
        // Swap collateral to debt token to repay flash loan
        if (params.collateralAsset != params.debtAsset) {
            address[] memory path = new address[](2);
            path[0] = params.collateralAsset;
            path[1] = params.debtAsset;
            
            IERC20(params.collateralAsset).approve(
                0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D, // Uniswap router
                collateralReceived
            );
            
            IDEXRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D).swapExactTokensForTokens(
                collateralReceived,
                amounts[0] + fees[0], // Must cover flash loan + fee
                path,
                address(this),
                block.timestamp + 300
            );
        }
        
        // Repay flash loan
        IERC20(tokens[0]).approve(msg.sender, amounts[0] + fees[0]);
        
        uint256 profit = IERC20(params.debtAsset).balanceOf(address(this)) - (amounts[0] + fees[0]);
        
        totalProfitExtracted += profit;
        successfulMEVs++;
        
        emit LiquidationExecuted(params.protocol, params.user, profit);
        
        return true;
    }
    
    function frontrunTransaction(
        address router,
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 minAmountOut,
        uint256 maxGasPrice
    ) external onlyAuthorized nonReentrant {
        require(tx.gasprice <= maxGasPrice, "Gas price too high");
        require(authorizedRouters[router], "Router not authorized");
        
        uint256 gasStart = gasleft();
        
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;
        
        IERC20(tokenIn).approve(router, amountIn);
        
        uint256[] memory amounts = IDEXRouter(router).swapExactTokensForTokens(
            amountIn,
            minAmountOut,
            path,
            address(this),
            block.timestamp + 300
        );
        
        uint256 gasUsed = gasStart - gasleft();
        emit MEVExecuted("frontrun", amounts[1] - amountIn, gasUsed);
    }
    
    function extractArbitrage(
        address[] calldata routers,
        address[][] calldata paths,
        uint256[] calldata amounts
    ) external onlyAuthorized nonReentrant {
        require(routers.length == paths.length, "Array length mismatch");
        require(paths.length == amounts.length, "Array length mismatch");
        
        uint256 gasStart = gasleft();
        uint256 totalProfit = 0;
        
        for (uint256 i = 0; i < routers.length; i++) {
            require(authorizedRouters[routers[i]], "Router not authorized");
            
            IERC20(paths[i][0]).approve(routers[i], amounts[i]);
            
            uint256[] memory swapAmounts = IDEXRouter(routers[i]).swapExactTokensForTokens(
                amounts[i],
                0,
                paths[i],
                address(this),
                block.timestamp + 300
            );
            
            totalProfit += swapAmounts[swapAmounts.length - 1];
        }
        
        totalProfitExtracted += totalProfit;
        successfulMEVs++;
        
        uint256 gasUsed = gasStart - gasleft();
        emit MEVExecuted("arbitrage", totalProfit, gasUsed);
    }
    
    function addAuthorizedCaller(address caller) external onlyOwner {
        authorizedCallers[caller] = true;
    }
    
    function removeAuthorizedCaller(address caller) external onlyOwner {
        authorizedCallers[caller] = false;
    }
    
    function addAuthorizedRouter(address router) external onlyOwner {
        authorizedRouters[router] = true;
    }
    
    function removeAuthorizedRouter(address router) external onlyOwner {
        authorizedRouters[router] = false;
    }
    
    function withdrawToken(address token, uint256 amount) external onlyOwner {
        IERC20(token).transfer(owner(), amount);
    }
    
    function withdrawETH() external onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }
    
    function getStats() external view returns (
        uint256 totalProfit,
        uint256 successfulMEVCount,
        uint256 contractBalance
    ) {
        return (
            totalProfitExtracted,
            successfulMEVs,
            address(this).balance
        );
    }
    
    receive() external payable {}
}
