// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

interface IFlashLoanProvider {
    function flashLoan(
        address receiver,
        address[] calldata tokens,
        uint256[] calldata amounts,
        bytes calldata data
    ) external;
}

interface IDEXRouter {
    function getAmountsOut(uint256 amountIn, address[] calldata path) 
        external view returns (uint256[] memory amounts);
    
    function swapExactTokensForTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external returns (uint256[] memory amounts);
}

interface IWETH {
    function deposit() external payable;
    function withdraw(uint256) external;
    function balanceOf(address) external view returns (uint256);
    function transfer(address, uint256) external returns (bool);
}

contract FlashLoanReceiver is ReentrancyGuard, AccessControl {
    using SafeERC20 for IERC20;
    
    bytes32 public constant EXECUTOR_ROLE = keccak256("EXECUTOR_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    
    address private constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    uint256 private constant MAX_SLIPPAGE = 300; // 3%
    uint256 private constant EMERGENCY_WITHDRAW_DELAY = 3 days;
    
    struct FlashLoanData {
        address[] routers;
        address[][] paths;
        uint256[] amounts;
        uint256[] minAmountsOut;
        bytes routerCalldata;
        uint256 deadline;
    }
    
    struct ProfitData {
        uint256 totalProfit;
        uint256 totalGasUsed;
        uint256 successfulTrades;
        uint256 failedTrades;
        mapping(address => uint256) tokenProfits;
    }
    
    mapping(address => bool) public authorizedProviders;
    mapping(address => bool) public authorizedRouters;
    mapping(bytes32 => bool) public executedTrades;
    mapping(address => AggregatorV3Interface) public priceFeeds;
    
    ProfitData public profitData;
    uint256 public emergencyWithdrawTime;
    bool public paused;
    
    event FlashLoanReceived(
        address indexed provider,
        address indexed token,
        uint256 amount
    );
    
    event ArbitrageExecuted(
        bytes32 indexed tradeId,
        uint256 profit,
        uint256 gasUsed
    );
    
    event EmergencyWithdraw(
        address indexed token,
        uint256 amount
    );
    
    modifier notPaused() {
        require(!paused, "Contract paused");
        _;
    }
    
    modifier onlyAuthorizedProvider() {
        require(authorizedProviders[msg.sender], "Unauthorized provider");
        _;
    }
    
    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
        _grantRole(EXECUTOR_ROLE, msg.sender);
        
        // Initialize authorized providers
        authorizedProviders[0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9] = true; // Aave
        authorizedProviders[0xBA12222222228d8Ba445958a75a0704d566BF2C8] = true; // Balancer
        
        // Initialize authorized routers
        authorizedRouters[0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D] = true; // Uniswap V2
        authorizedRouters[0xE592427A0AEce92De3Edee1F18E0157C05861564] = true; // Uniswap V3
        authorizedRouters[0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F] = true; // Sushiswap
        
        // Initialize price feeds (Chainlink)
        priceFeeds[WETH] = AggregatorV3Interface(0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419);
    }
    
    function executeFlashLoan(
        address provider,
        address[] calldata tokens,
        uint256[] calldata amounts,
        bytes calldata data
    ) external onlyRole(EXECUTOR_ROLE) notPaused nonReentrant {
        require(authorizedProviders[provider], "Unauthorized provider");
        
        bytes32 tradeId = keccak256(abi.encode(block.number, tokens, amounts));
        require(!executedTrades[tradeId], "Trade already executed");
        executedTrades[tradeId] = true;
        
        IFlashLoanProvider(provider).flashLoan(
            address(this),
            tokens,
            amounts,
            data
        );
    }
    
    function onFlashLoan(
        address initiator,
        address[] calldata tokens,
        uint256[] calldata amounts,
        uint256[] calldata fees,
        bytes calldata data
    ) external onlyAuthorizedProvider nonReentrant returns (bool) {
        require(initiator == address(this), "Invalid initiator");
        
        uint256 gasStart = gasleft();
        
        // Decode arbitrage parameters
        FlashLoanData memory flashData = abi.decode(data, (FlashLoanData));
        
        // Validate deadline
        require(block.timestamp <= flashData.deadline, "Deadline passed");
        
        uint256[] memory balancesBefore = new uint256[](tokens.length);
        for (uint256 i = 0; i < tokens.length; i++) {
            balancesBefore[i] = IERC20(tokens[i]).balanceOf(address(this));
            emit FlashLoanReceived(msg.sender, tokens[i], amounts[i]);
        }
        
        // Execute arbitrage
        uint256 totalProfit = _executeArbitrage(flashData, tokens);
        
        // Repay flash loans
        for (uint256 i = 0; i < tokens.length; i++) {
            uint256 amountOwed = amounts[i] + fees[i];
            uint256 balance = IERC20(tokens[i]).balanceOf(address(this));
            
            require(balance >= amountOwed, "Insufficient balance to repay");
            
            IERC20(tokens[i]).safeTransfer(msg.sender, amountOwed);
            
            // Record profit
            uint256 profit = balance - balancesBefore[i] - fees[i];
            if (profit > 0) {
                profitData.tokenProfits[tokens[i]] += profit;
                totalProfit += profit;
            }
        }
        
        // Update statistics
        uint256 gasUsed = gasStart - gasleft();
        profitData.totalGasUsed += gasUsed;
        
        if (totalProfit > 0) {
            profitData.totalProfit += totalProfit;
            profitData.successfulTrades++;
            
            emit ArbitrageExecuted(
                keccak256(abi.encode(block.number, tokens)),
                totalProfit,
                gasUsed
            );
        } else {
            profitData.failedTrades++;
        }
        
        return true;
    }
    
    function _executeArbitrage(
        FlashLoanData memory data,
        address[] calldata tokens
    ) private returns (uint256 profit) {
        // Validate routers
        for (uint256 i = 0; i < data.routers.length; i++) {
            require(authorizedRouters[data.routers[i]], "Unauthorized router");
        }
        
        // Execute swaps across multiple DEXs
        for (uint256 i = 0; i < data.routers.length; i++) {
            address router = data.routers[i];
            address[] memory path = data.paths[i];
            uint256 amountIn = data.amounts[i];
            uint256 minAmountOut = data.minAmountsOut[i];
            
            // Approve router
            IERC20(path[0]).safeApprove(router, amountIn);
            
            // Get expected output
            uint256[] memory expectedAmounts = IDEXRouter(router).getAmountsOut(amountIn, path);
            uint256 expectedOut = expectedAmounts[expectedAmounts.length - 1];
            
            // Validate slippage
            require(expectedOut >= minAmountOut, "Excessive slippage");
            
            // Check price oracle if available
            if (address(priceFeeds[path[0]]) != address(0) && 
                address(priceFeeds[path[path.length - 1]]) != address(0)) {
                require(_validatePriceOracle(path, amountIn, expectedOut), "Price manipulation detected");
            }
            
            // Execute swap
            try IDEXRouter(router).swapExactTokensForTokens(
                amountIn,
                minAmountOut,
                path,
                address(this),
                data.deadline
            ) returns (uint256[] memory amounts) {
                profit += amounts[amounts.length - 1] - amountIn;
            } catch {
                // Swap failed, revert approvals
                IERC20(path[0]).safeApprove(router, 0);
            }
        }
        
        return profit;
    }
    
    function _validatePriceOracle(
        address[] memory path,
        uint256 amountIn,
        uint256 expectedOut
    ) private view returns (bool) {
        // Get oracle prices
        (, int256 priceIn,,,) = priceFeeds[path[0]].latestRoundData();
        (, int256 priceOut,,,) = priceFeeds[path[path.length - 1]].latestRoundData();
        
        // Calculate expected output based on oracle
        uint256 oracleExpected = (amountIn * uint256(priceIn)) / uint256(priceOut);
        
        // Allow 5% deviation from oracle price
        uint256 deviation = (expectedOut > oracleExpected) 
            ? ((expectedOut - oracleExpected) * 10000) / oracleExpected
            : ((oracleExpected - expectedOut) * 10000) / oracleExpected;
            
        return deviation <= 500; // 5%
    }
    
    function simulateArbitrage(
        FlashLoanData calldata data,
        address[] calldata tokens
    ) external view returns (uint256 expectedProfit, uint256 gasEstimate) {
        uint256 totalIn = 0;
        uint256 totalOut = 0;
        
        for (uint256 i = 0; i < data.routers.length; i++) {
            require(authorizedRouters[data.routers[i]], "Unauthorized router");
            
            address[] memory path = data.paths[i];
            uint256 amountIn = data.amounts[i];
            
            uint256[] memory amounts = IDEXRouter(data.routers[i]).getAmountsOut(amountIn, path);
            
            totalIn += amountIn;
            totalOut += amounts[amounts.length - 1];
            
            // Estimate gas per swap
            gasEstimate += 150000; // Approximate gas for swap
        }
        
        expectedProfit = (totalOut > totalIn) ? totalOut - totalIn : 0;
        gasEstimate += 100000; // Base gas for flash loan
        
        return (expectedProfit, gasEstimate);
    }
    
    function withdrawProfit(address token) external onlyRole(ADMIN_ROLE) nonReentrant {
        uint256 balance = IERC20(token).balanceOf(address(this));
        require(balance > 0, "No balance");
        
        IERC20(token).safeTransfer(msg.sender, balance);
    }
    
    function emergencyWithdraw(address token) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(
            emergencyWithdrawTime != 0 && block.timestamp >= emergencyWithdrawTime,
            "Emergency withdraw not activated or delay not passed"
        );
        
        uint256 balance;
        if (token == address(0)) {
            balance = address(this).balance;
            payable(msg.sender).transfer(balance);
        } else {
            balance = IERC20(token).balanceOf(address(this));
            IERC20(token).safeTransfer(msg.sender, balance);
        }
        
        emit EmergencyWithdraw(token, balance);
    }
    
    function activateEmergencyWithdraw() external onlyRole(DEFAULT_ADMIN_ROLE) {
        emergencyWithdrawTime = block.timestamp + EMERGENCY_WITHDRAW_DELAY;
    }
    
    function setPaused(bool _paused) external onlyRole(ADMIN_ROLE) {
        paused = _paused;
    }
    
    function updateAuthorizedProvider(address provider, bool authorized) 
        external onlyRole(ADMIN_ROLE) {
        authorizedProviders[provider] = authorized;
    }
    
    function updateAuthorizedRouter(address router, bool authorized) 
        external onlyRole(ADMIN_ROLE) {
        authorizedRouters[router] = authorized;
    }
    
    function updatePriceFeed(address token, address feed) 
        external onlyRole(ADMIN_ROLE) {
        priceFeeds[token] = AggregatorV3Interface(feed);
    }
    
    function getProfit(address token) external view returns (uint256) {
        return profitData.tokenProfits[token];
    }
    
    function getStats() external view returns (
        uint256 totalProfit,
        uint256 totalGasUsed,
        uint256 successfulTrades,
        uint256 failedTrades
    ) {
        return (
            profitData.totalProfit,
            profitData.totalGasUsed,
            profitData.successfulTrades,
            profitData.failedTrades
        );
    }
    
    receive() external payable {
        if (msg.sender != WETH) {
            IWETH(WETH).deposit{value: msg.value}();
        }
    }
    
    fallback() external payable {
        revert("Invalid call");
    }
}