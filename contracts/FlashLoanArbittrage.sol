pragma solidity ^0.8.19;

import "@aave/core-v3/contracts/flashloan/base/FlashLoanSimpleReceiverBase.sol";
import "@aave/core-v3/contracts/interfaces/IPoolAddressesProvider.sol";
import "@aave/core-v3/contracts/interfaces/IPool.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol";

contract FlashLoanArbitrage is FlashLoanSimpleReceiverBase {
    ISwapRouter public immutable swapRouter;
    address public owner;
    
    struct ArbitrageParams {
        address tokenIn;
        address tokenOut;
        uint256 amountIn;
        uint24 fee;
        address exchange;
        uint256 minAmountOut;
    }
    
    event ArbitrageExecuted(
        address indexed token,
        uint256 borrowed,
        uint256 profit
    );
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    constructor(
        address _addressProvider,
        address _swapRouter
    ) FlashLoanSimpleReceiverBase(IPoolAddressesProvider(_addressProvider)) {
        swapRouter = ISwapRouter(_swapRouter);
        owner = msg.sender;
    }
    
    function executeArbitrage(
        address asset,
        uint256 amount,
        bytes calldata params
    ) external onlyOwner {
        POOL.flashLoanSimple(address(this), asset, amount, params, 0);
    }
    
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        require(msg.sender == address(POOL), "Caller must be pool");
        
        ArbitrageParams memory arbParams = abi.decode(params, (ArbitrageParams));
        
        uint256 amountOwed = amount + premium;
        uint256 profit = _performArbitrage(asset, amount, arbParams);
        
        require(IERC20(asset).balanceOf(address(this)) >= amountOwed, "Insufficient funds to repay");
        
        emit ArbitrageExecuted(asset, amount, profit);
        
        IERC20(asset).approve(address(POOL), amountOwed);
        
        return true;
    }
    
    function _performArbitrage(
        address asset,
        uint256 amount,
        ArbitrageParams memory params
    ) internal returns (uint256) {
        uint256 initialBalance = IERC20(asset).balanceOf(address(this));
        
        IERC20(asset).approve(address(swapRouter), amount);
        
        ISwapRouter.ExactInputSingleParams memory swapParams = ISwapRouter.ExactInputSingleParams({
            tokenIn: asset,
            tokenOut: params.tokenOut,
            fee: params.fee,
            recipient: address(this),
            deadline: block.timestamp + 300,
            amountIn: amount,
            amountOutMinimum: params.minAmountOut,
            sqrtPriceLimitX96: 0
        });
        
        uint256 amountOut = swapRouter.exactInputSingle(swapParams);
        
        IERC20(params.tokenOut).approve(address(swapRouter), amountOut);
        
        ISwapRouter.ExactInputSingleParams memory reverseParams = ISwapRouter.ExactInputSingleParams({
            tokenIn: params.tokenOut,
            tokenOut: asset,
            fee: params.fee,
            recipient: address(this),
            deadline: block.timestamp + 300,
            amountIn: amountOut,
            amountOutMinimum: amount,
            sqrtPriceLimitX96: 0
        });
        
        swapRouter.exactInputSingle(reverseParams);
        
        uint256 finalBalance = IERC20(asset).balanceOf(address(this));
        
        return finalBalance > initialBalance ? finalBalance - initialBalance : 0;
    }
    
    function withdrawToken(address token, uint256 amount) external onlyOwner {
        IERC20(token).transfer(owner, amount);
    }
    
    function withdrawETH() external onlyOwner {
        payable(owner).transfer(address(this).balance);
    }
    
    receive() external payable {}
}