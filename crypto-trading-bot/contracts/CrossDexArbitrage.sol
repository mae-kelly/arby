pragma solidity ^0.8.19;

import "./interfaces/IUniswapV3.sol";
import "./libraries/TransferHelper.sol";

contract CrossDexArbitrage {
    address private immutable owner;
    uint256 private constant MAX_SLIPPAGE = 300;
    
    struct SwapRoute {
        address tokenIn;
        address tokenOut;
        uint256 amountIn;
        address[] dexRouters;
        bytes[] swapData;
        uint256 minAmountOut;
    }
    
    struct TriangularArbParams {
        address tokenA;
        address tokenB;
        address tokenC;
        uint256 amountIn;
        address[] routers;
        uint24[] fees;
        uint256 minProfit;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    function executeCrossArbitrage(SwapRoute calldata route) external onlyOwner returns (uint256) {
        TransferHelper.safeTransferFrom(
            route.tokenIn,
            msg.sender,
            address(this),
            route.amountIn
        );
        
        uint256 currentAmount = route.amountIn;
        address currentToken = route.tokenIn;
        
        for (uint256 i = 0; i < route.dexRouters.length; i++) {
            TransferHelper.safeApprove(currentToken, route.dexRouters[i], currentAmount);
            
            (bool success, bytes memory result) = route.dexRouters[i].call(route.swapData[i]);
            require(success);
            
            if (i == route.dexRouters.length - 1) {
                currentToken = route.tokenOut;
                currentAmount = IERC20(route.tokenOut).balanceOf(address(this));
            }
        }
        
        require(currentAmount >= route.minAmountOut);
        
        TransferHelper.safeTransfer(route.tokenOut, owner, currentAmount);
        
        return currentAmount;
    }
    
    function executeTriangularArbitrage(TriangularArbParams calldata params) external onlyOwner returns (uint256) {
        TransferHelper.safeTransferFrom(
            params.tokenA,
            msg.sender,
            address(this),
            params.amountIn
        );
        
        uint256 amountB = _swap(
            params.tokenA,
            params.tokenB,
            params.amountIn,
            params.routers[0],
            params.fees[0]
        );
        
        uint256 amountC = _swap(
            params.tokenB,
            params.tokenC,
            amountB,
            params.routers[1],
            params.fees[1]
        );
        
        uint256 finalAmount = _swap(
            params.tokenC,
            params.tokenA,
            amountC,
            params.routers[2],
            params.fees[2]
        );
        
        require(finalAmount > params.amountIn + params.minProfit);
        
        TransferHelper.safeTransfer(params.tokenA, owner, finalAmount);
        
        return finalAmount - params.amountIn;
    }
    
    function _swap(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        address router,
        uint24 fee
    ) private returns (uint256) {
        TransferHelper.safeApprove(tokenIn, router, amountIn);
        
        ISwapRouter.ExactInputSingleParams memory params = ISwapRouter.ExactInputSingleParams({
            tokenIn: tokenIn,
            tokenOut: tokenOut,
            fee: fee,
            recipient: address(this),
            deadline: block.timestamp,
            amountIn: amountIn,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });
        
        uint256 amountOut = ISwapRouter(router).exactInputSingle(params);
        return amountOut;
    }
    
    function batchArbitrage(SwapRoute[] calldata routes) external onlyOwner {
        for (uint256 i = 0; i < routes.length; i++) {
            try this.executeCrossArbitrage(routes[i]) returns (uint256) {
            } catch {
                continue;
            }
        }
    }
    
    function calculateOptimalAmount(
        address tokenA,
        address tokenB,
        address router,
        uint256 reserveA,
        uint256 reserveB
    ) external pure returns (uint256) {
        uint256 optimal = sqrt(reserveA * reserveB * 997 * 1000) - (reserveA * 1000);
        return optimal / 997;
    }
    
    function sqrt(uint256 x) private pure returns (uint256 y) {
        uint256 z = (x + 1) / 2;
        y = x;
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
    }
    
    function withdrawToken(address token, uint256 amount) external onlyOwner {
        TransferHelper.safeTransfer(token, owner, amount);
    }
    
    function withdrawETH() external onlyOwner {
        payable(owner).transfer(address(this).balance);
    }
    
    receive() external payable {}
}