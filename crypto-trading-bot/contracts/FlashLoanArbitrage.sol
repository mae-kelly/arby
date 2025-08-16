pragma solidity ^0.8.19;

import "./interfaces/IFlashLoanReceiver.sol";
import "./interfaces/IAave.sol";
import "./interfaces/IUniswapV3.sol";
import "./interfaces/IBalancer.sol";
import "./libraries/TransferHelper.sol";

contract FlashLoanArbitrage is IFlashLoanReceiver {
    address private constant AAVE_LENDING_POOL = 0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2;
    address private constant BALANCER_VAULT = 0xBA12222222228d8Ba445958a75a0704d566BF2C8;
    address private immutable owner;
    
    struct ArbitrageParams {
        address[] tokens;
        uint256[] amounts;
        address[] swapPath;
        bytes swapData;
        uint256 expectedProfit;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    function executeArbitrage(
        address flashLoanProvider,
        address asset,
        uint256 amount,
        bytes calldata params
    ) external onlyOwner {
        if (flashLoanProvider == AAVE_LENDING_POOL) {
            address[] memory assets = new address[](1);
            assets[0] = asset;
            uint256[] memory amounts = new uint256[](1);
            amounts[0] = amount;
            uint256[] memory modes = new uint256[](1);
            modes[0] = 0;
            
            IAaveLendingPool(AAVE_LENDING_POOL).flashLoan(
                address(this),
                assets,
                amounts,
                modes,
                address(this),
                params,
                0
            );
        } else if (flashLoanProvider == BALANCER_VAULT) {
            IERC20[] memory tokens = new IERC20[](1);
            tokens[0] = IERC20(asset);
            uint256[] memory amounts = new uint256[](1);
            amounts[0] = amount;
            
            IBalancerVault(BALANCER_VAULT).flashLoan(
                IFlashLoanRecipient(address(this)),
                tokens,
                amounts,
                params
            );
        }
    }
    
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        require(msg.sender == AAVE_LENDING_POOL);
        require(initiator == address(this));
        
        ArbitrageParams memory arbParams = abi.decode(params, (ArbitrageParams));
        
        for (uint256 i = 0; i < arbParams.swapPath.length; i++) {
            address target = arbParams.swapPath[i];
            (bool success,) = target.call(arbParams.swapData);
            require(success);
        }
        
        for (uint256 i = 0; i < assets.length; i++) {
            uint256 amountOwing = amounts[i] + premiums[i];
            IERC20(assets[i]).approve(AAVE_LENDING_POOL, amountOwing);
        }
        
        return true;
    }
    
    function receiveFlashLoan(
        IERC20[] memory tokens,
        uint256[] memory amounts,
        uint256[] memory feeAmounts,
        bytes memory userData
    ) external override {
        require(msg.sender == BALANCER_VAULT);
        
        ArbitrageParams memory arbParams = abi.decode(userData, (ArbitrageParams));
        
        for (uint256 i = 0; i < arbParams.swapPath.length; i++) {
            address target = arbParams.swapPath[i];
            (bool success,) = target.call(arbParams.swapData);
            require(success);
        }
        
        for (uint256 i = 0; i < tokens.length; i++) {
            TransferHelper.safeTransfer(
                address(tokens[i]),
                BALANCER_VAULT,
                amounts[i] + feeAmounts[i]
            );
        }
    }
    
    function multiDexSwap(
        address[] calldata routers,
        bytes[] calldata swapCalldata,
        address[] calldata tokens,
        uint256[] calldata minAmounts
    ) external onlyOwner {
        for (uint256 i = 0; i < routers.length; i++) {
            (bool success, bytes memory result) = routers[i].call(swapCalldata[i]);
            require(success);
            
            if (i < tokens.length && i < minAmounts.length) {
                uint256 balance = IERC20(tokens[i]).balanceOf(address(this));
                require(balance >= minAmounts[i]);
            }
        }
    }
    
    function emergencyWithdraw(address token) external onlyOwner {
        if (token == address(0)) {
            payable(owner).transfer(address(this).balance);
        } else {
            uint256 balance = IERC20(token).balanceOf(address(this));
            TransferHelper.safeTransfer(token, owner, balance);
        }
    }
    
    receive() external payable {}
}