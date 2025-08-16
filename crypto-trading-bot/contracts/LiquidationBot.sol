pragma solidity ^0.8.19;

import "./interfaces/IAave.sol";
import "./interfaces/IFlashLoanReceiver.sol";
import "./libraries/TransferHelper.sol";

contract LiquidationBot is IFlashLoanReceiver {
    address private constant AAVE_POOL = 0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2;
    address private constant AAVE_ORACLE = 0x54586bE62E3c3580375aE3723C145253060Ca0C2;
    address private immutable owner;
    
    struct LiquidationParams {
        address collateralAsset;
        address debtAsset;
        address user;
        uint256 debtToCover;
        bool receiveAToken;
    }
    
    struct CompoundLiquidation {
        address cToken;
        address borrower;
        uint256 repayAmount;
        address cTokenCollateral;
    }
    
    mapping(address => bool) public authorizedCallers;
    
    modifier onlyAuthorized() {
        require(msg.sender == owner || authorizedCallers[msg.sender]);
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    function liquidateAave(LiquidationParams calldata params) external onlyAuthorized {
        uint256 debtBalance = IERC20(params.debtAsset).balanceOf(address(this));
        
        if (debtBalance < params.debtToCover) {
            _flashLoanForLiquidation(params);
        } else {
            _executeLiquidation(params);
        }
    }
    
    function _flashLoanForLiquidation(LiquidationParams memory params) private {
        address[] memory assets = new address[](1);
        assets[0] = params.debtAsset;
        
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = params.debtToCover;
        
        uint256[] memory modes = new uint256[](1);
        modes[0] = 0;
        
        bytes memory encodedParams = abi.encode(params);
        
        IAaveLendingPool(AAVE_POOL).flashLoan(
            address(this),
            assets,
            amounts,
            modes,
            address(this),
            encodedParams,
            0
        );
    }
    
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        require(msg.sender == AAVE_POOL);
        require(initiator == address(this));
        
        LiquidationParams memory liquidationParams = abi.decode(params, (LiquidationParams));
        
        _executeLiquidation(liquidationParams);
        
        uint256 profit = IERC20(liquidationParams.collateralAsset).balanceOf(address(this));
        
        if (profit > 0) {
            _swapCollateralToDebt(
                liquidationParams.collateralAsset,
                liquidationParams.debtAsset,
                profit
            );
        }
        
        for (uint256 i = 0; i < assets.length; i++) {
            uint256 amountOwing = amounts[i] + premiums[i];
            IERC20(assets[i]).approve(AAVE_POOL, amountOwing);
        }
        
        return true;
    }
    
    function _executeLiquidation(LiquidationParams memory params) private {
        IERC20(params.debtAsset).approve(AAVE_POOL, params.debtToCover);
        
        IAaveLendingPool(AAVE_POOL).liquidationCall(
            params.collateralAsset,
            params.debtAsset,
            params.user,
            params.debtToCover,
            params.receiveAToken
        );
    }
    
    function _swapCollateralToDebt(
        address collateral,
        address debt,
        uint256 amount
    ) private {
        bytes memory swapData = abi.encodeWithSignature(
            "swap(address,address,uint256)",
            collateral,
            debt,
            amount
        );
        
        (bool success,) = address(0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45).call(swapData);
        require(success);
    }
    
    function liquidateCompound(CompoundLiquidation calldata params) external onlyAuthorized {
        IERC20 underlying = IERC20(ICToken(params.cToken).underlying());
        underlying.approve(params.cToken, params.repayAmount);
        
        uint256 result = ICToken(params.cToken).liquidateBorrow(
            params.borrower,
            params.repayAmount,
            params.cTokenCollateral
        );
        
        require(result == 0);
        
        uint256 seizedTokens = ICToken(params.cTokenCollateral).balanceOf(address(this));
        if (seizedTokens > 0) {
            ICToken(params.cTokenCollateral).redeem(seizedTokens);
        }
    }
    
    function batchLiquidate(LiquidationParams[] calldata liquidations) external onlyAuthorized {
        for (uint256 i = 0; i < liquidations.length; i++) {
            try this.liquidateAave(liquidations[i]) {
            } catch {
                continue;
            }
        }
    }
    
    function checkProfitability(
        address collateralAsset,
        address debtAsset,
        uint256 debtAmount,
        uint256 bonus
    ) external view returns (uint256) {
        uint256 collateralPrice = IAaveOracle(AAVE_ORACLE).getAssetPrice(collateralAsset);
        uint256 debtPrice = IAaveOracle(AAVE_ORACLE).getAssetPrice(debtAsset);
        
        uint256 collateralValue = (debtAmount * debtPrice * (10000 + bonus)) / (collateralPrice * 10000);
        uint256 profit = (collateralValue * bonus) / 10000;
        
        return profit;
    }
    
    function setAuthorizedCaller(address caller, bool authorized) external {
        require(msg.sender == owner);
        authorizedCallers[caller] = authorized;
    }
    
    function emergencyWithdraw(address token) external {
        require(msg.sender == owner);
        
        if (token == address(0)) {
            payable(owner).transfer(address(this).balance);
        } else {
            uint256 balance = IERC20(token).balanceOf(address(this));
            TransferHelper.safeTransfer(token, owner, balance);
        }
    }
    
    receive() external payable {}
}

interface ICToken {
    function liquidateBorrow(address borrower, uint256 repayAmount, address cTokenCollateral) external returns (uint256);
    function underlying() external view returns (address);
    function balanceOf(address owner) external view returns (uint256);
    function redeem(uint256 redeemTokens) external returns (uint256);
}

interface IAaveOracle {
    function getAssetPrice(address asset) external view returns (uint256);
}