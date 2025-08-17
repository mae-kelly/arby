// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@aave/core-v3/contracts/flashloan/base/FlashLoanSimpleReceiverBase.sol";
import "@aave/core-v3/contracts/interfaces/IPoolAddressesProvider.sol";
import "@balancer-labs/v2-interfaces/contracts/vault/IVault.sol";
import "@balancer-labs/v2-interfaces/contracts/vault/IFlashLoanRecipient.sol";
import "@uniswap/v3-core/contracts/interfaces/callback/IUniswapV3FlashCallback.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";

interface IDyDxCallee {
    function callFunction(
        address sender,
        Account.Info memory accountInfo,
        bytes memory data
    ) external;
}

interface ISoloMargin {
    function operate(
        Account.Info[] memory accounts,
        Actions.ActionArgs[] memory actions
    ) external;
}

library Account {
    struct Info {
        address owner;
        uint256 number;
    }
}

library Actions {
    enum ActionType {
        Deposit,
        Withdraw,
        Transfer,
        Buy,
        Sell,
        Trade,
        Liquidate,
        Vaporize,
        Call
    }
    
    struct ActionArgs {
        ActionType actionType;
        uint256 accountId;
        Types.AssetAmount amount;
        uint256 primaryMarketId;
        uint256 secondaryMarketId;
        address otherAddress;
        uint256 otherAccountId;
        bytes data;
    }
}

library Types {
    enum AssetDenomination {
        Wei,
        Par
    }
    
    enum AssetReference {
        Delta,
        Target
    }
    
    struct AssetAmount {
        bool sign;
        AssetDenomination denomination;
        AssetReference ref;
        uint256 value;
    }
}

contract UniversalFlashLoan is 
    FlashLoanSimpleReceiverBase,
    IFlashLoanRecipient,
    IUniswapV3FlashCallback,
    IDyDxCallee 
{
    address private constant BALANCER_VAULT = 0xBA12222222228d8Ba445958a75a0704d566BF2C8;
    address private constant DYDX_SOLO_MARGIN = 0x1E0447b19BB6EcFdAe1e4AE1694b0C3659614e4e;
    address private constant MAKER_FLASH_LENDER = 0x60744434d6339a6B27d73d9Eda62b6F66a0a04FA;
    
    address private immutable owner;
    mapping(address => bool) private authorized;
    mapping(bytes32 => bool) private activeLoans;
    
    uint256 private constant MAX_SLIPPAGE = 300; // 3%
    uint256 private constant FLASH_LOAN_FEE_BASIS = 9; // 0.09%
    
    enum FlashLoanProvider {
        AAVE,
        BALANCER,
        UNISWAP_V3,
        DYDX,
        MAKER
    }
    
    struct FlashLoanParams {
        FlashLoanProvider provider;
        address[] assets;
        uint256[] amounts;
        bytes userData;
    }
    
    struct ArbitrageParams {
        address[] tokens;
        address[] routers;
        bytes[] swapData;
        uint256[] minAmountsOut;
    }
    
    event FlashLoanExecuted(
        FlashLoanProvider indexed provider,
        address indexed asset,
        uint256 amount,
        uint256 profit
    );
    
    event ArbitrageCompleted(
        address indexed token,
        uint256 profit,
        uint256 gasUsed
    );
    
    modifier onlyAuthorized() {
        require(authorized[msg.sender] || msg.sender == owner, "Unauthorized");
        _;
    }
    
    modifier nonReentrant() {
        bytes32 id = keccak256(abi.encode(block.number, msg.sender));
        require(!activeLoans[id], "Reentrant");
        activeLoans[id] = true;
        _;
        activeLoans[id] = false;
    }
    
    constructor(address _addressProvider) 
        FlashLoanSimpleReceiverBase(IPoolAddressesProvider(_addressProvider)) 
    {
        owner = msg.sender;
        authorized[msg.sender] = true;
    }
    
    function executeFlashLoan(FlashLoanParams calldata params) external onlyAuthorized nonReentrant {
        if (params.provider == FlashLoanProvider.AAVE) {
            _executeAaveFlashLoan(params);
        } else if (params.provider == FlashLoanProvider.BALANCER) {
            _executeBalancerFlashLoan(params);
        } else if (params.provider == FlashLoanProvider.UNISWAP_V3) {
            _executeUniswapV3FlashLoan(params);
        } else if (params.provider == FlashLoanProvider.DYDX) {
            _executeDyDxFlashLoan(params);
        } else if (params.provider == FlashLoanProvider.MAKER) {
            _executeMakerFlashLoan(params);
        } else {
            revert("Invalid provider");
        }
    }
    
    function _executeAaveFlashLoan(FlashLoanParams memory params) private {
        for (uint256 i = 0; i < params.assets.length; i++) {
            POOL.flashLoanSimple(
                address(this),
                params.assets[i],
                params.amounts[i],
                params.userData,
                0
            );
        }
    }
    
    function _executeBalancerFlashLoan(FlashLoanParams memory params) private {
        IVault(BALANCER_VAULT).flashLoan(
            IFlashLoanRecipient(address(this)),
            params.assets,
            params.amounts,
            params.userData
        );
    }
    
    function _executeUniswapV3FlashLoan(FlashLoanParams memory params) private {
        (address pool, uint256 amount0, uint256 amount1) = 
            abi.decode(params.userData, (address, uint256, uint256));
            
        IUniswapV3Pool(pool).flash(
            address(this),
            amount0,
            amount1,
            params.userData
        );
    }
    
    function _executeDyDxFlashLoan(FlashLoanParams memory params) private {
        Account.Info[] memory accounts = new Account.Info[](1);
        accounts[0] = Account.Info({owner: address(this), number: 1});
        
        Actions.ActionArgs[] memory actions = new Actions.ActionArgs[](3);
        
        // Withdraw
        actions[0] = Actions.ActionArgs({
            actionType: Actions.ActionType.Withdraw,
            accountId: 0,
            amount: Types.AssetAmount({
                sign: false,
                denomination: Types.AssetDenomination.Wei,
                ref: Types.AssetReference.Delta,
                value: params.amounts[0]
            }),
            primaryMarketId: 0, // WETH
            secondaryMarketId: 0,
            otherAddress: address(this),
            otherAccountId: 0,
            data: ""
        });
        
        // Call function
        actions[1] = Actions.ActionArgs({
            actionType: Actions.ActionType.Call,
            accountId: 0,
            amount: Types.AssetAmount({
                sign: false,
                denomination: Types.AssetDenomination.Wei,
                ref: Types.AssetReference.Delta,
                value: 0
            }),
            primaryMarketId: 0,
            secondaryMarketId: 0,
            otherAddress: address(this),
            otherAccountId: 0,
            data: params.userData
        });
        
        // Deposit back
        actions[2] = Actions.ActionArgs({
            actionType: Actions.ActionType.Deposit,
            accountId: 0,
            amount: Types.AssetAmount({
                sign: true,
                denomination: Types.AssetDenomination.Wei,
                ref: Types.AssetReference.Delta,
                value: params.amounts[0] + 2 // +2 wei fee
            }),
            primaryMarketId: 0,
            secondaryMarketId: 0,
            otherAddress: address(this),
            otherAccountId: 0,
            data: ""
        });
        
        ISoloMargin(DYDX_SOLO_MARGIN).operate(accounts, actions);
    }
    
    function _executeMakerFlashLoan(FlashLoanParams memory params) private {
        // Maker flash loan implementation
        bytes memory data = abi.encode(params.assets[0], params.amounts[0], params.userData);
        (bool success,) = MAKER_FLASH_LENDER.call(data);
        require(success, "Maker flash loan failed");
    }
    
    // AAVE callback
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        require(msg.sender == address(POOL), "Invalid caller");
        require(initiator == address(this), "Invalid initiator");
        
        uint256 balanceBefore = IERC20(asset).balanceOf(address(this));
        
        // Execute arbitrage
        _executeArbitrage(params);
        
        uint256 balanceAfter = IERC20(asset).balanceOf(address(this));
        uint256 profit = balanceAfter - balanceBefore - premium;
        
        require(profit > 0, "No profit");
        
        // Approve repayment
        IERC20(asset).approve(address(POOL), amount + premium);
        
        emit FlashLoanExecuted(FlashLoanProvider.AAVE, asset, amount, profit);
        
        return true;
    }
    
    // Balancer callback
    function receiveFlashLoan(
        IERC20[] memory tokens,
        uint256[] memory amounts,
        uint256[] memory feeAmounts,
        bytes memory userData
    ) external override {
        require(msg.sender == BALANCER_VAULT, "Invalid caller");
        
        uint256[] memory balancesBefore = new uint256[](tokens.length);
        for (uint256 i = 0; i < tokens.length; i++) {
            balancesBefore[i] = tokens[i].balanceOf(address(this));
        }
        
        // Execute arbitrage
        _executeArbitrage(userData);
        
        // Repay loans
        for (uint256 i = 0; i < tokens.length; i++) {
            uint256 balanceAfter = tokens[i].balanceOf(address(this));
            uint256 profit = balanceAfter - balancesBefore[i] - feeAmounts[i];
            
            require(profit > 0, "No profit");
            
            tokens[i].transfer(BALANCER_VAULT, amounts[i] + feeAmounts[i]);
            
            emit FlashLoanExecuted(
                FlashLoanProvider.BALANCER, 
                address(tokens[i]), 
                amounts[i], 
                profit
            );
        }
    }
    
    // UniswapV3 callback
    function uniswapV3FlashCallback(
        uint256 fee0,
        uint256 fee1,
        bytes calldata data
    ) external override {
        (address pool, uint256 amount0, uint256 amount1, bytes memory userData) = 
            abi.decode(data, (address, uint256, uint256, bytes));
            
        require(msg.sender == pool, "Invalid caller");
        
        // Execute arbitrage
        _executeArbitrage(userData);
        
        // Repay with fees
        if (amount0 > 0) {
            IERC20(IUniswapV3Pool(pool).token0()).transfer(pool, amount0 + fee0);
        }
        if (amount1 > 0) {
            IERC20(IUniswapV3Pool(pool).token1()).transfer(pool, amount1 + fee1);
        }
        
        emit FlashLoanExecuted(FlashLoanProvider.UNISWAP_V3, pool, amount0 + amount1, 0);
    }
    
    // DyDx callback
    function callFunction(
        address sender,
        Account.Info memory accountInfo,
        bytes memory data
    ) external override {
        require(msg.sender == DYDX_SOLO_MARGIN, "Invalid caller");
        require(sender == address(this), "Invalid sender");
        
        // Execute arbitrage
        _executeArbitrage(data);
    }
    
    function _executeArbitrage(bytes memory data) private {
        ArbitrageParams memory params = abi.decode(data, (ArbitrageParams));
        
        uint256 gasStart = gasleft();
        
        for (uint256 i = 0; i < params.routers.length; i++) {
            // Approve router
            IERC20(params.tokens[i]).approve(params.routers[i], type(uint256).max);
            
            // Execute swap
            (bool success, bytes memory result) = params.routers[i].call(params.swapData[i]);
            require(success, "Swap failed");
            
            // Verify output
            uint256 outputAmount = abi.decode(result, (uint256));
            require(outputAmount >= params.minAmountsOut[i], "Insufficient output");
        }
        
        uint256 gasUsed = gasStart - gasleft();
        
        emit ArbitrageCompleted(params.tokens[0], 0, gasUsed);
    }
    
    function recoverToken(address token, uint256 amount) external onlyAuthorized {
        if (token == address(0)) {
            payable(owner).transfer(amount);
        } else {
            IERC20(token).transfer(owner, amount);
        }
    }
    
    function setAuthorized(address user, bool status) external {
        require(msg.sender == owner, "Only owner");
        authorized[user] = status;
    }
    
    receive() external payable {}
}