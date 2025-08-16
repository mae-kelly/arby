pragma solidity ^0.8.19;

import "./IFlashLoanReceiver.sol";

interface IBalancerVault {
    function flashLoan(
        IFlashLoanRecipient recipient,
        IERC20[] memory tokens,
        uint256[] memory amounts,
        bytes memory userData
    ) external;
    
    enum SwapKind { GIVEN_IN, GIVEN_OUT }
    
    struct SingleSwap {
        bytes32 poolId;
        SwapKind kind;
        IAsset assetIn;
        IAsset assetOut;
        uint256 amount;
        bytes userData;
    }
    
    struct FundManagement {
        address sender;
        bool fromInternalBalance;
        address payable recipient;
        bool toInternalBalance;
    }
    
    function swap(
        SingleSwap memory singleSwap,
        FundManagement memory funds,
        uint256 limit,
        uint256 deadline
    ) external returns (uint256);
    
    struct BatchSwapStep {
        bytes32 poolId;
        uint256 assetInIndex;
        uint256 assetOutIndex;
        uint256 amount;
        bytes userData;
    }
    
    function batchSwap(
        SwapKind kind,
        BatchSwapStep[] memory swaps,
        IAsset[] memory assets,
        FundManagement memory funds,
        int256[] memory limits,
        uint256 deadline
    ) external payable returns (int256[] memory);
    
    function queryBatchSwap(
        SwapKind kind,
        BatchSwapStep[] memory swaps,
        IAsset[] memory assets,
        FundManagement memory funds
    ) external returns (int256[] memory assetDeltas);
    
    function getPool(bytes32 poolId) external view returns (address, PoolSpecialization);
    
    function getPoolTokens(bytes32 poolId) external view returns (
        IERC20[] memory tokens,
        uint256[] memory balances,
        uint256 lastChangeBlock
    );
    
    enum PoolSpecialization { GENERAL, MINIMAL_SWAP_INFO, TWO_TOKEN }
    
    struct JoinPoolRequest {
        IAsset[] assets;
        uint256[] maxAmountsIn;
        bytes userData;
        bool fromInternalBalance;
    }
    
    function joinPool(
        bytes32 poolId,
        address sender,
        address recipient,
        JoinPoolRequest memory request
    ) external payable;
    
    struct ExitPoolRequest {
        IAsset[] assets;
        uint256[] minAmountsOut;
        bytes userData;
        bool toInternalBalance;
    }
    
    function exitPool(
        bytes32 poolId,
        address sender,
        address payable recipient,
        ExitPoolRequest memory request
    ) external;
    
    function getInternalBalance(address user, IERC20[] memory tokens) external view returns (uint256[] memory);
    
    function manageUserBalance(UserBalanceOp[] memory ops) external payable;
    
    struct UserBalanceOp {
        UserBalanceOpKind kind;
        IAsset asset;
        uint256 amount;
        address sender;
        address payable recipient;
    }
    
    enum UserBalanceOpKind { DEPOSIT_INTERNAL, WITHDRAW_INTERNAL, TRANSFER_INTERNAL, TRANSFER_EXTERNAL }
}

interface IAsset {
}

interface IBalancerPool {
    function getPoolId() external view returns (bytes32);
    function getVault() external view returns (address);
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function getSwapFeePercentage() external view returns (uint256);
}