// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title TestnetFlashLoanSimulator
 * @dev Safe flash loan simulator for testnet testing
 * No real money at risk - perfect for learning!
 */
contract TestnetFlashLoanSimulator {
    
    mapping(address => uint256) public mockBalances;
    mapping(address => uint256) public totalBorrowed;
    
    uint256 public constant FLASH_LOAN_FEE = 9; // 0.09% (same as Aave)
    uint256 public constant MAX_LOAN_AMOUNT = 1000 * 10**18; // 1000 ETH max
    
    event FlashLoanSimulated(
        address indexed borrower,
        address indexed asset,
        uint256 amount,
        uint256 fee,
        bool success,
        string strategy
    );
    
    event FaucetUsed(address indexed user, uint256 amount);
    
    /**
     * @dev Simulate a flash loan (safe for testing)
     */
    function simulateFlashLoan(
        address asset,
        uint256 amount,
        bytes calldata params,
        string calldata strategy
    ) external {
        require(amount <= MAX_LOAN_AMOUNT, "Amount too large for testnet");
        
        // Calculate fee (same as real Aave)
        uint256 fee = (amount * FLASH_LOAN_FEE) / 10000;
        
        // Record the "loan"
        mockBalances[msg.sender] += amount;
        totalBorrowed[msg.sender] += amount;
        
        // Simulate callback execution
        bool success = _simulateArbitrageCallback(asset, amount, fee, params);
        
        // Simulate repayment
        if (success) {
            mockBalances[msg.sender] -= (amount + fee);
        }
        
        emit FlashLoanSimulated(
            msg.sender, 
            asset, 
            amount, 
            fee, 
            success, 
            strategy
        );
    }
    
    /**
     * @dev Simulate arbitrage execution logic
     */
    function _simulateArbitrageCallback(
        address asset,
        uint256 amount,
        uint256 fee,
        bytes calldata params
    ) private pure returns (bool) {
        // Decode arbitrage parameters
        (address dexA, address dexB, uint256 expectedProfit) = 
            abi.decode(params, (address, address, uint256));
        
        // Simulate arbitrage success based on profit expectations
        // In real testing, this would call actual DEX contracts
        return expectedProfit > fee;
    }
    
    /**
     * @dev Testnet faucet - get free tokens for testing!
     */
    function getTestTokens() external {
        uint256 faucetAmount = 10 * 10**18; // 10 test tokens
        mockBalances[msg.sender] += faucetAmount;
        
        emit FaucetUsed(msg.sender, faucetAmount);
    }
    
    /**
     * @dev Check your test token balance
     */
    function getBalance() external view returns (uint256) {
        return mockBalances[msg.sender];
    }
    
    /**
     * @dev Get total amount ever borrowed (for statistics)
     */
    function getTotalBorrowed() external view returns (uint256) {
        return totalBorrowed[msg.sender];
    }
    
    /**
     * @dev Reset your test account (for fresh testing)
     */
    function resetAccount() external {
        mockBalances[msg.sender] = 0;
        totalBorrowed[msg.sender] = 0;
    }
    
    /**
     * @dev Get contract statistics
     */
    function getStats() external view returns (
        uint256 maxLoanAmount,
        uint256 feeRate,
        string memory network
    ) {
        return (MAX_LOAN_AMOUNT, FLASH_LOAN_FEE, "TESTNET");
    }
}
