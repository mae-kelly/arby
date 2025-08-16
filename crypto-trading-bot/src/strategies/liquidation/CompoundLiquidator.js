const BaseStrategy = require('../base/BaseStrategy');
const { ethers } = require('ethers');

class CompoundLiquidator extends BaseStrategy {
    constructor(config, web3Provider, comptroller, priceOracle, cTokens, flashLoanProvider) {
        super(config);
        this.provider = web3Provider;
        this.comptroller = comptroller;
        this.priceOracle = priceOracle;
        this.cTokens = cTokens;
        this.flashLoanProvider = flashLoanProvider;
        this.liquidationContract = null;
        this.closeFactorMantissa = ethers.utils.parseEther('0.5');
        this.liquidationIncentiveMantissa = ethers.utils.parseEther('1.08');
        this.monitoredAccounts = new Set();
    }

    async initialize() {
        const contractFactory = new ethers.ContractFactory(
            this.config.liquidationABI,
            this.config.liquidationBytecode,
            this.provider.getSigner()
        );
        this.liquidationContract = await contractFactory.deploy(this.comptroller.address);
        await this.liquidationContract.deployed();
        
        await this.updateCloseFactorAndIncentive();
        await this.startAccountMonitoring();
        await super.initialize();
    }

    async calculateProfit(opportunity) {
        const { borrower, cTokenBorrowed, cTokenCollateral, repayAmount } = opportunity;
        
        const borrowedPrice = await this.priceOracle.getUnderlyingPrice(cTokenBorrowed.address);
        const collateralPrice = await this.priceOracle.getUnderlyingPrice(cTokenCollateral.address);
        const exchangeRate = await cTokenCollateral.exchangeRateStored();
        
        const repayAmountUSD = repayAmount.mul(borrowedPrice).div(ethers.utils.parseEther('1'));
        
        const seizeTokens = repayAmountUSD
            .mul(this.liquidationIncentiveMantissa)
            .div(collateralPrice)
            .mul(ethers.utils.parseEther('1'))
            .div(exchangeRate);
        
        const collateralValueUSD = seizeTokens
            .mul(exchangeRate)
            .mul(collateralPrice)
            .div(ethers.utils.parseEther('1'))
            .div(ethers.utils.parseEther('1'));
        
        const profit = collateralValueUSD.sub(repayAmountUSD);
        
        const flashLoanFee = repayAmount.mul(5).div(10000);
        const gasEstimate = await this.estimateGas(opportunity);
        
        return profit.sub(flashLoanFee).sub(gasEstimate);
    }

    async estimateGas(opportunity) {
        try {
            const gasEstimate = await this.liquidationContract.estimateGas.liquidateBorrow(
                opportunity.borrower,
                opportunity.cTokenBorrowed.address,
                opportunity.repayAmount,
                opportunity.cTokenCollateral.address
            );
            return gasEstimate.mul(ethers.utils.parseUnits('30', 'gwei'));
        } catch {
            return ethers.utils.parseUnits('500000', 'gwei');
        }
    }

    async executeStrategy(opportunity) {
        const startTime = Date.now();
        const { borrower, cTokenBorrowed, cTokenCollateral, repayAmount } = opportunity;
        
        try {
            const gasPrice = await this.provider.getGasPrice();
            const adjustedGasPrice = gasPrice.mul(115).div(100);
            
            const tx = await this.liquidationContract.liquidateBorrow(
                borrower,
                cTokenBorrowed.address,
                repayAmount,
                cTokenCollateral.address,
                {
                    gasPrice: adjustedGasPrice,
                    gasLimit: 800000
                }
            );
            
            const receipt = await tx.wait();
            const gasCost = receipt.gasUsed.mul(adjustedGasPrice);
            
            const liquidationEvent = receipt.logs.find(log =>
                log.topics[0] === ethers.utils.id('LiquidateBorrow(address,address,uint256,address,uint256)')
            );
            
            let profit = ethers.constants.Zero;
            if (liquidationEvent) {
                const decoded = ethers.utils.defaultAbiCoder.decode(
                    ['address', 'address', 'uint256', 'address', 'uint256'],
                    liquidationEvent.data
                );
                const seizeTokens = decoded[4];
                
                const collateralPrice = await this.priceOracle.getUnderlyingPrice(cTokenCollateral.address);
                const borrowedPrice = await this.priceOracle.getUnderlyingPrice(cTokenBorrowed.address);
                const exchangeRate = await cTokenCollateral.exchangeRateStored();
                
                const collateralValue = seizeTokens.mul(exchangeRate).mul(collateralPrice).div(ethers.utils.parseEther('1')).div(ethers.utils.parseEther('1'));
                const debtValue = repayAmount.mul(borrowedPrice).div(ethers.utils.parseEther('1'));
                
                profit = collateralValue.sub(debtValue);
            }
            
            return {
                success: receipt.status === 1,
                txHash: receipt.transactionHash,
                profit: ethers.utils.formatEther(profit),
                gasCost: ethers.utils.formatEther(gasCost),
                executionTime: Date.now() - startTime,
                liquidatedBorrower: borrower
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                profit: 0,
                gasCost: 0,
                executionTime: Date.now() - startTime
            };
        }
    }

    async findOpportunities() {
        const opportunities = [];
        const liquidatableAccounts = await this.findLiquidatableAccounts();
        
        for (const account of liquidatableAccounts) {
            try {
                const liquidity = await this.comptroller.getAccountLiquidity(account);
                
                if (liquidity[2].gt(0)) {
                    const accountAssets = await this.comptroller.getAssetsIn(account);
                    
                    for (const cTokenAddress of accountAssets) {
                        const cToken = this.cTokens[cTokenAddress];
                        if (!cToken) continue;
                        
                        const borrowBalance = await cToken.borrowBalanceStored(account);
                        if (borrowBalance.eq(0)) continue;
                        
                        const maxClose = borrowBalance.mul(this.closeFactorMantissa).div(ethers.utils.parseEther('1'));
                        
                        for (const [collateralAddress, collateralToken] of Object.entries(this.cTokens)) {
                            if (collateralAddress === cTokenAddress) continue;
                            
                            const collateralBalance = await collateralToken.balanceOf(account);
                            if (collateralBalance.eq(0)) continue;
                            
                            const profitEstimate = await this.calculateProfit({
                                borrower: account,
                                cTokenBorrowed: cToken,
                                cTokenCollateral: collateralToken,
                                repayAmount: maxClose
                            });
                            
                            if (profitEstimate.gt(ethers.utils.parseEther(this.minProfitThreshold.toString()))) {
                                opportunities.push({
                                    borrower: account,
                                    cTokenBorrowed: cToken,
                                    cTokenCollateral: collateralToken,
                                    repayAmount: maxClose,
                                    shortfall: liquidity[2],
                                    estimatedProfit: profitEstimate,
                                    priority: this.calculatePriority(liquidity[2], profitEstimate)
                                });
                            }
                        }
                    }
                }
            } catch (error) {
                continue;
            }
        }
        
        return opportunities.sort((a, b) => b.priority.sub(a.priority));
    }

    async findLiquidatableAccounts() {
        const liquidatableAccounts = [];
        const latestBlock = await this.provider.getBlockNumber();
        const fromBlock = latestBlock - 2000;
        
        const borrowEvents = await this.comptroller.queryFilter(
            this.comptroller.filters.MarketEntered(),
            fromBlock,
            latestBlock
        );
        
        const allAccounts = new Set([
            ...borrowEvents.map(event => event.args.account),
            ...Array.from(this.monitoredAccounts)
        ]);
        
        for (const account of allAccounts) {
            try {
                const liquidity = await this.comptroller.getAccountLiquidity(account);
                if (liquidity[2].gt(0)) {
                    liquidatableAccounts.push(account);
                    this.monitoredAccounts.add(account);
                }
            } catch (error) {
                continue;
            }
        }
        
        return liquidatableAccounts;
    }

    calculatePriority(shortfall, estimatedProfit) {
        const shortfallScore = shortfall.div(ethers.utils.parseEther('1000'));
        const profitScore = estimatedProfit.mul(10);
        return shortfallScore.add(profitScore);
    }

    async updateCloseFactorAndIncentive() {
        try {
            this.closeFactorMantissa = await this.comptroller.closeFactorMantissa();
            this.liquidationIncentiveMantissa = await this.comptroller.liquidationIncentiveMantissa();
        } catch (error) {
            this.emit('error', `Failed to update close factor and incentive: ${error.message}`);
        }
    }

    async startAccountMonitoring() {
        setInterval(async () => {
            await this.monitorAccountLiquidity();
        }, 15000);
        
        this.comptroller.on('MarketEntered', (cToken, account) => {
            this.monitoredAccounts.add(account);
        });
        
        for (const [address, cToken] of Object.entries(this.cTokens)) {
            cToken.on('Borrow', (borrower, borrowAmount, accountBorrows, totalBorrows) => {
                this.monitoredAccounts.add(borrower);
            });
        }
    }

    async monitorAccountLiquidity() {
        const accountsToRemove = new Set();
        
        for (const account of this.monitoredAccounts) {
            try {
                const liquidity = await this.comptroller.getAccountLiquidity(account);
                const assets = await this.comptroller.getAssetsIn(account);
                
                if (assets.length === 0) {
                    accountsToRemove.add(account);
                    continue;
                }
                
                if (liquidity[2].gt(0)) {
                    this.emit('liquidationOpportunity', {
                        account,
                        shortfall: liquidity[2],
                        liquidity: liquidity[1],
                        assets,
                        timestamp: Date.now()
                    });
                }
                
                if (liquidity[1].gt(ethers.utils.parseEther('10000')) && liquidity[2].eq(0)) {
                    accountsToRemove.add(account);
                }
            } catch (error) {
                accountsToRemove.add(account);
            }
        }
        
        for (const account of accountsToRemove) {
            this.monitoredAccounts.delete(account);
        }
    }

    async getOptimalRepayAmount(borrower, cTokenBorrowed, cTokenCollateral) {
        const borrowBalance = await cTokenBorrowed.borrowBalanceStored(borrower);
        const maxRepay = borrowBalance.mul(this.closeFactorMantissa).div(ethers.utils.parseEther('1'));
        
        const collateralBalance = await cTokenCollateral.balanceOf(borrower);
        const exchangeRate = await cTokenCollateral.exchangeRateStored();
        const collateralPrice = await this.priceOracle.getUnderlyingPrice(cTokenCollateral.address);
        const borrowedPrice = await this.priceOracle.getUnderlyingPrice(cTokenBorrowed.address);
        
        const maxCollateralValue = collateralBalance
            .mul(exchangeRate)
            .mul(collateralPrice)
            .div(ethers.utils.parseEther('1'))
            .div(ethers.utils.parseEther('1'));
        
        const maxRepayFromCollateral = maxCollateralValue
            .mul(ethers.utils.parseEther('1'))
            .div(this.liquidationIncentiveMantissa)
            .mul(ethers.utils.parseEther('1'))
            .div(borrowedPrice);
        
        return maxRepay.lt(maxRepayFromCollateral) ? maxRepay : maxRepayFromCollateral;
    }

    async simulateLiquidation(borrower, cTokenBorrowed, cTokenCollateral, repayAmount) {
        try {
            const liquidityBefore = await this.comptroller.getAccountLiquidity(borrower);
            
            const simulation = await this.liquidationContract.callStatic.liquidateBorrow(
                borrower,
                cTokenBorrowed.address,
                repayAmount,
                cTokenCollateral.address
            );
            
            return {
                success: true,
                shortfallBefore: liquidityBefore[2],
                seizeTokens: simulation.seizeTokens,
                actualRepayAmount: simulation.repayAmount
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    async calculateSeizeTokens(cTokenBorrowed, cTokenCollateral, repayAmount) {
        const borrowedPrice = await this.priceOracle.getUnderlyingPrice(cTokenBorrowed.address);
        const collateralPrice = await this.priceOracle.getUnderlyingPrice(cTokenCollateral.address);
        const exchangeRate = await cTokenCollateral.exchangeRateStored();
        
        const repayAmountUSD = repayAmount.mul(borrowedPrice).div(ethers.utils.parseEther('1'));
        
        const seizeTokens = repayAmountUSD
            .mul(this.liquidationIncentiveMantissa)
            .div(collateralPrice)
            .mul(ethers.utils.parseEther('1'))
            .div(exchangeRate);
        
        return seizeTokens;
    }

    async batchLiquidate(opportunities) {
        const results = [];
        const maxConcurrent = 2;
        
        for (let i = 0; i < opportunities.length; i += maxConcurrent) {
            const batch = opportunities.slice(i, i + maxConcurrent);
            const batchPromises = batch.map(opportunity => this.execute(opportunity));
            
            const batchResults = await Promise.allSettled(batchPromises);
            results.push(...batchResults.map(result => result.value || result.reason));
            
            if (i + maxConcurrent < opportunities.length) {
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
        
        return results;
    }

    async checkLiquidationViability(borrower, cTokenBorrowed, cTokenCollateral, repayAmount) {
        try {
            const borrowBalance = await cTokenBorrowed.borrowBalanceStored(borrower);
            const collateralBalance = await cTokenCollateral.balanceOf(borrower);
            const liquidity = await this.comptroller.getAccountLiquidity(borrower);
            
            if (liquidity[2].eq(0)) return { viable: false, reason: 'Account not underwater' };
            if (borrowBalance.eq(0)) return { viable: false, reason: 'No borrow balance' };
            if (collateralBalance.eq(0)) return { viable: false, reason: 'No collateral balance' };
            
            const maxRepay = borrowBalance.mul(this.closeFactorMantissa).div(ethers.utils.parseEther('1'));
            if (repayAmount.gt(maxRepay)) return { viable: false, reason: 'Repay amount exceeds close factor' };
            
            const seizeTokens = await this.calculateSeizeTokens(cTokenBorrowed, cTokenCollateral, repayAmount);
            if (seizeTokens.gt(collateralBalance)) return { viable: false, reason: 'Insufficient collateral to seize' };
            
            return { viable: true };
        } catch (error) {
            return { viable: false, reason: error.message };
        }
    }
}

module.exports = CompoundLiquidator;