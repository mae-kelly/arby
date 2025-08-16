const BaseStrategy = require('../base/BaseStrategy');
const { ethers } = require('ethers');

class AaveLiquidator extends BaseStrategy {
    constructor(config, web3Provider, aavePool, priceOracle, flashLoanProvider) {
        super(config);
        this.provider = web3Provider;
        this.aavePool = aavePool;
        this.priceOracle = priceOracle;
        this.flashLoanProvider = flashLoanProvider;
        this.liquidationContract = null;
        this.healthFactorThreshold = 1.0;
        this.monitoredUsers = new Set();
    }

    async initialize() {
        const contractFactory = new ethers.ContractFactory(
            this.config.liquidationABI,
            this.config.liquidationBytecode,
            this.provider.getSigner()
        );
        this.liquidationContract = await contractFactory.deploy(this.aavePool.address);
        await this.liquidationContract.deployed();
        
        await this.startHealthFactorMonitoring();
        await super.initialize();
    }

    async calculateProfit(opportunity) {
        const { user, collateralAsset, debtAsset, debtToCover } = opportunity;
        
        const userReserveData = await this.aavePool.getUserAccountData(user);
        const reserveData = await this.aavePool.getReserveData(collateralAsset);
        
        const liquidationBonus = reserveData.liquidationBonus / 10000;
        const collateralPrice = await this.priceOracle.getAssetPrice(collateralAsset);
        const debtPrice = await this.priceOracle.getAssetPrice(debtAsset);
        
        const collateralToReceive = (debtToCover * debtPrice / collateralPrice) * (1 + liquidationBonus);
        const profit = collateralToReceive - debtToCover * debtPrice / collateralPrice;
        
        const flashLoanFee = debtToCover * 0.0005;
        const gasEstimate = await this.estimateGas(opportunity);
        
        return profit - flashLoanFee - gasEstimate;
    }

    async estimateGas(opportunity) {
        try {
            const gasEstimate = await this.liquidationContract.estimateGas.executeLiquidation(
                opportunity.collateralAsset,
                opportunity.debtAsset,
                opportunity.user,
                opportunity.debtToCover,
                true
            );
            return gasEstimate.mul(ethers.utils.parseUnits('25', 'gwei'));
        } catch {
            return ethers.utils.parseUnits('600000', 'gwei');
        }
    }

    async executeStrategy(opportunity) {
        const startTime = Date.now();
        const { user, collateralAsset, debtAsset, debtToCover } = opportunity;
        
        try {
            const gasPrice = await this.provider.getGasPrice();
            const adjustedGasPrice = gasPrice.mul(120).div(100);
            
            const tx = await this.liquidationContract.executeLiquidation(
                collateralAsset,
                debtAsset,
                user,
                debtToCover,
                true,
                {
                    gasPrice: adjustedGasPrice,
                    gasLimit: 1000000
                }
            );
            
            const receipt = await tx.wait();
            const gasCost = receipt.gasUsed.mul(adjustedGasPrice);
            
            const liquidationEvent = receipt.logs.find(log =>
                log.topics[0] === ethers.utils.id('LiquidationCall(address,address,address,uint256,uint256,address,bool)')
            );
            
            let profit = 0;
            if (liquidationEvent) {
                const decoded = ethers.utils.defaultAbiCoder.decode(
                    ['address', 'address', 'address', 'uint256', 'uint256', 'address', 'bool'],
                    liquidationEvent.data
                );
                const liquidatedCollateralAmount = decoded[4];
                
                const collateralPrice = await this.priceOracle.getAssetPrice(collateralAsset);
                const debtPrice = await this.priceOracle.getAssetPrice(debtAsset);
                
                const collateralValue = liquidatedCollateralAmount.mul(collateralPrice).div(ethers.utils.parseEther('1'));
                const debtValue = debtToCover.mul(debtPrice).div(ethers.utils.parseEther('1'));
                
                profit = collateralValue.sub(debtValue);
            }
            
            return {
                success: receipt.status === 1,
                txHash: receipt.transactionHash,
                profit: ethers.utils.formatEther(profit),
                gasCost: ethers.utils.formatEther(gasCost),
                executionTime: Date.now() - startTime,
                liquidatedUser: user
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
        const liquidatableUsers = await this.findLiquidatableUsers();
        
        for (const user of liquidatableUsers) {
            try {
                const userReserveData = await this.aavePool.getUserAccountData(user);
                
                if (userReserveData.healthFactor.lt(ethers.utils.parseEther('1'))) {
                    const userReserves = await this.getUserReserves(user);
                    
                    for (const reserve of userReserves) {
                        if (reserve.currentVariableDebt.gt(0)) {
                            const maxLiquidationAmount = reserve.currentVariableDebt.div(2);
                            
                            const collateralReserves = userReserves.filter(r => 
                                r.currentATokenBalance.gt(0) && r.usageAsCollateralEnabled
                            );
                            
                            for (const collateral of collateralReserves) {
                                const profitEstimate = await this.calculateProfit({
                                    user,
                                    collateralAsset: collateral.asset,
                                    debtAsset: reserve.asset,
                                    debtToCover: maxLiquidationAmount
                                });
                                
                                if (profitEstimate > this.minProfitThreshold) {
                                    opportunities.push({
                                        user,
                                        collateralAsset: collateral.asset,
                                        debtAsset: reserve.asset,
                                        debtToCover: maxLiquidationAmount,
                                        healthFactor: userReserveData.healthFactor,
                                        estimatedProfit: profitEstimate,
                                        priority: this.calculatePriority(userReserveData.healthFactor, profitEstimate)
                                    });
                                }
                            }
                        }
                    }
                }
            } catch (error) {
                continue;
            }
        }
        
        return opportunities.sort((a, b) => b.priority - a.priority);
    }

    async findLiquidatableUsers() {
        const liquidatableUsers = [];
        const latestBlock = await this.provider.getBlockNumber();
        const fromBlock = latestBlock - 1000;
        
        const borrowEvents = await this.aavePool.queryFilter(
            this.aavePool.filters.Borrow(),
            fromBlock,
            latestBlock
        );
        
        const repayEvents = await this.aavePool.queryFilter(
            this.aavePool.filters.Repay(),
            fromBlock,
            latestBlock
        );
        
        const allUsers = new Set([
            ...borrowEvents.map(event => event.args.user),
            ...repayEvents.map(event => event.args.user),
            ...Array.from(this.monitoredUsers)
        ]);
        
        for (const user of allUsers) {
            try {
                const userAccountData = await this.aavePool.getUserAccountData(user);
                if (userAccountData.healthFactor.lt(ethers.utils.parseEther('1.05'))) {
                    liquidatableUsers.push(user);
                    this.monitoredUsers.add(user);
                }
            } catch (error) {
                continue;
            }
        }
        
        return liquidatableUsers;
    }

    async getUserReserves(user) {
        const reserves = [];
        const reservesList = await this.aavePool.getReservesList();
        
        for (const asset of reservesList) {
            try {
                const userReserveData = await this.aavePool.getUserReserveData(asset, user);
                
                if (userReserveData.currentATokenBalance.gt(0) || 
                    userReserveData.currentVariableDebt.gt(0) ||
                    userReserveData.currentStableDebt.gt(0)) {
                    
                    const reserveData = await this.aavePool.getReserveData(asset);
                    
                    reserves.push({
                        asset,
                        currentATokenBalance: userReserveData.currentATokenBalance,
                        currentVariableDebt: userReserveData.currentVariableDebt,
                        currentStableDebt: userReserveData.currentStableDebt,
                        usageAsCollateralEnabled: userReserveData.usageAsCollateralEnabled,
                        liquidationThreshold: reserveData.liquidationThreshold,
                        liquidationBonus: reserveData.liquidationBonus
                    });
                }
            } catch (error) {
                continue;
            }
        }
        
        return reserves;
    }

    calculatePriority(healthFactor, estimatedProfit) {
        const healthFactorScore = ethers.utils.parseEther('1').sub(healthFactor).mul(100);
        const profitScore = ethers.utils.parseEther(estimatedProfit.toString()).mul(10);
        return healthFactorScore.add(profitScore);
    }

    async startHealthFactorMonitoring() {
        setInterval(async () => {
            await this.monitorHealthFactors();
        }, 10000);
        
        this.aavePool.on('Borrow', (reserve, user, onBehalfOf, amount, borrowRateMode, borrowRate, referral) => {
            this.monitoredUsers.add(user);
        });
        
        this.aavePool.on('Repay', (reserve, user, repayer, amount) => {
            this.monitoredUsers.add(user);
        });
    }

    async monitorHealthFactors() {
        const usersToRemove = new Set();
        
        for (const user of this.monitoredUsers) {
            try {
                const userAccountData = await this.aavePool.getUserAccountData(user);
                
                if (userAccountData.totalCollateralETH.eq(0) && userAccountData.totalDebtETH.eq(0)) {
                    usersToRemove.add(user);
                    continue;
                }
                
                if (userAccountData.healthFactor.lt(ethers.utils.parseEther('1'))) {
                    this.emit('liquidationOpportunity', {
                        user,
                        healthFactor: userAccountData.healthFactor,
                        totalCollateral: userAccountData.totalCollateralETH,
                        totalDebt: userAccountData.totalDebtETH,
                        timestamp: Date.now()
                    });
                }
                
                if (userAccountData.healthFactor.gt(ethers.utils.parseEther('2'))) {
                    usersToRemove.add(user);
                }
            } catch (error) {
                usersToRemove.add(user);
            }
        }
        
        for (const user of usersToRemove) {
            this.monitoredUsers.delete(user);
        }
    }

    async getOptimalLiquidationAmount(user, collateralAsset, debtAsset) {
        const userReserveData = await this.aavePool.getUserReserveData(debtAsset, user);
        const maxLiquidation = userReserveData.currentVariableDebt.add(userReserveData.currentStableDebt).div(2);
        
        const collateralReserveData = await this.aavePool.getReserveData(collateralAsset);
        const debtReserveData = await this.aavePool.getReserveData(debtAsset);
        
        const collateralPrice = await this.priceOracle.getAssetPrice(collateralAsset);
        const debtPrice = await this.priceOracle.getAssetPrice(debtAsset);
        
        const liquidationBonus = collateralReserveData.liquidationBonus / 10000;
        const maxCollateralToLiquidate = userReserveData.currentATokenBalance;
        
        const maxDebtFromCollateral = maxCollateralToLiquidate
            .mul(collateralPrice)
            .div(debtPrice)
            .div(ethers.utils.parseUnits((1 + liquidationBonus).toString(), 18));
        
        return maxLiquidation.lt(maxDebtFromCollateral) ? maxLiquidation : maxDebtFromCollateral;
    }

    async simulateLiquidation(user, collateralAsset, debtAsset, debtToCover) {
        try {
            const userAccountDataBefore = await this.aavePool.getUserAccountData(user);
            
            const simulation = await this.liquidationContract.callStatic.executeLiquidation(
                collateralAsset,
                debtAsset,
                user,
                debtToCover,
                true
            );
            
            return {
                success: true,
                healthFactorBefore: userAccountDataBefore.healthFactor,
                collateralToReceive: simulation.collateralAmount,
                actualDebtToCover: simulation.debtAmount
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    async batchLiquidate(opportunities) {
        const batchSize = 3;
        const results = [];
        
        for (let i = 0; i < opportunities.length; i += batchSize) {
            const batch = opportunities.slice(i, i + batchSize);
            const batchPromises = batch.map(opportunity => this.execute(opportunity));
            
            const batchResults = await Promise.allSettled(batchPromises);
            results.push(...batchResults.map(result => result.value || result.reason));
            
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        return results;
    }
}

module.exports = AaveLiquidator;