const BaseStrategy = require('../base/BaseStrategy');
const { ethers } = require('ethers');

class MorphoLiquidator extends BaseStrategy {
    constructor(config, web3Provider, morphoAaveV3, morphoCompound, lens, flashLoanProvider) {
        super(config);
        this.provider = web3Provider;
        this.morphoAaveV3 = morphoAaveV3;
        this.morphoCompound = morphoCompound;
        this.lens = lens;
        this.flashLoanProvider = flashLoanProvider;
        this.liquidationContract = null;
        this.monitoredUsers = new Set();
        this.marketsData = new Map();
    }

    async initialize() {
        const contractFactory = new ethers.ContractFactory(
            this.config.liquidationABI,
            this.config.liquidationBytecode,
            this.provider.getSigner()
        );
        this.liquidationContract = await contractFactory.deploy(
            this.morphoAaveV3.address,
            this.morphoCompound.address
        );
        await this.liquidationContract.deployed();
        
        await this.loadMarketsData();
        await this.startUserMonitoring();
        await super.initialize();
    }

    async calculateProfit(opportunity) {
        const { user, marketId, seized, repaid, protocol } = opportunity;
        
        const morpho = protocol === 'aaveV3' ? this.morphoAaveV3 : this.morphoCompound;
        const market = this.marketsData.get(marketId);
        
        if (!market) return 0;
        
        const seizedValue = seized.mul(market.collateralPrice).div(ethers.utils.parseEther('1'));
        const repaidValue = repaid.mul(market.borrowPrice).div(ethers.utils.parseEther('1'));
        
        const profit = seizedValue.sub(repaidValue);
        const flashLoanFee = repaid.mul(5).div(10000);
        const gasEstimate = await this.estimateGas(opportunity);
        
        return profit.sub(flashLoanFee).sub(gasEstimate);
    }

    async estimateGas(opportunity) {
        try {
            const gasEstimate = await this.liquidationContract.estimateGas.liquidate(
                opportunity.marketId,
                opportunity.user,
                opportunity.repaid,
                opportunity.protocol === 'aaveV3'
            );
            return gasEstimate.mul(ethers.utils.parseUnits('25', 'gwei'));
        } catch {
            return ethers.utils.parseUnits('400000', 'gwei');
        }
    }

    async executeStrategy(opportunity) {
        const startTime = Date.now();
        const { user, marketId, repaid, protocol } = opportunity;
        
        try {
            const gasPrice = await this.provider.getGasPrice();
            const adjustedGasPrice = gasPrice.mul(110).div(100);
            
            const tx = await this.liquidationContract.liquidate(
                marketId,
                user,
                repaid,
                protocol === 'aaveV3',
                {
                    gasPrice: adjustedGasPrice,
                    gasLimit: 600000
                }
            );
            
            const receipt = await tx.wait();
            const gasCost = receipt.gasUsed.mul(adjustedGasPrice);
            
            const liquidationEvent = receipt.logs.find(log =>
                log.topics[0] === ethers.utils.id('Liquidation(bytes32,address,address,uint256,uint256)')
            );
            
            let profit = ethers.constants.Zero;
            if (liquidationEvent) {
                const decoded = ethers.utils.defaultAbiCoder.decode(
                    ['bytes32', 'address', 'address', 'uint256', 'uint256'],
                    liquidationEvent.data
                );
                const seizedAmount = decoded[3];
                const repaidAmount = decoded[4];
                
                const market = this.marketsData.get(marketId);
                if (market) {
                    const seizedValue = seizedAmount.mul(market.collateralPrice).div(ethers.utils.parseEther('1'));
                    const repaidValue = repaidAmount.mul(market.borrowPrice).div(ethers.utils.parseEther('1'));
                    profit = seizedValue.sub(repaidValue);
                }
            }
            
            return {
                success: receipt.status === 1,
                txHash: receipt.transactionHash,
                profit: ethers.utils.formatEther(profit),
                gasCost: ethers.utils.formatEther(gasCost),
                executionTime: Date.now() - startTime,
                liquidatedUser: user,
                marketId
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
        
        const aaveV3Opportunities = await this.findAaveV3Opportunities();
        const compoundOpportunities = await this.findCompoundOpportunities();
        
        opportunities.push(...aaveV3Opportunities, ...compoundOpportunities);
        
        return opportunities.sort((a, b) => b.estimatedProfit.sub(a.estimatedProfit));
    }

    async findAaveV3Opportunities() {
        const opportunities = [];
        
        try {
            const markets = await this.morphoAaveV3.marketsCreated();
            
            for (const marketId of markets) {
                const liquidatableUsers = await this.findLiquidatableUsersInMarket(marketId, 'aaveV3');
                
                for (const user of liquidatableUsers) {
                    try {
                        const userPosition = await this.lens.getUserHealthFactor(
                            this.morphoAaveV3.address,
                            user,
                            marketId
                        );
                        
                        if (userPosition.healthFactor.lt(ethers.utils.parseEther('1'))) {
                            const userMarketData = await this.lens.getUserMarketData(
                                this.morphoAaveV3.address,
                                marketId,
                                user
                            );
                            
                            if (userMarketData.borrowShares.gt(0)) {
                                const maxRepayable = userMarketData.borrowShares.div(2);
                                const seizedAmount = await this.calculateSeizedAmount(
                                    marketId,
                                    maxRepayable,
                                    'aaveV3'
                                );
                                
                                const profitEstimate = await this.calculateProfit({
                                    user,
                                    marketId,
                                    seized: seizedAmount,
                                    repaid: maxRepayable,
                                    protocol: 'aaveV3'
                                });
                                
                                if (profitEstimate.gt(ethers.utils.parseEther(this.minProfitThreshold.toString()))) {
                                    opportunities.push({
                                        user,
                                        marketId,
                                        repaid: maxRepayable,
                                        seized: seizedAmount,
                                        protocol: 'aaveV3',
                                        healthFactor: userPosition.healthFactor,
                                        estimatedProfit: profitEstimate,
                                        priority: this.calculatePriority(userPosition.healthFactor, profitEstimate)
                                    });
                                }
                            }
                        }
                    } catch (error) {
                        continue;
                    }
                }
            }
        } catch (error) {
            this.emit('error', `Error finding Aave V3 opportunities: ${error.message}`);
        }
        
        return opportunities;
    }

    async findCompoundOpportunities() {
        const opportunities = [];
        
        try {
            const markets = await this.morphoCompound.marketsCreated();
            
            for (const marketId of markets) {
                const liquidatableUsers = await this.findLiquidatableUsersInMarket(marketId, 'compound');
                
                for (const user of liquidatableUsers) {
                    try {
                        const userPosition = await this.lens.getUserHealthFactor(
                            this.morphoCompound.address,
                            user,
                            marketId
                        );
                        
                        if (userPosition.healthFactor.lt(ethers.utils.parseEther('1'))) {
                            const userMarketData = await this.lens.getUserMarketData(
                                this.morphoCompound.address,
                                marketId,
                                user
                            );
                            
                            if (userMarketData.borrowShares.gt(0)) {
                                const maxRepayable = userMarketData.borrowShares.div(2);
                                const seizedAmount = await this.calculateSeizedAmount(
                                    marketId,
                                    maxRepayable,
                                    'compound'
                                );
                                
                                const profitEstimate = await this.calculateProfit({
                                    user,
                                    marketId,
                                    seized: seizedAmount,
                                    repaid: maxRepayable,
                                    protocol: 'compound'
                                });
                                
                                if (profitEstimate.gt(ethers.utils.parseEther(this.minProfitThreshold.toString()))) {
                                    opportunities.push({
                                        user,
                                        marketId,
                                        repaid: maxRepayable,
                                        seized: seizedAmount,
                                        protocol: 'compound',
                                        healthFactor: userPosition.healthFactor,
                                        estimatedProfit: profitEstimate,
                                        priority: this.calculatePriority(userPosition.healthFactor, profitEstimate)
                                    });
                                }
                            }
                        }
                    } catch (error) {
                        continue;
                    }
                }
            }
        } catch (error) {
            this.emit('error', `Error finding Compound opportunities: ${error.message}`);
        }
        
        return opportunities;
    }

    async findLiquidatableUsersInMarket(marketId, protocol) {
        const users = new Set();
        const morpho = protocol === 'aaveV3' ? this.morphoAaveV3 : this.morphoCompound;
        
        try {
            const latestBlock = await this.provider.getBlockNumber();
            const fromBlock = latestBlock - 2000;
            
            const borrowEvents = await morpho.queryFilter(
                morpho.filters.Borrow(marketId),
                fromBlock,
                latestBlock
            );
            
            const supplyEvents = await morpho.queryFilter(
                morpho.filters.Supply(marketId),
                fromBlock,
                latestBlock
            );
            
            borrowEvents.forEach(event => users.add(event.args.user));
            supplyEvents.forEach(event => users.add(event.args.user));
            
            this.monitoredUsers.forEach(user => users.add(user));
        } catch (error) {
            this.emit('error', `Error finding users in market ${marketId}: ${error.message}`);
        }
        
        return Array.from(users);
    }

    async calculateSeizedAmount(marketId, repaidAmount, protocol) {
        try {
            const market = this.marketsData.get(marketId);
            if (!market) return ethers.constants.Zero;
            
            const liquidationIncentive = market.liquidationIncentive || ethers.utils.parseEther('1.05');
            
            const repaidValue = repaidAmount.mul(market.borrowPrice).div(ethers.utils.parseEther('1'));
            const seizedValue = repaidValue.mul(liquidationIncentive).div(ethers.utils.parseEther('1'));
            const seizedAmount = seizedValue.mul(ethers.utils.parseEther('1')).div(market.collateralPrice);
            
            return seizedAmount;
        } catch (error) {
            return ethers.constants.Zero;
        }
    }

    calculatePriority(healthFactor, estimatedProfit) {
        const healthFactorScore = ethers.utils.parseEther('1').sub(healthFactor).mul(100);
        const profitScore = estimatedProfit.mul(10);
        return healthFactorScore.add(profitScore);
    }

    async loadMarketsData() {
        try {
            const aaveV3Markets = await this.morphoAaveV3.marketsCreated();
            const compoundMarkets = await this.morphoCompound.marketsCreated();
            
            for (const marketId of aaveV3Markets) {
                const marketData = await this.lens.getMarketData(this.morphoAaveV3.address, marketId);
                this.marketsData.set(marketId, {
                    ...marketData,
                    protocol: 'aaveV3'
                });
            }
            
            for (const marketId of compoundMarkets) {
                const marketData = await this.lens.getMarketData(this.morphoCompound.address, marketId);
                this.marketsData.set(marketId, {
                    ...marketData,
                    protocol: 'compound'
                });
            }
        } catch (error) {
            this.emit('error', `Error loading markets data: ${error.message}`);
        }
    }

    async startUserMonitoring() {
        setInterval(async () => {
            await this.monitorUserHealthFactors();
        }, 12000);
        
        this.morphoAaveV3.on('Borrow', (marketId, user, onBehalfOf, assets, shares) => {
            this.monitoredUsers.add(user);
        });
        
        this.morphoAaveV3.on('Supply', (marketId, user, onBehalfOf, assets, shares) => {
            this.monitoredUsers.add(user);
        });
        
        this.morphoCompound.on('Borrow', (marketId, user, onBehalfOf, assets, shares) => {
            this.monitoredUsers.add(user);
        });
        
        this.morphoCompound.on('Supply', (marketId, user, onBehalfOf, assets, shares) => {
            this.monitoredUsers.add(user);
        });
    }

    async monitorUserHealthFactors() {
        const usersToRemove = new Set();
        
        for (const user of this.monitoredUsers) {
            try {
                const aaveV3HealthFactor = await this.lens.getUserHealthFactor(
                    this.morphoAaveV3.address,
                    user
                );
                
                const compoundHealthFactor = await this.lens.getUserHealthFactor(
                    this.morphoCompound.address,
                    user
                );
                
                const minHealthFactor = aaveV3HealthFactor.healthFactor.lt(compoundHealthFactor.healthFactor) 
                    ? aaveV3HealthFactor.healthFactor 
                    : compoundHealthFactor.healthFactor;
                
                if (minHealthFactor.lt(ethers.utils.parseEther('1'))) {
                    this.emit('liquidationOpportunity', {
                        user,
                        aaveV3HealthFactor: aaveV3HealthFactor.healthFactor,
                        compoundHealthFactor: compoundHealthFactor.healthFactor,
                        timestamp: Date.now()
                    });
                }
                
                if (minHealthFactor.gt(ethers.utils.parseEther('2'))) {
                    const hasActivePositions = await this.checkActivePositions(user);
                    if (!hasActivePositions) {
                        usersToRemove.add(user);
                    }
                }
            } catch (error) {
                usersToRemove.add(user);
            }
        }
        
        for (const user of usersToRemove) {
            this.monitoredUsers.delete(user);
        }
    }

    async checkActivePositions(user) {
        try {
            const aaveV3Markets = await this.morphoAaveV3.marketsCreated();
            const compoundMarkets = await this.morphoCompound.marketsCreated();
            
            for (const marketId of aaveV3Markets) {
                const userData = await this.lens.getUserMarketData(this.morphoAaveV3.address, marketId, user);
                if (userData.supplyShares.gt(0) || userData.borrowShares.gt(0)) {
                    return true;
                }
            }
            
            for (const marketId of compoundMarkets) {
                const userData = await this.lens.getUserMarketData(this.morphoCompound.address, marketId, user);
                if (userData.supplyShares.gt(0) || userData.borrowShares.gt(0)) {
                    return true;
                }
            }
            
            return false;
        } catch {
            return false;
        }
    }

    async simulateLiquidation(user, marketId, repaidAmount, protocol) {
        try {
            const simulation = await this.liquidationContract.callStatic.liquidate(
                marketId,
                user,
                repaidAmount,
                protocol === 'aaveV3'
            );
            
            return {
                success: true,
                seizedAmount: simulation.seizedAmount,
                actualRepaidAmount: simulation.repaidAmount
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    async getOptimalLiquidationAmount(user, marketId, protocol) {
        try {
            const morpho = protocol === 'aaveV3' ? this.morphoAaveV3 : this.morphoCompound;
            const userData = await this.lens.getUserMarketData(morpho.address, marketId, user);
            
            const maxLiquidation = userData.borrowShares.div(2);
            
            const market = this.marketsData.get(marketId);
            if (!market) return maxLiquidation;
            
            const maxSeizable = userData.supplyShares;
            const maxRepayFromCollateral = maxSeizable
                .mul(market.collateralPrice)
                .div(market.borrowPrice)
                .div(market.liquidationIncentive || ethers.utils.parseEther('1.05'));
            
            return maxLiquidation.lt(maxRepayFromCollateral) ? maxLiquidation : maxRepayFromCollateral;
        } catch {
            return ethers.constants.Zero;
        }
    }

    async batchLiquidate(opportunities) {
        const results = [];
        const batchSize = 2;
        
        for (let i = 0; i < opportunities.length; i += batchSize) {
            const batch = opportunities.slice(i, i + batchSize);
            const batchPromises = batch.map(opportunity => this.execute(opportunity));
            
            const batchResults = await Promise.allSettled(batchPromises);
            results.push(...batchResults.map(result => result.value || result.reason));
            
            if (i + batchSize < opportunities.length) {
                await new Promise(resolve => setTimeout(resolve, 1500));
            }
        }
        
        return results;
    }
}

module.exports = MorphoLiquidator;