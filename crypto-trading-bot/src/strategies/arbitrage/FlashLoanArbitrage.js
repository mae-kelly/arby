const BaseStrategy = require('../base/BaseStrategy');
const { ethers } = require('ethers');

class FlashLoanArbitrage extends BaseStrategy {
    constructor(config, web3Provider, aavePool, balancerVault, dexConnectors) {
        super(config);
        this.provider = web3Provider;
        this.aavePool = aavePool;
        this.balancerVault = balancerVault;
        this.dexConnectors = dexConnectors;
        this.flashLoanContract = null;
    }

    async initialize() {
        const contractFactory = new ethers.ContractFactory(
            this.config.flashLoanABI,
            this.config.flashLoanBytecode,
            this.provider.getSigner()
        );
        this.flashLoanContract = await contractFactory.deploy();
        await this.flashLoanContract.deployed();
        await super.initialize();
    }

    async calculateProfit(opportunity) {
        const { tokenAddress, amount, dexA, dexB } = opportunity;
        
        const priceA = await this.dexConnectors[dexA].getPrice(tokenAddress);
        const priceB = await this.dexConnectors[dexB].getPrice(tokenAddress);
        
        const priceDiff = Math.abs(priceA - priceB);
        const profit = (priceDiff / Math.min(priceA, priceB)) * amount;
        
        const flashLoanFee = amount * 0.0005;
        const dexFees = amount * 0.003 * 2;
        
        return profit - flashLoanFee - dexFees;
    }

    async estimateGas(opportunity) {
        try {
            const gasEstimate = await this.flashLoanContract.estimateGas.executeArbitrage(
                opportunity.tokenAddress,
                opportunity.amount,
                opportunity.dexA,
                opportunity.dexB,
                opportunity.minProfit
            );
            return gasEstimate.mul(ethers.utils.parseUnits('20', 'gwei'));
        } catch {
            return ethers.utils.parseUnits('500000', 'gwei');
        }
    }

    async executeStrategy(opportunity) {
        const startTime = Date.now();
        
        try {
            const gasPrice = await this.provider.getGasPrice();
            const adjustedGasPrice = gasPrice.mul(110).div(100);
            
            const tx = await this.flashLoanContract.executeArbitrage(
                opportunity.tokenAddress,
                opportunity.amount,
                opportunity.dexA,
                opportunity.dexB,
                opportunity.minProfit,
                {
                    gasPrice: adjustedGasPrice,
                    gasLimit: 800000
                }
            );
            
            const receipt = await tx.wait();
            const gasCost = receipt.gasUsed.mul(adjustedGasPrice);
            
            const logs = receipt.logs.find(log => 
                log.topics[0] === ethers.utils.id('ArbitrageExecuted(address,uint256,uint256)')
            );
            
            const profit = logs ? ethers.utils.defaultAbiCoder.decode(['uint256'], logs.data)[0] : 0;
            
            return {
                success: receipt.status === 1,
                txHash: receipt.transactionHash,
                profit: ethers.utils.formatEther(profit),
                gasCost: ethers.utils.formatEther(gasCost),
                executionTime: Date.now() - startTime
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
        const tokens = this.config.targetTokens;
        
        for (const token of tokens) {
            const prices = {};
            
            for (const [dexName, connector] of Object.entries(this.dexConnectors)) {
                try {
                    prices[dexName] = await connector.getPrice(token.address);
                } catch (error) {
                    continue;
                }
            }
            
            const dexNames = Object.keys(prices);
            for (let i = 0; i < dexNames.length; i++) {
                for (let j = i + 1; j < dexNames.length; j++) {
                    const dexA = dexNames[i];
                    const dexB = dexNames[j];
                    const priceA = prices[dexA];
                    const priceB = prices[dexB];
                    
                    const priceDiff = Math.abs(priceA - priceB) / Math.min(priceA, priceB);
                    
                    if (priceDiff > this.minProfitThreshold) {
                        const buyDex = priceA < priceB ? dexA : dexB;
                        const sellDex = priceA < priceB ? dexB : dexA;
                        
                        opportunities.push({
                            tokenAddress: token.address,
                            amount: ethers.utils.parseEther(token.defaultAmount),
                            dexA: buyDex,
                            dexB: sellDex,
                            priceDifference: priceDiff,
                            minProfit: ethers.utils.parseEther((priceDiff * 0.8).toString())
                        });
                    }
                }
            }
        }
        
        return opportunities.sort((a, b) => b.priceDifference - a.priceDifference);
    }
}

module.exports = FlashLoanArbitrage;