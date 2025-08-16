const BaseStrategy = require('../base/BaseStrategy');

class TriangularArbitrage extends BaseStrategy {
    constructor(config, dexConnector) {
        super(config);
        this.dex = dexConnector;
        this.triangles = [];
        this.priceCache = new Map();
        this.lastCacheUpdate = 0;
    }

    async initialize() {
        this.triangles = this.generateTriangles(this.config.tokens);
        await super.initialize();
    }

    generateTriangles(tokens) {
        const triangles = [];
        const pairs = this.config.tradingPairs;
        
        for (let i = 0; i < tokens.length; i++) {
            for (let j = i + 1; j < tokens.length; j++) {
                for (let k = j + 1; k < tokens.length; k++) {
                    const tokenA = tokens[i];
                    const tokenB = tokens[j];
                    const tokenC = tokens[k];
                    
                    const pairAB = `${tokenA}/${tokenB}`;
                    const pairBC = `${tokenB}/${tokenC}`;
                    const pairCA = `${tokenC}/${tokenA}`;
                    const pairBA = `${tokenB}/${tokenA}`;
                    const pairCB = `${tokenC}/${tokenB}`;
                    const pairAC = `${tokenA}/${tokenC}`;
                    
                    if (pairs.includes(pairAB) && pairs.includes(pairBC) && pairs.includes(pairCA)) {
                        triangles.push({
                            path: [tokenA, tokenB, tokenC, tokenA],
                            pairs: [pairAB, pairBC, pairCA],
                            direction: 'forward'
                        });
                    }
                    
                    if (pairs.includes(pairBA) && pairs.includes(pairCB) && pairs.includes(pairAC)) {
                        triangles.push({
                            path: [tokenA, tokenC, tokenB, tokenA],
                            pairs: [pairAC, pairCB, pairBA],
                            direction: 'reverse'
                        });
                    }
                }
            }
        }
        
        return triangles;
    }

    async calculateProfit(opportunity) {
        const { triangle, startAmount } = opportunity;
        let currentAmount = startAmount;
        let totalFees = 0;
        
        for (let i = 0; i < triangle.pairs.length; i++) {
            const pair = triangle.pairs[i];
            const price = await this.getPrice(pair);
            const fee = currentAmount * (this.dex.getFee(pair) || 0.003);
            
            currentAmount = (currentAmount - fee) * price;
            totalFees += fee;
        }
        
        return currentAmount - startAmount - totalFees;
    }

    async estimateGas(opportunity) {
        try {
            const gasEstimate = await this.dex.estimateTriangularArbitrageGas(
                opportunity.triangle.path,
                opportunity.startAmount
            );
            return gasEstimate * 1.2;
        } catch {
            return 300000;
        }
    }

    async executeStrategy(opportunity) {
        const startTime = Date.now();
        const { triangle, startAmount } = opportunity;
        
        try {
            const txHash = await this.dex.executeTriangularArbitrage(
                triangle.path,
                triangle.pairs,
                startAmount,
                opportunity.minProfit
            );
            
            const receipt = await this.dex.waitForTransaction(txHash);
            const logs = this.dex.parseTriangularArbitrageLogs(receipt.logs);
            
            const profit = logs.profit || 0;
            const gasCost = receipt.gasUsed * receipt.effectiveGasPrice;
            
            return {
                success: receipt.status === 1,
                txHash: receipt.transactionHash,
                profit: this.dex.formatUnits(profit),
                gasCost: this.dex.formatUnits(gasCost),
                executionTime: Date.now() - startTime,
                path: triangle.path
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
        await this.updatePriceCache();
        const opportunities = [];
        
        for (const triangle of this.triangles) {
            const startAmount = this.config.defaultTradeAmount;
            let currentAmount = startAmount;
            let isValid = true;
            
            for (const pair of triangle.pairs) {
                const price = this.priceCache.get(pair);
                if (!price) {
                    isValid = false;
                    break;
                }
                currentAmount *= price;
            }
            
            if (!isValid) continue;
            
            const profit = currentAmount - startAmount;
            const profitPercentage = profit / startAmount;
            
            if (profitPercentage > this.minProfitThreshold) {
                const liquidity = await this.checkLiquidity(triangle, startAmount);
                
                if (liquidity.sufficient) {
                    opportunities.push({
                        triangle,
                        startAmount,
                        expectedProfit: profit,
                        profitPercentage,
                        minProfit: profit * 0.9,
                        liquidity: liquidity.minLiquidity
                    });
                }
            }
        }
        
        return opportunities.sort((a, b) => b.profitPercentage - a.profitPercentage);
    }

    async updatePriceCache() {
        if (Date.now() - this.lastCacheUpdate < 2000) return;
        
        const promises = this.config.tradingPairs.map(async pair => {
            try {
                const price = await this.dex.getPrice(pair);
                this.priceCache.set(pair, price);
            } catch (error) {
                this.priceCache.delete(pair);
            }
        });
        
        await Promise.allSettled(promises);
        this.lastCacheUpdate = Date.now();
    }

    async getPrice(pair) {
        const cached = this.priceCache.get(pair);
        if (cached && Date.now() - this.lastCacheUpdate < 5000) {
            return cached;
        }
        
        try {
            const price = await this.dex.getPrice(pair);
            this.priceCache.set(pair, price);
            return price;
        } catch {
            return this.priceCache.get(pair) || 0;
        }
    }

    async checkLiquidity(triangle, amount) {
        let minLiquidity = Infinity;
        let currentAmount = amount;
        
        for (const pair of triangle.pairs) {
            try {
                const liquidity = await this.dex.getLiquidity(pair);
                const requiredLiquidity = currentAmount;
                
                if (liquidity < requiredLiquidity) {
                    return { sufficient: false, minLiquidity: liquidity };
                }
                
                minLiquidity = Math.min(minLiquidity, liquidity);
                const price = this.priceCache.get(pair) || await this.getPrice(pair);
                currentAmount *= price;
            } catch {
                return { sufficient: false, minLiquidity: 0 };
            }
        }
        
        return { sufficient: true, minLiquidity };
    }

    calculateOptimalAmount(triangle, maxAmount) {
        let optimalAmount = maxAmount;
        
        for (const pair of triangle.pairs) {
            const liquidity = this.dex.getLiquidity(pair);
            const maxForPair = liquidity * 0.1;
            optimalAmount = Math.min(optimalAmount, maxForPair);
        }
        
        return optimalAmount;
    }

    async simulateTriangle(triangle, amount) {
        let currentAmount = amount;
        const simulation = { steps: [], finalAmount: 0, profit: 0 };
        
        for (let i = 0; i < triangle.pairs.length; i++) {
            const pair = triangle.pairs[i];
            const price = await this.getPrice(pair);
            const fee = currentAmount * (this.dex.getFee(pair) || 0.003);
            const amountAfterFee = currentAmount - fee;
            const outputAmount = amountAfterFee * price;
            
            simulation.steps.push({
                pair,
                inputAmount: currentAmount,
                fee,
                price,
                outputAmount
            });
            
            currentAmount = outputAmount;
        }
        
        simulation.finalAmount = currentAmount;
        simulation.profit = currentAmount - amount;
        
        return simulation;
    }
}

module.exports = TriangularArbitrage;