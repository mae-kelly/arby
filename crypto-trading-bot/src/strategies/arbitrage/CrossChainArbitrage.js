const BaseStrategy = require('../base/BaseStrategy');

class CrossChainArbitrage extends BaseStrategy {
    constructor(config, chainConnectors, bridgeConnectors) {
        super(config);
        this.chains = chainConnectors;
        this.bridges = bridgeConnectors;
        this.chainBalances = new Map();
        this.bridgeFees = new Map();
        this.chainPrices = new Map();
    }

    async initialize() {
        for (const [chainId, connector] of Object.entries(this.chains)) {
            const balance = await connector.getBalance();
            this.chainBalances.set(chainId, balance);
        }
        
        await this.updateBridgeFees();
        await super.initialize();
    }

    async calculateProfit(opportunity) {
        const { tokenAddress, amount, sourceChain, targetChain, sourceDex, targetDex } = opportunity;
        
        const sourcePrice = await this.chains[sourceChain].getPrice(tokenAddress, sourceDex);
        const targetPrice = await this.chains[targetChain].getPrice(tokenAddress, targetDex);
        
        const priceDiff = targetPrice - sourcePrice;
        const grossProfit = priceDiff * amount;
        
        const bridgeFee = await this.getBridgeFee(tokenAddress, sourceChain, targetChain, amount);
        const sourceTradingFee = sourcePrice * amount * 0.003;
        const targetTradingFee = targetPrice * amount * 0.003;
        const gasFeesSource = await this.chains[sourceChain].estimateGasCost();
        const gasFeesTarget = await this.chains[targetChain].estimateGasCost();
        
        return grossProfit - bridgeFee - sourceTradingFee - targetTradingFee - gasFeesSource - gasFeesTarget;
    }

    async estimateGas(opportunity) {
        const sourceGas = await this.chains[opportunity.sourceChain].estimateGasCost();
        const targetGas = await this.chains[opportunity.targetChain].estimateGasCost();
        return sourceGas + targetGas;
    }

    async executeStrategy(opportunity) {
        const startTime = Date.now();
        const { tokenAddress, amount, sourceChain, targetChain, sourceDex, targetDex } = opportunity;
        
        try {
            const buyTx = await this.chains[sourceChain].buyToken(tokenAddress, amount, sourceDex);
            const buyReceipt = await this.chains[sourceChain].waitForTransaction(buyTx);
            
            if (buyReceipt.status !== 1) {
                throw new Error('Buy transaction failed');
            }
            
            const bridgeTx = await this.bridgeTokens(tokenAddress, amount, sourceChain, targetChain);
            const bridgeReceipt = await this.waitForBridgeCompletion(bridgeTx, targetChain);
            
            if (!bridgeReceipt.success) {
                throw new Error('Bridge transaction failed');
            }
            
            const sellTx = await this.chains[targetChain].sellToken(tokenAddress, amount, targetDex);
            const sellReceipt = await this.chains[targetChain].waitForTransaction(sellTx);
            
            if (sellReceipt.status !== 1) {
                throw new Error('Sell transaction failed');
            }
            
            const profit = this.calculateActualProfit(buyReceipt, sellReceipt, bridgeReceipt);
            const totalGasCost = buyReceipt.gasUsed + sellReceipt.gasUsed + bridgeReceipt.gasCost;
            
            return {
                success: true,
                buyTxHash: buyReceipt.transactionHash,
                bridgeTxHash: bridgeTx,
                sellTxHash: sellReceipt.transactionHash,
                profit,
                gasCost: totalGasCost,
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
        await this.updateChainPrices();
        const opportunities = [];
        const tokens = this.config.crossChainTokens;
        
        for (const token of tokens) {
            const chainIds = Object.keys(this.chains);
            
            for (let i = 0; i < chainIds.length; i++) {
                for (let j = 0; j < chainIds.length; j++) {
                    if (i === j) continue;
                    
                    const sourceChain = chainIds[i];
                    const targetChain = chainIds[j];
                    
                    const sourcePrices = this.chainPrices.get(`${sourceChain}:${token.address}`) || {};
                    const targetPrices = this.chainPrices.get(`${targetChain}:${token.address}`) || {};
                    
                    for (const [sourceDex, sourcePrice] of Object.entries(sourcePrices)) {
                        for (const [targetDex, targetPrice] of Object.entries(targetPrices)) {
                            const priceDiff = (targetPrice - sourcePrice) / sourcePrice;
                            
                            if (priceDiff > this.minProfitThreshold) {
                                const bridgeTime = await this.getBridgeTime(sourceChain, targetChain);
                                
                                if (bridgeTime < this.config.maxBridgeTime) {
                                    const maxAmount = await this.getMaxTradeAmount(
                                        token.address, 
                                        sourceChain, 
                                        targetChain
                                    );
                                    
                                    opportunities.push({
                                        tokenAddress: token.address,
                                        amount: Math.min(maxAmount, token.defaultAmount),
                                        sourceChain,
                                        targetChain,
                                        sourceDex,
                                        targetDex,
                                        sourcePrice,
                                        targetPrice,
                                        priceDiff,
                                        bridgeTime
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return opportunities.sort((a, b) => b.priceDiff - a.priceDiff);
    }

    async updateChainPrices() {
        const promises = [];
        
        for (const [chainId, connector] of Object.entries(this.chains)) {
            for (const token of this.config.crossChainTokens) {
                promises.push(
                    this.updateTokenPricesOnChain(chainId, token.address, connector)
                );
            }
        }
        
        await Promise.allSettled(promises);
    }

    async updateTokenPricesOnChain(chainId, tokenAddress, connector) {
        try {
            const dexes = this.config.supportedDexes[chainId] || [];
            const prices = {};
            
            for (const dex of dexes) {
                try {
                    const price = await connector.getPrice(tokenAddress, dex);
                    prices[dex] = price;
                } catch (error) {
                    continue;
                }
            }
            
            this.chainPrices.set(`${chainId}:${tokenAddress}`, prices);
        } catch (error) {
            this.emit('error', `Failed to update prices for ${chainId}:${tokenAddress}`);
        }
    }

    async bridgeTokens(tokenAddress, amount, sourceChain, targetChain) {
        const bridgeConnector = this.getBridgeConnector(sourceChain, targetChain);
        
        if (!bridgeConnector) {
            throw new Error(`No bridge available for ${sourceChain} -> ${targetChain}`);
        }
        
        return await bridgeConnector.bridgeTokens(tokenAddress, amount, targetChain);
    }

    async waitForBridgeCompletion(bridgeTxHash, targetChain) {
        const bridgeConnector = this.getBridgeConnector(null, targetChain);
        const maxWaitTime = 30 * 60 * 1000;
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWaitTime) {
            try {
                const status = await bridgeConnector.getBridgeStatus(bridgeTxHash);
                
                if (status.completed) {
                    return {
                        success: true,
                        gasCost: status.gasCost || 0,
                        actualAmount: status.actualAmount
                    };
                }
                
                if (status.failed) {
                    return { success: false, error: status.error };
                }
                
                await new Promise(resolve => setTimeout(resolve, 10000));
            } catch (error) {
                await new Promise(resolve => setTimeout(resolve, 5000));
            }
        }
        
        return { success: false, error: 'Bridge timeout' };
    }

    getBridgeConnector(sourceChain, targetChain) {
        const bridgeKey = sourceChain ? `${sourceChain}-${targetChain}` : targetChain;
        return this.bridges[bridgeKey] || this.bridges['default'];
    }

    async getBridgeFee(tokenAddress, sourceChain, targetChain, amount) {
        const cacheKey = `${tokenAddress}:${sourceChain}:${targetChain}`;
        const cached = this.bridgeFees.get(cacheKey);
        
        if (cached && Date.now() - cached.timestamp < 300000) {
            return cached.fee * amount;
        }
        
        try {
            const bridgeConnector = this.getBridgeConnector(sourceChain, targetChain);
            const fee = await bridgeConnector.getFee(tokenAddress, amount);
            
            this.bridgeFees.set(cacheKey, { fee: fee / amount, timestamp: Date.now() });
            return fee;
        } catch {
            return amount * 0.001;
        }
    }

    async getBridgeTime(sourceChain, targetChain) {
        const bridgeConnector = this.getBridgeConnector(sourceChain, targetChain);
        
        try {
            return await bridgeConnector.getEstimatedTime();
        } catch {
            return 15 * 60;
        }
    }

    async getMaxTradeAmount(tokenAddress, sourceChain, targetChain) {
        const sourceBalance = this.chainBalances.get(sourceChain);
        const bridgeLimits = await this.getBridgeLimits(tokenAddress, sourceChain, targetChain);
        
        const maxFromBalance = sourceBalance[tokenAddress] || 0;
        const maxFromBridge = bridgeLimits.maxAmount;
        
        return Math.min(maxFromBalance * 0.95, maxFromBridge);
    }

    async getBridgeLimits(tokenAddress, sourceChain, targetChain) {
        try {
            const bridgeConnector = this.getBridgeConnector(sourceChain, targetChain);
            return await bridgeConnector.getLimits(tokenAddress);
        } catch {
            return { maxAmount: 1000000, minAmount: 1 };
        }
    }

    async updateBridgeFees() {
        const promises = [];
        
        for (const token of this.config.crossChainTokens) {
            const chainIds = Object.keys(this.chains);
            
            for (let i = 0; i < chainIds.length; i++) {
                for (let j = 0; j < chainIds.length; j++) {
                    if (i !== j) {
                        promises.push(
                            this.getBridgeFee(token.address, chainIds[i], chainIds[j], 1)
                        );
                    }
                }
            }
        }
        
        await Promise.allSettled(promises);
    }

    calculateActualProfit(buyReceipt, sellReceipt, bridgeReceipt) {
        const buyAmount = this.parseTradeAmount(buyReceipt);
        const sellAmount = this.parseTradeAmount(sellReceipt);
        const bridgeFee = bridgeReceipt.gasCost || 0;
        
        return sellAmount - buyAmount - bridgeFee;
    }

    parseTradeAmount(receipt) {
        try {
            const transferLog = receipt.logs.find(log => 
                log.topics[0] === '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
            );
            
            if (transferLog && transferLog.data) {
                return parseInt(transferLog.data, 16);
            }
        } catch {}
        
        return 0;
    }
}

module.exports = CrossChainArbitrage;