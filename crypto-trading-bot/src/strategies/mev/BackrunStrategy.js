const BaseStrategy = require('../base/BaseStrategy');
const { ethers } = require('ethers');

class BackrunStrategy extends BaseStrategy {
    constructor(config, web3Provider, mempoolService, flashbotProvider, dexConnectors) {
        super(config);
        this.provider = web3Provider;
        this.mempoolService = mempoolService;
        this.flashbotProvider = flashbotProvider;
        this.dexConnectors = dexConnectors;
        this.pendingTransactions = new Map();
        this.backrunContract = null;
        this.targetTokens = new Set(config.targetTokens || []);
        this.minImpactThreshold = config.minImpactThreshold || 0.01;
    }

    async initialize() {
        const contractFactory = new ethers.ContractFactory(
            this.config.backrunABI,
            this.config.backrunBytecode,
            this.provider.getSigner()
        );
        this.backrunContract = await contractFactory.deploy();
        await this.backrunContract.deployed();
        
        await this.startMempoolMonitoring();
        await super.initialize();
    }

    async calculateProfit(opportunity) {
        const { targetTx, dex, tokenIn, tokenOut, impactedPrice, originalPrice, amount } = opportunity;
        
        const priceDiff = impactedPrice - originalPrice;
        const profit = Math.abs(priceDiff) * amount;
        
        const tradingFee = amount * (this.dexConnectors[dex].getFee() || 0.003);
        const gasEstimate = await this.estimateGas(opportunity);
        
        return profit - tradingFee - gasEstimate;
    }

    async estimateGas(opportunity) {
        try {
            const gasEstimate = await this.backrunContract.estimateGas.executeBackrun(
                opportunity.dex,
                opportunity.tokenIn,
                opportunity.tokenOut,
                opportunity.amount,
                opportunity.minAmountOut
            );
            return gasEstimate.mul(ethers.utils.parseUnits('30', 'gwei'));
        } catch {
            return ethers.utils.parseUnits('200000', 'gwei');
        }
    }

    async executeStrategy(opportunity) {
        const startTime = Date.now();
        const { targetTx, dex, tokenIn, tokenOut, amount, minAmountOut } = opportunity;
        
        try {
            const targetTxHash = targetTx.hash;
            const targetGasPrice = targetTx.gasPrice;
            const backrunGasPrice = targetGasPrice.add(ethers.utils.parseUnits('1', 'gwei'));
            
            const backrunTx = await this.backrunContract.populateTransaction.executeBackrun(
                dex,
                tokenIn,
                tokenOut,
                amount,
                minAmountOut,
                {
                    gasPrice: backrunGasPrice,
                    gasLimit: 300000
                }
            );
            
            const bundle = [
                {
                    signedTransaction: targetTx.rawTransaction
                },
                {
                    signer: this.provider.getSigner(),
                    transaction: backrunTx
                }
            ];
            
            const blockNumber = await this.provider.getBlockNumber();
            const targetBlockNumber = blockNumber + 1;
            
            const bundleResponse = await this.flashbotProvider.sendBundle(bundle, targetBlockNumber);
            
            if (bundleResponse.error) {
                throw new Error(`Bundle submission failed: ${bundleResponse.error.message}`);
            }
            
            const bundleResolution = await bundleResponse.wait();
            
            if (bundleResolution === 1) {
                const receipt = await this.provider.getTransactionReceipt(bundleResponse.bundleHash);
                const gasCost = receipt.gasUsed.mul(backrunGasPrice);
                
                const swapEvent = receipt.logs.find(log =>
                    log.topics[0] === ethers.utils.id('Swap(address,uint256,uint256,uint256,uint256,address)')
                );
                
                let profit = 0;
                if (swapEvent) {
                    const decoded = ethers.utils.defaultAbiCoder.decode(
                        ['address', 'uint256', 'uint256', 'uint256', 'uint256', 'address'],
                        swapEvent.data
                    );
                    const amountOut = decoded[4];
                    profit = amountOut.sub(amount);
                }
                
                return {
                    success: true,
                    bundleHash: bundleResponse.bundleHash,
                    txHash: receipt.transactionHash,
                    profit: ethers.utils.formatEther(profit),
                    gasCost: ethers.utils.formatEther(gasCost),
                    executionTime: Date.now() - startTime,
                    targetTxHash
                };
            } else {
                throw new Error('Bundle not included in block');
            }
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
        const recentTransactions = Array.from(this.pendingTransactions.values())
            .filter(tx => Date.now() - tx.timestamp < 30000);
        
        for (const pendingTx of recentTransactions) {
            try {
                const impact = await this.analyzePriceImpact(pendingTx);
                
                if (impact && Math.abs(impact.priceChange) > this.minImpactThreshold) {
                    const backrunAmount = this.calculateOptimalBackrunAmount(impact);
                    
                    if (backrunAmount > 0) {
                        const profitEstimate = await this.calculateProfit({
                            targetTx: pendingTx,
                            dex: impact.dex,
                            tokenIn: impact.tokenOut,
                            tokenOut: impact.tokenIn,
                            impactedPrice: impact.newPrice,
                            originalPrice: impact.originalPrice,
                            amount: backrunAmount
                        });
                        
                        if (profitEstimate > this.minProfitThreshold) {
                            opportunities.push({
                                targetTx: pendingTx,
                                dex: impact.dex,
                                tokenIn: impact.tokenOut,
                                tokenOut: impact.tokenIn,
                                amount: backrunAmount,
                                minAmountOut: backrunAmount.mul(99).div(100),
                                estimatedProfit: profitEstimate,
                                priceImpact: impact.priceChange,
                                priority: this.calculatePriority(impact.priceChange, profitEstimate)
                            });
                        }
                    }
                }
            } catch (error) {
                continue;
            }
        }
        
        return opportunities.sort((a, b) => b.priority - a.priority);
    }

    async startMempoolMonitoring() {
        this.mempoolService.on('pendingTransaction', (tx) => {
            this.handlePendingTransaction(tx);
        });
        
        setInterval(() => {
            this.cleanupOldTransactions();
        }, 60000);
    }

    async handlePendingTransaction(tx) {
        try {
            if (!tx.to || !tx.data || tx.data === '0x') return;
            
            const dexInfo = this.identifyDEX(tx.to, tx.data);
            if (!dexInfo) return;
            
            const swapInfo = await this.parseSwapTransaction(tx, dexInfo);
            if (!swapInfo) return;
            
            if (this.isTargetToken(swapInfo.tokenIn) || this.isTargetToken(swapInfo.tokenOut)) {
                this.pendingTransactions.set(tx.hash, {
                    ...tx,
                    dexInfo,
                    swapInfo,
                    timestamp: Date.now()
                });
                
                this.emit('targetTransactionDetected', {
                    txHash: tx.hash,
                    dex: dexInfo.name,
                    tokenIn: swapInfo.tokenIn,
                    tokenOut: swapInfo.tokenOut,
                    amountIn: swapInfo.amountIn
                });
            }
        } catch (error) {
            this.emit('error', `Error handling pending transaction: ${error.message}`);
        }
    }

    identifyDEX(to, data) {
        const dexRouters = {
            'uniswapV2': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
            'uniswapV3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
            'curve': '0x79a8C46DeA5aDa233ABaFFD40F3A0A2B1e5A4F27'
        };
        
        for (const [name, address] of Object.entries(dexRouters)) {
            if (to.toLowerCase() === address.toLowerCase()) {
                return { name, address };
            }
        }
        
        return null;
    }

    async parseSwapTransaction(tx, dexInfo) {
        try {
            const iface = this.dexConnectors[dexInfo.name].getInterface();
            const decoded = iface.parseTransaction({ data: tx.data });
            
            let tokenIn, tokenOut, amountIn;
            
            switch (decoded.name) {
                case 'swapExactTokensForTokens':
                case 'swapExactTokensForTokensSupportingFeeOnTransferTokens':
                    amountIn = decoded.args.amountIn;
                    tokenIn = decoded.args.path[0];
                    tokenOut = decoded.args.path[decoded.args.path.length - 1];
                    break;
                case 'exactInputSingle':
                    amountIn = decoded.args.amountIn;
                    tokenIn = decoded.args.tokenIn;
                    tokenOut = decoded.args.tokenOut;
                    break;
                default:
                    return null;
            }
            
            return { tokenIn, tokenOut, amountIn };
        } catch (error) {
            return null;
        }
    }

    async analyzePriceImpact(pendingTx) {
        if (!pendingTx.swapInfo) return null;
        
        const { tokenIn, tokenOut, amountIn } = pendingTx.swapInfo;
        const dex = pendingTx.dexInfo.name;
        
        try {
            const originalPrice = await this.dexConnectors[dex].getPrice(tokenIn, tokenOut);
            const simulatedPrice = await this.dexConnectors[dex].simulateSwap(tokenIn, tokenOut, amountIn);
            
            const priceChange = (simulatedPrice - originalPrice) / originalPrice;
            
            return {
                dex,
                tokenIn,
                tokenOut,
                originalPrice,
                newPrice: simulatedPrice,
                priceChange,
                amountIn
            };
        } catch (error) {
            return null;
        }
    }

    calculateOptimalBackrunAmount(impact) {
        const { priceChange, amountIn } = impact;
        
        const impactAmplifier = Math.abs(priceChange) * 100;
        const baseAmount = amountIn.div(10);
        
        return baseAmount.mul(Math.floor(impactAmplifier)).div(100);
    }

    calculatePriority(priceImpact, estimatedProfit) {
        const impactScore = Math.abs(priceImpact) * 1000;
        const profitScore = estimatedProfit * 100;
        return impactScore + profitScore;
    }

    isTargetToken(tokenAddress) {
        return this.targetTokens.has(tokenAddress.toLowerCase());
    }

    cleanupOldTransactions() {
        const cutoffTime = Date.now() - 120000;
        
        for (const [hash, tx] of this.pendingTransactions.entries()) {
            if (tx.timestamp < cutoffTime) {
                this.pendingTransactions.delete(hash);
            }
        }
    }

    async detectMEVOpportunities() {
        const opportunities = [];
        const recentTxs = Array.from(this.pendingTransactions.values())
            .filter(tx => Date.now() - tx.timestamp < 10000)
            .sort((a, b) => b.gasPrice - a.gasPrice);
        
        for (let i = 0; i < recentTxs.length - 1; i++) {
            const tx1 = recentTxs[i];
            const tx2 = recentTxs[i + 1];
            
            if (this.isArbitrageOpportunity(tx1, tx2)) {
                const arbitrageOpportunity = await this.createArbitrageOpportunity(tx1, tx2);
                if (arbitrageOpportunity) {
                    opportunities.push(arbitrageOpportunity);
                }
            }
        }
        
        return opportunities;
    }

    isArbitrageOpportunity(tx1, tx2) {
        if (!tx1.swapInfo || !tx2.swapInfo) return false;
        
        return (tx1.swapInfo.tokenOut === tx2.swapInfo.tokenIn &&
                tx1.swapInfo.tokenIn === tx2.swapInfo.tokenOut) ||
               (tx1.swapInfo.tokenIn === tx2.swapInfo.tokenIn &&
                tx1.swapInfo.tokenOut === tx2.swapInfo.tokenOut);
    }

    async createArbitrageOpportunity(tx1, tx2) {
        try {
            const impact1 = await this.analyzePriceImpact(tx1);
            const impact2 = await this.analyzePriceImpact(tx2);
            
            if (!impact1 || !impact2) return null;
            
            const combinedImpact = Math.abs(impact1.priceChange) + Math.abs(impact2.priceChange);
            
            if (combinedImpact > this.minImpactThreshold * 2) {
                return {
                    type: 'arbitrage',
                    tx1,
                    tx2,
                    impact1,
                    impact2,
                    combinedImpact,
                    estimatedProfit: combinedImpact * 10000
                };
            }
        } catch (error) {
            return null;
        }
        
        return null;
    }

    async executeArbitrageBundle(opportunity) {
        const { tx1, tx2, impact1, impact2 } = opportunity;
        
        try {
            const arbitrageTx = await this.backrunContract.populateTransaction.executeArbitrage(
                impact1.tokenIn,
                impact1.tokenOut,
                impact1.amountIn,
                impact2.dex,
                impact1.dex
            );
            
            const bundle = [
                { signedTransaction: tx1.rawTransaction },
                { signedTransaction: tx2.rawTransaction },
                { signer: this.provider.getSigner(), transaction: arbitrageTx }
            ];
            
            const blockNumber = await this.provider.getBlockNumber();
            const bundleResponse = await this.flashbotProvider.sendBundle(bundle, blockNumber + 1);
            
            return await bundleResponse.wait();
        } catch (error) {
            throw new Error(`Arbitrage bundle execution failed: ${error.message}`);
        }
    }
}

module.exports = BackrunStrategy;