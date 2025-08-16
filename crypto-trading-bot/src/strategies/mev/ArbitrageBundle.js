const BaseStrategy = require('../base/BaseStrategy');
const { ethers } = require('ethers');

class ArbitrageBundle extends BaseStrategy {
    constructor(config, web3Provider, flashbotProvider, dexConnectors, mempoolService) {
        super(config);
        this.provider = web3Provider;
        this.flashbotProvider = flashbotProvider;
        this.dexConnectors = dexConnectors;
        this.mempoolService = mempoolService;
        this.bundleContract = null;
        this.activeBundles = new Map();
        this.maxBundleSize = config.maxBundleSize || 3;
        this.bundleTimeout = config.bundleTimeout || 30000;
    }

    async initialize() {
        const contractFactory = new ethers.ContractFactory(
            this.config.bundleABI,
            this.config.bundleBytecode,
            this.provider.getSigner()
        );
        this.bundleContract = await contractFactory.deploy();
        await this.bundleContract.deployed();
        
        await this.startBundleMonitoring();
        await super.initialize();
    }

    async calculateProfit(opportunity) {
        const { bundle, totalGasEstimate } = opportunity;
        let totalProfit = 0;
        
        for (const operation of bundle.operations) {
            const { dexA, dexB, tokenIn, tokenOut, amount, priceA, priceB } = operation;
            
            const priceDiff = Math.abs(priceB - priceA);
            const operationProfit = (priceDiff / Math.min(priceA, priceB)) * amount;
            
            const fees = amount * 0.003 * 2;
            totalProfit += operationProfit - fees;
        }
        
        const gasCost = totalGasEstimate * bundle.gasPrice;
        const flashLoanFee = bundle.flashLoanAmount * 0.0005;
        
        return totalProfit - gasCost - flashLoanFee;
    }

    async estimateGas(opportunity) {
        const { bundle } = opportunity;
        let totalGas = ethers.BigNumber.from(0);
        
        try {
            for (const operation of bundle.operations) {
                const gasEstimate = await this.bundleContract.estimateGas.executeArbitrageOperation(
                    operation.dexA,
                    operation.dexB,
                    operation.tokenIn,
                    operation.tokenOut,
                    operation.amount
                );
                totalGas = totalGas.add(gasEstimate);
            }
            
            return totalGas.mul(ethers.utils.parseUnits('25', 'gwei'));
        } catch {
            return ethers.utils.parseUnits('800000', 'gwei');
        }
    }

    async executeStrategy(opportunity) {
        const startTime = Date.now();
        const { bundle } = opportunity;
        
        try {
            const gasPrice = await this.provider.getGasPrice();
            const priorityGasPrice = gasPrice.mul(130).div(100);
            
            const bundleTx = await this.bundleContract.populateTransaction.executeBundleArbitrage(
                bundle.operations,
                bundle.flashLoanAmount,
                bundle.flashLoanProvider,
                {
                    gasPrice: priorityGasPrice,
                    gasLimit: bundle.totalGasLimit
                }
            );
            
            const signedTx = await this.provider.getSigner().signTransaction(bundleTx);
            
            const flashbotBundle = [
                {
                    signedTransaction: signedTx
                }
            ];
            
            if (bundle.targetTransactions) {
                for (const targetTx of bundle.targetTransactions) {
                    flashbotBundle.unshift({
                        signedTransaction: targetTx.rawTransaction
                    });
                }
            }
            
            const blockNumber = await this.provider.getBlockNumber();
            const targetBlockNumber = blockNumber + 1;
            
            const bundleResponse = await this.flashbotProvider.sendBundle(
                flashbotBundle, 
                targetBlockNumber,
                {
                    minTimestamp: Math.floor(Date.now() / 1000),
                    maxTimestamp: Math.floor(Date.now() / 1000) + 120
                }
            );
            
            if (bundleResponse.error) {
                throw new Error(`Bundle submission failed: ${bundleResponse.error.message}`);
            }
            
            const bundleResolution = await bundleResponse.wait();
            
            if (bundleResolution === 1) {
                const receipt = await this.provider.getTransactionReceipt(bundleResponse.bundleHash);
                const gasCost = receipt.gasUsed.mul(priorityGasPrice);
                
                const bundleEvent = receipt.logs.find(log =>
                    log.topics[0] === ethers.utils.id('BundleExecuted(uint256,uint256,uint256)')
                );
                
                let profit = ethers.constants.Zero;
                if (bundleEvent) {
                    const decoded = ethers.utils.defaultAbiCoder.decode(
                        ['uint256', 'uint256', 'uint256'],
                        bundleEvent.data
                    );
                    profit = decoded[0];
                }
                
                return {
                    success: true,
                    bundleHash: bundleResponse.bundleHash,
                    profit: ethers.utils.formatEther(profit),
                    gasCost: ethers.utils.formatEther(gasCost),
                    executionTime: Date.now() - startTime,
                    operationsCount: bundle.operations.length
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
        
        const staticArbitrages = await this.findStaticArbitrageOpportunities();
        const dynamicArbitrages = await this.findDynamicArbitrageOpportunities();
        const mevBundles = await this.findMEVBundleOpportunities();
        
        opportunities.push(...staticArbitrages, ...dynamicArbitrages, ...mevBundles);
        
        return opportunities.sort((a, b) => b.estimatedProfit - a.estimatedProfit);
    }

    async findStaticArbitrageOpportunities() {
        const opportunities = [];
        const tokens = this.config.targetTokens;
        const dexNames = Object.keys(this.dexConnectors);
        
        for (const token of tokens) {
            const prices = {};
            
            for (const dex of dexNames) {
                try {
                    prices[dex] = await this.dexConnectors[dex].getPrice(token.address);
                } catch {
                    continue;
                }
            }
            
            const combinations = this.generateDexCombinations(dexNames, 2);
            
            for (const [dexA, dexB] of combinations) {
                if (!prices[dexA] || !prices[dexB]) continue;
                
                const priceDiff = Math.abs(prices[dexA] - prices[dexB]);
                const relativeSpread = priceDiff / Math.min(prices[dexA], prices[dexB]);
                
                if (relativeSpread > this.minProfitThreshold) {
                    const bundle = this.createArbitrageBundle([{
                        dexA: prices[dexA] < prices[dexB] ? dexA : dexB,
                        dexB: prices[dexA] < prices[dexB] ? dexB : dexA,
                        tokenIn: token.address,
                        tokenOut: token.address,
                        amount: ethers.utils.parseEther(token.defaultAmount),
                        priceA: Math.min(prices[dexA], prices[dexB]),
                        priceB: Math.max(prices[dexA], prices[dexB])
                    }]);
                    
                    const profitEstimate = await this.calculateProfit({ bundle });
                    
                    if (profitEstimate > this.minProfitThreshold) {
                        opportunities.push({
                            type: 'static',
                            bundle,
                            estimatedProfit: profitEstimate,
                            tokenAddress: token.address,
                            spread: relativeSpread
                        });
                    }
                }
            }
        }
        
        return opportunities;
    }

    async findDynamicArbitrageOpportunities() {
        const opportunities = [];
        const recentBlocks = await this.getRecentBlockTransactions(3);
        
        for (const blockTxs of recentBlocks) {
            const dexTransactions = blockTxs.filter(tx => this.isDexTransaction(tx));
            
            if (dexTransactions.length >= 2) {
                const bundles = await this.createDynamicBundles(dexTransactions);
                
                for (const bundle of bundles) {
                    const profitEstimate = await this.calculateProfit({ bundle });
                    
                    if (profitEstimate > this.minProfitThreshold) {
                        opportunities.push({
                            type: 'dynamic',
                            bundle,
                            estimatedProfit: profitEstimate,
                            blockNumber: bundle.blockNumber
                        });
                    }
                }
            }
        }
        
        return opportunities;
    }

    async findMEVBundleOpportunities() {
        const opportunities = [];
        const pendingTxs = await this.mempoolService.getPendingTransactions();
        
        const mevCandidates = pendingTxs.filter(tx => 
            this.isMEVTarget(tx) && Date.now() - tx.timestamp < 20000
        );
        
        if (mevCandidates.length >= 2) {
            const mevBundles = await this.createMEVBundles(mevCandidates);
            
            for (const bundle of mevBundles) {
                const profitEstimate = await this.calculateProfit({ bundle });
                
                if (profitEstimate > this.minProfitThreshold * 2) {
                    opportunities.push({
                        type: 'mev',
                        bundle,
                        estimatedProfit: profitEstimate,
                        targetTransactions: bundle.targetTransactions
                    });
                }
            }
        }
        
        return opportunities;
    }

    createArbitrageBundle(operations) {
        const totalAmount = operations.reduce((sum, op) => sum.add(op.amount), ethers.constants.Zero);
        
        return {
            operations,
            flashLoanAmount: totalAmount,
            flashLoanProvider: 'aave',
            totalGasLimit: operations.length * 200000 + 100000,
            gasPrice: ethers.utils.parseUnits('25', 'gwei'),
            timestamp: Date.now()
        };
    }

    async createDynamicBundles(dexTransactions) {
        const bundles = [];
        const maxOperations = Math.min(this.maxBundleSize, dexTransactions.length);
        
        for (let i = 2; i <= maxOperations; i++) {
            const combinations = this.generateTransactionCombinations(dexTransactions, i);
            
            for (const combo of combinations) {
                const operations = await this.analyzeTransactionCombo(combo);
                
                if (operations.length > 0) {
                    bundles.push(this.createArbitrageBundle(operations));
                }
            }
        }
        
        return bundles;
    }

    async createMEVBundles(mevCandidates) {
        const bundles = [];
        
        const sandwichableTransactions = mevCandidates.filter(tx => this.isSandwichable(tx));
        const arbitrageTransactions = mevCandidates.filter(tx => this.isArbitrageable(tx));
        
        for (const sandwich of sandwichableTransactions) {
            const frontrunOperation = await this.createFrontrunOperation(sandwich);
            const backrunOperation = await this.createBackrunOperation(sandwich);
            
            if (frontrunOperation && backrunOperation) {
                bundles.push({
                    operations: [frontrunOperation, backrunOperation],
                    targetTransactions: [sandwich],
                    flashLoanAmount: frontrunOperation.amount,
                    flashLoanProvider: 'balancer',
                    totalGasLimit: 400000,
                    gasPrice: sandwich.gasPrice.add(ethers.utils.parseUnits('2', 'gwei')),
                    timestamp: Date.now()
                });
            }
        }
        
        for (let i = 0; i < arbitrageTransactions.length - 1; i++) {
            const tx1 = arbitrageTransactions[i];
            const tx2 = arbitrageTransactions[i + 1];
            
            const arbitrageOperation = await this.createArbitrageOperation(tx1, tx2);
            
            if (arbitrageOperation) {
                bundles.push({
                    operations: [arbitrageOperation],
                    targetTransactions: [tx1, tx2],
                    flashLoanAmount: arbitrageOperation.amount,
                    flashLoanProvider: 'aave',
                    totalGasLimit: 300000,
                    gasPrice: Math.max(tx1.gasPrice, tx2.gasPrice).add(ethers.utils.parseUnits('1', 'gwei')),
                    timestamp: Date.now()
                });
            }
        }
        
        return bundles;
    }

    generateDexCombinations(dexes, size) {
        const combinations = [];
        
        for (let i = 0; i < dexes.length; i++) {
            for (let j = i + 1; j < dexes.length; j++) {
                if (size === 2) {
                    combinations.push([dexes[i], dexes[j]]);
                }
            }
        }
        
        return combinations;
    }

    generateTransactionCombinations(transactions, size) {
        const combinations = [];
        
        function combine(start, currentCombo) {
            if (currentCombo.length === size) {
                combinations.push([...currentCombo]);
                return;
            }
            
            for (let i = start; i < transactions.length; i++) {
                currentCombo.push(transactions[i]);
                combine(i + 1, currentCombo);
                currentCombo.pop();
            }
        }
        
        combine(0, []);
        return combinations.slice(0, 100);
    }

    async analyzeTransactionCombo(transactions) {
        const operations = [];
        
        for (let i = 0; i < transactions.length - 1; i++) {
            const tx1 = transactions[i];
            const tx2 = transactions[i + 1];
            
            const operation = await this.createOperationFromTxPair(tx1, tx2);
            if (operation) {
                operations.push(operation);
            }
        }
        
        return operations;
    }

    async createOperationFromTxPair(tx1, tx2) {
        try {
            const tx1Info = await this.parseTransaction(tx1);
            const tx2Info = await this.parseTransaction(tx2);
            
            if (!tx1Info || !tx2Info) return null;
            
            if (tx1Info.tokenOut === tx2Info.tokenIn || tx1Info.tokenIn === tx2Info.tokenOut) {
                return {
                    dexA: tx1Info.dex,
                    dexB: tx2Info.dex,
                    tokenIn: tx1Info.tokenIn,
                    tokenOut: tx1Info.tokenOut,
                    amount: tx1Info.amountIn,
                    priceA: tx1Info.price,
                    priceB: tx2Info.price
                };
            }
        } catch {
            return null;
        }
        
        return null;
    }

    async parseTransaction(tx) {
        for (const [dexName, connector] of Object.entries(this.dexConnectors)) {
            try {
                const parsed = await connector.parseTransaction(tx);
                if (parsed) {
                    return { ...parsed, dex: dexName };
                }
            } catch {
                continue;
            }
        }
        
        return null;
    }

    isDexTransaction(tx) {
        const dexAddresses = Object.values(this.dexConnectors).map(connector => 
            connector.getRouterAddress().toLowerCase()
        );
        
        return dexAddresses.includes(tx.to?.toLowerCase());
    }

    isMEVTarget(tx) {
        return this.isDexTransaction(tx) && 
               tx.gasPrice.gt(ethers.utils.parseUnits('20', 'gwei')) &&
               tx.value.gt(ethers.utils.parseEther('0.1'));
    }

    isSandwichable(tx) {
        return this.isDexTransaction(tx) && 
               tx.value.gt(ethers.utils.parseEther('1')) &&
               tx.gasPrice.lt(ethers.utils.parseUnits('50', 'gwei'));
    }

    isArbitrageable(tx) {
        return this.isDexTransaction(tx);
    }

    async createFrontrunOperation(sandwichTx) {
        try {
            const txInfo = await this.parseTransaction(sandwichTx);
            if (!txInfo) return null;
            
            return {
                dexA: txInfo.dex,
                dexB: txInfo.dex,
                tokenIn: txInfo.tokenIn,
                tokenOut: txInfo.tokenOut,
                amount: txInfo.amountIn.div(5),
                priceA: txInfo.price,
                priceB: txInfo.price * 1.02
            };
        } catch {
            return null;
        }
    }

    async createBackrunOperation(sandwichTx) {
        try {
            const txInfo = await this.parseTransaction(sandwichTx);
            if (!txInfo) return null;
            
            return {
                dexA: txInfo.dex,
                dexB: txInfo.dex,
                tokenIn: txInfo.tokenOut,
                tokenOut: txInfo.tokenIn,
                amount: txInfo.amountIn.div(5),
                priceA: txInfo.price * 1.02,
                priceB: txInfo.price
            };
        } catch {
            return null;
        }
    }

    async createArbitrageOperation(tx1, tx2) {
        try {
            const tx1Info = await this.parseTransaction(tx1);
            const tx2Info = await this.parseTransaction(tx2);
            
            if (!tx1Info || !tx2Info) return null;
            
            if (tx1Info.tokenIn === tx2Info.tokenOut && tx1Info.tokenOut === tx2Info.tokenIn) {
                const avgAmount = tx1Info.amountIn.add(tx2Info.amountIn).div(2);
                
                return {
                    dexA: tx1Info.dex,
                    dexB: tx2Info.dex,
                    tokenIn: tx1Info.tokenIn,
                    tokenOut: tx1Info.tokenOut,
                    amount: avgAmount.div(2),
                    priceA: tx1Info.price,
                    priceB: tx2Info.price
                };
            }
        } catch {
            return null;
        }
        
        return null;
    }

    async getRecentBlockTransactions(blockCount) {
        const currentBlock = await this.provider.getBlockNumber();
        const blocks = [];
        
        for (let i = 0; i < blockCount; i++) {
            try {
                const block = await this.provider.getBlockWithTransactions(currentBlock - i);
                blocks.push(block.transactions);
            } catch {
                continue;
            }
        }
        
        return blocks;
    }

    async startBundleMonitoring() {
        setInterval(async () => {
            await this.cleanupExpiredBundles();
        }, 30000);
        
        setInterval(async () => {
            await this.monitorBundlePerformance();
        }, 60000);
    }

    async cleanupExpiredBundles() {
        const currentTime = Date.now();
        
        for (const [bundleId, bundleInfo] of this.activeBundles.entries()) {
            if (currentTime - bundleInfo.timestamp > this.bundleTimeout) {
                this.activeBundles.delete(bundleId);
            }
        }
    }

    async monitorBundlePerformance() {
        for (const [bundleId, bundleInfo] of this.activeBundles.entries()) {
            try {
                const bundleStatus = await this.flashbotProvider.getBundleStats(bundleId);
                
                if (bundleStatus.isIncluded) {
                    this.emit('bundleIncluded', {
                        bundleId,
                        blockNumber: bundleStatus.blockNumber,
                        profit: bundleStatus.profit
                    });
                    this.activeBundles.delete(bundleId);
                } else if (bundleStatus.isFailed) {
                    this.emit('bundleFailed', {
                        bundleId,
                        reason: bundleStatus.failureReason
                    });
                    this.activeBundles.delete(bundleId);
                }
            } catch {
                continue;
            }
        }
    }
}

module.exports = ArbitrageBundle;