const EventEmitter = require('events');
const WebSocket = require('ws');
const { ethers } = require('ethers');

class MempoolService extends EventEmitter {
    constructor(config, chainConnectors) {
        super();
        this.config = config;
        this.chainConnectors = chainConnectors;
        this.pendingTransactions = new Map();
        this.transactionFilters = new Map();
        this.subscribers = new Map();
        this.websockets = new Map();
        this.gasAuctions = new Map();
        this.mevOpportunities = new Map();
        this.cleanupIntervals = new Map();
    }

    async initialize() {
        for (const [chainId, connector] of Object.entries(this.chainConnectors)) {
            await this.initializeChainMempool(chainId, connector);
        }
        
        this.startCleanupRoutines();
        this.emit('initialized');
    }

    async initializeChainMempool(chainId, connector) {
        this.pendingTransactions.set(chainId, new Map());
        this.gasAuctions.set(chainId, new Map());
        this.mevOpportunities.set(chainId, []);

        try {
            if (chainId === '1') {
                await this.setupEthereumMempool(connector);
            } else {
                await this.setupGenericMempool(chainId, connector);
            }
        } catch (error) {
            this.emit('error', `Failed to setup mempool for chain ${chainId}: ${error.message}`);
        }
    }

    async setupEthereumMempool(connector) {
        try {
            const ws = new WebSocket('wss://api.blocknative.com/v0');
            
            ws.on('open', () => {
                ws.send(JSON.stringify({
                    categoryCode: 'initialize',
                    eventCode: 'checkDappId',
                    dappId: this.config.blocknativeApiKey
                }));

                ws.send(JSON.stringify({
                    categoryCode: 'configs',
                    eventCode: 'put',
                    config: {
                        scope: 'global',
                        filters: [{
                            'to': this.config.targetContracts || []
                        }]
                    }
                }));
            });

            ws.on('message', (data) => {
                try {
                    const message = JSON.parse(data);
                    this.handleBlocknativeMessage(message);
                } catch (error) {
                    this.emit('error', `Failed to parse Blocknative message: ${error.message}`);
                }
            });

            this.websockets.set('1-blocknative', ws);
        } catch (error) {
            await this.setupGenericMempool('1', connector);
        }
    }

    async setupGenericMempool(chainId, connector) {
        connector.provider.on('pending', async (txHash) => {
            try {
                const tx = await connector.provider.getTransaction(txHash);
                if (tx) {
                    this.processPendingTransaction(chainId, tx);
                }
            } catch (error) {
                // Ignore failed fetches
            }
        });
    }

    handleBlocknativeMessage(message) {
        if (message.status === 'pending') {
            const tx = {
                hash: message.hash,
                from: message.from,
                to: message.to,
                value: ethers.BigNumber.from(message.value || '0'),
                gasPrice: ethers.BigNumber.from(message.gasPrice || '0'),
                gasLimit: ethers.BigNumber.from(message.gas || '0'),
                data: message.input || '0x',
                nonce: message.nonce,
                timestamp: Date.now()
            };
            
            this.processPendingTransaction('1', tx);
        }
    }

    processPendingTransaction(chainId, tx) {
        if (!tx.hash || !tx.from) return;

        const chainTxs = this.pendingTransactions.get(chainId);
        
        const txData = {
            ...tx,
            chainId,
            timestamp: Date.now(),
            gasPrice: tx.gasPrice || ethers.BigNumber.from(0),
            value: tx.value || ethers.BigNumber.from(0),
            analyzed: false
        };

        chainTxs.set(tx.hash, txData);

        this.analyzePendingTransaction(chainId, txData);
        
        this.emit('pendingTransaction', txData);
        
        this.checkForMEVOpportunities(chainId, txData);
        this.checkGasAuction(chainId, txData);
        this.notifySubscribers(chainId, txData);
    }

    analyzePendingTransaction(chainId, tx) {
        try {
            if (!tx.to || !tx.data || tx.data === '0x') {
                tx.type = 'transfer';
                tx.analyzed = true;
                return;
            }

            const methodId = tx.data.slice(0, 10);
            
            switch (methodId) {
                case '0xa9059cbb':
                    tx.type = 'erc20_transfer';
                    break;
                case '0x38ed1739':
                    tx.type = 'uniswap_swap_exact_tokens_for_tokens';
                    tx.dex = 'uniswap';
                    break;
                case '0x7ff36ab5':
                    tx.type = 'uniswap_swap_exact_eth_for_tokens';
                    tx.dex = 'uniswap';
                    break;
                case '0x18cbafe5':
                    tx.type = 'uniswap_swap_exact_tokens_for_eth';
                    tx.dex = 'uniswap';
                    break;
                case '0x128acb08':
                    tx.type = 'uniswap_v3_exact_input_single';
                    tx.dex = 'uniswap_v3';
                    break;
                case '0x414bf389':
                    tx.type = 'curve_exchange';
                    tx.dex = 'curve';
                    break;
                case '0x945bcec9':
                    tx.type = 'balancer_batch_swap';
                    tx.dex = 'balancer';
                    break;
                case '0xe8e33700':
                    tx.type = 'contract_deployment';
                    break;
                default:
                    tx.type = 'contract_interaction';
            }

            if (tx.type.includes('swap') || tx.type.includes('exchange')) {
                tx.isTrading = true;
                this.extractTradingInfo(chainId, tx);
            }

            tx.analyzed = true;
        } catch (error) {
            tx.analyzed = false;
            this.emit('error', `Failed to analyze transaction ${tx.hash}: ${error.message}`);
        }
    }

    extractTradingInfo(chainId, tx) {
        try {
            const connector = this.chainConnectors[chainId];
            if (!connector) return;

            if (tx.type === 'uniswap_swap_exact_tokens_for_tokens') {
                const decoded = this.decodeUniswapSwap(tx.data);
                if (decoded) {
                    tx.swapInfo = decoded;
                }
            } else if (tx.type === 'uniswap_v3_exact_input_single') {
                const decoded = this.decodeUniswapV3Swap(tx.data);
                if (decoded) {
                    tx.swapInfo = decoded;
                }
            }

            if (tx.swapInfo) {
                this.emit('tradingTransaction', {
                    chainId,
                    hash: tx.hash,
                    type: tx.type,
                    dex: tx.dex,
                    swapInfo: tx.swapInfo,
                    gasPrice: tx.gasPrice,
                    timestamp: tx.timestamp
                });
            }
        } catch (error) {
            this.emit('error', `Failed to extract trading info: ${error.message}`);
        }
    }

    decodeUniswapSwap(data) {
        try {
            const iface = new ethers.utils.Interface([
                'function swapExactTokensForTokens(uint amountIn, uint amountOutMin, address[] calldata path, address to, uint deadline)'
            ]);
            
            const decoded = iface.parseTransaction({ data });
            return {
                amountIn: decoded.args.amountIn,
                amountOutMin: decoded.args.amountOutMin,
                path: decoded.args.path,
                deadline: decoded.args.deadline
            };
        } catch (error) {
            return null;
        }
    }

    decodeUniswapV3Swap(data) {
        try {
            const iface = new ethers.utils.Interface([
                'function exactInputSingle((address tokenIn, address tokenOut, uint24 fee, address recipient, uint256 deadline, uint256 amountIn, uint256 amountOutMinimum, uint160 sqrtPriceLimitX96))'
            ]);
            
            const decoded = iface.parseTransaction({ data });
            return {
                tokenIn: decoded.args[0].tokenIn,
                tokenOut: decoded.args[0].tokenOut,
                fee: decoded.args[0].fee,
                amountIn: decoded.args[0].amountIn,
                amountOutMinimum: decoded.args[0].amountOutMinimum
            };
        } catch (error) {
            return null;
        }
    }

    checkForMEVOpportunities(chainId, tx) {
        if (!tx.isTrading) return;

        const chainOpportunities = this.mevOpportunities.get(chainId);
        const recentTxs = Array.from(this.pendingTransactions.get(chainId).values())
            .filter(t => t.isTrading && Date.now() - t.timestamp < 30000);

        for (const otherTx of recentTxs) {
            if (otherTx.hash === tx.hash) continue;

            const opportunity = this.detectArbitrageOpportunity(tx, otherTx);
            if (opportunity) {
                chainOpportunities.push({
                    type: 'arbitrage',
                    transactions: [tx.hash, otherTx.hash],
                    opportunity,
                    timestamp: Date.now()
                });

                this.emit('mevOpportunity', {
                    chainId,
                    type: 'arbitrage',
                    opportunity,
                    transactions: [tx, otherTx]
                });
            }

            const sandwichOpp = this.detectSandwichOpportunity(tx, otherTx);
            if (sandwichOpp) {
                chainOpportunities.push({
                    type: 'sandwich',
                    targetTransaction: tx.hash,
                    opportunity: sandwichOpp,
                    timestamp: Date.now()
                });

                this.emit('mevOpportunity', {
                    chainId,
                    type: 'sandwich',
                    opportunity: sandwichOpp,
                    targetTransaction: tx
                });
            }
        }
    }

    detectArbitrageOpportunity(tx1, tx2) {
        if (!tx1.swapInfo || !tx2.swapInfo) return null;

        const path1 = tx1.swapInfo.path || [tx1.swapInfo.tokenIn, tx1.swapInfo.tokenOut];
        const path2 = tx2.swapInfo.path || [tx2.swapInfo.tokenIn, tx2.swapInfo.tokenOut];

        if (path1.length === 2 && path2.length === 2) {
            if (path1[0] === path2[1] && path1[1] === path2[0]) {
                return {
                    token0: path1[0],
                    token1: path1[1],
                    dex1: tx1.dex,
                    dex2: tx2.dex,
                    amount1: tx1.swapInfo.amountIn,
                    amount2: tx2.swapInfo.amountIn
                };
            }
        }

        return null;
    }

    detectSandwichOpportunity(targetTx, otherTx) {
        if (!targetTx.swapInfo || targetTx.gasPrice.lt(ethers.utils.parseUnits('20', 'gwei'))) {
            return null;
        }

        const targetAmount = targetTx.swapInfo.amountIn;
        if (targetAmount.lt(ethers.utils.parseEther('1'))) {
            return null;
        }

        return {
            targetTx: targetTx.hash,
            tokenIn: targetTx.swapInfo.tokenIn || targetTx.swapInfo.path[0],
            tokenOut: targetTx.swapInfo.tokenOut || targetTx.swapInfo.path[targetTx.swapInfo.path.length - 1],
            targetAmount,
            dex: targetTx.dex,
            estimatedProfit: targetAmount.div(100)
        };
    }

    checkGasAuction(chainId, tx) {
        const chainAuctions = this.gasAuctions.get(chainId);
        const key = `${tx.from}-${tx.nonce}`;
        
        if (chainAuctions.has(key)) {
            const existing = chainAuctions.get(key);
            if (tx.gasPrice.gt(existing.gasPrice)) {
                existing.gasPrice = tx.gasPrice;
                existing.hash = tx.hash;
                existing.updatedAt = Date.now();
                
                this.emit('gasAuction', {
                    chainId,
                    from: tx.from,
                    nonce: tx.nonce,
                    oldGasPrice: existing.gasPrice,
                    newGasPrice: tx.gasPrice,
                    newHash: tx.hash
                });
            }
        } else {
            chainAuctions.set(key, {
                from: tx.from,
                nonce: tx.nonce,
                hash: tx.hash,
                gasPrice: tx.gasPrice,
                createdAt: Date.now(),
                updatedAt: Date.now()
            });
        }
    }

    notifySubscribers(chainId, tx) {
        const filters = this.transactionFilters.get(chainId) || [];
        
        for (const filter of filters) {
            if (this.matchesFilter(tx, filter.criteria)) {
                filter.callback(tx);
            }
        }
    }

    matchesFilter(tx, criteria) {
        if (criteria.to && tx.to !== criteria.to) return false;
        if (criteria.from && tx.from !== criteria.from) return false;
        if (criteria.type && tx.type !== criteria.type) return false;
        if (criteria.minGasPrice && tx.gasPrice.lt(criteria.minGasPrice)) return false;
        if (criteria.maxGasPrice && tx.gasPrice.gt(criteria.maxGasPrice)) return false;
        if (criteria.minValue && tx.value.lt(criteria.minValue)) return false;
        if (criteria.dex && tx.dex !== criteria.dex) return false;
        
        return true;
    }

    subscribeToTransactions(chainId, criteria, callback) {
        if (!this.transactionFilters.has(chainId)) {
            this.transactionFilters.set(chainId, []);
        }
        
        const filterId = `${chainId}-${Date.now()}-${Math.random()}`;
        this.transactionFilters.get(chainId).push({
            id: filterId,
            criteria,
            callback
        });
        
        return filterId;
    }

    unsubscribeFromTransactions(filterId) {
        for (const [chainId, filters] of this.transactionFilters.entries()) {
            const index = filters.findIndex(f => f.id === filterId);
            if (index !== -1) {
                filters.splice(index, 1);
                return true;
            }
        }
        return false;
    }

    getPendingTransactions(chainId, filter = {}) {
        const chainTxs = this.pendingTransactions.get(chainId);
        if (!chainTxs) return [];

        let transactions = Array.from(chainTxs.values());
        
        if (filter.type) {
            transactions = transactions.filter(tx => tx.type === filter.type);
        }
        
        if (filter.isTrading) {
            transactions = transactions.filter(tx => tx.isTrading);
        }
        
        if (filter.minGasPrice) {
            transactions = transactions.filter(tx => tx.gasPrice.gte(filter.minGasPrice));
        }
        
        if (filter.since) {
            transactions = transactions.filter(tx => tx.timestamp > filter.since);
        }

        return transactions.sort((a, b) => b.timestamp - a.timestamp);
    }

    getMEVOpportunities(chainId, type = null) {
        const opportunities = this.mevOpportunities.get(chainId) || [];
        
        if (type) {
            return opportunities.filter(opp => opp.type === type);
        }
        
        return opportunities;
    }

    getGasAuctions(chainId) {
        const auctions = this.gasAuctions.get(chainId) || new Map();
        return Array.from(auctions.values());
    }

    getTransactionByHash(chainId, hash) {
        const chainTxs = this.pendingTransactions.get(chainId);
        return chainTxs?.get(hash) || null;
    }

    async getTransactionPosition(chainId, hash) {
        try {
            const connector = this.chainConnectors[chainId];
            if (!connector) return null;

            const tx = await connector.provider.getTransaction(hash);
            if (!tx) return null;

            const block = await connector.provider.getBlock('pending', true);
            const position = block.transactions.findIndex(txHash => txHash === hash);
            
            return {
                position: position + 1,
                totalPending: block.transactions.length,
                gasPrice: tx.gasPrice,
                estimatedWaitTime: this.estimateWaitTime(chainId, tx.gasPrice)
            };
        } catch (error) {
            return null;
        }
    }

    estimateWaitTime(chainId, gasPrice) {
        const gasPriceGwei = parseFloat(ethers.utils.formatUnits(gasPrice, 'gwei'));
        
        if (chainId === '1') {
            if (gasPriceGwei > 50) return '< 1 minute';
            if (gasPriceGwei > 30) return '1-3 minutes';
            if (gasPriceGwei > 20) return '3-5 minutes';
            return '5+ minutes';
        } else if (chainId === '56') {
            if (gasPriceGwei > 10) return '< 30 seconds';
            if (gasPriceGwei > 5) return '30-60 seconds';
            return '1-2 minutes';
        } else if (chainId === '137') {
            if (gasPriceGwei > 40) return '< 30 seconds';
            if (gasPriceGwei > 30) return '30-60 seconds';
            return '1-2 minutes';
        }
        
        return 'Unknown';
    }

    startCleanupRoutines() {
        for (const chainId of Object.keys(this.chainConnectors)) {
            const interval = setInterval(() => {
                this.cleanupOldTransactions(chainId);
                this.cleanupOldOpportunities(chainId);
                this.cleanupOldAuctions(chainId);
            }, 60000);
            
            this.cleanupIntervals.set(chainId, interval);
        }
    }

    cleanupOldTransactions(chainId) {
        const chainTxs = this.pendingTransactions.get(chainId);
        if (!chainTxs) return;

        const cutoff = Date.now() - 300000;
        for (const [hash, tx] of chainTxs.entries()) {
            if (tx.timestamp < cutoff) {
                chainTxs.delete(hash);
            }
        }
    }

    cleanupOldOpportunities(chainId) {
        const opportunities = this.mevOpportunities.get(chainId);
        if (!opportunities) return;

        const cutoff = Date.now() - 60000;
        const filtered = opportunities.filter(opp => opp.timestamp > cutoff);
        this.mevOpportunities.set(chainId, filtered);
    }

    cleanupOldAuctions(chainId) {
        const auctions = this.gasAuctions.get(chainId);
        if (!auctions) return;

        const cutoff = Date.now() - 180000;
        for (const [key, auction] of auctions.entries()) {
            if (auction.updatedAt < cutoff) {
                auctions.delete(key);
            }
        }
    }

    getStats(chainId) {
        const chainTxs = this.pendingTransactions.get(chainId);
        const opportunities = this.mevOpportunities.get(chainId) || [];
        const auctions = this.gasAuctions.get(chainId) || new Map();

        if (!chainTxs) return null;

        const transactions = Array.from(chainTxs.values());
        const tradingTxs = transactions.filter(tx => tx.isTrading);
        
        return {
            totalPending: transactions.length,
            tradingTransactions: tradingTxs.length,
            mevOpportunities: opportunities.length,
            gasAuctions: auctions.size,
            averageGasPrice: this.calculateAverageGasPrice(transactions),
            lastUpdate: Date.now()
        };
    }

    calculateAverageGasPrice(transactions) {
        if (transactions.length === 0) return '0';

        const total = transactions.reduce((sum, tx) => 
            sum.add(tx.gasPrice), ethers.BigNumber.from(0)
        );
        
        return ethers.utils.formatUnits(total.div(transactions.length), 'gwei');
    }

    cleanup() {
        for (const interval of this.cleanupIntervals.values()) {
            clearInterval(interval);
        }
        
        for (const ws of this.websockets.values()) {
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        }
        
        this.cleanupIntervals.clear();
        this.websockets.clear();
    }
}

module.exports = MempoolService;