const { ethers } = require('ethers');
const EventEmitter = require('events');

class Engine extends EventEmitter {
    constructor(bot) {
        super();
        this.bot = bot;
        this.strategies = [];
        this.scanners = new Map();
        this.opportunities = new Map();
        this.processing = new Set();
        this.intervals = [];
        this.blockSubscriptions = new Map();
        this.mempoolWatchers = new Map();
        this.priceFeeds = new Map();
    }
    
    async initialize() {
        await this.setupScanners();
        await this.setupPriceFeeds();
        await this.setupMempoolWatchers();
        this.setupBlockListeners();
    }
    
    async setupScanners() {
        for (const network of Object.keys(this.bot.providers)) {
            this.scanners.set(network, {
                dex: await this.createDexScanner(network),
                lending: await this.createLendingScanner(network),
                arbitrage: await this.createArbitrageScanner(network)
            });
        }
    }
    
    async createDexScanner(network) {
        const provider = this.bot.providers[network];
        const config = this.bot.config.dex[network];
        
        const scanner = {
            pools: new Map(),
            routers: new Map(),
            factories: new Map()
        };
        
        for (const [protocol, addresses] of Object.entries(config)) {
            scanner.routers.set(protocol, new ethers.Contract(
                addresses.router,
                this.bot.config.abis[protocol].router,
                provider
            ));
            
            if (addresses.factory) {
                scanner.factories.set(protocol, new ethers.Contract(
                    addresses.factory,
                    this.bot.config.abis[protocol].factory,
                    provider
                ));
            }
        }
        
        return scanner;
    }
    
    async createLendingScanner(network) {
        const provider = this.bot.providers[network];
        const config = this.bot.config.lending[network];
        
        const scanner = {
            pools: new Map(),
            oracles: new Map(),
            positions: new Map()
        };
        
        for (const [protocol, addresses] of Object.entries(config)) {
            if (addresses.pool) {
                scanner.pools.set(protocol, new ethers.Contract(
                    addresses.pool,
                    this.bot.config.abis[protocol].pool,
                    provider
                ));
            }
            
            if (addresses.oracle) {
                scanner.oracles.set(protocol, new ethers.Contract(
                    addresses.oracle,
                    this.bot.config.abis[protocol].oracle,
                    provider
                ));
            }
        }
        
        return scanner;
    }
    
    async createArbitrageScanner(network) {
        return {
            pairs: new Map(),
            routes: new Map(),
            profits: new Map()
        };
    }
    
    async setupPriceFeeds() {
        for (const [network, provider] of Object.entries(this.bot.providers)) {
            const feeds = new Map();
            
            for (const [token, address] of Object.entries(this.bot.config.tokens[network])) {
                feeds.set(token, {
                    address,
                    decimals: await this.getTokenDecimals(provider, address),
                    price: ethers.BigNumber.from(0),
                    lastUpdate: 0
                });
            }
            
            this.priceFeeds.set(network, feeds);
        }
    }
    
    async getTokenDecimals(provider, address) {
        if (address === ethers.constants.AddressZero) return 18;
        
        const contract = new ethers.Contract(
            address,
            ['function decimals() view returns (uint8)'],
            provider
        );
        
        try {
            return await contract.decimals();
        } catch {
            return 18;
        }
    }
    
    async setupMempoolWatchers() {
        for (const [network, provider] of Object.entries(this.bot.providers)) {
            if (provider.on) {
                const watcher = {
                    transactions: new Map(),
                    patterns: new Map()
                };
                
                provider.on('pending', async (txHash) => {
                    try {
                        const tx = await provider.getTransaction(txHash);
                        if (tx && tx.data && tx.data.length > 10) {
                            await this.analyzeMempoolTransaction(network, tx);
                        }
                    } catch {}
                });
                
                this.mempoolWatchers.set(network, watcher);
            }
        }
    }
    
    setupBlockListeners() {
        for (const [network, provider] of Object.entries(this.bot.providers)) {
            provider.on('block', async (blockNumber) => {
                await this.processBlock(network, blockNumber);
            });
        }
    }
    
    async processBlock(network, blockNumber) {
        const block = await this.bot.providers[network].getBlockWithTransactions(blockNumber);
        
        await this.updatePrices(network);
        await this.scanOpportunities(network, block);
        await this.checkLiquidations(network);
        
        this.emit('block_processed', { network, blockNumber, timestamp: block.timestamp });
    }
    
    async updatePrices(network) {
        const scanner = this.scanners.get(network);
        const feeds = this.priceFeeds.get(network);
        
        for (const [protocol, router] of scanner.dex.routers) {
            for (const [token, feed] of feeds) {
                if (token === 'ETH') continue;
                
                try {
                    const price = await this.getTokenPrice(network, protocol, feed.address);
                    feed.price = price;
                    feed.lastUpdate = Date.now();
                } catch {}
            }
        }
    }
    
    async getTokenPrice(network, protocol, tokenAddress) {
        const scanner = this.scanners.get(network);
        const router = scanner.dex.routers.get(protocol);
        
        if (!router) return ethers.BigNumber.from(0);
        
        const weth = this.bot.config.tokens[network].WETH;
        const amountIn = ethers.utils.parseEther('1');
        
        try {
            const amounts = await router.getAmountsOut(amountIn, [tokenAddress, weth]);
            return amounts[1];
        } catch {
            return ethers.BigNumber.from(0);
        }
    }
    
    async scanOpportunities(network, block) {
        const opportunities = [];
        
        const dexOps = await this.scanDexArbitrage(network);
        const lendingOps = await this.scanLiquidations(network);
        const crossChainOps = await this.scanCrossChain(network);
        
        opportunities.push(...dexOps, ...lendingOps, ...crossChainOps);
        
        for (const opportunity of opportunities) {
            if (await this.validateOpportunity(opportunity)) {
                this.opportunities.set(opportunity.id, opportunity);
                this.bot.emit('opportunity', opportunity);
            }
        }
    }
    
    async scanDexArbitrage(network) {
        const opportunities = [];
        const scanner = this.scanners.get(network);
        const tokens = this.bot.config.tokens[network];
        
        for (const [tokenA, addressA] of Object.entries(tokens)) {
            for (const [tokenB, addressB] of Object.entries(tokens)) {
                if (tokenA === tokenB) continue;
                
                const prices = await this.getPricesAcrossDexes(network, addressA, addressB);
                const arbitrage = this.calculateArbitrage(prices);
                
                if (arbitrage.profit.gt(ethers.utils.parseEther('0.01'))) {
                    opportunities.push({
                        id: `${network}-${tokenA}-${tokenB}-${Date.now()}`,
                        type: 'dex_arbitrage',
                        network,
                        tokenIn: addressA,
                        tokenOut: addressB,
                        amountIn: arbitrage.optimalAmount,
                        expectedProfit: arbitrage.profit,
                        path: arbitrage.path,
                        timestamp: Date.now()
                    });
                }
            }
        }
        
        return opportunities;
    }
    
    async getPricesAcrossDexes(network, tokenA, tokenB) {
        const scanner = this.scanners.get(network);
        const prices = new Map();
        
        for (const [protocol, router] of scanner.dex.routers) {
            try {
                const amountIn = ethers.utils.parseEther('1');
                const amounts = await router.getAmountsOut(amountIn, [tokenA, tokenB]);
                prices.set(protocol, {
                    rate: amounts[1].mul(10000).div(amountIn),
                    router: router.address
                });
            } catch {}
        }
        
        return prices;
    }
    
    calculateArbitrage(prices) {
        let maxRate = ethers.BigNumber.from(0);
        let minRate = ethers.constants.MaxUint256;
        let buyDex = null;
        let sellDex = null;
        
        for (const [dex, data] of prices) {
            if (data.rate.gt(maxRate)) {
                maxRate = data.rate;
                sellDex = dex;
            }
            if (data.rate.lt(minRate)) {
                minRate = data.rate;
                buyDex = dex;
            }
        }
        
        const spread = maxRate.sub(minRate);
        const profit = spread.mul(10000).div(minRate);
        
        return {
            profit,
            optimalAmount: ethers.utils.parseEther('10'),
            path: [buyDex, sellDex]
        };
    }
    
    async scanLiquidations(network) {
        const opportunities = [];
        const scanner = this.scanners.get(network);
        
        for (const [protocol, pool] of scanner.lending.pools) {
            const positions = await this.getUnhealthyPositions(network, protocol, pool);
            
            for (const position of positions) {
                if (position.healthFactor.lt(ethers.utils.parseEther('1'))) {
                    opportunities.push({
                        id: `${network}-${protocol}-${position.user}-${Date.now()}`,
                        type: 'liquidation',
                        network,
                        protocol,
                        user: position.user,
                        collateral: position.collateral,
                        debt: position.debt,
                        amount: position.debtAmount,
                        bonus: position.liquidationBonus,
                        timestamp: Date.now()
                    });
                }
            }
        }
        
        return opportunities;
    }
    
    async getUnhealthyPositions(network, protocol, pool) {
        const positions = [];
        
        try {
            const filter = pool.filters.Borrow();
            const events = await pool.queryFilter(filter, -1000);
            
            for (const event of events) {
                const user = event.args.user || event.args.onBehalfOf;
                const data = await pool.getUserAccountData(user);
                
                if (data.healthFactor.lt(ethers.utils.parseEther('1.1'))) {
                    positions.push({
                        user,
                        healthFactor: data.healthFactor,
                        collateral: event.args.reserve,
                        debt: event.args.reserve,
                        debtAmount: data.totalDebtETH,
                        liquidationBonus: ethers.BigNumber.from('500')
                    });
                }
            }
        } catch {}
        
        return positions;
    }
    
    async scanCrossChain(network) {
        const opportunities = [];
        const tokens = this.bot.config.tokens[network];
        
        for (const [token, address] of Object.entries(tokens)) {
            const prices = await this.getCrossChainPrices(token);
            
            for (const [networkA, priceA] of prices) {
                for (const [networkB, priceB] of prices) {
                    if (networkA === networkB) continue;
                    
                    const spread = priceB.sub(priceA).mul(10000).div(priceA);
                    
                    if (spread.gt(200)) {
                        opportunities.push({
                            id: `cross-${token}-${networkA}-${networkB}-${Date.now()}`,
                            type: 'cross_chain',
                            token,
                            buyNetwork: networkA,
                            sellNetwork: networkB,
                            buyPrice: priceA,
                            sellPrice: priceB,
                            spread,
                            timestamp: Date.now()
                        });
                    }
                }
            }
        }
        
        return opportunities;
    }
    
    async getCrossChainPrices(token) {
        const prices = new Map();
        
        for (const [network, feeds] of this.priceFeeds) {
            const feed = feeds.get(token);
            if (feed && feed.price.gt(0)) {
                prices.set(network, feed.price);
            }
        }
        
        return prices;
    }
    
    async checkLiquidations(network) {
        const scanner = this.scanners.get(network);
        
        for (const [protocol, oracle] of scanner.lending.oracles) {
            const prices = await this.getOraclePrices(oracle);
            await this.updateLiquidationThresholds(network, protocol, prices);
        }
    }
    
    async getOraclePrices(oracle) {
        const prices = new Map();
        
        try {
            const assets = await oracle.getAssetsList();
            
            for (const asset of assets) {
                const price = await oracle.getAssetPrice(asset);
                prices.set(asset, price);
            }
        } catch {}
        
        return prices;
    }
    
    async updateLiquidationThresholds(network, protocol, prices) {
        const scanner = this.scanners.get(network);
        const positions = scanner.lending.positions.get(protocol) || new Map();
        
        for (const [user, position] of positions) {
            const collateralPrice = prices.get(position.collateral);
            const debtPrice = prices.get(position.debt);
            
            if (collateralPrice && debtPrice) {
                const healthFactor = collateralPrice.mul(position.collateralAmount)
                    .mul(position.ltv)
                    .div(debtPrice.mul(position.debtAmount).mul(10000));
                
                position.healthFactor = healthFactor;
            }
        }
    }
    
    async analyzeMempoolTransaction(network, tx) {
        const opportunities = [];
        
        if (this.isLargeDexTrade(tx)) {
            const impact = await this.calculatePriceImpact(network, tx);
            if (impact.profitable) {
                opportunities.push({
                    id: `mempool-${tx.hash}`,
                    type: 'backrun',
                    network,
                    targetTx: tx.hash,
                    impact,
                    timestamp: Date.now()
                });
            }
        }
        
        for (const opportunity of opportunities) {
            this.bot.emit('opportunity', opportunity);
        }
    }
    
    isLargeDexTrade(tx) {
        const signatures = [
            '0x38ed1739',
            '0x7ff36ab5',
            '0x18cbafe5',
            '0xfb3bdb41'
        ];
        
        return signatures.some(sig => tx.data.startsWith(sig)) && 
               tx.value && tx.value.gt(ethers.utils.parseEther('10'));
    }
    
    async calculatePriceImpact(network, tx) {
        return {
            profitable: Math.random() > 0.9,
            expectedProfit: ethers.utils.parseEther('0.1')
        };
    }
    
    async validateOpportunity(opportunity) {
        if (this.processing.has(opportunity.id)) return false;
        
        const gasPrice = await this.bot.providers[opportunity.network].getGasPrice();
        const estimatedGas = ethers.BigNumber.from('500000');
        const gasCost = gasPrice.mul(estimatedGas);
        
        if (opportunity.expectedProfit && opportunity.expectedProfit.lte(gasCost.mul(2))) {
            return false;
        }
        
        return true;
    }
    
    addStrategy(strategy) {
        this.strategies.push(strategy);
    }
    
    async start() {
        this.intervals.push(
            setInterval(() => this.cleanupOpportunities(), 5000),
            setInterval(() => this.updateMetrics(), 10000)
        );
    }
    
    async stop() {
        for (const interval of this.intervals) {
            clearInterval(interval);
        }
        
        this.intervals = [];
        this.opportunities.clear();
        this.processing.clear();
    }
    
    cleanupOpportunities() {
        const now = Date.now();
        const maxAge = 10000;
        
        for (const [id, opportunity] of this.opportunities) {
            if (now - opportunity.timestamp > maxAge) {
                this.opportunities.delete(id);
            }
        }
    }
    
    updateMetrics() {
        this.emit('metrics', {
            opportunities: this.opportunities.size,
            processing: this.processing.size,
            strategies: this.strategies.length
        });
    }
}

module.exports = Engine;