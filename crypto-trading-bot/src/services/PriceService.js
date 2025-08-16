const EventEmitter = require('events');
const WebSocket = require('ws');

class PriceService extends EventEmitter {
    constructor(config, exchangeConnectors, dexConnectors, chainConnectors) {
        super();
        this.config = config;
        this.exchangeConnectors = exchangeConnectors;
        this.dexConnectors = dexConnectors;
        this.chainConnectors = chainConnectors;
        this.prices = new Map();
        this.priceHistory = new Map();
        this.websockets = new Map();
        this.updateIntervals = new Map();
        this.priceFeeds = new Map();
        this.lastUpdate = new Map();
        this.volatility = new Map();
    }

    async initialize() {
        await this.setupPriceFeeds();
        await this.startRealTimeUpdates();
        this.startPriceHistory();
        this.emit('initialized');
    }

    async setupPriceFeeds() {
        for (const token of this.config.tokens) {
            this.prices.set(token.symbol, new Map());
            this.priceHistory.set(token.symbol, []);
            this.volatility.set(token.symbol, 0);
        }

        for (const [name, exchange] of Object.entries(this.exchangeConnectors)) {
            try {
                await this.setupExchangeFeed(name, exchange);
            } catch (error) {
                this.emit('error', `Failed to setup ${name} feed: ${error.message}`);
            }
        }

        for (const [name, dex] of Object.entries(this.dexConnectors)) {
            try {
                await this.setupDexFeed(name, dex);
            } catch (error) {
                this.emit('error', `Failed to setup ${name} feed: ${error.message}`);
            }
        }
    }

    async setupExchangeFeed(exchangeName, exchange) {
        const pairs = this.config.tradingPairs || [];
        
        for (const pair of pairs) {
            try {
                await exchange.subscribeToTicker(pair);
                
                exchange.on('tickerUpdate', (symbol, ticker) => {
                    this.updatePrice(symbol, exchangeName, ticker.price, 'CEX');
                });
            } catch (error) {
                continue;
            }
        }
    }

    async setupDexFeed(dexName, dex) {
        const tokens = this.config.tokens || [];
        
        for (const token of tokens) {
            const updateInterval = setInterval(async () => {
                try {
                    const price = await dex.getPrice(token.address, this.config.baseTokens.WETH);
                    if (price > 0) {
                        this.updatePrice(token.symbol, dexName, price, 'DEX');
                    }
                } catch (error) {
                    this.emit('error', `Failed to get ${token.symbol} price from ${dexName}: ${error.message}`);
                }
            }, this.config.updateInterval || 5000);
            
            this.updateIntervals.set(`${dexName}-${token.symbol}`, updateInterval);
        }
    }

    updatePrice(symbol, source, price, type) {
        if (!this.prices.has(symbol)) {
            this.prices.set(symbol, new Map());
        }

        const tokenPrices = this.prices.get(symbol);
        const oldPrice = tokenPrices.get(source);
        
        tokenPrices.set(source, {
            price: parseFloat(price),
            timestamp: Date.now(),
            type,
            change: oldPrice ? ((price - oldPrice.price) / oldPrice.price) * 100 : 0
        });

        this.updatePriceHistory(symbol, price);
        this.calculateVolatility(symbol);
        this.lastUpdate.set(`${symbol}-${source}`, Date.now());

        this.emit('priceUpdate', {
            symbol,
            source,
            price: parseFloat(price),
            type,
            timestamp: Date.now()
        });

        this.checkPriceArbitrage(symbol);
    }

    updatePriceHistory(symbol, price) {
        if (!this.priceHistory.has(symbol)) {
            this.priceHistory.set(symbol, []);
        }

        const history = this.priceHistory.get(symbol);
        history.push({
            price: parseFloat(price),
            timestamp: Date.now()
        });

        if (history.length > 1000) {
            history.shift();
        }
    }

    calculateVolatility(symbol) {
        const history = this.priceHistory.get(symbol);
        if (!history || history.length < 10) return;

        const prices = history.slice(-20).map(h => h.price);
        const returns = [];
        
        for (let i = 1; i < prices.length; i++) {
            returns.push(Math.log(prices[i] / prices[i - 1]));
        }

        const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
        const volatility = Math.sqrt(variance) * Math.sqrt(8760);

        this.volatility.set(symbol, volatility);
    }

    checkPriceArbitrage(symbol) {
        const tokenPrices = this.prices.get(symbol);
        if (!tokenPrices || tokenPrices.size < 2) return;

        const priceArray = Array.from(tokenPrices.entries()).map(([source, data]) => ({
            source,
            price: data.price,
            type: data.type,
            timestamp: data.timestamp
        }));

        priceArray.sort((a, b) => a.price - b.price);
        
        const minPrice = priceArray[0];
        const maxPrice = priceArray[priceArray.length - 1];
        const spread = (maxPrice.price - minPrice.price) / minPrice.price;

        if (spread > this.config.arbitrageThreshold || 0.005) {
            this.emit('arbitrageOpportunity', {
                symbol,
                buySource: minPrice.source,
                sellSource: maxPrice.source,
                buyPrice: minPrice.price,
                sellPrice: maxPrice.price,
                spread,
                profit: spread * 100,
                timestamp: Date.now()
            });
        }
    }

    getPrice(symbol, source = null) {
        if (!this.prices.has(symbol)) return null;

        const tokenPrices = this.prices.get(symbol);
        
        if (source) {
            return tokenPrices.get(source) || null;
        }

        const allPrices = Array.from(tokenPrices.values())
            .filter(p => Date.now() - p.timestamp < 30000);
        
        if (allPrices.length === 0) return null;

        const avgPrice = allPrices.reduce((sum, p) => sum + p.price, 0) / allPrices.length;
        return {
            price: avgPrice,
            sources: allPrices.length,
            timestamp: Math.max(...allPrices.map(p => p.timestamp))
        };
    }

    getAllPrices(symbol) {
        return this.prices.get(symbol) || new Map();
    }

    getPriceHistory(symbol, timeframe = 3600000) {
        const history = this.priceHistory.get(symbol) || [];
        const cutoff = Date.now() - timeframe;
        return history.filter(h => h.timestamp > cutoff);
    }

    getVolatility(symbol) {
        return this.volatility.get(symbol) || 0;
    }

    getPriceSpread(symbol) {
        const tokenPrices = this.prices.get(symbol);
        if (!tokenPrices || tokenPrices.size < 2) return null;

        const prices = Array.from(tokenPrices.values())
            .filter(p => Date.now() - p.timestamp < 30000)
            .map(p => p.price);

        if (prices.length < 2) return null;

        const min = Math.min(...prices);
        const max = Math.max(...prices);
        
        return {
            min,
            max,
            spread: (max - min) / min,
            spreadBps: ((max - min) / min) * 10000
        };
    }

    async refreshPrice(symbol, source) {
        try {
            if (this.exchangeConnectors[source]) {
                const ticker = await this.exchangeConnectors[source].fetchTicker(symbol);
                this.updatePrice(symbol, source, ticker.last, 'CEX');
            } else if (this.dexConnectors[source]) {
                const token = this.config.tokens.find(t => t.symbol === symbol);
                if (token) {
                    const price = await this.dexConnectors[source].getPrice(
                        token.address, 
                        this.config.baseTokens.WETH
                    );
                    this.updatePrice(symbol, source, price, 'DEX');
                }
            }
        } catch (error) {
            this.emit('error', `Failed to refresh ${symbol} price from ${source}: ${error.message}`);
        }
    }

    async startRealTimeUpdates() {
        setInterval(async () => {
            for (const symbol of this.config.tokens.map(t => t.symbol)) {
                for (const source of Object.keys(this.dexConnectors)) {
                    const lastUpdateKey = `${symbol}-${source}`;
                    const lastUpdate = this.lastUpdate.get(lastUpdateKey) || 0;
                    
                    if (Date.now() - lastUpdate > 30000) {
                        await this.refreshPrice(symbol, source);
                    }
                }
            }
        }, 10000);
    }

    startPriceHistory() {
        setInterval(() => {
            for (const [symbol, tokenPrices] of this.prices.entries()) {
                const avgPrice = this.getAveragePrice(symbol);
                if (avgPrice) {
                    this.updatePriceHistory(symbol, avgPrice);
                }
            }
        }, 60000);
    }

    getAveragePrice(symbol) {
        const tokenPrices = this.prices.get(symbol);
        if (!tokenPrices) return null;

        const recentPrices = Array.from(tokenPrices.values())
            .filter(p => Date.now() - p.timestamp < 30000);
        
        if (recentPrices.length === 0) return null;

        return recentPrices.reduce((sum, p) => sum + p.price, 0) / recentPrices.length;
    }

    getTopArbitrageOpportunities(limit = 10) {
        const opportunities = [];
        
        for (const symbol of this.config.tokens.map(t => t.symbol)) {
            const spread = this.getPriceSpread(symbol);
            if (spread && spread.spreadBps > 50) {
                const tokenPrices = this.prices.get(symbol);
                const priceArray = Array.from(tokenPrices.entries());
                
                const minPriceEntry = priceArray.reduce((min, curr) => 
                    curr[1].price < min[1].price ? curr : min
                );
                const maxPriceEntry = priceArray.reduce((max, curr) => 
                    curr[1].price > max[1].price ? curr : max
                );

                opportunities.push({
                    symbol,
                    buySource: minPriceEntry[0],
                    sellSource: maxPriceEntry[0],
                    buyPrice: minPriceEntry[1].price,
                    sellPrice: maxPriceEntry[1].price,
                    spread: spread.spread,
                    spreadBps: spread.spreadBps,
                    volume: this.getTradeVolume(symbol),
                    timestamp: Date.now()
                });
            }
        }

        return opportunities
            .sort((a, b) => b.spreadBps - a.spreadBps)
            .slice(0, limit);
    }

    getTradeVolume(symbol) {
        return 1000000;
    }

    async getPriceFromChain(tokenAddress, chainId) {
        try {
            const connector = this.chainConnectors[chainId];
            if (!connector) return null;

            return await connector.getPrice(tokenAddress);
        } catch (error) {
            return null;
        }
    }

    async getMultiChainPrices(tokenAddress) {
        const prices = {};
        
        for (const [chainId, connector] of Object.entries(this.chainConnectors)) {
            try {
                const price = await connector.getPrice(tokenAddress);
                if (price > 0) {
                    prices[chainId] = {
                        price,
                        timestamp: Date.now()
                    };
                }
            } catch (error) {
                continue;
            }
        }

        return prices;
    }

    startPriceAlerts() {
        this.on('priceUpdate', (data) => {
            const { symbol, price, source } = data;
            const alerts = this.config.priceAlerts?.[symbol] || [];
            
            for (const alert of alerts) {
                if (alert.type === 'above' && price >= alert.price) {
                    this.emit('priceAlert', {
                        symbol,
                        type: 'above',
                        targetPrice: alert.price,
                        currentPrice: price,
                        source,
                        timestamp: Date.now()
                    });
                } else if (alert.type === 'below' && price <= alert.price) {
                    this.emit('priceAlert', {
                        symbol,
                        type: 'below',
                        targetPrice: alert.price,
                        currentPrice: price,
                        source,
                        timestamp: Date.now()
                    });
                }
            }
        });
    }

    getPriceDeviation(symbol) {
        const tokenPrices = this.prices.get(symbol);
        if (!tokenPrices || tokenPrices.size < 2) return null;

        const prices = Array.from(tokenPrices.values())
            .filter(p => Date.now() - p.timestamp < 30000)
            .map(p => p.price);

        const avg = prices.reduce((sum, p) => sum + p, 0) / prices.length;
        const deviations = prices.map(p => Math.abs(p - avg) / avg);
        const maxDeviation = Math.max(...deviations);

        return {
            average: avg,
            maxDeviation,
            standardDeviation: Math.sqrt(
                deviations.reduce((sum, d) => sum + d * d, 0) / deviations.length
            )
        };
    }

    async getExternalPrice(symbol) {
        try {
            const response = await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${symbol}&vs_currencies=usd`);
            const data = await response.json();
            return data[symbol]?.usd || null;
        } catch (error) {
            return null;
        }
    }

    cleanup() {
        for (const interval of this.updateIntervals.values()) {
            clearInterval(interval);
        }
        
        for (const ws of this.websockets.values()) {
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        }
        
        this.updateIntervals.clear();
        this.websockets.clear();
    }

    getStats() {
        const stats = {
            totalTokens: this.prices.size,
            totalSources: 0,
            averageSpread: 0,
            topVolatile: [],
            lastUpdate: Date.now()
        };

        for (const [symbol, tokenPrices] of this.prices.entries()) {
            stats.totalSources += tokenPrices.size;
            
            const volatility = this.getVolatility(symbol);
            stats.topVolatile.push({ symbol, volatility });
        }

        stats.topVolatile.sort((a, b) => b.volatility - a.volatility);
        stats.topVolatile = stats.topVolatile.slice(0, 5);

        return stats;
    }
}

module.exports = PriceService;