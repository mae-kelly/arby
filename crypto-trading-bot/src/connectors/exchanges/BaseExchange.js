const EventEmitter = require('events');
const ccxt = require('ccxt');

class BaseExchange extends EventEmitter {
    constructor(config) {
        super();
        this.config = config;
        this.exchange = null;
        this.isConnected = false;
        this.lastPriceUpdate = new Map();
        this.orderBook = new Map();
        this.balances = {};
        this.rateLimits = new Map();
        this.retryCount = 0;
        this.maxRetries = 3;
    }

    async initialize() {
        this.exchange = new ccxt[this.config.exchangeId]({
            apiKey: this.config.apiKey,
            secret: this.config.secret,
            password: this.config.passphrase,
            sandbox: false,
            enableRateLimit: true,
            timeout: 30000,
            options: {
                adjustForTimeDifference: true,
                recvWindow: 10000
            }
        });

        await this.exchange.loadMarkets();
        this.isConnected = true;
        await this.updateBalances();
        this.emit('connected');
    }

    async fetchTicker(symbol) {
        try {
            const ticker = await this.exchange.fetchTicker(symbol);
            this.lastPriceUpdate.set(symbol, Date.now());
            return ticker;
        } catch (error) {
            throw new Error(`Failed to fetch ticker for ${symbol}: ${error.message}`);
        }
    }

    async fetchOrderBook(symbol, limit = 20) {
        try {
            const orderBook = await this.exchange.fetchOrderBook(symbol, limit);
            this.orderBook.set(symbol, {
                ...orderBook,
                timestamp: Date.now()
            });
            return orderBook;
        } catch (error) {
            throw new Error(`Failed to fetch order book for ${symbol}: ${error.message}`);
        }
    }

    async createOrder(symbol, type, side, amount, price, params = {}) {
        try {
            const order = await this.exchange.createOrder(symbol, type, side, amount, price, params);
            this.emit('orderCreated', order);
            return order;
        } catch (error) {
            throw new Error(`Failed to create order: ${error.message}`);
        }
    }

    async createMarketBuyOrder(symbol, amount, price = null, params = {}) {
        return await this.createOrder(symbol, 'market', 'buy', amount, price, params);
    }

    async createMarketSellOrder(symbol, amount, price = null, params = {}) {
        return await this.createOrder(symbol, 'market', 'sell', amount, price, params);
    }

    async createLimitBuyOrder(symbol, amount, price, params = {}) {
        return await this.createOrder(symbol, 'limit', 'buy', amount, price, params);
    }

    async createLimitSellOrder(symbol, amount, price, params = {}) {
        return await this.createOrder(symbol, 'limit', 'sell', amount, price, params);
    }

    async cancelOrder(id, symbol) {
        try {
            const result = await this.exchange.cancelOrder(id, symbol);
            this.emit('orderCancelled', { id, symbol });
            return result;
        } catch (error) {
            throw new Error(`Failed to cancel order ${id}: ${error.message}`);
        }
    }

    async fetchOrder(id, symbol) {
        try {
            return await this.exchange.fetchOrder(id, symbol);
        } catch (error) {
            throw new Error(`Failed to fetch order ${id}: ${error.message}`);
        }
    }

    async fetchOpenOrders(symbol = null) {
        try {
            return await this.exchange.fetchOpenOrders(symbol);
        } catch (error) {
            throw new Error(`Failed to fetch open orders: ${error.message}`);
        }
    }

    async fetchBalance() {
        try {
            this.balances = await this.exchange.fetchBalance();
            return this.balances;
        } catch (error) {
            throw new Error(`Failed to fetch balance: ${error.message}`);
        }
    }

    async updateBalances() {
        await this.fetchBalance();
        this.emit('balanceUpdated', this.balances);
    }

    getBalance(currency) {
        return this.balances[currency] || { free: 0, used: 0, total: 0 };
    }

    async getPrice(symbol) {
        const cached = this.lastPriceUpdate.get(symbol);
        if (cached && Date.now() - cached < 5000) {
            const orderBook = this.orderBook.get(symbol);
            if (orderBook && orderBook.bids.length > 0 && orderBook.asks.length > 0) {
                return (orderBook.bids[0][0] + orderBook.asks[0][0]) / 2;
            }
        }

        const ticker = await this.fetchTicker(symbol);
        return ticker.last;
    }

    async getLiquidity(symbol) {
        const orderBook = await this.fetchOrderBook(symbol, 10);
        const bidLiquidity = orderBook.bids.reduce((sum, [price, amount]) => sum + amount, 0);
        const askLiquidity = orderBook.asks.reduce((sum, [price, amount]) => sum + amount, 0);
        return Math.min(bidLiquidity, askLiquidity);
    }

    async getSpread(symbol) {
        const orderBook = await this.fetchOrderBook(symbol, 1);
        if (orderBook.bids.length === 0 || orderBook.asks.length === 0) {
            return 0;
        }
        const bid = orderBook.bids[0][0];
        const ask = orderBook.asks[0][0];
        return (ask - bid) / bid;
    }

    async withdraw(currency, amount, address, tag = null, params = {}) {
        try {
            const withdrawal = await this.exchange.withdraw(currency, amount, address, tag, params);
            this.emit('withdrawalCreated', withdrawal);
            return withdrawal;
        } catch (error) {
            throw new Error(`Failed to withdraw ${amount} ${currency}: ${error.message}`);
        }
    }

    async fetchDeposits(currency = null, since = null, limit = null, params = {}) {
        try {
            return await this.exchange.fetchDeposits(currency, since, limit, params);
        } catch (error) {
            throw new Error(`Failed to fetch deposits: ${error.message}`);
        }
    }

    async fetchWithdrawals(currency = null, since = null, limit = null, params = {}) {
        try {
            return await this.exchange.fetchWithdrawals(currency, since, limit, params);
        } catch (error) {
            throw new Error(`Failed to fetch withdrawals: ${error.message}`);
        }
    }

    async retryOperation(operation, ...args) {
        for (let i = 0; i < this.maxRetries; i++) {
            try {
                return await operation.apply(this, args);
            } catch (error) {
                if (i === this.maxRetries - 1) throw error;
                await this.sleep(Math.pow(2, i) * 1000);
            }
        }
    }

    checkRateLimit(endpoint) {
        const lastCall = this.rateLimits.get(endpoint);
        const minInterval = this.exchange.rateLimit || 1000;
        
        if (lastCall && Date.now() - lastCall < minInterval) {
            return false;
        }
        
        this.rateLimits.set(endpoint, Date.now());
        return true;
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async executeArbitrage(buySymbol, sellSymbol, amount) {
        try {
            const buyOrder = await this.createMarketBuyOrder(buySymbol, amount);
            const sellOrder = await this.createMarketSellOrder(sellSymbol, amount);
            
            return {
                buyOrder,
                sellOrder,
                profit: sellOrder.average - buyOrder.average,
                timestamp: Date.now()
            };
        } catch (error) {
            throw new Error(`Arbitrage execution failed: ${error.message}`);
        }
    }

    async getMarketDepth(symbol, limit = 50) {
        const orderBook = await this.fetchOrderBook(symbol, limit);
        
        let bidDepth = 0;
        let askDepth = 0;
        
        for (const [price, amount] of orderBook.bids) {
            bidDepth += price * amount;
        }
        
        for (const [price, amount] of orderBook.asks) {
            askDepth += price * amount;
        }
        
        return { bidDepth, askDepth, total: bidDepth + askDepth };
    }

    async waitForOrderFill(orderId, symbol, timeout = 30000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            const order = await this.fetchOrder(orderId, symbol);
            
            if (order.status === 'closed') {
                return order;
            }
            
            await this.sleep(1000);
        }
        
        throw new Error(`Order ${orderId} not filled within timeout`);
    }

    disconnect() {
        this.isConnected = false;
        this.emit('disconnected');
    }

    getExchangeInfo() {
        return {
            id: this.config.exchangeId,
            name: this.exchange.name,
            countries: this.exchange.countries,
            rateLimit: this.exchange.rateLimit,
            fees: this.exchange.fees,
            markets: Object.keys(this.exchange.markets)
        };
    }
}

module.exports = BaseExchange;