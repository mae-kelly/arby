const BaseExchange = require('./BaseExchange');
const WebSocket = require('ws');

class Binance extends BaseExchange {
    constructor(config) {
        super({
            ...config,
            exchangeId: 'binance'
        });
        this.websockets = new Map();
        this.streamData = new Map();
        this.listenKey = null;
        this.keepAliveInterval = null;
    }

    async initialize() {
        await super.initialize();
        
        this.exchange.options = {
            ...this.exchange.options,
            adjustForTimeDifference: true,
            recvWindow: 5000,
            timeDifference: 0
        };

        await this.startUserDataStream();
        await this.syncServerTime();
    }

    async syncServerTime() {
        try {
            const serverTime = await this.exchange.fetchTime();
            const localTime = Date.now();
            this.exchange.options.timeDifference = serverTime - localTime;
        } catch (error) {
            this.emit('error', `Failed to sync server time: ${error.message}`);
        }
    }

    async startUserDataStream() {
        try {
            const response = await this.exchange.privatePostUserDataStream();
            this.listenKey = response.listenKey;
            
            this.keepAliveInterval = setInterval(async () => {
                try {
                    await this.exchange.privatePutUserDataStream({ listenKey: this.listenKey });
                } catch (error) {
                    this.emit('error', `Failed to keep alive user data stream: ${error.message}`);
                }
            }, 30 * 60 * 1000);

            this.connectUserDataWebSocket();
        } catch (error) {
            this.emit('error', `Failed to start user data stream: ${error.message}`);
        }
    }

    connectUserDataWebSocket() {
        const ws = new WebSocket(`wss://stream.binance.com:9443/ws/${this.listenKey}`);
        
        ws.on('open', () => {
            this.emit('userDataStreamConnected');
        });

        ws.on('message', (data) => {
            try {
                const message = JSON.parse(data);
                this.handleUserDataMessage(message);
            } catch (error) {
                this.emit('error', `Failed to parse user data message: ${error.message}`);
            }
        });

        ws.on('error', (error) => {
            this.emit('error', `User data WebSocket error: ${error.message}`);
        });

        ws.on('close', () => {
            setTimeout(() => this.connectUserDataWebSocket(), 5000);
        });

        this.websockets.set('userData', ws);
    }

    handleUserDataMessage(message) {
        switch (message.e) {
            case 'executionReport':
                this.emit('orderUpdate', {
                    symbol: message.s,
                    orderId: message.i,
                    clientOrderId: message.c,
                    side: message.S.toLowerCase(),
                    type: message.o.toLowerCase(),
                    status: message.X.toLowerCase(),
                    executedQty: parseFloat(message.z),
                    price: parseFloat(message.p),
                    avgPrice: parseFloat(message.Z) / parseFloat(message.z) || 0
                });
                break;
            case 'outboundAccountPosition':
                this.balances[message.a] = {
                    free: parseFloat(message.f),
                    used: parseFloat(message.l),
                    total: parseFloat(message.f) + parseFloat(message.l)
                };
                this.emit('balanceUpdate', this.balances);
                break;
        }
    }

    async subscribeToTicker(symbol) {
        const streamName = `${symbol.toLowerCase()}@ticker`;
        const wsUrl = `wss://stream.binance.com:9443/ws/${streamName}`;
        
        const ws = new WebSocket(wsUrl);
        
        ws.on('open', () => {
            this.emit('tickerStreamConnected', symbol);
        });

        ws.on('message', (data) => {
            try {
                const ticker = JSON.parse(data);
                this.streamData.set(symbol, {
                    symbol: ticker.s,
                    price: parseFloat(ticker.c),
                    bid: parseFloat(ticker.b),
                    ask: parseFloat(ticker.a),
                    volume: parseFloat(ticker.v),
                    change: parseFloat(ticker.P),
                    timestamp: ticker.E
                });
                this.emit('tickerUpdate', symbol, this.streamData.get(symbol));
            } catch (error) {
                this.emit('error', `Failed to parse ticker data: ${error.message}`);
            }
        });

        this.websockets.set(`ticker_${symbol}`, ws);
    }

    async subscribeToOrderBook(symbol, levels = 20) {
        const streamName = `${symbol.toLowerCase()}@depth${levels}@100ms`;
        const wsUrl = `wss://stream.binance.com:9443/ws/${streamName}`;
        
        const ws = new WebSocket(wsUrl);
        
        ws.on('message', (data) => {
            try {
                const depth = JSON.parse(data);
                const orderBook = {
                    symbol: depth.s,
                    bids: depth.b.map(([price, qty]) => [parseFloat(price), parseFloat(qty)]),
                    asks: depth.a.map(([price, qty]) => [parseFloat(price), parseFloat(qty)]),
                    timestamp: depth.E
                };
                
                this.orderBook.set(symbol, orderBook);
                this.emit('orderBookUpdate', symbol, orderBook);
            } catch (error) {
                this.emit('error', `Failed to parse order book data: ${error.message}`);
            }
        });

        this.websockets.set(`depth_${symbol}`, ws);
    }

    async createOCOOrder(symbol, side, quantity, price, stopPrice, stopLimitPrice) {
        try {
            const params = {
                symbol,
                side: side.toUpperCase(),
                quantity,
                price,
                stopPrice,
                stopLimitPrice,
                stopLimitTimeInForce: 'GTC'
            };

            const order = await this.exchange.privatePostOrderOco(params);
            this.emit('ocoOrderCreated', order);
            return order;
        } catch (error) {
            throw new Error(`Failed to create OCO order: ${error.message}`);
        }
    }

    async createStopLossOrder(symbol, side, quantity, stopPrice, price = null) {
        try {
            const params = {
                symbol,
                side: side.toUpperCase(),
                type: price ? 'STOP_LOSS_LIMIT' : 'STOP_LOSS',
                quantity,
                stopPrice,
                timeInForce: 'GTC'
            };

            if (price) {
                params.price = price;
            }

            const order = await this.exchange.createOrder(symbol, params.type, side, quantity, price || stopPrice, params);
            return order;
        } catch (error) {
            throw new Error(`Failed to create stop loss order: ${error.message}`);
        }
    }

    async getKlines(symbol, interval, limit = 500, startTime = null, endTime = null) {
        try {
            const params = {
                symbol,
                interval,
                limit
            };

            if (startTime) params.startTime = startTime;
            if (endTime) params.endTime = endTime;

            const klines = await this.exchange.publicGetKlines(params);
            return klines.map(kline => ({
                openTime: kline[0],
                open: parseFloat(kline[1]),
                high: parseFloat(kline[2]),
                low: parseFloat(kline[3]),
                close: parseFloat(kline[4]),
                volume: parseFloat(kline[5]),
                closeTime: kline[6],
                quoteVolume: parseFloat(kline[7]),
                trades: kline[8]
            }));
        } catch (error) {
            throw new Error(`Failed to get klines: ${error.message}`);
        }
    }

    async get24hrStats(symbol = null) {
        try {
            const params = symbol ? { symbol } : {};
            const stats = await this.exchange.publicGetTicker24hr(params);
            
            if (Array.isArray(stats)) {
                return stats.map(stat => this.format24hrStat(stat));
            } else {
                return this.format24hrStat(stats);
            }
        } catch (error) {
            throw new Error(`Failed to get 24hr stats: ${error.message}`);
        }
    }

    format24hrStat(stat) {
        return {
            symbol: stat.symbol,
            price: parseFloat(stat.lastPrice),
            priceChange: parseFloat(stat.priceChange),
            priceChangePercent: parseFloat(stat.priceChangePercent),
            volume: parseFloat(stat.volume),
            quoteVolume: parseFloat(stat.quoteVolume),
            high: parseFloat(stat.highPrice),
            low: parseFloat(stat.lowPrice),
            open: parseFloat(stat.openPrice),
            trades: stat.count
        };
    }

    async getAveragePrice(symbol) {
        try {
            const response = await this.exchange.publicGetAvgPrice({ symbol });
            return parseFloat(response.price);
        } catch (error) {
            throw new Error(`Failed to get average price: ${error.message}`);
        }
    }

    async getExchangeInfo() {
        try {
            const info = await this.exchange.publicGetExchangeInfo();
            return {
                timezone: info.timezone,
                serverTime: info.serverTime,
                rateLimits: info.rateLimits,
                symbols: info.symbols.map(symbol => ({
                    symbol: symbol.symbol,
                    status: symbol.status,
                    baseAsset: symbol.baseAsset,
                    quoteAsset: symbol.quoteAsset,
                    baseAssetPrecision: symbol.baseAssetPrecision,
                    quotePrecision: symbol.quotePrecision,
                    filters: symbol.filters
                }))
            };
        } catch (error) {
            throw new Error(`Failed to get exchange info: ${error.message}`);
        }
    }

    async cancelAllOrders(symbol) {
        try {
            const result = await this.exchange.privateDeleteOpenOrders({ symbol });
            this.emit('allOrdersCancelled', symbol);
            return result;
        } catch (error) {
            throw new Error(`Failed to cancel all orders for ${symbol}: ${error.message}`);
        }
    }

    async getAccountSnapshot(type = 'SPOT') {
        try {
            const snapshot = await this.exchange.privateGetAccountSnapshot({
                type,
                timestamp: Date.now(),
                recvWindow: 5000
            });
            return snapshot;
        } catch (error) {
            throw new Error(`Failed to get account snapshot: ${error.message}`);
        }
    }

    async createMarginOrder(symbol, side, type, quantity, price = null, params = {}) {
        try {
            const marginParams = {
                ...params,
                symbol,
                side: side.toUpperCase(),
                type: type.toUpperCase(),
                quantity,
                isIsolated: params.isIsolated || false
            };

            if (price) {
                marginParams.price = price;
            }

            const order = await this.exchange.privatePostMarginOrder(marginParams);
            return order;
        } catch (error) {
            throw new Error(`Failed to create margin order: ${error.message}`);
        }
    }

    async borrowMargin(asset, amount, isIsolated = false, symbol = null) {
        try {
            const params = {
                asset,
                amount,
                isIsolated
            };

            if (isIsolated && symbol) {
                params.symbol = symbol;
            }

            const result = await this.exchange.privatePostMarginLoan(params);
            return result;
        } catch (error) {
            throw new Error(`Failed to borrow margin: ${error.message}`);
        }
    }

    async repayMargin(asset, amount, isIsolated = false, symbol = null) {
        try {
            const params = {
                asset,
                amount,
                isIsolated
            };

            if (isIsolated && symbol) {
                params.symbol = symbol;
            }

            const result = await this.exchange.privatePostMarginRepay(params);
            return result;
        } catch (error) {
            throw new Error(`Failed to repay margin: ${error.message}`);
        }
    }

    getStreamData(symbol) {
        return this.streamData.get(symbol);
    }

    closeWebSocket(name) {
        const ws = this.websockets.get(name);
        if (ws) {
            ws.close();
            this.websockets.delete(name);
        }
    }

    disconnect() {
        for (const [name, ws] of this.websockets.entries()) {
            ws.close();
        }
        this.websockets.clear();

        if (this.keepAliveInterval) {
            clearInterval(this.keepAliveInterval);
        }

        if (this.listenKey) {
            this.exchange.privateDeleteUserDataStream({ listenKey: this.listenKey }).catch(() => {});
        }

        super.disconnect();
    }
}

module.exports = Binance;