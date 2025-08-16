const BaseExchange = require('./BaseExchange');
const WebSocket = require('ws');
const crypto = require('crypto');

class Coinbase extends BaseExchange {
    constructor(config) {
        super({
            ...config,
            exchangeId: 'coinbasepro'
        });
        this.wsUrl = 'wss://ws-feed.pro.coinbase.com';
        this.websocket = null;
        this.subscriptions = new Set();
        this.channels = new Map();
        this.sequenceNumber = 0;
        this.orderBookSequence = new Map();
    }

    async initialize() {
        await super.initialize();
        
        this.exchange.options = {
            ...this.exchange.options,
            advanced: true,
            pro: true
        };

        await this.connectWebSocket();
    }

    async connectWebSocket() {
        this.websocket = new WebSocket(this.wsUrl);
        
        this.websocket.on('open', () => {
            this.emit('websocketConnected');
            this.authenticateWebSocket();
        });

        this.websocket.on('message', (data) => {
            try {
                const message = JSON.parse(data);
                this.handleWebSocketMessage(message);
            } catch (error) {
                this.emit('error', `Failed to parse WebSocket message: ${error.message}`);
            }
        });

        this.websocket.on('error', (error) => {
            this.emit('error', `WebSocket error: ${error.message}`);
        });

        this.websocket.on('close', () => {
            setTimeout(() => this.connectWebSocket(), 5000);
        });
    }

    authenticateWebSocket() {
        const timestamp = Date.now() / 1000;
        const method = 'GET';
        const requestPath = '/users/self/verify';
        const body = '';
        
        const message = timestamp + method + requestPath + body;
        const signature = crypto.createHmac('sha256', Buffer.from(this.config.secret, 'base64'))
            .update(message)
            .digest('base64');

        const authMessage = {
            type: 'subscribe',
            channels: ['user', 'heartbeat'],
            timestamp: timestamp.toString(),
            key: this.config.apiKey,
            signature: signature,
            passphrase: this.config.passphrase
        };

        this.websocket.send(JSON.stringify(authMessage));
    }

    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'subscriptions':
                this.emit('subscriptionConfirmed', message.channels);
                break;
            case 'heartbeat':
                this.emit('heartbeat', message);
                break;
            case 'ticker':
                this.handleTickerMessage(message);
                break;
            case 'l2update':
                this.handleLevel2Update(message);
                break;
            case 'snapshot':
                this.handleSnapshot(message);
                break;
            case 'received':
            case 'open':
            case 'done':
            case 'match':
                this.handleOrderMessage(message);
                break;
            case 'error':
                this.emit('error', `WebSocket error: ${message.message}`);
                break;
        }
    }

    handleTickerMessage(message) {
        this.streamData.set(message.product_id, {
            symbol: message.product_id,
            price: parseFloat(message.price),
            bid: parseFloat(message.best_bid),
            ask: parseFloat(message.best_ask),
            volume: parseFloat(message.volume_24h),
            time: message.time,
            sequence: message.sequence
        });
        this.emit('tickerUpdate', message.product_id, this.streamData.get(message.product_id));
    }

    handleLevel2Update(message) {
        const symbol = message.product_id;
        let orderBook = this.orderBook.get(symbol);
        
        if (!orderBook || message.sequence <= this.orderBookSequence.get(symbol)) {
            return;
        }

        for (const [side, price, size] of message.changes) {
            const priceLevel = parseFloat(price);
            const sizeLevel = parseFloat(size);
            
            if (side === 'buy') {
                if (sizeLevel === 0) {
                    orderBook.bids = orderBook.bids.filter(([p]) => p !== priceLevel);
                } else {
                    const index = orderBook.bids.findIndex(([p]) => p === priceLevel);
                    if (index >= 0) {
                        orderBook.bids[index] = [priceLevel, sizeLevel];
                    } else {
                        orderBook.bids.push([priceLevel, sizeLevel]);
                        orderBook.bids.sort((a, b) => b[0] - a[0]);
                    }
                }
            } else {
                if (sizeLevel === 0) {
                    orderBook.asks = orderBook.asks.filter(([p]) => p !== priceLevel);
                } else {
                    const index = orderBook.asks.findIndex(([p]) => p === priceLevel);
                    if (index >= 0) {
                        orderBook.asks[index] = [priceLevel, sizeLevel];
                    } else {
                        orderBook.asks.push([priceLevel, sizeLevel]);
                        orderBook.asks.sort((a, b) => a[0] - b[0]);
                    }
                }
            }
        }

        this.orderBookSequence.set(symbol, message.sequence);
        this.emit('orderBookUpdate', symbol, orderBook);
    }

    handleSnapshot(message) {
        const orderBook = {
            symbol: message.product_id,
            bids: message.bids.map(([price, size]) => [parseFloat(price), parseFloat(size)]),
            asks: message.asks.map(([price, size]) => [parseFloat(price), parseFloat(size)]),
            timestamp: Date.now()
        };

        this.orderBook.set(message.product_id, orderBook);
        this.orderBookSequence.set(message.product_id, 0);
        this.emit('orderBookSnapshot', message.product_id, orderBook);
    }

    handleOrderMessage(message) {
        this.emit('orderUpdate', {
            type: message.type,
            orderId: message.order_id,
            clientOrderId: message.client_oid,
            symbol: message.product_id,
            side: message.side,
            orderType: message.order_type,
            size: parseFloat(message.size || 0),
            price: parseFloat(message.price || 0),
            remainingSize: parseFloat(message.remaining_size || 0),
            reason: message.reason,
            sequence: message.sequence,
            time: message.time
        });
    }

    async subscribeToTicker(symbol) {
        const subscribeMessage = {
            type: 'subscribe',
            product_ids: [symbol],
            channels: ['ticker']
        };

        this.websocket.send(JSON.stringify(subscribeMessage));
        this.subscriptions.add(`ticker_${symbol}`);
    }

    async subscribeToLevel2(symbol) {
        const subscribeMessage = {
            type: 'subscribe',
            product_ids: [symbol],
            channels: ['level2']
        };

        this.websocket.send(JSON.stringify(subscribeMessage));
        this.subscriptions.add(`level2_${symbol}`);
    }

    async subscribeToMatches(symbol) {
        const subscribeMessage = {
            type: 'subscribe',
            product_ids: [symbol],
            channels: ['matches']
        };

        this.websocket.send(JSON.stringify(subscribeMessage));
        this.subscriptions.add(`matches_${symbol}`);
    }

    async createAdvancedOrder(symbol, side, orderType, size, params = {}) {
        try {
            const orderParams = {
                product_id: symbol,
                side,
                type: orderType,
                size: size.toString(),
                ...params
            };

            if (params.price) {
                orderParams.price = params.price.toString();
            }

            if (params.stop) {
                orderParams.stop = params.stop;
                orderParams.stop_price = params.stopPrice.toString();
            }

            if (params.timeInForce) {
                orderParams.time_in_force = params.timeInForce;
            }

            const order = await this.exchange.privatePostOrders(orderParams);
            this.emit('advancedOrderCreated', order);
            return order;
        } catch (error) {
            throw new Error(`Failed to create advanced order: ${error.message}`);
        }
    }

    async createStopOrder(symbol, side, size, stopPrice, limitPrice = null) {
        const params = {
            stop: side === 'buy' ? 'entry' : 'loss',
            stopPrice
        };

        if (limitPrice) {
            params.price = limitPrice;
            return await this.createAdvancedOrder(symbol, side, 'limit', size, params);
        } else {
            return await this.createAdvancedOrder(symbol, side, 'market', size, params);
        }
    }

    async getFills(orderId = null, productId = null, before = null, after = null, limit = 100) {
        try {
            const params = { limit };
            
            if (orderId) params.order_id = orderId;
            if (productId) params.product_id = productId;
            if (before) params.before = before;
            if (after) params.after = after;

            const fills = await this.exchange.privateGetFills(params);
            return fills.map(fill => ({
                orderId: fill.order_id,
                tradeId: fill.trade_id,
                productId: fill.product_id,
                side: fill.side,
                size: parseFloat(fill.size),
                price: parseFloat(fill.price),
                fee: parseFloat(fill.fee),
                createdAt: fill.created_at,
                liquidity: fill.liquidity,
                settled: fill.settled
            }));
        } catch (error) {
            throw new Error(`Failed to get fills: ${error.message}`);
        }
    }

    async getAccounts() {
        try {
            const accounts = await this.exchange.privateGetAccounts();
            return accounts.map(account => ({
                id: account.id,
                currency: account.currency,
                balance: parseFloat(account.balance),
                available: parseFloat(account.available),
                hold: parseFloat(account.hold),
                profileId: account.profile_id,
                tradingEnabled: account.trading_enabled
            }));
        } catch (error) {
            throw new Error(`Failed to get accounts: ${error.message}`);
        }
    }

    async getAccountHistory(accountId, before = null, after = null, limit = 100) {
        try {
            const params = { limit };
            
            if (before) params.before = before;
            if (after) params.after = after;

            const history = await this.exchange.privateGetAccountsAccountIdLedger(
                { account_id: accountId, ...params }
            );
            
            return history.map(entry => ({
                id: entry.id,
                amount: parseFloat(entry.amount),
                balance: parseFloat(entry.balance),
                type: entry.type,
                details: entry.details,
                createdAt: entry.created_at
            }));
        } catch (error) {
            throw new Error(`Failed to get account history: ${error.message}`);
        }
    }

    async getProductStats(symbol) {
        try {
            const stats = await this.exchange.publicGetProductsProductIdStats({
                product_id: symbol
            });
            
            return {
                open: parseFloat(stats.open),
                high: parseFloat(stats.high),
                low: parseFloat(stats.low),
                volume: parseFloat(stats.volume),
                last: parseFloat(stats.last),
                volume30Day: parseFloat(stats.volume_30day)
            };
        } catch (error) {
            throw new Error(`Failed to get product stats: ${error.message}`);
        }
    }

    async getCandles(symbol, start, end, granularity) {
        try {
            const params = {
                product_id: symbol,
                start: start,
                end: end,
                granularity: granularity
            };

            const candles = await this.exchange.publicGetProductsProductIdCandles(params);
            return candles.map(candle => ({
                time: candle[0],
                low: candle[1],
                high: candle[2],
                open: candle[3],
                close: candle[4],
                volume: candle[5]
            }));
        } catch (error) {
            throw new Error(`Failed to get candles: ${error.message}`);
        }
    }

    async getOrderStatus(orderId) {
        try {
            const order = await this.exchange.privateGetOrdersOrderId({ order_id: orderId });
            return {
                id: order.id,
                price: parseFloat(order.price || 0),
                size: parseFloat(order.size),
                productId: order.product_id,
                side: order.side,
                type: order.type,
                timeInForce: order.time_in_force,
                postOnly: order.post_only,
                createdAt: order.created_at,
                fillFees: parseFloat(order.fill_fees),
                filledSize: parseFloat(order.filled_size),
                executedValue: parseFloat(order.executed_value),
                status: order.status,
                settled: order.settled
            };
        } catch (error) {
            throw new Error(`Failed to get order status: ${error.message}`);
        }
    }

    async cancelAllOrdersForProduct(productId) {
        try {
            const result = await this.exchange.privateDeleteOrders({ product_id: productId });
            this.emit('allOrdersCancelled', productId);
            return result;
        } catch (error) {
            throw new Error(`Failed to cancel all orders for ${productId}: ${error.message}`);
        }
    }

    unsubscribeFromChannel(channel, productIds = []) {
        const unsubscribeMessage = {
            type: 'unsubscribe',
            channels: [channel]
        };

        if (productIds.length > 0) {
            unsubscribeMessage.product_ids = productIds;
        }

        this.websocket.send(JSON.stringify(unsubscribeMessage));
    }

    getStreamData(symbol) {
        return this.streamData.get(symbol);
    }

    disconnect() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        this.subscriptions.clear();
        this.channels.clear();
        super.disconnect();
    }
}

module.exports = Coinbase;