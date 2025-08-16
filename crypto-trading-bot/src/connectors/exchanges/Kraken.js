const BaseExchange = require('./BaseExchange');
const WebSocket = require('ws');
const crypto = require('crypto');

class Kraken extends BaseExchange {
    constructor(config) {
        super({
            ...config,
            exchangeId: 'kraken'
        });
        this.wsUrl = 'wss://ws.kraken.com';
        this.wsAuthUrl = 'wss://ws-auth.kraken.com';
        this.publicWs = null;
        this.privateWs = null;
        this.subscriptionId = 0;
        this.channelMap = new Map();
        this.assetPairs = new Map();
    }

    async initialize() {
        await super.initialize();
        
        this.exchange.options = {
            ...this.exchange.options,
            adjustForTimeDifference: false
        };

        await this.loadAssetPairs();
        await this.connectPublicWebSocket();
        await this.connectPrivateWebSocket();
    }

    async loadAssetPairs() {
        try {
            const assetPairs = await this.exchange.publicGetAssetPairs();
            for (const [pair, info] of Object.entries(assetPairs.result)) {
                this.assetPairs.set(pair, {
                    altname: info.altname,
                    wsname: info.wsname,
                    base: info.base,
                    quote: info.quote,
                    pairDecimals: info.pair_decimals,
                    lotDecimals: info.lot_decimals,
                    fees: info.fees
                });
            }
        } catch (error) {
            this.emit('error', `Failed to load asset pairs: ${error.message}`);
        }
    }

    async connectPublicWebSocket() {
        this.publicWs = new WebSocket(this.wsUrl);
        
        this.publicWs.on('open', () => {
            this.emit('publicWebSocketConnected');
        });

        this.publicWs.on('message', (data) => {
            try {
                const message = JSON.parse(data);
                this.handlePublicMessage(message);
            } catch (error) {
                this.emit('error', `Failed to parse public WebSocket message: ${error.message}`);
            }
        });

        this.publicWs.on('error', (error) => {
            this.emit('error', `Public WebSocket error: ${error.message}`);
        });

        this.publicWs.on('close', () => {
            setTimeout(() => this.connectPublicWebSocket(), 5000);
        });
    }

    async connectPrivateWebSocket() {
        try {
            const tokenResponse = await this.exchange.privatePostGetWebSocketsToken();
            const token = tokenResponse.result.token;

            this.privateWs = new WebSocket(this.wsAuthUrl);
            
            this.privateWs.on('open', () => {
                this.authenticatePrivateWebSocket(token);
            });

            this.privateWs.on('message', (data) => {
                try {
                    const message = JSON.parse(data);
                    this.handlePrivateMessage(message);
                } catch (error) {
                    this.emit('error', `Failed to parse private WebSocket message: ${error.message}`);
                }
            });

            this.privateWs.on('error', (error) => {
                this.emit('error', `Private WebSocket error: ${error.message}`);
            });

            this.privateWs.on('close', () => {
                setTimeout(() => this.connectPrivateWebSocket(), 10000);
            });
        } catch (error) {
            this.emit('error', `Failed to connect private WebSocket: ${error.message}`);
        }
    }

    authenticatePrivateWebSocket(token) {
        const authMessage = {
            event: 'subscribe',
            subscription: {
                name: 'ownTrades',
                token: token
            }
        };

        this.privateWs.send(JSON.stringify(authMessage));

        const balanceMessage = {
            event: 'subscribe',
            subscription: {
                name: 'openOrders',
                token: token
            }
        };

        this.privateWs.send(JSON.stringify(balanceMessage));
    }

    handlePublicMessage(message) {
        if (Array.isArray(message)) {
            const channelId = message[0];
            const data = message[1];
            const channelName = message[2];
            const pair = message[3];

            switch (channelName) {
                case 'ticker':
                    this.handleTickerData(data, pair);
                    break;
                case 'book':
                    this.handleOrderBookData(data, pair, channelId);
                    break;
                case 'trade':
                    this.handleTradeData(data, pair);
                    break;
                case 'ohlc':
                    this.handleOHLCData(data, pair);
                    break;
            }
        } else if (message.event === 'subscriptionStatus') {
            this.handleSubscriptionStatus(message);
        } else if (message.event === 'heartbeat') {
            this.emit('heartbeat');
        }
    }

    handlePrivateMessage(message) {
        if (Array.isArray(message)) {
            const data = message[0];
            const channelName = message[1];

            switch (channelName) {
                case 'ownTrades':
                    this.handleOwnTrades(data);
                    break;
                case 'openOrders':
                    this.handleOpenOrders(data);
                    break;
            }
        } else if (message.event === 'subscriptionStatus') {
            this.emit('privateSubscriptionStatus', message);
        }
    }

    handleTickerData(data, pair) {
        const ticker = {
            symbol: pair,
            ask: parseFloat(data.a[0]),
            askVolume: parseFloat(data.a[2]),
            bid: parseFloat(data.b[0]),
            bidVolume: parseFloat(data.b[2]),
            close: parseFloat(data.c[0]),
            volume: parseFloat(data.v[1]),
            vwap: parseFloat(data.p[1]),
            trades: parseInt(data.t[1]),
            low: parseFloat(data.l[1]),
            high: parseFloat(data.h[1]),
            open: parseFloat(data.o)
        };

        this.streamData.set(pair, ticker);
        this.emit('tickerUpdate', pair, ticker);
    }

    handleOrderBookData(data, pair, channelId) {
        let orderBook = this.orderBook.get(pair);
        
        if (!orderBook) {
            orderBook = { bids: [], asks: [], timestamp: Date.now() };
            this.orderBook.set(pair, orderBook);
        }

        if (data.bs) {
            orderBook.bids = data.bs.map(([price, volume]) => [parseFloat(price), parseFloat(volume)]);
        }
        
        if (data.as) {
            orderBook.asks = data.as.map(([price, volume]) => [parseFloat(price), parseFloat(volume)]);
        }

        if (data.b) {
            for (const [price, volume, timestamp] of data.b) {
                const priceLevel = parseFloat(price);
                const volumeLevel = parseFloat(volume);
                
                if (volumeLevel === 0) {
                    orderBook.bids = orderBook.bids.filter(([p]) => p !== priceLevel);
                } else {
                    const index = orderBook.bids.findIndex(([p]) => p === priceLevel);
                    if (index >= 0) {
                        orderBook.bids[index] = [priceLevel, volumeLevel];
                    } else {
                        orderBook.bids.push([priceLevel, volumeLevel]);
                        orderBook.bids.sort((a, b) => b[0] - a[0]);
                    }
                }
            }
        }

        if (data.a) {
            for (const [price, volume, timestamp] of data.a) {
                const priceLevel = parseFloat(price);
                const volumeLevel = parseFloat(volume);
                
                if (volumeLevel === 0) {
                    orderBook.asks = orderBook.asks.filter(([p]) => p !== priceLevel);
                } else {
                    const index = orderBook.asks.findIndex(([p]) => p === priceLevel);
                    if (index >= 0) {
                        orderBook.asks[index] = [priceLevel, volumeLevel];
                    } else {
                        orderBook.asks.push([priceLevel, volumeLevel]);
                        orderBook.asks.sort((a, b) => a[0] - b[0]);
                    }
                }
            }
        }

        orderBook.timestamp = Date.now();
        this.emit('orderBookUpdate', pair, orderBook);
    }

    handleOwnTrades(trades) {
        for (const [tradeId, trade] of Object.entries(trades)) {
            this.emit('tradeUpdate', {
                tradeId,
                orderId: trade.ordertxid,
                pair: trade.pair,
                time: parseFloat(trade.time),
                type: trade.type,
                orderType: trade.ordertype,
                price: parseFloat(trade.price),
                cost: parseFloat(trade.cost),
                fee: parseFloat(trade.fee),
                vol: parseFloat(trade.vol),
                margin: parseFloat(trade.margin || 0),
                misc: trade.misc
            });
        }
    }

    handleOpenOrders(orders) {
        for (const [orderId, order] of Object.entries(orders)) {
            this.emit('orderUpdate', {
                orderId,
                refId: order.refid,
                userRef: order.userref,
                status: order.status,
                openTime: parseFloat(order.opentm),
                startTime: parseFloat(order.starttm),
                expireTime: parseFloat(order.expiretm),
                description: order.descr,
                vol: parseFloat(order.vol),
                volExec: parseFloat(order.vol_exec),
                cost: parseFloat(order.cost),
                fee: parseFloat(order.fee),
                price: parseFloat(order.price),
                stopPrice: parseFloat(order.stopprice || 0),
                limitPrice: parseFloat(order.limitprice || 0),
                misc: order.misc,
                oFlags: order.oflags
            });
        }
    }

    handleSubscriptionStatus(message) {
        if (message.status === 'subscribed') {
            this.channelMap.set(message.channelID, {
                channel: message.subscription.name,
                pair: message.pair
            });
            this.emit('subscribed', message);
        } else if (message.status === 'error') {
            this.emit('subscriptionError', message);
        }
    }

    async subscribeToTicker(pair) {
        const subscribeMessage = {
            event: 'subscribe',
            pair: [pair],
            subscription: {
                name: 'ticker'
            }
        };

        this.publicWs.send(JSON.stringify(subscribeMessage));
    }

    async subscribeToOrderBook(pair, depth = 10) {
        const subscribeMessage = {
            event: 'subscribe',
            pair: [pair],
            subscription: {
                name: 'book',
                depth: depth
            }
        };

        this.publicWs.send(JSON.stringify(subscribeMessage));
    }

    async subscribeToTrades(pair) {
        const subscribeMessage = {
            event: 'subscribe',
            pair: [pair],
            subscription: {
                name: 'trade'
            }
        };

        this.publicWs.send(JSON.stringify(subscribeMessage));
    }

    async addOrder(pair, type, orderType, volume, price = null, params = {}) {
        try {
            const orderParams = {
                pair,
                type,
                ordertype: orderType,
                volume: volume.toString()
            };

            if (price) {
                orderParams.price = price.toString();
            }

            if (params.leverage) {
                orderParams.leverage = params.leverage.toString();
            }

            if (params.stopLoss) {
                orderParams['close[ordertype]'] = 'stop-loss';
                orderParams['close[price]'] = params.stopLoss.toString();
            }

            if (params.takeProfit) {
                orderParams['close[ordertype]'] = 'take-profit';
                orderParams['close[price]'] = params.takeProfit.toString();
            }

            if (params.startTime) {
                orderParams.starttm = params.startTime;
            }

            if (params.expireTime) {
                orderParams.expiretm = params.expireTime;
            }

            if (params.postOnly) {
                orderParams.oflags = 'post';
            }

            const result = await this.exchange.privatePostAddOrder(orderParams);
            return result.result;
        } catch (error) {
            throw new Error(`Failed to add order: ${error.message}`);
        }
    }

    async cancelOrder(txid) {
        try {
            const result = await this.exchange.privatePostCancelOrder({ txid });
            return result.result;
        } catch (error) {
            throw new Error(`Failed to cancel order: ${error.message}`);
        }
    }

    async getOrderInfo(txid, trades = false) {
        try {
            const result = await this.exchange.privatePostQueryOrders({
                txid,
                trades
            });
            return result.result;
        } catch (error) {
            throw new Error(`Failed to get order info: ${error.message}`);
        }
    }

    async getTradesHistory(type = 'all', trades = false, start = null, end = null, ofs = null) {
        try {
            const params = { type, trades };
            
            if (start) params.start = start;
            if (end) params.end = end;
            if (ofs) params.ofs = ofs;

            const result = await this.exchange.privatePostTradesHistory(params);
            return result.result;
        } catch (error) {
            throw new Error(`Failed to get trades history: ${error.message}`);
        }
    }

    async getDepositMethods(asset) {
        try {
            const result = await this.exchange.privatePostDepositMethods({ asset });
            return result.result;
        } catch (error) {
            throw new Error(`Failed to get deposit methods: ${error.message}`);
        }
    }

    async getDepositAddresses(asset, method, new_address = false) {
        try {
            const result = await this.exchange.privatePostDepositAddresses({
                asset,
                method,
                new: new_address
            });
            return result.result;
        } catch (error) {
            throw new Error(`Failed to get deposit addresses: ${error.message}`);
        }
    }

    async getWithdrawInfo(asset, key, amount) {
        try {
            const result = await this.exchange.privatePostWithdrawInfo({
                asset,
                key,
                amount: amount.toString()
            });
            return result.result;
        } catch (error) {
            throw new Error(`Failed to get withdraw info: ${error.message}`);
        }
    }

    async withdrawFunds(asset, key, amount) {
        try {
            const result = await this.exchange.privatePostWithdraw({
                asset,
                key,
                amount: amount.toString()
            });
            return result.result;
        } catch (error) {
            throw new Error(`Failed to withdraw funds: ${error.message}`);
        }
    }

    async getWithdrawStatus(asset, method) {
        try {
            const result = await this.exchange.privatePostWithdrawStatus({
                asset,
                method
            });
            return result.result;
        } catch (error) {
            throw new Error(`Failed to get withdraw status: ${error.message}`);
        }
    }

    async getSystemStatus() {
        try {
            const result = await this.exchange.publicGetSystemStatus();
            return result.result;
        } catch (error) {
            throw new Error(`Failed to get system status: ${error.message}`);
        }
    }

    async getAssetInfo(info = 'info', aclass = 'currency', assets = []) {
        try {
            const params = { info, aclass };
            
            if (assets.length > 0) {
                params.asset = assets.join(',');
            }

            const result = await this.exchange.publicGetAssets(params);
            return result.result;
        } catch (error) {
            throw new Error(`Failed to get asset info: ${error.message}`);
        }
    }

    getAssetPairInfo(pair) {
        return this.assetPairs.get(pair);
    }

    formatPairName(symbol) {
        for (const [pair, info] of this.assetPairs.entries()) {
            if (info.altname === symbol || info.wsname === symbol) {
                return pair;
            }
        }
        return symbol;
    }

    getStreamData(pair) {
        return this.streamData.get(pair);
    }

    disconnect() {
        if (this.publicWs) {
            this.publicWs.close();
            this.publicWs = null;
        }
        
        if (this.privateWs) {
            this.privateWs.close();
            this.privateWs = null;
        }
        
        this.channelMap.clear();
        super.disconnect();
    }
}

module.exports = Kraken;