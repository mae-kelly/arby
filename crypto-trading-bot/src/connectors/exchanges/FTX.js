const BaseExchange = require('./BaseExchange');
const WebSocket = require('ws');
const crypto = require('crypto');

class FTX extends BaseExchange {
    constructor(config) {
        super({
            ...config,
            exchangeId: 'ftx'
        });
        this.wsUrl = 'wss://ftx.com/ws/';
        this.websocket = null;
        this.subscriptions = new Set();
        this.channels = new Map();
        this.authenticated = false;
        this.requestId = 1;
        this.subaccount = config.subaccount || null;
    }

    async initialize() {
        await super.initialize();
        
        this.exchange.options = {
            ...this.exchange.options,
            'FTX-SUBACCOUNT': this.subaccount
        };

        await this.connectWebSocket();
    }

    async connectWebSocket() {
        this.websocket = new WebSocket(this.wsUrl);
        
        this.websocket.on('open', () => {
            this.emit('websocketConnected');
            this.authenticate();
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
            this.authenticated = false;
            setTimeout(() => this.connectWebSocket(), 5000);
        });

        setInterval(() => {
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({ op: 'ping' }));
            }
        }, 15000);
    }

    authenticate() {
        const ts = Date.now();
        const signature = crypto
            .createHmac('sha256', this.config.secret)
            .update(ts + 'websocket_login')
            .digest('hex');

        const authMessage = {
            op: 'login',
            args: {
                key: this.config.apiKey,
                sign: signature,
                time: ts
            }
        };

        if (this.subaccount) {
            authMessage.args.subaccount = this.subaccount;
        }

        this.websocket.send(JSON.stringify(authMessage));
    }

    handleWebSocketMessage(message) {
        if (message.type === 'error') {
            this.emit('error', `WebSocket error: ${message.msg}`);
            return;
        }

        if (message.type === 'subscribed') {
            this.emit('subscribed', message);
            return;
        }

        if (message.type === 'info' && message.code === 20001) {
            this.authenticated = true;
            this.emit('authenticated');
            return;
        }

        if (message.type === 'pong') {
            this.emit('pong');
            return;
        }

        switch (message.channel) {
            case 'ticker':
                this.handleTickerMessage(message);
                break;
            case 'orderbook':
                this.handleOrderBookMessage(message);
                break;
            case 'trades':
                this.handleTradesMessage(message);
                break;
            case 'orders':
                this.handleOrdersMessage(message);
                break;
            case 'fills':
                this.handleFillsMessage(message);
                break;
        }
    }

    handleTickerMessage(message) {
        const ticker = message.data;
        this.streamData.set(message.market, {
            symbol: message.market,
            bid: ticker.bid,
            ask: ticker.ask,
            bidSize: ticker.bidSize,
            askSize: ticker.askSize,
            last: ticker.last,
            time: ticker.time
        });
        this.emit('tickerUpdate', message.market, this.streamData.get(message.market));
    }

    handleOrderBookMessage(message) {
        const data = message.data;
        let orderBook = this.orderBook.get(message.market);

        if (message.type === 'partial') {
            orderBook = {
                bids: data.bids.map(([price, size]) => [price, size]),
                asks: data.asks.map(([price, size]) => [price, size]),
                timestamp: data.time,
                checksum: data.checksum
            };
            this.orderBook.set(message.market, orderBook);
        } else if (message.type === 'update') {
            if (orderBook) {
                for (const [price, size] of data.bids) {
                    if (size === 0) {
                        orderBook.bids = orderBook.bids.filter(([p]) => p !== price);
                    } else {
                        const index = orderBook.bids.findIndex(([p]) => p === price);
                        if (index >= 0) {
                            orderBook.bids[index] = [price, size];
                        } else {
                            orderBook.bids.push([price, size]);
                            orderBook.bids.sort((a, b) => b[0] - a[0]);
                        }
                    }
                }

                for (const [price, size] of data.asks) {
                    if (size === 0) {
                        orderBook.asks = orderBook.asks.filter(([p]) => p !== price);
                    } else {
                        const index = orderBook.asks.findIndex(([p]) => p === price);
                        if (index >= 0) {
                            orderBook.asks[index] = [price, size];
                        } else {
                            orderBook.asks.push([price, size]);
                            orderBook.asks.sort((a, b) => a[0] - b[0]);
                        }
                    }
                }

                orderBook.timestamp = data.time;
                orderBook.checksum = data.checksum;
            }
        }

        this.emit('orderBookUpdate', message.market, orderBook);
    }

    handleTradesMessage(message) {
        for (const trade of message.data) {
            this.emit('tradeUpdate', {
                market: message.market,
                id: trade.id,
                price: trade.price,
                size: trade.size,
                side: trade.side,
                time: trade.time,
                liquidation: trade.liquidation
            });
        }
    }

    handleOrdersMessage(message) {
        const order = message.data;
        this.emit('orderUpdate', {
            id: order.id,
            clientId: order.clientId,
            market: order.market,
            type: order.type,
            side: order.side,
            size: order.size,
            price: order.price,
            status: order.status,
            filledSize: order.filledSize,
            remainingSize: order.remainingSize,
            avgFillPrice: order.avgFillPrice,
            createdAt: order.createdAt
        });
    }

    handleFillsMessage(message) {
        const fill = message.data;
        this.emit('fillUpdate', {
            id: fill.id,
            market: fill.market,
            future: fill.future,
            side: fill.side,
            size: fill.size,
            price: fill.price,
            type: fill.type,
            time: fill.time,
            orderId: fill.orderId,
            tradeId: fill.tradeId,
            fee: fill.fee,
            feeCurrency: fill.feeCurrency,
            feeRate: fill.feeRate,
            liquidity: fill.liquidity
        });
    }

    async subscribe(channel, market = null) {
        const subscribeMessage = {
            op: 'subscribe',
            channel: channel
        };

        if (market) {
            subscribeMessage.market = market;
        }

        this.websocket.send(JSON.stringify(subscribeMessage));
        this.subscriptions.add(`${channel}:${market}`);
    }

    async unsubscribe(channel, market = null) {
        const unsubscribeMessage = {
            op: 'unsubscribe',
            channel: channel
        };

        if (market) {
            unsubscribeMessage.market = market;
        }

        this.websocket.send(JSON.stringify(unsubscribeMessage));
        this.subscriptions.delete(`${channel}:${market}`);
    }

    async subscribeToTicker(market) {
        await this.subscribe('ticker', market);
    }

    async subscribeToOrderBook(market) {
        await this.subscribe('orderbook', market);
    }

    async subscribeToTrades(market) {
        await this.subscribe('trades', market);
    }

    async subscribeToOrders() {
        if (this.authenticated) {
            await this.subscribe('orders');
        }
    }

    async subscribeToFills() {
        if (this.authenticated) {
            await this.subscribe('fills');
        }
    }

    async placeOrder(market, side, price, size, type = 'limit', params = {}) {
        try {
            const orderParams = {
                market,
                side,
                price: type === 'market' ? null : price,
                type,
                size,
                ...params
            };

            if (params.reduceOnly) {
                orderParams.reduceOnly = true;
            }

            if (params.ioc) {
                orderParams.ioc = true;
            }

            if (params.postOnly) {
                orderParams.postOnly = true;
            }

            if (params.clientId) {
                orderParams.clientId = params.clientId;
            }

            const order = await this.exchange.privatePostOrders(orderParams);
            this.emit('orderPlaced', order.result);
            return order.result;
        } catch (error) {
            throw new Error(`Failed to place order: ${error.message}`);
        }
    }

    async modifyOrder(orderId, price = null, size = null, clientId = null) {
        try {
            const params = { order_id: orderId };
            
            if (price !== null) params.price = price;
            if (size !== null) params.size = size;
            if (clientId !== null) params.clientId = clientId;

            const order = await this.exchange.privatePostOrdersOrderIdModify(params);
            return order.result;
        } catch (error) {
            throw new Error(`Failed to modify order: ${error.message}`);
        }
    }

    async getOrderHistory(market = null, startTime = null, endTime = null, limit = 100) {
        try {
            const params = { limit };
            
            if (market) params.market = market;
            if (startTime) params.start_time = startTime;
            if (endTime) params.end_time = endTime;

            const orders = await this.exchange.privateGetOrdersHistory(params);
            return orders.result;
        } catch (error) {
            throw new Error(`Failed to get order history: ${error.message}`);
        }
    }

    async getFills(market = null, startTime = null, endTime = null, limit = 100) {
        try {
            const params = { limit };
            
            if (market) params.market = market;
            if (startTime) params.start_time = startTime;
            if (endTime) params.end_time = endTime;

            const fills = await this.exchange.privateGetFills(params);
            return fills.result;
        } catch (error) {
            throw new Error(`Failed to get fills: ${error.message}`);
        }
    }

    async getPositions(showAvgPrice = false) {
        try {
            const positions = await this.exchange.privateGetPositions({
                showAvgPrice
            });
            return positions.result;
        } catch (error) {
            throw new Error(`Failed to get positions: ${error.message}`);
        }
    }

    async getPosition(market) {
        try {
            const position = await this.exchange.privateGetPositionsMarket({
                market
            });
            return position.result;
        } catch (error) {
            throw new Error(`Failed to get position for ${market}: ${error.message}`);
        }
    }

    async getLeverageTokens() {
        try {
            const tokens = await this.exchange.publicGetLt();
            return tokens.result;
        } catch (error) {
            throw new Error(`Failed to get leverage tokens: ${error.message}`);
        }
    }

    async getLeverageTokenBalance(token) {
        try {
            const balance = await this.exchange.privateGetLtTokenBalances({
                token
            });
            return balance.result;
        } catch (error) {
            throw new Error(`Failed to get leverage token balance: ${error.message}`);
        }
    }

    async redeemLeverageToken(token, size) {
        try {
            const result = await this.exchange.privatePostLtTokenRedeem({
                token,
                size
            });
            return result.result;
        } catch (error) {
            throw new Error(`Failed to redeem leverage token: ${error.message}`);
        }
    }

    async getSubaccounts() {
        try {
            const subaccounts = await this.exchange.privateGetSubaccounts();
            return subaccounts.result;
        } catch (error) {
            throw new Error(`Failed to get subaccounts: ${error.message}`);
        }
    }

    async createSubaccount(nickname) {
        try {
            const subaccount = await this.exchange.privatePostSubaccounts({
                nickname
            });
            return subaccount.result;
        } catch (error) {
            throw new Error(`Failed to create subaccount: ${error.message}`);
        }
    }

    async transferBetweenSubaccounts(coin, size, source, destination) {
        try {
            const transfer = await this.exchange.privatePostSubaccountsTransfer({
                coin,
                size,
                source,
                destination
            });
            return transfer.result;
        } catch (error) {
            throw new Error(`Failed to transfer between subaccounts: ${error.message}`);
        }
    }

    async getMarkets() {
        try {
            const markets = await this.exchange.publicGetMarkets();
            return markets.result;
        } catch (error) {
            throw new Error(`Failed to get markets: ${error.message}`);
        }
    }

    async getMarket(market) {
        try {
            const marketData = await this.exchange.publicGetMarketsMarket({
                market
            });
            return marketData.result;
        } catch (error) {
            throw new Error(`Failed to get market data: ${error.message}`);
        }
    }

    async getHistoricalPrices(market, resolution, startTime, endTime) {
        try {
            const candles = await this.exchange.publicGetMarketsMarketCandles({
                market_name: market,
                resolution,
                start_time: startTime,
                end_time: endTime
            });
            return candles.result;
        } catch (error) {
            throw new Error(`Failed to get historical prices: ${error.message}`);
        }
    }

    async getFutures() {
        try {
            const futures = await this.exchange.publicGetFutures();
            return futures.result;
        } catch (error) {
            throw new Error(`Failed to get futures: ${error.message}`);
        }
    }

    async getFuture(future) {
        try {
            const futureData = await this.exchange.publicGetFuturesFuture({
                future
            });
            return futureData.result;
        } catch (error) {
            throw new Error(`Failed to get future data: ${error.message}`);
        }
    }

    async getFundingRates(future = null, startTime = null, endTime = null) {
        try {
            const params = {};
            
            if (future) params.future = future;
            if (startTime) params.start_time = startTime;
            if (endTime) params.end_time = endTime;

            const rates = await this.exchange.publicGetFundingRates(params);
            return rates.result;
        } catch (error) {
            throw new Error(`Failed to get funding rates: ${error.message}`);
        }
    }

    async getIndexWeights(index) {
        try {
            const weights = await this.exchange.publicGetIndexesIndexWeights({
                index
            });
            return weights.result;
        } catch (error) {
            throw new Error(`Failed to get index weights: ${error.message}`);
        }
    }

    async getAccountInfo() {
        try {
            const account = await this.exchange.privateGetAccount();
            return account.result;
        } catch (error) {
            throw new Error(`Failed to get account info: ${error.message}`);
        }
    }

    async changeAccountLeverage(leverage) {
        try {
            const result = await this.exchange.privatePostAccountLeverage({
                leverage
            });
            return result.result;
        } catch (error) {
            throw new Error(`Failed to change account leverage: ${error.message}`);
        }
    }

    async getDepositAddress(coin, method = null) {
        try {
            const params = { coin };
            if (method) params.method = method;

            const address = await this.exchange.privateGetWalletDepositAddressCoin(params);
            return address.result;
        } catch (error) {
            throw new Error(`Failed to get deposit address: ${error.message}`);
        }
    }

    async requestWithdrawal(coin, size, address, tag = null, method = null, password = null) {
        try {
            const params = {
                coin,
                size,
                address
            };

            if (tag) params.tag = tag;
            if (method) params.method = method;
            if (password) params.password = password;

            const withdrawal = await this.exchange.privatePostWalletWithdrawals(params);
            return withdrawal.result;
        } catch (error) {
            throw new Error(`Failed to request withdrawal: ${error.message}`);
        }
    }

    async getWithdrawalHistory(startTime = null, endTime = null) {
        try {
            const params = {};
            
            if (startTime) params.start_time = startTime;
            if (endTime) params.end_time = endTime;

            const withdrawals = await this.exchange.privateGetWalletWithdrawals(params);
            return withdrawals.result;
        } catch (error) {
            throw new Error(`Failed to get withdrawal history: ${error.message}`);
        }
    }

    async getDepositHistory(startTime = null, endTime = null) {
        try {
            const params = {};
            
            if (startTime) params.start_time = startTime;
            if (endTime) params.end_time = endTime;

            const deposits = await this.exchange.privateGetWalletDeposits(params);
            return deposits.result;
        } catch (error) {
            throw new Error(`Failed to get deposit history: ${error.message}`);
        }
    }

    getStreamData(market) {
        return this.streamData.get(market);
    }

    isAuthenticated() {
        return this.authenticated;
    }

    disconnect() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        this.authenticated = false;
        this.subscriptions.clear();
        this.channels.clear();
        super.disconnect();
    }
}

module.exports = FTX;