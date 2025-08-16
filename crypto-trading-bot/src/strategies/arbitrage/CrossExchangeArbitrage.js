const BaseStrategy = require('../base/BaseStrategy');
const ccxt = require('ccxt');

class CrossExchangeArbitrage extends BaseStrategy {
    constructor(config, exchangeConnectors) {
        super(config);
        this.exchanges = exchangeConnectors;
        this.balances = new Map();
        this.orderBooks = new Map();
        this.lastPriceUpdate = new Map();
    }

    async initialize() {
        for (const [name, exchange] of Object.entries(this.exchanges)) {
            await exchange.loadMarkets();
            const balance = await exchange.fetchBalance();
            this.balances.set(name, balance);
        }
        await super.initialize();
    }

    async calculateProfit(opportunity) {
        const { symbol, buyExchange, sellExchange, buyPrice, sellPrice, amount } = opportunity;
        
        const grossProfit = (sellPrice - buyPrice) * amount;
        const buyFee = buyPrice * amount * (this.exchanges[buyExchange].fees.trading.maker || 0.001);
        const sellFee = sellPrice * amount * (this.exchanges[sellExchange].fees.trading.taker || 0.001);
        const transferFee = this.getTransferFee(symbol, buyExchange, sellExchange);
        
        return grossProfit - buyFee - sellFee - transferFee;
    }

    async estimateGas(opportunity) {
        return 0;
    }

    async executeStrategy(opportunity) {
        const startTime = Date.now();
        const { symbol, buyExchange, sellExchange, buyPrice, sellPrice, amount } = opportunity;
        
        try {
            const buyOrder = await this.exchanges[buyExchange].createMarketBuyOrder(symbol, amount);
            const sellOrder = await this.exchanges[sellExchange].createMarketSellOrder(symbol, amount);
            
            const buyFilled = buyOrder.filled || 0;
            const sellFilled = sellOrder.filled || 0;
            const actualAmount = Math.min(buyFilled, sellFilled);
            
            const profit = (sellOrder.average - buyOrder.average) * actualAmount;
            const fees = (buyOrder.fee?.cost || 0) + (sellOrder.fee?.cost || 0);
            
            return {
                success: buyOrder.status === 'closed' && sellOrder.status === 'closed',
                buyOrderId: buyOrder.id,
                sellOrderId: sellOrder.id,
                profit: profit - fees,
                gasCost: 0,
                executionTime: Date.now() - startTime,
                actualAmount
            };
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
        const symbols = this.config.tradingPairs;
        
        await this.updateOrderBooks(symbols);
        
        for (const symbol of symbols) {
            const exchangeNames = Object.keys(this.exchanges);
            
            for (let i = 0; i < exchangeNames.length; i++) {
                for (let j = i + 1; j < exchangeNames.length; j++) {
                    const exchangeA = exchangeNames[i];
                    const exchangeB = exchangeNames[j];
                    
                    const orderBookA = this.orderBooks.get(`${exchangeA}:${symbol}`);
                    const orderBookB = this.orderBooks.get(`${exchangeB}:${symbol}`);
                    
                    if (!orderBookA || !orderBookB) continue;
                    
                    const bidA = orderBookA.bids[0]?.[0];
                    const askA = orderBookA.asks[0]?.[0];
                    const bidB = orderBookB.bids[0]?.[0];
                    const askB = orderBookB.asks[0]?.[0];
                    
                    if (!bidA || !askA || !bidB || !askB) continue;
                    
                    if (bidA > askB) {
                        const spread = (bidA - askB) / askB;
                        if (spread > this.minProfitThreshold) {
                            const maxAmount = Math.min(
                                orderBookA.bids[0][1],
                                orderBookB.asks[0][1],
                                this.getMaxTradeAmount(symbol, exchangeA, exchangeB)
                            );
                            
                            opportunities.push({
                                symbol,
                                buyExchange: exchangeB,
                                sellExchange: exchangeA,
                                buyPrice: askB,
                                sellPrice: bidA,
                                amount: maxAmount,
                                spread
                            });
                        }
                    }
                    
                    if (bidB > askA) {
                        const spread = (bidB - askA) / askA;
                        if (spread > this.minProfitThreshold) {
                            const maxAmount = Math.min(
                                orderBookB.bids[0][1],
                                orderBookA.asks[0][1],
                                this.getMaxTradeAmount(symbol, exchangeB, exchangeA)
                            );
                            
                            opportunities.push({
                                symbol,
                                buyExchange: exchangeA,
                                sellExchange: exchangeB,
                                buyPrice: askA,
                                sellPrice: bidB,
                                amount: maxAmount,
                                spread
                            });
                        }
                    }
                }
            }
        }
        
        return opportunities.sort((a, b) => b.spread - a.spread);
    }

    async updateOrderBooks(symbols) {
        const promises = [];
        
        for (const symbol of symbols) {
            for (const [exchangeName, exchange] of Object.entries(this.exchanges)) {
                const key = `${exchangeName}:${symbol}`;
                const lastUpdate = this.lastPriceUpdate.get(key) || 0;
                
                if (Date.now() - lastUpdate > 1000) {
                    promises.push(
                        exchange.fetchOrderBook(symbol, 5).then(orderBook => {
                            this.orderBooks.set(key, orderBook);
                            this.lastPriceUpdate.set(key, Date.now());
                        }).catch(() => {})
                    );
                }
            }
        }
        
        await Promise.allSettled(promises);
    }

    getMaxTradeAmount(symbol, buyExchange, sellExchange) {
        const buyBalance = this.balances.get(buyExchange);
        const sellBalance = this.balances.get(sellExchange);
        
        if (!buyBalance || !sellBalance) return 0;
        
        const [base, quote] = symbol.split('/');
        const buyQuoteBalance = buyBalance.free?.[quote] || 0;
        const sellBaseBalance = sellBalance.free?.[base] || 0;
        
        const currentPrice = this.orderBooks.get(`${buyExchange}:${symbol}`)?.asks[0]?.[0] || 1;
        const maxBuyAmount = buyQuoteBalance / currentPrice * 0.95;
        const maxSellAmount = sellBaseBalance * 0.95;
        
        return Math.min(maxBuyAmount, maxSellAmount);
    }

    getTransferFee(symbol, fromExchange, toExchange) {
        const [base] = symbol.split('/');
        const feeMap = this.config.transferFees || {};
        return feeMap[base] || 0;
    }

    async rebalanceInventory() {
        for (const [exchangeName, exchange] of Object.entries(this.exchanges)) {
            try {
                const balance = await exchange.fetchBalance();
                this.balances.set(exchangeName, balance);
            } catch (error) {
                this.emit('error', `Failed to update balance for ${exchangeName}: ${error.message}`);
            }
        }
    }
}

module.exports = CrossExchangeArbitrage;