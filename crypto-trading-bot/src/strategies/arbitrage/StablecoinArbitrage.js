const BaseStrategy = require('../base/BaseStrategy');

class StablecoinArbitrage extends BaseStrategy {
    constructor(config, dexConnectors, cexConnectors) {
        super(config);
        this.dexConnectors = dexConnectors;
        this.cexConnectors = cexConnectors;
        this.stablecoins = config.stablecoins;
        this.depegThreshold = config.depegThreshold || 0.003;
        this.priceFeeds = new Map();
        this.liquidityCache = new Map();
    }

    async initialize() {
        for (const stablecoin of this.stablecoins) {
            await this.initializeStablecoinMonitoring(stablecoin);
        }
        await super.initialize();
    }

    async calculateProfit(opportunity) {
        const { stablecoin, buyVenue, sellVenue, buyPrice, sellPrice, amount } = opportunity;
        
        const grossProfit = (sellPrice - buyPrice) * amount;
        const buyFee = this.getTradingFee(buyVenue, stablecoin) * buyPrice * amount;
        const sellFee = this.getTradingFee(sellVenue, stablecoin) * sellPrice * amount;
        const transferFee = this.getTransferFee(stablecoin, buyVenue, sellVenue, amount);
        
        return grossProfit - buyFee - sellFee - transferFee;
    }

    async estimateGas(opportunity) {
        if (this.isDexVenue(opportunity.buyVenue) || this.isDexVenue(opportunity.sellVenue)) {
            return 150000;
        }
        return 0;
    }

    async executeStrategy(opportunity) {
        const startTime = Date.now();
        const { stablecoin, buyVenue, sellVenue, buyPrice, sellPrice, amount } = opportunity;
        
        try {
            const buyOrder = await this.executeBuy(stablecoin, amount, buyVenue, buyPrice);
            const sellOrder = await this.executeSell(stablecoin, amount, sellVenue, sellPrice);
            
            const actualBuyAmount = buyOrder.filled || 0;
            const actualSellAmount = sellOrder.filled || 0;
            const executedAmount = Math.min(actualBuyAmount, actualSellAmount);
            
            const profit = (sellOrder.average - buyOrder.average) * executedAmount;
            const fees = (buyOrder.fee?.cost || 0) + (sellOrder.fee?.cost || 0);
            const gasCost = buyOrder.gasCost + sellOrder.gasCost;
            
            return {
                success: buyOrder.success && sellOrder.success,
                buyOrderId: buyOrder.id,
                sellOrderId: sellOrder.id,
                profit: profit - fees,
                gasCost,
                executionTime: Date.now() - startTime,
                executedAmount
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
        await this.updatePriceFeeds();
        const opportunities = [];
        
        for (const stablecoin of this.stablecoins) {
            const prices = this.priceFeeds.get(stablecoin.symbol) || {};
            const venues = Object.keys(prices);
            
            for (let i = 0; i < venues.length; i++) {
                for (let j = i + 1; j < venues.length; j++) {
                    const venueA = venues[i];
                    const venueB = venues[j];
                    const priceA = prices[venueA];
                    const priceB = prices[venueB];
                    
                    if (!priceA || !priceB) continue;
                    
                    const spread = Math.abs(priceA - priceB);
                    const relativeSpread = spread / Math.min(priceA, priceB);
                    
                    if (relativeSpread > this.minProfitThreshold) {
                        const buyVenue = priceA < priceB ? venueA : venueB;
                        const sellVenue = priceA < priceB ? venueB : venueA;
                        const buyPrice = Math.min(priceA, priceB);
                        const sellPrice = Math.max(priceA, priceB);
                        
                        const maxAmount = await this.getMaxTradeAmount(
                            stablecoin.symbol, 
                            buyVenue, 
                            sellVenue
                        );
                        
                        if (maxAmount > stablecoin.minTradeAmount) {
                            const isDepegEvent = this.isDepegEvent(stablecoin.symbol, buyPrice, sellPrice);
                            
                            opportunities.push({
                                stablecoin: stablecoin.symbol,
                                buyVenue,
                                sellVenue,
                                buyPrice,
                                sellPrice,
                                amount: Math.min(maxAmount, stablecoin.maxTradeAmount),
                                spread: relativeSpread,
                                isDepegEvent,
                                urgency: isDepegEvent ? 'HIGH' : 'NORMAL'
                            });
                        }
                    }
                }
            }
        }
        
        return opportunities.sort((a, b) => {
            if (a.urgency === 'HIGH' && b.urgency !== 'HIGH') return -1;
            if (b.urgency === 'HIGH' && a.urgency !== 'HIGH') return 1;
            return b.spread - a.spread;
        });
    }

    async updatePriceFeeds() {
        const promises = [];
        
        for (const stablecoin of this.stablecoins) {
            const venues = [...Object.keys(this.dexConnectors), ...Object.keys(this.cexConnectors)];
            
            for (const venue of venues) {
                promises.push(
                    this.updatePriceForVenue(stablecoin.symbol, venue)
                );
            }
        }
        
        await Promise.allSettled(promises);
    }

    async updatePriceForVenue(symbol, venue) {
        try {
            let price;
            
            if (this.isDexVenue(venue)) {
                price = await this.dexConnectors[venue].getPrice(symbol);
            } else {
                price = await this.cexConnectors[venue].fetchTicker(symbol + '/USD');
                price = price.last;
            }
            
            if (!this.priceFeeds.has(symbol)) {
                this.priceFeeds.set(symbol, {});
            }
            
            this.priceFeeds.get(symbol)[venue] = price;
        } catch (error) {
            if (this.priceFeeds.has(symbol)) {
                delete this.priceFeeds.get(symbol)[venue];
            }
        }
    }

    async executeBuy(stablecoin, amount, venue, maxPrice) {
        try {
            if (this.isDexVenue(venue)) {
                const tx = await this.dexConnectors[venue].buyToken(stablecoin, amount, maxPrice);
                const receipt = await this.dexConnectors[venue].waitForTransaction(tx);
                
                return {
                    success: receipt.status === 1,
                    id: tx,
                    filled: amount,
                    average: maxPrice,
                    fee: { cost: receipt.gasUsed * receipt.effectiveGasPrice },
                    gasCost: receipt.gasUsed * receipt.effectiveGasPrice
                };
            } else {
                const order = await this.cexConnectors[venue].createMarketBuyOrder(
                    stablecoin + '/USD', 
                    amount
                );
                
                return {
                    success: order.status === 'closed',
                    id: order.id,
                    filled: order.filled,
                    average: order.average,
                    fee: order.fee,
                    gasCost: 0
                };
            }
        } catch (error) {
            return {
                success: false,
                error: error.message,
                gasCost: 0
            };
        }
    }

    async executeSell(stablecoin, amount, venue, minPrice) {
        try {
            if (this.isDexVenue(venue)) {
                const tx = await this.dexConnectors[venue].sellToken(stablecoin, amount, minPrice);
                const receipt = await this.dexConnectors[venue].waitForTransaction(tx);
                
                return {
                    success: receipt.status === 1,
                    id: tx,
                    filled: amount,
                    average: minPrice,
                    fee: { cost: receipt.gasUsed * receipt.effectiveGasPrice },
                    gasCost: receipt.gasUsed * receipt.effectiveGasPrice
                };
            } else {
                const order = await this.cexConnectors[venue].createMarketSellOrder(
                    stablecoin + '/USD', 
                    amount
                );
                
                return {
                    success: order.status === 'closed',
                    id: order.id,
                    filled: order.filled,
                    average: order.average,
                    fee: order.fee,
                    gasCost: 0
                };
            }
        } catch (error) {
            return {
                success: false,
                error: error.message,
                gasCost: 0
            };
        }
    }

    isDepegEvent(symbol, buyPrice, sellPrice) {
        const avgPrice = (buyPrice + sellPrice) / 2;
        const deviation = Math.abs(avgPrice - 1.0);
        return deviation > this.depegThreshold;
    }

    async getMaxTradeAmount(symbol, buyVenue, sellVenue) {
        const buyLiquidity = await this.getLiquidity(symbol, buyVenue);
        const sellLiquidity = await this.getLiquidity(symbol, sellVenue);
        
        return Math.min(buyLiquidity, sellLiquidity) * 0.1;
    }

    async getLiquidity(symbol, venue) {
        const cacheKey = `${symbol}:${venue}`;
        const cached = this.liquidityCache.get(cacheKey);
        
        if (cached && Date.now() - cached.timestamp < 30000) {
            return cached.liquidity;
        }
        
        try {
            let liquidity;
            
            if (this.isDexVenue(venue)) {
                liquidity = await this.dexConnectors[venue].getLiquidity(symbol);
            } else {
                const orderBook = await this.cexConnectors[venue].fetchOrderBook(symbol + '/USD', 10);
                const bidLiquidity = orderBook.bids.reduce((sum, [price, amount]) => sum + amount, 0);
                const askLiquidity = orderBook.asks.reduce((sum, [price, amount]) => sum + amount, 0);
                liquidity = Math.min(bidLiquidity, askLiquidity);
            }
            
            this.liquidityCache.set(cacheKey, { liquidity, timestamp: Date.now() });
            return liquidity;
        } catch {
            return 1000;
        }
    }

    getTradingFee(venue, stablecoin) {
        if (this.isDexVenue(venue)) {
            return this.dexConnectors[venue].getFee(stablecoin) || 0.003;
        } else {
            return this.cexConnectors[venue].fees?.trading?.maker || 0.001;
        }
    }

    getTransferFee(stablecoin, fromVenue, toVenue, amount) {
        if (this.isDexVenue(fromVenue) && this.isDexVenue(toVenue)) {
            return 0;
        }
        
        const feeMap = this.config.transferFees || {};
        const flatFee = feeMap[stablecoin] || 1;
        const percentageFee = amount * 0.0001;
        
        return flatFee + percentageFee;
    }

    isDexVenue(venue) {
        return this.dexConnectors.hasOwnProperty(venue);
    }

    async initializeStablecoinMonitoring(stablecoin) {
        setInterval(async () => {
            await this.checkForDepegEvents(stablecoin);
        }, 5000);
    }

    async checkForDepegEvents(stablecoin) {
        const prices = this.priceFeeds.get(stablecoin.symbol) || {};
        const priceValues = Object.values(prices).filter(p => p);
        
        if (priceValues.length === 0) return;
        
        const avgPrice = priceValues.reduce((sum, price) => sum + price, 0) / priceValues.length;
        const maxDeviation = Math.max(...priceValues.map(p => Math.abs(p - 1.0)));
        
        if (maxDeviation > this.depegThreshold) {
            this.emit('depegDetected', {
                stablecoin: stablecoin.symbol,
                avgPrice,
                maxDeviation,
                prices,
                timestamp: Date.now()
            });
        }
    }

    async emergencyArbitrage(depegEvent) {
        const { stablecoin, prices } = depegEvent;
        const opportunities = [];
        
        const sortedPrices = Object.entries(prices).sort(([,a], [,b]) => a - b);
        const lowestPrice = sortedPrices[0];
        const highestPrice = sortedPrices[sortedPrices.length - 1];
        
        if (lowestPrice[1] < 0.98 && highestPrice[1] > 1.01) {
            const maxAmount = await this.getMaxTradeAmount(stablecoin, lowestPrice[0], highestPrice[0]);
            
            opportunities.push({
                stablecoin,
                buyVenue: lowestPrice[0],
                sellVenue: highestPrice[0],
                buyPrice: lowestPrice[1],
                sellPrice: highestPrice[1],
                amount: maxAmount,
                spread: (highestPrice[1] - lowestPrice[1]) / lowestPrice[1],
                isDepegEvent: true,
                urgency: 'EMERGENCY'
            });
        }
        
        return opportunities;
    }
}

module.exports = StablecoinArbitrage;