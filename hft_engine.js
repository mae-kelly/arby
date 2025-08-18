const WebSocket = require('ws');
const axios = require('axios');

class RealHFTEngine {
    constructor() {
        this.webhook = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3";
        this.orderBook = {};
        this.lastPrices = {};
        this.running = false;
        this.reconnectDelay = 5000;
    }

    async sendAlert(message) {
        try {
            await axios.post(this.webhook, {
                content: `⚡ REAL HFT: ${message}`
            }, { timeout: 5000 });
            console.log(`Alert: ${message}`);
        } catch (error) {
            console.error("Discord error:", error.message);
        }
    }

    analyzeRealOrderBook(symbol) {
        const orderBook = this.orderBook[symbol];
        if (!orderBook || !orderBook.bids || !orderBook.asks) return null;

        const bidPrice = parseFloat(orderBook.bids[0][0]);
        const askPrice = parseFloat(orderBook.asks[0][0]);
        const bidSize = parseFloat(orderBook.bids[0][1]);
        const askSize = parseFloat(orderBook.asks[0][1]);
        
        const spread = askPrice - bidPrice;
        const midPrice = (askPrice + bidPrice) / 2;
        const spreadBps = (spread / midPrice) * 10000; // Basis points
        
        // Calculate order book imbalance
        const bidDepth = orderBook.bids.slice(0, 5).reduce((sum, [price, size]) => sum + parseFloat(size), 0);
        const askDepth = orderBook.asks.slice(0, 5).reduce((sum, [price, size]) => sum + parseFloat(size), 0);
        const imbalance = (bidDepth - askDepth) / (bidDepth + askDepth);

        return {
            symbol,
            bidPrice,
            askPrice,
            spread,
            spreadBps,
            midPrice,
            imbalance,
            bidDepth,
            askDepth,
            bidSize,
            askSize
        };
    }

    async detectRealOpportunities(analysis) {
        const { symbol, spreadBps, imbalance, midPrice } = analysis;
        
        // Real HFT opportunity detection
        const opportunities = [];
        
        // Tight spread opportunity (good for market making)
        if (spreadBps < 5) { // Less than 0.5 basis points
            opportunities.push({
                type: "TIGHT_SPREAD",
                symbol,
                description: `Very tight spread: ${spreadBps.toFixed(2)} bps`,
                profitPotential: "Low risk market making"
            });
        }
        
        // Order book imbalance (directional opportunity)
        if (Math.abs(imbalance) > 0.3) {
            const direction = imbalance > 0 ? "BULLISH" : "BEARISH";
            opportunities.push({
                type: "IMBALANCE",
                symbol,
                direction,
                description: `Strong ${direction.toLowerCase()} imbalance: ${(imbalance * 100).toFixed(1)}%`,
                profitPotential: "Directional momentum"
            });
        }
        
        // Wide spread opportunity (arbitrage potential)
        if (spreadBps > 20) { // More than 2 basis points
            opportunities.push({
                type: "WIDE_SPREAD",
                symbol,
                description: `Wide spread: ${spreadBps.toFixed(2)} bps`,
                profitPotential: "Arbitrage opportunity"
            });
        }
        
        // Alert on significant opportunities
        for (const opp of opportunities) {
            if (opp.type === "WIDE_SPREAD" && spreadBps > 50) {
                await this.sendAlert(`${opp.type}: ${opp.description} on ${symbol}`);
            }
        }
        
        return opportunities;
    }

    connectRealWebSocket() {
        console.log('Connecting to OKX WebSocket...');
        const ws = new WebSocket('wss://ws.okx.com:8443/ws/v5/public');
        
        ws.on('open', () => {
            console.log('✅ Real WebSocket connected to OKX');
            this.sendAlert("Connected to live OKX order book feeds");
            
            // Subscribe to real order books
            const subscriptions = [
                { "op": "subscribe", "args": [{ "channel": "books5", "instId": "BTC-USDT" }] },
                { "op": "subscribe", "args": [{ "channel": "books5", "instId": "ETH-USDT" }] },
                { "op": "subscribe", "args": [{ "channel": "books5", "instId": "SOL-USDT" }] }
            ];

            subscriptions.forEach(sub => {
                ws.send(JSON.stringify(sub));
                console.log(`📊 Subscribed to ${sub.args[0].instId} order book`);
            });
        });

        ws.on('message', async (data) => {
            try {
                const message = JSON.parse(data);
                
                if (message.data && message.arg && message.arg.channel === 'books5') {
                    const symbol = message.arg.instId;
                    const bookData = message.data[0];
                    
                    // Store real order book data
                    this.orderBook[symbol] = {
                        bids: bookData.bids,
                        asks: bookData.asks,
                        timestamp: bookData.ts
                    };

                    // Analyze the real order book
                    const analysis = this.analyzeRealOrderBook(symbol);
                    if (analysis) {
                        console.log(`${symbol}: Mid=$${analysis.midPrice} | Spread=${analysis.spreadBps.toFixed(2)}bps | Imbalance=${(analysis.imbalance*100).toFixed(1)}%`);
                        
                        // Detect real trading opportunities
                        const opportunities = await this.detectRealOpportunities(analysis);
                        
                        if (opportunities.length > 0) {
                            console.log(`  🎯 ${opportunities.length} opportunities detected:`);
                            opportunities.forEach(opp => {
                                console.log(`     ${opp.type}: ${opp.description}`);
                            });
                        }
                    }
                }
            } catch (error) {
                console.error('WebSocket message error:', error.message);
            }
        });

        ws.on('error', (error) => {
            console.error('❌ WebSocket error:', error.message);
            setTimeout(() => this.connectRealWebSocket(), this.reconnectDelay);
        });

        ws.on('close', (code, reason) => {
            console.log(`WebSocket disconnected (${code}): ${reason}`);
            setTimeout(() => this.connectRealWebSocket(), this.reconnectDelay);
        });

        return ws;
    }

    start() {
        this.running = true;
        this.sendAlert("Real HFT Engine started - monitoring live order books");
        this.connectRealWebSocket();
        
        // Status update every 30 seconds
        setInterval(() => {
            const symbols = Object.keys(this.orderBook);
            console.log(`📈 Monitoring ${symbols.length} symbols: ${symbols.join(', ')}`);
        }, 30000);
    }

    stop() {
        this.running = false;
        this.sendAlert("Real HFT Engine stopped");
    }
}

const hftEngine = new RealHFTEngine();
hftEngine.start();

process.on('SIGINT', () => {
    console.log('Stopping Real HFT Engine...');
    hftEngine.stop();
    process.exit(0);
});
