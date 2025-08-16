const EventEmitter = require('events');

class BaseStrategy extends EventEmitter {
    constructor(config) {
        super();
        this.config = config;
        this.isActive = false;
        this.positions = new Map();
        this.metrics = {
            totalTrades: 0,
            successfulTrades: 0,
            totalProfit: 0,
            totalGasCost: 0,
            averageProfit: 0
        };
        this.minProfitThreshold = config.minProfitThreshold || 0.005;
        this.maxGasPrice = config.maxGasPrice || 100e9;
        this.maxSlippage = config.maxSlippage || 0.02;
    }

    async initialize() {
        this.isActive = true;
        this.emit('initialized');
    }

    async execute(opportunity) {
        if (!this.isActive) return null;
        
        try {
            const profitEstimate = await this.calculateProfit(opportunity);
            if (profitEstimate < this.minProfitThreshold) return null;

            const gasEstimate = await this.estimateGas(opportunity);
            if (gasEstimate > this.maxGasPrice) return null;

            const result = await this.executeStrategy(opportunity);
            await this.updateMetrics(result);
            
            this.emit('tradeExecuted', result);
            return result;
        } catch (error) {
            this.emit('error', error);
            return null;
        }
    }

    async calculateProfit(opportunity) {
        throw new Error('calculateProfit must be implemented');
    }

    async estimateGas(opportunity) {
        throw new Error('estimateGas must be implemented');
    }

    async executeStrategy(opportunity) {
        throw new Error('executeStrategy must be implemented');
    }

    async updateMetrics(result) {
        this.metrics.totalTrades++;
        if (result.success) {
            this.metrics.successfulTrades++;
            this.metrics.totalProfit += result.profit;
        }
        this.metrics.totalGasCost += result.gasCost;
        this.metrics.averageProfit = this.metrics.totalProfit / this.metrics.successfulTrades;
    }

    getMetrics() {
        return { ...this.metrics };
    }

    stop() {
        this.isActive = false;
        this.emit('stopped');
    }
}

module.exports = BaseStrategy;