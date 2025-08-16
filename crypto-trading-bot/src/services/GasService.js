const EventEmitter = require('events');
const { ethers } = require('ethers');

class GasService extends EventEmitter {
    constructor(config, chainConnectors) {
        super();
        this.config = config;
        this.chainConnectors = chainConnectors;
        this.gasTrackers = new Map();
        this.gasHistory = new Map();
        this.gasStations = new Map();
        this.updateIntervals = new Map();
        this.baseFeeHistory = new Map();
        this.priorityFeeHistory = new Map();
    }

    async initialize() {
        for (const [chainId, connector] of Object.entries(this.chainConnectors)) {
            await this.initializeChainGasTracking(chainId, connector);
        }
        
        this.startGasTracking();
        this.startBaseFeeTracking();
        this.emit('initialized');
    }

    async initializeChainGasTracking(chainId, connector) {
        this.gasTrackers.set(chainId, {
            fast: ethers.BigNumber.from(0),
            standard: ethers.BigNumber.from(0),
            safe: ethers.BigNumber.from(0),
            baseFee: ethers.BigNumber.from(0),
            maxPriorityFee: ethers.BigNumber.from(0),
            lastUpdate: 0
        });

        this.gasHistory.set(chainId, []);
        this.baseFeeHistory.set(chainId, []);
        this.priorityFeeHistory.set(chainId, []);

        await this.updateGasPrice(chainId);
    }

    async updateGasPrice(chainId) {
        try {
            const connector = this.chainConnectors[chainId];
            if (!connector) return;

            const gasPrice = await connector.provider.getGasPrice();
            const block = await connector.provider.getBlock('latest');
            
            let baseFee = ethers.BigNumber.from(0);
            let maxPriorityFee = ethers.BigNumber.from(0);
            
            if (block.baseFeePerGas) {
                baseFee = block.baseFeePerGas;
                maxPriorityFee = gasPrice.sub(baseFee);
            }

            const gasData = {
                fast: gasPrice.mul(120).div(100),
                standard: gasPrice,
                safe: gasPrice.mul(80).div(100),
                baseFee,
                maxPriorityFee,
                lastUpdate: Date.now(),
                blockNumber: block.number
            };

            if (chainId === '1') {
                await this.updateEthereumGasStation();
                const ethGasStation = this.gasStations.get('1');
                if (ethGasStation) {
                    gasData.fast = ethers.utils.parseUnits(ethGasStation.fast.toString(), 'gwei');
                    gasData.standard = ethers.utils.parseUnits(ethGasStation.standard.toString(), 'gwei');
                    gasData.safe = ethers.utils.parseUnits(ethGasStation.safe.toString(), 'gwei');
                }
            } else if (chainId === '137') {
                await this.updatePolygonGasStation();
                const polygonGasStation = this.gasStations.get('137');
                if (polygonGasStation) {
                    gasData.fast = ethers.utils.parseUnits(polygonGasStation.fast.toString(), 'gwei');
                    gasData.standard = ethers.utils.parseUnits(polygonGasStation.standard.toString(), 'gwei');
                    gasData.safe = ethers.utils.parseUnits(polygonGasStation.safe.toString(), 'gwei');
                }
            }

            this.gasTrackers.set(chainId, gasData);
            this.updateGasHistory(chainId, gasData);
            
            this.emit('gasUpdate', {
                chainId,
                ...gasData,
                formattedPrices: {
                    fast: ethers.utils.formatUnits(gasData.fast, 'gwei'),
                    standard: ethers.utils.formatUnits(gasData.standard, 'gwei'),
                    safe: ethers.utils.formatUnits(gasData.safe, 'gwei'),
                    baseFee: ethers.utils.formatUnits(gasData.baseFee, 'gwei')
                }
            });

        } catch (error) {
            this.emit('error', `Failed to update gas price for chain ${chainId}: ${error.message}`);
        }
    }

    async updateEthereumGasStation() {
        try {
            const response = await fetch('https://ethgasstation.info/api/ethgasAPI.json');
            const data = await response.json();
            
            this.gasStations.set('1', {
                fast: data.fast / 10,
                standard: data.average / 10,
                safe: data.safeLow / 10,
                timestamp: Date.now()
            });
        } catch (error) {
            try {
                const response = await fetch('https://api.etherscan.io/api?module=gastracker&action=gasoracle');
                const data = await response.json();
                
                if (data.status === '1') {
                    this.gasStations.set('1', {
                        fast: parseFloat(data.result.FastGasPrice),
                        standard: parseFloat(data.result.StandardGasPrice),
                        safe: parseFloat(data.result.SafeGasPrice),
                        timestamp: Date.now()
                    });
                }
            } catch (fallbackError) {
                this.emit('error', `Failed to fetch Ethereum gas station data: ${fallbackError.message}`);
            }
        }
    }

    async updatePolygonGasStation() {
        try {
            const response = await fetch('https://gasstation.polygon.technology/v2');
            const data = await response.json();
            
            this.gasStations.set('137', {
                fast: data.fast.maxFee,
                standard: data.standard.maxFee,
                safe: data.safeLow?.maxFee || data.standard.maxFee * 0.8,
                timestamp: Date.now()
            });
        } catch (error) {
            this.emit('error', `Failed to fetch Polygon gas station data: ${error.message}`);
        }
    }

    updateGasHistory(chainId, gasData) {
        if (!this.gasHistory.has(chainId)) {
            this.gasHistory.set(chainId, []);
        }

        const history = this.gasHistory.get(chainId);
        history.push({
            fast: gasData.fast,
            standard: gasData.standard,
            safe: gasData.safe,
            baseFee: gasData.baseFee,
            timestamp: Date.now(),
            blockNumber: gasData.blockNumber
        });

        if (history.length > 1000) {
            history.shift();
        }

        if (gasData.baseFee.gt(0)) {
            const baseFeeHistory = this.baseFeeHistory.get(chainId) || [];
            baseFeeHistory.push({
                baseFee: gasData.baseFee,
                timestamp: Date.now(),
                blockNumber: gasData.blockNumber
            });
            
            if (baseFeeHistory.length > 500) {
                baseFeeHistory.shift();
            }
            
            this.baseFeeHistory.set(chainId, baseFeeHistory);
        }
    }

    startGasTracking() {
        for (const chainId of Object.keys(this.chainConnectors)) {
            const interval = setInterval(async () => {
                await this.updateGasPrice(chainId);
            }, this.config.gasUpdateInterval || 15000);
            
            this.updateIntervals.set(chainId, interval);
        }
    }

    startBaseFeeTracking() {
        for (const [chainId, connector] of Object.entries(this.chainConnectors)) {
            if (chainId === '1' || chainId === '137') {
                connector.provider.on('block', async (blockNumber) => {
                    try {
                        const block = await connector.provider.getBlock(blockNumber);
                        if (block.baseFeePerGas) {
                            this.updateBaseFeeHistory(chainId, block.baseFeePerGas, blockNumber);
                        }
                    } catch (error) {
                        // Ignore errors
                    }
                });
            }
        }
    }

    updateBaseFeeHistory(chainId, baseFee, blockNumber) {
        const history = this.baseFeeHistory.get(chainId) || [];
        history.push({
            baseFee,
            timestamp: Date.now(),
            blockNumber
        });

        if (history.length > 100) {
            history.shift();
        }

        this.baseFeeHistory.set(chainId, history);
        
        this.emit('baseFeeUpdate', {
            chainId,
            baseFee,
            blockNumber,
            formatted: ethers.utils.formatUnits(baseFee, 'gwei')
        });
    }

    getGasPrice(chainId, speed = 'standard') {
        const gasData = this.gasTrackers.get(chainId);
        if (!gasData) return null;

        return gasData[speed] || gasData.standard;
    }

    getOptimalGasPrice(chainId, urgency = 'normal') {
        const gasData = this.gasTrackers.get(chainId);
        if (!gasData) return null;

        const history = this.gasHistory.get(chainId) || [];
        if (history.length < 5) {
            return gasData.standard;
        }

        const recentHistory = history.slice(-10);
        const avgGasPrice = recentHistory.reduce((sum, h) => 
            sum.add(h.standard), ethers.BigNumber.from(0)
        ).div(recentHistory.length);

        switch (urgency) {
            case 'low':
                return avgGasPrice.mul(90).div(100);
            case 'high':
                return avgGasPrice.mul(130).div(100);
            case 'urgent':
                return avgGasPrice.mul(150).div(100);
            default:
                return avgGasPrice.mul(110).div(100);
        }
    }

    getEIP1559GasPrice(chainId, urgency = 'normal') {
        const gasData = this.gasTrackers.get(chainId);
        if (!gasData || gasData.baseFee.eq(0)) {
            return this.getOptimalGasPrice(chainId, urgency);
        }

        const baseFeeHistory = this.baseFeeHistory.get(chainId) || [];
        if (baseFeeHistory.length < 3) {
            return {
                maxFeePerGas: gasData.standard,
                maxPriorityFeePerGas: ethers.utils.parseUnits('2', 'gwei')
            };
        }

        const recentBaseFees = baseFeeHistory.slice(-5).map(h => h.baseFee);
        const avgBaseFee = recentBaseFees.reduce((sum, bf) => 
            sum.add(bf), ethers.BigNumber.from(0)
        ).div(recentBaseFees.length);

        const baseFeeMultiplier = {
            low: 110,
            normal: 120,
            high: 140,
            urgent: 160
        }[urgency] || 120;

        const priorityFeeMultiplier = {
            low: ethers.utils.parseUnits('1', 'gwei'),
            normal: ethers.utils.parseUnits('2', 'gwei'),
            high: ethers.utils.parseUnits('3', 'gwei'),
            urgent: ethers.utils.parseUnits('5', 'gwei')
        }[urgency] || ethers.utils.parseUnits('2', 'gwei');

        const maxFeePerGas = avgBaseFee.mul(baseFeeMultiplier).div(100).add(priorityFeeMultiplier);
        
        return {
            maxFeePerGas,
            maxPriorityFeePerGas: priorityFeeMultiplier
        };
    }

    async estimateTransactionCost(chainId, gasLimit, urgency = 'normal') {
        const gasPrice = this.getOptimalGasPrice(chainId, urgency);
        if (!gasPrice) return null;

        const totalCost = gasPrice.mul(gasLimit);
        const connector = this.chainConnectors[chainId];
        
        let nativeTokenPrice = 1;
        try {
            if (chainId === '1') {
                nativeTokenPrice = await this.getETHPrice();
            } else if (chainId === '56') {
                nativeTokenPrice = await this.getBNBPrice();
            } else if (chainId === '137') {
                nativeTokenPrice = await this.getMATICPrice();
            }
        } catch (error) {
            // Use fallback price
        }

        return {
            gasPrice,
            gasLimit,
            totalCost,
            costInNative: ethers.utils.formatEther(totalCost),
            costInUSD: parseFloat(ethers.utils.formatEther(totalCost)) * nativeTokenPrice,
            urgency
        };
    }

    async getETHPrice() {
        try {
            const response = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd');
            const data = await response.json();
            return data.ethereum.usd;
        } catch (error) {
            return 2000;
        }
    }

    async getBNBPrice() {
        try {
            const response = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=binancecoin&vs_currencies=usd');
            const data = await response.json();
            return data.binancecoin.usd;
        } catch (error) {
            return 300;
        }
    }

    async getMATICPrice() {
        try {
            const response = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=matic-network&vs_currencies=usd');
            const data = await response.json();
            return data['matic-network'].usd;
        } catch (error) {
            return 1;
        }
    }

    getGasTrend(chainId, timeframe = 3600000) {
        const history = this.gasHistory.get(chainId) || [];
        const cutoff = Date.now() - timeframe;
        const recentHistory = history.filter(h => h.timestamp > cutoff);

        if (recentHistory.length < 2) return 'stable';

        const firstPrice = recentHistory[0].standard;
        const lastPrice = recentHistory[recentHistory.length - 1].standard;
        const change = lastPrice.sub(firstPrice).mul(100).div(firstPrice);

        if (change.gt(10)) return 'rising';
        if (change.lt(-10)) return 'falling';
        return 'stable';
    }

    getGasStats(chainId) {
        const gasData = this.gasTrackers.get(chainId);
        const history = this.gasHistory.get(chainId) || [];
        
        if (!gasData || history.length === 0) return null;

        const recentHistory = history.slice(-50);
        const prices = recentHistory.map(h => parseFloat(ethers.utils.formatUnits(h.standard, 'gwei')));
        
        const avg = prices.reduce((sum, p) => sum + p, 0) / prices.length;
        const min = Math.min(...prices);
        const max = Math.max(...prices);
        
        return {
            current: {
                fast: ethers.utils.formatUnits(gasData.fast, 'gwei'),
                standard: ethers.utils.formatUnits(gasData.standard, 'gwei'),
                safe: ethers.utils.formatUnits(gasData.safe, 'gwei'),
                baseFee: ethers.utils.formatUnits(gasData.baseFee, 'gwei')
            },
            statistics: {
                average: avg.toFixed(2),
                minimum: min.toFixed(2),
                maximum: max.toFixed(2),
                trend: this.getGasTrend(chainId)
            },
            lastUpdate: gasData.lastUpdate
        };
    }

    async predictGasPrice(chainId, blocksAhead = 5) {
        const history = this.gasHistory.get(chainId) || [];
        if (history.length < 10) return this.getGasPrice(chainId);

        const recentPrices = history.slice(-20).map(h => 
            parseFloat(ethers.utils.formatUnits(h.standard, 'gwei'))
        );

        const weights = recentPrices.map((_, i) => (i + 1) / recentPrices.length);
        const weightedAvg = recentPrices.reduce((sum, price, i) => 
            sum + price * weights[i], 0
        ) / weights.reduce((sum, w) => sum + w, 0);

        const volatility = this.calculateGasVolatility(chainId);
        const trend = this.getGasTrend(chainId);
        
        let prediction = weightedAvg;
        
        if (trend === 'rising') {
            prediction *= (1 + volatility * 0.5);
        } else if (trend === 'falling') {
            prediction *= (1 - volatility * 0.3);
        }

        return ethers.utils.parseUnits(prediction.toFixed(2), 'gwei');
    }

    calculateGasVolatility(chainId) {
        const history = this.gasHistory.get(chainId) || [];
        if (history.length < 10) return 0;

        const prices = history.slice(-20).map(h => 
            parseFloat(ethers.utils.formatUnits(h.standard, 'gwei'))
        );

        const returns = [];
        for (let i = 1; i < prices.length; i++) {
            returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
        }

        const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
        
        return Math.sqrt(variance);
    }

    async getOptimalTransactionTiming(chainId, targetGasPrice) {
        const currentGas = this.getGasPrice(chainId);
        if (!currentGas) return 'now';

        const targetGwei = parseFloat(ethers.utils.formatUnits(targetGasPrice, 'gwei'));
        const currentGwei = parseFloat(ethers.utils.formatUnits(currentGas, 'gwei'));

        if (currentGwei <= targetGwei) {
            return 'now';
        }

        const trend = this.getGasTrend(chainId);
        const volatility = this.calculateGasVolatility(chainId);

        if (trend === 'falling' && volatility < 0.1) {
            return 'wait_5_minutes';
        } else if (trend === 'rising') {
            return 'send_immediately';
        } else {
            return 'wait_15_minutes';
        }
    }

    cleanup() {
        for (const interval of this.updateIntervals.values()) {
            clearInterval(interval);
        }
        this.updateIntervals.clear();
    }
}

module.exports = GasService;