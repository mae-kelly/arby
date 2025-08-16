const EventEmitter = require('events');
const { ethers } = require('ethers');

class OracleService extends EventEmitter {
    constructor(config, chainConnectors) {
        super();
        this.config = config;
        this.chainConnectors = chainConnectors;
        this.oracles = new Map();
        this.priceFeeds = new Map();
        this.lastUpdates = new Map();
        this.deviationThreshold = config.deviationThreshold || 0.05;
        this.updateIntervals = new Map();
        this.aggregatedPrices = new Map();
    }

    async initialize() {
        await this.setupChainlinkOracles();
        await this.setupBandProtocolOracles();
        await this.setupTellorOracles();
        await this.setupUniswapV3Oracles();
        
        this.startPriceAggregation();
        this.startDeviationMonitoring();
        this.emit('initialized');
    }

    async setupChainlinkOracles() {
        const chainlinkFeeds = this.config.chainlinkFeeds || {};
        
        for (const [chainId, feeds] of Object.entries(chainlinkFeeds)) {
            const connector = this.chainConnectors[chainId];
            if (!connector) continue;

            for (const [pair, feedAddress] of Object.entries(feeds)) {
                try {
                    const priceFeed = new ethers.Contract(
                        feedAddress,
                        [
                            'function latestRoundData() view returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound)',
                            'function decimals() view returns (uint8)',
                            'function description() view returns (string)'
                        ],
                        connector.provider
                    );

                    const key = `chainlink-${chainId}-${pair}`;
                    this.oracles.set(key, {
                        type: 'chainlink',
                        chainId,
                        pair,
                        contract: priceFeed,
                        address: feedAddress
                    });

                    await this.updateChainlinkPrice(key);
                } catch (error) {
                    this.emit('error', `Failed to setup Chainlink feed ${pair}: ${error.message}`);
                }
            }
        }
    }

    async setupBandProtocolOracles() {
        const bandFeeds = this.config.bandFeeds || {};
        
        for (const [chainId, config] of Object.entries(bandFeeds)) {
            const connector = this.chainConnectors[chainId];
            if (!connector) continue;

            try {
                const bandContract = new ethers.Contract(
                    config.contractAddress,
                    [
                        'function getReferenceData(string memory _base, string memory _quote) view returns (uint256 rate, uint256 lastUpdatedBase, uint256 lastUpdatedQuote)',
                        'function getReferenceDataBulk(string[] memory _bases, string[] memory _quotes) view returns (uint256[] memory rates, uint256[] memory lastUpdatedBases, uint256[] memory lastUpdatedQuotes)'
                    ],
                    connector.provider
                );

                for (const pair of config.pairs || []) {
                    const [base, quote] = pair.split('/');
                    const key = `band-${chainId}-${pair}`;
                    
                    this.oracles.set(key, {
                        type: 'band',
                        chainId,
                        pair,
                        base,
                        quote,
                        contract: bandContract
                    });

                    await this.updateBandPrice(key);
                }
            } catch (error) {
                this.emit('error', `Failed to setup Band Protocol oracles: ${error.message}`);
            }
        }
    }

    async setupTellorOracles() {
        const tellorFeeds = this.config.tellorFeeds || {};
        
        for (const [chainId, config] of Object.entries(tellorFeeds)) {
            const connector = this.chainConnectors[chainId];
            if (!connector) continue;

            try {
                const tellorContract = new ethers.Contract(
                    config.contractAddress,
                    [
                        'function getCurrentValue(bytes32 _queryId) view returns (bool ifRetrieve, bytes memory value, uint256 _timestampRetrieved)',
                        'function getDataBefore(bytes32 _queryId, uint256 _timestamp) view returns (bool _ifRetrieve, bytes memory _value, uint256 _timestampRetrieved)'
                    ],
                    connector.provider
                );

                for (const [pair, queryId] of Object.entries(config.queryIds || {})) {
                    const key = `tellor-${chainId}-${pair}`;
                    
                    this.oracles.set(key, {
                        type: 'tellor',
                        chainId,
                        pair,
                        contract: tellorContract,
                        queryId
                    });

                    await this.updateTellorPrice(key);
                }
            } catch (error) {
                this.emit('error', `Failed to setup Tellor oracles: ${error.message}`);
            }
        }
    }

    async setupUniswapV3Oracles() {
        const uniswapFeeds = this.config.uniswapV3Feeds || {};
        
        for (const [chainId, pools] of Object.entries(uniswapFeeds)) {
            const connector = this.chainConnectors[chainId];
            if (!connector) continue;

            for (const [pair, poolAddress] of Object.entries(pools)) {
                try {
                    const poolContract = new ethers.Contract(
                        poolAddress,
                        [
                            'function slot0() view returns (uint160 sqrtPriceX96, int24 tick, uint16 observationIndex, uint16 observationCardinality, uint16 observationCardinalityNext, uint8 feeProtocol, bool unlocked)',
                            'function observe(uint32[] calldata secondsAgos) view returns (int56[] memory tickCumulatives, uint160[] memory secondsPerLiquidityCumulativeX128s)'
                        ],
                        connector.provider
                    );

                    const key = `uniswap-${chainId}-${pair}`;
                    this.oracles.set(key, {
                        type: 'uniswap_v3',
                        chainId,
                        pair,
                        contract: poolContract,
                        address: poolAddress
                    });

                    await this.updateUniswapPrice(key);
                } catch (error) {
                    this.emit('error', `Failed to setup Uniswap V3 oracle ${pair}: ${error.message}`);
                }
            }
        }
    }

    async updateChainlinkPrice(oracleKey) {
        try {
            const oracle = this.oracles.get(oracleKey);
            if (!oracle) return;

            const [roundId, answer, startedAt, updatedAt, answeredInRound] = await oracle.contract.latestRoundData();
            const decimals = await oracle.contract.decimals();
            
            const price = parseFloat(ethers.utils.formatUnits(answer, decimals));
            
            this.priceFeeds.set(oracleKey, {
                price,
                timestamp: updatedAt.toNumber() * 1000,
                roundId: roundId.toString(),
                source: 'chainlink',
                pair: oracle.pair,
                decimals,
                lastFetched: Date.now()
            });

            this.lastUpdates.set(oracleKey, Date.now());
            
            this.emit('priceUpdate', {
                source: 'chainlink',
                pair: oracle.pair,
                price,
                timestamp: updatedAt.toNumber() * 1000
            });

        } catch (error) {
            this.emit('error', `Failed to update Chainlink price for ${oracleKey}: ${error.message}`);
        }
    }

    async updateBandPrice(oracleKey) {
        try {
            const oracle = this.oracles.get(oracleKey);
            if (!oracle) return;

            const [rate, lastUpdatedBase, lastUpdatedQuote] = await oracle.contract.getReferenceData(
                oracle.base,
                oracle.quote
            );
            
            const price = parseFloat(ethers.utils.formatUnits(rate, 18));
            const timestamp = Math.min(lastUpdatedBase, lastUpdatedQuote).toNumber() * 1000;
            
            this.priceFeeds.set(oracleKey, {
                price,
                timestamp,
                source: 'band',
                pair: oracle.pair,
                lastFetched: Date.now()
            });

            this.lastUpdates.set(oracleKey, Date.now());
            
            this.emit('priceUpdate', {
                source: 'band',
                pair: oracle.pair,
                price,
                timestamp
            });

        } catch (error) {
            this.emit('error', `Failed to update Band price for ${oracleKey}: ${error.message}`);
        }
    }

    async updateTellorPrice(oracleKey) {
        try {
            const oracle = this.oracles.get(oracleKey);
            if (!oracle) return;

            const [ifRetrieve, value, timestampRetrieved] = await oracle.contract.getCurrentValue(oracle.queryId);
            
            if (!ifRetrieve) return;
            
            const price = parseFloat(ethers.utils.formatUnits(ethers.BigNumber.from(value), 18));
            const timestamp = timestampRetrieved.toNumber() * 1000;
            
            this.priceFeeds.set(oracleKey, {
                price,
                timestamp,
                source: 'tellor',
                pair: oracle.pair,
                lastFetched: Date.now()
            });

            this.lastUpdates.set(oracleKey, Date.now());
            
            this.emit('priceUpdate', {
                source: 'tellor',
                pair: oracle.pair,
                price,
                timestamp
            });

        } catch (error) {
            this.emit('error', `Failed to update Tellor price for ${oracleKey}: ${error.message}`);
        }
    }

    async updateUniswapPrice(oracleKey) {
        try {
            const oracle = this.oracles.get(oracleKey);
            if (!oracle) return;

            const [sqrtPriceX96] = await oracle.contract.slot0();
            
            const price = this.sqrtPriceX96ToPrice(sqrtPriceX96);
            
            this.priceFeeds.set(oracleKey, {
                price,
                timestamp: Date.now(),
                source: 'uniswap_v3',
                pair: oracle.pair,
                sqrtPriceX96: sqrtPriceX96.toString(),
                lastFetched: Date.now()
            });

            this.lastUpdates.set(oracleKey, Date.now());
            
            this.emit('priceUpdate', {
                source: 'uniswap_v3',
                pair: oracle.pair,
                price,
                timestamp: Date.now()
            });

        } catch (error) {
            this.emit('error', `Failed to update Uniswap price for ${oracleKey}: ${error.message}`);
        }
    }

    sqrtPriceX96ToPrice(sqrtPriceX96) {
        const Q96 = ethers.BigNumber.from(2).pow(96);
        const price = sqrtPriceX96.mul(sqrtPriceX96).div(Q96).div(Q96);
        return parseFloat(ethers.utils.formatEther(price));
    }

    async getPrice(pair, sources = []) {
        const aggregatedPrice = this.aggregatedPrices.get(pair);
        
        if (sources.length === 0) {
            return aggregatedPrice || null;
        }

        const sourcePrices = [];
        for (const source of sources) {
            const oracleKey = `${source}-${pair}`;
            const priceData = this.priceFeeds.get(oracleKey);
            if (priceData && this.isPriceRecent(priceData)) {
                sourcePrices.push(priceData);
            }
        }

        if (sourcePrices.length === 0) return null;
        if (sourcePrices.length === 1) return sourcePrices[0];

        return this.aggregatePrices(sourcePrices);
    }

    aggregatePrices(priceData) {
        if (priceData.length === 0) return null;
        
        const validPrices = priceData.filter(p => p.price > 0);
        if (validPrices.length === 0) return null;

        const prices = validPrices.map(p => p.price);
        const weights = validPrices.map(p => this.calculateWeight(p));
        
        const weightedSum = prices.reduce((sum, price, i) => sum + price * weights[i], 0);
        const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
        
        const aggregatedPrice = weightedSum / totalWeight;
        const median = this.calculateMedian(prices);
        const deviation = this.calculateDeviation(prices);

        return {
            price: aggregatedPrice,
            median,
            deviation,
            sources: validPrices.map(p => p.source),
            timestamp: Math.max(...validPrices.map(p => p.timestamp)),
            confidence: this.calculateConfidence(prices, deviation)
        };
    }

    calculateWeight(priceData) {
        const age = Date.now() - priceData.timestamp;
        const ageWeight = Math.max(0, 1 - age / (24 * 60 * 60 * 1000));
        
        const sourceWeights = {
            chainlink: 1.0,
            band: 0.8,
            tellor: 0.6,
            uniswap_v3: 0.7
        };
        
        const sourceWeight = sourceWeights[priceData.source] || 0.5;
        
        return ageWeight * sourceWeight;
    }

    calculateMedian(prices) {
        const sorted = [...prices].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0 ? 
            (sorted[mid - 1] + sorted[mid]) / 2 : 
            sorted[mid];
    }

    calculateDeviation(prices) {
        const mean = prices.reduce((sum, price) => sum + price, 0) / prices.length;
        const variance = prices.reduce((sum, price) => sum + Math.pow(price - mean, 2), 0) / prices.length;
        return Math.sqrt(variance) / mean;
    }

    calculateConfidence(prices, deviation) {
        if (prices.length < 2) return 0.5;
        if (deviation < 0.01) return 1.0;
        if (deviation < 0.05) return 0.8;
        if (deviation < 0.1) return 0.6;
        return 0.4;
    }

    isPriceRecent(priceData, maxAge = 3600000) {
        return Date.now() - priceData.timestamp < maxAge;
    }

    startPriceAggregation() {
        const interval = setInterval(() => {
            this.aggregateAllPrices();
        }, this.config.aggregationInterval || 30000);
        
        this.updateIntervals.set('aggregation', interval);
    }

    aggregateAllPrices() {
        const pairs = new Set();
        
        for (const [key, oracle] of this.oracles.entries()) {
            pairs.add(oracle.pair);
        }

        for (const pair of pairs) {
            const oracleKeys = Array.from(this.oracles.keys())
                .filter(key => this.oracles.get(key).pair === pair);
            
            const priceData = oracleKeys
                .map(key => this.priceFeeds.get(key))
                .filter(data => data && this.isPriceRecent(data));

            if (priceData.length > 0) {
                const aggregated = this.aggregatePrices(priceData);
                this.aggregatedPrices.set(pair, aggregated);
                
                this.emit('aggregatedPrice', {
                    pair,
                    ...aggregated
                });
            }
        }
    }

    startDeviationMonitoring() {
        const interval = setInterval(() => {
            this.monitorPriceDeviations();
        }, this.config.deviationCheckInterval || 60000);
        
        this.updateIntervals.set('deviation', interval);
    }

    monitorPriceDeviations() {
        for (const [pair, aggregatedPrice] of this.aggregatedPrices.entries()) {
            if (aggregatedPrice.deviation > this.deviationThreshold) {
                this.emit('priceDeviation', {
                    pair,
                    deviation: aggregatedPrice.deviation,
                    threshold: this.deviationThreshold,
                    sources: aggregatedPrice.sources,
                    timestamp: Date.now()
                });
            }
        }
    }

    async refreshAllPrices() {
        const updatePromises = [];
        
        for (const [key, oracle] of this.oracles.entries()) {
            switch (oracle.type) {
                case 'chainlink':
                    updatePromises.push(this.updateChainlinkPrice(key));
                    break;
                case 'band':
                    updatePromises.push(this.updateBandPrice(key));
                    break;
                case 'tellor':
                    updatePromises.push(this.updateTellorPrice(key));
                    break;
                case 'uniswap_v3':
                    updatePromises.push(this.updateUniswapPrice(key));
                    break;
            }
        }

        await Promise.allSettled(updatePromises);
        this.aggregateAllPrices();
    }

    async getTWAP(pair, period = 3600) {
        const oracleKey = `uniswap-1-${pair}`;
        const oracle = this.oracles.get(oracleKey);
        
        if (!oracle || oracle.type !== 'uniswap_v3') {
            return null;
        }

        try {
            const secondsAgos = [period, 0];
            const [tickCumulatives] = await oracle.contract.observe(secondsAgos);
            
            const tickCumulativesDelta = tickCumulatives[1].sub(tickCumulatives[0]);
            const timeWeightedAverageTick = tickCumulativesDelta.div(period);
            
            const price = Math.pow(1.0001, timeWeightedAverageTick.toNumber());
            
            return {
                price,
                period,
                pair,
                timestamp: Date.now(),
                source: 'uniswap_v3_twap'
            };

        } catch (error) {
            this.emit('error', `Failed to get TWAP for ${pair}: ${error.message}`);
            return null;
        }
    }

    async getHistoricalPrice(pair, timestamp, source = null) {
        if (source === 'tellor') {
            const oracleKey = `tellor-1-${pair}`;
            const oracle = this.oracles.get(oracleKey);
            
            if (oracle) {
                try {
                    const [ifRetrieve, value, timestampRetrieved] = await oracle.contract.getDataBefore(
                        oracle.queryId,
                        Math.floor(timestamp / 1000)
                    );
                    
                    if (ifRetrieve) {
                        return {
                            price: parseFloat(ethers.utils.formatUnits(ethers.BigNumber.from(value), 18)),
                            timestamp: timestampRetrieved.toNumber() * 1000,
                            source: 'tellor',
                            pair
                        };
                    }
                } catch (error) {
                    this.emit('error', `Failed to get historical Tellor price: ${error.message}`);
                }
            }
        }

        return null;
    }

    validatePrice(pair, price, maxDeviation = 0.1) {
        const aggregatedPrice = this.aggregatedPrices.get(pair);
        if (!aggregatedPrice) return { valid: false, reason: 'No reference price' };

        const deviation = Math.abs(price - aggregatedPrice.price) / aggregatedPrice.price;
        
        if (deviation > maxDeviation) {
            return {
                valid: false,
                reason: 'Price deviation too high',
                deviation,
                reference: aggregatedPrice.price,
                threshold: maxDeviation
            };
        }

        return { valid: true, deviation, reference: aggregatedPrice.price };
    }

    getOracleStatus() {
        const status = {
            totalOracles: this.oracles.size,
            activeOracles: 0,
            inactiveOracles: 0,
            lastUpdate: 0,
            sources: {}
        };

        for (const [key, oracle] of this.oracles.entries()) {
            const lastUpdate = this.lastUpdates.get(key) || 0;
            const isActive = Date.now() - lastUpdate < 300000;
            
            if (isActive) {
                status.activeOracles++;
            } else {
                status.inactiveOracles++;
            }
            
            status.lastUpdate = Math.max(status.lastUpdate, lastUpdate);
            
            if (!status.sources[oracle.type]) {
                status.sources[oracle.type] = { active: 0, inactive: 0 };
            }
            
            if (isActive) {
                status.sources[oracle.type].active++;
            } else {
                status.sources[oracle.type].inactive++;
            }
        }

        return status;
    }

    getAllPrices() {
        return Object.fromEntries(this.aggregatedPrices);
    }

    cleanup() {
        for (const interval of this.updateIntervals.values()) {
            clearInterval(interval);
        }
        this.updateIntervals.clear();
    }
}

module.exports = OracleService;