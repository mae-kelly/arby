const { ethers } = require('ethers');
const EventEmitter = require('events');
const Engine = require('./engine');
const Executor = require('./executor');
const Manager = require('./manager');

class Bot extends EventEmitter {
    constructor(config) {
        super();
        this.config = config;
        this.isRunning = false;
        this.providers = {};
        this.wallets = {};
        this.contracts = {};
        this.strategies = new Map();
        this.performance = {
            startTime: Date.now(),
            totalProfit: 0,
            totalGasUsed: 0,
            successfulTrades: 0,
            failedTrades: 0
        };
        
        this.engine = new Engine(this);
        this.executor = new Executor(this);
        this.manager = new Manager(this);
    }
    
    async initialize() {
        await this.setupProviders();
        await this.setupWallets();
        await this.loadContracts();
        await this.engine.initialize();
        await this.executor.initialize();
        await this.manager.initialize();
        
        this.setupEventHandlers();
        this.emit('initialized');
    }
    
    async setupProviders() {
        for (const [network, config] of Object.entries(this.config.networks)) {
            if (config.websocket) {
                this.providers[network] = new ethers.providers.WebSocketProvider(config.websocket);
            } else {
                this.providers[network] = new ethers.providers.JsonRpcProvider(config.rpc);
            }
            
            this.providers[network].on('block', (blockNumber) => {
                this.emit('block', { network, blockNumber });
            });
            
            this.providers[network].on('pending', (tx) => {
                this.emit('pending', { network, tx });
            });
        }
    }
    
    async setupWallets() {
        for (const [network, provider] of Object.entries(this.providers)) {
            this.wallets[network] = new ethers.Wallet(this.config.privateKey, provider);
        }
    }
    
    async loadContracts() {
        for (const [network, wallet] of Object.entries(this.wallets)) {
            this.contracts[network] = {
                flashLoanArbitrage: new ethers.Contract(
                    this.config.contracts[network].flashLoanArbitrage,
                    this.config.abis.flashLoanArbitrage,
                    wallet
                ),
                crossDexArbitrage: new ethers.Contract(
                    this.config.contracts[network].crossDexArbitrage,
                    this.config.abis.crossDexArbitrage,
                    wallet
                ),
                liquidationBot: new ethers.Contract(
                    this.config.contracts[network].liquidationBot,
                    this.config.abis.liquidationBot,
                    wallet
                )
            };
        }
    }
    
    setupEventHandlers() {
        this.on('opportunity', async (opportunity) => {
            if (this.isRunning) {
                await this.executor.execute(opportunity);
            }
        });
        
        this.on('executed', (result) => {
            this.updatePerformance(result);
        });
        
        this.on('error', (error) => {
            this.handleError(error);
        });
    }
    
    async start() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        await this.engine.start();
        await this.manager.start();
        
        this.emit('started');
    }
    
    async stop() {
        if (!this.isRunning) return;
        
        this.isRunning = false;
        await this.engine.stop();
        await this.manager.stop();
        
        for (const provider of Object.values(this.providers)) {
            if (provider.destroy) {
                await provider.destroy();
            }
        }
        
        this.emit('stopped');
    }
    
    registerStrategy(name, strategy) {
        strategy.bot = this;
        this.strategies.set(name, strategy);
        this.engine.addStrategy(strategy);
    }
    
    updatePerformance(result) {
        if (result.success) {
            this.performance.successfulTrades++;
            this.performance.totalProfit += result.profit;
        } else {
            this.performance.failedTrades++;
        }
        
        this.performance.totalGasUsed += result.gasUsed || 0;
        
        this.emit('performance', this.performance);
    }
    
    handleError(error) {
        const critical = [
            'INSUFFICIENT_FUNDS',
            'CONTRACT_ERROR',
            'PROVIDER_ERROR'
        ];
        
        if (critical.includes(error.code)) {
            this.stop();
        }
    }
    
    async getBalance(network, token) {
        const wallet = this.wallets[network];
        
        if (token === 'ETH') {
            return await wallet.getBalance();
        }
        
        const tokenContract = new ethers.Contract(
            token,
            ['function balanceOf(address) view returns (uint256)'],
            wallet.provider
        );
        
        return await tokenContract.balanceOf(wallet.address);
    }
    
    async estimateGas(network, transaction) {
        try {
            const estimate = await this.providers[network].estimateGas(transaction);
            const gasPrice = await this.providers[network].getGasPrice();
            
            return {
                gasLimit: estimate.mul(110).div(100),
                gasPrice: gasPrice.mul(this.config.gasPriceMultiplier || 120).div(100),
                maxPriorityFeePerGas: ethers.utils.parseUnits(this.config.priorityFee || '2', 'gwei'),
                maxFeePerGas: gasPrice.mul(150).div(100)
            };
        } catch (error) {
            throw new Error(`Gas estimation failed: ${error.message}`);
        }
    }
    
    async sendTransaction(network, transaction) {
        const wallet = this.wallets[network];
        const gasConfig = await this.estimateGas(network, transaction);
        
        const tx = await wallet.sendTransaction({
            ...transaction,
            ...gasConfig,
            nonce: await wallet.getTransactionCount('pending')
        });
        
        return await tx.wait();
    }
    
    async flashLoan(network, params) {
        const contract = this.contracts[network].flashLoanArbitrage;
        
        const tx = await contract.executeArbitrage(
            params.provider,
            params.asset,
            params.amount,
            params.data
        );
        
        return await tx.wait();
    }
    
    async liquidate(network, params) {
        const contract = this.contracts[network].liquidationBot;
        
        const tx = await contract.liquidateAave({
            collateralAsset: params.collateral,
            debtAsset: params.debt,
            user: params.user,
            debtToCover: params.amount,
            receiveAToken: params.receiveAToken || false
        });
        
        return await tx.wait();
    }
    
    getMetrics() {
        const uptime = Date.now() - this.performance.startTime;
        const totalTrades = this.performance.successfulTrades + this.performance.failedTrades;
        
        return {
            uptime,
            totalTrades,
            successRate: totalTrades > 0 ? this.performance.successfulTrades / totalTrades : 0,
            totalProfit: this.performance.totalProfit,
            avgGasPerTrade: totalTrades > 0 ? this.performance.totalGasUsed / totalTrades : 0,
            profitPerHour: this.performance.totalProfit / (uptime / 3600000)
        };
    }
}

module.exports = Bot;