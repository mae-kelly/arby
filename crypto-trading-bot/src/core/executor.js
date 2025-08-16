const { ethers } = require('ethers');
const EventEmitter = require('events');

class Executor extends EventEmitter {
    constructor(bot) {
        super();
        this.bot = bot;
        this.queue = [];
        this.executing = new Map();
        this.results = new Map();
        this.gasTracker = new Map();
        this.nonces = new Map();
        this.flashbots = null;
        this.maxConcurrent = 5;
        this.retryAttempts = 3;
    }
    
    async initialize() {
        await this.setupFlashbots();
        await this.setupGasTracking();
        this.startExecutionLoop();
    }
    
    async setupFlashbots() {
        if (this.bot.config.flashbots && this.bot.config.flashbots.enabled) {
            const { FlashbotsBundleProvider } = require('@flashbots/ethers-provider-bundle');
            
            this.flashbots = await FlashbotsBundleProvider.create(
                this.bot.providers.ethereum,
                this.bot.wallets.ethereum,
                this.bot.config.flashbots.relay || 'https://relay.flashbots.net'
            );
        }
    }
    
    async setupGasTracking() {
        for (const network of Object.keys(this.bot.providers)) {
            this.gasTracker.set(network, {
                base: ethers.BigNumber.from('0'),
                priority: ethers.BigNumber.from('0'),
                lastUpdate: 0
            });
            
            await this.updateGasPrice(network);
        }
    }
    
    async updateGasPrice(network) {
        const provider = this.bot.providers[network];
        
        try {
            if (provider.getFeeData) {
                const feeData = await provider.getFeeData();
                this.gasTracker.set(network, {
                    base: feeData.gasPrice || ethers.BigNumber.from('0'),
                    priority: feeData.maxPriorityFeePerGas || ethers.BigNumber.from('0'),
                    lastUpdate: Date.now()
                });
            } else {
                const gasPrice = await provider.getGasPrice();
                this.gasTracker.set(network, {
                    base: gasPrice,
                    priority: ethers.BigNumber.from('0'),
                    lastUpdate: Date.now()
                });
            }
        } catch {}
    }
    
    startExecutionLoop() {
        setInterval(async () => {
            await this.processQueue();
        }, 100);
        
        setInterval(async () => {
            for (const network of this.gasTracker.keys()) {
                await this.updateGasPrice(network);
            }
        }, 5000);
    }
    
    async execute(opportunity) {
        const executionId = `exec-${opportunity.id}`;
        
        if (this.executing.has(executionId)) {
            return this.results.get(executionId);
        }
        
        this.executing.set(executionId, opportunity);
        
        try {
            const result = await this.routeExecution(opportunity);
            
            this.results.set(executionId, result);
            this.bot.emit('executed', result);
            
            return result;
        } catch (error) {
            const failedResult = {
                success: false,
                opportunity,
                error: error.message,
                timestamp: Date.now()
            };
            
            this.results.set(executionId, failedResult);
            this.bot.emit('execution_failed', failedResult);
            
            return failedResult;
        } finally {
            this.executing.delete(executionId);
        }
    }
    
    async routeExecution(opportunity) {
        switch (opportunity.type) {
            case 'dex_arbitrage':
                return await this.executeDexArbitrage(opportunity);
            case 'liquidation':
                return await this.executeLiquidation(opportunity);
            case 'cross_chain':
                return await this.executeCrossChain(opportunity);
            case 'backrun':
                return await this.executeBackrun(opportunity);
            case 'triangular':
                return await this.executeTriangular(opportunity);
            default:
                throw new Error(`Unknown opportunity type: ${opportunity.type}`);
        }
    }
    
    async executeDexArbitrage(opportunity) {
        const { network, tokenIn, tokenOut, amountIn, path } = opportunity;
        const wallet = this.bot.wallets[network];
        
        const balance = await this.bot.getBalance(network, tokenIn);
        if (balance.lt(amountIn)) {
            return await this.executeFlashLoanArbitrage(opportunity);
        }
        
        const contract = this.bot.contracts[network].crossDexArbitrage;
        const gasConfig = await this.getOptimalGas(network, opportunity);
        
        const tx = await contract.executeCrossArbitrage({
            tokenIn,
            tokenOut,
            amountIn,
            dexRouters: path.map(p => this.bot.config.dex[network][p].router),
            swapData: await this.encodeSwapData(network, path, tokenIn, tokenOut, amountIn),
            minAmountOut: opportunity.expectedProfit.mul(95).div(100)
        }, gasConfig);
        
        const receipt = await tx.wait();
        
        return {
            success: receipt.status === 1,
            opportunity,
            txHash: receipt.transactionHash,
            gasUsed: receipt.gasUsed,
            profit: await this.calculateProfit(receipt, opportunity),
            timestamp: Date.now()
        };
    }
    
    async executeFlashLoanArbitrage(opportunity) {
        const { network, tokenIn, amountIn } = opportunity;
        
        const provider = this.selectFlashLoanProvider(network, tokenIn, amountIn);
        const params = await this.encodeArbitrageParams(opportunity);
        
        const contract = this.bot.contracts[network].flashLoanArbitrage;
        const gasConfig = await this.getOptimalGas(network, opportunity);
        
        const tx = await contract.executeArbitrage(
            provider,
            tokenIn,
            amountIn,
            params,
            gasConfig
        );
        
        const receipt = await tx.wait();
        
        return {
            success: receipt.status === 1,
            opportunity,
            txHash: receipt.transactionHash,
            gasUsed: receipt.gasUsed,
            profit: await this.calculateProfit(receipt, opportunity),
            timestamp: Date.now()
        };
    }
    
    selectFlashLoanProvider(network, asset, amount) {
        const providers = this.bot.config.flashLoanProviders[network];
        
        if (providers.balancer && providers.balancer.assets.includes(asset)) {
            return providers.balancer.vault;
        }
        
        return providers.aave.pool;
    }
    
    async encodeArbitrageParams(opportunity) {
        const swapPath = [];
        const swapData = [];
        
        for (let i = 0; i < opportunity.path.length; i++) {
            const dex = opportunity.path[i];
            const router = this.bot.config.dex[opportunity.network][dex].router;
            
            swapPath.push(router);
            swapData.push(await this.encodeSwapData(
                opportunity.network,
                [dex],
                opportunity.tokenIn,
                opportunity.tokenOut,
                opportunity.amountIn
            ));
        }
        
        return ethers.utils.defaultAbiCoder.encode(
            ['address[]', 'uint256[]', 'address[]', 'bytes', 'uint256'],
            [
                [opportunity.tokenIn, opportunity.tokenOut],
                [opportunity.amountIn],
                swapPath,
                ethers.utils.concat(swapData),
                opportunity.expectedProfit
            ]
        );
    }
    
    async encodeSwapData(network, path, tokenIn, tokenOut, amountIn) {
        const swapData = [];
        
        for (const dex of path) {
            const config = this.bot.config.dex[network][dex];
            
            if (dex === 'uniswapV3') {
                swapData.push(
                    ethers.utils.defaultAbiCoder.encode(
                        ['address', 'address', 'uint24', 'address', 'uint256', 'uint256', 'uint256', 'uint160'],
                        [tokenIn, tokenOut, 3000, this.bot.wallets[network].address, Date.now() + 300, amountIn, 0, 0]
                    )
                );
            } else {
                swapData.push(
                    ethers.utils.defaultAbiCoder.encode(
                        ['uint256', 'uint256', 'address[]', 'address', 'uint256'],
                        [amountIn, 0, [tokenIn, tokenOut], this.bot.wallets[network].address, Date.now() + 300]
                    )
                );
            }
        }
        
        return swapData;
    }
    
    async executeLiquidation(opportunity) {
        const { network, protocol, user, collateral, debt, amount } = opportunity;
        
        const result = await this.bot.liquidate(network, {
            collateral,
            debt,
            user,
            amount,
            receiveAToken: false
        });
        
        return {
            success: result.status === 1,
            opportunity,
            txHash: result.transactionHash,
            gasUsed: result.gasUsed,
            profit: await this.calculateLiquidationProfit(result, opportunity),
            timestamp: Date.now()
        };
    }
    
    async executeCrossChain(opportunity) {
        const { token, buyNetwork, sellNetwork, buyPrice, sellPrice } = opportunity;
        
        const buyTx = await this.executeBuy(buyNetwork, token, buyPrice);
        
        if (!buyTx.success) {
            return buyTx;
        }
        
        await this.waitForBridge(token, buyNetwork, sellNetwork);
        
        const sellTx = await this.executeSell(sellNetwork, token, sellPrice);
        
        return {
            success: sellTx.success,
            opportunity,
            buyTx: buyTx.txHash,
            sellTx: sellTx.txHash,
            profit: sellTx.success ? sellPrice.sub(buyPrice) : ethers.BigNumber.from(0),
            timestamp: Date.now()
        };
    }
    
    async executeBuy(network, token, price) {
        const wallet = this.bot.wallets[network];
        const router = this.bot.config.dex[network].uniswapV2.router;
        
        const contract = new ethers.Contract(router, this.bot.config.abis.uniswapV2.router, wallet);
        
        const tx = await contract.swapExactETHForTokens(
            0,
            [this.bot.config.tokens[network].WETH, token],
            wallet.address,
            Date.now() + 300,
            { value: ethers.utils.parseEther('1') }
        );
        
        const receipt = await tx.wait();
        
        return {
            success: receipt.status === 1,
            txHash: receipt.transactionHash
        };
    }
    
    async executeSell(network, token, price) {
        const wallet = this.bot.wallets[network];
        const router = this.bot.config.dex[network].uniswapV2.router;
        
        const tokenContract = new ethers.Contract(token, ['function balanceOf(address) view returns (uint256)', 'function approve(address, uint256)'], wallet);
        const balance = await tokenContract.balanceOf(wallet.address);
        
        await tokenContract.approve(router, balance);
        
        const contract = new ethers.Contract(router, this.bot.config.abis.uniswapV2.router, wallet);
        
        const tx = await contract.swapExactTokensForETH(
            balance,
            0,
            [token, this.bot.config.tokens[network].WETH],
            wallet.address,
            Date.now() + 300
        );
        
        const receipt = await tx.wait();
        
        return {
            success: receipt.status === 1,
            txHash: receipt.transactionHash
        };
    }
    
    async waitForBridge(token, fromNetwork, toNetwork) {
        await new Promise(resolve => setTimeout(resolve, 60000));
    }
    
    async executeBackrun(opportunity) {
        const { network, targetTx, impact } = opportunity;
        
        if (this.flashbots && network === 'ethereum') {
            return await this.executeFlashbotsBundle(opportunity);
        }
        
        const wallet = this.bot.wallets[network];
        const provider = this.bot.providers[network];
        
        const targetReceipt = await provider.waitForTransaction(targetTx, 1);
        
        if (!targetReceipt) {
            throw new Error('Target transaction not found');
        }
        
        const backrunTx = await this.createBackrunTransaction(network, targetReceipt, impact);
        const result = await wallet.sendTransaction(backrunTx);
        const receipt = await result.wait();
        
        return {
            success: receipt.status === 1,
            opportunity,
            txHash: receipt.transactionHash,
            gasUsed: receipt.gasUsed,
            profit: impact.expectedProfit,
            timestamp: Date.now()
        };
    }
    
    async executeFlashbotsBundle(opportunity) {
        const { targetTx, impact } = opportunity;
        
        const blockNumber = await this.bot.providers.ethereum.getBlockNumber();
        const targetBlock = blockNumber + 1;
        
        const backrunTx = await this.createBackrunTransaction('ethereum', null, impact);
        
        const bundle = [
            { signedTransaction: targetTx },
            { signedTransaction: await this.bot.wallets.ethereum.signTransaction(backrunTx) }
        ];
        
        const bundleResponse = await this.flashbots.sendBundle(bundle, targetBlock);
        
        if ('error' in bundleResponse) {
            throw new Error(bundleResponse.error.message);
        }
        
        const resolution = await bundleResponse.wait();
        
        return {
            success: resolution === 0,
            opportunity,
            bundleHash: bundleResponse.bundleHash,
            profit: impact.expectedProfit,
            timestamp: Date.now()
        };
    }
    
    async createBackrunTransaction(network, targetReceipt, impact) {
        return {
            to: this.bot.contracts[network].crossDexArbitrage.address,
            data: '0x',
            value: ethers.utils.parseEther('0.1'),
            gasLimit: ethers.BigNumber.from('500000'),
            maxFeePerGas: ethers.utils.parseUnits('100', 'gwei'),
            maxPriorityFeePerGas: ethers.utils.parseUnits('10', 'gwei')
        };
    }
    
    async executeTriangular(opportunity) {
        const { network, tokenA, tokenB, tokenC, amountIn } = opportunity;
        
        const contract = this.bot.contracts[network].crossDexArbitrage;
        const gasConfig = await this.getOptimalGas(network, opportunity);
        
        const tx = await contract.executeTriangularArbitrage({
            tokenA,
            tokenB,
            tokenC,
            amountIn,
            routers: opportunity.routers,
            fees: opportunity.fees,
            minProfit: opportunity.expectedProfit.mul(90).div(100)
        }, gasConfig);
        
        const receipt = await tx.wait();
        
        return {
            success: receipt.status === 1,
            opportunity,
            txHash: receipt.transactionHash,
            gasUsed: receipt.gasUsed,
            profit: await this.calculateProfit(receipt, opportunity),
            timestamp: Date.now()
        };
    }
    
    async getOptimalGas(network, opportunity) {
        const tracker = this.gasTracker.get(network);
        
        if (Date.now() - tracker.lastUpdate > 10000) {
            await this.updateGasPrice(network);
        }
        
        const multiplier = this.getGasMultiplier(opportunity);
        
        return {
            gasLimit: ethers.BigNumber.from('800000'),
            maxFeePerGas: tracker.base.mul(multiplier).div(100),
            maxPriorityFeePerGas: tracker.priority.mul(multiplier).div(100)
        };
    }
    
    getGasMultiplier(opportunity) {
        const profit = opportunity.expectedProfit || ethers.BigNumber.from(0);
        
        if (profit.gt(ethers.utils.parseEther('1'))) {
            return 150;
        } else if (profit.gt(ethers.utils.parseEther('0.5'))) {
            return 130;
        } else if (profit.gt(ethers.utils.parseEther('0.1'))) {
            return 115;
        }
        
        return 105;
    }
    
    async calculateProfit(receipt, opportunity) {
        const gasUsed = receipt.gasUsed;
        const gasPrice = receipt.effectiveGasPrice || receipt.gasPrice;
        const gasCost = gasUsed.mul(gasPrice);
        
        return opportunity.expectedProfit.sub(gasCost);
    }
    
    async calculateLiquidationProfit(receipt, opportunity) {
        const bonus = opportunity.bonus || ethers.BigNumber.from('500');
        const profit = opportunity.amount.mul(bonus).div(10000);
        
        const gasUsed = receipt.gasUsed;
        const gasPrice = receipt.effectiveGasPrice || receipt.gasPrice;
        const gasCost = gasUsed.mul(gasPrice);
        
        return profit.sub(gasCost);
    }
    
    async processQueue() {
        const concurrent = this.executing.size;
        
        if (concurrent >= this.maxConcurrent) {
            return;
        }
        
        const available = this.maxConcurrent - concurrent;
        const opportunities = this.queue.splice(0, available);
        
        for (const opportunity of opportunities) {
            this.execute(opportunity).catch(error => {
                this.bot.emit('error', {
                    type: 'execution_error',
                    opportunity,
                    error: error.message
                });
            });
        }
    }
    
    queueExecution(opportunity) {
        this.queue.push(opportunity);
    }
    
    clearQueue() {
        this.queue = [];
    }
    
    getQueueSize() {
        return this.queue.length;
    }
    
    getExecutingCount() {
        return this.executing.size;
    }
}

module.exports = Executor;