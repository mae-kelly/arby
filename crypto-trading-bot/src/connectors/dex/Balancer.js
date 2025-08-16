const { ethers } = require('ethers');

class Balancer {
    constructor(config, provider) {
        this.provider = provider;
        this.config = config;
        this.vaultAddress = '0xBA12222222228d8Ba445958a75a0704d566BF2C8';
        this.routerAddress = '0xBA12222222228d8Ba445958a75a0704d566BF2C8';
        this.queriesAddress = '0xE39B5e3B6D74016b2F6A9673D7d7493B6DF549d5';
        
        this.vaultContract = new ethers.Contract(
            this.vaultAddress,
            config.vaultABI,
            provider.getSigner()
        );
        
        this.queriesContract = new ethers.Contract(
            this.queriesAddress,
            config.queriesABI,
            provider
        );
        
        this.pools = new Map();
        this.poolTokens = new Map();
    }

    async initialize() {
        await this.loadPools();
    }

    async loadPools() {
        const poolIds = this.config.poolIds || [];
        
        for (const poolId of poolIds) {
            try {
                const pool = await this.getPool(poolId);
                if (pool) {
                    this.pools.set(poolId, pool);
                }
            } catch (error) {
                continue;
            }
        }
    }

    async getPool(poolId) {
        try {
            const [tokens, balances] = await this.vaultContract.getPoolTokens(poolId);
            
            return {
                poolId,
                tokens: tokens.filter(token => token !== ethers.constants.AddressZero),
                balances: balances.filter(balance => !balance.eq(0)),
                lastChangeBlock: 0
            };
        } catch (error) {
            return null;
        }
    }

    async getPrice(tokenA, tokenB) {
        try {
            const pool = await this.findPoolWithTokens(tokenA, tokenB);
            if (!pool) return 0;

            const amount = ethers.utils.parseEther('1');
            const swapRequest = this.buildSwapRequest(tokenA, tokenB, amount, pool.poolId);
            
            const result = await this.queriesContract.querySwap(swapRequest);
            return parseFloat(ethers.utils.formatEther(result));
        } catch (error) {
            return 0;
        }
    }

    async findPoolWithTokens(tokenA, tokenB) {
        for (const [poolId, pool] of this.pools.entries()) {
            const hasTokenA = pool.tokens.some(token => token.toLowerCase() === tokenA.toLowerCase());
            const hasTokenB = pool.tokens.some(token => token.toLowerCase() === tokenB.toLowerCase());
            
            if (hasTokenA && hasTokenB) {
                return pool;
            }
        }
        return null;
    }

    buildSwapRequest(tokenIn, tokenOut, amount, poolId) {
        const singleSwap = {
            poolId: poolId,
            kind: 0,
            assetIn: tokenIn,
            assetOut: tokenOut,
            amount: amount,
            userData: '0x'
        };

        const funds = {
            sender: ethers.constants.AddressZero,
            fromInternalBalance: false,
            recipient: ethers.constants.AddressZero,
            toInternalBalance: false
        };

        return { singleSwap, funds, limit: 0, deadline: 0 };
    }

    async swap(tokenIn, tokenOut, amount, minAmountOut, recipient = null) {
        try {
            const pool = await this.findPoolWithTokens(tokenIn, tokenOut);
            if (!pool) throw new Error('Pool not found');

            const singleSwap = {
                poolId: pool.poolId,
                kind: 0,
                assetIn: tokenIn,
                assetOut: tokenOut,
                amount: amount,
                userData: '0x'
            };

            const funds = {
                sender: await this.provider.getSigner().getAddress(),
                fromInternalBalance: false,
                recipient: recipient || await this.provider.getSigner().getAddress(),
                toInternalBalance: false
            };

            const deadline = Math.floor(Date.now() / 1000) + 300;

            const tx = await this.vaultContract.swap(singleSwap, funds, minAmountOut, deadline);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to swap: ${error.message}`);
        }
    }

    async batchSwap(swaps, assets, limits, deadline = null) {
        try {
            const funds = {
                sender: await this.provider.getSigner().getAddress(),
                fromInternalBalance: false,
                recipient: await this.provider.getSigner().getAddress(),
                toInternalBalance: false
            };

            const swapDeadline = deadline || Math.floor(Date.now() / 1000) + 300;

            const tx = await this.vaultContract.batchSwap(0, swaps, assets, funds, limits, swapDeadline);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to batch swap: ${error.message}`);
        }
    }

    async joinPool(poolId, sender, recipient, request) {
        try {
            const tx = await this.vaultContract.joinPool(poolId, sender, recipient, request);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to join pool: ${error.message}`);
        }
    }

    async exitPool(poolId, sender, recipient, request) {
        try {
            const tx = await this.vaultContract.exitPool(poolId, sender, recipient, request);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to exit pool: ${error.message}`);
        }
    }

    async flashLoan(tokens, amounts, userData) {
        try {
            const tx = await this.vaultContract.flashLoan(
                await this.provider.getSigner().getAddress(),
                tokens,
                amounts,
                userData
            );
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to execute flash loan: ${error.message}`);
        }
    }

    async getLiquidity(tokenA, tokenB) {
        try {
            const pool = await this.findPoolWithTokens(tokenA, tokenB);
            if (!pool) return 0;

            const tokenAIndex = pool.tokens.findIndex(token => token.toLowerCase() === tokenA.toLowerCase());
            const tokenBIndex = pool.tokens.findIndex(token => token.toLowerCase() === tokenB.toLowerCase());

            if (tokenAIndex === -1 || tokenBIndex === -1) return 0;

            const balanceA = parseFloat(ethers.utils.formatEther(pool.balances[tokenAIndex]));
            const balanceB = parseFloat(ethers.utils.formatEther(pool.balances[tokenBIndex]));

            return Math.min(balanceA, balanceB);
        } catch (error) {
            return 0;
        }
    }

    async simulateSwap(tokenIn, tokenOut, amountIn) {
        try {
            const pool = await this.findPoolWithTokens(tokenIn, tokenOut);
            if (!pool) return 0;

            const swapRequest = this.buildSwapRequest(tokenIn, tokenOut, amountIn, pool.poolId);
            const result = await this.queriesContract.querySwap(swapRequest);
            
            const price = parseFloat(ethers.utils.formatEther(
                result.mul(ethers.utils.parseEther('1')).div(amountIn)
            ));
            
            return price;
        } catch (error) {
            return 0;
        }
    }

    async estimateGasCost(method = 'swap', params = []) {
        try {
            const gasEstimate = await this.vaultContract.estimateGas[method](...params);
            const gasPrice = await this.provider.getGasPrice();
            return gasEstimate.mul(gasPrice);
        } catch (error) {
            return ethers.utils.parseUnits('200000', 'gwei');
        }
    }

    getFee(poolId = null) {
        return 0.001;
    }

    getRouterAddress() {
        return this.vaultAddress;
    }

    async parseTransaction(tx) {
        try {
            if (tx.to && tx.to.toLowerCase() === this.vaultAddress.toLowerCase()) {
                const iface = new ethers.utils.Interface(this.config.vaultABI);
                const decoded = iface.parseTransaction({ data: tx.data });

                if (decoded.name === 'swap') {
                    const singleSwap = decoded.args.singleSwap;
                    return {
                        tokenIn: singleSwap.assetIn,
                        tokenOut: singleSwap.assetOut,
                        amountIn: singleSwap.amount,
                        price: await this.getPrice(singleSwap.assetIn, singleSwap.assetOut)
                    };
                }

                if (decoded.name === 'batchSwap') {
                    const swaps = decoded.args.swaps;
                    const assets = decoded.args.assets;
                    
                    if (swaps.length > 0) {
                        const firstSwap = swaps[0];
                        return {
                            tokenIn: assets[firstSwap.assetInIndex],
                            tokenOut: assets[firstSwap.assetOutIndex],
                            amountIn: firstSwap.amount,
                            price: await this.getPrice(assets[firstSwap.assetInIndex], assets[firstSwap.assetOutIndex])
                        };
                    }
                }
            }
            return null;
        } catch (error) {
            return null;
        }
    }

    async getInterface() {
        return new ethers.utils.Interface(this.config.vaultABI);
    }

    async executeTriangularArbitrage(path, pairs, startAmount, minProfit) {
        try {
            const swaps = [];
            const assets = [];
            const assetMap = new Map();
            let assetIndex = 0;

            for (const asset of path) {
                if (!assetMap.has(asset)) {
                    assetMap.set(asset, assetIndex);
                    assets.push(asset);
                    assetIndex++;
                }
            }

            for (let i = 0; i < path.length - 1; i++) {
                const tokenIn = path[i];
                const tokenOut = path[i + 1];
                const pool = await this.findPoolWithTokens(tokenIn, tokenOut);
                
                if (!pool) throw new Error(`Pool not found for ${tokenIn} -> ${tokenOut}`);

                swaps.push({
                    poolId: pool.poolId,
                    assetInIndex: assetMap.get(tokenIn),
                    assetOutIndex: assetMap.get(tokenOut),
                    amount: i === 0 ? startAmount : 0,
                    userData: '0x'
                });
            }

            const limits = new Array(assets.length).fill(0);
            limits[0] = startAmount;
            limits[limits.length - 1] = startAmount.add(minProfit).mul(-1);

            const tx = await this.batchSwap(swaps, assets, limits);
            return tx.transactionHash;
        } catch (error) {
            throw new Error(`Failed to execute triangular arbitrage: ${error.message}`);
        }
    }

    async buyToken(tokenAddress, amount, dexName, maxPrice = null) {
        const wethAddress = this.config.wethAddress;
        const amountOut = await this.simulateSwap(wethAddress, tokenAddress, amount);
        const minAmountOut = ethers.utils.parseEther((amountOut * 0.95).toString());

        return await this.swap(wethAddress, tokenAddress, amount, minAmountOut);
    }

    async sellToken(tokenAddress, amount, dexName, minPrice = null) {
        const wethAddress = this.config.wethAddress;
        const amountOut = await this.simulateSwap(tokenAddress, wethAddress, amount);
        const minAmountOut = ethers.utils.parseEther((amountOut * 0.95).toString());

        return await this.swap(tokenAddress, wethAddress, amount, minAmountOut);
    }

    async waitForTransaction(txHash) {
        return await this.provider.waitForTransaction(txHash);
    }

    formatUnits(value, decimals = 18) {
        return ethers.utils.formatUnits(value, decimals);
    }

    async getPoolBalance(poolId, tokenAddress) {
        try {
            const pool = this.pools.get(poolId);
            if (!pool) return ethers.BigNumber.from(0);

            const tokenIndex = pool.tokens.findIndex(token => token.toLowerCase() === tokenAddress.toLowerCase());
            if (tokenIndex === -1) return ethers.BigNumber.from(0);

            return pool.balances[tokenIndex];
        } catch (error) {
            return ethers.BigNumber.from(0);
        }
    }

    async getPoolWeights(poolId) {
        try {
            const poolContract = new ethers.Contract(
                poolId.slice(0, 42),
                this.config.poolABI,
                this.provider
            );

            const weights = await poolContract.getNormalizedWeights();
            return weights;
        } catch (error) {
            return [];
        }
    }

    async getSwapFee(poolId) {
        try {
            const poolContract = new ethers.Contract(
                poolId.slice(0, 42),
                this.config.poolABI,
                this.provider
            );

            const swapFee = await poolContract.getSwapFeePercentage();
            return swapFee;
        } catch (error) {
            return ethers.utils.parseEther('0.001');
        }
    }

    async calculateSpotPrice(poolId, tokenIn, tokenOut) {
        try {
            const pool = this.pools.get(poolId);
            if (!pool) return ethers.BigNumber.from(0);

            const tokenInIndex = pool.tokens.findIndex(token => token.toLowerCase() === tokenIn.toLowerCase());
            const tokenOutIndex = pool.tokens.findIndex(token => token.toLowerCase() === tokenOut.toLowerCase());

            if (tokenInIndex === -1 || tokenOutIndex === -1) return ethers.BigNumber.from(0);

            const balanceIn = pool.balances[tokenInIndex];
            const balanceOut = pool.balances[tokenOutIndex];
            const weights = await this.getPoolWeights(poolId);

            if (weights.length === 0) {
                return balanceOut.mul(ethers.utils.parseEther('1')).div(balanceIn);
            }

            const weightIn = weights[tokenInIndex];
            const weightOut = weights[tokenOutIndex];

            return balanceOut.mul(weightIn).div(balanceIn.mul(weightOut));
        } catch (error) {
            return ethers.BigNumber.from(0);
        }
    }

    async getPoolInfo(poolId) {
        try {
            const pool = this.pools.get(poolId);
            if (!pool) return null;

            const weights = await this.getPoolWeights(poolId);
            const swapFee = await this.getSwapFee(poolId);

            return {
                poolId,
                tokens: pool.tokens,
                balances: pool.balances,
                weights,
                swapFee,
                poolType: this.getPoolType(poolId)
            };
        } catch (error) {
            return null;
        }
    }

    getPoolType(poolId) {
        const poolTypeIndex = parseInt(poolId.slice(-4), 16);
        switch (poolTypeIndex) {
            case 0: return 'WEIGHTED';
            case 1: return 'STABLE';
            case 2: return 'META_STABLE';
            case 3: return 'PHANTOM_STABLE';
            case 4: return 'COMPOSABLE_STABLE';
            default: return 'UNKNOWN';
        }
    }

    async queryBatchSwap(swaps, assets) {
        try {
            const funds = {
                sender: ethers.constants.AddressZero,
                fromInternalBalance: false,
                recipient: ethers.constants.AddressZero,
                toInternalBalance: false
            };

            return await this.queriesContract.queryBatchSwap(0, swaps, assets, funds);
        } catch (error) {
            return [];
        }
    }

    async findOptimalSwapPath(tokenIn, tokenOut, amountIn) {
        const directPool = await this.findPoolWithTokens(tokenIn, tokenOut);
        
        if (directPool) {
            const expectedOutput = await this.simulateSwap(tokenIn, tokenOut, amountIn);
            return {
                path: [tokenIn, tokenOut],
                pools: [directPool.poolId],
                expectedOutput: ethers.utils.parseEther(expectedOutput.toString()),
                hops: 1
            };
        }

        return null;
    }

    async getFlashLoanFee() {
        try {
            return await this.vaultContract.getProtocolFeesCollector();
        } catch (error) {
            return ethers.BigNumber.from(0);
        }
    }

    async getAllPoolIds() {
        return Array.from(this.pools.keys());
    }

    async refreshPoolData(poolId) {
        try {
            const [tokens, balances] = await this.vaultContract.getPoolTokens(poolId);
            
            const pool = {
                poolId,
                tokens: tokens.filter(token => token !== ethers.constants.AddressZero),
                balances: balances.filter(balance => !balance.eq(0)),
                lastChangeBlock: await this.provider.getBlockNumber()
            };

            this.pools.set(poolId, pool);
            return pool;
        } catch (error) {
            return null;
        }
    }

    async getPoolLiquidity(poolId) {
        try {
            const pool = this.pools.get(poolId);
            if (!pool) return ethers.BigNumber.from(0);

            return pool.balances.reduce((sum, balance) => sum.add(balance), ethers.BigNumber.from(0));
        } catch (error) {
            return ethers.BigNumber.from(0);
        }
    }
}

module.exports = Balancer;