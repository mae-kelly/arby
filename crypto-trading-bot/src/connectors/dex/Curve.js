const { ethers } = require('ethers');

class Curve {
    constructor(config, provider) {
        this.provider = provider;
        this.config = config;
        this.registryAddress = '0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5';
        this.factoryAddress = '0xB9fC157394Af804a3578134A6585C0dc9cc990d4';
        this.addressProviderAddress = '0x0000000022D53366457F9d5E68Ec105046FC4383';
        
        this.registryContract = new ethers.Contract(
            this.registryAddress,
            config.registryABI,
            provider
        );
        
        this.pools = new Map();
        this.poolsInfo = new Map();
        this.coins = new Map();
    }

    async initialize() {
        await this.loadPools();
    }

    async loadPools() {
        try {
            const poolCount = await this.registryContract.pool_count();
            
            for (let i = 0; i < poolCount; i++) {
                try {
                    const poolAddress = await this.registryContract.pool_list(i);
                    const poolInfo = await this.getPoolInfo(poolAddress);
                    
                    if (poolInfo) {
                        this.pools.set(poolAddress.toLowerCase(), poolInfo);
                        
                        for (let j = 0; j < poolInfo.coins.length; j++) {
                            const coinAddress = poolInfo.coins[j];
                            if (coinAddress !== ethers.constants.AddressZero) {
                                if (!this.coins.has(coinAddress.toLowerCase())) {
                                    this.coins.set(coinAddress.toLowerCase(), []);
                                }
                                this.coins.get(coinAddress.toLowerCase()).push({
                                    pool: poolAddress,
                                    index: j
                                });
                            }
                        }
                    }
                } catch (error) {
                    continue;
                }
            }
        } catch (error) {
            throw new Error(`Failed to load pools: ${error.message}`);
        }
    }

    async getPoolInfo(poolAddress) {
        try {
            const coins = await this.registryContract.get_coins(poolAddress);
            const balances = await this.registryContract.get_balances(poolAddress);
            const amplificationParameter = await this.registryContract.get_A(poolAddress);
            const fees = await this.registryContract.get_fees(poolAddress);
            
            const poolContract = new ethers.Contract(
                poolAddress,
                this.config.poolABI,
                this.provider
            );

            return {
                address: poolAddress,
                contract: poolContract,
                coins: coins.filter(coin => coin !== ethers.constants.AddressZero),
                balances: balances.filter(balance => !balance.eq(0)),
                A: amplificationParameter,
                fee: fees[0],
                adminFee: fees[1],
                n_coins: coins.filter(coin => coin !== ethers.constants.AddressZero).length
            };
        } catch (error) {
            return null;
        }
    }

    async getPool(tokenA, tokenB) {
        const poolsWithTokenA = this.coins.get(tokenA.toLowerCase()) || [];
        const poolsWithTokenB = this.coins.get(tokenB.toLowerCase()) || [];
        
        for (const poolA of poolsWithTokenA) {
            for (const poolB of poolsWithTokenB) {
                if (poolA.pool === poolB.pool) {
                    return this.pools.get(poolA.pool.toLowerCase());
                }
            }
        }
        
        return null;
    }

    async getPrice(tokenA, tokenB) {
        try {
            const pool = await this.getPool(tokenA, tokenB);
            if (!pool) return 0;

            const tokenAInfo = this.coins.get(tokenA.toLowerCase()).find(coin => coin.pool === pool.address);
            const tokenBInfo = this.coins.get(tokenB.toLowerCase()).find(coin => coin.pool === pool.address);
            
            if (!tokenAInfo || !tokenBInfo) return 0;

            const amount = ethers.utils.parseEther('1');
            const dy = await pool.contract.get_dy(tokenAInfo.index, tokenBInfo.index, amount);
            
            return parseFloat(ethers.utils.formatEther(dy));
        } catch (error) {
            return 0;
        }
    }

    async getDy(poolAddress, i, j, dx) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) return ethers.BigNumber.from(0);

            return await pool.contract.get_dy(i, j, dx);
        } catch (error) {
            return ethers.BigNumber.from(0);
        }
    }

    async exchange(poolAddress, i, j, dx, minDy) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) throw new Error('Pool not found');

            const tx = await pool.contract.exchange(i, j, dx, minDy);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to exchange: ${error.message}`);
        }
    }

    async exchangeUnderlying(poolAddress, i, j, dx, minDy) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) throw new Error('Pool not found');

            const tx = await pool.contract.exchange_underlying(i, j, dx, minDy);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to exchange underlying: ${error.message}`);
        }
    }

    async addLiquidity(poolAddress, amounts, minMintAmount) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) throw new Error('Pool not found');

            const tx = await pool.contract.add_liquidity(amounts, minMintAmount);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to add liquidity: ${error.message}`);
        }
    }

    async removeLiquidity(poolAddress, tokenAmount, minAmounts) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) throw new Error('Pool not found');

            const tx = await pool.contract.remove_liquidity(tokenAmount, minAmounts);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to remove liquidity: ${error.message}`);
        }
    }

    async removeLiquidityOneCoin(poolAddress, tokenAmount, i, minAmount) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) throw new Error('Pool not found');

            const tx = await pool.contract.remove_liquidity_one_coin(tokenAmount, i, minAmount);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to remove liquidity one coin: ${error.message}`);
        }
    }

    async calcTokenAmount(poolAddress, amounts, isDeposit = true) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) return ethers.BigNumber.from(0);

            return await pool.contract.calc_token_amount(amounts, isDeposit);
        } catch (error) {
            return ethers.BigNumber.from(0);
        }
    }

    async calcWithdrawOneCoin(poolAddress, tokenAmount, i) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) return ethers.BigNumber.from(0);

            return await pool.contract.calc_withdraw_one_coin(tokenAmount, i);
        } catch (error) {
            return ethers.BigNumber.from(0);
        }
    }

    async getLiquidity(tokenA, tokenB) {
        try {
            const pool = await this.getPool(tokenA, tokenB);
            if (!pool) return 0;

            const totalSupply = await pool.contract.totalSupply();
            return parseFloat(ethers.utils.formatEther(totalSupply));
        } catch (error) {
            return 0;
        }
    }

    async simulateSwap(tokenIn, tokenOut, amountIn) {
        try {
            const pool = await this.getPool(tokenIn, tokenOut);
            if (!pool) return 0;

            const tokenInInfo = this.coins.get(tokenIn.toLowerCase()).find(coin => coin.pool === pool.address);
            const tokenOutInfo = this.coins.get(tokenOut.toLowerCase()).find(coin => coin.pool === pool.address);
            
            if (!tokenInInfo || !tokenOutInfo) return 0;

            const dy = await this.getDy(pool.address, tokenInInfo.index, tokenOutInfo.index, amountIn);
            const price = parseFloat(ethers.utils.formatEther(
                dy.mul(ethers.utils.parseEther('1')).div(amountIn)
            ));
            
            return price;
        } catch (error) {
            return 0;
        }
    }

    async estimateGasCost(poolAddress, method = 'exchange', params = []) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) return ethers.utils.parseUnits('150000', 'gwei');

            const gasEstimate = await pool.contract.estimateGas[method](...params);
            const gasPrice = await this.provider.getGasPrice();
            return gasEstimate.mul(gasPrice);
        } catch (error) {
            return ethers.utils.parseUnits('150000', 'gwei');
        }
    }

    getFee(poolAddress = null) {
        if (poolAddress) {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (pool) {
                return parseFloat(ethers.utils.formatUnits(pool.fee, 8));
            }
        }
        return 0.0004;
    }

    getRouterAddress() {
        return this.registryAddress;
    }

    async parseTransaction(tx) {
        try {
            for (const [poolAddress, pool] of this.pools.entries()) {
                if (tx.to && tx.to.toLowerCase() === poolAddress) {
                    const iface = new ethers.utils.Interface(this.config.poolABI);
                    const decoded = iface.parseTransaction({ data: tx.data });

                    if (decoded.name === 'exchange') {
                        const tokenInIndex = decoded.args.i;
                        const tokenOutIndex = decoded.args.j;
                        const amountIn = decoded.args.dx;
                        
                        return {
                            tokenIn: pool.coins[tokenInIndex],
                            tokenOut: pool.coins[tokenOutIndex],
                            amountIn: amountIn,
                            price: await this.getPrice(pool.coins[tokenInIndex], pool.coins[tokenOutIndex])
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
        return new ethers.utils.Interface(this.config.poolABI);
    }

    async executeTriangularArbitrage(path, pairs, startAmount, minProfit) {
        try {
            let currentAmount = startAmount;
            const transactions = [];

            for (let i = 0; i < path.length - 1; i++) {
                const tokenIn = path[i];
                const tokenOut = path[i + 1];
                const pool = await this.getPool(tokenIn, tokenOut);
                
                if (!pool) throw new Error(`Pool not found for ${tokenIn} -> ${tokenOut}`);

                const tokenInInfo = this.coins.get(tokenIn.toLowerCase()).find(coin => coin.pool === pool.address);
                const tokenOutInfo = this.coins.get(tokenOut.toLowerCase()).find(coin => coin.pool === pool.address);
                
                const minDy = i === path.length - 2 ? startAmount.add(minProfit) : 0;
                
                const tx = await this.exchange(
                    pool.address,
                    tokenInInfo.index,
                    tokenOutInfo.index,
                    currentAmount,
                    minDy
                );
                
                transactions.push(tx);
                currentAmount = await this.getDy(pool.address, tokenInInfo.index, tokenOutInfo.index, currentAmount);
            }

            return transactions[transactions.length - 1].transactionHash;
        } catch (error) {
            throw new Error(`Failed to execute triangular arbitrage: ${error.message}`);
        }
    }

    async buyToken(tokenAddress, amount, dexName, maxPrice = null) {
        const wethAddress = this.config.wethAddress;
        const pool = await this.getPool(wethAddress, tokenAddress);
        
        if (!pool) throw new Error('Pool not found');

        const wethInfo = this.coins.get(wethAddress.toLowerCase()).find(coin => coin.pool === pool.address);
        const tokenInfo = this.coins.get(tokenAddress.toLowerCase()).find(coin => coin.pool === pool.address);
        
        const dy = await this.getDy(pool.address, wethInfo.index, tokenInfo.index, amount);
        const minDy = dy.mul(95).div(100);

        return await this.exchange(pool.address, wethInfo.index, tokenInfo.index, amount, minDy);
    }

    async sellToken(tokenAddress, amount, dexName, minPrice = null) {
        const wethAddress = this.config.wethAddress;
        const pool = await this.getPool(tokenAddress, wethAddress);
        
        if (!pool) throw new Error('Pool not found');

        const tokenInfo = this.coins.get(tokenAddress.toLowerCase()).find(coin => coin.pool === pool.address);
        const wethInfo = this.coins.get(wethAddress.toLowerCase()).find(coin => coin.pool === pool.address);
        
        const dy = await this.getDy(pool.address, tokenInfo.index, wethInfo.index, amount);
        const minDy = dy.mul(95).div(100);

        return await this.exchange(pool.address, tokenInfo.index, wethInfo.index, amount, minDy);
    }

    async waitForTransaction(txHash) {
        return await this.provider.waitForTransaction(txHash);
    }

    formatUnits(value, decimals = 18) {
        return ethers.utils.formatUnits(value, decimals);
    }

    async getPoolBalances(poolAddress) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) return [];

            const balances = [];
            for (let i = 0; i < pool.n_coins; i++) {
                const balance = await pool.contract.balances(i);
                balances.push(balance);
            }
            return balances;
        } catch (error) {
            return [];
        }
    }

    async getVirtualPrice(poolAddress) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) return ethers.BigNumber.from(0);

            return await pool.contract.get_virtual_price();
        } catch (error) {
            return ethers.BigNumber.from(0);
        }
    }

    async getAmplificationParameter(poolAddress) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) return ethers.BigNumber.from(0);

            return await pool.contract.A();
        } catch (error) {
            return ethers.BigNumber.from(0);
        }
    }

    async getAdminBalances(poolAddress) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) return [];

            const adminBalances = [];
            for (let i = 0; i < pool.n_coins; i++) {
                try {
                    const adminBalance = await pool.contract.admin_balances(i);
                    adminBalances.push(adminBalance);
                } catch {
                    adminBalances.push(ethers.BigNumber.from(0));
                }
            }
            return adminBalances;
        } catch (error) {
            return [];
        }
    }

    async calculateSlippage(poolAddress, i, j, dx) {
        try {
            const dy = await this.getDy(poolAddress, i, j, dx);
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) return 0;

            const currentPrice = await this.getPrice(pool.coins[i], pool.coins[j]);
            const executionPrice = parseFloat(ethers.utils.formatEther(
                dy.mul(ethers.utils.parseEther('1')).div(dx)
            ));

            return Math.abs((executionPrice - currentPrice) / currentPrice);
        } catch (error) {
            return 0;
        }
    }

    async findBestPool(tokenA, tokenB) {
        const possiblePools = [];
        const poolsWithA = this.coins.get(tokenA.toLowerCase()) || [];
        const poolsWithB = this.coins.get(tokenB.toLowerCase()) || [];

        for (const poolA of poolsWithA) {
            for (const poolB of poolsWithB) {
                if (poolA.pool === poolB.pool) {
                    const pool = this.pools.get(poolA.pool.toLowerCase());
                    if (pool) {
                        const liquidity = await this.getLiquidity(tokenA, tokenB);
                        possiblePools.push({
                            ...pool,
                            liquidity,
                            tokenAIndex: poolA.index,
                            tokenBIndex: poolB.index
                        });
                    }
                }
            }
        }

        return possiblePools.sort((a, b) => b.liquidity - a.liquidity)[0] || null;
    }

    async getMetaPoolInfo(poolAddress) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) return null;

            const basePool = await pool.contract.base_pool();
            const baseVirtualPrice = await this.getVirtualPrice(basePool);
            const metaVirtualPrice = await this.getVirtualPrice(poolAddress);

            return {
                basePool,
                baseVirtualPrice,
                metaVirtualPrice,
                isMetaPool: basePool !== ethers.constants.AddressZero
            };
        } catch (error) {
            return null;
        }
    }

    async getPoolStats(poolAddress) {
        try {
            const pool = this.pools.get(poolAddress.toLowerCase());
            if (!pool) return null;

            const [balances, virtualPrice, totalSupply, A] = await Promise.all([
                this.getPoolBalances(poolAddress),
                this.getVirtualPrice(poolAddress),
                pool.contract.totalSupply(),
                this.getAmplificationParameter(poolAddress)
            ]);

            return {
                address: poolAddress,
                balances,
                virtualPrice,
                totalSupply,
                amplificationParameter: A,
                fee: pool.fee,
                adminFee: pool.adminFee,
                nCoins: pool.n_coins,
                coins: pool.coins
            };
        } catch (error) {
            return null;
        }
    }

    getAllPools() {
        return Array.from(this.pools.values());
    }

    getPoolsForToken(tokenAddress) {
        return this.coins.get(tokenAddress.toLowerCase()) || [];
    }

    async getOptimalSwapRoute(tokenIn, tokenOut, amountIn) {
        const directPool = await this.getPool(tokenIn, tokenOut);
        
        if (directPool) {
            const tokenInInfo = this.coins.get(tokenIn.toLowerCase()).find(coin => coin.pool === directPool.address);
            const tokenOutInfo = this.coins.get(tokenOut.toLowerCase()).find(coin => coin.pool === directPool.address);
            
            const dy = await this.getDy(directPool.address, tokenInInfo.index, tokenOutInfo.index, amountIn);
            
            return {
                route: [tokenIn, tokenOut],
                pools: [directPool.address],
                expectedOutput: dy,
                priceImpact: await this.calculateSlippage(directPool.address, tokenInInfo.index, tokenOutInfo.index, amountIn)
            };
        }

        return null;
    }
}

module.exports = Curve;