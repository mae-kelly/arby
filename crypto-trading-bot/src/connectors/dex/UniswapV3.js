const { ethers } = require('ethers');
const { Pool, Position, nearestUsableTick, TickMath, SqrtPriceMath } = require('@uniswap/v3-sdk');
const { Token, CurrencyAmount, TradeType, Percent } = require('@uniswap/sdk-core');
const { AlphaRouter, SwapType } = require('@uniswap/smart-order-router');

class UniswapV3 {
    constructor(config, provider) {
        this.provider = provider;
        this.config = config;
        this.factoryAddress = '0x1F98431c8aD98523631AE4a59f267346ea31F984';
        this.routerAddress = '0xE592427A0AEce92De3Edee1F18E0157C05861564';
        this.quoterAddress = '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6';
        this.nonfungiblePositionManagerAddress = '0xC36442b4a4522E871399CD717aBDD847Ab11FE88';
        
        this.factoryContract = new ethers.Contract(
            this.factoryAddress,
            config.factoryABI,
            provider
        );
        
        this.routerContract = new ethers.Contract(
            this.routerAddress,
            config.routerABI,
            provider.getSigner()
        );
        
        this.quoterContract = new ethers.Contract(
            this.quoterAddress,
            config.quoterABI,
            provider
        );
        
        this.alphaRouter = new AlphaRouter({
            chainId: config.chainId,
            provider: provider
        });
        
        this.pools = new Map();
        this.tokens = new Map();
    }

    async initialize() {
        await this.loadCommonTokens();
    }

    async loadCommonTokens() {
        const tokenList = this.config.tokens || [];
        for (const tokenInfo of tokenList) {
            const token = new Token(
                this.config.chainId,
                tokenInfo.address,
                tokenInfo.decimals,
                tokenInfo.symbol,
                tokenInfo.name
            );
            this.tokens.set(tokenInfo.address.toLowerCase(), token);
        }
    }

    async getPool(tokenA, tokenB, fee) {
        const poolKey = `${tokenA.toLowerCase()}-${tokenB.toLowerCase()}-${fee}`;
        
        if (this.pools.has(poolKey)) {
            return this.pools.get(poolKey);
        }

        try {
            const poolAddress = await this.factoryContract.getPool(tokenA, tokenB, fee);
            
            if (poolAddress === ethers.constants.AddressZero) {
                return null;
            }

            const poolContract = new ethers.Contract(
                poolAddress,
                this.config.poolABI,
                this.provider
            );

            const [slot0, liquidity, fee0, fee1] = await Promise.all([
                poolContract.slot0(),
                poolContract.liquidity(),
                poolContract.feeGrowthGlobal0X128(),
                poolContract.feeGrowthGlobal1X128()
            ]);

            const tokenAObj = this.tokens.get(tokenA.toLowerCase());
            const tokenBObj = this.tokens.get(tokenB.toLowerCase());

            if (!tokenAObj || !tokenBObj) {
                throw new Error('Token not found in token list');
            }

            const pool = new Pool(
                tokenAObj,
                tokenBObj,
                fee,
                slot0.sqrtPriceX96.toString(),
                liquidity.toString(),
                slot0.tick
            );

            this.pools.set(poolKey, {
                pool,
                contract: poolContract,
                address: poolAddress
            });

            return this.pools.get(poolKey);
        } catch (error) {
            throw new Error(`Failed to get pool: ${error.message}`);
        }
    }

    async getPrice(tokenIn, tokenOut, fee = 3000) {
        try {
            const poolInfo = await this.getPool(tokenIn, tokenOut, fee);
            if (!poolInfo) {
                throw new Error('Pool not found');
            }

            const tokenAObj = this.tokens.get(tokenIn.toLowerCase());
            const tokenBObj = this.tokens.get(tokenOut.toLowerCase());

            const price = poolInfo.pool.token0Price;
            return tokenAObj.equals(poolInfo.pool.token0) ? 
                parseFloat(price.toSignificant(6)) : 
                parseFloat(price.invert().toSignificant(6));
        } catch (error) {
            throw new Error(`Failed to get price: ${error.message}`);
        }
    }

    async quoteExactInputSingle(tokenIn, tokenOut, amountIn, fee = 3000) {
        try {
            const params = {
                tokenIn,
                tokenOut,
                fee,
                amountIn,
                sqrtPriceLimitX96: 0
            };

            const quote = await this.quoterContract.callStatic.quoteExactInputSingle(params);
            return quote.amountOut;
        } catch (error) {
            throw new Error(`Failed to quote exact input: ${error.message}`);
        }
    }

    async swapExactInputSingle(tokenIn, tokenOut, amountIn, amountOutMinimum, fee = 3000, recipient = null) {
        try {
            const params = {
                tokenIn,
                tokenOut,
                fee,
                recipient: recipient || await this.provider.getSigner().getAddress(),
                deadline: Math.floor(Date.now() / 1000) + 300,
                amountIn,
                amountOutMinimum,
                sqrtPriceLimitX96: 0
            };

            const tx = await this.routerContract.exactInputSingle(params);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to swap exact input single: ${error.message}`);
        }
    }

    async swapExactInput(path, amountIn, amountOutMinimum, recipient = null) {
        try {
            const params = {
                path,
                recipient: recipient || await this.provider.getSigner().getAddress(),
                deadline: Math.floor(Date.now() / 1000) + 300,
                amountIn,
                amountOutMinimum
            };

            const tx = await this.routerContract.exactInput(params);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to swap exact input: ${error.message}`);
        }
    }

    async swapExactOutputSingle(tokenIn, tokenOut, amountOut, amountInMaximum, fee = 3000, recipient = null) {
        try {
            const params = {
                tokenIn,
                tokenOut,
                fee,
                recipient: recipient || await this.provider.getSigner().getAddress(),
                deadline: Math.floor(Date.now() / 1000) + 300,
                amountOut,
                amountInMaximum,
                sqrtPriceLimitX96: 0
            };

            const tx = await this.routerContract.exactOutputSingle(params);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to swap exact output single: ${error.message}`);
        }
    }

    async getLiquidity(tokenA, tokenB, fee = 3000) {
        try {
            const poolInfo = await this.getPool(tokenA, tokenB, fee);
            if (!poolInfo) {
                return 0;
            }

            const liquidity = await poolInfo.contract.liquidity();
            return parseFloat(ethers.utils.formatUnits(liquidity, 18));
        } catch (error) {
            return 0;
        }
    }

    async simulateSwap(tokenIn, tokenOut, amountIn, fee = 3000) {
        try {
            const tokenInObj = this.tokens.get(tokenIn.toLowerCase());
            const tokenOutObj = this.tokens.get(tokenOut.toLowerCase());
            
            if (!tokenInObj || !tokenOutObj) {
                throw new Error('Token not found');
            }

            const route = await this.alphaRouter.route(
                CurrencyAmount.fromRawAmount(tokenInObj, amountIn.toString()),
                tokenOutObj,
                TradeType.EXACT_INPUT,
                {
                    recipient: await this.provider.getSigner().getAddress(),
                    slippageTolerance: new Percent(50, 10000),
                    deadline: Math.floor(Date.now() / 1000) + 300,
                    type: SwapType.UNIVERSAL_ROUTER
                }
            );

            if (!route) {
                throw new Error('No route found');
            }

            return route.quote.quotient.toString();
        } catch (error) {
            const quote = await this.quoteExactInputSingle(tokenIn, tokenOut, amountIn, fee);
            return quote.toString();
        }
    }

    async addLiquidity(tokenA, tokenB, fee, amountA, amountB, tickLower, tickUpper, recipient = null) {
        try {
            const poolInfo = await this.getPool(tokenA, tokenB, fee);
            if (!poolInfo) {
                throw new Error('Pool not found');
            }

            const positionManager = new ethers.Contract(
                this.nonfungiblePositionManagerAddress,
                this.config.positionManagerABI,
                this.provider.getSigner()
            );

            const params = {
                token0: poolInfo.pool.token0.address,
                token1: poolInfo.pool.token1.address,
                fee,
                tickLower: nearestUsableTick(tickLower, poolInfo.pool.tickSpacing),
                tickUpper: nearestUsableTick(tickUpper, poolInfo.pool.tickSpacing),
                amount0Desired: poolInfo.pool.token0.address === tokenA ? amountA : amountB,
                amount1Desired: poolInfo.pool.token0.address === tokenA ? amountB : amountA,
                amount0Min: 0,
                amount1Min: 0,
                recipient: recipient || await this.provider.getSigner().getAddress(),
                deadline: Math.floor(Date.now() / 1000) + 300
            };

            const tx = await positionManager.mint(params);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to add liquidity: ${error.message}`);
        }
    }

    async removeLiquidity(tokenId, liquidity, amount0Min = 0, amount1Min = 0) {
        try {
            const positionManager = new ethers.Contract(
                this.nonfungiblePositionManagerAddress,
                this.config.positionManagerABI,
                this.provider.getSigner()
            );

            const params = {
                tokenId,
                liquidity,
                amount0Min,
                amount1Min,
                deadline: Math.floor(Date.now() / 1000) + 300
            };

            const tx = await positionManager.decreaseLiquidity(params);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to remove liquidity: ${error.message}`);
        }
    }

    async collectFees(tokenId, amount0Max = ethers.constants.MaxUint128, amount1Max = ethers.constants.MaxUint128) {
        try {
            const positionManager = new ethers.Contract(
                this.nonfungiblePositionManagerAddress,
                this.config.positionManagerABI,
                this.provider.getSigner()
            );

            const params = {
                tokenId,
                recipient: await this.provider.getSigner().getAddress(),
                amount0Max,
                amount1Max
            };

            const tx = await positionManager.collect(params);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to collect fees: ${error.message}`);
        }
    }

    async getPosition(tokenId) {
        try {
            const positionManager = new ethers.Contract(
                this.nonfungiblePositionManagerAddress,
                this.config.positionManagerABI,
                this.provider
            );

            const position = await positionManager.positions(tokenId);
            return {
                nonce: position.nonce,
                operator: position.operator,
                token0: position.token0,
                token1: position.token1,
                fee: position.fee,
                tickLower: position.tickLower,
                tickUpper: position.tickUpper,
                liquidity: position.liquidity,
                feeGrowthInside0LastX128: position.feeGrowthInside0LastX128,
                feeGrowthInside1LastX128: position.feeGrowthInside1LastX128,
                tokensOwed0: position.tokensOwed0,
                tokensOwed1: position.tokensOwed1
            };
        } catch (error) {
            throw new Error(`Failed to get position: ${error.message}`);
        }
    }

    async getPoolState(tokenA, tokenB, fee = 3000) {
        try {
            const poolInfo = await this.getPool(tokenA, tokenB, fee);
            if (!poolInfo) {
                return null;
            }

            const [slot0, liquidity] = await Promise.all([
                poolInfo.contract.slot0(),
                poolInfo.contract.liquidity()
            ]);

            return {
                sqrtPriceX96: slot0.sqrtPriceX96,
                tick: slot0.tick,
                observationIndex: slot0.observationIndex,
                observationCardinality: slot0.observationCardinality,
                observationCardinalityNext: slot0.observationCardinalityNext,
                feeProtocol: slot0.feeProtocol,
                unlocked: slot0.unlocked,
                liquidity: liquidity
            };
        } catch (error) {
            throw new Error(`Failed to get pool state: ${error.message}`);
        }
    }

    async estimateGasCost(method, params) {
        try {
            const gasEstimate = await this.routerContract.estimateGas[method](params);
            const gasPrice = await this.provider.getGasPrice();
            return gasEstimate.mul(gasPrice);
        } catch (error) {
            return ethers.utils.parseUnits('150000', 'gwei');
        }
    }

    getFee() {
        return 0.003;
    }

    getRouterAddress() {
        return this.routerAddress;
    }

    async parseTransaction(tx) {
        try {
            const iface = new ethers.utils.Interface(this.config.routerABI);
            const decoded = iface.parseTransaction({ data: tx.data });

            if (decoded.name === 'exactInputSingle') {
                return {
                    tokenIn: decoded.args.params.tokenIn,
                    tokenOut: decoded.args.params.tokenOut,
                    amountIn: decoded.args.params.amountIn,
                    price: await this.getPrice(decoded.args.params.tokenIn, decoded.args.params.tokenOut)
                };
            }

            if (decoded.name === 'exactInput') {
                const path = this.decodePath(decoded.args.params.path);
                return {
                    tokenIn: path[0].tokenAddress,
                    tokenOut: path[path.length - 1].tokenAddress,
                    amountIn: decoded.args.params.amountIn,
                    price: await this.getPrice(path[0].tokenAddress, path[path.length - 1].tokenAddress)
                };
            }

            return null;
        } catch (error) {
            return null;
        }
    }

    decodePath(path) {
        const tokens = [];
        let offset = 0;

        while (offset < path.length) {
            const tokenAddress = '0x' + path.slice(offset + 2, offset + 42).toLowerCase();
            tokens.push({ tokenAddress });
            
            offset += 40;
            if (offset < path.length) {
                const fee = parseInt(path.slice(offset + 2, offset + 8), 16);
                tokens[tokens.length - 1].fee = fee;
                offset += 6;
            }
        }

        return tokens;
    }

    async getInterface() {
        return new ethers.utils.Interface(this.config.routerABI);
    }

    async executeTriangularArbitrage(path, pairs, startAmount, minProfit) {
        try {
            const swapPath = this.encodePath(path, pairs);
            
            const params = {
                path: swapPath,
                recipient: await this.provider.getSigner().getAddress(),
                deadline: Math.floor(Date.now() / 1000) + 300,
                amountIn: startAmount,
                amountOutMinimum: startAmount.add(minProfit)
            };

            const tx = await this.routerContract.exactInput(params);
            return tx.hash;
        } catch (error) {
            throw new Error(`Failed to execute triangular arbitrage: ${error.message}`);
        }
    }

    encodePath(tokens, fees) {
        let path = '0x';
        
        for (let i = 0; i < tokens.length; i++) {
            path += tokens[i].slice(2);
            
            if (i < fees.length) {
                const feeHex = fees[i].toString(16).padStart(6, '0');
                path += feeHex;
            }
        }
        
        return path;
    }

    async buyToken(tokenAddress, amount, dexName, maxPrice = null) {
        const wethAddress = this.config.wethAddress;
        const amountOut = await this.quoteExactInputSingle(
            wethAddress, 
            tokenAddress, 
            amount, 
            3000
        );
        
        return await this.swapExactInputSingle(
            wethAddress,
            tokenAddress,
            amount,
            amountOut.mul(95).div(100),
            3000
        );
    }

    async sellToken(tokenAddress, amount, dexName, minPrice = null) {
        const wethAddress = this.config.wethAddress;
        const amountOut = await this.quoteExactInputSingle(
            tokenAddress, 
            wethAddress, 
            amount, 
            3000
        );
        
        return await this.swapExactInputSingle(
            tokenAddress,
            wethAddress,
            amount,
            amountOut.mul(95).div(100),
            3000
        );
    }

    async waitForTransaction(txHash) {
        return await this.provider.waitForTransaction(txHash);
    }

    formatUnits(value, decimals = 18) {
        return ethers.utils.formatUnits(value, decimals);
    }
}

module.exports = UniswapV3;