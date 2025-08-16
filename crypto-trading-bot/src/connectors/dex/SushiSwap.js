const { ethers } = require('ethers');

class SushiSwap {
    constructor(config, provider) {
        this.provider = provider;
        this.config = config;
        this.factoryAddress = '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac';
        this.routerAddress = '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F';
        this.wethAddress = config.wethAddress;
        
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
        
        this.pairs = new Map();
        this.reserves = new Map();
    }

    async initialize() {
        await this.loadCommonPairs();
    }

    async loadCommonPairs() {
        const tokenPairs = this.config.tokenPairs || [];
        for (const [tokenA, tokenB] of tokenPairs) {
            await this.getPair(tokenA, tokenB);
        }
    }

    async getPair(tokenA, tokenB) {
        const pairKey = this.getPairKey(tokenA, tokenB);
        
        if (this.pairs.has(pairKey)) {
            return this.pairs.get(pairKey);
        }

        try {
            const pairAddress = await this.factoryContract.getPair(tokenA, tokenB);
            
            if (pairAddress === ethers.constants.AddressZero) {
                return null;
            }

            const pairContract = new ethers.Contract(
                pairAddress,
                this.config.pairABI,
                this.provider
            );

            const pairInfo = {
                address: pairAddress,
                contract: pairContract,
                token0: tokenA.toLowerCase() < tokenB.toLowerCase() ? tokenA : tokenB,
                token1: tokenA.toLowerCase() < tokenB.toLowerCase() ? tokenB : tokenA
            };

            this.pairs.set(pairKey, pairInfo);
            return pairInfo;
        } catch (error) {
            throw new Error(`Failed to get pair: ${error.message}`);
        }
    }

    async getReserves(tokenA, tokenB) {
        const pairKey = this.getPairKey(tokenA, tokenB);
        const cached = this.reserves.get(pairKey);
        
        if (cached && Date.now() - cached.timestamp < 10000) {
            return cached.reserves;
        }

        try {
            const pair = await this.getPair(tokenA, tokenB);
            if (!pair) {
                return { reserve0: 0, reserve1: 0 };
            }

            const reserves = await pair.contract.getReserves();
            const reserveData = {
                reserve0: reserves._reserve0,
                reserve1: reserves._reserve1,
                blockTimestampLast: reserves._blockTimestampLast
            };

            this.reserves.set(pairKey, {
                reserves: reserveData,
                timestamp: Date.now()
            });

            return reserveData;
        } catch (error) {
            return { reserve0: ethers.BigNumber.from(0), reserve1: ethers.BigNumber.from(0) };
        }
    }

    async getPrice(tokenA, tokenB) {
        try {
            const reserves = await this.getReserves(tokenA, tokenB);
            
            if (reserves.reserve0.eq(0) || reserves.reserve1.eq(0)) {
                return 0;
            }

            const pair = await this.getPair(tokenA, tokenB);
            if (!pair) return 0;

            const isToken0 = tokenA.toLowerCase() === pair.token0.toLowerCase();
            
            if (isToken0) {
                return parseFloat(ethers.utils.formatUnits(
                    reserves.reserve1.mul(ethers.utils.parseEther('1')).div(reserves.reserve0),
                    18
                ));
            } else {
                return parseFloat(ethers.utils.formatUnits(
                    reserves.reserve0.mul(ethers.utils.parseEther('1')).div(reserves.reserve1),
                    18
                ));
            }
        } catch (error) {
            return 0;
        }
    }

    async getAmountOut(amountIn, tokenIn, tokenOut) {
        try {
            const path = [tokenIn, tokenOut];
            const amounts = await this.routerContract.getAmountsOut(amountIn, path);
            return amounts[amounts.length - 1];
        } catch (error) {
            return ethers.BigNumber.from(0);
        }
    }

    async getAmountsOut(amountIn, path) {
        try {
            return await this.routerContract.getAmountsOut(amountIn, path);
        } catch (error) {
            return [];
        }
    }

    async swapExactTokensForTokens(amountIn, amountOutMin, path, to = null, deadline = null) {
        try {
            const recipient = to || await this.provider.getSigner().getAddress();
            const swapDeadline = deadline || Math.floor(Date.now() / 1000) + 300;

            const tx = await this.routerContract.swapExactTokensForTokens(
                amountIn,
                amountOutMin,
                path,
                recipient,
                swapDeadline
            );

            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to swap exact tokens for tokens: ${error.message}`);
        }
    }

    async swapTokensForExactTokens(amountOut, amountInMax, path, to = null, deadline = null) {
        try {
            const recipient = to || await this.provider.getSigner().getAddress();
            const swapDeadline = deadline || Math.floor(Date.now() / 1000) + 300;

            const tx = await this.routerContract.swapTokensForExactTokens(
                amountOut,
                amountInMax,
                path,
                recipient,
                swapDeadline
            );

            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to swap tokens for exact tokens: ${error.message}`);
        }
    }

    async swapExactETHForTokens(amountOutMin, path, to = null, deadline = null, value) {
        try {
            const recipient = to || await this.provider.getSigner().getAddress();
            const swapDeadline = deadline || Math.floor(Date.now() / 1000) + 300;

            const tx = await this.routerContract.swapExactETHForTokens(
                amountOutMin,
                path,
                recipient,
                swapDeadline,
                { value }
            );

            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to swap exact ETH for tokens: ${error.message}`);
        }
    }

    async swapTokensForExactETH(amountOut, amountInMax, path, to = null, deadline = null) {
        try {
            const recipient = to || await this.provider.getSigner().getAddress();
            const swapDeadline = deadline || Math.floor(Date.now() / 1000) + 300;

            const tx = await this.routerContract.swapTokensForExactETH(
                amountOut,
                amountInMax,
                path,
                recipient,
                swapDeadline
            );

            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to swap tokens for exact ETH: ${error.message}`);
        }
    }

    async addLiquidity(tokenA, tokenB, amountADesired, amountBDesired, amountAMin, amountBMin, to = null, deadline = null) {
        try {
            const recipient = to || await this.provider.getSigner().getAddress();
            const liquidityDeadline = deadline || Math.floor(Date.now() / 1000) + 300;

            const tx = await this.routerContract.addLiquidity(
                tokenA,
                tokenB,
                amountADesired,
                amountBDesired,
                amountAMin,
                amountBMin,
                recipient,
                liquidityDeadline
            );

            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to add liquidity: ${error.message}`);
        }
    }

    async removeLiquidity(tokenA, tokenB, liquidity, amountAMin, amountBMin, to = null, deadline = null) {
        try {
            const recipient = to || await this.provider.getSigner().getAddress();
            const liquidityDeadline = deadline || Math.floor(Date.now() / 1000) + 300;

            const tx = await this.routerContract.removeLiquidity(
                tokenA,
                tokenB,
                liquidity,
                amountAMin,
                amountBMin,
                recipient,
                liquidityDeadline
            );

            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to remove liquidity: ${error.message}`);
        }
    }

    async getLiquidity(tokenA, tokenB) {
        try {
            const reserves = await this.getReserves(tokenA, tokenB);
            const reserve0 = parseFloat(ethers.utils.formatEther(reserves.reserve0));
            const reserve1 = parseFloat(ethers.utils.formatEther(reserves.reserve1));
            return Math.min(reserve0, reserve1);
        } catch (error) {
            return 0;
        }
    }

    async simulateSwap(tokenIn, tokenOut, amountIn) {
        try {
            const amountOut = await this.getAmountOut(amountIn, tokenIn, tokenOut);
            const price = parseFloat(ethers.utils.formatEther(
                amountOut.mul(ethers.utils.parseEther('1')).div(amountIn)
            ));
            return price;
        } catch (error) {
            return 0;
        }
    }

    async estimateGasCost(method = 'swapExactTokensForTokens', params = []) {
        try {
            const gasEstimate = await this.routerContract.estimateGas[method](...params);
            const gasPrice = await this.provider.getGasPrice();
            return gasEstimate.mul(gasPrice);
        } catch (error) {
            return ethers.utils.parseUnits('200000', 'gwei');
        }
    }

    getFee(pair = null) {
        return 0.003;
    }

    getRouterAddress() {
        return this.routerAddress;
    }

    async parseTransaction(tx) {
        try {
            const iface = new ethers.utils.Interface(this.config.routerABI);
            const decoded = iface.parseTransaction({ data: tx.data });

            if (decoded.name === 'swapExactTokensForTokens') {
                const path = decoded.args.path;
                return {
                    tokenIn: path[0],
                    tokenOut: path[path.length - 1],
                    amountIn: decoded.args.amountIn,
                    price: await this.getPrice(path[0], path[path.length - 1])
                };
            }

            if (decoded.name === 'swapExactETHForTokens') {
                const path = decoded.args.path;
                return {
                    tokenIn: path[0],
                    tokenOut: path[path.length - 1],
                    amountIn: tx.value,
                    price: await this.getPrice(path[0], path[path.length - 1])
                };
            }

            return null;
        } catch (error) {
            return null;
        }
    }

    async getInterface() {
        return new ethers.utils.Interface(this.config.routerABI);
    }

    async executeTriangularArbitrage(path, pairs, startAmount, minProfit) {
        try {
            const amountOutMin = startAmount.add(minProfit);
            
            const tx = await this.swapExactTokensForTokens(
                startAmount,
                amountOutMin,
                path
            );

            return tx.transactionHash;
        } catch (error) {
            throw new Error(`Failed to execute triangular arbitrage: ${error.message}`);
        }
    }

    async buyToken(tokenAddress, amount, dexName, maxPrice = null) {
        const path = [this.wethAddress, tokenAddress];
        const amountsOut = await this.getAmountsOut(amount, path);
        const amountOutMin = amountsOut[1].mul(95).div(100);

        return await this.swapExactTokensForTokens(
            amount,
            amountOutMin,
            path
        );
    }

    async sellToken(tokenAddress, amount, dexName, minPrice = null) {
        const path = [tokenAddress, this.wethAddress];
        const amountsOut = await this.getAmountsOut(amount, path);
        const amountOutMin = amountsOut[1].mul(95).div(100);

        return await this.swapExactTokensForTokens(
            amount,
            amountOutMin,
            path
        );
    }

    async waitForTransaction(txHash) {
        return await this.provider.waitForTransaction(txHash);
    }

    formatUnits(value, decimals = 18) {
        return ethers.utils.formatUnits(value, decimals);
    }

    getPairKey(tokenA, tokenB) {
        const sortedTokens = [tokenA.toLowerCase(), tokenB.toLowerCase()].sort();
        return `${sortedTokens[0]}-${sortedTokens[1]}`;
    }

    async getOptimalAmount(tokenA, tokenB, amountA) {
        try {
            const reserves = await this.getReserves(tokenA, tokenB);
            const pair = await this.getPair(tokenA, tokenB);
            
            if (!pair || reserves.reserve0.eq(0) || reserves.reserve1.eq(0)) {
                return ethers.BigNumber.from(0);
            }

            const isToken0 = tokenA.toLowerCase() === pair.token0.toLowerCase();
            const reserveA = isToken0 ? reserves.reserve0 : reserves.reserve1;
            const reserveB = isToken0 ? reserves.reserve1 : reserves.reserve0;

            return amountA.mul(reserveB).div(reserveA);
        } catch (error) {
            return ethers.BigNumber.from(0);
        }
    }

    async quote(amountA, reserveA, reserveB) {
        if (amountA.eq(0) || reserveA.eq(0) || reserveB.eq(0)) {
            return ethers.BigNumber.from(0);
        }
        return amountA.mul(reserveB).div(reserveA);
    }

    async getAmountIn(amountOut, reserveIn, reserveOut) {
        if (amountOut.eq(0) || reserveIn.eq(0) || reserveOut.eq(0)) {
            return ethers.BigNumber.from(0);
        }

        const numerator = reserveIn.mul(amountOut).mul(1000);
        const denominator = reserveOut.sub(amountOut).mul(997);
        return numerator.div(denominator).add(1);
    }

    async getPairInfo(tokenA, tokenB) {
        const pair = await this.getPair(tokenA, tokenB);
        if (!pair) return null;

        const reserves = await this.getReserves(tokenA, tokenB);
        const totalSupply = await pair.contract.totalSupply();
        
        return {
            address: pair.address,
            token0: pair.token0,
            token1: pair.token1,
            reserve0: reserves.reserve0,
            reserve1: reserves.reserve1,
            totalSupply: totalSupply,
            blockTimestampLast: reserves.blockTimestampLast
        };
    }

    async getAllPairs() {
        const allPairsLength = await this.factoryContract.allPairsLength();
        const pairs = [];

        for (let i = 0; i < allPairsLength; i++) {
            try {
                const pairAddress = await this.factoryContract.allPairs(i);
                pairs.push(pairAddress);
            } catch (error) {
                continue;
            }
        }

        return pairs;
    }

    async calculatePriceImpact(tokenIn, tokenOut, amountIn) {
        try {
            const currentPrice = await this.getPrice(tokenIn, tokenOut);
            const amountOut = await this.getAmountOut(amountIn, tokenIn, tokenOut);
            const executionPrice = parseFloat(ethers.utils.formatEther(
                amountOut.mul(ethers.utils.parseEther('1')).div(amountIn)
            ));
            
            return Math.abs((executionPrice - currentPrice) / currentPrice);
        } catch (error) {
            return 0;
        }
    }
}

module.exports = SushiSwap;