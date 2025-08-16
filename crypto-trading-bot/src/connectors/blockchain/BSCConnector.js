const { ethers } = require('ethers');
const EventEmitter = require('events');

class BSCConnector extends EventEmitter {
    constructor(config) {
        super();
        this.config = config;
        this.provider = null;
        this.signer = null;
        this.gasTracker = new Map();
        this.nonce = null;
        this.chainId = 56;
        this.blockTime = 3000;
    }

    async initialize() {
        this.provider = new ethers.providers.JsonRpcProvider(this.config.rpcUrl, {
            name: 'bsc',
            chainId: this.chainId
        });

        if (this.config.privateKey) {
            this.signer = new ethers.Wallet(this.config.privateKey, this.provider);
        }

        await this.updateGasSettings();
        await this.updateNonce();
        
        this.startGasTracking();
        this.emit('connected');
    }

    async updateGasSettings() {
        try {
            const gasPrice = await this.provider.getGasPrice();
            
            this.gasTracker.set('standard', ethers.utils.parseUnits('5', 'gwei'));
            this.gasTracker.set('fast', ethers.utils.parseUnits('6', 'gwei'));
            this.gasTracker.set('fastest', ethers.utils.parseUnits('10', 'gwei'));
            this.gasTracker.set('current', gasPrice);
            
        } catch (error) {
            this.emit('error', `Failed to update gas settings: ${error.message}`);
        }
    }

    async updateNonce() {
        if (this.signer) {
            this.nonce = await this.provider.getTransactionCount(this.signer.address, 'pending');
        }
    }

    startGasTracking() {
        setInterval(async () => {
            await this.updateGasSettings();
        }, 10000);
    }

    async getBalance(address = null) {
        try {
            const targetAddress = address || this.signer?.address;
            if (!targetAddress) throw new Error('No address provided');

            const balance = await this.provider.getBalance(targetAddress);
            return {
                BNB: {
                    balance: ethers.utils.formatEther(balance),
                    wei: balance
                }
            };
        } catch (error) {
            throw new Error(`Failed to get balance: ${error.message}`);
        }
    }

    async getTokenBalance(tokenAddress, holderAddress = null) {
        try {
            const targetAddress = holderAddress || this.signer?.address;
            if (!targetAddress) throw new Error('No address provided');

            const tokenContract = new ethers.Contract(
                tokenAddress,
                ['function balanceOf(address) view returns (uint256)',
                 'function decimals() view returns (uint8)',
                 'function symbol() view returns (string)'],
                this.provider
            );

            const [balance, decimals, symbol] = await Promise.all([
                tokenContract.balanceOf(targetAddress),
                tokenContract.decimals(),
                tokenContract.symbol()
            ]);

            return {
                balance: ethers.utils.formatUnits(balance, decimals),
                wei: balance,
                decimals,
                symbol
            };
        } catch (error) {
            throw new Error(`Failed to get token balance: ${error.message}`);
        }
    }

    async getPrice(tokenAddress, dex = 'pancakeswap') {
        try {
            const wbnbAddress = '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c';
            
            if (dex === 'pancakeswap') {
                const routerAddress = '0x10ED43C718714eb63d5aA57B78B54704E256024E';
                const routerContract = new ethers.Contract(
                    routerAddress,
                    ['function getAmountsOut(uint,address[]) view returns (uint[])'],
                    this.provider
                );

                const amount = ethers.utils.parseEther('1');
                const path = [tokenAddress, wbnbAddress];
                const amounts = await routerContract.getAmountsOut(amount, path);

                return parseFloat(ethers.utils.formatEther(amounts[1]));
            }

            return 0;
        } catch (error) {
            return 0;
        }
    }

    async estimateGasCost(to = null, data = '0x', value = 0) {
        try {
            const gasLimit = await this.provider.estimateGas({
                to,
                data,
                value: ethers.BigNumber.from(value),
                from: this.signer?.address
            });

            const gasPrice = this.gasTracker.get('standard');
            return gasLimit.mul(gasPrice);
        } catch (error) {
            return ethers.utils.parseUnits('200000', 'gwei');
        }
    }

    async sendTransaction(to, data = '0x', value = 0, gasSettings = {}) {
        try {
            if (!this.signer) throw new Error('No signer available');

            const tx = {
                to,
                data,
                value: ethers.BigNumber.from(value),
                nonce: this.nonce,
                gasLimit: gasSettings.gasLimit || 300000,
                gasPrice: gasSettings.gasPrice || this.gasTracker.get('fast'),
                chainId: this.chainId
            };

            const signedTx = await this.signer.signTransaction(tx);
            const response = await this.provider.sendTransaction(signedTx);
            
            this.nonce++;
            this.emit('transactionSent', response);
            
            return response;
        } catch (error) {
            await this.updateNonce();
            throw new Error(`Failed to send transaction: ${error.message}`);
        }
    }

    async waitForTransaction(txHash, confirmations = 1, timeout = 180000) {
        try {
            const receipt = await this.provider.waitForTransaction(txHash, confirmations, timeout);
            this.emit('transactionConfirmed', receipt);
            return receipt;
        } catch (error) {
            throw new Error(`Transaction timeout or failed: ${error.message}`);
        }
    }

    async buyToken(tokenAddress, amount, dexAddress, maxPrice = null) {
        try {
            const routerAddress = dexAddress || '0x10ED43C718714eb63d5aA57B78B54704E256024E';
            const wbnbAddress = '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c';
            
            const routerContract = new ethers.Contract(
                routerAddress,
                ['function swapExactETHForTokens(uint,address[],address,uint) external payable returns (uint[])'],
                this.signer
            );

            const path = [wbnbAddress, tokenAddress];
            const deadline = Math.floor(Date.now() / 1000) + 300;
            const amountOutMin = 0;

            const tx = await routerContract.swapExactETHForTokens(
                amountOutMin,
                path,
                this.signer.address,
                deadline,
                { value: amount }
            );
            
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to buy token: ${error.message}`);
        }
    }

    async sellToken(tokenAddress, amount, dexAddress, minPrice = null) {
        try {
            const routerAddress = dexAddress || '0x10ED43C718714eb63d5aA57B78B54704E256024E';
            const wbnbAddress = '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c';
            
            const routerContract = new ethers.Contract(
                routerAddress,
                ['function swapExactTokensForETH(uint,uint,address[],address,uint) external returns (uint[])'],
                this.signer
            );

            const path = [tokenAddress, wbnbAddress];
            const deadline = Math.floor(Date.now() / 1000) + 300;
            const amountOutMin = 0;

            const tx = await routerContract.swapExactTokensForETH(
                amount,
                amountOutMin,
                path,
                this.signer.address,
                deadline
            );
            
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to sell token: ${error.message}`);
        }
    }

    async approveToken(tokenAddress, spenderAddress, amount = null) {
        try {
            const tokenContract = new ethers.Contract(
                tokenAddress,
                ['function approve(address,uint256) external returns (bool)'],
                this.signer
            );

            const approveAmount = amount || ethers.constants.MaxUint256;
            const tx = await tokenContract.approve(spenderAddress, approveAmount);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to approve token: ${error.message}`);
        }
    }

    async getAllowance(tokenAddress, ownerAddress, spenderAddress) {
        try {
            const tokenContract = new ethers.Contract(
                tokenAddress,
                ['function allowance(address,address) view returns (uint256)'],
                this.provider
            );

            return await tokenContract.allowance(ownerAddress, spenderAddress);
        } catch (error) {
            return ethers.BigNumber.from(0);
        }
    }

    async getBlockNumber() {
        return await this.provider.getBlockNumber();
    }

    async getBlock(blockNumber = 'latest') {
        return await this.provider.getBlock(blockNumber);
    }

    async getTransaction(txHash) {
        return await this.provider.getTransaction(txHash);
    }

    async getTransactionReceipt(txHash) {
        return await this.provider.getTransactionReceipt(txHash);
    }

    async getLogs(filter) {
        return await this.provider.getLogs(filter);
    }

    async getContractEvents(contractAddress, abi, eventName, fromBlock = 0, toBlock = 'latest') {
        try {
            const contract = new ethers.Contract(contractAddress, abi, this.provider);
            const filter = contract.filters[eventName]();
            
            return await contract.queryFilter(filter, fromBlock, toBlock);
        } catch (error) {
            throw new Error(`Failed to get contract events: ${error.message}`);
        }
    }

    async callContract(contractAddress, abi, methodName, params = [], value = 0) {
        try {
            const contract = new ethers.Contract(contractAddress, abi, this.signer || this.provider);
            
            if (value > 0) {
                return await contract[methodName](...params, { value });
            } else {
                return await contract[methodName](...params);
            }
        } catch (error) {
            throw new Error(`Failed to call contract: ${error.message}`);
        }
    }

    async deployContract(abi, bytecode, constructorArgs = [], gasLimit = 2000000) {
        try {
            if (!this.signer) throw new Error('No signer available');

            const factory = new ethers.ContractFactory(abi, bytecode, this.signer);
            const contract = await factory.deploy(...constructorArgs, { gasLimit });
            await contract.deployed();
            
            return contract;
        } catch (error) {
            throw new Error(`Failed to deploy contract: ${error.message}`);
        }
    }

    async getGasPrice(speed = 'standard') {
        return this.gasTracker.get(speed) || await this.provider.getGasPrice();
    }

    async estimateGasLimit(to, data = '0x', value = 0) {
        try {
            return await this.provider.estimateGas({
                to,
                data,
                value: ethers.BigNumber.from(value),
                from: this.signer?.address
            });
        } catch (error) {
            return ethers.BigNumber.from(300000);
        }
    }

    async getCurrentNonce(address = null) {
        const targetAddress = address || this.signer?.address;
        return await this.provider.getTransactionCount(targetAddress, 'pending');
    }

    async getNetworkInfo() {
        try {
            const network = await this.provider.getNetwork();
            const block = await this.provider.getBlock('latest');
            const gasPrice = await this.provider.getGasPrice();

            return {
                chainId: network.chainId,
                name: 'Binance Smart Chain',
                blockNumber: block.number,
                blockTime: Date.now() - (block.timestamp * 1000),
                gasPrice: ethers.utils.formatUnits(gasPrice, 'gwei')
            };
        } catch (error) {
            throw new Error(`Failed to get network info: ${error.message}`);
        }
    }

    async monitorMempool(callback) {
        this.provider.on('pending', async (txHash) => {
            try {
                const tx = await this.provider.getTransaction(txHash);
                if (tx) {
                    callback(tx);
                }
            } catch (error) {
                // Ignore failed transaction fetches
            }
        });
    }

    async subscribeToBlocks(callback) {
        this.provider.on('block', async (blockNumber) => {
            try {
                const block = await this.provider.getBlock(blockNumber, true);
                callback(block);
            } catch (error) {
                this.emit('error', `Failed to fetch block ${blockNumber}: ${error.message}`);
            }
        });
    }

    async getTokenInfo(tokenAddress) {
        try {
            const tokenContract = new ethers.Contract(
                tokenAddress,
                [
                    'function name() view returns (string)',
                    'function symbol() view returns (string)',
                    'function decimals() view returns (uint8)',
                    'function totalSupply() view returns (uint256)'
                ],
                this.provider
            );

            const [name, symbol, decimals, totalSupply] = await Promise.all([
                tokenContract.name(),
                tokenContract.symbol(),
                tokenContract.decimals(),
                tokenContract.totalSupply()
            ]);

            return {
                address: tokenAddress,
                name,
                symbol,
                decimals,
                totalSupply: ethers.utils.formatUnits(totalSupply, decimals)
            };
        } catch (error) {
            throw new Error(`Failed to get token info: ${error.message}`);
        }
    }

    async batchCall(calls) {
        try {
            const multicallAddress = '0x41263cBA59EB80dC200F3E2544eda4ed6A90E76C';
            const multicallABI = [
                'function aggregate(tuple(address target, bytes callData)[] calls) returns (uint256 blockNumber, bytes[] returnData)'
            ];

            const multicall = new ethers.Contract(multicallAddress, multicallABI, this.provider);
            const result = await multicall.aggregate(calls);
            
            return result.returnData;
        } catch (error) {
            throw new Error(`Failed to execute batch call: ${error.message}`);
        }
    }

    async waitForConfirmations(txHash, requiredConfirmations = 5) {
        let confirmations = 0;
        let receipt = null;

        while (confirmations < requiredConfirmations) {
            receipt = await this.provider.getTransactionReceipt(txHash);
            
            if (receipt) {
                const currentBlock = await this.provider.getBlockNumber();
                confirmations = currentBlock - receipt.blockNumber + 1;
                
                if (confirmations < requiredConfirmations) {
                    await new Promise(resolve => setTimeout(resolve, this.blockTime));
                }
            } else {
                await new Promise(resolve => setTimeout(resolve, this.blockTime));
            }
        }

        return receipt;
    }

    async getPancakeSwapPairs() {
        try {
            const factoryAddress = '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73';
            const factoryContract = new ethers.Contract(
                factoryAddress,
                ['function allPairsLength() view returns (uint256)',
                 'function allPairs(uint256) view returns (address)'],
                this.provider
            );

            const pairsLength = await factoryContract.allPairsLength();
            const pairs = [];

            for (let i = 0; i < Math.min(pairsLength, 1000); i++) {
                try {
                    const pairAddress = await factoryContract.allPairs(i);
                    pairs.push(pairAddress);
                } catch (error) {
                    continue;
                }
            }

            return pairs;
        } catch (error) {
            throw new Error(`Failed to get PancakeSwap pairs: ${error.message}`);
        }
    }

    async getBNBPrice() {
        try {
            const busdAddress = '0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56';
            const wbnbAddress = '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c';
            const routerAddress = '0x10ED43C718714eb63d5aA57B78B54704E256024E';
            
            const routerContract = new ethers.Contract(
                routerAddress,
                ['function getAmountsOut(uint,address[]) view returns (uint[])'],
                this.provider
            );

            const amount = ethers.utils.parseEther('1');
            const path = [wbnbAddress, busdAddress];
            const amounts = await routerContract.getAmountsOut(amount, path);

            return parseFloat(ethers.utils.formatEther(amounts[1]));
        } catch (error) {
            return 0;
        }
    }

    async getValidatorInfo() {
        try {
            const validatorContract = new ethers.Contract(
                '0x0000000000000000000000000000000000001000',
                ['function getValidators() view returns (address[])'],
                this.provider
            );

            const validators = await validatorContract.getValidators();
            return validators;
        } catch (error) {
            return [];
        }
    }

    async calculateOptimalGasPrice() {
        try {
            const block = await this.provider.getBlock('latest');
            const transactions = await Promise.all(
                block.transactions.slice(0, 10).map(txHash => 
                    this.provider.getTransaction(txHash)
                )
            );

            const gasPrices = transactions
                .filter(tx => tx && tx.gasPrice)
                .map(tx => tx.gasPrice)
                .sort((a, b) => a.sub(b));

            if (gasPrices.length === 0) {
                return this.gasTracker.get('standard');
            }

            const medianIndex = Math.floor(gasPrices.length / 2);
            return gasPrices[medianIndex];
        } catch (error) {
            return this.gasTracker.get('standard');
        }
    }

    async swapTokensOnPancakeSwap(tokenIn, tokenOut, amountIn, slippage = 0.5) {
        try {
            const routerAddress = '0x10ED43C718714eb63d5aA57B78B54704E256024E';
            const routerContract = new ethers.Contract(
                routerAddress,
                [
                    'function getAmountsOut(uint,address[]) view returns (uint[])',
                    'function swapExactTokensForTokens(uint,uint,address[],address,uint) external returns (uint[])'
                ],
                this.signer
            );

            const path = [tokenIn, tokenOut];
            const amounts = await routerContract.getAmountsOut(amountIn, path);
            const amountOutMin = amounts[1].mul(100 - Math.floor(slippage * 100)).div(100);
            const deadline = Math.floor(Date.now() / 1000) + 300;

            const tx = await routerContract.swapExactTokensForTokens(
                amountIn,
                amountOutMin,
                path,
                this.signer.address,
                deadline
            );

            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to swap tokens on PancakeSwap: ${error.message}`);
        }
    }

    async addLiquidityToPancakeSwap(tokenA, tokenB, amountA, amountB, slippage = 0.5) {
        try {
            const routerAddress = '0x10ED43C718714eb63d5aA57B78B54704E256024E';
            const routerContract = new ethers.Contract(
                routerAddress,
                ['function addLiquidity(address,address,uint,uint,uint,uint,address,uint) external returns (uint,uint,uint)'],
                this.signer
            );

            const amountAMin = amountA.mul(100 - Math.floor(slippage * 100)).div(100);
            const amountBMin = amountB.mul(100 - Math.floor(slippage * 100)).div(100);
            const deadline = Math.floor(Date.now() / 1000) + 300;

            const tx = await routerContract.addLiquidity(
                tokenA,
                tokenB,
                amountA,
                amountB,
                amountAMin,
                amountBMin,
                this.signer.address,
                deadline
            );

            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to add liquidity to PancakeSwap: ${error.message}`);
        }
    }

    disconnect() {
        if (this.provider) {
            this.provider.removeAllListeners();
        }
        this.emit('disconnected');
    }

    formatUnits(value, decimals = 18) {
        return ethers.utils.formatUnits(value, decimals);
    }

    parseUnits(value, decimals = 18) {
        return ethers.utils.parseUnits(value.toString(), decimals);
    }

    isAddress(address) {
        return ethers.utils.isAddress(address);
    }

    getAddress() {
        return this.signer?.address || null;
    }

    async sign(message) {
        if (!this.signer) throw new Error('No signer available');
        return await this.signer.signMessage(message);
    }

    async signTypedData(domain, types, value) {
        if (!this.signer) throw new Error('No signer available');
        return await this.signer._signTypedData(domain, types, value);
    }
}

module.exports = BSCConnector;