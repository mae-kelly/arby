const { ethers } = require('ethers');
const EventEmitter = require('events');

class EthereumConnector extends EventEmitter {
    constructor(config) {
        super();
        this.config = config;
        this.provider = null;
        this.signer = null;
        this.gasTracker = new Map();
        this.nonce = null;
        this.chainId = 1;
        this.blockTime = 12000;
    }

    async initialize() {
        this.provider = new ethers.providers.JsonRpcProvider(this.config.rpcUrl, {
            name: 'ethereum',
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
            const block = await this.provider.getBlock('latest');
            
            this.gasTracker.set('standard', gasPrice);
            this.gasTracker.set('fast', gasPrice.mul(120).div(100));
            this.gasTracker.set('fastest', gasPrice.mul(150).div(100));
            this.gasTracker.set('baseFee', block.baseFeePerGas || gasPrice);
            
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
        }, 15000);
    }

    async getBalance(address = null) {
        try {
            const targetAddress = address || this.signer?.address;
            if (!targetAddress) throw new Error('No address provided');

            const balance = await this.provider.getBalance(targetAddress);
            return {
                ETH: {
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

    async getPrice(tokenAddress, dex = 'uniswap') {
        try {
            const wethAddress = '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2';
            
            if (dex === 'uniswap') {
                const quoterAddress = '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6';
                const quoterContract = new ethers.Contract(
                    quoterAddress,
                    ['function quoteExactInputSingle(address,address,uint24,uint256,uint160) view returns (uint256)'],
                    this.provider
                );

                const amount = ethers.utils.parseEther('1');
                const quote = await quoterContract.quoteExactInputSingle(
                    tokenAddress,
                    wethAddress,
                    3000,
                    amount,
                    0
                );

                return parseFloat(ethers.utils.formatEther(quote));
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

            const gasPrice = this.gasTracker.get('standard') || await this.provider.getGasPrice();
            return gasLimit.mul(gasPrice);
        } catch (error) {
            return ethers.utils.parseUnits('100000', 'gwei');
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
                gasLimit: gasSettings.gasLimit || 200000,
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

    async waitForTransaction(txHash, confirmations = 1, timeout = 300000) {
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
            const routerAddress = dexAddress || '0xE592427A0AEce92De3Edee1F18E0157C05861564';
            const wethAddress = '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2';
            
            const routerContract = new ethers.Contract(
                routerAddress,
                ['function exactInputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160)) external returns (uint256)'],
                this.signer
            );

            const params = {
                tokenIn: wethAddress,
                tokenOut: tokenAddress,
                fee: 3000,
                recipient: this.signer.address,
                deadline: Math.floor(Date.now() / 1000) + 300,
                amountIn: amount,
                amountOutMinimum: 0,
                sqrtPriceLimitX96: 0
            };

            const tx = await routerContract.exactInputSingle(params);
            return await tx.wait();
        } catch (error) {
            throw new Error(`Failed to buy token: ${error.message}`);
        }
    }

    async sellToken(tokenAddress, amount, dexAddress, minPrice = null) {
        try {
            const routerAddress = dexAddress || '0xE592427A0AEce92De3Edee1F18E0157C05861564';
            const wethAddress = '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2';
            
            const routerContract = new ethers.Contract(
                routerAddress,
                ['function exactInputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160)) external returns (uint256)'],
                this.signer
            );

            const params = {
                tokenIn: tokenAddress,
                tokenOut: wethAddress,
                fee: 3000,
                recipient: this.signer.address,
                deadline: Math.floor(Date.now() / 1000) + 300,
                amountIn: amount,
                amountOutMinimum: 0,
                sqrtPriceLimitX96: 0
            };

            const tx = await routerContract.exactInputSingle(params);
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

    async deployContract(abi, bytecode, constructorArgs = [], gasLimit = 3000000) {
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
            return ethers.BigNumber.from(200000);
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
                name: network.name,
                blockNumber: block.number,
                blockTime: Date.now() - (block.timestamp * 1000),
                gasPrice: ethers.utils.formatUnits(gasPrice, 'gwei'),
                baseFee: block.baseFeePerGas ? ethers.utils.formatUnits(block.baseFeePerGas, 'gwei') : null
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
            const multicallAddress = '0x5BA1e12693Dc8F9c48aAD8770482f4739bEeD696';
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

    async waitForConfirmations(txHash, requiredConfirmations = 3) {
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

module.exports = EthereumConnector;