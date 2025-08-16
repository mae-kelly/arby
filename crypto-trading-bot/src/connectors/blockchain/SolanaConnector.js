const { Connection, PublicKey, Keypair, Transaction, SystemProgram, LAMPORTS_PER_SOL } = require('@solana/web3.js');
const { Token, TOKEN_PROGRAM_ID, ASSOCIATED_TOKEN_PROGRAM_ID } = require('@solana/spl-token');
const { Jupiter } = require('@jup-ag/core');
const EventEmitter = require('events');

class SolanaConnector extends EventEmitter {
    constructor(config) {
        super();
        this.config = config;
        this.connection = null;
        this.wallet = null;
        this.jupiter = null;
        this.tokenAccounts = new Map();
        this.priceCache = new Map();
        this.blockTime = 400;
    }

    async initialize() {
        this.connection = new Connection(this.config.rpcUrl, 'confirmed');
        
        if (this.config.privateKey) {
            const secretKey = new Uint8Array(JSON.parse(this.config.privateKey));
            this.wallet = Keypair.fromSecretKey(secretKey);
        }

        await this.initializeJupiter();
        await this.loadTokenAccounts();
        
        this.emit('connected');
    }

    async initializeJupiter() {
        try {
            this.jupiter = await Jupiter.load({
                connection: this.connection,
                cluster: 'mainnet-beta',
                user: this.wallet
            });
        } catch (error) {
            this.emit('error', `Failed to initialize Jupiter: ${error.message}`);
        }
    }

    async loadTokenAccounts() {
        if (!this.wallet) return;

        try {
            const tokenAccounts = await this.connection.getParsedTokenAccountsByOwner(
                this.wallet.publicKey,
                { programId: TOKEN_PROGRAM_ID }
            );

            for (const { account, pubkey } of tokenAccounts.value) {
                const mintAddress = account.data.parsed.info.mint;
                const balance = account.data.parsed.info.tokenAmount;
                
                this.tokenAccounts.set(mintAddress, {
                    address: pubkey.toString(),
                    mint: mintAddress,
                    balance: balance.uiAmount,
                    decimals: balance.decimals
                });
            }
        } catch (error) {
            this.emit('error', `Failed to load token accounts: ${error.message}`);
        }
    }

    async getBalance(address = null) {
        try {
            const targetAddress = address ? new PublicKey(address) : this.wallet?.publicKey;
            if (!targetAddress) throw new Error('No address provided');

            const balance = await this.connection.getBalance(targetAddress);
            return {
                SOL: {
                    balance: balance / LAMPORTS_PER_SOL,
                    lamports: balance
                }
            };
        } catch (error) {
            throw new Error(`Failed to get balance: ${error.message}`);
        }
    }

    async getTokenBalance(mintAddress, holderAddress = null) {
        try {
            const targetAddress = holderAddress ? new PublicKey(holderAddress) : this.wallet?.publicKey;
            if (!targetAddress) throw new Error('No address provided');

            const tokenAccounts = await this.connection.getParsedTokenAccountsByOwner(
                targetAddress,
                { mint: new PublicKey(mintAddress) }
            );

            if (tokenAccounts.value.length === 0) {
                return { balance: 0, lamports: 0, decimals: 9 };
            }

            const tokenAccount = tokenAccounts.value[0];
            const balance = tokenAccount.account.data.parsed.info.tokenAmount;

            return {
                balance: balance.uiAmount,
                lamports: balance.amount,
                decimals: balance.decimals
            };
        } catch (error) {
            throw new Error(`Failed to get token balance: ${error.message}`);
        }
    }

    async getPrice(mintAddress, outputMint = 'So11111111111111111111111111111111111111112') {
        try {
            const cacheKey = `${mintAddress}-${outputMint}`;
            const cached = this.priceCache.get(cacheKey);
            
            if (cached && Date.now() - cached.timestamp < 30000) {
                return cached.price;
            }

            if (!this.jupiter) {
                await this.initializeJupiter();
            }

            const routes = await this.jupiter.computeRoutes({
                inputMint: new PublicKey(mintAddress),
                outputMint: new PublicKey(outputMint),
                amount: 1000000,
                slippageBps: 50
            });

            if (routes.routesInfos.length === 0) return 0;

            const bestRoute = routes.routesInfos[0];
            const price = bestRoute.outAmount / 1000000;

            this.priceCache.set(cacheKey, {
                price,
                timestamp: Date.now()
            });

            return price;
        } catch (error) {
            return 0;
        }
    }

    async estimateGasCost(instructions = []) {
        try {
            const recentBlockhash = await this.connection.getLatestBlockhash();
            const transaction = new Transaction({ recentBlockhash: recentBlockhash.blockhash });
            
            if (instructions.length === 0) {
                transaction.add(
                    SystemProgram.transfer({
                        fromPubkey: this.wallet.publicKey,
                        toPubkey: this.wallet.publicKey,
                        lamports: 1
                    })
                );
            } else {
                instructions.forEach(ix => transaction.add(ix));
            }

            const fees = await transaction.getEstimatedFee(this.connection);
            return fees || 5000;
        } catch (error) {
            return 5000;
        }
    }

    async sendTransaction(transaction, signers = []) {
        try {
            if (!this.wallet) throw new Error('No wallet available');

            const recentBlockhash = await this.connection.getLatestBlockhash();
            transaction.recentBlockhash = recentBlockhash.blockhash;
            transaction.feePayer = this.wallet.publicKey;

            const allSigners = [this.wallet, ...signers];
            transaction.sign(...allSigners);

            const signature = await this.connection.sendRawTransaction(transaction.serialize());
            this.emit('transactionSent', { signature });

            return signature;
        } catch (error) {
            throw new Error(`Failed to send transaction: ${error.message}`);
        }
    }

    async waitForTransaction(signature, commitment = 'confirmed', timeout = 60000) {
        try {
            const confirmation = await this.connection.confirmTransaction(signature, commitment);
            
            if (confirmation.value.err) {
                throw new Error(`Transaction failed: ${confirmation.value.err}`);
            }

            this.emit('transactionConfirmed', { signature, confirmation });
            return confirmation;
        } catch (error) {
            throw new Error(`Transaction timeout or failed: ${error.message}`);
        }
    }

    async buyToken(mintAddress, amount, dexName = 'jupiter', maxPrice = null) {
        try {
            const solMint = 'So11111111111111111111111111111111111111112';
            
            const routes = await this.jupiter.computeRoutes({
                inputMint: new PublicKey(solMint),
                outputMint: new PublicKey(mintAddress),
                amount: amount * LAMPORTS_PER_SOL,
                slippageBps: 100
            });

            if (routes.routesInfos.length === 0) {
                throw new Error('No routes found');
            }

            const { execute } = await this.jupiter.exchange({
                routeInfo: routes.routesInfos[0]
            });

            const swapResult = await execute();
            return swapResult;
        } catch (error) {
            throw new Error(`Failed to buy token: ${error.message}`);
        }
    }

    async sellToken(mintAddress, amount, dexName = 'jupiter', minPrice = null) {
        try {
            const solMint = 'So11111111111111111111111111111111111111112';
            const tokenAccount = this.tokenAccounts.get(mintAddress);
            
            if (!tokenAccount) {
                throw new Error('Token account not found');
            }

            const tokenAmount = amount * Math.pow(10, tokenAccount.decimals);
            
            const routes = await this.jupiter.computeRoutes({
                inputMint: new PublicKey(mintAddress),
                outputMint: new PublicKey(solMint),
                amount: tokenAmount,
                slippageBps: 100
            });

            if (routes.routesInfos.length === 0) {
                throw new Error('No routes found');
            }

            const { execute } = await this.jupiter.exchange({
                routeInfo: routes.routesInfos[0]
            });

            const swapResult = await execute();
            return swapResult;
        } catch (error) {
            throw new Error(`Failed to sell token: ${error.message}`);
        }
    }

    async createTokenAccount(mintAddress, owner = null) {
        try {
            const mintPublicKey = new PublicKey(mintAddress);
            const ownerPublicKey = owner ? new PublicKey(owner) : this.wallet.publicKey;

            const associatedTokenAddress = await Token.getAssociatedTokenAddress(
                ASSOCIATED_TOKEN_PROGRAM_ID,
                TOKEN_PROGRAM_ID,
                mintPublicKey,
                ownerPublicKey
            );

            const transaction = new Transaction().add(
                Token.createAssociatedTokenAccountInstruction(
                    ASSOCIATED_TOKEN_PROGRAM_ID,
                    TOKEN_PROGRAM_ID,
                    mintPublicKey,
                    associatedTokenAddress,
                    ownerPublicKey,
                    this.wallet.publicKey
                )
            );

            const signature = await this.sendTransaction(transaction);
            await this.waitForTransaction(signature);

            return associatedTokenAddress.toString();
        } catch (error) {
            throw new Error(`Failed to create token account: ${error.message}`);
        }
    }

    async transferToken(mintAddress, destinationAddress, amount) {
        try {
            const mintPublicKey = new PublicKey(mintAddress);
            const destinationPublicKey = new PublicKey(destinationAddress);

            const sourceTokenAccount = await Token.getAssociatedTokenAddress(
                ASSOCIATED_TOKEN_PROGRAM_ID,
                TOKEN_PROGRAM_ID,
                mintPublicKey,
                this.wallet.publicKey
            );

            const destinationTokenAccount = await Token.getAssociatedTokenAddress(
                ASSOCIATED_TOKEN_PROGRAM_ID,
                TOKEN_PROGRAM_ID,
                mintPublicKey,
                destinationPublicKey
            );

            const tokenInfo = this.tokenAccounts.get(mintAddress);
            const transferAmount = amount * Math.pow(10, tokenInfo?.decimals || 9);

            const transaction = new Transaction().add(
                Token.createTransferInstruction(
                    TOKEN_PROGRAM_ID,
                    sourceTokenAccount,
                    destinationTokenAccount,
                    this.wallet.publicKey,
                    [],
                    transferAmount
                )
            );

            const signature = await this.sendTransaction(transaction);
            return await this.waitForTransaction(signature);
        } catch (error) {
            throw new Error(`Failed to transfer token: ${error.message}`);
        }
    }

    async getBlockNumber() {
        return await this.connection.getSlot();
    }

    async getBlock(slot = null) {
        const targetSlot = slot || await this.connection.getSlot();
        return await this.connection.getBlock(targetSlot);
    }

    async getTransaction(signature) {
        return await this.connection.getTransaction(signature);
    }

    async getLogs(programId, commitment = 'confirmed') {
        const logs = await this.connection.onLogs(
            new PublicKey(programId),
            (logs) => {
                this.emit('logs', logs);
            },
            commitment
        );
        return logs;
    }

    async getTokenInfo(mintAddress) {
        try {
            const mintPublicKey = new PublicKey(mintAddress);
            const mintInfo = await this.connection.getParsedAccountInfo(mintPublicKey);

            if (!mintInfo.value) {
                throw new Error('Token not found');
            }

            const data = mintInfo.value.data.parsed.info;
            return {
                address: mintAddress,
                decimals: data.decimals,
                supply: data.supply,
                mintAuthority: data.mintAuthority,
                freezeAuthority: data.freezeAuthority
            };
        } catch (error) {
            throw new Error(`Failed to get token info: ${error.message}`);
        }
    }

    async getNetworkInfo() {
        try {
            const slot = await this.connection.getSlot();
            const blockTime = await this.connection.getBlockTime(slot);
            const version = await this.connection.getVersion();

            return {
                slot,
                blockTime,
                version,
                cluster: this.config.cluster || 'mainnet-beta'
            };
        } catch (error) {
            throw new Error(`Failed to get network info: ${error.message}`);
        }
    }

    async subscribeToBlocks(callback) {
        const subscription = this.connection.onSlotChange((slotInfo) => {
            callback(slotInfo);
        });
        
        return subscription;
    }

    async subscribeToTokenAccount(tokenAccountAddress, callback) {
        const subscription = this.connection.onAccountChange(
            new PublicKey(tokenAccountAddress),
            (accountInfo) => {
                callback(accountInfo);
            }
        );
        
        return subscription;
    }

    async getTokenAccounts(ownerAddress = null) {
        try {
            const owner = ownerAddress ? new PublicKey(ownerAddress) : this.wallet.publicKey;
            
            const tokenAccounts = await this.connection.getParsedTokenAccountsByOwner(
                owner,
                { programId: TOKEN_PROGRAM_ID }
            );

            return tokenAccounts.value.map(({ account, pubkey }) => ({
                address: pubkey.toString(),
                mint: account.data.parsed.info.mint,
                balance: account.data.parsed.info.tokenAmount.uiAmount,
                decimals: account.data.parsed.info.tokenAmount.decimals
            }));
        } catch (error) {
            throw new Error(`Failed to get token accounts: ${error.message}`);
        }
    }

    async simulateSwap(inputMint, outputMint, amount) {
        try {
            const routes = await this.jupiter.computeRoutes({
                inputMint: new PublicKey(inputMint),
                outputMint: new PublicKey(outputMint),
                amount: amount,
                slippageBps: 50
            });

            if (routes.routesInfos.length === 0) return 0;

            const bestRoute = routes.routesInfos[0];
            return bestRoute.outAmount / amount;
        } catch (error) {
            return 0;
        }
    }

    async getMarketData(mintAddress) {
        try {
            const response = await fetch(`https://api.coingecko.com/api/v3/coins/solana/contract/${mintAddress}`);
            const data = await response.json();
            
            return {
                price: data.market_data?.current_price?.usd || 0,
                marketCap: data.market_data?.market_cap?.usd || 0,
                volume24h: data.market_data?.total_volume?.usd || 0,
                priceChange24h: data.market_data?.price_change_percentage_24h || 0
            };
        } catch (error) {
            return { price: 0, marketCap: 0, volume24h: 0, priceChange24h: 0 };
        }
    }

    async executeArbitrageRoute(route, amount) {
        try {
            const transactions = [];
            let currentAmount = amount;

            for (let i = 0; i < route.length - 1; i++) {
                const inputMint = route[i];
                const outputMint = route[i + 1];

                const routes = await this.jupiter.computeRoutes({
                    inputMint: new PublicKey(inputMint),
                    outputMint: new PublicKey(outputMint),
                    amount: currentAmount,
                    slippageBps: 50
                });

                if (routes.routesInfos.length === 0) {
                    throw new Error(`No route found for ${inputMint} -> ${outputMint}`);
                }

                const { execute } = await this.jupiter.exchange({
                    routeInfo: routes.routesInfos[0]
                });

                const swapResult = await execute();
                transactions.push(swapResult);
                currentAmount = routes.routesInfos[0].outAmount;
            }

            return {
                success: true,
                transactions,
                finalAmount: currentAmount,
                profit: currentAmount - amount
            };
        } catch (error) {
            throw new Error(`Failed to execute arbitrage route: ${error.message}`);
        }
    }

    async getRecentPriorityFees() {
        try {
            const recentSlots = await this.connection.getRecentPrioritizationFees({
                lockedWritableAccounts: [this.wallet.publicKey]
            });

            if (recentSlots.length === 0) return 0;

            const fees = recentSlots.map(slot => slot.prioritizationFee);
            const averageFee = fees.reduce((sum, fee) => sum + fee, 0) / fees.length;
            
            return Math.ceil(averageFee);
        } catch (error) {
            return 1000;
        }
    }

    async batchTransactions(transactions) {
        try {
            const signatures = [];
            
            for (const transaction of transactions) {
                const signature = await this.sendTransaction(transaction);
                signatures.push(signature);
                
                await new Promise(resolve => setTimeout(resolve, this.blockTime));
            }

            const confirmations = await Promise.all(
                signatures.map(sig => this.waitForTransaction(sig))
            );

            return {
                signatures,
                confirmations,
                success: confirmations.every(conf => !conf.value.err)
            };
        } catch (error) {
            throw new Error(`Failed to batch transactions: ${error.message}`);
        }
    }

    disconnect() {
        this.connection = null;
        this.emit('disconnected');
    }

    getAddress() {
        return this.wallet?.publicKey.toString() || null;
    }

    async sign(message) {
        if (!this.wallet) throw new Error('No wallet available');
        
        const messageBytes = Buffer.from(message, 'utf8');
        const signature = this.wallet.secretKey.slice(32);
        
        return signature;
    }

    formatUnits(value, decimals = 9) {
        return value / Math.pow(10, decimals);
    }

    parseUnits(value, decimals = 9) {
        return Math.floor(value * Math.pow(10, decimals));
    }

    isAddress(address) {
        try {
            new PublicKey(address);
            return true;
        } catch {
            return false;
        }
    }
}

module.exports = SolanaConnector;