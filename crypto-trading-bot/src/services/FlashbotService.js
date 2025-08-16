const EventEmitter = require('events');
const { ethers } = require('ethers');
const { FlashbotsBundleProvider, FlashbotsBundleResolution } = require('@flashbots/ethers-provider-bundle');

class FlashbotService extends EventEmitter {
    constructor(config, provider, authSigner) {
        super();
        this.config = config;
        this.provider = provider;
        this.authSigner = authSigner;
        this.flashbotsProvider = null;
        this.bundleQueue = new Map();
        this.bundleHistory = new Map();
        this.relayEndpoints = [
            'https://relay.flashbots.net',
            'https://rpc.beaverbuild.org',
            'https://builder0x69.io',
            'https://api.blocknative.com/v1/auction'
        ];
        this.activeBundles = new Map();
        this.bundleStats = {
            submitted: 0,
            included: 0,
            failed: 0,
            totalProfit: ethers.BigNumber.from(0)
        };
    }

    async initialize() {
        try {
            this.flashbotsProvider = await FlashbotsBundleProvider.create(
                this.provider,
                this.authSigner,
                this.config.flashbotsRelay || this.relayEndpoints[0],
                'mainnet'
            );
            
            this.emit('initialized');
        } catch (error) {
            this.emit('error', `Failed to initialize Flashbots: ${error.message}`);
            throw error;
        }
    }

    async sendBundle(transactions, targetBlockNumber, options = {}) {
        try {
            const bundleId = `bundle_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            
            const bundle = transactions.map(tx => {
                if (typeof tx === 'string') {
                    return { signedTransaction: tx };
                }
                return tx;
            });

            const bundleRequest = {
                transactions: bundle,
                blockNumber: targetBlockNumber,
                minTimestamp: options.minTimestamp,
                maxTimestamp: options.maxTimestamp,
                revertingTxHashes: options.revertingTxHashes || []
            };

            this.bundleQueue.set(bundleId, {
                ...bundleRequest,
                id: bundleId,
                createdAt: Date.now(),
                status: 'pending',
                attempts: 0
            });