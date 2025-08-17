use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use rayon::prelude::*;
use dashmap::DashMap;
use parking_lot::Mutex;
use crossbeam::channel::{bounded, Receiver, Sender};
use futures::stream::{FuturesUnordered, StreamExt};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use reqwest::Client;
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangePrice {
    pub exchange: String,
    pub symbol: String,
    pub bid: f64,
    pub ask: f64,
    pub volume: f64,
    pub timestamp: u64,
    pub chain_id: u32,
    pub is_dex: bool,
    pub liquidity_usd: f64,
    pub gas_price_gwei: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    pub id: String,
    pub strategy: String,
    pub symbol: String,
    pub buy_exchange: String,
    pub sell_exchange: String,
    pub buy_price: f64,
    pub sell_price: f64,
    pub spread: f64,
    pub profit_estimate: f64,
    pub volume: f64,
    pub confidence: f64,
    pub timestamp: u64,
    pub execution_time_ms: u64,
    pub gas_cost_eth: f64,
    pub net_profit: f64,
    pub chain_from: String,
    pub chain_to: String,
    pub flash_loan_available: bool,
    pub mev_risk: f64,
}

#[derive(Debug, Clone)]
pub struct ExchangeConfig {
    pub name: String,
    pub api_url: String,
    pub websocket_url: String,
    pub fee_rate: f64,
    pub rate_limit: u32,
    pub chain: String,
    pub supports_flash_loans: bool,
}

pub struct UltraFastArbitrageEngine {
    price_cache: Arc<DashMap<String, ExchangePrice>>,
    opportunities: Arc<DashMap<String, ArbitrageOpportunity>>,
    exchange_configs: HashMap<String, ExchangeConfig>,
    opportunity_sender: Sender<ArbitrageOpportunity>,
    opportunity_receiver: Receiver<ArbitrageOpportunity>,
    total_opportunities: Arc<Mutex<u64>>,
    total_profit: Arc<Mutex<f64>>,
    http_client: Client,
    gas_prices: Arc<DashMap<String, f64>>,
    token_prices: Arc<DashMap<String, f64>>,
    min_profit: f64,
    max_slippage: f64,
}

impl UltraFastArbitrageEngine {
    pub fn new(min_profit: f64, max_slippage: f64) -> Self {
        let (sender, receiver) = bounded(100000);
        
        let mut exchange_configs = HashMap::new();
        
        // Tier 1 CEX Configurations
        exchange_configs.insert("binance".to_string(), ExchangeConfig {
            name: "binance".to_string(),
            api_url: "https://api.binance.com/api/v3".to_string(),
            websocket_url: "wss://stream.binance.com:9443/ws".to_string(),
            fee_rate: 0.001,
            rate_limit: 1200,
            chain: "cex".to_string(),
            supports_flash_loans: false,
        });
        
        exchange_configs.insert("okx".to_string(), ExchangeConfig {
            name: "okx".to_string(),
            api_url: "https://www.okx.com/api/v5".to_string(),
            websocket_url: "wss://ws.okx.com:8443/ws/v5/public".to_string(),
            fee_rate: 0.001,
            rate_limit: 600,
            chain: "cex".to_string(),
            supports_flash_loans: false,
        });
        
        exchange_configs.insert("coinbase".to_string(), ExchangeConfig {
            name: "coinbase".to_string(),
            api_url: "https://api.exchange.coinbase.com".to_string(),
            websocket_url: "wss://ws-feed.exchange.coinbase.com".to_string(),
            fee_rate: 0.005,
            rate_limit: 10,
            chain: "cex".to_string(),
            supports_flash_loans: false,
        });
        
        exchange_configs.insert("bybit".to_string(), ExchangeConfig {
            name: "bybit".to_string(),
            api_url: "https://api.bybit.com/v5".to_string(),
            websocket_url: "wss://stream.bybit.com/v5/public/spot".to_string(),
            fee_rate: 0.001,
            rate_limit: 120,
            chain: "cex".to_string(),
            supports_flash_loans: false,
        });
        
        exchange_configs.insert("kucoin".to_string(), ExchangeConfig {
            name: "kucoin".to_string(),
            api_url: "https://api.kucoin.com/api/v1".to_string(),
            websocket_url: "wss://ws-api.kucoin.com/endpoint".to_string(),
            fee_rate: 0.001,
            rate_limit: 45,
            chain: "cex".to_string(),
            supports_flash_loans: false,
        });
        
        // Ethereum DEX Configurations
        exchange_configs.insert("uniswap_v3".to_string(), ExchangeConfig {
            name: "uniswap_v3".to_string(),
            api_url: "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3".to_string(),
            websocket_url: "".to_string(),
            fee_rate: 0.003,
            rate_limit: 1000,
            chain: "ethereum".to_string(),
            supports_flash_loans: true,
        });
        
        exchange_configs.insert("sushiswap".to_string(), ExchangeConfig {
            name: "sushiswap".to_string(),
            api_url: "https://api.thegraph.com/subgraphs/name/sushiswap/exchange".to_string(),
            websocket_url: "".to_string(),
            fee_rate: 0.003,
            rate_limit: 1000,
            chain: "ethereum".to_string(),
            supports_flash_loans: true,
        });
        
        exchange_configs.insert("curve".to_string(), ExchangeConfig {
            name: "curve".to_string(),
            api_url: "https://api.curve.fi/api/getPools/all".to_string(),
            websocket_url: "".to_string(),
            fee_rate: 0.0004,
            rate_limit: 500,
            chain: "ethereum".to_string(),
            supports_flash_loans: true,
        });
        
        // BSC DEX Configurations
        exchange_configs.insert("pancakeswap".to_string(), ExchangeConfig {
            name: "pancakeswap".to_string(),
            api_url: "https://api.thegraph.com/subgraphs/name/pancakeswap/exchange".to_string(),
            websocket_url: "".to_string(),
            fee_rate: 0.0025,
            rate_limit: 1000,
            chain: "bsc".to_string(),
            supports_flash_loans: true,
        });
        
        // Polygon DEX Configurations
        exchange_configs.insert("quickswap".to_string(), ExchangeConfig {
            name: "quickswap".to_string(),
            api_url: "https://api.thegraph.com/subgraphs/name/sameepsi/quickswap06".to_string(),
            websocket_url: "".to_string(),
            fee_rate: 0.003,
            rate_limit: 1000,
            chain: "polygon".to_string(),
            supports_flash_loans: true,
        });
        
        // Arbitrum DEX Configurations
        exchange_configs.insert("uniswap_arbitrum".to_string(), ExchangeConfig {
            name: "uniswap_arbitrum".to_string(),
            api_url: "https://api.thegraph.com/subgraphs/name/ianlapham/arbitrum-minimal".to_string(),
            websocket_url: "".to_string(),
            fee_rate: 0.003,
            rate_limit: 1000,
            chain: "arbitrum".to_string(),
            supports_flash_loans: true,
        });
        
        Self {
            price_cache: Arc::new(DashMap::new()),
            opportunities: Arc::new(DashMap::new()),
            exchange_configs,
            opportunity_sender: sender,
            opportunity_receiver: receiver,
            total_opportunities: Arc::new(Mutex::new(0)),
            total_profit: Arc::new(Mutex::new(0.0)),
            http_client: Client::new(),
            gas_prices: Arc::new(DashMap::new()),
            token_prices: Arc::new(DashMap::new()),
            min_profit,
            max_slippage,
        }
    }
    
    pub async fn start_scanning(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("🦀 Starting Ultra-Fast Rust Arbitrage Engine");
        
        let mut futures = FuturesUnordered::new();
        
        // Price monitoring workers
        for worker_id in 0..32 {
            let price_cache = self.price_cache.clone();
            let exchange_configs = self.exchange_configs.clone();
            let http_client = self.http_client.clone();
            
            futures.push(tokio::spawn(async move {
                Self::price_monitoring_worker(worker_id, price_cache, exchange_configs, http_client).await
            }));
        }
        
        // Arbitrage scanning workers
        for worker_id in 0..16 {
            let price_cache = self.price_cache.clone();
            let opportunities = self.opportunities.clone();
            let sender = self.opportunity_sender.clone();
            let total_opps = self.total_opportunities.clone();
            let total_profit = self.total_profit.clone();
            let min_profit = self.min_profit;
            let max_slippage = self.max_slippage;
            
            futures.push(tokio::spawn(async move {
                Self::arbitrage_scanning_worker(
                    worker_id, price_cache, opportunities, sender, 
                    total_opps, total_profit, min_profit, max_slippage
                ).await
            }));
        }
        
        // Opportunity processing worker
        let receiver = self.opportunity_receiver.clone();
        futures.push(tokio::spawn(async move {
            Self::opportunity_processing_worker(receiver).await
        }));
        
        // Gas price monitoring
        let gas_prices = self.gas_prices.clone();
        let http_client = self.http_client.clone();
        futures.push(tokio::spawn(async move {
            Self::gas_price_monitoring_worker(gas_prices, http_client).await
        }));
        
        while let Some(result) = futures.next().await {
            if let Err(e) = result {
                eprintln!("Worker error: {:?}", e);
            }
        }
        
        Ok(())
    }
    
    async fn price_monitoring_worker(
        worker_id: usize,
        price_cache: Arc<DashMap<String, ExchangePrice>>,
        exchange_configs: HashMap<String, ExchangeConfig>,
        http_client: Client,
    ) {
        let mut iteration = 0u64;
        
        loop {
            let start_time = Instant::now();
            
            // Parallel price fetching for all exchanges
            let exchange_futures: Vec<_> = exchange_configs
                .iter()
                .map(|(name, config)| {
                    let client = http_client.clone();
                    let cache = price_cache.clone();
                    let exchange_name = name.clone();
                    let exchange_config = config.clone();
                    
                    tokio::spawn(async move {
                        if let Ok(prices) = Self::fetch_exchange_prices(&client, &exchange_config).await {
                            for price in prices {
                                let key = format!("{}:{}", exchange_name, price.symbol);
                                cache.insert(key, price);
                            }
                        }
                    })
                })
                .collect();
            
            futures::future::join_all(exchange_futures).await;
            
            iteration += 1;
            let elapsed = start_time.elapsed();
            
            if iteration % 100 == 0 {
                println!("🔥 Price Worker {}: Iteration {} - {} prices cached in {:?}", 
                        worker_id, iteration, price_cache.len(), elapsed);
            }
            
            // Maintain 10ms cycle time
            if elapsed < Duration::from_millis(10) {
                tokio::time::sleep(Duration::from_millis(10) - elapsed).await;
            }
        }
    }
    
    async fn fetch_exchange_prices(
        client: &Client,
        config: &ExchangeConfig,
    ) -> Result<Vec<ExchangePrice>, Box<dyn std::error::Error + Send + Sync>> {
        let mut prices = Vec::new();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        
        match config.name.as_str() {
            "binance" => {
                let url = format!("{}/ticker/24hr", config.api_url);
                let response: Value = client.get(&url).send().await?.json().await?;
                
                if let Some(tickers) = response.as_array() {
                    for ticker in tickers.iter().take(200) {
                        if let (Some(symbol), Some(bid), Some(ask), Some(volume)) = (
                            ticker["symbol"].as_str(),
                            ticker["bidPrice"].as_str().and_then(|s| s.parse::<f64>().ok()),
                            ticker["askPrice"].as_str().and_then(|s| s.parse::<f64>().ok()),
                            ticker["volume"].as_str().and_then(|s| s.parse::<f64>().ok()),
                        ) {
                            if symbol.ends_with("USDT") && bid > 0.0 && ask > 0.0 && volume > 0.0 {
                                prices.push(ExchangePrice {
                                    exchange: config.name.clone(),
                                    symbol: symbol.to_string(),
                                    bid,
                                    ask,
                                    volume,
                                    timestamp,
                                    chain_id: 0, // CEX
                                    is_dex: false,
                                    liquidity_usd: volume * (bid + ask) / 2.0,
                                    gas_price_gwei: 0.0,
                                });
                            }
                        }
                    }
                }
            },
            "okx" => {
                let url = format!("{}/market/tickers?instType=SPOT", config.api_url);
                let response: Value = client.get(&url).send().await?.json().await?;
                
                if let Some(data) = response["data"].as_array() {
                    for ticker in data.iter().take(200) {
                        if let (Some(symbol), Some(bid), Some(ask), Some(volume)) = (
                            ticker["instId"].as_str(),
                            ticker["bidPx"].as_str().and_then(|s| s.parse::<f64>().ok()),
                            ticker["askPx"].as_str().and_then(|s| s.parse::<f64>().ok()),
                            ticker["vol24h"].as_str().and_then(|s| s.parse::<f64>().ok()),
                        ) {
                            if symbol.ends_with("-USDT") && bid > 0.0 && ask > 0.0 && volume > 0.0 {
                                prices.push(ExchangePrice {
                                    exchange: config.name.clone(),
                                    symbol: symbol.replace("-", "/"),
                                    bid,
                                    ask,
                                    volume,
                                    timestamp,
                                    chain_id: 0, // CEX
                                    is_dex: false,
                                    liquidity_usd: volume * (bid + ask) / 2.0,
                                    gas_price_gwei: 0.0,
                                });
                            }
                        }
                    }
                }
            },
            "uniswap_v3" => {
                // Generate mock DEX prices for demonstration
                let base_tokens = ["WETH", "WBTC", "USDC", "DAI", "LINK", "UNI", "AAVE", "CRV"];
                let base_prices = [2500.0, 43000.0, 1.0, 1.0, 15.0, 6.0, 85.0, 0.75];
                
                for (i, token) in base_tokens.iter().enumerate() {
                    let base_price = base_prices[i];
                    let spread = base_price * 0.003; // 0.3% spread for DEX
                    
                    prices.push(ExchangePrice {
                        exchange: config.name.clone(),
                        symbol: format!("{}/USDT", token),
                        bid: base_price - spread / 2.0,
                        ask: base_price + spread / 2.0,
                        volume: 1000000.0 + (i as f64 * 50000.0),
                        timestamp,
                        chain_id: 1, // Ethereum
                        is_dex: true,
                        liquidity_usd: 5000000.0 + (i as f64 * 200000.0),
                        gas_price_gwei: 30.0,
                    });
                }
            },
            _ => {
                // Generate mock prices for other exchanges
                Self::generate_mock_prices(config, timestamp, &mut prices);
            }
        }
        
        Ok(prices)
    }
    
    fn generate_mock_prices(config: &ExchangeConfig, timestamp: u64, prices: &mut Vec<ExchangePrice>) {
        let symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT", "AVAX/USDT", "MATIC/USDT", "DOT/USDT"];
        let base_prices = [43000.0, 2500.0, 300.0, 0.5, 100.0, 35.0, 0.8, 7.0];
        
        for (i, (symbol, base_price)) in symbols.iter().zip(base_prices.iter()).enumerate() {
            let variation = (config.name.len() as f64 * 0.001) + (i as f64 * 0.0005);
            let price_multiplier = 1.0 + (variation - 0.003);
            let adjusted_price = base_price * price_multiplier;
            
            let spread_rate = if config.is_dex { 0.003 } else { 0.001 };
            let spread = adjusted_price * spread_rate;
            
            let chain_id = match config.chain.as_str() {
                "ethereum" => 1,
                "bsc" => 56,
                "polygon" => 137,
                "arbitrum" => 42161,
                "optimism" => 10,
                "avalanche" => 43114,
                _ => 0,
            };
            
            prices.push(ExchangePrice {
                exchange: config.name.clone(),
                symbol: symbol.to_string(),
                bid: adjusted_price - spread / 2.0,
                ask: adjusted_price + spread / 2.0,
                volume: 500000.0 + (i as f64 * 100000.0),
                timestamp,
                chain_id,
                is_dex: config.supports_flash_loans,
                liquidity_usd: 2000000.0 + (i as f64 * 150000.0),
                gas_price_gwei: if chain_id > 0 { 25.0 } else { 0.0 },
            });
        }
    }
    
    async fn arbitrage_scanning_worker(
        worker_id: usize,
        price_cache: Arc<DashMap<String, ExchangePrice>>,
        opportunities: Arc<DashMap<String, ArbitrageOpportunity>>,
        sender: Sender<ArbitrageOpportunity>,
        total_opps: Arc<Mutex<u64>>,
        total_profit: Arc<Mutex<f64>>,
        min_profit: f64,
        max_slippage: f64,
    ) {
        let mut iteration = 0u64;
        
        loop {
            let start_time = Instant::now();
            
            // Extract current prices
            let current_prices: Vec<(String, ExchangePrice)> = price_cache
                .iter()
                .map(|entry| (entry.key().clone(), entry.value().clone()))
                .collect();
            
            if current_prices.len() < 2 {
                tokio::time::sleep(Duration::from_millis(10)).await;
                continue;
            }
            
            // Parallel arbitrage detection
            let detected_opportunities: Vec<ArbitrageOpportunity> = current_prices
                .par_iter()
                .flat_map(|(key1, price1)| {
                    current_prices.par_iter().filter_map(move |(key2, price2)| {
                        if key1 >= key2 || price1.symbol != price2.symbol {
                            return None;
                        }
                        
                        Self::detect_cross_exchange_opportunity(price1, price2, min_profit, max_slippage)
                    })
                })
                .chain(Self::detect_triangular_opportunities(&current_prices, min_profit))
                .chain(Self::detect_flash_loan_opportunities(&current_prices, min_profit, max_slippage))
                .chain(Self::detect_cross_chain_opportunities(&current_prices, min_profit))
                .collect();
            
            // Process detected opportunities
            for opportunity in detected_opportunities {
                if opportunity.confidence > 0.7 && opportunity.net_profit > min_profit {
                    let opp_id = opportunity.id.clone();
                    
                    // Send to processing queue
                    let _ = sender.try_send(opportunity.clone());
                    
                    // Cache opportunity
                    opportunities.insert(opp_id, opportunity.clone());
                    
                    // Update statistics
                    {
                        let mut total_count = total_opps.lock();
                        *total_count += 1;
                    }
                    {
                        let mut total_p = total_profit.lock();
                        *total_p += opportunity.net_profit;
                    }
                }
            }
            
            iteration += 1;
            let elapsed = start_time.elapsed();
            
            if iteration % 1000 == 0 {
                let total_count = *total_opps.lock();
                let total_p = *total_profit.lock();
                println!("⚡ Scan Worker {}: Iteration {} - Found {} opportunities worth ${:.2} in {:?}", 
                        worker_id, iteration, total_count, total_p, elapsed);
            }
            
            // Maintain microsecond-level scanning
            if elapsed < Duration::from_micros(500) {
                tokio::time::sleep(Duration::from_micros(500) - elapsed).await;
            }
        }
    }
    
    fn detect_cross_exchange_opportunity(
        price1: &ExchangePrice,
        price2: &ExchangePrice,
        min_profit: f64,
        max_slippage: f64,
    ) -> Option<ArbitrageOpportunity> {
        let spread = if price1.ask < price2.bid {
            (price2.bid - price1.ask) / price1.ask
        } else if price2.ask < price1.bid {
            (price1.bid - price2.ask) / price2.ask
        } else {
            return None;
        };
        
        if spread < 0.0001 {
            return None;
        }
        
        let trade_size = 25000.0;
        let gross_profit = trade_size * spread;
        
        // Calculate fees and costs
        let trading_fees = trade_size * 0.002; // 0.2% total trading fees
        let gas_cost = Self::calculate_gas_cost(&price1.chain_id, &price2.chain_id, trade_size);
        let slippage_cost = Self::calculate_slippage_impact(trade_size, price1.liquidity_usd, price2.liquidity_usd);
        
        let net_profit = gross_profit - trading_fees - gas_cost - slippage_cost;
        
        if net_profit < min_profit || slippage_cost / trade_size > max_slippage {
            return None;
        }
        
        let confidence = Self::calculate_confidence_score(spread, price1.volume.min(price2.volume), slippage_cost / trade_size);
        
        Some(ArbitrageOpportunity {
            id: format!("cross_{}_{}_{}_{}", 
                       price1.exchange, price2.exchange, price1.symbol.replace("/", "_"),
                       SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()),
            strategy: "cross_exchange".to_string(),
            symbol: price1.symbol.clone(),
            buy_exchange: if price1.ask < price2.bid { price1.exchange.clone() } else { price2.exchange.clone() },
            sell_exchange: if price1.ask < price2.bid { price2.exchange.clone() } else { price1.exchange.clone() },
            buy_price: if price1.ask < price2.bid { price1.ask } else { price2.ask },
            sell_price: if price1.ask < price2.bid { price2.bid } else { price1.bid },
            spread: spread * 100.0,
            profit_estimate: gross_profit,
            volume: price1.volume.min(price2.volume),
            confidence,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            execution_time_ms: 100,
            gas_cost_eth: gas_cost / 2500.0, // Assuming ETH price
            net_profit,
            chain_from: Self::chain_id_to_name(price1.chain_id),
            chain_to: Self::chain_id_to_name(price2.chain_id),
            flash_loan_available: price1.is_dex || price2.is_dex,
            mev_risk: if price1.is_dex || price2.is_dex { 0.1 } else { 0.0 },
        })
    }
    
    fn detect_triangular_opportunities(
        prices: &[(String, ExchangePrice)],
        min_profit: f64,
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        // Group by exchange
        let mut exchange_groups: HashMap<String, Vec<&ExchangePrice>> = HashMap::new();
        for (_, price) in prices {
            exchange_groups.entry(price.exchange.clone()).or_insert_with(Vec::new).push(price);
        }
        
        // Find triangular opportunities within each exchange
        for (exchange, exchange_prices) in exchange_groups {
            if exchange_prices.len() < 3 {
                continue;
            }
            
            // Define common triangular paths
            let triangular_paths = [
                ("BTC/USDT", "ETH/BTC", "ETH/USDT"),
                ("BNB/USDT", "ETH/BNB", "ETH/USDT"),
                ("ADA/USDT", "ETH/ADA", "ETH/USDT"),
                ("SOL/USDT", "ETH/SOL", "ETH/USDT"),
                ("AVAX/USDT", "ETH/AVAX", "ETH/USDT"),
            ];
            
            for (pair1, pair2, pair3) in &triangular_paths {
                let price1 = exchange_prices.iter().find(|p| &p.symbol == pair1);
                let price2 = exchange_prices.iter().find(|p| &p.symbol == pair2);
                let price3 = exchange_prices.iter().find(|p| &p.symbol == pair3);
                
                if let (Some(p1), Some(p2), Some(p3)) = (price1, price2, price3) {
                    let rate1 = p1.bid;
                    let rate2 = p2.bid;
                    let rate3 = 1.0 / p3.ask;
                    
                    let final_rate = rate1 * rate2 * rate3;
                    let profit_rate = final_rate - 1.0;
                    
                    if profit_rate > 0.0005 {
                        let trade_size = 15000.0;
                        let gross_profit = trade_size * profit_rate;
                        let fees = trade_size * 0.003; // 3 trades * 0.1% each
                        let gas_cost = if p1.is_dex { 150.0 } else { 0.0 };
                        let net_profit = gross_profit - fees - gas_cost;
                        
                        if net_profit > min_profit {
                            opportunities.push(ArbitrageOpportunity {
                                id: format!("tri_{}_{}_{}_{}_{}", 
                                           exchange, pair1.replace("/", "_"), pair2.replace("/", "_"), pair3.replace("/", "_"),
                                           SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()),
                                strategy: "triangular".to_string(),
                                symbol: format!("{}-{}-{}", pair1, pair2, pair3),
                                buy_exchange: exchange.clone(),
                                sell_exchange: exchange.clone(),
                                buy_price: rate1,
                                sell_price: final_rate,
                                spread: profit_rate * 100.0,
                                profit_estimate: gross_profit,
                                volume: p1.volume.min(p2.volume).min(p3.volume),
                                confidence: 0.8,
                                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                                execution_time_ms: 150,
                                gas_cost_eth: gas_cost / 2500.0,
                                net_profit,
                                chain_from: Self::chain_id_to_name(p1.chain_id),
                                chain_to: Self::chain_id_to_name(p1.chain_id),
                                flash_loan_available: p1.is_dex,
                                mev_risk: if p1.is_dex { 0.15 } else { 0.0 },
                            });
                        }
                    }
                }
            }
        }
        
        opportunities
    }
    
    fn detect_flash_loan_opportunities(
        prices: &[(String, ExchangePrice)],
        min_profit: f64,
        max_slippage: f64,
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        // Separate DEX and CEX prices
        let dex_prices: Vec<&ExchangePrice> = prices.iter().map(|(_, p)| p).filter(|p| p.is_dex).collect();
        let cex_prices: Vec<&ExchangePrice> = prices.iter().map(|(_, p)| p).filter(|p| !p.is_dex).collect();
        
        for dex_price in &dex_prices {
            for cex_price in &cex_prices {
                if dex_price.symbol == cex_price.symbol && dex_price.chain_id == 1 {
                    let spread = if dex_price.ask < cex_price.bid {
                        (cex_price.bid - dex_price.ask) / dex_price.ask
                    } else if cex_price.ask < dex_price.bid {
                        (dex_price.bid - cex_price.ask) / cex_price.ask
                    } else {
                        continue;
                    };
                    
                    if spread < 0.002 {
                        continue;
                    }
                    
                    let flash_loan_amount = 100000.0;
                    let gross_profit = flash_loan_amount * spread;
                    let flash_loan_fee = flash_loan_amount * 0.0005; // 0.05% Aave fee
                    let gas_cost = 300.0; // High gas for complex flash loan
                    let slippage_cost = Self::calculate_slippage_impact(flash_loan_amount, dex_price.liquidity_usd, cex_price.liquidity_usd);
                    
                    let net_profit = gross_profit - flash_loan_fee - gas_cost - slippage_cost;
                    
                    if net_profit > min_profit * 5.0 && slippage_cost / flash_loan_amount < max_slippage {
                        let confidence = Self::calculate_confidence_score(spread, flash_loan_amount, slippage_cost / flash_loan_amount);
                        
                        opportunities.push(ArbitrageOpportunity {
                            id: format!("flash_{}_{}_{}_{}", 
                                       dex_price.exchange, cex_price.exchange, dex_price.symbol.replace("/", "_"),
                                       SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()),
                            strategy: "flash_loan".to_string(),
                            symbol: dex_price.symbol.clone(),
                            buy_exchange: if dex_price.ask < cex_price.bid { dex_price.exchange.clone() } else { cex_price.exchange.clone() },
                            sell_exchange: if dex_price.ask < cex_price.bid { cex_price.exchange.clone() } else { dex_price.exchange.clone() },
                            buy_price: if dex_price.ask < cex_price.bid { dex_price.ask } else { cex_price.ask },
                            sell_price: if dex_price.ask < cex_price.bid { cex_price.bid } else { dex_price.bid },
                            spread: spread * 100.0,
                            profit_estimate: gross_profit,
                            volume: flash_loan_amount,
                            confidence,
                            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                            execution_time_ms: 300,
                            gas_cost_eth: gas_cost / 2500.0,
                            net_profit,
                            chain_from: Self::chain_id_to_name(dex_price.chain_id),
                            chain_to: "cex".to_string(),
                            flash_loan_available: true,
                            mev_risk: 0.2,
                        });
                    }
                }
            }
        }
        
        opportunities
    }
    
    fn detect_cross_chain_opportunities(
        prices: &[(String, ExchangePrice)],
        min_profit: f64,
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        // Group prices by symbol and chain
        let mut symbol_chain_map: HashMap<String, HashMap<u32, Vec<&ExchangePrice>>> = HashMap::new();
        
        for (_, price) in prices {
            if price.is_dex {
                symbol_chain_map
                    .entry(price.symbol.clone())
                    .or_insert_with(HashMap::new)
                    .entry(price.chain_id)
                    .or_insert_with(Vec::new)
                    .push(price);
            }
        }
        
        // Find cross-chain opportunities
        for (symbol, chain_map) in symbol_chain_map {
            if chain_map.len() < 2 {
                continue;
            }
            
            let chains: Vec<u32> = chain_map.keys().cloned().collect();
            
            for i in 0..chains.len() {
                for j in i + 1..chains.len() {
                    let chain1 = chains[i];
                    let chain2 = chains[j];
                    
                    if let (Some(prices1), Some(prices2)) = (chain_map.get(&chain1), chain_map.get(&chain2)) {
                        // Find best prices on each chain
                        let best_ask1 = prices1.iter().min_by(|a, b| a.ask.partial_cmp(&b.ask).unwrap());
                        let best_bid2 = prices2.iter().max_by(|a, b| a.bid.partial_cmp(&b.bid).unwrap());
                        
                        if let (Some(ask_price), Some(bid_price)) = (best_ask1, best_bid2) {
                            if ask_price.ask < bid_price.bid {
                                let spread = (bid_price.bid - ask_price.ask) / ask_price.ask;
                                
                                if spread > 0.005 { // Higher threshold for cross-chain
                                    let trade_size = 50000.0;
                                    let gross_profit = trade_size * spread;
                                    let bridge_fee = trade_size * 0.001; // 0.1% bridge fee
                                    let gas_cost = 200.0; // Gas on both chains
                                    let time_cost = trade_size * 0.0001; // Time value cost
                                    
                                    let net_profit = gross_profit - bridge_fee - gas_cost - time_cost;
                                    
                                    if net_profit > min_profit * 10.0 {
                                        opportunities.push(ArbitrageOpportunity {
                                            id: format!("crosschain_{}_{}_{}_{}_{}", 
                                                       Self::chain_id_to_name(chain1), Self::chain_id_to_name(chain2), 
                                                       symbol.replace("/", "_"), ask_price.exchange, bid_price.exchange),
                                            strategy: "cross_chain".to_string(),
                                            symbol: symbol.clone(),
                                            buy_exchange: ask_price.exchange.clone(),
                                            sell_exchange: bid_price.exchange.clone(),
                                            buy_price: ask_price.ask,
                                            sell_price: bid_price.bid,
                                            spread: spread * 100.0,
                                            profit_estimate: gross_profit,
                                            volume: trade_size,
                                            confidence: 0.7, // Lower confidence for cross-chain
                                            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                                            execution_time_ms: 30000, // 30 seconds for cross-chain
                                            gas_cost_eth: gas_cost / 2500.0,
                                            net_profit,
                                            chain_from: Self::chain_id_to_name(chain1),
                                            chain_to: Self::chain_id_to_name(chain2),
                                            flash_loan_available: false,
                                            mev_risk: 0.05, // Lower MEV risk for cross-chain
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        opportunities
    }
    
    fn calculate_gas_cost(chain1: &u32, chain2: &u32, trade_size: f64) -> f64 {
        let base_gas = match (chain1, chain2) {
            (0, 0) => 0.0, // CEX to CEX
            (1, 1) => 100.0, // Ethereum
            (56, 56) => 5.0, // BSC
            (137, 137) => 2.0, // Polygon
            (42161, 42161) => 10.0, // Arbitrum
            (10, 10) => 8.0, // Optimism
            _ => 50.0, // Cross-chain
        };
        
        // Scale with trade size
        base_gas * (1.0 + trade_size / 100000.0)
    }
    
    fn calculate_slippage_impact(trade_size: f64, liquidity1: f64, liquidity2: f64) -> f64 {
        let avg_liquidity = (liquidity1 + liquidity2) / 2.0;
        
        if avg_liquidity <= 0.0 {
            return trade_size; // Maximum slippage
        }
        
        let impact_factor = trade_size / avg_liquidity;
        let slippage = impact_factor.sqrt() * 0.1 * trade_size;
        
        slippage.min(trade_size * 0.05) // Cap at 5%
    }
    
    fn calculate_confidence_score(spread: f64, volume: f64, slippage_rate: f64) -> f64 {
        let spread_score = (spread * 1000.0).min(1.0).max(0.0);
        let volume_score = (volume / 1000000.0).min(1.0).max(0.0);
        let slippage_score = (1.0 - slippage_rate).min(1.0).max(0.0);
        
        (spread_score * 0.4 + volume_score * 0.3 + slippage_score * 0.3)
    }
    
    fn chain_id_to_name(chain_id: u32) -> String {
        match chain_id {
            0 => "cex".to_string(),
            1 => "ethereum".to_string(),
            56 => "bsc".to_string(),
            137 => "polygon".to_string(),
            42161 => "arbitrum".to_string(),
            10 => "optimism".to_string(),
            43114 => "avalanche".to_string(),
            250 => "fantom".to_string(),
            _ => format!("chain_{}", chain_id),
        }
    }
    
    async fn gas_price_monitoring_worker(
        gas_prices: Arc<DashMap<String, f64>>,
        http_client: Client,
    ) {
        loop {
            // Update gas prices for all chains
            let gas_endpoints = [
                ("ethereum", "https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey=K4SEVFZ3PI8STM73VKV84C8PYZJUK7HB2G"),
                ("bsc", "https://api.bscscan.com/api?module=gastracker&action=gasoracle&apikey=K4SEVFZ3PI8STM73VKV84C8PYZJUK7HB2G"),
                ("polygon", "https://api.polygonscan.com/api?module=gastracker&action=gasoracle&apikey=K4SEVFZ3PI8STM73VKV84C8PYZJUK7HB2G"),
            ];
            
            for (chain, endpoint) in &gas_endpoints {
                if let Ok(response) = http_client.get(*endpoint).send().await {
                    if let Ok(data) = response.json::<Value>().await {
                        if let Some(gas_price) = data["result"]["FastGasPrice"].as_str().and_then(|s| s.parse::<f64>().ok()) {
                            gas_prices.insert(chain.to_string(), gas_price);
                        }
                    }
                }
            }
            
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
    }
    
    async fn opportunity_processing_worker(receiver: Receiver<ArbitrageOpportunity>) {
        let mut processed_count = 0u64;
        let mut total_profit = 0.0;
        let mut last_log = Instant::now();
        
        while let Ok(opportunity) = receiver.recv() {
            processed_count += 1;
            total_profit += opportunity.net_profit;
            
            // Log high-value opportunities immediately
            if opportunity.net_profit > 500.0 {
                println!("🚨 HIGH VALUE: ${:.2} {} {} -> {} | {:.3}% spread | {:.1}% confidence",
                        opportunity.net_profit,
                        opportunity.symbol,
                        opportunity.buy_exchange,
                        opportunity.sell_exchange,
                        opportunity.spread,
                        opportunity.confidence * 100.0);
            }
            
            // Periodic summary logging
            if last_log.elapsed() > Duration::from_secs(10) {
                println!("💰 Processed {} opportunities | Total potential: ${:.2} | Rate: {:.1}/sec",
                        processed_count,
                        total_profit,
                        processed_count as f64 / last_log.elapsed().as_secs() as f64);
                last_log = Instant::now();
            }
        }
    }
    
    pub fn update_price(&self, exchange: String, symbol: String, bid: f64, ask: f64, volume: f64) {
        let key = format!("{}:{}", exchange, symbol);
        let price = ExchangePrice {
            exchange,
            symbol,
            bid,
            ask,
            volume,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            chain_id: 1, // Default to Ethereum
            is_dex: false,
            liquidity_usd: volume * (bid + ask) / 2.0,
            gas_price_gwei: 30.0,
        };
        
        self.price_cache.insert(key, price);
    }
    
    pub fn get_opportunities(&self) -> Vec<String> {
        self.opportunities
            .iter()
            .take(100)
            .map(|entry| serde_json::to_string(entry.value()).unwrap_or_default())
            .collect()
    }
    
    pub fn get_stats(&self) -> (u64, f64, usize) {
        let total_opps = *self.total_opportunities.lock();
        let total_profit = *self.total_profit.lock();
        let cache_size = self.price_cache.len();
        
        (total_opps, total_profit, cache_size)
    }
    
    pub fn clear_old_data(&self) {
        let cutoff_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() - 300; // 5 minutes
        
        self.price_cache.retain(|_, price| price.timestamp > cutoff_time);
        self.opportunities.retain(|_, opp| opp.timestamp > cutoff_time);
    }
}

// Python bindings
#[pyclass]
pub struct PyUltraFastEngine {
    engine: UltraFastArbitrageEngine,
}

#[pymethods]
impl PyUltraFastEngine {
    #[new]
    pub fn new(min_profit: f64, max_slippage: f64) -> Self {
        Self {
            engine: UltraFastArbitrageEngine::new(min_profit, max_slippage),
        }
    }
    
    pub fn update_price(&self, exchange: String, symbol: String, bid: f64, ask: f64, volume: f64) {
        self.engine.update_price(exchange, symbol, bid, ask, volume);
    }
    
    pub fn get_opportunities(&self) -> Vec<String> {
        self.engine.get_opportunities()
    }
    
    pub fn get_stats(&self) -> (u64, f64, usize) {
        self.engine.get_stats()
    }
    
    pub fn clear_old_data(&self) {
        self.engine.clear_old_data();
    }
    
    pub fn start_async_scanning(&self) -> PyResult<()> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.spawn(async move {
            // This would normally start the scanning in the background
            // For the Python interface, we'll return immediately
        });
        Ok(())
    }
}

#[pymodule]
fn ultra_fast_arbitrage(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyUltraFastEngine>()?;
    Ok(())
}