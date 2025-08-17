use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use futures::stream::{FuturesUnordered, StreamExt};
use dashmap::DashMap;
use rayon::prelude::*;
use crossbeam::channel::{bounded, Receiver, Sender};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    pub id: String,
    pub strategy: ArbitrageStrategy,
    pub token_pair: TokenPair,
    pub buy_exchange: String,
    pub sell_exchange: String,
    pub buy_price: f64,
    pub sell_price: f64,
    pub spread_percentage: f64,
    pub potential_profit: f64,
    pub volume_24h: f64,
    pub liquidity: f64,
    pub execution_time_ms: u64,
    pub gas_cost_wei: u64,
    pub slippage_impact: f64,
    pub confidence_score: f64,
    pub timestamp: u64,
    pub chain_id: u32,
    pub dex_router: Option<String>,
    pub flash_loan_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArbitrageStrategy {
    CrossExchange,
    TriangularArbitrage,
    FlashLoanArbitrage,
    CrossChainArbitrage,
    StatisticalArbitrage,
    MevArbitrage,
    LiquidationArbitrage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPair {
    pub base: String,
    pub quote: String,
    pub address_base: Option<String>,
    pub address_quote: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangePrice {
    pub exchange: String,
    pub symbol: String,
    pub bid: f64,
    pub ask: f64,
    pub last: f64,
    pub volume: f64,
    pub timestamp: u64,
    pub chain_id: u32,
    pub is_dex: bool,
    pub liquidity_usd: f64,
}

#[derive(Debug, Clone)]
pub struct MarketData {
    prices: Arc<DashMap<String, ExchangePrice>>,
    opportunities: Arc<DashMap<String, ArbitrageOpportunity>>,
    last_update: Arc<RwLock<Instant>>,
}

impl MarketData {
    pub fn new() -> Self {
        Self {
            prices: Arc::new(DashMap::new()),
            opportunities: Arc::new(DashMap::new()),
            last_update: Arc::new(RwLock::new(Instant::now())),
        }
    }

    pub fn update_price(&self, key: String, price: ExchangePrice) {
        self.prices.insert(key, price);
    }

    pub fn get_price(&self, key: &str) -> Option<ExchangePrice> {
        self.prices.get(key).map(|entry| entry.clone())
    }

    pub fn add_opportunity(&self, opportunity: ArbitrageOpportunity) {
        self.opportunities.insert(opportunity.id.clone(), opportunity);
    }

    pub fn get_top_opportunities(&self, limit: usize) -> Vec<ArbitrageOpportunity> {
        let mut opps: Vec<_> = self.opportunities
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        
        opps.par_sort_by(|a, b| b.potential_profit.partial_cmp(&a.potential_profit).unwrap());
        opps.into_iter().take(limit).collect()
    }
}

pub struct HyperArbitrageEngine {
    market_data: MarketData,
    opportunity_sender: Sender<ArbitrageOpportunity>,
    opportunity_receiver: Receiver<ArbitrageOpportunity>,
    active_exchanges: Vec<String>,
    supported_chains: Vec<u32>,
    min_profit_threshold: f64,
    max_slippage: f64,
    concurrent_workers: usize,
}

impl HyperArbitrageEngine {
    pub fn new(
        min_profit_threshold: f64,
        max_slippage: f64,
        concurrent_workers: usize,
    ) -> Self {
        let (sender, receiver) = bounded(10000);
        
        Self {
            market_data: MarketData::new(),
            opportunity_sender: sender,
            opportunity_receiver: receiver,
            active_exchanges: vec![
                "binance".to_string(),
                "okx".to_string(),
                "coinbase".to_string(),
                "kucoin".to_string(),
                "bybit".to_string(),
                "gate".to_string(),
                "huobi".to_string(),
                "kraken".to_string(),
                "uniswap_v3".to_string(),
                "pancakeswap".to_string(),
                "sushiswap".to_string(),
                "quickswap".to_string(),
                "traderjoe".to_string(),
                "balancer".to_string(),
                "curve".to_string(),
            ],
            supported_chains: vec![1, 56, 137, 42161, 10, 43114, 250, 25, 100, 1284, 1285],
            min_profit_threshold,
            max_slippage,
            concurrent_workers,
        }
    }

    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("🦀 Starting Rust HyperArbitrage Engine with {} workers", self.concurrent_workers);
        
        let mut futures = FuturesUnordered::new();
        
        for worker_id in 0..self.concurrent_workers {
            let market_data = self.market_data.clone();
            let sender = self.opportunity_sender.clone();
            let exchanges = self.active_exchanges.clone();
            let chains = self.supported_chains.clone();
            let min_profit = self.min_profit_threshold;
            let max_slip = self.max_slippage;
            
            futures.push(tokio::spawn(async move {
                Self::arbitrage_worker(worker_id, market_data, sender, exchanges, chains, min_profit, max_slip).await
            }));
        }
        
        futures.push(tokio::spawn({
            let market_data = self.market_data.clone();
            async move {
                Self::price_updater_worker(market_data).await
            }
        }));
        
        futures.push(tokio::spawn({
            let receiver = self.opportunity_receiver.clone();
            async move {
                Self::opportunity_processor_worker(receiver).await
            }
        }));
        
        while let Some(result) = futures.next().await {
            if let Err(e) = result {
                eprintln!("Worker error: {:?}", e);
            }
        }
        
        Ok(())
    }

    async fn arbitrage_worker(
        worker_id: usize,
        market_data: MarketData,
        sender: Sender<ArbitrageOpportunity>,
        exchanges: Vec<String>,
        chains: Vec<u32>,
        min_profit: f64,
        max_slippage: f64,
    ) {
        let mut iteration = 0u64;
        
        loop {
            let start_time = Instant::now();
            
            let prices: Vec<(String, ExchangePrice)> = market_data.prices
                .iter()
                .map(|entry| (entry.key().clone(), entry.value().clone()))
                .collect();
            
            if prices.len() < 2 {
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
            
            let opportunities = Self::find_arbitrage_opportunities_parallel(
                &prices,
                &exchanges,
                &chains,
                min_profit,
                max_slippage,
            );
            
            for opportunity in opportunities {
                if opportunity.confidence_score > 0.7 {
                    let _ = sender.try_send(opportunity.clone());
                    market_data.add_opportunity(opportunity);
                }
            }
            
            iteration += 1;
            let elapsed = start_time.elapsed();
            
            if iteration % 1000 == 0 {
                println!(
                    "🔥 Worker {} - Iteration {}: Processed {} price pairs in {:?}",
                    worker_id, iteration, prices.len(), elapsed
                );
            }
            
            if elapsed < Duration::from_millis(1) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
    }

    fn find_arbitrage_opportunities_parallel(
        prices: &[(String, ExchangePrice)],
        exchanges: &[String],
        chains: &[u32],
        min_profit: f64,
        max_slippage: f64,
    ) -> Vec<ArbitrageOpportunity> {
        prices
            .par_iter()
            .flat_map(|(key1, price1)| {
                prices.par_iter().filter_map(move |(key2, price2)| {
                    if key1 >= key2 || price1.symbol != price2.symbol {
                        return None;
                    }
                    
                    Self::calculate_cross_exchange_opportunity(price1, price2, min_profit, max_slippage)
                })
            })
            .chain(Self::find_triangular_opportunities_parallel(prices, min_profit))
            .chain(Self::find_flash_loan_opportunities_parallel(prices, min_profit, max_slippage))
            .collect()
    }

    fn calculate_cross_exchange_opportunity(
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
        
        if spread < min_profit {
            return None;
        }
        
        let volume = price1.volume.min(price2.volume);
        let potential_profit = spread * volume * 0.001;
        let slippage_impact = Self::calculate_slippage_impact(volume, price1.liquidity_usd);
        
        if slippage_impact > max_slippage {
            return None;
        }
        
        let confidence_score = Self::calculate_confidence_score(spread, volume, slippage_impact);
        
        Some(ArbitrageOpportunity {
            id: format!("cross_{}_{}_{}_{}", 
                       price1.exchange, price2.exchange, price1.symbol, 
                       std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()),
            strategy: ArbitrageStrategy::CrossExchange,
            token_pair: TokenPair {
                base: price1.symbol.split('/').next().unwrap_or("").to_string(),
                quote: price1.symbol.split('/').nth(1).unwrap_or("").to_string(),
                address_base: None,
                address_quote: None,
            },
            buy_exchange: if price1.ask < price2.bid { price1.exchange.clone() } else { price2.exchange.clone() },
            sell_exchange: if price1.ask < price2.bid { price2.exchange.clone() } else { price1.exchange.clone() },
            buy_price: if price1.ask < price2.bid { price1.ask } else { price2.ask },
            sell_price: if price1.ask < price2.bid { price2.bid } else { price1.bid },
            spread_percentage: spread * 100.0,
            potential_profit,
            volume_24h: volume,
            liquidity: price1.liquidity_usd.min(price2.liquidity_usd),
            execution_time_ms: 150,
            gas_cost_wei: if price1.is_dex || price2.is_dex { 150000 } else { 0 },
            slippage_impact,
            confidence_score,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            chain_id: price1.chain_id,
            dex_router: None,
            flash_loan_available: true,
        })
    }

    fn find_triangular_opportunities_parallel(
        prices: &[(String, ExchangePrice)],
        min_profit: f64,
    ) -> Vec<ArbitrageOpportunity> {
        let exchange_groups: HashMap<String, Vec<&ExchangePrice>> = prices
            .iter()
            .fold(HashMap::new(), |mut acc, (_, price)| {
                acc.entry(price.exchange.clone()).or_insert_with(Vec::new).push(price);
                acc
            });
        
        exchange_groups
            .par_iter()
            .flat_map(|(exchange, exchange_prices)| {
                Self::find_triangular_in_exchange(exchange_prices, exchange, min_profit)
            })
            .collect()
    }

    fn find_triangular_in_exchange(
        prices: &[&ExchangePrice],
        exchange: &str,
        min_profit: f64,
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        for i in 0..prices.len() {
            for j in i + 1..prices.len() {
                for k in j + 1..prices.len() {
                    if let Some(opp) = Self::calculate_triangular_opportunity(
                        prices[i], prices[j], prices[k], exchange, min_profit
                    ) {
                        opportunities.push(opp);
                    }
                }
            }
        }
        
        opportunities
    }

    fn calculate_triangular_opportunity(
        price1: &ExchangePrice,
        price2: &ExchangePrice,
        price3: &ExchangePrice,
        exchange: &str,
        min_profit: f64,
    ) -> Option<ArbitrageOpportunity> {
        let rate1 = price1.bid;
        let rate2 = price2.bid;
        let rate3 = 1.0 / price3.ask;
        
        let final_rate = rate1 * rate2 * rate3;
        let profit_rate = final_rate - 1.0;
        
        if profit_rate < min_profit {
            return None;
        }
        
        let volume = price1.volume.min(price2.volume).min(price3.volume);
        let potential_profit = profit_rate * volume * 0.001;
        
        Some(ArbitrageOpportunity {
            id: format!("tri_{}_{}_{}_{}_{}", 
                       exchange, price1.symbol, price2.symbol, price3.symbol,
                       std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()),
            strategy: ArbitrageStrategy::TriangularArbitrage,
            token_pair: TokenPair {
                base: price1.symbol.split('/').next().unwrap_or("").to_string(),
                quote: price1.symbol.split('/').nth(1).unwrap_or("").to_string(),
                address_base: None,
                address_quote: None,
            },
            buy_exchange: exchange.to_string(),
            sell_exchange: exchange.to_string(),
            buy_price: rate1,
            sell_price: final_rate,
            spread_percentage: profit_rate * 100.0,
            potential_profit,
            volume_24h: volume,
            liquidity: price1.liquidity_usd,
            execution_time_ms: 200,
            gas_cost_wei: if price1.is_dex { 300000 } else { 0 },
            slippage_impact: Self::calculate_slippage_impact(volume, price1.liquidity_usd),
            confidence_score: Self::calculate_confidence_score(profit_rate, volume, 0.01),
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            chain_id: price1.chain_id,
            dex_router: None,
            flash_loan_available: true,
        })
    }

    fn find_flash_loan_opportunities_parallel(
        prices: &[(String, ExchangePrice)],
        min_profit: f64,
        max_slippage: f64,
    ) -> Vec<ArbitrageOpportunity> {
        let dex_prices: Vec<_> = prices.iter().filter(|(_, p)| p.is_dex).collect();
        let cex_prices: Vec<_> = prices.iter().filter(|(_, p)| !p.is_dex).collect();
        
        dex_prices
            .par_iter()
            .flat_map(|(_, dex_price)| {
                cex_prices.par_iter().filter_map(|(_, cex_price)| {
                    if dex_price.symbol == cex_price.symbol {
                        Self::calculate_flash_loan_opportunity(dex_price, cex_price, min_profit, max_slippage)
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    fn calculate_flash_loan_opportunity(
        dex_price: &ExchangePrice,
        cex_price: &ExchangePrice,
        min_profit: f64,
        max_slippage: f64,
    ) -> Option<ArbitrageOpportunity> {
        let spread = if dex_price.ask < cex_price.bid {
            (cex_price.bid - dex_price.ask) / dex_price.ask
        } else if cex_price.ask < dex_price.bid {
            (dex_price.bid - cex_price.ask) / cex_price.ask
        } else {
            return None;
        };
        
        if spread < min_profit * 2.0 {
            return None;
        }
        
        let flash_loan_amount = 100000.0;
        let flash_loan_fee = flash_loan_amount * 0.0005;
        let gross_profit = flash_loan_amount * spread;
        let net_profit = gross_profit - flash_loan_fee;
        
        if net_profit < 50.0 {
            return None;
        }
        
        let slippage_impact = Self::calculate_slippage_impact(flash_loan_amount, dex_price.liquidity_usd);
        
        if slippage_impact > max_slippage {
            return None;
        }
        
        Some(ArbitrageOpportunity {
            id: format!("flash_{}_{}_{}_{}", 
                       dex_price.exchange, cex_price.exchange, dex_price.symbol,
                       std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()),
            strategy: ArbitrageStrategy::FlashLoanArbitrage,
            token_pair: TokenPair {
                base: dex_price.symbol.split('/').next().unwrap_or("").to_string(),
                quote: dex_price.symbol.split('/').nth(1).unwrap_or("").to_string(),
                address_base: None,
                address_quote: None,
            },
            buy_exchange: if dex_price.ask < cex_price.bid { dex_price.exchange.clone() } else { cex_price.exchange.clone() },
            sell_exchange: if dex_price.ask < cex_price.bid { cex_price.exchange.clone() } else { dex_price.exchange.clone() },
            buy_price: if dex_price.ask < cex_price.bid { dex_price.ask } else { cex_price.ask },
            sell_price: if dex_price.ask < cex_price.bid { cex_price.bid } else { dex_price.bid },
            spread_percentage: spread * 100.0,
            potential_profit: net_profit,
            volume_24h: flash_loan_amount,
            liquidity: dex_price.liquidity_usd,
            execution_time_ms: 300,
            gas_cost_wei: 400000,
            slippage_impact,
            confidence_score: Self::calculate_confidence_score(spread, flash_loan_amount, slippage_impact),
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            chain_id: dex_price.chain_id,
            dex_router: Some("0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45".to_string()),
            flash_loan_available: true,
        })
    }

    fn calculate_slippage_impact(volume: f64, liquidity: f64) -> f64 {
        if liquidity <= 0.0 {
            return 1.0;
        }
        
        let impact_factor = volume / liquidity;
        impact_factor.sqrt() * 0.1
    }

    fn calculate_confidence_score(spread: f64, volume: f64, slippage: f64) -> f64 {
        let spread_score = (spread * 1000.0).min(1.0);
        let volume_score = (volume / 1000000.0).min(1.0);
        let slippage_score = 1.0 - slippage.min(1.0);
        
        (spread_score * 0.4 + volume_score * 0.3 + slippage_score * 0.3).max(0.0).min(1.0)
    }

    async fn price_updater_worker(market_data: MarketData) {
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        
        loop {
            interval.tick().await;
            
            let mock_prices = Self::generate_mock_prices();
            
            for (key, price) in mock_prices {
                market_data.update_price(key, price);
            }
            
            let mut last_update = market_data.last_update.write().await;
            *last_update = Instant::now();
        }
    }

    fn generate_mock_prices() -> Vec<(String, ExchangePrice)> {
        let exchanges = ["binance", "okx", "coinbase", "uniswap_v3", "pancakeswap"];
        let symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"];
        let chains = [1u32, 56, 137, 42161, 10];
        
        let mut prices = Vec::new();
        let base_time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        
        for (i, exchange) in exchanges.iter().enumerate() {
            for (j, symbol) in symbols.iter().enumerate() {
                let base_price = match *symbol {
                    "BTC/USDT" => 43000.0,
                    "ETH/USDT" => 2500.0,
                    "BNB/USDT" => 300.0,
                    "ADA/USDT" => 0.5,
                    "SOL/USDT" => 100.0,
                    _ => 1.0,
                };
                
                let variation = (i as f64 * 0.001) + (j as f64 * 0.0005);
                let price_variation = 1.0 + (variation - 0.002);
                
                let adjusted_price = base_price * price_variation;
                let spread = adjusted_price * 0.001;
                
                let price = ExchangePrice {
                    exchange: exchange.to_string(),
                    symbol: symbol.to_string(),
                    bid: adjusted_price - spread,
                    ask: adjusted_price + spread,
                    last: adjusted_price,
                    volume: 1000000.0 + (i * j) as f64 * 50000.0,
                    timestamp: base_time,
                    chain_id: chains[i % chains.len()],
                    is_dex: exchange.contains("swap") || exchange.contains("uniswap"),
                    liquidity_usd: 5000000.0 + (i * j) as f64 * 100000.0,
                };
                
                let key = format!("{}_{}", exchange, symbol.replace("/", "_"));
                prices.push((key, price));
            }
        }
        
        prices
    }

    async fn opportunity_processor_worker(receiver: Receiver<ArbitrageOpportunity>) {
        let mut processed_count = 0u64;
        let mut total_profit = 0.0f64;
        
        while let Ok(opportunity) = receiver.recv() {
            processed_count += 1;
            total_profit += opportunity.potential_profit;
            
            if processed_count % 100 == 0 {
                println!(
                    "💰 Processed {} opportunities | Total potential profit: ${:.2} | Latest: {} spread {:.3}%",
                    processed_count,
                    total_profit,
                    opportunity.strategy.serialize_variant_name(),
                    opportunity.spread_percentage
                );
            }
            
            if opportunity.potential_profit > 1000.0 {
                println!(
                    "🚨 HIGH VALUE OPPORTUNITY: ${:.2} profit | {} {} -> {} | {:.3}% spread",
                    opportunity.potential_profit,
                    opportunity.token_pair.base,
                    opportunity.buy_exchange,
                    opportunity.sell_exchange,
                    opportunity.spread_percentage
                );
            }
        }
    }
}

impl ArbitrageStrategy {
    fn serialize_variant_name(&self) -> &'static str {
        match self {
            ArbitrageStrategy::CrossExchange => "CrossExchange",
            ArbitrageStrategy::TriangularArbitrage => "TriangularArbitrage",
            ArbitrageStrategy::FlashLoanArbitrage => "FlashLoanArbitrage",
            ArbitrageStrategy::CrossChainArbitrage => "CrossChainArbitrage",
            ArbitrageStrategy::StatisticalArbitrage => "StatisticalArbitrage",
            ArbitrageStrategy::MevArbitrage => "MevArbitrage",
            ArbitrageStrategy::LiquidationArbitrage => "LiquidationArbitrage",
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let engine = HyperArbitrageEngine::new(0.0001, 0.005, 16);
    engine.start().await
}