use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Price {
    pub bid: f64,
    pub ask: f64,
    pub volume: f64,
    pub timestamp: u64,
    pub exchange: String,
    pub chain: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    pub strategy: String,
    pub symbol: String,
    pub buy_exchange: String,
    pub sell_exchange: String,
    pub buy_price: f64,
    pub sell_price: f64,
    pub spread: f64,
    pub profit_estimate: f64,
    pub volume: f64,
    pub gas_cost: f64,
    pub net_profit: f64,
    pub confidence: f64,
    pub chain_from: String,
    pub chain_to: String,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct Exchange {
    pub name: String,
    pub chain: String,
    pub fee_rate: f64,
    pub min_trade_size: f64,
    pub max_trade_size: f64,
    pub api_endpoint: String,
    pub websocket_endpoint: String,
}

pub struct ArbitrageEngine {
    pub exchanges: Vec<Exchange>,
    pub price_cache: Arc<RwLock<HashMap<String, HashMap<String, Price>>>>,
    pub opportunities: Arc<RwLock<Vec<ArbitrageOpportunity>>>,
    pub gas_prices: Arc<RwLock<HashMap<String, f64>>>,
    pub eth_price: Arc<RwLock<f64>>,
}

impl ArbitrageEngine {
    pub fn new() -> Self {
        let exchanges = vec![
            Exchange {
                name: "Binance".to_string(),
                chain: "CEX".to_string(),
                fee_rate: 0.001,
                min_trade_size: 10.0,
                max_trade_size: 1000000.0,
                api_endpoint: "https://api.binance.com".to_string(),
                websocket_endpoint: "wss://stream.binance.com:9443".to_string(),
            },
            Exchange {
                name: "Coinbase".to_string(),
                chain: "CEX".to_string(),
                fee_rate: 0.005,
                min_trade_size: 1.0,
                max_trade_size: 500000.0,
                api_endpoint: "https://api.exchange.coinbase.com".to_string(),
                websocket_endpoint: "wss://ws-feed.exchange.coinbase.com".to_string(),
            },
            Exchange {
                name: "Uniswap".to_string(),
                chain: "Ethereum".to_string(),
                fee_rate: 0.003,
                min_trade_size: 1.0,
                max_trade_size: 10000000.0,
                api_endpoint: "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3".to_string(),
                websocket_endpoint: "".to_string(),
            },
            Exchange {
                name: "PancakeSwap".to_string(),
                chain: "BSC".to_string(),
                fee_rate: 0.0025,
                min_trade_size: 1.0,
                max_trade_size: 5000000.0,
                api_endpoint: "https://api.pancakeswap.info".to_string(),
                websocket_endpoint: "".to_string(),
            },
            Exchange {
                name: "SushiSwap".to_string(),
                chain: "Ethereum".to_string(),
                fee_rate: 0.003,
                min_trade_size: 1.0,
                max_trade_size: 2000000.0,
                api_endpoint: "https://api.sushi.com".to_string(),
                websocket_endpoint: "".to_string(),
            },
            Exchange {
                name: "QuickSwap".to_string(),
                chain: "Polygon".to_string(),
                fee_rate: 0.003,
                min_trade_size: 1.0,
                max_trade_size: 1000000.0,
                api_endpoint: "https://api.quickswap.exchange".to_string(),
                websocket_endpoint: "".to_string(),
            },
            Exchange {
                name: "TraderJoe".to_string(),
                chain: "Avalanche".to_string(),
                fee_rate: 0.003,
                min_trade_size: 1.0,
                max_trade_size: 1000000.0,
                api_endpoint: "https://api.traderjoexyz.com".to_string(),
                websocket_endpoint: "".to_string(),
            },
            Exchange {
                name: "SpookySwap".to_string(),
                chain: "Fantom".to_string(),
                fee_rate: 0.002,
                min_trade_size: 1.0,
                max_trade_size: 500000.0,
                api_endpoint: "https://api.spookyswap.finance".to_string(),
                websocket_endpoint: "".to_string(),
            },
            Exchange {
                name: "Raydium".to_string(),
                chain: "Solana".to_string(),
                fee_rate: 0.0025,
                min_trade_size: 1.0,
                max_trade_size: 2000000.0,
                api_endpoint: "https://api.raydium.io".to_string(),
                websocket_endpoint: "".to_string(),
            },
            Exchange {
                name: "Orca".to_string(),
                chain: "Solana".to_string(),
                fee_rate: 0.003,
                min_trade_size: 1.0,
                max_trade_size: 1000000.0,
                api_endpoint: "https://api.orca.so".to_string(),
                websocket_endpoint: "".to_string(),
            },
        ];

        ArbitrageEngine {
            exchanges,
            price_cache: Arc::new(RwLock::new(HashMap::new())),
            opportunities: Arc::new(RwLock::new(Vec::new())),
            gas_prices: Arc::new(RwLock::new(HashMap::new())),
            eth_price: Arc::new(RwLock::new(4000.0)),
        }
    }

    pub async fn scan_all_opportunities(&self) -> Vec<ArbitrageOpportunity> {
        let price_cache = self.price_cache.read().await;
        let mut all_opportunities = Vec::new();

        let symbols: Vec<String> = price_cache.keys().cloned().collect();
        
        // Parallel processing of symbols
        let opportunities: Vec<Vec<ArbitrageOpportunity>> = symbols
            .par_iter()
            .map(|symbol| {
                self.scan_symbol_opportunities(symbol, &price_cache)
            })
            .collect();

        for symbol_opportunities in opportunities {
            all_opportunities.extend(symbol_opportunities);
        }

        // Sort by net profit
        all_opportunities.sort_by(|a, b| b.net_profit.partial_cmp(&a.net_profit).unwrap());
        
        all_opportunities
    }

    fn scan_symbol_opportunities(
        &self,
        symbol: &str,
        price_cache: &HashMap<String, HashMap<String, Price>>,
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();

        if let Some(symbol_prices) = price_cache.get(symbol) {
            // Cross-exchange arbitrage
            opportunities.extend(self.find_cross_exchange_opportunities(symbol, symbol_prices));
            
            // Cross-chain arbitrage
            opportunities.extend(self.find_cross_chain_opportunities(symbol, symbol_prices));
            
            // Flash loan arbitrage
            opportunities.extend(self.find_flash_loan_opportunities(symbol, symbol_prices));
            
            // Triangular arbitrage
            opportunities.extend(self.find_triangular_opportunities(symbol, price_cache));
        }

        opportunities
    }

    fn find_cross_exchange_opportunities(
        &self,
        symbol: &str,
        prices: &HashMap<String, Price>,
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        let exchanges: Vec<&String> = prices.keys().collect();

        for i in 0..exchanges.len() {
            for j in i + 1..exchanges.len() {
                let exchange1 = exchanges[i];
                let exchange2 = exchanges[j];
                
                if let (Some(price1), Some(price2)) = (prices.get(exchange1), prices.get(exchange2)) {
                    // Check both directions
                    if price1.ask < price2.bid {
                        let spread = (price2.bid - price1.ask) / price1.ask;
                        if spread > 0.002 {
                            let opportunity = self.calculate_arbitrage_profit(
                                "Cross-Exchange",
                                symbol,
                                exchange1,
                                exchange2,
                                price1.ask,
                                price2.bid,
                                spread,
                                price1.volume.min(price2.volume),
                                &price1.chain,
                                &price2.chain,
                            );
                            opportunities.push(opportunity);
                        }
                    }
                    
                    if price2.ask < price1.bid {
                        let spread = (price1.bid - price2.ask) / price2.ask;
                        if spread > 0.002 {
                            let opportunity = self.calculate_arbitrage_profit(
                                "Cross-Exchange",
                                symbol,
                                exchange2,
                                exchange1,
                                price2.ask,
                                price1.bid,
                                spread,
                                price1.volume.min(price2.volume),
                                &price2.chain,
                                &price1.chain,
                            );
                            opportunities.push(opportunity);
                        }
                    }
                }
            }
        }

        opportunities
    }

    fn find_cross_chain_opportunities(
        &self,
        symbol: &str,
        prices: &HashMap<String, Price>,
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        // Group by chain
        let mut chain_prices: HashMap<String, Vec<&Price>> = HashMap::new();
        for price in prices.values() {
            chain_prices.entry(price.chain.clone()).or_insert_with(Vec::new).push(price);
        }

        // Find best prices on each chain
        for (chain1, prices1) in &chain_prices {
            for (chain2, prices2) in &chain_prices {
                if chain1 != chain2 {
                    let best_ask1 = prices1.iter().min_by(|a, b| a.ask.partial_cmp(&b.ask).unwrap());
                    let best_bid2 = prices2.iter().max_by(|a, b| a.bid.partial_cmp(&b.bid).unwrap());
                    
                    if let (Some(ask_price), Some(bid_price)) = (best_ask1, best_bid2) {
                        if ask_price.ask < bid_price.bid {
                            let spread = (bid_price.bid - ask_price.ask) / ask_price.ask;
                            if spread > 0.005 { // Higher threshold for cross-chain
                                let opportunity = self.calculate_cross_chain_profit(
                                    symbol,
                                    &ask_price.exchange,
                                    &bid_price.exchange,
                                    ask_price.ask,
                                    bid_price.bid,
                                    spread,
                                    ask_price.volume.min(bid_price.volume),
                                    chain1,
                                    chain2,
                                );
                                opportunities.push(opportunity);
                            }
                        }
                    }
                }
            }
        }

        opportunities
    }

    fn find_flash_loan_opportunities(
        &self,
        symbol: &str,
        prices: &HashMap<String, Price>,
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        // Find DEX vs CEX opportunities suitable for flash loans
        let dex_prices: Vec<&Price> = prices.values().filter(|p| p.chain != "CEX").collect();
        let cex_prices: Vec<&Price> = prices.values().filter(|p| p.chain == "CEX").collect();

        for dex_price in &dex_prices {
            for cex_price in &cex_prices {
                if dex_price.ask < cex_price.bid {
                    let spread = (cex_price.bid - dex_price.ask) / dex_price.ask;
                    if spread > 0.005 { // Higher threshold for flash loans
                        let opportunity = self.calculate_flash_loan_profit(
                            symbol,
                            &dex_price.exchange,
                            &cex_price.exchange,
                            dex_price.ask,
                            cex_price.bid,
                            spread,
                            dex_price.volume.min(cex_price.volume),
                            &dex_price.chain,
                        );
                        opportunities.push(opportunity);
                    }
                }
            }
        }

        opportunities
    }

    fn find_triangular_opportunities(
        &self,
        _symbol: &str,
        price_cache: &HashMap<String, HashMap<String, Price>>,
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        // Define triangular paths
        let triangles = vec![
            ("BTCUSDT", "ETHBTC", "ETHUSDT"),
            ("BNBUSDT", "ETHBNB", "ETHUSDT"),
            ("ADAUSDT", "ETHADA", "ETHUSDT"),
        ];

        for (pair1, pair2, pair3) in triangles {
            if let (Some(prices1), Some(prices2), Some(prices3)) = (
                price_cache.get(pair1),
                price_cache.get(pair2),
                price_cache.get(pair3),
            ) {
                for exchange in self.exchanges.iter().filter(|e| e.chain == "CEX") {
                    if let (Some(p1), Some(p2), Some(p3)) = (
                        prices1.get(&exchange.name),
                        prices2.get(&exchange.name),
                        prices3.get(&exchange.name),
                    ) {
                        let rate1 = p1.bid;
                        let rate2 = p2.bid;
                        let rate3 = 1.0 / p3.ask;
                        
                        let final_rate = rate1 * rate2 * rate3;
                        let profit_rate = final_rate - 1.0;
                        
                        if profit_rate > 0.001 {
                            let trade_size = 10000.0;
                            let gross_profit = trade_size * profit_rate;
                            let fees = trade_size * (exchange.fee_rate * 3.0); // 3 trades
                            let net_profit = gross_profit - fees;
                            
                            if net_profit > 5.0 {
                                opportunities.push(ArbitrageOpportunity {
                                    strategy: "Triangular".to_string(),
                                    symbol: format!("{}-{}-{}", pair1, pair2, pair3),
                                    buy_exchange: exchange.name.clone(),
                                    sell_exchange: exchange.name.clone(),
                                    buy_price: rate1,
                                    sell_price: final_rate,
                                    spread: profit_rate * 100.0,
                                    profit_estimate: gross_profit,
                                    volume: p1.volume.min(p2.volume).min(p3.volume),
                                    gas_cost: 0.0,
                                    net_profit,
                                    confidence: 0.7,
                                    chain_from: exchange.chain.clone(),
                                    chain_to: exchange.chain.clone(),
                                    execution_time_ms: 500,
                                });
                            }
                        }
                    }
                }
            }
        }

        opportunities
    }

    fn calculate_arbitrage_profit(
        &self,
        strategy: &str,
        symbol: &str,
        buy_exchange: &str,
        sell_exchange: &str,
        buy_price: f64,
        sell_price: f64,
        spread: f64,
        volume: f64,
        chain_from: &str,
        chain_to: &str,
    ) -> ArbitrageOpportunity {
        let trade_size = 10000.0; // $10k position
        let gross_profit = trade_size * spread;
        let fees = trade_size * 0.002; // 0.2% total fees
        let gas_cost = if chain_from != "CEX" || chain_to != "CEX" { 25.0 } else { 0.0 };
        let net_profit = gross_profit - fees - gas_cost;

        ArbitrageOpportunity {
            strategy: strategy.to_string(),
            symbol: symbol.to_string(),
            buy_exchange: buy_exchange.to_string(),
            sell_exchange: sell_exchange.to_string(),
            buy_price,
            sell_price,
            spread: spread * 100.0,
            profit_estimate: gross_profit,
            volume,
            gas_cost,
            net_profit,
            confidence: 0.8,
            chain_from: chain_from.to_string(),
            chain_to: chain_to.to_string(),
            execution_time_ms: 200,
        }
    }

    fn calculate_cross_chain_profit(
        &self,
        symbol: &str,
        buy_exchange: &str,
        sell_exchange: &str,
        buy_price: f64,
        sell_price: f64,
        spread: f64,
        volume: f64,
        chain_from: &str,
        chain_to: &str,
    ) -> ArbitrageOpportunity {
        let trade_size = 25000.0; // Larger size for cross-chain
        let gross_profit = trade_size * spread;
        let bridge_fee = trade_size * 0.001; // 0.1% bridge fee
        let gas_cost = 50.0; // Higher gas for cross-chain
        let net_profit = gross_profit - bridge_fee - gas_cost;

        ArbitrageOpportunity {
            strategy: "Cross-Chain".to_string(),
            symbol: symbol.to_string(),
            buy_exchange: buy_exchange.to_string(),
            sell_exchange: sell_exchange.to_string(),
            buy_price,
            sell_price,
            spread: spread * 100.0,
            profit_estimate: gross_profit,
            volume,
            gas_cost: gas_cost + bridge_fee,
            net_profit,
            confidence: 0.75,
            chain_from: chain_from.to_string(),
            chain_to: chain_to.to_string(),
            execution_time_ms: 5000, // Cross-chain takes longer
        }
    }

    fn calculate_flash_loan_profit(
        &self,
        symbol: &str,
        buy_exchange: &str,
        sell_exchange: &str,
        buy_price: f64,
        sell_price: f64,
        spread: f64,
        volume: f64,
        chain: &str,
    ) -> ArbitrageOpportunity {
        let trade_size = 100000.0; // Large flash loan size
        let gross_profit = trade_size * spread;
        let flash_loan_fee = trade_size * 0.0005; // 0.05% flash loan fee
        let gas_cost = 75.0; // Higher gas for flash loan complexity
        let net_profit = gross_profit - flash_loan_fee - gas_cost;

        ArbitrageOpportunity {
            strategy: "Flash Loan".to_string(),
            symbol: symbol.to_string(),
            buy_exchange: buy_exchange.to_string(),
            sell_exchange: sell_exchange.to_string(),
            buy_price,
            sell_price,
            spread: spread * 100.0,
            profit_estimate: gross_profit,
            volume,
            gas_cost: gas_cost + flash_loan_fee,
            net_profit,
            confidence: 0.9,
            chain_from: chain.to_string(),
            chain_to: "CEX".to_string(),
            execution_time_ms: 1000,
        }
    }

    pub async fn update_price(
        &self,
        symbol: String,
        exchange: String,
        price: Price,
    ) {
        let mut cache = self.price_cache.write().await;
        cache.entry(symbol).or_insert_with(HashMap::new).insert(exchange, price);
    }

    pub async fn get_top_opportunities(&self, limit: usize) -> Vec<ArbitrageOpportunity> {
        let opportunities = self.opportunities.read().await;
        opportunities.iter().take(limit).cloned().collect()
    }
}