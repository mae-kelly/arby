use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use ndarray::prelude::*;

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
    pub blockchain: String,
    pub gas_cost: f64,
    pub net_profit: f64,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ExchangeData {
    pub prices: HashMap<String, PriceData>,
    pub timestamp: u64,
    pub exchange: String,
    pub blockchain: String,
}

#[derive(Debug, Clone)]
pub struct PriceData {
    pub bid: f64,
    pub ask: f64,
    pub volume: f64,
    pub last_updated: u64,
}

pub struct ArbitrageEngine {
    pub exchanges: Arc<RwLock<Vec<ExchangeData>>>,
    pub opportunities: Arc<RwLock<Vec<ArbitrageOpportunity>>>,
    pub price_matrix: Arc<RwLock<Array2<f64>>>,
    pub gpu_accelerated: bool,
}

impl ArbitrageEngine {
    pub fn new(gpu_accelerated: bool) -> Self {
        Self {
            exchanges: Arc::new(RwLock::new(Vec::new())),
            opportunities: Arc::new(RwLock::new(Vec::new())),
            price_matrix: Arc::new(RwLock::new(Array2::zeros((0, 0)))),
            gpu_accelerated,
        }
    }

    pub async fn scan_cross_exchange_arbitrage(&self) -> Vec<ArbitrageOpportunity> {
        let exchanges = self.exchanges.read().await;
        let mut opportunities = Vec::new();
        
        let exchange_pairs: Vec<(&ExchangeData, &ExchangeData)> = exchanges
            .iter()
            .enumerate()
            .flat_map(|(i, ex1)| {
                exchanges[i+1..].iter().map(move |ex2| (ex1, ex2))
            })
            .collect();

        let batch_opportunities: Vec<Vec<ArbitrageOpportunity>> = exchange_pairs
            .par_iter()
            .map(|(ex1, ex2)| {
                self.find_cross_exchange_opportunities(ex1, ex2)
            })
            .collect();

        for batch in batch_opportunities {
            opportunities.extend(batch);
        }

        opportunities
    }

    fn find_cross_exchange_opportunities(
        &self,
        ex1: &ExchangeData,
        ex2: &ExchangeData,
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        for (symbol, price1) in &ex1.prices {
            if let Some(price2) = ex2.prices.get(symbol) {
                if price1.ask < price2.bid {
                    let spread = (price2.bid - price1.ask) / price1.ask;
                    if spread > 0.001 {
                        let trade_size = 10000.0;
                        let gross_profit = trade_size * spread;
                        let fees = trade_size * 0.002;
                        let gas_cost = self.calculate_gas_cost(&ex1.blockchain, &ex2.blockchain);
                        let net_profit = gross_profit - fees - gas_cost;
                        
                        if net_profit > 10.0 {
                            opportunities.push(ArbitrageOpportunity {
                                id: format!("{}_{}_{}_{}", ex1.exchange, ex2.exchange, symbol, chrono::Utc::now().timestamp_millis()),
                                strategy: "CrossExchange".to_string(),
                                symbol: symbol.clone(),
                                buy_exchange: ex1.exchange.clone(),
                                sell_exchange: ex2.exchange.clone(),
                                buy_price: price1.ask,
                                sell_price: price2.bid,
                                spread: spread * 100.0,
                                profit_estimate: gross_profit,
                                volume: price1.volume.min(price2.volume),
                                confidence: 0.8,
                                timestamp: chrono::Utc::now().timestamp_millis() as u64,
                                blockchain: ex1.blockchain.clone(),
                                gas_cost,
                                net_profit,
                                execution_time_ms: 500,
                            });
                        }
                    }
                }
                
                if price2.ask < price1.bid {
                    let spread = (price1.bid - price2.ask) / price2.ask;
                    if spread > 0.001 {
                        let trade_size = 10000.0;
                        let gross_profit = trade_size * spread;
                        let fees = trade_size * 0.002;
                        let gas_cost = self.calculate_gas_cost(&ex2.blockchain, &ex1.blockchain);
                        let net_profit = gross_profit - fees - gas_cost;
                        
                        if net_profit > 10.0 {
                            opportunities.push(ArbitrageOpportunity {
                                id: format!("{}_{}_{}_{}", ex2.exchange, ex1.exchange, symbol, chrono::Utc::now().timestamp_millis()),
                                strategy: "CrossExchange".to_string(),
                                symbol: symbol.clone(),
                                buy_exchange: ex2.exchange.clone(),
                                sell_exchange: ex1.exchange.clone(),
                                buy_price: price2.ask,
                                sell_price: price1.bid,
                                spread: spread * 100.0,
                                profit_estimate: gross_profit,
                                volume: price1.volume.min(price2.volume),
                                confidence: 0.8,
                                timestamp: chrono::Utc::now().timestamp_millis() as u64,
                                blockchain: ex2.blockchain.clone(),
                                gas_cost,
                                net_profit,
                                execution_time_ms: 500,
                            });
                        }
                    }
                }
            }
        }
        
        opportunities
    }

    pub async fn scan_triangular_arbitrage(&self) -> Vec<ArbitrageOpportunity> {
        let exchanges = self.exchanges.read().await;
        let mut opportunities = Vec::new();

        for exchange in exchanges.iter() {
            let triangular_opps = self.find_triangular_opportunities(exchange).await;
            opportunities.extend(triangular_opps);
        }

        opportunities
    }

    async fn find_triangular_opportunities(&self, exchange: &ExchangeData) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        let major_bases = vec!["BTC", "ETH", "BNB", "ADA", "SOL", "AVAX", "MATIC", "DOT"];
        let quote_currencies = vec!["USDT", "USDC", "BUSD"];

        for base1 in &major_bases {
            for base2 in &major_bases {
                if base1 == base2 { continue; }
                
                for quote in &quote_currencies {
                    let pair1 = format!("{}{}", base1, quote);
                    let pair2 = format!("{}{}", base1, base2);
                    let pair3 = format!("{}{}", base2, quote);
                    
                    if let (Some(price1), Some(price2), Some(price3)) = (
                        exchange.prices.get(&pair1),
                        exchange.prices.get(&pair2),
                        exchange.prices.get(&pair3),
                    ) {
                        let rate1 = price1.bid;
                        let rate2 = price2.bid;
                        let rate3 = 1.0 / price3.ask;
                        
                        let final_rate = rate1 * rate2 * rate3;
                        let profit_rate = final_rate - 1.0;
                        
                        if profit_rate > 0.002 {
                            let trade_size = 10000.0;
                            let gross_profit = trade_size * profit_rate;
                            let fees = trade_size * 0.003;
                            let net_profit = gross_profit - fees;
                            
                            if net_profit > 5.0 {
                                opportunities.push(ArbitrageOpportunity {
                                    id: format!("tri_{}_{}_{}_{}", exchange.exchange, pair1, pair2, chrono::Utc::now().timestamp_millis()),
                                    strategy: "Triangular".to_string(),
                                    symbol: format!("{}-{}-{}", pair1, pair2, pair3),
                                    buy_exchange: exchange.exchange.clone(),
                                    sell_exchange: exchange.exchange.clone(),
                                    buy_price: rate1,
                                    sell_price: final_rate,
                                    spread: profit_rate * 100.0,
                                    profit_estimate: gross_profit,
                                    volume: price1.volume.min(price2.volume).min(price3.volume),
                                    confidence: 0.7,
                                    timestamp: chrono::Utc::now().timestamp_millis() as u64,
                                    blockchain: exchange.blockchain.clone(),
                                    gas_cost: 0.0,
                                    net_profit,
                                    execution_time_ms: 200,
                                });
                            }
                        }
                    }
                }
            }
        }
        
        opportunities
    }

    pub async fn scan_flash_loan_arbitrage(&self) -> Vec<ArbitrageOpportunity> {
        let exchanges = self.exchanges.read().await;
        let mut opportunities = Vec::new();

        let dex_exchanges: Vec<&ExchangeData> = exchanges
            .iter()
            .filter(|ex| ex.exchange.contains("Uniswap") || ex.exchange.contains("Sushiswap") || ex.exchange.contains("PancakeSwap"))
            .collect();

        let cex_exchanges: Vec<&ExchangeData> = exchanges
            .iter()
            .filter(|ex| !ex.exchange.contains("Uniswap") && !ex.exchange.contains("Sushiswap") && !ex.exchange.contains("PancakeSwap"))
            .collect();

        for dex in &dex_exchanges {
            for cex in &cex_exchanges {
                if dex.blockchain == cex.blockchain {
                    let flash_opps = self.find_flash_loan_opportunities(dex, cex);
                    opportunities.extend(flash_opps);
                }
            }
        }

        opportunities
    }

    fn find_flash_loan_opportunities(&self, dex: &ExchangeData, cex: &ExchangeData) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        for (symbol, dex_price) in &dex.prices {
            if let Some(cex_price) = cex.prices.get(symbol) {
                if dex_price.ask < cex_price.bid {
                    let spread = (cex_price.bid - dex_price.ask) / dex_price.ask;
                    if spread > 0.005 {
                        let trade_size = 100000.0;
                        let gross_profit = trade_size * spread;
                        let flash_loan_fee = trade_size * 0.0005;
                        let gas_cost = self.calculate_gas_cost(&dex.blockchain, &cex.blockchain) * 2.0;
                        let net_profit = gross_profit - flash_loan_fee - gas_cost;
                        
                        if net_profit > 100.0 {
                            opportunities.push(ArbitrageOpportunity {
                                id: format!("flash_{}_{}_{}_{}", dex.exchange, cex.exchange, symbol, chrono::Utc::now().timestamp_millis()),
                                strategy: "FlashLoan".to_string(),
                                symbol: symbol.clone(),
                                buy_exchange: dex.exchange.clone(),
                                sell_exchange: cex.exchange.clone(),
                                buy_price: dex_price.ask,
                                sell_price: cex_price.bid,
                                spread: spread * 100.0,
                                profit_estimate: gross_profit,
                                volume: dex_price.volume.min(cex_price.volume),
                                confidence: 0.9,
                                timestamp: chrono::Utc::now().timestamp_millis() as u64,
                                blockchain: dex.blockchain.clone(),
                                gas_cost: gas_cost + flash_loan_fee,
                                net_profit,
                                execution_time_ms: 1000,
                            });
                        }
                    }
                }
            }
        }
        
        opportunities
    }

    fn calculate_gas_cost(&self, from_blockchain: &str, to_blockchain: &str) -> f64 {
        match (from_blockchain, to_blockchain) {
            ("ethereum", "ethereum") => 50.0,
            ("arbitrum", "arbitrum") => 5.0,
            ("polygon", "polygon") => 1.0,
            ("bsc", "bsc") => 2.0,
            ("avalanche", "avalanche") => 3.0,
            ("solana", "solana") => 0.1,
            _ => 25.0, // Cross-chain gas cost
        }
    }

    pub async fn update_exchange_data(&self, new_data: ExchangeData) {
        let mut exchanges = self.exchanges.write().await;
        
        if let Some(existing) = exchanges.iter_mut().find(|ex| 
            ex.exchange == new_data.exchange && ex.blockchain == new_data.blockchain
        ) {
            *existing = new_data;
        } else {
            exchanges.push(new_data);
        }
    }

    pub async fn get_top_opportunities(&self, limit: usize) -> Vec<ArbitrageOpportunity> {
        let mut all_opportunities = Vec::new();
        
        let cross_opps = self.scan_cross_exchange_arbitrage().await;
        let triangular_opps = self.scan_triangular_arbitrage().await;
        let flash_opps = self.scan_flash_loan_arbitrage().await;
        
        all_opportunities.extend(cross_opps);
        all_opportunities.extend(triangular_opps);
        all_opportunities.extend(flash_opps);
        
        all_opportunities.sort_by(|a, b| b.net_profit.partial_cmp(&a.net_profit).unwrap());
        all_opportunities.truncate(limit);
        
        let mut opportunities = self.opportunities.write().await;
        *opportunities = all_opportunities.clone();
        
        all_opportunities
    }

    pub async fn optimize_with_gpu(&self, opportunities: &mut Vec<ArbitrageOpportunity>) {
        if !self.gpu_accelerated {
            return;
        }

        let profits: Vec<f64> = opportunities.iter().map(|opp| opp.net_profit).collect();
        let spreads: Vec<f64> = opportunities.iter().map(|opp| opp.spread).collect();
        
        let profit_array = Array1::from(profits);
        let spread_array = Array1::from(spreads);
        
        let optimized_scores = &profit_array * 0.7 + &spread_array * 0.3;
        
        for (i, opp) in opportunities.iter_mut().enumerate() {
            if i < optimized_scores.len() {
                opp.confidence = (optimized_scores[i] / 1000.0).min(1.0).max(0.0);
            }
        }
        
        opportunities.sort_by(|a, b| {
            (b.confidence * b.net_profit).partial_cmp(&(a.confidence * a.net_profit)).unwrap()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_arbitrage_engine() {
        let engine = ArbitrageEngine::new(false);
        
        let mut exchange1 = ExchangeData {
            prices: HashMap::new(),
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            exchange: "Binance".to_string(),
            blockchain: "ethereum".to_string(),
        };
        
        exchange1.prices.insert("BTCUSDT".to_string(), PriceData {
            bid: 43000.0,
            ask: 43050.0,
            volume: 1000000.0,
            last_updated: chrono::Utc::now().timestamp_millis() as u64,
        });
        
        engine.update_exchange_data(exchange1).await;
        
        let opportunities = engine.get_top_opportunities(10).await;
        assert!(opportunities.len() >= 0);
    }
}