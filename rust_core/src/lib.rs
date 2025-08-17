use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use rayon::prelude::*;
use dashmap::DashMap;
use parking_lot::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangePrice {
    pub exchange: String,
    pub symbol: String,
    pub bid: f64,
    pub ask: f64,
    pub volume: f64,
    pub timestamp: u64,
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
    pub confidence: f64,
    pub gas_cost: f64,
    pub net_profit: f64,
}

pub struct UltraFastArbitrageEngine {
    price_cache: Arc<DashMap<String, ExchangePrice>>,
    opportunities: Arc<Mutex<Vec<ArbitrageOpportunity>>>,
    exchange_fees: HashMap<String, f64>,
    min_profit: f64,
}

impl UltraFastArbitrageEngine {
    pub fn new(min_profit: f64) -> Self {
        let mut exchange_fees = HashMap::new();
        exchange_fees.insert("binance".to_string(), 0.001);
        exchange_fees.insert("coinbase".to_string(), 0.005);
        exchange_fees.insert("okx".to_string(), 0.001);
        exchange_fees.insert("bybit".to_string(), 0.001);
        exchange_fees.insert("huobi".to_string(), 0.002);
        exchange_fees.insert("kucoin".to_string(), 0.001);
        exchange_fees.insert("gate".to_string(), 0.002);
        exchange_fees.insert("mexc".to_string(), 0.002);
        exchange_fees.insert("bitget".to_string(), 0.001);
        exchange_fees.insert("uniswap_v3".to_string(), 0.003);
        exchange_fees.insert("sushiswap".to_string(), 0.003);
        exchange_fees.insert("curve".to_string(), 0.0004);
        exchange_fees.insert("balancer".to_string(), 0.0005);
        exchange_fees.insert("1inch".to_string(), 0.003);

        Self {
            price_cache: Arc::new(DashMap::new()),
            opportunities: Arc::new(Mutex::new(Vec::new())),
            exchange_fees,
            min_profit,
        }
    }

    pub fn update_price(&self, price: ExchangePrice) {
        let key = format!("{}:{}", price.exchange, price.symbol);
        self.price_cache.insert(key, price);
    }

    pub fn scan_cross_exchange_arbitrage(&self) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        let mut symbol_prices: HashMap<String, Vec<ExchangePrice>> = HashMap::new();

        for entry in self.price_cache.iter() {
            let price = entry.value();
            symbol_prices
                .entry(price.symbol.clone())
                .or_insert_with(Vec::new)
                .push(price.clone());
        }

        symbol_prices
            .par_iter()
            .map(|(symbol, prices)| {
                let mut symbol_opportunities = Vec::new();
                
                for i in 0..prices.len() {
                    for j in i + 1..prices.len() {
                        let price1 = &prices[i];
                        let price2 = &prices[j];

                        let spread1 = (price2.bid - price1.ask) / price1.ask;
                        let spread2 = (price1.bid - price2.ask) / price2.ask;

                        if spread1 > 0.002 {
                            let trade_size = 10000.0;
                            let gross_profit = trade_size * spread1;
                            let fee1 = trade_size * self.exchange_fees.get(&price1.exchange).unwrap_or(&0.001);
                            let fee2 = trade_size * self.exchange_fees.get(&price2.exchange).unwrap_or(&0.001);
                            let net_profit = gross_profit - fee1 - fee2;

                            if net_profit > self.min_profit {
                                symbol_opportunities.push(ArbitrageOpportunity {
                                    strategy: "cross_exchange".to_string(),
                                    symbol: symbol.clone(),
                                    buy_exchange: price1.exchange.clone(),
                                    sell_exchange: price2.exchange.clone(),
                                    buy_price: price1.ask,
                                    sell_price: price2.bid,
                                    spread: spread1 * 100.0,
                                    profit_estimate: gross_profit,
                                    volume: price1.volume.min(price2.volume),
                                    confidence: 0.8,
                                    gas_cost: fee1 + fee2,
                                    net_profit,
                                });
                            }
                        }

                        if spread2 > 0.002 {
                            let trade_size = 10000.0;
                            let gross_profit = trade_size * spread2;
                            let fee1 = trade_size * self.exchange_fees.get(&price2.exchange).unwrap_or(&0.001);
                            let fee2 = trade_size * self.exchange_fees.get(&price1.exchange).unwrap_or(&0.001);
                            let net_profit = gross_profit - fee1 - fee2;

                            if net_profit > self.min_profit {
                                symbol_opportunities.push(ArbitrageOpportunity {
                                    strategy: "cross_exchange".to_string(),
                                    symbol: symbol.clone(),
                                    buy_exchange: price2.exchange.clone(),
                                    sell_exchange: price1.exchange.clone(),
                                    buy_price: price2.ask,
                                    sell_price: price1.bid,
                                    spread: spread2 * 100.0,
                                    profit_estimate: gross_profit,
                                    volume: price1.volume.min(price2.volume),
                                    confidence: 0.8,
                                    gas_cost: fee1 + fee2,
                                    net_profit,
                                });
                            }
                        }
                    }
                }
                symbol_opportunities
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|mut ops| opportunities.append(&mut ops));

        opportunities
    }

    pub fn scan_triangular_arbitrage(&self, exchange: &str) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        let exchange_prices: Vec<ExchangePrice> = self
            .price_cache
            .iter()
            .filter(|entry| entry.value().exchange == exchange)
            .map(|entry| entry.value().clone())
            .collect();

        let triangular_paths = vec![
            ("BTCUSDT", "ETHBTC", "ETHUSDT"),
            ("BNBUSDT", "ETHBNB", "ETHUSDT"),
            ("ADAUSDT", "ETHADA", "ETHUSDT"),
            ("DOTUSDT", "ETHDOT", "ETHUSDT"),
            ("LINKUSDT", "ETHLINK", "ETHUSDT"),
        ];

        for (pair1, pair2, pair3) in triangular_paths {
            let price1 = exchange_prices.iter().find(|p| p.symbol == pair1);
            let price2 = exchange_prices.iter().find(|p| p.symbol == pair2);
            let price3 = exchange_prices.iter().find(|p| p.symbol == pair3);

            if let (Some(p1), Some(p2), Some(p3)) = (price1, price2, price3) {
                let rate1 = p1.bid;
                let rate2 = p2.bid;
                let rate3 = 1.0 / p3.ask;

                let final_rate = rate1 * rate2 * rate3;
                let profit_rate = final_rate - 1.0;

                if profit_rate > 0.001 {
                    let trade_size = 10000.0;
                    let gross_profit = trade_size * profit_rate;
                    let fee = trade_size * 0.003;
                    let net_profit = gross_profit - fee;

                    if net_profit > self.min_profit {
                        opportunities.push(ArbitrageOpportunity {
                            strategy: "triangular".to_string(),
                            symbol: format!("{}-{}-{}", pair1, pair2, pair3),
                            buy_exchange: exchange.to_string(),
                            sell_exchange: exchange.to_string(),
                            buy_price: rate1,
                            sell_price: final_rate,
                            spread: profit_rate * 100.0,
                            profit_estimate: gross_profit,
                            volume: p1.volume.min(p2.volume).min(p3.volume),
                            confidence: 0.7,
                            gas_cost: fee,
                            net_profit,
                        });
                    }
                }
            }
        }

        opportunities
    }

    pub fn scan_flash_loan_arbitrage(&self, gas_price_gwei: f64, eth_price: f64) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        let mut symbol_prices: HashMap<String, Vec<ExchangePrice>> = HashMap::new();

        for entry in self.price_cache.iter() {
            let price = entry.value();
            if price.exchange.contains("uniswap") || price.exchange.contains("sushiswap") || 
               price.exchange.contains("curve") || price.exchange == "binance" || 
               price.exchange == "coinbase" || price.exchange == "okx" {
                symbol_prices
                    .entry(price.symbol.clone())
                    .or_insert_with(Vec::new)
                    .push(price.clone());
            }
        }

        symbol_prices
            .par_iter()
            .map(|(symbol, prices)| {
                let mut symbol_opportunities = Vec::new();
                let dex_prices: Vec<&ExchangePrice> = prices.iter()
                    .filter(|p| p.exchange.contains("uniswap") || p.exchange.contains("sushiswap") || p.exchange.contains("curve"))
                    .collect();
                let cex_prices: Vec<&ExchangePrice> = prices.iter()
                    .filter(|p| p.exchange == "binance" || p.exchange == "coinbase" || p.exchange == "okx")
                    .collect();

                for dex_price in &dex_prices {
                    for cex_price in &cex_prices {
                        let spread = (cex_price.bid - dex_price.ask) / dex_price.ask;
                        
                        if spread > 0.005 {
                            let trade_size = 100000.0;
                            let gross_profit = trade_size * spread;
                            let flash_loan_fee = trade_size * 0.0005;
                            let gas_cost = (400000.0 * gas_price_gwei * 1e9 / 1e18) * eth_price;
                            let net_profit = gross_profit - flash_loan_fee - gas_cost;

                            if net_profit > 50.0 {
                                symbol_opportunities.push(ArbitrageOpportunity {
                                    strategy: "flash_loan".to_string(),
                                    symbol: symbol.clone(),
                                    buy_exchange: dex_price.exchange.clone(),
                                    sell_exchange: cex_price.exchange.clone(),
                                    buy_price: dex_price.ask,
                                    sell_price: cex_price.bid,
                                    spread: spread * 100.0,
                                    profit_estimate: gross_profit,
                                    volume: dex_price.volume.min(cex_price.volume),
                                    confidence: 0.9,
                                    gas_cost: flash_loan_fee + gas_cost,
                                    net_profit,
                                });
                            }
                        }
                    }
                }
                symbol_opportunities
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|mut ops| opportunities.append(&mut ops));

        opportunities
    }

    pub fn get_all_opportunities(&self) -> Vec<ArbitrageOpportunity> {
        let cross_exchange = self.scan_cross_exchange_arbitrage();
        let triangular_binance = self.scan_triangular_arbitrage("binance");
        let triangular_okx = self.scan_triangular_arbitrage("okx");
        let flash_loan = self.scan_flash_loan_arbitrage(20.0, 2500.0);

        let mut all_opportunities = Vec::new();
        all_opportunities.extend(cross_exchange);
        all_opportunities.extend(triangular_binance);
        all_opportunities.extend(triangular_okx);
        all_opportunities.extend(flash_loan);

        all_opportunities.sort_by(|a, b| b.net_profit.partial_cmp(&a.net_profit).unwrap());
        all_opportunities.truncate(100);
        all_opportunities
    }

    pub fn clear_old_prices(&self, max_age_seconds: u64) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.price_cache.retain(|_, price| {
            current_time - price.timestamp < max_age_seconds
        });
    }
}

#[pyclass]
pub struct PyArbitrageEngine {
    engine: UltraFastArbitrageEngine,
}

#[pymethods]
impl PyArbitrageEngine {
    #[new]
    pub fn new(min_profit: f64) -> Self {
        Self {
            engine: UltraFastArbitrageEngine::new(min_profit),
        }
    }

    pub fn update_price(&self, exchange: String, symbol: String, bid: f64, ask: f64, volume: f64) {
        let price = ExchangePrice {
            exchange,
            symbol,
            bid,
            ask,
            volume,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        self.engine.update_price(price);
    }

    pub fn scan_opportunities(&self) -> Vec<String> {
        let opportunities = self.engine.get_all_opportunities();
        opportunities
            .into_iter()
            .map(|opp| serde_json::to_string(&opp).unwrap())
            .collect()
    }

    pub fn clear_old_data(&self) {
        self.engine.clear_old_prices(60);
    }

    pub fn get_cache_size(&self) -> usize {
        self.engine.price_cache.len()
    }
}

#[pymodule]
fn arbitrage_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyArbitrageEngine>()?;
    Ok(())
}