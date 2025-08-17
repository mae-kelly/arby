use std::sync::{Arc, Mutex};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use rayon::prelude::*;
use crossbeam::channel::{bounded, unbounded};
use dashmap::DashMap;
use parking_lot::RwLock as PLRwLock;
use smallvec::SmallVec;
use ahash::{AHashMap, AHashSet};

const MAX_PATH_LENGTH: usize = 8;
const MIN_PROFIT_THRESHOLD: f64 = 0.001;
const MAX_CONCURRENT_PATHS: usize = 100000;
const CACHE_SIZE: usize = 1000000;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Market {
    pub exchange_id: u32,
    pub symbol_id: u32,
    pub bid: f64,
    pub ask: f64,
    pub bid_volume: f64,
    pub ask_volume: f64,
    pub fee: f64,
    pub timestamp: u64,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct ArbitragePath {
    pub markets: [u32; MAX_PATH_LENGTH],
    pub exchanges: [u32; MAX_PATH_LENGTH],
    pub profit: f64,
    pub volume: f64,
    pub confidence: f64,
    pub length: u8,
}

pub struct ArbitrageEngine {
    markets: Arc<DashMap<u64, Market>>,
    graph: Arc<PLRwLock<AHashMap<u32, Vec<u32>>>>,
    paths: Arc<RwLock<Vec<ArbitragePath>>>,
    cache: Arc<DashMap<u64, f64>>,
    execution_queue: Arc<Mutex<VecDeque<ArbitragePath>>>,
    stats: Arc<DashMap<String, f64>>,
}

impl ArbitrageEngine {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            markets: Arc::new(DashMap::with_capacity(100000)),
            graph: Arc::new(PLRwLock::new(AHashMap::with_capacity(10000))),
            paths: Arc::new(RwLock::new(Vec::with_capacity(10000))),
            cache: Arc::new(DashMap::with_capacity(CACHE_SIZE)),
            execution_queue: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            stats: Arc::new(DashMap::new()),
        })
    }

    pub fn update_market(&self, market: Market) {
        let key = ((market.exchange_id as u64) << 32) | (market.symbol_id as u64);
        
        // Update with timestamp check
        if let Some(mut existing) = self.markets.get_mut(&key) {
            if market.timestamp > existing.timestamp {
                *existing = market.clone();
            }
        } else {
            self.markets.insert(key, market.clone());
        }
        
        // Update graph
        let mut graph = self.graph.write();
        graph.entry(market.symbol_id)
            .or_insert_with(Vec::new)
            .push(market.exchange_id);
    }

    pub fn find_arbitrage_parallel(&self) -> Vec<ArbitragePath> {
        let markets = self.markets.clone();
        let graph = self.graph.read().clone();
        
        // Parallel search across all symbols
        let paths: Vec<ArbitragePath> = graph
            .par_iter()
            .flat_map(|(symbol_id, exchanges)| {
                self.find_paths_for_symbol(*symbol_id, exchanges, &markets)
            })
            .filter(|path| path.profit > MIN_PROFIT_THRESHOLD)
            .collect();
        
        // Sort by profit
        let mut sorted_paths = paths;
        sorted_paths.par_sort_unstable_by(|a, b| {
            b.profit.partial_cmp(&a.profit).unwrap()
        });
        
        sorted_paths.truncate(1000);
        sorted_paths
    }

    fn find_paths_for_symbol(
        &self,
        symbol_id: u32,
        exchanges: &[u32],
        markets: &DashMap<u64, Market>,
    ) -> Vec<ArbitragePath> {
        let mut paths = Vec::new();
        
        // Check direct arbitrage between exchanges
        for i in 0..exchanges.len() {
            for j in i + 1..exchanges.len() {
                let key1 = ((exchanges[i] as u64) << 32) | (symbol_id as u64);
                let key2 = ((exchanges[j] as u64) << 32) | (symbol_id as u64);
                
                if let (Some(m1), Some(m2)) = (markets.get(&key1), markets.get(&key2)) {
                    // Check for arbitrage opportunity
                    let profit = self.calculate_direct_arbitrage(&m1, &m2);
                    
                    if profit > MIN_PROFIT_THRESHOLD {
                        let mut path = ArbitragePath {
                            markets: [0; MAX_PATH_LENGTH],
                            exchanges: [0; MAX_PATH_LENGTH],
                            profit,
                            volume: (m1.bid_volume.min(m2.ask_volume)),
                            confidence: self.calculate_confidence(&m1, &m2),
                            length: 2,
                        };
                        
                        path.markets[0] = symbol_id;
                        path.markets[1] = symbol_id;
                        path.exchanges[0] = exchanges[i];
                        path.exchanges[1] = exchanges[j];
                        
                        paths.push(path);
                    }
                }
            }
        }
        
        // Check triangular arbitrage
        paths.extend(self.find_triangular_paths(symbol_id, exchanges, markets));
        
        paths
    }

    fn calculate_direct_arbitrage(&self, m1: &Market, m2: &Market) -> f64 {
        // Buy at m1.ask, sell at m2.bid
        let buy_price = m1.ask * (1.0 + m1.fee);
        let sell_price = m2.bid * (1.0 - m2.fee);
        
        if sell_price > buy_price {
            (sell_price - buy_price) / buy_price
        } else {
            // Try reverse
            let buy_price = m2.ask * (1.0 + m2.fee);
            let sell_price = m1.bid * (1.0 - m1.fee);
            
            if sell_price > buy_price {
                (sell_price - buy_price) / buy_price
            } else {
                0.0
            }
        }
    }

    fn find_triangular_paths(
        &self,
        start_symbol: u32,
        exchanges: &[u32],
        markets: &DashMap<u64, Market>,
    ) -> Vec<ArbitragePath> {
        let mut paths = Vec::new();
        
        // Find all symbols connected to start_symbol
        let connected_symbols = self.find_connected_symbols(start_symbol, markets);
        
        for middle_symbol in connected_symbols {
            // Find symbols connected to middle_symbol
            let final_symbols = self.find_connected_symbols(middle_symbol, markets);
            
            for end_symbol in final_symbols {
                if end_symbol == start_symbol {
                    continue; // Skip if it's back to start
                }
                
                // Check if we can complete the triangle
                if self.has_market(end_symbol, start_symbol, markets) {
                    // Calculate triangular arbitrage
                    let profit = self.calculate_triangular_profit(
                        start_symbol,
                        middle_symbol,
                        end_symbol,
                        exchanges,
                        markets,
                    );
                    
                    if profit > MIN_PROFIT_THRESHOLD {
                        let mut path = ArbitragePath {
                            markets: [0; MAX_PATH_LENGTH],
                            exchanges: [0; MAX_PATH_LENGTH],
                            profit,
                            volume: 0.0,
                            confidence: 0.8,
                            length: 3,
                        };
                        
                        path.markets[0] = start_symbol;
                        path.markets[1] = middle_symbol;
                        path.markets[2] = end_symbol;
                        
                        paths.push(path);
                    }
                }
            }
        }
        
        paths
    }

    fn find_connected_symbols(&self, symbol: u32, markets: &DashMap<u64, Market>) -> Vec<u32> {
        let mut connected = HashSet::new();
        
        for entry in markets.iter() {
            let market = entry.value();
            if market.symbol_id == symbol {
                // This is a simplification - in reality, you'd parse the symbol pairs
                connected.insert((market.symbol_id + 1) % 1000);
            }
        }
        
        connected.into_iter().collect()
    }

    fn has_market(&self, symbol1: u32, symbol2: u32, markets: &DashMap<u64, Market>) -> bool {
        markets.iter().any(|entry| {
            let market = entry.value();
            market.symbol_id == symbol1 || market.symbol_id == symbol2
        })
    }

    fn calculate_triangular_profit(
        &self,
        s1: u32,
        s2: u32,
        s3: u32,
        exchanges: &[u32],
        markets: &DashMap<u64, Market>,
    ) -> f64 {
        // Simplified triangular calculation
        let mut total_rate = 1.0;
        
        // Step 1: s1 -> s2
        if let Some(market) = self.get_best_market(s1, s2, exchanges, markets) {
            total_rate *= market.bid / market.ask;
            total_rate *= 1.0 - market.fee;
        }
        
        // Step 2: s2 -> s3
        if let Some(market) = self.get_best_market(s2, s3, exchanges, markets) {
            total_rate *= market.bid / market.ask;
            total_rate *= 1.0 - market.fee;
        }
        
        // Step 3: s3 -> s1
        if let Some(market) = self.get_best_market(s3, s1, exchanges, markets) {
            total_rate *= market.bid / market.ask;
            total_rate *= 1.0 - market.fee;
        }
        
        if total_rate > 1.0 {
            total_rate - 1.0
        } else {
            0.0
        }
    }

    fn get_best_market(
        &self,
        s1: u32,
        s2: u32,
        exchanges: &[u32],
        markets: &DashMap<u64, Market>,
    ) -> Option<Market> {
        let mut best_market = None;
        let mut best_rate = 0.0;
        
        for exchange in exchanges {
            let key = ((*exchange as u64) << 32) | (s1 as u64);
            if let Some(market) = markets.get(&key) {
                let rate = market.bid / market.ask;
                if rate > best_rate {
                    best_rate = rate;
                    best_market = Some(market.clone());
                }
            }
        }
        
        best_market
    }

    fn calculate_confidence(&self, m1: &Market, m2: &Market) -> f64 {
        let volume_factor = (m1.bid_volume.min(m2.ask_volume) / 10000.0).min(1.0);
        let time_factor = 1.0 / (1.0 + (Instant::now().elapsed().as_secs() as f64 - m1.timestamp as f64).abs() / 60.0);
        let spread_factor = 1.0 - ((m1.ask - m1.bid) / m1.bid).min(0.1);
        
        (volume_factor * time_factor * spread_factor).max(0.0).min(1.0)
    }

    pub async fn execute_arbitrage(&self, path: ArbitragePath) -> Result<String, String> {
        // Update statistics
        self.stats.insert("total_attempts".to_string(), 
            self.stats.get("total_attempts").map(|v| *v).unwrap_or(0.0) + 1.0);
        
        // Simulate execution
        let success = path.confidence > 0.5 && path.profit > MIN_PROFIT_THRESHOLD;
        
        if success {
            self.stats.insert("successful_trades".to_string(),
                self.stats.get("successful_trades").map(|v| *v).unwrap_or(0.0) + 1.0);
            self.stats.insert("total_profit".to_string(),
                self.stats.get("total_profit").map(|v| *v).unwrap_or(0.0) + path.profit);
            
            Ok(format!("Executed: profit={:.4}%, confidence={:.2}", 
                path.profit * 100.0, path.confidence))
        } else {
            Err("Execution failed".to_string())
        }
    }

    pub fn get_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        for entry in self.stats.iter() {
            stats.insert(entry.key().clone(), *entry.value());
        }
        stats
    }
}

// FFI exports for Python/C++ interop
#[no_mangle]
pub extern "C" fn create_engine() -> *mut ArbitrageEngine {
    let engine = ArbitrageEngine::new();
    Arc::into_raw(engine) as *mut ArbitrageEngine
}

#[no_mangle]
pub extern "C" fn destroy_engine(engine: *mut ArbitrageEngine) {
    if !engine.is_null() {
        unsafe {
            let _ = Arc::from_raw(engine);
        }
    }
}

#[no_mangle]
pub extern "C" fn update_market(
    engine: *mut ArbitrageEngine,
    exchange_id: u32,
    symbol_id: u32,
    bid: f64,
    ask: f64,
    bid_volume: f64,
    ask_volume: f64,
    fee: f64,
) {
    if engine.is_null() {
        return;
    }
    
    unsafe {
        let engine = &*engine;
        let market = Market {
            exchange_id,
            symbol_id,
            bid,
            ask,
            bid_volume,
            ask_volume,
            fee,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        engine.update_market(market);
    }
}

#[no_mangle]
pub extern "C" fn find_arbitrage(
    engine: *mut ArbitrageEngine,
    paths_out: *mut ArbitragePath,
    max_paths: usize,
) -> usize {
    if engine.is_null() || paths_out.is_null() {
        return 0;
    }
    
    unsafe {
        let engine = &*engine;
        let paths = engine.find_arbitrage_parallel();
        
        let count = paths.len().min(max_paths);
        for i in 0..count {
            *paths_out.add(i) = paths[i].clone();
        }
        
        count
    }
}