use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[derive(Debug, Clone)]
pub struct SimpleMarket {
    pub exchange_id: u32,
    pub symbol_id: u32,
    pub bid: f64,
    pub ask: f64,
    pub volume: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct ArbitragePath {
    pub profit: f64,
    pub confidence: f64,
    pub exchanges: [u32; 8],
    pub length: u8,
}

pub struct ArbitrageEngine {
    markets: HashMap<u64, SimpleMarket>,
    opportunities: Vec<ArbitragePath>,
}

impl ArbitrageEngine {
    pub fn new() -> Self {
        Self {
            markets: HashMap::new(),
            opportunities: Vec::new(),
        }
    }
    
    pub fn add_market(&mut self, market: SimpleMarket) {
        let key = ((market.exchange_id as u64) << 32) | (market.symbol_id as u64);
        self.markets.insert(key, market);
    }
    
    pub fn find_arbitrage(&mut self) -> Vec<ArbitragePath> {
        let mut paths = Vec::new();
        
        // Group markets by symbol
        let mut by_symbol: HashMap<u32, Vec<&SimpleMarket>> = HashMap::new();
        for market in self.markets.values() {
            by_symbol.entry(market.symbol_id).or_default().push(market);
        }
        
        // Find arbitrage between exchanges
        for markets in by_symbol.values() {
            if markets.len() < 2 {
                continue;
            }
            
            for i in 0..markets.len() {
                for j in i+1..markets.len() {
                    let m1 = markets[i];
                    let m2 = markets[j];
                    
                    // Check both directions
                    if m2.bid > m1.ask {
                        let profit = (m2.bid - m1.ask) / m1.ask;
                        if profit > 0.001 {
                            let mut path = ArbitragePath {
                                profit,
                                confidence: 0.8,
                                exchanges: [0; 8],
                                length: 2,
                            };
                            path.exchanges[0] = m1.exchange_id;
                            path.exchanges[1] = m2.exchange_id;
                            paths.push(path);
                        }
                    }
                    
                    if m1.bid > m2.ask {
                        let profit = (m1.bid - m2.ask) / m2.ask;
                        if profit > 0.001 {
                            let mut path = ArbitragePath {
                                profit,
                                confidence: 0.8,
                                exchanges: [0; 8],
                                length: 2,
                            };
                            path.exchanges[0] = m2.exchange_id;
                            path.exchanges[1] = m1.exchange_id;
                            paths.push(path);
                        }
                    }
                }
            }
        }
        
        // Sort by profit
        paths.sort_by(|a, b| b.profit.partial_cmp(&a.profit).unwrap());
        paths.truncate(100);
        
        self.opportunities = paths.clone();
        paths
    }
    
    pub fn get_opportunities(&self) -> &Vec<ArbitragePath> {
        &self.opportunities
    }
}

// C FFI exports
#[no_mangle]
pub extern "C" fn create_engine() -> *mut ArbitrageEngine {
    Box::into_raw(Box::new(ArbitrageEngine::new()))
}

#[no_mangle]
pub extern "C" fn destroy_engine(ptr: *mut ArbitrageEngine) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn add_market(
    ptr: *mut ArbitrageEngine,
    exchange_id: u32,
    symbol_id: u32,
    bid: f64,
    ask: f64,
    volume: f64,
) {
    if ptr.is_null() {
        return;
    }
    
    unsafe {
        let engine = &mut *ptr;
        let market = SimpleMarket {
            exchange_id,
            symbol_id,
            bid,
            ask,
            volume,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        engine.add_market(market);
    }
}

#[no_mangle]
pub extern "C" fn find_opportunities(
    ptr: *mut ArbitrageEngine,
    paths_out: *mut ArbitragePath,
    max_paths: usize,
) -> usize {
    if ptr.is_null() || paths_out.is_null() {
        return 0;
    }
    
    unsafe {
        let engine = &mut *ptr;
        let paths = engine.find_arbitrage();
        
        let count = paths.len().min(max_paths);
        for i in 0..count {
            *paths_out.add(i) = paths[i].clone();
        }
        
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arbitrage() {
        let mut engine = ArbitrageEngine::new();
        
        // Add markets
        engine.add_market(SimpleMarket {
            exchange_id: 1,
            symbol_id: 1,
            bid: 100.0,
            ask: 101.0,
            volume: 1000.0,
            timestamp: 0,
        });
        
        engine.add_market(SimpleMarket {
            exchange_id: 2,
            symbol_id: 1,
            bid: 102.0,
            ask: 103.0,
            volume: 1000.0,
            timestamp: 0,
        });
        
        let paths = engine.find_arbitrage();
        assert!(!paths.is_empty());
        assert!(paths[0].profit > 0.0);
    }
}
