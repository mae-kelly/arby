// Simple Rust library that will compile
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SimpleArbitrage {
    pub exchanges: HashMap<String, f64>,
}

impl SimpleArbitrage {
    pub fn new() -> Self {
        Self {
            exchanges: HashMap::new(),
        }
    }
    
    pub fn add_price(&mut self, exchange: String, price: f64) {
        self.exchanges.insert(exchange, price);
    }
    
    pub fn find_arbitrage(&self) -> Option<(String, String, f64)> {
        let mut best_opportunity = None;
        let mut max_profit = 0.0;
        
        for (buy_exchange, buy_price) in &self.exchanges {
            for (sell_exchange, sell_price) in &self.exchanges {
                if buy_exchange != sell_exchange {
                    let profit = (sell_price - buy_price) / buy_price;
                    if profit > max_profit {
                        max_profit = profit;
                        best_opportunity = Some((
                            buy_exchange.clone(),
                            sell_exchange.clone(),
                            profit * 100.0
                        ));
                    }
                }
            }
        }
        
        best_opportunity
    }
}

// Export C functions for Python interop
#[no_mangle]
pub extern "C" fn create_arbitrage() -> *mut SimpleArbitrage {
    Box::into_raw(Box::new(SimpleArbitrage::new()))
}

#[no_mangle]
pub extern "C" fn destroy_arbitrage(ptr: *mut SimpleArbitrage) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn add_exchange_price(
    ptr: *mut SimpleArbitrage,
    exchange: *const std::os::raw::c_char,
    price: f64
) {
    if ptr.is_null() || exchange.is_null() {
        return;
    }
    
    unsafe {
        let arb = &mut *ptr;
        let exchange_str = std::ffi::CStr::from_ptr(exchange)
            .to_string_lossy()
            .to_string();
        arb.add_price(exchange_str, price);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arbitrage() {
        let mut arb = SimpleArbitrage::new();
        arb.add_price("binance".to_string(), 100.0);
        arb.add_price("coinbase".to_string(), 101.0);
        
        let result = arb.find_arbitrage();
        assert!(result.is_some());
        
        if let Some((buy, sell, profit)) = result {
            assert_eq!(buy, "binance");
            assert_eq!(sell, "coinbase");
            assert!(profit > 0.0);
        }
    }
}