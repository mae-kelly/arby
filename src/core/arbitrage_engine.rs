use std::sync::{Arc, Mutex};
use std::collections::{HashSet, VecDeque};
use tokio::sync::RwLock;
use ethers::prelude::*;
use rayon::prelude::*;
use crossbeam::channel;
use dashmap::DashMap;
use parking_lot::RwLock as PLRwLock;
use smallvec::SmallVec;
use ahash::AHashMap;

const MAX_PATH_LENGTH: usize = 5;
const PROFIT_THRESHOLD: f64 = 0.001;
const MAX_CONCURRENT_PATHS: usize = 10000;

#[derive(Clone, Debug)]
pub struct Token {
    address: H160,
    symbol: String,
    decimals: u8,
    chain_id: u64,
}

#[derive(Clone, Debug)]
pub struct Market {
    id: String,
    token_a: Token,
    token_b: Token,
    reserves_a: U256,
    reserves_b: U256,
    fee: u32,
    exchange: String,
    chain_id: u64,
    gas_cost: U256,
}

#[derive(Clone)]
pub struct ArbitragePath {
    markets: SmallVec<[Market; 8]>,
    profit: f64,
    gas_cost: U256,
    execution_time: u64,
}

pub struct ArbitrageEngine {
    markets: Arc<DashMap<String, Market>>,
    graph: Arc<PLRwLock<AHashMap<String, Vec<String>>>>,
    paths: Arc<RwLock<Vec<ArbitragePath>>>,
    execution_queue: Arc<Mutex<VecDeque<ArbitragePath>>>,
    profit_tracker: Arc<DashMap<String, f64>>,
}

impl ArbitrageEngine {
    pub fn new() -> Self {
        Self {
            markets: Arc::new(DashMap::new()),
            graph: Arc::new(PLRwLock::new(AHashMap::new())),
            paths: Arc::new(RwLock::new(Vec::new())),
            execution_queue: Arc::new(Mutex::new(VecDeque::new())),
            profit_tracker: Arc::new(DashMap::new()),
        }
    }

    pub async fn update_market(&self, market: Market) {
        let market_id = market.id.clone();
        self.markets.insert(market_id.clone(), market.clone());
        
        let mut graph = self.graph.write();
        let token_a = market.token_a.symbol.clone();
        let token_b = market.token_b.symbol.clone();
        
        graph.entry(token_a.clone())
            .or_insert_with(Vec::new)
            .push(market_id.clone());
        graph.entry(token_b)
            .or_insert_with(Vec::new)
            .push(market_id);
    }

    pub async fn find_arbitrage_paths(&self) -> Vec<ArbitragePath> {
        let markets = self.markets.clone();
        let graph = self.graph.read().clone();
        
        let (tx, rx) = channel::unbounded();
        
        graph.par_iter().for_each(|(start_token, _)| {
            let paths = self.dfs_paths(
                start_token,
                start_token,
                vec![],
                HashSet::new(),
                &graph,
                &markets,
                0,
            );
            
            for path in paths {
                if self.calculate_profit(&path.markets) > PROFIT_THRESHOLD {
                    tx.send(path).unwrap();
                }
            }
        });
        
        drop(tx);
        rx.into_iter().collect()
    }

    fn dfs_paths(
        &self,
        current: &str,
        target: &str,
        path: Vec<Market>,
        visited: HashSet<String>,
        graph: &AHashMap<String, Vec<String>>,
        markets: &DashMap<String, Market>,
        depth: usize,
    ) -> Vec<ArbitragePath> {
        if depth > MAX_PATH_LENGTH {
            return vec![];
        }
        
        if current == target && !path.is_empty() {
            let path_slice: Vec<Market> = path.clone();
            let profit = self.calculate_profit(&path_slice);
            if profit > PROFIT_THRESHOLD {
                return vec![ArbitragePath {
                    markets: SmallVec::from_vec(path),
                    profit,
                    gas_cost: self.calculate_gas(&path_slice),
                    execution_time: 0,
                }];
            }
        }
        
        let mut results = vec![];
        
        if let Some(market_ids) = graph.get(current) {
            for market_id in market_ids {
                if visited.contains(market_id) {
                    continue;
                }
                
                if let Some(market) = markets.get(market_id) {
                    let mut new_visited = visited.clone();
                    new_visited.insert(market_id.clone());
                    
                    let mut new_path = path.clone();
                    new_path.push(market.clone());
                    
                    let next_token = if market.token_a.symbol == current {
                        &market.token_b.symbol
                    } else {
                        &market.token_a.symbol
                    };
                    
                    let sub_paths = self.dfs_paths(
                        next_token,
                        target,
                        new_path,
                        new_visited,
                        graph,
                        markets,
                        depth + 1,
                    );
                    
                    results.extend(sub_paths);
                }
            }
        }
        
        results
    }

    fn calculate_profit(&self, path: &[Market]) -> f64 {
        let mut amount = 1000000.0;
        
        for market in path {
            let fee_multiplier = 1.0 - (market.fee as f64 / 1000000.0);
            
            let reserves_a = market.reserves_a.as_u128() as f64;
            let reserves_b = market.reserves_b.as_u128() as f64;
            
            let amount_out = (amount * fee_multiplier * reserves_b) / 
                            (reserves_a + amount * fee_multiplier);
            
            amount = amount_out;
        }
        
        (amount - 1000000.0) / 1000000.0
    }

    fn calculate_gas(&self, path: &[Market]) -> U256 {
        let mut total = U256::zero();
        for market in path {
            total = total + market.gas_cost;
        }
        total
    }

    pub async fn execute_arbitrage(&self, path: ArbitragePath) -> Result<H256, Box<dyn std::error::Error>> {
        let encoded = self.encode_path(&path);
        let tx_hash = self.submit_transaction(encoded).await?;
        
        self.profit_tracker.insert(
            tx_hash.to_string(),
            path.profit,
        );
        
        Ok(tx_hash)
    }

    fn encode_path(&self, path: &ArbitragePath) -> Vec<u8> {
        let mut encoded = Vec::new();
        
        encoded.extend_from_slice(&(path.markets.len() as u32).to_be_bytes());
        
        for market in &path.markets {
            encoded.extend_from_slice(market.token_a.address.as_bytes());
            encoded.extend_from_slice(market.token_b.address.as_bytes());
            encoded.extend_from_slice(&market.exchange.as_bytes()[..32.min(market.exchange.len())]);
            encoded.extend_from_slice(&market.chain_id.to_be_bytes());
        }
        
        encoded
    }

    async fn submit_transaction(&self, data: Vec<u8>) -> Result<H256, Box<dyn std::error::Error>> {
        let provider = Provider::<Http>::try_from("http://localhost:8545")?;
        let wallet = "0x0000000000000000000000000000000000000000"
            .parse::<LocalWallet>()?;
        let client = SignerMiddleware::new(provider, wallet);
        
        let tx = TransactionRequest::new()
            .data(data)
            .gas(U256::from(3000000))
            .gas_price(U256::from(50_000_000_000u64));
        
        let pending_tx = client.send_transaction(tx, None).await?;
        let receipt = pending_tx.await?;
        
        Ok(receipt.unwrap().transaction_hash)
    }

    pub async fn optimize_paths(engine: Arc<Self>) {
        loop {
            let paths = engine.find_arbitrage_paths().await;
            
            let optimized: Vec<ArbitragePath> = paths
                .into_par_iter()
                .filter(|p| p.profit > PROFIT_THRESHOLD)
                .collect();
            
            {
                let mut queue = engine.execution_queue.lock().unwrap();
                for path in optimized {
                    queue.push_back(path);
                }
            }
            
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    }

    pub async fn run(self: Arc<Self>) {
        let engine = self.clone();
        
        tokio::spawn(async move {
            Self::optimize_paths(engine).await;
        });
        
        loop {
            let path = {
                let mut queue = self.execution_queue.lock().unwrap();
                queue.pop_front()
            };
            
            if let Some(path) = path {
                match self.execute_arbitrage(path).await {
                    Ok(tx) => println!("Executed: {}", tx),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        }
    }
}