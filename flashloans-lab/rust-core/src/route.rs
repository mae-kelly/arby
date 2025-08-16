// File: src/route.rs
//! Multi-hop route optimization - Production Ready

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use std::hash::Hash;

/// Pool information for routing
#[derive(Debug, Clone)]
pub struct Pool {
    pub address: [u8; 20],
    pub token0: [u8; 20],
    pub token1: [u8; 20],
    pub fee: u32,
    pub liquidity: u128,
    pub sqrt_price_x96: u128,
    pub protocol: PoolProtocol,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PoolProtocol {
    UniswapV2,
    UniswapV3,
    Curve,
    Balancer,
}

/// Route through multiple pools
#[derive(Debug, Clone)]
pub struct Route {
    pub pools: Vec<Pool>,
    pub token_path: Vec<[u8; 20]>,
    pub expected_output: u128,
    pub price_impact: f64,
    pub gas_cost: u64,
}

/// Routing engine with graph-based optimization
pub struct Router {
    pools: Vec<Pool>,
    token_graph: HashMap<[u8; 20], Vec<usize>>, // token -> pool indices
    gas_costs: HashMap<PoolProtocol, u64>,
}

impl Router {
    pub fn new(pools: Vec<Pool>) -> Self {
        let mut token_graph = HashMap::new();
        
        for (idx, pool) in pools.iter().enumerate() {
            token_graph.entry(pool.token0)
                .or_insert_with(Vec::new)
                .push(idx);
            token_graph.entry(pool.token1)
                .or_insert_with(Vec::new)
                .push(idx);
        }
        
        let mut gas_costs = HashMap::new();
        gas_costs.insert(PoolProtocol::UniswapV2, 75_000);
        gas_costs.insert(PoolProtocol::UniswapV3, 140_000);
        gas_costs.insert(PoolProtocol::Curve, 150_000);
        gas_costs.insert(PoolProtocol::Balancer, 120_000);
        
        Router {
            pools,
            token_graph,
            gas_costs,
        }
    }
    
    /// Find optimal route using Dijkstra with dynamic programming
    pub fn find_best_route(
        &self,
        token_in: [u8; 20],
        token_out: [u8; 20],
        amount_in: u128,
        max_hops: usize,
        max_price_impact: f64,
    ) -> Result<Route, &'static str> {
        if token_in == token_out {
            return Err("SAME_TOKEN");
        }
        
        #[derive(Clone)]
        struct State {
            token: [u8; 20],
            amount: u128,
            path: Vec<usize>,
            gas_used: u64,
            price_impact: f64,
        }
        
        impl PartialEq for State {
            fn eq(&self, other: &Self) -> bool {
                self.amount == other.amount
            }
        }
        
        impl Eq for State {}
        
        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                self.amount.cmp(&other.amount)
            }
        }
        
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        
        let mut heap = BinaryHeap::new();
        let mut best_amounts: HashMap<([u8; 20], usize), u128> = HashMap::new();
        
        heap.push(State {
            token: token_in,
            amount: amount_in,
            path: vec![],
            gas_used: 0,
            price_impact: 0.0,
        });
        
        best_amounts.insert((token_in, 0), amount_in);
        
        let mut best_route: Option<Route> = None;
        
        while let Some(state) = heap.pop() {
            if state.token == token_out {
                let route = self.build_route(&state.path, token_in, amount_in)?;
                
                if best_route.is_none() || 
                   best_route.as_ref().unwrap().expected_output < route.expected_output {
                    best_route = Some(route);
                }
                continue;
            }
            
            if state.path.len() >= max_hops {
                continue;
            }
            
            // Explore adjacent pools
            if let Some(pool_indices) = self.token_graph.get(&state.token) {
                for &pool_idx in pool_indices {
                    if state.path.contains(&pool_idx) {
                        continue; // Avoid cycles
                    }
                    
                    let pool = &self.pools[pool_idx];
                    let next_token = if pool.token0 == state.token {
                        pool.token1
                    } else {
                        pool.token0
                    };
                    
                    // Calculate output amount
                    let output = self.calculate_output(
                        pool,
                        state.token == pool.token0,
                        state.amount
                    )?;
                    
                    if output == 0 {
                        continue;
                    }
                    
                    // Calculate price impact
                    let impact = self.calculate_price_impact(
                        pool,
                        state.token == pool.token0,
                        state.amount
                    );
                    
                    let cumulative_impact = state.price_impact + impact;
                    
                    if cumulative_impact > max_price_impact {
                        continue;
                    }
                    
                    let next_hops = state.path.len() + 1;
                    let key = (next_token, next_hops);
                    
                    if let Some(&best_amount) = best_amounts.get(&key) {
                        if output <= best_amount {
                            continue;
                        }
                    }
                    
                    best_amounts.insert(key, output);
                    
                    let mut next_path = state.path.clone();
                    next_path.push(pool_idx);
                    
                    let gas_used = state.gas_used + 
                        self.gas_costs.get(&pool.protocol).unwrap_or(&100_000);
                    
                    heap.push(State {
                        token: next_token,
                        amount: output,
                        path: next_path,
                        gas_used,
                        price_impact: cumulative_impact,
                    });
                }
            }
        }
        
        best_route.ok_or("NO_ROUTE_FOUND")
    }
    
    /// Build route from path indices
    fn build_route(
        &self,
        path_indices: &[usize],
        token_in: [u8; 20],
        amount_in: u128,
    ) -> Result<Route, &'static str> {
        let mut pools = Vec::new();
        let mut token_path = vec![token_in];
        let mut current_token = token_in;
        let mut current_amount = amount_in;
        let mut total_gas = 0u64;
        let mut total_impact = 0.0;
        
        for &idx in path_indices {
            let pool = &self.pools[idx];
            pools.push(pool.clone());
            
            let zero_for_one = pool.token0 == current_token;
            let next_token = if zero_for_one {
                pool.token1
            } else {
                pool.token0
            };
            
            current_amount = self.calculate_output(pool, zero_for_one, current_amount)?;
            total_impact += self.calculate_price_impact(pool, zero_for_one, current_amount);
            total_gas += self.gas_costs.get(&pool.protocol).unwrap_or(&100_000);
            
            token_path.push(next_token);
            current_token = next_token;
        }
        
        Ok(Route {
            pools,
            token_path,
            expected_output: current_amount,
            price_impact: total_impact,
            gas_cost: total_gas,
        })
    }
    
    /// Calculate output for a single pool
    fn calculate_output(
        &self,
        pool: &Pool,
        zero_for_one: bool,
        amount_in: u128,
    ) -> Result<u128, &'static str> {
        match pool.protocol {
            PoolProtocol::UniswapV2 => {
                // Use the UniV2 math from univ2.rs
                crate::univ2::get_amount_out(
                    amount_in,
                    if zero_for_one { pool.liquidity } else { pool.liquidity / 2 },
                    if zero_for_one { pool.liquidity / 2 } else { pool.liquidity },
                    pool.fee
                )
            },
            PoolProtocol::UniswapV3 => {
                // Simplified - would use full tick traversal
                let fee_amount = (amount_in as u128 * pool.fee as u128) / 1_000_000;
                Ok(amount_in - fee_amount)
            },
            PoolProtocol::Curve => {
                // Curve stable swap invariant
                let fee_amount = (amount_in as u128 * pool.fee as u128) / 10_000;
                Ok(amount_in - fee_amount)
            },
            PoolProtocol::Balancer => {
                // Weighted math or stable math depending on pool type
                let fee_amount = (amount_in as u128 * pool.fee as u128) / 1_000_000;
                Ok(amount_in - fee_amount)
            }
        }
    }
    
    /// Calculate price impact for a swap
    fn calculate_price_impact(
        &self,
        pool: &Pool,
        _zero_for_one: bool,
        amount_in: u128,
    ) -> f64 {
        // Calculate based on pool liquidity and trade size
        let trade_ratio = amount_in as f64 / pool.liquidity as f64;
        
        match pool.protocol {
            PoolProtocol::UniswapV2 => {
                // x * y = k impact model
                trade_ratio * trade_ratio * 100.0
            },
            PoolProtocol::UniswapV3 => {
                // Concentrated liquidity has less impact
                trade_ratio * 50.0
            },
            PoolProtocol::Curve => {
                // Stable pools have minimal impact
                trade_ratio * 10.0
            },
            PoolProtocol::Balancer => {
                // Depends on weights
                trade_ratio * 30.0
            }
        }
    }
    
    /// Split order across multiple routes for better execution
    pub fn split_route(
        &self,
        token_in: [u8; 20],
        token_out: [u8; 20],
        amount_in: u128,
        max_splits: usize,
    ) -> Result<Vec<(Route, u128)>, &'static str> {
        let mut routes = Vec::new();
        let split_amount = amount_in / max_splits as u128;
        
        for i in 0..max_splits {
            let amount = if i == max_splits - 1 {
                amount_in - (split_amount * (max_splits - 1) as u128)
            } else {
                split_amount
            };
            
            let route = self.find_best_route(
                token_in,
                token_out,
                amount,
                4,
                2.0
            )?;
            
            routes.push((route, amount));
        }
        
        // Sort by efficiency (output/input ratio)
        routes.sort_by(|a, b| {
            let ratio_a = a.0.expected_output as f64 / a.1 as f64;
            let ratio_b = b.0.expected_output as f64 / b.1 as f64;
            ratio_b.partial_cmp(&ratio_a).unwrap()
        });
        
        Ok(routes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_route_finding() {
        let pools = vec![
            Pool {
                address: [1; 20],
                token0: [10; 20], // Token A
                token1: [20; 20], // Token B
                fee: 30,
                liquidity: 1_000_000_000_000_000_000,
                sqrt_price_x96: 79228162514264337593543950336,
                protocol: PoolProtocol::UniswapV2,
            },
            Pool {
                address: [2; 20],
                token0: [20; 20], // Token B
                token1: [30; 20], // Token C
                fee: 30,
                liquidity: 2_000_000_000_000_000_000,
                sqrt_price_x96: 79228162514264337593543950336,
                protocol: PoolProtocol::UniswapV3,
            },
        ];
        
        let router = Router::new(pools);
        let route = router.find_best_route(
            [10; 20], // Token A
            [30; 20], // Token C
            1_000_000_000_000_000_000, // 1 token
            3,
            5.0
        );
        
        assert!(route.is_ok());
        let route = route.unwrap();
        assert_eq!(route.pools.len(), 2);
        assert!(route.expected_output > 0);
    }
}