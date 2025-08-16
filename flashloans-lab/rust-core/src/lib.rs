// File: src/lib.rs
use pyo3::prelude::*;

mod univ2;
mod univ3;
mod route;

#[pymodule]
fn flashloans_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(univ2_exact_in, m)?)?;
    m.add_function(wrap_pyfunction!(univ3_exact_in, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_gas, m)?)?;
    Ok(())
}

#[pyfunction]
fn univ2_exact_in(
    reserve_in: u128,
    reserve_out: u128,
    amount_in: u128,
    fee_bps: u32
) -> PyResult<u128> {
    let amount_in_with_fee = amount_in * (10000 - fee_bps) as u128;
    let numerator = amount_in_with_fee * reserve_out;
    let denominator = reserve_in * 10000 + amount_in_with_fee;
    
    Ok(numerator / denominator)
}

#[pyfunction]
fn univ3_exact_in(
    _sqrt_price_x96: u128,
    _liquidity: u128,
    amount_in: u128,
    fee_tier: u32
) -> PyResult<u128> {
    // Simplified UniV3 math
    let fee_multiplier = 1_000_000 - fee_tier;
    let amount_after_fee = amount_in * fee_multiplier as u128 / 1_000_000;
    
    // Would implement full tick traversal here
    Ok(amount_after_fee)
}

#[pyfunction]
fn estimate_gas(tx_type: &str, chain: &str) -> PyResult<u64> {
    let base_gas = match chain {
        "ethereum" => 21_000,
        "arbitrum" => 15_000,
        "base" => 15_000,
        _ => 21_000,
    };
    
    let operation_gas = match tx_type {
        "univ2_swap" => 75_000,
        "univ3_swap" => 140_000,
        "flash_loan" => 200_000,
        _ => 100_000,
    };
    
    Ok(base_gas + operation_gas)
}