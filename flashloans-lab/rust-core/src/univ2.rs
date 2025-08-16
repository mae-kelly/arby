// File: src/univ2.rs
//! UniswapV2 AMM math implementations - Production Ready

use num_bigint::{BigUint, ToBigUint};
use num_traits::{One, ToPrimitive};

/// Calculate exact output amount for UniswapV2 swap
/// Uses full precision arithmetic to avoid rounding errors
pub fn get_amount_out(
    amount_in: u128,
    reserve_in: u128,
    reserve_out: u128,
    fee: u32, // Fee in basis points (30 = 0.3%)
) -> Result<u128, &'static str> {
    if amount_in == 0 {
        return Err("INSUFFICIENT_INPUT_AMOUNT");
    }
    if reserve_in == 0 || reserve_out == 0 {
        return Err("INSUFFICIENT_LIQUIDITY");
    }
    
    // Use BigUint for precise calculation
    let amount_in_big = amount_in.to_biguint().unwrap();
    let reserve_in_big = reserve_in.to_biguint().unwrap();
    let reserve_out_big = reserve_out.to_biguint().unwrap();
    let fee_multiplier = (10000u32 - fee).to_biguint().unwrap();
    let ten_thousand = 10000u32.to_biguint().unwrap();
    
    // amount_in_with_fee = amount_in * (10000 - fee)
    let amount_in_with_fee = &amount_in_big * &fee_multiplier;
    
    // numerator = amount_in_with_fee * reserve_out
    let numerator = &amount_in_with_fee * &reserve_out_big;
    
    // denominator = reserve_in * 10000 + amount_in_with_fee
    let denominator = &reserve_in_big * &ten_thousand + &amount_in_with_fee;
    
    // amount_out = numerator / denominator
    let amount_out = numerator / denominator;
    
    amount_out.to_u128().ok_or("OVERFLOW")
}

/// Calculate required input amount for desired output in UniswapV2
pub fn get_amount_in(
    amount_out: u128,
    reserve_in: u128,
    reserve_out: u128,
    fee: u32,
) -> Result<u128, &'static str> {
    if amount_out == 0 {
        return Err("INSUFFICIENT_OUTPUT_AMOUNT");
    }
    if reserve_in == 0 || reserve_out == 0 {
        return Err("INSUFFICIENT_LIQUIDITY");
    }
    if amount_out >= reserve_out {
        return Err("EXCESSIVE_OUTPUT_AMOUNT");
    }
    
    let amount_out_big = amount_out.to_biguint().unwrap();
    let reserve_in_big = reserve_in.to_biguint().unwrap();
    let reserve_out_big = reserve_out.to_biguint().unwrap();
    let fee_multiplier = (10000u32 - fee).to_biguint().unwrap();
    let ten_thousand = 10000u32.to_biguint().unwrap();
    let one = BigUint::one();
    
    // numerator = reserve_in * amount_out * 10000
    let numerator = &reserve_in_big * &amount_out_big * &ten_thousand;
    
    // denominator = (reserve_out - amount_out) * (10000 - fee)
    let denominator = (&reserve_out_big - &amount_out_big) * &fee_multiplier;
    
    // amount_in = (numerator / denominator) + 1
    let amount_in = (numerator / denominator) + one;
    
    amount_in.to_u128().ok_or("OVERFLOW")
}

/// Quote function for price calculation without fee
pub fn quote(
    amount_a: u128,
    reserve_a: u128,
    reserve_b: u128,
) -> Result<u128, &'static str> {
    if amount_a == 0 {
        return Err("INSUFFICIENT_AMOUNT");
    }
    if reserve_a == 0 || reserve_b == 0 {
        return Err("INSUFFICIENT_LIQUIDITY");
    }
    
    let amount_a_big = amount_a.to_biguint().unwrap();
    let reserve_a_big = reserve_a.to_biguint().unwrap();
    let reserve_b_big = reserve_b.to_biguint().unwrap();
    
    // amount_b = amount_a * reserve_b / reserve_a
    let amount_b = amount_a_big * reserve_b_big / reserve_a_big;
    
    amount_b.to_u128().ok_or("OVERFLOW")
}

/// Calculate amounts for multi-hop swap
pub fn get_amounts_out(
    amount_in: u128,
    path: &[(u128, u128, u32)], // Vec of (reserve_in, reserve_out, fee)
) -> Result<Vec<u128>, &'static str> {
    if path.len() < 1 {
        return Err("INVALID_PATH");
    }
    
    let mut amounts = vec![amount_in];
    
    for (reserve_in, reserve_out, fee) in path {
        let amount_out = get_amount_out(
            *amounts.last().unwrap(),
            *reserve_in,
            *reserve_out,
            *fee
        )?;
        amounts.push(amount_out);
    }
    
    Ok(amounts)
}

/// Calculate required input for multi-hop swap
pub fn get_amounts_in(
    amount_out: u128,
    path: &[(u128, u128, u32)], // Vec of (reserve_in, reserve_out, fee)
) -> Result<Vec<u128>, &'static str> {
    if path.len() < 1 {
        return Err("INVALID_PATH");
    }
    
    let mut amounts = vec![amount_out];
    
    for (reserve_in, reserve_out, fee) in path.iter().rev() {
        let amount_in = get_amount_in(
            amounts[0],
            *reserve_in,
            *reserve_out,
            *fee
        )?;
        amounts.insert(0, amount_in);
    }
    
    Ok(amounts)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_get_amount_out() {
        // Test with 0.3% fee (30 basis points)
        let amount_out = get_amount_out(
            1000000000000000000, // 1 token
            10000000000000000000000, // 10000 tokens reserve in
            20000000000000000000000, // 20000 tokens reserve out
            30 // 0.3% fee
        ).unwrap();
        
        // Should get approximately 1.997 tokens out
        assert!(amount_out > 1990000000000000000);
        assert!(amount_out < 2000000000000000000);
    }
}