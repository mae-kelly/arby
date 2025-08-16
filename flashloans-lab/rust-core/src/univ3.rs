//! UniswapV3 concentrated liquidity AMM math - Production Ready

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};

const Q96: u128 = 0x1000000000000000000000000; // 2^96

/// Full tick information
#[derive(Debug, Clone)]
pub struct Tick {
    pub liquidity_gross: u128,
    pub liquidity_net: i128,
    pub fee_growth_outside_0_x128: u128,
    pub fee_growth_outside_1_x128: u128,
    pub tick_cumulative_outside: i64,
    pub seconds_per_liquidity_outside_x128: u128,
    pub seconds_outside: u32,
    pub initialized: bool,
}

/// Swap state during computation
#[derive(Debug, Clone)]
pub struct SwapState {
    pub amount_specified_remaining: i128,
    pub amount_calculated: i128,
    pub sqrt_price_x96: u128,
    pub tick: i32,
    pub fee_growth_global_x128: u128,
    pub protocol_fee: u128,
    pub liquidity: u128,
}

/// Step computation result
#[derive(Debug, Clone)]
pub struct StepComputations {
    pub sqrt_price_start_x96: u128,
    pub tick_next: i32,
    pub initialized: bool,
    pub sqrt_price_next_x96: u128,
    pub amount_in: u128,
    pub amount_out: u128,
    pub fee_amount: u128,
}

/// Get amount0 delta between two sqrt prices
pub fn get_amount0_delta(
    sqrt_ratio_a_x96: u128,
    sqrt_ratio_b_x96: u128,
    liquidity: u128,
    round_up: bool,
) -> Result<u128, &'static str> {
    if sqrt_ratio_a_x96 > sqrt_ratio_b_x96 {
        return get_amount0_delta(sqrt_ratio_b_x96, sqrt_ratio_a_x96, liquidity, round_up);
    }
    
    let numerator = BigUint::from(liquidity) * BigUint::from(Q96) * 
                    (BigUint::from(sqrt_ratio_b_x96) - BigUint::from(sqrt_ratio_a_x96));
    let denominator = BigUint::from(sqrt_ratio_b_x96) * BigUint::from(sqrt_ratio_a_x96);
    
    let result = if round_up {
        (numerator + &denominator - BigUint::one()) / denominator
    } else {
        numerator / denominator
    };
    
    result.to_u128().ok_or("OVERFLOW")
}

/// Get amount1 delta between two sqrt prices
pub fn get_amount1_delta(
    sqrt_ratio_a_x96: u128,
    sqrt_ratio_b_x96: u128,
    liquidity: u128,
    round_up: bool,
) -> Result<u128, &'static str> {
    if sqrt_ratio_a_x96 > sqrt_ratio_b_x96 {
        return get_amount1_delta(sqrt_ratio_b_x96, sqrt_ratio_a_x96, liquidity, round_up);
    }
    
    let delta = BigUint::from(sqrt_ratio_b_x96) - BigUint::from(sqrt_ratio_a_x96);
    let result = if round_up {
        let numerator = BigUint::from(liquidity) * delta;
        (numerator + BigUint::from(Q96) - BigUint::one()) / BigUint::from(Q96)
    } else {
        BigUint::from(liquidity) * delta / BigUint::from(Q96)
    };
    
    result.to_u128().ok_or("OVERFLOW")
}

/// Get next sqrt price from input
pub fn get_next_sqrt_price_from_input(
    sqrt_price_x96: u128,
    liquidity: u128,
    amount_in: u128,
    zero_for_one: bool,
) -> Result<u128, &'static str> {
    if sqrt_price_x96 == 0 || liquidity == 0 {
        return Err("INVALID_PRICE_OR_LIQUIDITY");
    }
    
    if zero_for_one {
        get_next_sqrt_price_from_amount0_rounding_up(
            sqrt_price_x96,
            liquidity,
            amount_in,
            true
        )
    } else {
        get_next_sqrt_price_from_amount1_rounding_down(
            sqrt_price_x96,
            liquidity,
            amount_in,
            true
        )
    }
}

/// Get next sqrt price from output
pub fn get_next_sqrt_price_from_output(
    sqrt_price_x96: u128,
    liquidity: u128,
    amount_out: u128,
    zero_for_one: bool,
) -> Result<u128, &'static str> {
    if sqrt_price_x96 == 0 || liquidity == 0 {
        return Err("INVALID_PRICE_OR_LIQUIDITY");
    }
    
    if zero_for_one {
        get_next_sqrt_price_from_amount1_rounding_down(
            sqrt_price_x96,
            liquidity,
            amount_out,
            false
        )
    } else {
        get_next_sqrt_price_from_amount0_rounding_up(
            sqrt_price_x96,
            liquidity,
            amount_out,
            false
        )
    }
}

fn get_next_sqrt_price_from_amount0_rounding_up(
    sqrt_price_x96: u128,
    liquidity: u128,
    amount: u128,
    add: bool,
) -> Result<u128, &'static str> {
    if amount == 0 {
        return Ok(sqrt_price_x96);
    }
    
    let numerator1 = BigUint::from(liquidity) * BigUint::from(Q96);
    
    if add {
        let product = BigUint::from(amount) * BigUint::from(sqrt_price_x96);
        if product.clone() / BigUint::from(amount) == BigUint::from(sqrt_price_x96) {
            let denominator = numerator1.clone() + product;
            if denominator >= numerator1 {
                let result = (numerator1 * BigUint::from(sqrt_price_x96) + &denominator - BigUint::one()) / denominator;
                return result.to_u128().ok_or("OVERFLOW");
            }
        }
        
        let result = (numerator1 + BigUint::from(amount) * BigUint::from(sqrt_price_x96) - BigUint::one()) / 
                     BigUint::from(amount) + BigUint::from(sqrt_price_x96);
        result.to_u128().ok_or("OVERFLOW")
    } else {
        let product = BigUint::from(amount) * BigUint::from(sqrt_price_x96);
        if product.clone() / BigUint::from(amount) == BigUint::from(sqrt_price_x96) && numerator1 > product {
            let denominator = &numerator1 - &product;
            let result = (numerator1 * BigUint::from(sqrt_price_x96) + &denominator - BigUint::one()) / denominator;
            result.to_u128().ok_or("OVERFLOW")
        } else {
            Err("ARITHMETIC_ERROR")
        }
    }
}

fn get_next_sqrt_price_from_amount1_rounding_down(
    sqrt_price_x96: u128,
    liquidity: u128,
    amount: u128,
    add: bool,
) -> Result<u128, &'static str> {
    if add {
        let quotient = if amount <= u128::MAX {
            (BigUint::from(amount) * BigUint::from(Q96)) / BigUint::from(liquidity)
        } else {
            BigUint::from(amount) / (BigUint::from(liquidity) / BigUint::from(Q96))
        };
        
        let result = BigUint::from(sqrt_price_x96) + quotient;
        result.to_u128().ok_or("OVERFLOW")
    } else {
        let quotient = if amount <= u128::MAX {
            (BigUint::from(amount) * BigUint::from(Q96) + BigUint::from(liquidity) - BigUint::one()) / 
            BigUint::from(liquidity)
        } else {
            BigUint::from(amount) / (BigUint::from(liquidity) / BigUint::from(Q96))
        };
        
        if BigUint::from(sqrt_price_x96) > quotient {
            let result = BigUint::from(sqrt_price_x96) - quotient;
            result.to_u128().ok_or("OVERFLOW")
        } else {
            Err("ARITHMETIC_ERROR")
        }
    }
}

/// Compute swap step
pub fn compute_swap_step(
    sqrt_price_current_x96: u128,
    sqrt_price_target_x96: u128,
    liquidity: u128,
    amount_remaining: i128,
    fee_pips: u32,
) -> Result<StepComputations, &'static str> {
    let zero_for_one = sqrt_price_current_x96 >= sqrt_price_target_x96;
    let exact_in = amount_remaining >= 0;
    
    let mut step = StepComputations {
        sqrt_price_start_x96: sqrt_price_current_x96,
        tick_next: 0,
        initialized: false,
        sqrt_price_next_x96: sqrt_price_target_x96,
        amount_in: 0,
        amount_out: 0,
        fee_amount: 0,
    };
    
    let amount_remaining_less_fee = if exact_in {
        let fee_amount = (amount_remaining.abs() as u128 * fee_pips as u128) / 1_000_000;
        amount_remaining - fee_amount as i128
    } else {
        amount_remaining
    };
    
    if exact_in {
        step.amount_in = if zero_for_one {
            get_amount0_delta(sqrt_price_target_x96, sqrt_price_current_x96, liquidity, true)?
        } else {
            get_amount1_delta(sqrt_price_current_x96, sqrt_price_target_x96, liquidity, true)?
        };
        
        if amount_remaining_less_fee.abs() as u128 >= step.amount_in {
            step.sqrt_price_next_x96 = sqrt_price_target_x96;
        } else {
            step.sqrt_price_next_x96 = get_next_sqrt_price_from_input(
                sqrt_price_current_x96,
                liquidity,
                amount_remaining_less_fee.abs() as u128,
                zero_for_one
            )?;
        }
    } else {
        step.amount_out = if zero_for_one {
            get_amount1_delta(sqrt_price_target_x96, sqrt_price_current_x96, liquidity, false)?
        } else {
            get_amount0_delta(sqrt_price_current_x96, sqrt_price_target_x96, liquidity, false)?
        };
        
        if amount_remaining.abs() as u128 >= step.amount_out {
            step.sqrt_price_next_x96 = sqrt_price_target_x96;
        } else {
            step.sqrt_price_next_x96 = get_next_sqrt_price_from_output(
                sqrt_price_current_x96,
                liquidity,
                amount_remaining.abs() as u128,
                zero_for_one
            )?;
        }
    }
    
    let max = sqrt_price_target_x96 == step.sqrt_price_next_x96;
    
    if zero_for_one {
        if !max || !exact_in {
            step.amount_in = get_amount0_delta(
                step.sqrt_price_next_x96,
                sqrt_price_current_x96,
                liquidity,
                true
            )?;
        }
        if !max || exact_in {
            step.amount_out = get_amount1_delta(
                step.sqrt_price_next_x96,
                sqrt_price_current_x96,
                liquidity,
                false
            )?;
        }
    } else {
        if !max || !exact_in {
            step.amount_in = get_amount1_delta(
                sqrt_price_current_x96,
                step.sqrt_price_next_x96,
                liquidity,
                true
            )?;
        }
        if !max || exact_in {
            step.amount_out = get_amount0_delta(
                sqrt_price_current_x96,
                step.sqrt_price_next_x96,
                liquidity,
                false
            )?;
        }
    }
    
    if !exact_in && step.amount_out > amount_remaining.abs() as u128 {
        step.amount_out = amount_remaining.abs() as u128;
    }
    
    if exact_in && step.sqrt_price_next_x96 != sqrt_price_target_x96 {
        step.fee_amount = (amount_remaining.abs() as u128) - step.amount_in;
    } else {
        step.fee_amount = (step.amount_in * fee_pips as u128) / (1_000_000 - fee_pips as u128);
    }
    
    Ok(step)
}

/// Get tick at sqrt ratio
pub fn get_tick_at_sqrt_ratio(sqrt_price_x96: u128) -> Result<i32, &'static str> {
    // MIN_SQRT_RATIO = 4295128739
    // MAX_SQRT_RATIO is too large for u128, using BigUint
    if sqrt_price_x96 < 4295128739 {
        return Err("SQRT_RATIO_OUT_OF_BOUNDS");
    }
    
    let max_sqrt_ratio = BigUint::parse_bytes(b"1461446703485210103287273052203988822378723970342", 10).unwrap();
    if BigUint::from(sqrt_price_x96) > max_sqrt_ratio {
        return Err("SQRT_RATIO_OUT_OF_BOUNDS");
    }
    
    let ratio = BigUint::from(sqrt_price_x96) * BigUint::from(sqrt_price_x96) / BigUint::from(Q96);
    
    let mut r = ratio.clone();
    let mut msb = 0u8;
    
    // Binary search for most significant bit
    let checks = vec![
        (BigUint::parse_bytes(b"340282366920938463463374607431768211455", 10).unwrap(), 128),
        (BigUint::from(0xFFFFFFFFFFFFFFFFu64), 64),
        (BigUint::from(0xFFFFFFFFu32), 32),
        (BigUint::from(0xFFFFu16), 16),
        (BigUint::from(0xFFu8), 8),
        (BigUint::from(0xFu8), 4),
        (BigUint::from(0x3u8), 2),
        (BigUint::from(0x1u8), 1),
    ];
    
    for (mask, shift) in checks {
        let f = if r > mask { 1 } else { 0 };
        msb |= f << shift;
        r >>= f * shift;
    }
    
    // Fix: Use i128 directly to avoid overflow
    // The value (msb - 128) << 64 needs to be calculated in i128 space
    let msb_adjusted = (msb as i128) - 128;
    let log_2 = msb_adjusted.wrapping_shl(64);
    
    // Calculate log base 1.0001
    let log_base = (log_2 as f64 / (1i128 << 64) as f64) / (10000_f64.ln() / 9999_f64.ln());
    let tick = log_base.floor() as i32;
    
    Ok(tick)
}

/// Get sqrt ratio at tick
pub fn get_sqrt_ratio_at_tick(tick: i32) -> Result<u128, &'static str> {
    if tick < -887272 || tick > 887272 {
        return Err("TICK_OUT_OF_BOUNDS");
    }
    
    let abs_tick = tick.abs() as u32;
    
    // Q128 as BigUint with explicit type
    let q128: BigUint = BigUint::from(1u8) << 128;
    
    let mut ratio = if abs_tick & 0x1 != 0 {
        BigUint::from(0xfffcb933bd6fad37aa2d162d1a594001u128)
    } else {
        q128.clone()
    };
    
    // Compute 1.0001^tick using binary decomposition
    let multipliers = vec![
        (0x2, BigUint::from(0xfff97272373d413259a46990580e213au128)),
        (0x4, BigUint::from(0xfff2e50f5f656932ef12357cf3c7fdccu128)),
        (0x8, BigUint::from(0xffe5caca7e10e4e61c3624eaa0941cd0u128)),
        (0x10, BigUint::from(0xffcb9843d60f6159c9db58835c926644u128)),
        (0x20, BigUint::from(0xff973b41fa98c081472e6896dfb254c0u128)),
        (0x40, BigUint::from(0xff2ea16466c96a3843ec78b326b52861u128)),
        (0x80, BigUint::from(0xfe5dee046a99a2a811c461f1969c3053u128)),
        (0x100, BigUint::from(0xfcbe86c7900a88aedcffc83b479aa3a4u128)),
        (0x200, BigUint::from(0xf987a7253ac413176f2b074cf7815e54u128)),
        (0x400, BigUint::from(0xf3392b0822b70005940c7a398e4b70f3u128)),
        (0x800, BigUint::from(0xe7159475a2c29b7443b29c7fa6e889d9u128)),
        (0x1000, BigUint::from(0xd097f3bdfd2022b8845ad8f792aa5825u128)),
        (0x2000, BigUint::from(0xa9f746462d870fdf8a65dc1f90e061e5u128)),
        (0x4000, BigUint::from(0x70d869a156d2a1b890bb3df62baf32f7u128)),
        (0x8000, BigUint::from(0x31be135f97d08fd981231505542fcfa6u128)),
        (0x10000, BigUint::from(0x9aa508b5b7a84e1c677de54f3e99bc9u128)),
        (0x20000, BigUint::from(0x5d6af8dedb81196699c329225ee604u128)),
        (0x40000, BigUint::from(0x2216e584f5fa1ea926041bedfe98u128)),
        (0x80000, BigUint::from(0x48a170391f7dc42444e8fa2u128)),
    ];
    
    for (mask, multiplier) in multipliers {
        if abs_tick & mask != 0 {
            ratio = ratio * multiplier >> 128;
        }
    }
    
    if tick > 0 {
        ratio = BigUint::from(u128::MAX) / ratio;
    }
    
    // Clone ratio before the shift operation to avoid move
    let sqrt_price_x96: BigUint = (ratio.clone() >> 32) + if ratio % (BigUint::one() << 32) == BigUint::zero() { 
        BigUint::zero() 
    } else { 
        BigUint::one() 
    };
    
    sqrt_price_x96.to_u128().ok_or("OVERFLOW")
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sqrt_price_math() {
        let sqrt_price = 79228162514264337593543950336u128; // 1:1 price
        let tick = get_tick_at_sqrt_ratio(sqrt_price).unwrap();
        assert_eq!(tick, 0);
        
        let sqrt_price_recovered = get_sqrt_ratio_at_tick(0).unwrap();
        assert!((sqrt_price_recovered as i128 - sqrt_price as i128).abs() < 100000);
    }
}