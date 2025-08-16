"""
AMM Math - Python bindings to Rust core for exact DEX calculations
"""
import sys
import math
from typing import Tuple, Optional
from decimal import Decimal

# This would normally bind to Rust via pyo3, but for demo we implement in Python
class AMMCalculator:
    
    @staticmethod
    def uniswap_v2_get_amount_out(
        amount_in: int,
        reserve_in: int, 
        reserve_out: int,
        fee_bps: int = 30  # 0.3% = 30 bps
    ) -> int:
        """Calculate exact output for Uniswap V2 swap"""
        if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
            return 0
            
        # Apply fee: amount_in_with_fee = amount_in * (10000 - fee_bps) / 10000
        amount_in_with_fee = amount_in * (10000 - fee_bps)
        numerator = amount_in_with_fee * reserve_out
        denominator = (reserve_in * 10000) + amount_in_with_fee
        
        return numerator // denominator
    
    @staticmethod
    def uniswap_v2_get_amount_in(
        amount_out: int,
        reserve_in: int,
        reserve_out: int, 
        fee_bps: int = 30
    ) -> int:
        """Calculate exact input needed for Uniswap V2 swap"""
        if amount_out <= 0 or reserve_in <= 0 or reserve_out <= 0:
            return 0
            
        if amount_out >= reserve_out:
            raise ValueError("Insufficient liquidity")
            
        numerator = reserve_in * amount_out * 10000
        denominator = (reserve_out - amount_out) * (10000 - fee_bps)
        
        return (numerator // denominator) + 1
    
    @staticmethod 
    def sqrt_price_x96_to_price(sqrt_price_x96: int, decimals0: int = 18, decimals1: int = 18) -> float:
        """Convert Uniswap V3 sqrtPriceX96 to human readable price"""
        if sqrt_price_x96 <= 0:
            return 0.0
            
        # price = (sqrtPriceX96 / 2^96)^2 * 10^(decimals0 - decimals1)
        sqrt_price = sqrt_price_x96 / (2 ** 96)
        price = sqrt_price ** 2
        
        # Adjust for decimals
        decimal_adjustment = 10 ** (decimals0 - decimals1)
        return price * decimal_adjustment
    
    @staticmethod
    def tick_to_sqrt_price_x96(tick: int) -> int:
        """Convert tick to sqrtPriceX96"""
        # sqrt(1.0001^tick) * 2^96
        sqrt_price = math.sqrt(1.0001 ** tick)
        return int(sqrt_price * (2 ** 96))
    
    @staticmethod
    def sqrt_price_x96_to_tick(sqrt_price_x96: int) -> int:
        """Convert sqrtPriceX96 to tick"""
        if sqrt_price_x96 <= 0:
            return 0
            
        sqrt_price = sqrt_price_x96 / (2 ** 96)
        price = sqrt_price ** 2
        
        # tick = log(price) / log(1.0001)
        return int(math.log(price) / math.log(1.0001))
    
    @staticmethod
    def uniswap_v3_get_amount_out(
        amount_in: int,
        sqrt_price_x96: int,
        liquidity: int,
        fee: int,
        zero_for_one: bool
    ) -> Tuple[int, int]:  # amount_out, new_sqrt_price_x96
        """
        Calculate Uniswap V3 swap output (simplified single-tick)
        In production, this would handle tick crossing in Rust
        """
        if amount_in <= 0 or liquidity <= 0:
            return 0, sqrt_price_x96
            
        # Apply fee
        amount_in_less_fee = amount_in - (amount_in * fee // 1_000_000)
        
        if zero_for_one:  # Selling token0 for token1
            # Simplified calculation - in reality needs tick crossing logic
            sqrt_price = sqrt_price_x96 / (2 ** 96)
            
            # Delta sqrt price = amount_in / liquidity
            delta_sqrt_price = amount_in_less_fee / liquidity
            new_sqrt_price = sqrt_price - delta_sqrt_price
            
            if new_sqrt_price <= 0:
                return 0, sqrt_price_x96
                
            new_sqrt_price_x96 = int(new_sqrt_price * (2 ** 96))
            
            # amount_out = liquidity * (sqrt_price - new_sqrt_price)
            amount_out = int(liquidity * (sqrt_price - new_sqrt_price))
            
        else:  # Selling token1 for token0
            sqrt_price = sqrt_price_x96 / (2 ** 96)
            
            # For token1 -> token0: delta_sqrt_price = amount_in / (liquidity * sqrt_price)  
            delta_sqrt_price = amount_in_less_fee / (liquidity * sqrt_price)
            new_sqrt_price = sqrt_price + delta_sqrt_price
            new_sqrt_price_x96 = int(new_sqrt_price * (2 ** 96))
            
            # amount_out = liquidity * (new_sqrt_price - sqrt_price) / (sqrt_price * new_sqrt_price)
            amount_out = int(liquidity * delta_sqrt_price / sqrt_price)
            
        return max(0, amount_out), new_sqrt_price_x96
    
    @staticmethod
    def solidly_get_amount_out(
        amount_in: int,
        reserve_in: int,
        reserve_out: int,
        stable: bool = False,
        fee_bps: int = 20  # 0.2% for stable, 0.3% for volatile
    ) -> int:
        """Calculate Solidly-style AMM output"""
        if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
            return 0
            
        amount_in_with_fee = amount_in * (10000 - fee_bps) // 10000
        
        if stable:
            # Stable swap: more complex curve math
            # Simplified for demo - use constant sum with curve adjustment
            xy = reserve_in * reserve_out
            x = reserve_in + amount_in_with_fee
            y = xy // x
            return reserve_out - y
        else:
            # Volatile: same as Uniswap V2
            return AMMCalculator.uniswap_v2_get_amount_out(
                amount_in, reserve_in, reserve_out, fee_bps
            )

# For production, this would be:
# import rust_core
# class AMMCalculator:
#     @staticmethod 
#     def uniswap_v2_get_amount_out(*args):
#         return rust_core.univ2_get_amount_out(*args)
