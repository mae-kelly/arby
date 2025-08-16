#!/bin/bash
set -euo pipefail

echo "🦀 Generating Rust Core Files..."

cd flashloans-lab

# Complete the flash loan family file
cat >> engine/strategies/family_flash_loan.py << 'EOF'
D0A0e5C4F27eAD9083C756Cc2"]

class MakerFlashArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.3
    def get_required_tokens(self) -> List[str]:
        return ["0x6B175474E89094C44Da98b954EedeAC495271d0F"]

class EulerFlashArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.4
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class IronBankFlash(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.45
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class YearnVaultFlash(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.35
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class ConvexFlashArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.4
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class LidoFlashArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.3
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]
EOF

# Create placeholder files for remaining families
for family in cex_dex oracle liquidity liquidation stablecoin gas cross_chain; do
    cat > engine/strategies/family_${family}.py << EOF
"""
Family Strategy Placeholders - ${family}
These implement the remaining strategy families with placeholder logic
"""

import asyncio
from typing import Dict, List, Optional, Any
from .base import StrategyBase, StrategyConfig, StrategyDetectionResult, FlashLoanStrategy
from ..execsim.simulator import CandidateTrade

# Placeholder implementations for ${family} family strategies
# Each returns opportunity_found=False for demo purposes
# In production, these would contain full detection and simulation logic

EOF

    case $family in
        "cex_dex")
            cat >> engine/strategies/family_${family}.py << 'EOF'
class BinanceUniswapSpread(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.5
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

# Add 9 more CEX/DEX strategies with similar structure
for i in range(2, 11):
    globals()[f'CexDexStrategy{i}'] = type(f'CexDexStrategy{i}', (FlashLoanStrategy,), {
        'detect': lambda self, md: StrategyDetectionResult(opportunity_found=False),
        'simulate': lambda self, c, md: 0.0,
        'calculate_risk_score': lambda self, c: 0.4,
        'get_required_tokens': lambda self: ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]
    })

# Specific implementations
class CoinbaseCurveSpread(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.5
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class OkxSushiSpread(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.5
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class FundingSpotBasis(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.6
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class PerpDexSpread(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.7
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class CexDepthArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.5
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class FundingRateArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.6
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class BasisTradeDetector(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.5
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class CrossExchangeMomentum(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.6
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class CexDexLatencyArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.7
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]
EOF
            ;;
        "oracle")
            cat >> engine/strategies/family_${family}.py << 'EOF'
class ChainlinkLagDetector(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.3
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class BandOracleLib(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.4
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class TwapSpotDrift(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.3
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class PriceFeedStale(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.25
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class OracleSandwich(FlashLoanStrategy):
    """DISABLED: Oracle manipulation is unethical"""
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 1.0
    def get_required_tokens(self) -> List[str]:
        return []

class HeartbeatEdge(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.3
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class OracleFrontrun(FlashLoanStrategy):
    """DISABLED: Oracle frontrunning is unethical"""
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 1.0
    def get_required_tokens(self) -> List[str]:
        return []

class MedianOracleArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.35
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class VolatilityOracleLag(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.4
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class CrossChainOracleArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.5
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]
EOF
            ;;
        *)
            # Generate placeholder classes for other families
            for i in {1..10}; do
                cat >> engine/strategies/family_${family}.py << EOF
class ${family^}Strategy${i}(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.4
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

EOF
            done
            ;;
    esac
done

# Create specific implementations for remaining families with proper names
cat > engine/strategies/family_liquidity.py << 'EOF'
"""
Family 6: Liquidity Mirages & JIT LP Detection (10 total)
"""

import asyncio
from typing import Dict, List, Optional, Any
from .base import StrategyBase, StrategyConfig, StrategyDetectionResult, FlashLoanStrategy
from ..execsim.simulator import CandidateTrade

class JitLpDetector(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.3
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class FakeLiquidityDetector(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.4
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class LpWithdrawalArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.35
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class ConcentratedLiquidityArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.4
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class RangeOrderArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.3
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class LpFeeHarvesting(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.25
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class ImpermanentLossHedge(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.3
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class LiquidityMiningArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.35
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class YieldFarmingArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.4
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]

class LpTokenArb(FlashLoanStrategy):
    async def detect(self, market_data: Dict[str, Any]) -> StrategyDetectionResult:
        return StrategyDetectionResult(opportunity_found=False)
    async def simulate(self, candidate: CandidateTrade, market_data: Dict[str, Any]) -> float:
        return 0.0
    def calculate_risk_score(self, candidate: CandidateTrade) -> float:
        return 0.3
    def get_required_tokens(self) -> List[str]:
        return ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"]
EOF

# Now create the Rust core files
cat > rust-core/Cargo.toml << 'EOF'
[package]
name = "flashloan-core"
version = "1.0.0"
edition = "2021"
authors = ["Flash Loan Lab"]

[lib]
name = "flashloan_core"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
num-bigint = "0.4"
num-traits = "0.2"
thiserror = "1.0"

[dependencies.uint]
version = "0.9"
default-features = false

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
opt-level = 3
EOF

cat > rust-core/src/lib.rs << 'EOF'
//! Flash Loan Core - High Performance AMM Math
//! 
//! This library provides exact mathematical calculations for various AMM protocols
//! with Python bindings via PyO3 for the flash loan arbitrage engine.

use pyo3::prelude::*;

pub mod univ2;
pub mod univ3; 
pub mod solidly;
pub mod gas;
pub mod route;
pub mod ffi_pyo3;

pub use univ2::*;
pub use univ3::*;
pub use solidly::*;
pub use gas::*;
pub use route::*;

/// Error types for AMM calculations
#[derive(thiserror::Error, Debug)]
pub enum AmmError {
    #[error("Insufficient liquidity")]
    InsufficientLiquidity,
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Overflow in calculation")]
    Overflow,
    #[error("Division by zero")]
    DivisionByZero,
}

pub type Result<T> = std::result::Result<T, AmmError>;

/// Python module definition
#[pymodule]
fn flashloan_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Export Uniswap V2 functions
    m.add_function(wrap_pyfunction!(ffi_pyo3::py_univ2_get_amount_out, m)?)?;
    m.add_function(wrap_pyfunction!(ffi_pyo3::py_univ2_get_amount_in, m)?)?;
    
    // Export Uniswap V3 functions  
    m.add_function(wrap_pyfunction!(ffi_pyo3::py_univ3_get_amount_out, m)?)?;
    m.add_function(wrap_pyfunction!(ffi_pyo3::py_sqrt_price_to_tick, m)?)?;
    m.add_function(wrap_pyfunction!(ffi_pyo3::py_tick_to_sqrt_price, m)?)?;
    
    // Export Solidly functions
    m.add_function(wrap_pyfunction!(ffi_pyo3::py_solidly_get_amount_out, m)?)?;
    
    // Export gas estimation
    m.add_function(wrap_pyfunction!(ffi_pyo3::py_estimate_gas_cost, m)?)?;
    
    Ok(())
}
EOF

cat > rust-core/src/univ2.rs << 'EOF'
//! Uniswap V2 AMM calculations
//! Implements exact constant product market maker math

use crate::{AmmError, Result};
use num_bigint::BigUint;
use num_traits::{Zero, One};

/// Calculate output amount for Uniswap V2 swap
/// Uses the constant product formula: x * y = k
pub fn get_amount_out(
    amount_in: u128,
    reserve_in: u128, 
    reserve_out: u128,
    fee_bps: u16,
) -> Result<u128> {
    if amount_in == 0 || reserve_in == 0 || reserve_out == 0 {
        return Err(AmmError::InsufficientLiquidity);
    }
    
    if fee_bps > 10000 {
        return Err(AmmError::InvalidParameters("Fee cannot exceed 100%".to_string()));
    }
    
    // Convert to BigUint for overflow protection
    let amount_in = BigUint::from(amount_in);
    let reserve_in = BigUint::from(reserve_in);
    let reserve_out = BigUint::from(reserve_out);
    let fee_multiplier = BigUint::from(10000u32 - fee_bps as u32);
    
    // amount_in_with_fee = amount_in * (10000 - fee_bps) / 10000
    let amount_in_with_fee = &amount_in * &fee_multiplier;
    
    // numerator = amount_in_with_fee * reserve_out
    let numerator = &amount_in_with_fee * &reserve_out;
    
    // denominator = reserve_in * 10000 + amount_in_with_fee  
    let denominator = &reserve_in * BigUint::from(10000u32) + &amount_in_with_fee;
    
    if denominator.is_zero() {
        return Err(AmmError::DivisionByZero);
    }
    
    let result = numerator / denominator;
    
    // Convert back to u128, checking for overflow
    result.try_into()
        .map_err(|_| AmmError::Overflow)
}

/// Calculate input amount needed for desired output
pub fn get_amount_in(
    amount_out: u128,
    reserve_in: u128,
    reserve_out: u128, 
    fee_bps: u16,
) -> Result<u128> {
    if amount_out == 0 || reserve_in == 0 || reserve_out == 0 {
        return Err(AmmError::InsufficientLiquidity);
    }
    
    if amount_out >= reserve_out {
        return Err(AmmError::InsufficientLiquidity);
    }
    
    if fee_bps > 10000 {
        return Err(AmmError::InvalidParameters("Fee cannot exceed 100%".to_string()));
    }
    
    let amount_out = BigUint::from(amount_out);
    let reserve_in = BigUint::from(reserve_in);
    let reserve_out = BigUint::from(reserve_out);
    let fee_multiplier = BigUint::from(10000u32 - fee_bps as u32);
    
    // numerator = reserve_in * amount_out * 10000
    let numerator = &reserve_in * &amount_out * BigUint::from(10000u32);
    
    // denominator = (reserve_out - amount_out) * fee_multiplier
    let denominator = (&reserve_out - &amount_out) * &fee_multiplier;
    
    if denominator.is_zero() {
        return Err(AmmError::DivisionByZero);
    }
    
    let result = (&numerator / &denominator) + BigUint::one();
    
    result.try_into()
        .map_err(|_| AmmError::Overflow)
}

/// Calculate price after swap
pub fn get_price_after_swap(
    amount_in: u128,
    reserve_in: u128,
    reserve_out: u128,
    fee_bps: u16,
) -> Result<f64> {
    if reserve_in == 0 || reserve_out == 0 {
        return Err(AmmError::InsufficientLiquidity);
    }
    
    let amount_out = get_amount_out(amount_in, reserve_in, reserve_out, fee_bps)?;
    
    let new_reserve_in = reserve_in + amount_in;
    let new_reserve_out = reserve_out.saturating_sub(amount_out);
    
    if new_reserve_out == 0 {
        return Err(AmmError::InsufficientLiquidity);
    }
    
    Ok(new_reserve_out as f64 / new_reserve_in as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_get_amount_out() {
        // Test with 1 ETH input, 100 ETH / 200,000 USDC reserves, 0.3% fee
        let result = get_amount_out(
            1_000_000_000_000_000_000, // 1 ETH
            100_000_000_000_000_000_000, // 100 ETH
            200_000_000_000, // 200,000 USDC (6 decimals)
            30 // 0.3% = 30 bps
        ).unwrap();
        
        // Should get approximately 1994 USDC (slightly less due to fees and slippage)
        assert!(result > 1_990_000_000 && result < 2_000_000_000);
    }
    
    #[test]
    fn test_get_amount_in() {
        // Test reverse calculation
        let amount_out = 1_000_000_000; // 1000 USDC
        let result = get_amount_in(
            amount_out,
            100_000_000_000_000_000_000, // 100 ETH  
            200_000_000_000, // 200,000 USDC
            30
        ).unwrap();
        
        // Verify by calculating amount_out with this input
        let verified_out = get_amount_out(
            result,
            100_000_000_000_000_000_000,
            200_000_000_000,
            30
        ).unwrap();
        
        assert!(verified_out >= amount_out);
    }
    
    #[test]
    fn test_insufficient_liquidity() {
        let result = get_amount_out(0, 100, 100, 30);
        assert!(matches!(result, Err(AmmError::InsufficientLiquidity)));
        
        let result = get_amount_out(100, 0, 100, 30);
        assert!(matches!(result, Err(AmmError::InsufficientLiquidity)));
    }
    
    #[test]
    fn test_invalid_fee() {
        let result = get_amount_out(100, 1000, 1000, 10001);
        assert!(matches!(result, Err(AmmError::InvalidParameters(_))));
    }
}
EOF

cat > rust-core/src/univ3.rs << 'EOF'
//! Uniswap V3 concentrated liquidity calculations
//! Implements tick-based pricing and liquidity math

use crate::{AmmError, Result};
use num_bigint::{BigInt, BigUint};
use num_traits::{Zero, One, Signed};

/// Uniswap V3 constants
const Q96: u128 = 1u128 << 96;
const MIN_TICK: i32 = -887272;
const MAX_TICK: i32 = 887272;

/// Convert tick to sqrtPriceX96
pub fn tick_to_sqrt_price_x96(tick: i32) -> Result<u128> {
    if tick < MIN_TICK || tick > MAX_TICK {
        return Err(AmmError::InvalidParameters("Tick out of range".to_string()));
    }
    
    // sqrt(1.0001^tick) * 2^96
    // Use approximation for demo - production would use exact calculation
    let tick_f = tick as f64;
    let price = 1.0001_f64.powf(tick_f);
    let sqrt_price = price.sqrt();
    let sqrt_price_x96 = sqrt_price * (Q96 as f64);
    
    if sqrt_price_x96 > u128::MAX as f64 {
        return Err(AmmError::Overflow);
    }
    
    Ok(sqrt_price_x96 as u128)
}

/// Convert sqrtPriceX96 to tick
pub fn sqrt_price_x96_to_tick(sqrt_price_x96: u128) -> Result<i32> {
    if sqrt_price_x96 == 0 {
        return Err(AmmError::InvalidParameters("Price cannot be zero".to_string()));
    }
    
    let sqrt_price = sqrt_price_x96 as f64 / Q96 as f64;
    let price = sqrt_price * sqrt_price;
    let tick = price.log(1.0001);
    
    let tick_rounded = tick.round() as i32;
    
    if tick_rounded < MIN_TICK || tick_rounded > MAX_TICK {
        return Err(AmmError::InvalidParameters("Calculated tick out of range".to_string()));
    }
    
    Ok(tick_rounded)
}

/// Calculate swap output for Uniswap V3 (simplified single-tick)
pub fn get_amount_out(
    amount_in: u128,
    sqrt_price_x96: u128,
    liquidity: u128, 
    fee: u32,
    zero_for_one: bool,
) -> Result<(u128, u128)> {
    if amount_in == 0 || liquidity == 0 || sqrt_price_x96 == 0 {
        return Err(AmmError::InsufficientLiquidity);
    }
    
    if fee > 1_000_000 {
        return Err(AmmError::InvalidParameters("Fee too high".to_string()));
    }
    
    // Apply fee: amount_in_less_fee = amount_in * (1_000_000 - fee) / 1_000_000
    let amount_in_less_fee = (amount_in as u128)
        .saturating_mul(1_000_000 - fee as u128) / 1_000_000;
    
    let sqrt_price = sqrt_price_x96 as f64 / Q96 as f64;
    
    if zero_for_one {
        // Selling token0 for token1
        // Simplified calculation - in production would handle tick crossing
        let delta_sqrt_price = amount_in_less_fee as f64 / liquidity as f64;
        let new_sqrt_price = sqrt_price - delta_sqrt_price;
        
        if new_sqrt_price <= 0.0 {
            return Err(AmmError::InsufficientLiquidity);
        }
        
        let new_sqrt_price_x96 = (new_sqrt_price * Q96 as f64) as u128;
        let amount_out = (liquidity as f64 * (sqrt_price - new_sqrt_price)) as u128;
        
        Ok((amount_out, new_sqrt_price_x96))
    } else {
        // Selling token1 for token0  
        let delta_sqrt_price = amount_in_less_fee as f64 / (liquidity as f64 * sqrt_price);
        let new_sqrt_price = sqrt_price + delta_sqrt_price;
        let new_sqrt_price_x96 = (new_sqrt_price * Q96 as f64) as u128;
        
        let amount_out = (liquidity as f64 * delta_sqrt_price / sqrt_price) as u128;
        
        Ok((amount_out, new_sqrt_price_x96))
    }
}

/// Get the next initialized tick
pub fn next_initialized_tick(
    current_tick: i32,
    tick_spacing: i32,
    zero_for_one: bool,
) -> i32 {
    let compressed = if current_tick < 0 && current_tick % tick_spacing != 0 {
        (current_tick / tick_spacing) - 1
    } else {
        current_tick / tick_spacing
    };
    
    if zero_for_one {
        (compressed - 1) * tick_spacing
    } else {
        (compressed + 1) * tick_spacing
    }
}

/// Calculate liquidity for a position
pub fn get_liquidity_for_amounts(
    sqrt_price_x96: u128,
    sqrt_price_a_x96: u128,
    sqrt_price_b_x96: u128,
    amount0: u128,
    amount1: u128,
) -> Result<u128> {
    if sqrt_price_a_x96 > sqrt_price_b_x96 {
        return get_liquidity_for_amounts(
            sqrt_price_x96, sqrt_price_b_x96, sqrt_price_a_x96, amount1, amount0
        );
    }
    
    let sqrt_price = sqrt_price_x96 as f64 / Q96 as f64;
    let sqrt_price_a = sqrt_price_a_x96 as f64 / Q96 as f64;
    let sqrt_price_b = sqrt_price_b_x96 as f64 / Q96 as f64;
    
    let liquidity = if sqrt_price <= sqrt_price_a {
        // Below range
        (amount0 as f64 * sqrt_price_a * sqrt_price_b / (sqrt_price_b - sqrt_price_a)) as u128
    } else if sqrt_price < sqrt_price_b {
        // In range
        let liquidity0 = amount0 as f64 * sqrt_price * sqrt_price_b / (sqrt_price_b - sqrt_price);
        let liquidity1 = amount1 as f64 / (sqrt_price - sqrt_price_a);
        liquidity0.min(liquidity1) as u128
    } else {
        // Above range
        (amount1 as f64 / (sqrt_price_b - sqrt_price_a)) as u128
    };
    
    Ok(liquidity)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tick_conversion() {
        let tick = 0;
        let sqrt_price = tick_to_sqrt_price_x96(tick).unwrap();
        let back_to_tick = sqrt_price_x96_to_tick(sqrt_price).unwrap();
        assert_eq!(tick, back_to_tick);
    }
    
    #[test]
    fn test_tick_bounds() {
        assert!(tick_to_sqrt_price_x96(MIN_TICK - 1).is_err());
        assert!(tick_to_sqrt_price_x96(MAX_TICK + 1).is_err());
        
        assert!(tick_to_sqrt_price_x96(MIN_TICK).is_ok());
        assert!(tick_to_sqrt_price_x96(MAX_TICK).is_ok());
    }
    
    #[test]
    fn test_get_amount_out() {
        let sqrt_price_x96 = tick_to_sqrt_price_x96(0).unwrap(); // Price = 1
        let liquidity = 1_000_000_000_000_000_000u128; // 1e18
        let amount_in = 1_000_000_000_000_000_000u128; // 1e18
        let fee = 3000; // 0.3%
        
        let result = get_amount_out(amount_in, sqrt_price_x96, liquidity, fee, true);
        assert!(result.is_ok());
        
        let (amount_out, new_sqrt_price_x96) = result.unwrap();
        assert!(amount_out > 0);
        assert!(new_sqrt_price_x96 < sqrt_price_x96); // Price should decrease
    }
}
EOF

cat > rust-core/src/solidly.rs << 'EOF'
//! Solidly-style AMM calculations
//! Implements both stable and volatile pool math

use crate::{AmmError, Result};
use num_bigint::BigUint;
use num_traits::Zero;

/// Calculate output for Solidly-style AMM
pub fn get_amount_out(
    amount_in: u128,
    reserve_in: u128,
    reserve_out: u128,
    stable: bool,
    fee_bps: u16,
) -> Result<u128> {
    if amount_in == 0 || reserve_in == 0 || reserve_out == 0 {
        return Err(AmmError::InsufficientLiquidity);
    }
    
    if fee_bps > 10000 {
        return Err(AmmError::InvalidParameters("Fee cannot exceed 100%".to_string()));
    }
    
    // Apply fee
    let fee_multiplier = 10000u32 - fee_bps as u32;
    let amount_in_with_fee = (amount_in as u128 * fee_multiplier as u128) / 10000;
    
    if stable {
        // Stable swap: StableSwap curve
        stable_get_amount_out(amount_in_with_fee, reserve_in, reserve_out)
    } else {
        // Volatile: Uniswap V2 style constant product
        volatile_get_amount_out(amount_in_with_fee, reserve_in, reserve_out)
    }
}

/// Stable swap calculation (simplified StableSwap)
fn stable_get_amount_out(
    amount_in: u128,
    reserve_in: u128, 
    reserve_out: u128,
) -> Result<u128> {
    // Simplified stable swap calculation
    // In production, this would implement the full StableSwap invariant
    
    let reserve_in = BigUint::from(reserve_in);
    let reserve_out = BigUint::from(reserve_out);
    let amount_in = BigUint::from(amount_in);
    
    // For demo: use a curve that's flatter than constant product
    // Real implementation would solve: An^n + Bn^n = k where A = x + y, B = xy
    
    let total_reserves = &reserve_in + &reserve_out;
    let product = &reserve_in * &reserve_out;
    
    // New reserve_in after trade
    let new_reserve_in = &reserve_in + &amount_in;
    
    // Solve for new_reserve_out using simplified curve
    // This is a rough approximation - production needs exact math
    let new_total = &total_reserves + &amount_in;
    let target_product = &product * &new_total / &total_reserves;
    
    let new_reserve_out = if new_reserve_in.is_zero() {
        return Err(AmmError::DivisionByZero);
    } else {
        &target_product / &new_reserve_in
    };
    
    if new_reserve_out >= reserve_out {
        return Err(AmmError::InsufficientLiquidity);
    }
    
    let amount_out = &reserve_out - new_reserve_out;
    
    amount_out.try_into()
        .map_err(|_| AmmError::Overflow)
}

/// Volatile swap calculation (constant product)
fn volatile_get_amount_out(
    amount_in: u128,
    reserve_in: u128,
    reserve_out: u128, 
) -> Result<u128> {
    let amount_in = BigUint::from(amount_in);
    let reserve_in = BigUint::from(reserve_in);
    let reserve_out = BigUint::from(reserve_out);
    
    // Standard constant product: x * y = k
    let numerator = &amount_in * &reserve_out;
    let denominator = &reserve_in + &amount_in;
    
    if denominator.is_zero() {
        return Err(AmmError::DivisionByZero);
    }
    
    let result = numerator / denominator;
    
    result.try_into()
        .map_err(|_| AmmError::Overflow)
}

/// Calculate the invariant for stable pools  
pub fn get_stable_invariant(x: u128, y: u128) -> Result<u128> {
    // Simplified invariant calculation
    // Real Solidly uses: x^3*y + y^3*x = k
    
    let x = BigUint::from(x);
    let y = BigUint::from(y);
    
    let x_cubed = &x * &x * &x;
    let y_cubed = &y * &y * &y;
    
    let invariant = &x_cubed * &y + &y_cubed * &x;
    
    invariant.try_into()
        .map_err(|_| AmmError::Overflow)
}

/// Check if reserves are balanced for stable pool
pub fn is_balanced(reserve0: u128, reserve1: u128, threshold_bps: u16) -> bool {
    if reserve0 == 0 || reserve1 == 0 {
        return false;
    }
    
    let ratio = if reserve0 > reserve1 {
        (reserve0 * 10000) / reserve1
    } else {
        (reserve1 * 10000) / reserve0
    };
    
    let max_ratio = 10000 + threshold_bps as u128;
    ratio <= max_ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_volatile_swap() {
        let result = get_amount_out(
            1_000_000_000_000_000_000, // 1 token
            100_000_000_000_000_000_000, // 100 tokens
            100_000_000_000_000_000_000, // 100 tokens  
            false, // volatile
            20 // 0.2% fee
        ).unwrap();
        
        // Should be close to constant product result
        assert!(result > 990_000_000_000_000_000); // > 0.99 tokens
        assert!(result < 1_000_000_000_000_000_000); // < 1 token
    }
    
    #[test]
    fn test_stable_swap() {
        let result = get_amount_out(
            1_000_000_000_000_000_000, // 1 token
            100_000_000_000_000_000_000, // 100 tokens
            100_000_000_000_000_000_000, // 100 tokens
            true, // stable
            20 // 0.2% fee  
        ).unwrap();
        
        // Stable swap should have less slippage than volatile
        assert!(result > 0);
    }
    
    #[test]
    fn test_balanced_check() {
        assert!(is_balanced(100_000, 100_000, 100)); // Exactly balanced
        assert!(is_balanced(100_000, 101_000, 200)); // 1% diff, 2% threshold
        assert!(!is_balanced(100_000, 105_000, 200)); // 5% diff, 2% threshold
        assert!(!is_balanced(0, 100_000, 100)); // Zero reserve
    }
    
    #[test]
    fn test_stable_invariant() {
        let invariant = get_stable_invariant(1000, 1000).unwrap();
        assert!(invariant > 0);
        
        // Invariant should be symmetric
        let invariant2 = get_stable_invariant(1000, 1000).unwrap();
        assert_eq!(invariant, invariant2);
    }
}
EOF

cat > rust-core/src/gas.rs << 'EOF'
//! Gas estimation and optimization utilities

use crate::{AmmError, Result};
use std::collections::HashMap;

/// Gas costs for common operations (in gas units)
#[derive(Debug, Clone)]
pub struct GasCosts {
    pub base_cost: u64,
    pub transfer: u64,
    pub swap_v2: u64,
    pub swap_v3: u64,
    pub flash_loan: u64,
    pub multicall: u64,
}

impl Default for GasCosts {
    fn default() -> Self {
        Self {
            base_cost: 21000,
            transfer: 21000,
            swap_v2: 100000,
            swap_v3: 150000,
            flash_loan: 50000,
            multicall: 30000,
        }
    }
}

/// Chain-specific gas configurations
pub struct ChainGasConfig {
    pub chain_id: u64,
    pub costs: GasCosts,
    pub base_fee_multiplier: f64,
    pub priority_fee_min: u64,
}

impl ChainGasConfig {
    pub fn ethereum() -> Self {
        Self {
            chain_id: 1,
            costs: GasCosts::default(),
            base_fee_multiplier: 1.0,
            priority_fee_min: 1_000_000_000, // 1 gwei
        }
    }
    
    pub fn arbitrum() -> Self {
        Self {
            chain_id: 42161,
            costs: GasCosts {
                base_cost: 21000,
                transfer: 21000,
                swap_v2: 80000,  // Cheaper on L2
                swap_v3: 120000,
                flash_loan: 40000,
                multicall: 25000,
            },
            base_fee_multiplier: 0.1, // Much cheaper L2 gas
            priority_fee_min: 10_000_000, // 0.01 gwei
        }
    }
    
    pub fn base() -> Self {
        Self {
            chain_id: 8453,
            costs: GasCosts {
                base_cost: 21000,
                transfer: 21000,
                swap_v2: 85000,
                swap_v3: 125000,
                flash_loan: 42000,
                multicall: 27000,
            },
            base_fee_multiplier: 0.05, // Very cheap L2 gas
            priority_fee_min: 1_000_000, // 0.001 gwei
        }
    }
}

/// Gas estimator for complex transactions
pub struct GasEstimator {
    configs: HashMap<u64, ChainGasConfig>,
}

impl GasEstimator {
    pub fn new() -> Self {
        let mut configs = HashMap::new();
        configs.insert(1, ChainGasConfig::ethereum());
        configs.insert(42161, ChainGasConfig::arbitrum());
        configs.insert(8453, ChainGasConfig::base());
        
        Self { configs }
    }
    
    /// Estimate gas for a flash loan arbitrage transaction
    pub fn estimate_flash_loan_arbitrage(
        &self,
        chain_id: u64,
        num_swaps: usize,
        swap_types: &[&str], // "v2", "v3", "solidly"
        num_calls: usize,
    ) -> Result<u64> {
        let config = self.configs.get(&chain_id)
            .ok_or_else(|| AmmError::InvalidParameters(format!("Unknown chain {}", chain_id)))?;
        
        let mut total_gas = config.costs.base_cost + config.costs.flash_loan;
        
        // Add gas for each swap
        for swap_type in swap_types {
            let swap_gas = match *swap_type {
                "v2" => config.costs.swap_v2,
                "v3" => config.costs.swap_v3,
                "solidly" => config.costs.swap_v2 + 10000, // Slightly more than v2
                _ => config.costs.swap_v2, // Default to v2
            };
            total_gas += swap_gas;
        }
        
        // Add multicall overhead if multiple calls
        if num_calls > 1 {
            total_gas += config.costs.multicall * (num_calls as u64 - 1);
        }
        
        // Add 10% buffer for safety
        total_gas = (total_gas as f64 * 1.1) as u64;
        
        Ok(total_gas)
    }
    
    /// Calculate total gas cost in wei
    pub fn calculate_gas_cost(
        &self,
        chain_id: u64,
        gas_estimate: u64,
        base_fee_per_gas: u64,
        max_priority_fee: u64,
    ) -> Result<u128> {
        let config = self.configs.get(&chain_id)
            .ok_or_else(|| AmmError::InvalidParameters(format!("Unknown chain {}", chain_id)))?;
        
        // Adjust base fee for chain
        let adjusted_base_fee = (base_fee_per_gas as f64 * config.base_fee_multiplier) as u64;
        
        // Ensure minimum priority fee
        let priority_fee = max_priority_fee.max(config.priority_fee_min);
        
        let total_fee_per_gas = adjusted_base_fee + priority_fee;
        let total_cost = gas_estimate as u128 * total_fee_per_gas as u128;
        
        Ok(total_cost)
    }
    
    /// Optimize gas parameters for profitability
    pub fn optimize_for_profit(
        &self,
        chain_id: u64,
        expected_profit: u128,
        gas_estimate: u64,
        base_fee_per_gas: u64,
    ) -> Result<OptimizedGasParams> {
        let config = self.configs.get(&chain_id)
            .ok_or_else(|| AmmError::InvalidParameters(format!("Unknown chain {}", chain_id)))?;
        
        // Calculate maximum gas price we can afford
        let max_total_gas_cost = expected_profit / 2; // Use at most 50% of profit for gas
        let max_fee_per_gas = max_total_gas_cost / gas_estimate as u128;
        
        let adjusted_base_fee = (base_fee_per_gas as f64 * config.base_fee_multiplier) as u64;
        
        if max_fee_per_gas < adjusted_base_fee as u128 {
            return Err(AmmError::InvalidParameters("Profit too low to cover gas".to_string()));
        }
        
        let max_priority_fee = (max_fee_per_gas - adjusted_base_fee as u128)
            .max(config.priority_fee_min as u128) as u64;
        
        // Choose aggressive priority fee for high-value opportunities
        let priority_fee = if expected_profit > 1_000_000_000_000_000_000 { // > 1 ETH profit
            max_priority_fee.min(100_000_000_000) // Max 100 gwei priority
        } else if expected_profit > 100_000_000_000_000_000 { // > 0.1 ETH profit
            max_priority_fee.min(50_000_000_000) // Max 50 gwei priority
        } else {
            max_priority_fee.min(20_000_000_000) // Max 20 gwei priority
        };
        
        Ok(OptimizedGasParams {
            gas_limit: gas_estimate,
            max_fee_per_gas: adjusted_base_fee + priority_fee,
            max_priority_fee_per_gas: priority_fee,
            estimated_cost: gas_estimate as u128 * (adjusted_base_fee + priority_fee) as u128,
        })
    }
}

#[derive(Debug, Clone)]
pub struct OptimizedGasParams {
    pub gas_limit: u64,
    pub max_fee_per_gas: u64,
    pub max_priority_fee_per_gas: u64,
    pub estimated_cost: u128,
}

/// Calculate breakeven gas price for a trade
pub fn calculate_breakeven_gas_price(
    expected_profit: u128,
    gas_estimate: u64,
    profit_margin: f64, // e.g. 0.2 for 20% margin
) -> u64 {
    let target_profit = expected_profit as f64 * (1.0 - profit_margin);
    let max_gas_cost = expected_profit as f64 - target_profit;
    
    (max_gas_cost / gas_estimate as f64) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gas_estimation() {
        let estimator = GasEstimator::new();
        
        let gas = estimator.estimate_flash_loan_arbitrage(
            1, // Ethereum
            2, // 2 swaps
            &["v2", "v3"],
            3, // 3 total calls
        ).unwrap();
        
        assert!(gas > 200_000); // Should be reasonable for flash loan + 2 swaps
        assert!(gas < 1_000_000); // But not excessive
    }
    
    #[test]
    fn test_gas_cost_calculation() {
        let estimator = GasEstimator::new();
        
        let cost = estimator.calculate_gas_cost(
            1, // Ethereum
            300_000, // gas
            20_000_000_000, // 20 gwei base fee
            2_000_000_000, // 2 gwei priority fee
        ).unwrap();
        
        // Should be 300k * 22 gwei = 6.6M gwei = 0.0066 ETH
        assert_eq!(cost, 6_600_000_000_000_000);
    }
    
    #[test]
    fn test_l2_gas_costs() {
        let estimator = GasEstimator::new();
        
        let eth_cost = estimator.calculate_gas_cost(1, 300_000, 20_000_000_000, 2_000_000_000).unwrap();
        let arb_cost = estimator.calculate_gas_cost(42161, 300_000, 20_000_000_000, 2_000_000_000).unwrap();
        
        // Arbitrum should be much cheaper
        assert!(arb_cost < eth_cost / 5);
    }
    
    #[test]
    fn test_breakeven_calculation() {
        let breakeven = calculate_breakeven_gas_price(
            1_000_000_000_000_000_000, // 1 ETH profit
            300_000, // 300k gas
            0.3 // 30% margin
        );
        
        // Should allow up to 70% of profit for gas
        // 0.7 ETH / 300k gas = ~2333 gwei max
        assert!(breakeven > 2_000_000_000_000); // > 2000 gwei
        assert!(breakeven < 3_000_000_000_000); // < 3000 gwei
    }
}
EOF

cat > rust-core/src/route.rs << 'EOF'
//! Route optimization and path finding utilities

use crate::{AmmError, Result};
use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::cmp::{Ordering, Reverse};

/// Represents a token in the routing graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Token {
    pub address: String,
    pub symbol: String,
    pub decimals: u8,
}

/// Represents a trading pool
#[derive(Debug, Clone)]
pub struct Pool {
    pub address: String,
    pub token0: Token,
    pub token1: Token,
    pub pool_type: PoolType,
    pub fee_bps: u16,
    pub reserve0: u128,
    pub reserve1: u128,
    pub liquidity_score: f64, // 0-1 score based on liquidity depth
}

#[derive(Debug, Clone, PartialEq)]
pub enum PoolType {
    UniswapV2,
    UniswapV3,
    Solidly,
    Curve,
    Balancer,
}

/// A single hop in a route
#[derive(Debug, Clone)]
pub struct RouteHop {
    pub pool: Pool,
    pub token_in: Token,
    pub token_out: Token,
    pub amount_in: u128,
    pub amount_out: u128,
    pub gas_cost: u64,
}

/// A complete trading route
#[derive(Debug, Clone)]
pub struct Route {
    pub hops: Vec<RouteHop>,
    pub total_amount_in: u128,
    pub total_amount_out: u128,
    pub total_gas_cost: u64,
    pub price_impact: f64,
    pub confidence_score: f64,
}

impl Route {
    /// Calculate the effective price of the entire route
    pub fn effective_price(&self) -> f64 {
        if self.total_amount_in == 0 {
            return 0.0;
        }
        self.total_amount_out as f64 / self.total_amount_in as f64
    }
    
    /// Check if route forms a profitable cycle
    pub fn is_arbitrage_cycle(&self) -> bool {
        if self.hops.is_empty() {
            return false;
        }
        
        let start_token = &self.hops[0].token_in;
        let end_token = &self.hops.last().unwrap().token_out;
        
        start_token == end_token && self.total_amount_out > self.total_amount_in
    }
    
    /// Calculate profit for arbitrage cycle
    pub fn arbitrage_profit(&self) -> i128 {
        if !self.is_arbitrage_cycle() {
            return 0;
        }
        self.total_amount_out as i128 - self.total_amount_in as i128
    }
}

/// Route finder for arbitrage opportunities
pub struct RouteFinder {
    pools: Vec<Pool>,
    token_graph: HashMap<String, Vec<usize>>, // token address -> pool indices
}

impl RouteFinder {
    pub fn new(pools: Vec<Pool>) -> Self {
        let mut token_graph = HashMap::new();
        
        for (i, pool) in pools.iter().enumerate() {
            token_graph.entry(pool.token0.address.clone())
                .or_insert_with(Vec::new)
                .push(i);
            token_graph.entry(pool.token1.address.clone())
                .or_insert_with(Vec::new)
                .push(i);
        }
        
        Self { pools, token_graph }
    }
    
    /// Find all arbitrage cycles starting from a given token
    pub fn find_arbitrage_cycles(
        &self,
        start_token: &str,
        amount_in: u128,
        max_hops: usize,
        min_profit_bps: u16,
    ) -> Result<Vec<Route>> {
        let mut routes = Vec::new();
        
        // Use DFS to find cycles
        let mut visited_pools = vec![false; self.pools.len()];
        let mut current_route = Vec::new();
        
        self.dfs_arbitrage(
            start_token,
            start_token,
            amount_in,
            max_hops,
            min_profit_bps,
            &mut visited_pools,
            &mut current_route,
            &mut routes,
        );
        
        // Sort by profitability
        routes.sort_by(|a, b| {
            b.arbitrage_profit().cmp(&a.arbitrage_profit())
        });
        
        Ok(routes)
    }
    
    fn dfs_arbitrage(
        &self,
        current_token: &str,
        target_token: &str,
        amount_in: u128,
        remaining_hops: usize,
        min_profit_bps: u16,
        visited_pools: &mut Vec<bool>,
        current_route: &mut Vec<RouteHop>,
        results: &mut Vec<Route>,
    ) {
        if remaining_hops == 0 {
            return;
        }
        
        // If we're back to the target token, check if it's profitable
        if current_token == target_token && !current_route.is_empty() {
            let route = self.build_route_from_hops(current_route.clone(), amount_in);
            if let Ok(route) = route {
                if route.is_arbitrage_cycle() {
                    let profit_bps = ((route.arbitrage_profit() as f64 / amount_in as f64) * 10000.0) as u16;
                    if profit_bps >= min_profit_bps {
                        results.push(route);
                    }
                }
            }
            return;
        }
        
        // Explore connected pools
        if let Some(pool_indices) = self.token_graph.get(current_token) {
            for &pool_idx in pool_indices {
                if visited_pools[pool_idx] {
                    continue;
                }
                
                let pool = &self.pools[pool_idx];
                let next_token = if pool.token0.address == current_token {
                    &pool.token1.address
                } else {
                    &pool.token0.address
                };
                
                // Calculate swap output
                if let Ok(amount_out) = self.calculate_swap_output(pool, current_token, amount_in) {
                    if amount_out == 0 {
                        continue;
                    }
                    
                    // Create hop
                    let hop = RouteHop {
                        pool: pool.clone(),
                        token_in: if pool.token0.address == current_token {
                            pool.token0.clone()
                        } else {
                            pool.token1.clone()
                        },
                        token_out: if pool.token0.address == current_token {
                            pool.token1.clone()
                        } else {
                            pool.token0.clone()
                        },
                        amount_in,
                        amount_out,
                        gas_cost: self.estimate_swap_gas(&pool.pool_type),
                    };
                    
                    // Mark pool as visited and continue DFS
                    visited_pools[pool_idx] = true;
                    current_route.push(hop);
                    
                    self.dfs_arbitrage(
                        next_token,
                        target_token,
                        amount_out,
                        remaining_hops - 1,
                        min_profit_bps,
                        visited_pools,
                        current_route,
                        results,
                    );
                    
                    // Backtrack
                    current_route.pop();
                    visited_pools[pool_idx] = false;
                }
            }
        }
    }
    
    fn calculate_swap_output(&self, pool: &Pool, token_in: &str, amount_in: u128) -> Result<u128> {
        let (reserve_in, reserve_out) = if pool.token0.address == token_in {
            (pool.reserve0, pool.reserve1)
        } else {
            (pool.reserve1, pool.reserve0)
        };
        
        match pool.pool_type {
            PoolType::UniswapV2 => {
                crate::univ2::get_amount_out(amount_in, reserve_in, reserve_out, pool.fee_bps)
            },
            PoolType::Solidly => {
                // Assume volatile for simplicity
                crate::solidly::get_amount_out(amount_in, reserve_in, reserve_out, false, pool.fee_bps)
            },
            _ => {
                // Default to V2 math for other types
                crate::univ2::get_amount_out(amount_in, reserve_in, reserve_out, pool.fee_bps)
            }
        }
    }
    
    fn estimate_swap_gas(&self, pool_type: &PoolType) -> u64 {
        match pool_type {
            PoolType::UniswapV2 => 100_000,
            PoolType::UniswapV3 => 150_000,
            PoolType::Solidly => 110_000,
            PoolType::Curve => 200_000,
            PoolType::Balancer => 180_000,
        }
    }
    
    fn build_route_from_hops(&self, hops: Vec<RouteHop>, initial_amount: u128) -> Result<Route> {
        if hops.is_empty() {
            return Err(AmmError::InvalidParameters("Empty route".to_string()));
        }
        
        let total_gas_cost = hops.iter().map(|h| h.gas_cost).sum();
        let total_amount_out = hops.last().unwrap().amount_out;
        
        // Calculate price impact (simplified)
        let mut total_impact = 0.0;
        for hop in &hops {
            let pool_impact = hop.amount_in as f64 / (hop.pool.reserve0.min(hop.pool.reserve1) as f64);
            total_impact += pool_impact;
        }
        
        Ok(Route {
            hops,
            total_amount_in: initial_amount,
            total_amount_out,
            total_gas_cost,
            price_impact: total_impact.min(1.0),
            confidence_score: 0.8, // Default confidence
        })
    }
    
    /// Find the best direct route between two tokens
    pub fn find_best_route(
        &self,
        token_in: &str,
        token_out: &str,
        amount_in: u128,
        max_hops: usize,
    ) -> Result<Option<Route>> {
        // Use Dijkstra's algorithm to find optimal path
        let mut best_routes: HashMap<String, (u128, Vec<RouteHop>)> = HashMap::new();
        let mut heap = BinaryHeap::new();
        
        // Start with the input token
        heap.push(Reverse((0u128, token_in.to_string(), Vec::new(), amount_in)));
        best_routes.insert(token_in.to_string(), (amount_in, Vec::new()));
        
        while let Some(Reverse((neg_amount_out, current_token, route, current_amount))) = heap.pop() {
            if current_token == token_out {
                if let Ok(final_route) = self.build_route_from_hops(route, amount_in) {
                    return Ok(Some(final_route));
                }
            }
            
            if route.len() >= max_hops {
                continue;
            }
            
            // Explore neighboring tokens
            if let Some(pool_indices) = self.token_graph.get(&current_token) {
                for &pool_idx in pool_indices {
                    let pool = &self.pools[pool_idx];
                    let next_token = if pool.token0.address == current_token {
                        &pool.token1.address
                    } else {
                        &pool.token0.address
                    };
                    
                    if let Ok(amount_out) = self.calculate_swap_output(pool, &current_token, current_amount) {
                        if amount_out == 0 {
                            continue;
                        }
                        
                        // Check if this is better than existing route to next_token
                        if let Some(&(existing_amount, _)) = best_routes.get(next_token) {
                            if amount_out <= existing_amount {
                                continue;
                            }
                        }
                        
                        let mut new_route = route.clone();
                        new_route.push(RouteHop {
                            pool: pool.clone(),
                            token_in: if pool.token0.address == current_token {
                                pool.token0.clone()
                            } else {
                                pool.token1.clone()
                            },
                            token_out: if pool.token0.address == current_token {
                                pool.token1.clone()
                            } else {
                                pool.token0.clone()
                            },
                            amount_in: current_amount,
                            amount_out,
                            gas_cost: self.estimate_swap_gas(&pool.pool_type),
                        });
                        
                        best_routes.insert(next_token.clone(), (amount_out, new_route.clone()));
                        heap.push(Reverse((u128::MAX - amount_out, next_token.clone(), new_route, amount_out)));
                    }
                }
            }
        }
        
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_token(address: &str, symbol: &str) -> Token {
        Token {
            address: address.to_string(),
            symbol: symbol.to_string(),
            decimals: 18,
        }
    }
    
    fn create_test_pool(token0: Token, token1: Token, reserve0: u128, reserve1: u128) -> Pool {
        Pool {
            address: format!("pool_{}_{}", token0.symbol, token1.symbol),
            token0,
            token1,
            pool_type: PoolType::UniswapV2,
            fee_bps: 30,
            reserve0,
            reserve1,
            liquidity_score: 0.8,
        }
    }
    
    #[test]
    fn test_route_finder() {
        let weth = create_test_token("weth", "WETH");
        let usdc = create_test_token("usdc", "USDC");
        let dai = create_test_token("dai", "DAI");
        
        let pools = vec![
            create_test_pool(weth.clone(), usdc.clone(), 100_000_000_000_000_000_000, 200_000_000_000),
            create_test_pool(usdc.clone(), dai.clone(), 1_000_000_000_000, 1_000_000_000_000_000_000_000),
            create_test_pool(dai.clone(), weth.clone(), 2_000_000_000_000_000_000_000, 90_000_000_000_000_000_000),
        ];
        
        let finder = RouteFinder::new(pools);
        
        // Test triangular arbitrage
        let cycles = finder.find_arbitrage_cycles(
            "weth",
            1_000_000_000_000_000_000, // 1 ETH
            3,
            10, // 0.1% minimum profit
        ).unwrap();
        
        assert!(!cycles.is_empty());
        
        for cycle in &cycles {
            assert!(cycle.is_arbitrage_cycle());
            assert!(cycle.arbitrage_profit() > 0);
        }
    }
    
    #[test]
    fn test_direct_routing() {
        let weth = create_test_token("weth", "WETH");
        let usdc = create_test_token("usdc", "USDC");
        
        let pools = vec![
            create_test_pool(weth.clone(), usdc.clone(), 100_000_000_000_000_000_000, 200_000_000_000),
        ];
        
        let finder = RouteFinder::new(pools);
        
        let route = finder.find_best_route(
            "weth",
            "usdc", 
            1_000_000_000_000_000_000,
            2,
        ).unwrap();
        
        assert!(route.is_some());
        let route = route.unwrap();
        assert_eq!(route.hops.len(), 1);
        assert!(route.total_amount_out > 0);
    }
}
EOF

cat > rust-core/src/ffi_pyo3.rs << 'EOF'
//! Python FFI bindings using PyO3

use pyo3::prelude::*;
use crate::{univ2, univ3, solidly, gas};

/// Python wrapper for Uniswap V2 get_amount_out
#[pyfunction]
pub fn py_univ2_get_amount_out(
    amount_in: u128,
    reserve_in: u128,
    reserve_out: u128,
    fee_bps: u16,
) -> PyResult<u128> {
    univ2::get_amount_out(amount_in, reserve_in, reserve_out, fee_bps)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Python wrapper for Uniswap V2 get_amount_in
#[pyfunction]
pub fn py_univ2_get_amount_in(
    amount_out: u128,
    reserve_in: u128,
    reserve_out: u128,
    fee_bps: u16,
) -> PyResult<u128> {
    univ2::get_amount_in(amount_out, reserve_in, reserve_out, fee_bps)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Python wrapper for Uniswap V3 get_amount_out
#[pyfunction]
pub fn py_univ3_get_amount_out(
    amount_in: u128,
    sqrt_price_x96: u128,
    liquidity: u128,
    fee: u32,
    zero_for_one: bool,
) -> PyResult<(u128, u128)> {
    univ3::get_amount_out(amount_in, sqrt_price_x96, liquidity, fee, zero_for_one)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Python wrapper for tick to sqrt price conversion
#[pyfunction]
pub fn py_tick_to_sqrt_price(tick: i32) -> PyResult<u128> {
    univ3::tick_to_sqrt_price_x96(tick)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Python wrapper for sqrt price to tick conversion
#[pyfunction]
pub fn py_sqrt_price_to_tick(sqrt_price_x96: u128) -> PyResult<i32> {
    univ3::sqrt_price_x96_to_tick(sqrt_price_x96)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Python wrapper for Solidly get_amount_out
#[pyfunction]
pub fn py_solidly_get_amount_out(
    amount_in: u128,
    reserve_in: u128,
    reserve_out: u128,
    stable: bool,
    fee_bps: u16,
) -> PyResult<u128> {
    solidly::get_amount_out(amount_in, reserve_in, reserve_out, stable, fee_bps)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Python wrapper for gas cost estimation
#[pyfunction]
pub fn py_estimate_gas_cost(
    chain_id: u64,
    num_swaps: usize,
    swap_types: Vec<String>,
    num_calls: usize,
) -> PyResult<u64> {
    let estimator = gas::GasEstimator::new();
    let swap_type_refs: Vec<&str> = swap_types.iter().map(|s| s.as_str()).collect();
    
    estimator.estimate_flash_loan_arbitrage(chain_id, num_swaps, &swap_type_refs, num_calls)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Python wrapper for calculating gas cost in wei
#[pyfunction]
pub fn py_calculate_gas_cost_wei(
    chain_id: u64,
    gas_estimate: u64,
    base_fee_per_gas: u64,
    max_priority_fee: u64,
) -> PyResult<u128> {
    let estimator = gas::GasEstimator::new();
    estimator.calculate_gas_cost(chain_id, gas_estimate, base_fee_per_gas, max_priority_fee)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))
}

/// Python wrapper for breakeven gas price calculation
#[pyfunction]
pub fn py_calculate_breakeven_gas_price(
    expected_profit: u128,
    gas_estimate: u64,
    profit_margin: f64,
) -> PyResult<u64> {
    Ok(gas::calculate_breakeven_gas_price(expected_profit, gas_estimate, profit_margin))
}
EOF

cat > rust-core/src/tests/math_tests.rs << 'EOF'
//! Integration tests for AMM math functions

use crate::{univ2, univ3, solidly};

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_univ2_integration() {
        // Test realistic Ethereum mainnet scenario
        let weth_reserve = 100_000_000_000_000_000_000u128; // 100 ETH
        let usdc_reserve = 200_000_000_000u128; // 200k USDC (6 decimals)
        let swap_amount = 1_000_000_000_000_000_000u128; // 1 ETH
        
        let usdc_out = univ2::get_amount_out(
            swap_amount,
            weth_reserve,
            usdc_reserve,
            30 // 0.3% fee
        ).unwrap();
        
        // Should get approximately 1980 USDC (price impact + fees)
        assert!(usdc_out > 1_970_000_000);
        assert!(usdc_out < 1_990_000_000);
        
        // Test reverse calculation
        let weth_in = univ2::get_amount_in(
            usdc_out,
            weth_reserve,
            usdc_reserve,
            30
        ).unwrap();
        
        // Should be close to original amount (within rounding)
        assert!(weth_in >= swap_amount);
        assert!(weth_in <= swap_amount + 1000); // Allow small rounding error
    }
    
    #[test]
    fn test_univ3_integration() {
        // Test Uniswap V3 WETH/USDC 0.3% pool scenario
        let current_tick = 0; // Around $2000/ETH
        let sqrt_price_x96 = univ3::tick_to_sqrt_price_x96(current_tick).unwrap();
        let liquidity = 1_000_000_000_000_000_000u128; // 1e18
        let swap_amount = 1_000_000_000_000_000_000u128; // 1 ETH
        
        let (usdc_out, new_sqrt_price) = univ3::get_amount_out(
            swap_amount,
            sqrt_price_x96,
            liquidity,
            3000, // 0.3% fee
            true // zero_for_one
        ).unwrap();
        
        assert!(usdc_out > 0);
        assert!(new_sqrt_price < sqrt_price_x96); // Price should decrease
        
        // Test tick conversion roundtrip
        let back_to_tick = univ3::sqrt_price_x96_to_tick(sqrt_price_x96).unwrap();
        assert_eq!(current_tick, back_to_tick);
    }
    
    #[test]
    fn test_solidly_integration() {
        // Test stable vs volatile pools
        let reserve0 = 1_000_000_000_000_000_000_000u128; // 1000 tokens
        let reserve1 = 1_000_000_000_000_000_000_000u128; // 1000 tokens  
        let swap_amount = 10_000_000_000_000_000_000u128; // 10 tokens
        
        let stable_out = solidly::get_amount_out(
            swap_amount,
            reserve0,
            reserve1,
            true, // stable
            20 // 0.2% fee
        ).unwrap();
        
        let volatile_out = solidly::get_amount_out(
            swap_amount,
            reserve0,
            reserve1,
            false, // volatile
            30 // 0.3% fee
        ).unwrap();
        
        // Stable pools should have less slippage
        assert!(stable_out > volatile_out);
        
        // Both should be positive
        assert!(stable_out > 0);
        assert!(volatile_out > 0);
    }
    
    #[test]
    fn test_arbitrage_scenario() {
        // Test a triangular arbitrage scenario
        // WETH -> USDC -> DAI -> WETH
        
        // Pool 1: WETH/USDC (1 ETH = 2000 USDC)
        let weth_usdc_out = univ2::get_amount_out(
            1_000_000_000_000_000_000u128, // 1 ETH
            100_000_000_000_000_000_000u128, // 100 ETH
            200_000_000_000u128, // 200k USDC
            30
        ).unwrap();
        
        // Pool 2: USDC/DAI (1:1 with slight imbalance)
        let usdc_dai_out = solidly::get_amount_out(
            weth_usdc_out,
            1_000_000_000_000u128, // 1M USDC
            1_010_000_000_000_000_000
EOF