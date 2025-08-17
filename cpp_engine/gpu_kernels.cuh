#ifndef GPU_KERNELS_CUH
#define GPU_KERNELS_CUH

#include <cuda_runtime.h>

// Core arbitrage detection kernels
__global__ void find_arbitrage_opportunities_kernel(
    const float* bids, const float* asks, const float* volumes,
    float* spreads, int* indices, int n, float min_profit);

__global__ void calculate_triangular_arbitrage_kernel(
    const float* prices_a, const float* prices_b, const float* prices_c,
    float* profit_rates, int* valid_triangles, int n, float min_profit);

__global__ void calculate_flash_loan_profits_kernel(
    const float* dex_prices, const float* cex_prices, const float* volumes,
    float* profits, int* profitable_pairs, int n_dex, int n_cex,
    float gas_cost_usd, float flash_loan_fee_rate, float min_profit);

// Portfolio optimization kernels
__global__ void calculate_portfolio_metrics_kernel(
    const float* position_values, const float* position_pnls,
    const float* position_weights, float* portfolio_metrics,
    int n_positions, float risk_free_rate);

__global__ void optimize_portfolio_weights_kernel(
    const float* expected_returns, const float* covariance_matrix,
    float* optimal_weights, int n_assets, float risk_aversion);

// Cross-chain arbitrage detection
__global__ void detect_cross_chain_arbitrage_kernel(
    const float* eth_prices, const float* polygon_prices, const float* arbitrum_prices,
    const float* optimism_prices, const float* base_prices,
    float* arbitrage_profits, int* best_chains, int n_tokens,
    float bridge_cost_percentage, float min_profit);

// Advanced arbitrage strategies
__global__ void calculate_option_arbitrage_kernel(
    const float* spot_prices, const float* call_prices, const float* put_prices,
    const float* strikes, const float* expiries, float* arbitrage_values,
    int n_options, float risk_free_rate, float current_time);

__global__ void detect_statistical_arbitrage_kernel(
    const float* price_series_a, const float* price_series_b,
    float* correlation_coeffs, float* spread_zscores, float* signals,
    int series_length, int n_pairs, int lookback_period);

// Risk management kernels
__global__ void calculate_var_and_cvar_kernel(
    const float* portfolio_returns, float* var_result, float* cvar_result,
    int n_scenarios, float confidence_level);

// Device utility functions
__device__ __forceinline__ float atomicMaxFloat(float* address, float val);
__device__ void warp_reduce_sum(volatile float* sdata, int tid);

#endif // GPU_KERNELS_CUH