#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <vector>
#include <memory>
#include <cmath>
#include <stdio.h>

#define THREADS_PER_BLOCK 1024
#define MAX_EXCHANGES 100
#define MAX_SYMBOLS 50000
#define WARP_SIZE 32

struct PriceData {
    float bid;
    float ask;
    float volume;
    float liquidity;
    uint64_t timestamp;
    uint32_t exchange_id;
    uint32_t chain_id;
    float gas_price;
    bool is_dex;
};

struct ArbitrageResult {
    float spread;
    float gross_profit;
    float net_profit;
    float confidence;
    float execution_time_ms;
    float gas_cost;
    float slippage_impact;
    uint32_t buy_exchange;
    uint32_t sell_exchange;
    uint32_t buy_chain;
    uint32_t sell_chain;
    uint32_t strategy_type; // 0=cross_ex, 1=triangular, 2=flash_loan, 3=cross_chain
    bool profitable;
};

struct TriangularPath {
    uint32_t token1_id;
    uint32_t token2_id;
    uint32_t token3_id;
    uint32_t exchange_id;
};

struct GasPrice {
    uint32_t chain_id;
    float price_gwei;
    float eth_price_usd;
};

// Device memory structure for ultra-fast access
struct GPUMarketData {
    PriceData* prices;
    ArbitrageResult* results;
    TriangularPath* triangular_paths;
    GasPrice* gas_prices;
    float* token_correlations;
    uint32_t* profitable_indices;
    uint32_t num_prices;
    uint32_t num_results;
    uint32_t num_triangular;
    uint32_t num_chains;
};

// Optimized device functions
__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float calculate_slippage_impact(float trade_size, float liquidity) {
    if (liquidity <= 0.0f) return 1.0f;
    float impact = sqrtf(trade_size / liquidity) * 0.1f;
    return fminf(impact, 0.05f); // Cap at 5%
}

__device__ __forceinline__ float calculate_gas_cost(uint32_t chain_id, float gas_price, float eth_price, uint32_t gas_limit) {
    if (chain_id == 0) return 0.0f; // CEX
    
    float multiplier = 1.0f;
    switch (chain_id) {
        case 1: multiplier = 1.0f; break;    // Ethereum
        case 56: multiplier = 0.1f; break;   // BSC
        case 137: multiplier = 0.05f; break; // Polygon
        case 42161: multiplier = 0.3f; break; // Arbitrum
        case 10: multiplier = 0.2f; break;   // Optimism
        default: multiplier = 0.5f; break;
    }
    
    return (gas_price * gas_limit * multiplier * eth_price) / 1e9f;
}

__device__ __forceinline__ float calculate_confidence(float spread, float volume, float slippage) {
    float spread_score = fminf(spread * 1000.0f, 1.0f);
    float volume_score = fminf(logf(volume + 1.0f) / 15.0f, 1.0f);
    float slippage_score = fmaxf(1.0f - slippage, 0.0f);
    return spread_score * 0.4f + volume_score * 0.3f + slippage_score * 0.3f;
}

// Ultra-fast cross-exchange arbitrage kernel
__global__ void cross_exchange_arbitrage_kernel(
    const PriceData* __restrict__ prices,
    ArbitrageResult* __restrict__ results,
    const GasPrice* __restrict__ gas_prices,
    uint32_t num_prices,
    uint32_t num_chains,
    float min_profit_threshold,
    float max_slippage_threshold
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_combinations = (num_prices * (num_prices - 1)) / 2;
    
    if (tid >= total_combinations) return;
    
    // Convert linear index to pair indices using optimized calculation
    uint32_t i = 0, j = 1;
    uint32_t remaining = tid;
    while (remaining >= num_prices - i - 1) {
        remaining -= (num_prices - i - 1);
        i++;
        j = i + 1;
    }
    j += remaining;
    
    if (i >= num_prices || j >= num_prices) return;
    
    const PriceData& price1 = prices[i];
    const PriceData& price2 = prices[j];
    
    // Skip if different symbols (assuming symbol is encoded in upper bits of exchange_id)
    if ((price1.exchange_id >> 16) != (price2.exchange_id >> 16)) {
        results[tid].profitable = false;
        return;
    }
    
    // Calculate arbitrage opportunity in both directions
    float spread = 0.0f;
    float buy_price = 0.0f;
    float sell_price = 0.0f;
    uint32_t buy_exchange = i;
    uint32_t sell_exchange = j;
    uint32_t buy_chain = price1.chain_id;
    uint32_t sell_chain = price2.chain_id;
    
    // Direction 1: Buy from exchange i, sell to exchange j
    if (price1.ask > 0.0f && price2.bid > 0.0f && price1.ask < price2.bid) {
        spread = (price2.bid - price1.ask) / price1.ask;
        buy_price = price1.ask;
        sell_price = price2.bid;
    }
    // Direction 2: Buy from exchange j, sell to exchange i
    else if (price2.ask > 0.0f && price1.bid > 0.0f && price2.ask < price1.bid) {
        spread = (price1.bid - price2.ask) / price2.ask;
        buy_price = price2.ask;
        sell_price = price1.bid;
        buy_exchange = j;
        sell_exchange = i;
        buy_chain = price2.chain_id;
        sell_chain = price1.chain_id;
    }
    else {
        results[tid].profitable = false;
        return;
    }
    
    if (spread < 0.0001f) { // Minimum 0.01% spread
        results[tid].profitable = false;
        return;
    }
    
    // Dynamic trade sizing based on liquidity and spread
    float base_trade_size = 25000.0f;
    float spread_multiplier = fminf(spread * 100.0f, 3.0f);
    float liquidity_factor = fminf(sqrtf(fminf(price1.liquidity, price2.liquidity) / 1000000.0f), 2.0f);
    float trade_size = base_trade_size * spread_multiplier * liquidity_factor;
    
    // Calculate costs with chain-specific gas prices
    float gas_cost_buy = 0.0f;
    float gas_cost_sell = 0.0f;
    
    for (uint32_t g = 0; g < num_chains; g++) {
        if (gas_prices[g].chain_id == buy_chain) {
            gas_cost_buy = calculate_gas_cost(buy_chain, gas_prices[g].price_gwei, gas_prices[g].eth_price_usd, 200000);
        }
        if (gas_prices[g].chain_id == sell_chain) {
            gas_cost_sell = calculate_gas_cost(sell_chain, gas_prices[g].price_gwei, gas_prices[g].eth_price_usd, 150000);
        }
    }
    
    float total_gas_cost = gas_cost_buy + gas_cost_sell;
    
    // Cross-chain bridge costs
    float bridge_cost = 0.0f;
    if (buy_chain != sell_chain && buy_chain > 0 && sell_chain > 0) {
        bridge_cost = trade_size * 0.001f; // 0.1% bridge fee
    }
    
    // Trading fees
    float trading_fees = trade_size * 0.002f; // 0.2% total trading fees
    
    // Slippage calculation
    float slippage_impact = calculate_slippage_impact(trade_size, fminf(price1.liquidity, price2.liquidity));
    float slippage_cost = trade_size * slippage_impact;
    
    // MEV protection cost for DEX trades
    float mev_cost = 0.0f;
    if (price1.is_dex || price2.is_dex) {
        mev_cost = trade_size * 0.0005f; // 0.05% MEV protection
    }
    
    float gross_profit = trade_size * spread;
    float total_costs = trading_fees + total_gas_cost + bridge_cost + slippage_cost + mev_cost;
    float net_profit = gross_profit - total_costs;
    
    // Calculate execution time
    float execution_time = 100.0f; // Base CEX-CEX time
    if (price1.is_dex || price2.is_dex) execution_time += 50.0f; // DEX overhead
    if (buy_chain != sell_chain) execution_time += 30000.0f; // Cross-chain delay
    
    // Calculate confidence score
    float confidence = calculate_confidence(spread, fminf(price1.volume, price2.volume), slippage_impact);
    
    // Apply filters
    bool profitable = (net_profit > min_profit_threshold) && 
                     (slippage_impact < max_slippage_threshold) &&
                     (confidence > 0.5f) &&
                     (spread > 0.0001f);
    
    // Store results
    results[tid].spread = spread * 100.0f; // Convert to percentage
    results[tid].gross_profit = gross_profit;
    results[tid].net_profit = net_profit;
    results[tid].confidence = confidence;
    results[tid].execution_time_ms = execution_time;
    results[tid].gas_cost = total_gas_cost;
    results[tid].slippage_impact = slippage_impact * 100.0f;
    results[tid].buy_exchange = buy_exchange;
    results[tid].sell_exchange = sell_exchange;
    results[tid].buy_chain = buy_chain;
    results[tid].sell_chain = sell_chain;
    results[tid].strategy_type = 0; // Cross-exchange
    results[tid].profitable = profitable;
}

// Ultra-fast triangular arbitrage kernel
__global__ void triangular_arbitrage_kernel(
    const PriceData* __restrict__ prices,
    const TriangularPath* __restrict__ paths,
    ArbitrageResult* __restrict__ results,
    const GasPrice* __restrict__ gas_prices,
    uint32_t num_paths,
    uint32_t num_prices,
    uint32_t num_chains,
    float min_profit_threshold
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_paths) return;
    
    const TriangularPath& path = paths[tid];
    
    // Find prices for the three tokens on the specified exchange
    const PriceData* price1 = nullptr;
    const PriceData* price2 = nullptr;
    const PriceData* price3 = nullptr;
    
    for (uint32_t i = 0; i < num_prices; i++) {
        uint32_t token_id = prices[i].exchange_id >> 16;
        uint32_t exchange_id = prices[i].exchange_id & 0xFFFF;
        
        if (exchange_id == path.exchange_id) {
            if (token_id == path.token1_id) price1 = &prices[i];
            else if (token_id == path.token2_id) price2 = &prices[i];
            else if (token_id == path.token3_id) price3 = &prices[i];
        }
    }
    
    if (!price1 || !price2 || !price3) {
        results[tid].profitable = false;
        return;
    }
    
    // Calculate triangular arbitrage: Token1 -> Token2 -> Token3 -> Token1
    float rate1 = price1->bid; // Sell Token1 for Token2
    float rate2 = price2->bid; // Sell Token2 for Token3
    float rate3 = 1.0f / price3->ask; // Buy Token1 with Token3
    
    float final_rate = rate1 * rate2 * rate3;
    float profit_rate = final_rate - 1.0f;
    
    if (profit_rate < 0.0005f) { // Minimum 0.05% profit
        results[tid].profitable = false;
        return;
    }
    
    // Dynamic trade sizing
    float min_volume = fminf(fminf(price1->volume, price2->volume), price3->volume);
    float base_trade_size = 15000.0f;
    float volume_factor = fminf(sqrtf(min_volume / 500000.0f), 2.0f);
    float profit_factor = fminf(profit_rate * 200.0f, 2.0f);
    float trade_size = base_trade_size * volume_factor * profit_factor;
    
    float gross_profit = trade_size * profit_rate;
    
    // Calculate costs
    float trading_fees = trade_size * 0.003f; // 3 trades * 0.1% each
    
    float gas_cost = 0.0f;
    if (price1->is_dex) {
        for (uint32_t g = 0; g < num_chains; g++) {
            if (gas_prices[g].chain_id == price1->chain_id) {
                gas_cost = calculate_gas_cost(price1->chain_id, gas_prices[g].price_gwei, 
                                            gas_prices[g].eth_price_usd, 400000); // Complex tri-arb
                break;
            }
        }
    }
    
    // Slippage for three consecutive trades
    float avg_liquidity = (price1->liquidity + price2->liquidity + price3->liquidity) / 3.0f;
    float slippage_impact = calculate_slippage_impact(trade_size, avg_liquidity) * 1.5f; // Amplified for 3 trades
    float slippage_cost = trade_size * slippage_impact;
    
    // MEV risk is higher for triangular arbitrage
    float mev_cost = 0.0f;
    if (price1->is_dex) {
        mev_cost = trade_size * 0.001f; // 0.1% MEV protection
    }
    
    float net_profit = gross_profit - trading_fees - gas_cost - slippage_cost - mev_cost;
    
    // Execution time for three sequential trades
    float execution_time = price1->is_dex ? 300.0f : 150.0f;
    
    float confidence = calculate_confidence(profit_rate, min_volume, slippage_impact) * 0.9f; // Lower confidence for tri-arb
    
    bool profitable = (net_profit > min_profit_threshold) && 
                     (confidence > 0.6f) &&
                     (slippage_impact < 0.03f); // 3% max slippage for tri-arb
    
    // Store results
    results[tid].spread = profit_rate * 100.0f;
    results[tid].gross_profit = gross_profit;
    results[tid].net_profit = net_profit;
    results[tid].confidence = confidence;
    results[tid].execution_time_ms = execution_time;
    results[tid].gas_cost = gas_cost;
    results[tid].slippage_impact = slippage_impact * 100.0f;
    results[tid].buy_exchange = path.exchange_id;
    results[tid].sell_exchange = path.exchange_id;
    results[tid].buy_chain = price1->chain_id;
    results[tid].sell_chain = price1->chain_id;
    results[tid].strategy_type = 1; // Triangular
    results[tid].profitable = profitable;
}

// Flash loan arbitrage kernel
__global__ void flash_loan_arbitrage_kernel(
    const PriceData* __restrict__ prices,
    ArbitrageResult* __restrict__ results,
    const GasPrice* __restrict__ gas_prices,
    uint32_t num_prices,
    uint32_t num_chains,
    float min_profit_threshold,
    float max_slippage_threshold
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_prices) return;
    
    const PriceData& dex_price = prices[tid];
    
    // Only process DEX prices
    if (!dex_price.is_dex || dex_price.chain_id != 1) { // Ethereum only for flash loans
        results[tid].profitable = false;
        return;
    }
    
    // Find corresponding CEX price
    const PriceData* cex_price = nullptr;
    uint32_t dex_token_id = dex_price.exchange_id >> 16;
    
    for (uint32_t i = 0; i < num_prices; i++) {
        if (i == tid) continue;
        
        uint32_t token_id = prices[i].exchange_id >> 16;
        if (token_id == dex_token_id && !prices[i].is_dex) {
            cex_price = &prices[i];
            break;
        }
    }
    
    if (!cex_price) {
        results[tid].profitable = false;
        return;
    }
    
    // Calculate spread in both directions
    float spread = 0.0f;
    float buy_price = 0.0f;
    float sell_price = 0.0f;
    bool dex_to_cex = false;
    
    if (dex_price.ask < cex_price->bid) {
        spread = (cex_price->bid - dex_price.ask) / dex_price.ask;
        buy_price = dex_price.ask;
        sell_price = cex_price->bid;
        dex_to_cex = true;
    } else if (cex_price->ask < dex_price.bid) {
        spread = (dex_price.bid - cex_price->ask) / cex_price->ask;
        buy_price = cex_price->ask;
        sell_price = dex_price.bid;
        dex_to_cex = false;
    } else {
        results[tid].profitable = false;
        return;
    }
    
    if (spread < 0.003f) { // Minimum 0.3% spread for flash loans
        results[tid].profitable = false;
        return;
    }
    
    // Flash loan sizing - much larger than regular arbitrage
    float base_loan_size = 200000.0f;
    float spread_multiplier = fminf(spread * 50.0f, 5.0f);
    float liquidity_factor = fminf(sqrtf(dex_price.liquidity / 5000000.0f), 3.0f);
    float loan_amount = base_loan_size * spread_multiplier * liquidity_factor;
    
    float gross_profit = loan_amount * spread;
    
    // Flash loan fee (Aave: 0.05%)
    float flash_loan_fee = loan_amount * 0.0005f;
    
    // Gas costs for complex flash loan transaction
    float gas_cost = 0.0f;
    for (uint32_t g = 0; g < num_chains; g++) {
        if (gas_prices[g].chain_id == 1) { // Ethereum
            gas_cost = calculate_gas_cost(1, gas_prices[g].price_gwei, 
                                        gas_prices[g].eth_price_usd, 800000); // High gas limit
            break;
        }
    }
    
    // Slippage impact for large trade
    float slippage_impact = calculate_slippage_impact(loan_amount, dex_price.liquidity) * 1.2f;
    float slippage_cost = loan_amount * slippage_impact;
    
    // MEV protection cost
    float mev_cost = loan_amount * 0.001f; // 0.1% MEV protection
    
    // Execution risk premium
    float execution_risk = loan_amount * 0.0002f; // 0.02% execution risk
    
    float net_profit = gross_profit - flash_loan_fee - gas_cost - slippage_cost - mev_cost - execution_risk;
    
    float execution_time = 500.0f; // Flash loan execution time
    
    float confidence = calculate_confidence(spread, loan_amount, slippage_impact) * 0.95f; // High confidence for flash loans
    
    bool profitable = (net_profit > min_profit_threshold * 10.0f) && // Higher threshold for flash loans
                     (slippage_impact < max_slippage_threshold) &&
                     (confidence > 0.8f) &&
                     (spread > 0.003f);
    
    // Store results
    results[tid].spread = spread * 100.0f;
    results[tid].gross_profit = gross_profit;
    results[tid].net_profit = net_profit;
    results[tid].confidence = confidence;
    results[tid].execution_time_ms = execution_time;
    results[tid].gas_cost = gas_cost + flash_loan_fee;
    results[tid].slippage_impact = slippage_impact * 100.0f;
    results[tid].buy_exchange = dex_to_cex ? tid : 999; // 999 represents CEX
    results[tid].sell_exchange = dex_to_cex ? 999 : tid;
    results[tid].buy_chain = dex_to_cex ? 1 : 0;
    results[tid].sell_chain = dex_to_cex ? 0 : 1;
    results[tid].strategy_type = 2; // Flash loan
    results[tid].profitable = profitable;
}

// Cross-chain arbitrage kernel
__global__ void cross_chain_arbitrage_kernel(
    const PriceData* __restrict__ prices,
    ArbitrageResult* __restrict__ results,
    const GasPrice* __restrict__ gas_prices,
    uint32_t num_prices,
    uint32_t num_chains,
    float min_profit_threshold
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_combinations = (num_prices * (num_prices - 1)) / 2;
    
    if (tid >= total_combinations) return;
    
    // Convert linear index to pair indices
    uint32_t i = 0, j = 1;
    uint32_t remaining = tid;
    while (remaining >= num_prices - i - 1) {
        remaining -= (num_prices - i - 1);
        i++;
        j = i + 1;
    }
    j += remaining;
    
    const PriceData& price1 = prices[i];
    const PriceData& price2 = prices[j];
    
    // Only process cross-chain opportunities (different chains, same token, both DEX)
    if (price1.chain_id == price2.chain_id || 
        !price1.is_dex || !price2.is_dex ||
        (price1.exchange_id >> 16) != (price2.exchange_id >> 16)) {
        results[tid].profitable = false;
        return;
    }
    
    // Calculate cross-chain arbitrage opportunity
    float spread = 0.0f;
    float buy_price = 0.0f;
    float sell_price = 0.0f;
    uint32_t buy_chain = price1.chain_id;
    uint32_t sell_chain = price2.chain_id;
    
    if (price1.ask < price2.bid) {
        spread = (price2.bid - price1.ask) / price1.ask;
        buy_price = price1.ask;
        sell_price = price2.bid;
    } else if (price2.ask < price1.bid) {
        spread = (price1.bid - price2.ask) / price2.ask;
        buy_price = price2.ask;
        sell_price = price1.bid;
        buy_chain = price2.chain_id;
        sell_chain = price1.chain_id;
    } else {
        results[tid].profitable = false;
        return;
    }
    
    if (spread < 0.008f) { // Minimum 0.8% spread for cross-chain
        results[tid].profitable = false;
        return;
    }
    
    // Cross-chain trade sizing
    float base_trade_size = 75000.0f;
    float spread_multiplier = fminf(spread * 20.0f, 4.0f);
    float liquidity_factor = fminf(sqrtf(fminf(price1.liquidity, price2.liquidity) / 2000000.0f), 2.5f);
    float trade_size = base_trade_size * spread_multiplier * liquidity_factor;
    
    float gross_profit = trade_size * spread;
    
    // Bridge costs (varies by chain pair)
    float bridge_fee_rate = 0.002f; // 0.2% default
    if ((buy_chain == 1 && sell_chain == 137) || (buy_chain == 137 && sell_chain == 1)) {
        bridge_fee_rate = 0.001f; // Lower for ETH-Polygon
    } else if ((buy_chain == 1 && sell_chain == 42161) || (buy_chain == 42161 && sell_chain == 1)) {
        bridge_fee_rate = 0.0015f; // Lower for ETH-Arbitrum
    }
    
    float bridge_cost = trade_size * bridge_fee_rate;
    
    // Gas costs on both chains
    float gas_cost_buy = 0.0f;
    float gas_cost_sell = 0.0f;
    
    for (uint32_t g = 0; g < num_chains; g++) {
        if (gas_prices[g].chain_id == buy_chain) {
            gas_cost_buy = calculate_gas_cost(buy_chain, gas_prices[g].price_gwei, gas_prices[g].eth_price_usd, 300000);
        }
        if (gas_prices[g].chain_id == sell_chain) {
            gas_cost_sell = calculate_gas_cost(sell_chain, gas_prices[g].price_gwei, gas_prices[g].eth_price_usd, 200000);
        }
    }
    
    float total_gas_cost = gas_cost_buy + gas_cost_sell;
    
    // Time value cost (opportunity cost for bridge delay)
    float time_value_cost = trade_size * 0.0001f; // 0.01% time cost
    
    // Slippage on both sides
    float slippage_impact = (calculate_slippage_impact(trade_size, price1.liquidity) + 
                           calculate_slippage_impact(trade_size, price2.liquidity)) / 2.0f;
    float slippage_cost = trade_size * slippage_impact;
    
    float net_profit = gross_profit - bridge_cost - total_gas_cost - time_value_cost - slippage_cost;
    
    // Long execution time for cross-chain
    float execution_time = 45000.0f; // 45 seconds average
    
    float confidence = calculate_confidence(spread, fminf(price1.volume, price2.volume), slippage_impact) * 0.8f; // Lower confidence
    
    bool profitable = (net_profit > min_profit_threshold * 20.0f) && // Much higher threshold for cross-chain
                     (confidence > 0.7f) &&
                     (spread > 0.008f);
    
    // Store results
    results[tid].spread = spread * 100.0f;
    results[tid].gross_profit = gross_profit;
    results[tid].net_profit = net_profit;
    results[tid].confidence = confidence;
    results[tid].execution_time_ms = execution_time;
    results[tid].gas_cost = total_gas_cost + bridge_cost;
    results[tid].slippage_impact = slippage_impact * 100.0f;
    results[tid].buy_exchange = i;
    results[tid].sell_exchange = j;
    results[tid].buy_chain = buy_chain;
    results[tid].sell_chain = sell_chain;
    results[tid].strategy_type = 3; // Cross-chain
    results[tid].profitable = profitable;
}

// Kernel to filter and sort profitable opportunities
__global__ void filter_and_rank_opportunities_kernel(
    const ArbitrageResult* __restrict__ input_results,
    ArbitrageResult* __restrict__ output_results,
    uint32_t* __restrict__ profitable_count,
    uint32_t num_results
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ uint32_t shared_count;
    if (threadIdx.x == 0) shared_count = 0;
    __syncthreads();
    
    if (tid < num_results && input_results[tid].profitable) {
        uint32_t index = atomicAdd(&shared_count, 1);
        if (index < 1000) { // Limit to top 1000 opportunities
            output_results[index] = input_results[tid];
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(profitable_count, shared_count);
    }
}

// Advanced market analysis kernel
__global__ void market_analysis_kernel(
    const PriceData* __restrict__ prices,
    float* __restrict__ volatility_scores,
    float* __restrict__ liquidity_scores,
    float* __restrict__ momentum_scores,
    uint32_t num_prices,
    uint32_t analysis_window
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_prices) return;
    
    const PriceData& current_price = prices[tid];
    
    // Calculate volatility score
    float price_variance = 0.0f;
    float mean_price = (current_price.bid + current_price.ask) / 2.0f;
    uint32_t comparison_count = 0;
    
    for (uint32_t i = 0; i < num_prices; i++) {
        if (i != tid && (prices[i].exchange_id >> 16) == (current_price.exchange_id >> 16)) {
            float other_price = (prices[i].bid + prices[i].ask) / 2.0f;
            float price_diff = fabsf(mean_price - other_price) / mean_price;
            price_variance += price_diff * price_diff;
            comparison_count++;
        }
    }
    
    volatility_scores[tid] = comparison_count > 0 ? sqrtf(price_variance / comparison_count) : 0.0f;
    
    // Calculate liquidity score (normalized)
    liquidity_scores[tid] = fminf(logf(current_price.liquidity + 1.0f) / 20.0f, 1.0f);
    
    // Calculate momentum score (price position relative to others)
    uint32_t higher_count = 0;
    uint32_t total_count = 0;
    
    for (uint32_t i = 0; i < num_prices; i++) {
        if ((prices[i].exchange_id >> 16) == (current_price.exchange_id >> 16)) {
            float other_price = (prices[i].bid + prices[i].ask) / 2.0f;
            if (mean_price > other_price) higher_count++;
            total_count++;
        }
    }
    
    momentum_scores[tid] = total_count > 0 ? (float)higher_count / total_count : 0.5f;
}

// Risk assessment kernel
__global__ void risk_assessment_kernel(
    const ArbitrageResult* __restrict__ opportunities,
    float* __restrict__ risk_scores,
    uint32_t num_opportunities,
    float portfolio_exposure,
    float max_position_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_opportunities) return;
    
    const ArbitrageResult& opp = opportunities[tid];
    
    if (!opp.profitable) {
        risk_scores[tid] = 1.0f; // Maximum risk for unprofitable
        return;
    }
    
    float risk_score = 0.0f;
    
    // Slippage risk (0-0.3)
    risk_score += fminf(opp.slippage_impact / 100.0f / 0.05f, 1.0f) * 0.3f;
    
    // Execution time risk (0-0.2)
    float time_risk = fminf(opp.execution_time_ms / 60000.0f, 1.0f); // 1 minute max
    risk_score += time_risk * 0.2f;
    
    // Strategy-specific risk (0-0.2)
    float strategy_risk = 0.0f;
    switch (opp.strategy_type) {
        case 0: strategy_risk = 0.1f; break; // Cross-exchange
        case 1: strategy_risk = 0.15f; break; // Triangular
        case 2: strategy_risk = 0.05f; break; // Flash loan (lower risk)
        case 3: strategy_risk = 0.25f; break; // Cross-chain (higher risk)
    }
    risk_score += strategy_risk * 0.2f;
    
    // Confidence inverse risk (0-0.2)
    risk_score += (1.0f - opp.confidence) * 0.2f;
    
    // Portfolio concentration risk (0-0.1)
    float position_ratio = opp.net_profit / max_position_size;
    risk_score += fminf(position_ratio, 1.0f) * 0.1f;
    
    risk_scores[tid] = fminf(risk_score, 1.0f);
}

// Host interface functions
extern "C" {
    
// Initialize GPU market data structure
GPUMarketData* initialize_gpu_market_data(uint32_t max_prices, uint32_t max_results) {
    GPUMarketData* gpu_data = new GPUMarketData();
    
    cudaMalloc(&gpu_data->prices, max_prices * sizeof(PriceData));
    cudaMalloc(&gpu_data->results, max_results * sizeof(ArbitrageResult));
    cudaMalloc(&gpu_data->gas_prices, 20 * sizeof(GasPrice)); // Support 20 chains
    cudaMalloc(&gpu_data->profitable_indices, max_results * sizeof(uint32_t));
    
    // Initialize triangular paths
    std::vector<TriangularPath> tri_paths;
    uint32_t major_tokens[] = {0, 1, 2, 3, 4, 5}; // BTC, ETH, BNB, ADA, SOL, AVAX
    uint32_t major_exchanges[] = {0, 1, 2}; // Binance, OKX, Coinbase
    
    for (uint32_t ex : major_exchanges) {
        for (uint32_t i = 0; i < 6; i++) {
            for (uint32_t j = i + 1; j < 6; j++) {
                for (uint32_t k = j + 1; k < 6; k++) {
                    tri_paths.push_back({major_tokens[i], major_tokens[j], major_tokens[k], ex});
                }
            }
        }
    }
    
    gpu_data->num_triangular = tri_paths.size();
    cudaMalloc(&gpu_data->triangular_paths, gpu_data->num_triangular * sizeof(TriangularPath));
    cudaMemcpy(gpu_data->triangular_paths, tri_paths.data(), 
               gpu_data->num_triangular * sizeof(TriangularPath), cudaMemcpyHostToDevice);
    
    return gpu_data;
}

// Update GPU price data
void update_gpu_prices(GPUMarketData* gpu_data, PriceData* host_prices, uint32_t num_prices) {
    cudaMemcpy(gpu_data->prices, host_prices, num_prices * sizeof(PriceData), cudaMemcpyHostToDevice);
    gpu_data->num_prices = num_prices;
}

// Update GPU gas prices
void update_gpu_gas_prices(GPUMarketData* gpu_data, GasPrice* host_gas_prices, uint32_t num_chains) {
    cudaMemcpy(gpu_data->gas_prices, host_gas_prices, num_chains * sizeof(GasPrice), cudaMemcpyHostToDevice);
    gpu_data->num_chains = num_chains;
}

// Execute arbitrage scanning on GPU
uint32_t scan_arbitrage_opportunities_gpu(
    GPUMarketData* gpu_data,
    ArbitrageResult* host_results,
    uint32_t max_results,
    float min_profit_threshold,
    float max_slippage_threshold
) {
    if (gpu_data->num_prices < 2) return 0;
    
    // Calculate grid dimensions
    uint32_t cross_ex_combinations = (gpu_data->num_prices * (gpu_data->num_prices - 1)) / 2;
    
    dim3 cross_ex_grid((cross_ex_combinations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 block(THREADS_PER_BLOCK);
    
    // Launch cross-exchange arbitrage kernel
    cross_exchange_arbitrage_kernel<<<cross_ex_grid, block>>>(
        gpu_data->prices, gpu_data->results, gpu_data->gas_prices,
        gpu_data->num_prices, gpu_data->num_chains,
        min_profit_threshold, max_slippage_threshold
    );
    
    // Launch triangular arbitrage kernel
    if (gpu_data->num_triangular > 0) {
        dim3 tri_grid((gpu_data->num_triangular + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        triangular_arbitrage_kernel<<<tri_grid, block>>>(
            gpu_data->prices, gpu_data->triangular_paths, 
            gpu_data->results + cross_ex_combinations, gpu_data->gas_prices,
            gpu_data->num_triangular, gpu_data->num_prices, gpu_data->num_chains,
            min_profit_threshold
        );
    }
    
    // Launch flash loan arbitrage kernel
    dim3 flash_grid((gpu_data->num_prices + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    flash_loan_arbitrage_kernel<<<flash_grid, block>>>(
        gpu_data->prices, gpu_data->results + cross_ex_combinations + gpu_data->num_triangular,
        gpu_data->gas_prices, gpu_data->num_prices, gpu_data->num_chains,
        min_profit_threshold, max_slippage_threshold
    );
    
    // Launch cross-chain arbitrage kernel
    cross_chain_arbitrage_kernel<<<cross_ex_grid, block>>>(
        gpu_data->prices, gpu_data->results + cross_ex_combinations + gpu_data->num_triangular + gpu_data->num_prices,
        gpu_data->gas_prices, gpu_data->num_prices, gpu_data->num_chains,
        min_profit_threshold
    );
    
    cudaDeviceSynchronize();
    
    // Copy results back to host
    uint32_t total_results = cross_ex_combinations + gpu_data->num_triangular + gpu_data->num_prices + cross_ex_combinations;
    uint32_t copy_count = fminf(total_results, max_results);
    
    cudaMemcpy(host_results, gpu_data->results, copy_count * sizeof(ArbitrageResult), cudaMemcpyDeviceToHost);
    
    return copy_count;
}

// Cleanup GPU resources
void cleanup_gpu_market_data(GPUMarketData* gpu_data) {
    if (gpu_data) {
        cudaFree(gpu_data->prices);
        cudaFree(gpu_data->results);
        cudaFree(gpu_data->triangular_paths);
        cudaFree(gpu_data->gas_prices);
        cudaFree(gpu_data->profitable_indices);
        delete gpu_data;
    }
}

}