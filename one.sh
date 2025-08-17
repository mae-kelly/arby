# Generate C++/CUDA GPU Engine
cat > cpp_cuda/CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(HyperArbitrageGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto")

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# CUDA architecture detection
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89) # A100, RTX 30/40xx, H100
endif()

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/include/python3.10)

# Main GPU arbitrage engine
add_library(hyperarbitrage_gpu SHARED
    src/gpu_arbitrage_engine.cu
    src/price_calculator.cu
    src/opportunity_detector.cu
    src/execution_engine.cpp
    src/market_data_processor.cu
    src/cross_chain_optimizer.cu
    src/flash_loan_calculator.cu
    src/mev_protection.cu
    src/risk_calculator.cu
    src/portfolio_optimizer.cu
)

target_link_libraries(hyperarbitrage_gpu
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_CURAND_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
    Threads::Threads
    python3.10
)

# Set CUDA properties for maximum performance
set_target_properties(hyperarbitrage_gpu PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Python bindings
add_library(hyperarbitrage_py SHARED
    src/python_bindings.cpp
)

target_link_libraries(hyperarbitrage_py
    hyperarbitrage_gpu
    python3.10
)
EOF

# Generate main GPU arbitrage engine
cat > cpp_cuda/src/gpu_arbitrage_engine.cu << 'EOF'
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <cooperative_groups.h>
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <thread>
#include <atomic>

namespace cg = cooperative_groups;

constexpr int MAX_EXCHANGES = 200;
constexpr int MAX_TOKENS = 50000;
constexpr int MAX_CHAINS = 50;
constexpr int THREADS_PER_BLOCK = 1024;
constexpr int MAX_OPPORTUNITIES = 1000000;
constexpr int WARP_SIZE = 32;

struct __align__(16) GPUPrice {
    float bid;
    float ask;
    float volume_24h;
    float liquidity_usd;
    uint32_t timestamp_ms;
    uint16_t exchange_id;
    uint16_t chain_id;
    uint32_t token_id;
    float gas_price_gwei;
    uint8_t is_dex;
    uint8_t supports_flash_loans;
    uint16_t padding;
};

struct __align__(16) GPUOpportunity {
    uint64_t id;
    uint32_t token_id;
    uint16_t buy_exchange_id;
    uint16_t sell_exchange_id;
    uint16_t buy_chain_id;
    uint16_t sell_chain_id;
    float buy_price;
    float sell_price;
    float spread_percentage;
    float gross_profit_usd;
    float net_profit_usd;
    float gas_cost_usd;
    float execution_time_ms;
    float confidence_score;
    float volume_limit_usd;
    float slippage_percentage;
    uint64_t timestamp_ns;
    uint8_t strategy_type; // 0=cross_ex, 1=triangular, 2=flash_loan, 3=cross_chain
    uint8_t mev_protection;
    uint8_t flash_loan_available;
    uint8_t padding;
};

struct __align__(8) ExchangeConfig {
    uint16_t id;
    uint16_t chain_id;
    uint16_t fee_bps;
    uint8_t is_dex;
    uint8_t supports_flash_loans;
    float min_trade_usd;
    float max_trade_usd;
};

struct __align__(8) ChainConfig {
    uint16_t id;
    uint16_t block_time_ms;
    float gas_multiplier;
    float bridge_fee_bps;
};

class HyperArbitrageGPU {
private:
    // GPU memory pointers
    GPUPrice* d_prices;
    GPUOpportunity* d_opportunities;
    ExchangeConfig* d_exchange_configs;
    ChainConfig* d_chain_configs;
    float* d_price_matrix; // [exchange][token] flattened
    uint32_t* d_opportunity_counts;
    
    // Host data
    std::vector<GPUPrice> h_prices;
    std::vector<ExchangeConfig> h_exchange_configs;
    std::vector<ChainConfig> h_chain_configs;
    
    // CUDA streams for parallel processing
    cudaStream_t* streams;
    int num_streams;
    
    // Performance tracking
    std::atomic<uint64_t> total_scans{0};
    std::atomic<uint64_t> total_opportunities{0};
    std::atomic<double> total_profit_potential{0.0};
    
    size_t max_prices;
    size_t max_opportunities;
    
public:
    HyperArbitrageGPU(size_t max_prices = MAX_TOKENS * MAX_EXCHANGES, 
                      size_t max_opps = MAX_OPPORTUNITIES,
                      int num_streams = 16) 
        : max_prices(max_prices), max_opportunities(max_opps), num_streams(num_streams) {
        
        initialize_gpu_memory();
        initialize_cuda_streams();
        initialize_exchange_configs();
        initialize_chain_configs();
        
        std::cout << "🚀 HyperArbitrageGPU initialized with " << num_streams << " CUDA streams" << std::endl;
    }
    
    ~HyperArbitrageGPU() {
        cleanup_gpu_memory();
        cleanup_cuda_streams();
    }
    
private:
    void initialize_gpu_memory() {
        // Allocate GPU memory
        cudaMalloc(&d_prices, max_prices * sizeof(GPUPrice));
        cudaMalloc(&d_opportunities, max_opportunities * sizeof(GPUOpportunity));
        cudaMalloc(&d_exchange_configs, MAX_EXCHANGES * sizeof(ExchangeConfig));
        cudaMalloc(&d_chain_configs, MAX_CHAINS * sizeof(ChainConfig));
        cudaMalloc(&d_price_matrix, MAX_EXCHANGES * MAX_TOKENS * sizeof(float));
        cudaMalloc(&d_opportunity_counts, num_streams * sizeof(uint32_t));
        
        // Initialize with zeros
        cudaMemset(d_prices, 0, max_prices * sizeof(GPUPrice));
        cudaMemset(d_opportunities, 0, max_opportunities * sizeof(GPUOpportunity));
        cudaMemset(d_opportunity_counts, 0, num_streams * sizeof(uint32_t));
        
        // Check for allocation errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error("GPU memory allocation failed: " + std::string(cudaGetErrorString(error)));
        }
        
        std::cout << "✅ GPU memory allocated: " 
                  << (max_prices * sizeof(GPUPrice) + max_opportunities * sizeof(GPUOpportunity)) / (1024*1024) 
                  << " MB" << std::endl;
    }
    
    void initialize_cuda_streams() {
        streams = new cudaStream_t[num_streams];
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamCreate(&streams[i]);
        }
    }
    
    void initialize_exchange_configs() {
        h_exchange_configs.resize(MAX_EXCHANGES);
        
        // Major CEX
        h_exchange_configs[0] = {0, 0, 10, 0, 0, 10.0f, 10000000.0f}; // Binance
        h_exchange_configs[1] = {1, 0, 50, 0, 0, 1.0f, 5000000.0f};   // Coinbase
        h_exchange_configs[2] = {2, 0, 10, 0, 0, 10.0f, 10000000.0f}; // OKX
        h_exchange_configs[3] = {3, 0, 10, 0, 0, 5.0f, 5000000.0f};   // Bybit
        h_exchange_configs[4] = {4, 0, 20, 0, 0, 1.0f, 2000000.0f};   // Huobi
        h_exchange_configs[5] = {5, 0, 10, 0, 0, 1.0f, 3000000.0f};   // KuCoin
        h_exchange_configs[6] = {6, 0, 20, 0, 0, 1.0f, 1000000.0f};   // Gate.io
        h_exchange_configs[7] = {7, 0, 26, 0, 0, 1.0f, 2000000.0f};   // Kraken
        
        // Ethereum DEX
        h_exchange_configs[10] = {10, 1, 30, 1, 1, 1.0f, 50000000.0f}; // Uniswap V3
        h_exchange_configs[11] = {11, 1, 30, 1, 1, 1.0f, 20000000.0f}; // Uniswap V2
        h_exchange_configs[12] = {12, 1, 30, 1, 1, 1.0f, 10000000.0f}; // SushiSwap
        h_exchange_configs[13] = {13, 1, 4, 1, 1, 1.0f, 100000000.0f};  // Curve
        h_exchange_configs[14] = {14, 1, 5, 1, 1, 1.0f, 30000000.0f};   // Balancer
        h_exchange_configs[15] = {15, 1, 30, 1, 0, 1.0f, 5000000.0f};   // 1inch
        
        // BSC DEX
        h_exchange_configs[20] = {20, 56, 25, 1, 1, 1.0f, 20000000.0f}; // PancakeSwap
        h_exchange_configs[21] = {21, 56, 10, 1, 0, 1.0f, 5000000.0f};  // Biswap
        h_exchange_configs[22] = {22, 56, 30, 1, 0, 1.0f, 2000000.0f};  // BakerySwap
        
        // Polygon DEX
        h_exchange_configs[30] = {30, 137, 30, 1, 1, 1.0f, 10000000.0f}; // QuickSwap
        h_exchange_configs[31] = {31, 137, 30, 1, 0, 1.0f, 5000000.0f};  // SushiSwap Polygon
        
        // Arbitrum DEX
        h_exchange_configs[40] = {40, 42161, 30, 1, 1, 1.0f, 15000000.0f}; // Uniswap Arbitrum
        h_exchange_configs[41] = {41, 42161, 30, 1, 0, 1.0f, 8000000.0f};  // SushiSwap Arbitrum
        h_exchange_configs[42] = {42, 42161, 5, 1, 1, 1.0f, 50000000.0f};  // Curve Arbitrum
        
        // Optimism DEX
        h_exchange_configs[50] = {50, 10, 30, 1, 1, 1.0f, 10000000.0f}; // Uniswap Optimism
        h_exchange_configs[51] = {51, 10, 5, 1, 1, 1.0f, 30000000.0f};  // Curve Optimism
        
        // Avalanche DEX
        h_exchange_configs[60] = {60, 43114, 30, 1, 1, 1.0f, 10000000.0f}; // TraderJoe
        h_exchange_configs[61] = {61, 43114, 30, 1, 0, 1.0f, 5000000.0f};  // Pangolin
        
        // Fantom DEX
        h_exchange_configs[70] = {70, 250, 20, 1, 1, 1.0f, 5000000.0f}; // SpookySwap
        h_exchange_configs[71] = {71, 250, 5, 1, 1, 1.0f, 20000000.0f}; // Curve Fantom
        
        // Solana DEX
        h_exchange_configs[80] = {80, 900, 25, 1, 1, 1.0f, 50000000.0f}; // Raydium
        h_exchange_configs[81] = {81, 900, 30, 1, 0, 1.0f, 20000000.0f}; // Orca
        h_exchange_configs[82] = {82, 900, 50, 1, 0, 1.0f, 100000000.0f}; // Jupiter
        
        // Copy to GPU
        cudaMemcpy(d_exchange_configs, h_exchange_configs.data(), 
                   h_exchange_configs.size() * sizeof(ExchangeConfig), cudaMemcpyHostToDevice);
    }
    
    void initialize_chain_configs() {
        h_chain_configs.resize(MAX_CHAINS);
        
        h_chain_configs[0] = {0, 0, 0.0f, 0.0f};        // CEX (no chain)
        h_chain_configs[1] = {1, 12000, 1.0f, 100.0f};  // Ethereum
        h_chain_configs[56] = {56, 3000, 0.1f, 50.0f};  // BSC
        h_chain_configs[137] = {137, 2000, 0.05f, 30.0f}; // Polygon
        h_chain_configs[42161] = {42161, 1000, 0.3f, 100.0f}; // Arbitrum
        h_chain_configs[10] = {10, 2000, 0.2f, 80.0f};  // Optimism
        h_chain_configs[43114] = {43114, 2000, 0.15f, 60.0f}; // Avalanche
        h_chain_configs[250] = {250, 1000, 0.08f, 40.0f}; // Fantom
        h_chain_configs[900] = {900, 400, 0.01f, 20.0f}; // Solana (custom ID)
        
        // Copy to GPU
        cudaMemcpy(d_chain_configs, h_chain_configs.data(), 
                   h_chain_configs.size() * sizeof(ChainConfig), cudaMemcpyHostToDevice);
    }
    
    void cleanup_gpu_memory() {
        cudaFree(d_prices);
        cudaFree(d_opportunities);
        cudaFree(d_exchange_configs);
        cudaFree(d_chain_configs);
        cudaFree(d_price_matrix);
        cudaFree(d_opportunity_counts);
    }
    
    void cleanup_cuda_streams() {
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamDestroy(streams[i]);
        }
        delete[] streams;
    }

public:
    void update_prices(const std::vector<GPUPrice>& prices) {
        if (prices.size() > max_prices) {
            std::cerr << "⚠️  Price update exceeds maximum capacity" << std::endl;
            return;
        }
        
        h_prices = prices;
        
        // Async copy to GPU
        cudaMemcpyAsync(d_prices, prices.data(), 
                       prices.size() * sizeof(GPUPrice), 
                       cudaMemcpyHostToDevice, streams[0]);
    }
    
    std::vector<GPUOpportunity> scan_all_opportunities(float min_profit_usd = 10.0f, 
                                                      float max_slippage = 0.05f) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        const size_t price_count = h_prices.size();
        if (price_count < 2) {
            return {};
        }
        
        // Reset opportunity counts
        cudaMemset(d_opportunity_counts, 0, num_streams * sizeof(uint32_t));
        
        // Calculate grid dimensions
        const size_t total_combinations = (price_count * (price_count - 1)) / 2;
        const dim3 block_size(THREADS_PER_BLOCK);
        const dim3 grid_size((total_combinations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        
        // Launch cross-exchange arbitrage kernel
        cross_exchange_arbitrage_kernel<<<grid_size, block_size, 0, streams[0]>>>(
            d_prices, d_opportunities, d_exchange_configs, d_chain_configs,
            price_count, min_profit_usd, max_slippage, 0 // offset for opportunities
        );
        
        // Launch triangular arbitrage kernel
        triangular_arbitrage_kernel<<<grid_size, block_size, 0, streams[1]>>>(
            d_prices, d_opportunities, d_exchange_configs, d_chain_configs,
            price_count, min_profit_usd, total_combinations // offset
        );
        
        // Launch flash loan arbitrage kernel
        flash_loan_arbitrage_kernel<<<grid_size, block_size, 0, streams[2]>>>(
            d_prices, d_opportunities, d_exchange_configs, d_chain_configs,
            price_count, min_profit_usd, max_slippage, total_combinations * 2 // offset
        );
        
        // Launch cross-chain arbitrage kernel
        cross_chain_arbitrage_kernel<<<grid_size, block_size, 0, streams[3]>>>(
            d_prices, d_opportunities, d_exchange_configs, d_chain_configs,
            price_count, min_profit_usd, max_slippage, total_combinations * 3 // offset
        );
        
        // Synchronize all streams
        for (int i = 0; i < 4; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        
        // Copy results back to host
        std::vector<GPUOpportunity> opportunities(max_opportunities);
        cudaMemcpy(opportunities.data(), d_opportunities, 
                   max_opportunities * sizeof(GPUOpportunity), cudaMemcpyDeviceToHost);
        
        // Filter out empty opportunities and sort by profit
        opportunities.erase(
            std::remove_if(opportunities.begin(), opportunities.end(),
                          [](const GPUOpportunity& opp) { return opp.net_profit_usd <= 0.0f; }),
            opportunities.end());
        
        std::sort(opportunities.begin(), opportunities.end(),
                 [](const GPUOpportunity& a, const GPUOpportunity& b) {
                     return a.net_profit_usd > b.net_profit_usd;
                 });
        
        // Update performance metrics
        total_scans++;
        total_opportunities += opportunities.size();
        for (const auto& opp : opportunities) {
            total_profit_potential += opp.net_profit_usd;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        if (total_scans % 100 == 0) {
            std::cout << "🔥 GPU Scan #" << total_scans 
                      << ": Found " << opportunities.size() << " opportunities in " 
                      << duration.count() << "μs ("
                      << (total_combinations * 4) / duration.count() << " ops/μs)" << std::endl;
        }
        
        return opportunities;
    }
    
    std::vector<GPUOpportunity> get_top_opportunities(size_t limit = 100) {
        auto all_opportunities = scan_all_opportunities();
        
        if (all_opportunities.size() <= limit) {
            return all_opportunities;
        }
        
        return std::vector<GPUOpportunity>(all_opportunities.begin(), 
                                          all_opportunities.begin() + limit);
    }
    
    void print_performance_stats() {
        std::cout << "\n📊 HyperArbitrage GPU Performance Stats:" << std::endl;
        std::cout << "   Total Scans: " << total_scans << std::endl;
        std::cout << "   Total Opportunities: " << total_opportunities << std::endl;
        std::cout << "   Total Profit Potential: $" << total_profit_potential << std::endl;
        std::cout << "   Avg Opportunities/Scan: " << (total_scans > 0 ? total_opportunities / total_scans : 0) << std::endl;
    }
};

// CUDA Kernels Implementation
__global__ void cross_exchange_arbitrage_kernel(
    const GPUPrice* __restrict__ prices,
    GPUOpportunity* __restrict__ opportunities,
    const ExchangeConfig* __restrict__ exchange_configs,
    const ChainConfig* __restrict__ chain_configs,
    size_t price_count,
    float min_profit_usd,
    float max_slippage,
    size_t opportunity_offset
) {
    const size_t total_combinations = (price_count * (price_count - 1)) / 2;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_combinations) return;
    
    // Convert linear index to pair indices
    size_t i = 0, j = 1;
    size_t remaining = tid;
    while (remaining >= price_count - i - 1) {
        remaining -= (price_count - i - 1);
        i++;
        j = i + 1;
    }
    j += remaining;
    
    const GPUPrice& price1 = prices[i];
    const GPUPrice& price2 = prices[j];
    
    // Only process same token on different exchanges
    if (price1.token_id != price2.token_id || price1.exchange_id == price2.exchange_id) {
        return;
    }
    
    // Calculate arbitrage opportunity
    float buy_price, sell_price;
    uint16_t buy_exchange, sell_exchange;
    uint16_t buy_chain, sell_chain;
    
    if (price1.ask < price2.bid) {
        buy_price = price1.ask;
        sell_price = price2.bid;
        buy_exchange = price1.exchange_id;
        sell_exchange = price2.exchange_id;
        buy_chain = price1.chain_id;
        sell_chain = price2.chain_id;
    } else if (price2.ask < price1.bid) {
        buy_price = price2.ask;
        sell_price = price1.bid;
        buy_exchange = price2.exchange_id;
        sell_exchange = price1.exchange_id;
        buy_chain = price2.chain_id;
        sell_chain = price1.chain_id;
    } else {
        return;
    }
    
    const float spread = (sell_price - buy_price) / buy_price;
    if (spread < 0.0001f) return; // Minimum 0.01% spread
    
    // Get exchange configurations
    const ExchangeConfig& buy_config = exchange_configs[buy_exchange];
    const ExchangeConfig& sell_config = exchange_configs[sell_exchange];
    
    // Calculate trade size based on liquidity and volume
    const float max_by_volume = fminf(price1.volume_24h, price2.volume_24h) * 0.01f;
    const float max_by_liquidity = fminf(price1.liquidity_usd, price2.liquidity_usd) * 0.05f;
    const float trade_size_usd = fminf(fmaxf(max_by_volume, 1000.0f), fminf(max_by_liquidity, 100000.0f));
    
    // Calculate costs
    const float trading_fees = trade_size_usd * (buy_config.fee_bps + sell_config.fee_bps) / 10000.0f;
    
    // Gas cost calculation
    float gas_cost = 0.0f;
    if (buy_config.is_dex || sell_config.is_dex) {
        const ChainConfig& buy_chain_config = chain_configs[buy_chain];
        const ChainConfig& sell_chain_config = chain_configs[sell_chain];
        
        const float gas_limit = (buy_chain != sell_chain) ? 400000.0f : 200000.0f;
        const float eth_price = 2500.0f; // Could be dynamic
        
        gas_cost = (gas_limit * price1.gas_price_gwei * buy_chain_config.gas_multiplier * eth_price) / 1e9f;
        if (buy_chain != sell_chain) {
            gas_cost += (gas_limit * price2.gas_price_gwei * sell_chain_config.gas_multiplier * eth_price) / 1e9f;
        }
    }
    
    // Bridge cost for cross-chain
    const float bridge_cost = (buy_chain != sell_chain) ? trade_size_usd * 0.001f : 0.0f;
    
    // Slippage calculation
    const float avg_liquidity = (price1.liquidity_usd + price2.liquidity_usd) * 0.5f;
    const float slippage_impact = (avg_liquidity > 0.0f) ? 
        fminf(sqrtf(trade_size_usd / avg_liquidity) * 0.1f, 0.05f) : 0.05f;
    
    if (slippage_impact > max_slippage) return;
    
    const float slippage_cost = trade_size_usd * slippage_impact;
    const float gross_profit = trade_size_usd * spread;
    const float total_costs = trading_fees + gas_cost + bridge_cost + slippage_cost;
    const float net_profit = gross_profit - total_costs;
    
    if (net_profit < min_profit_usd) return;
    
    // Execution time estimation
    uint16_t execution_time_ms;
    if (buy_chain != sell_chain) {
        execution_time_ms = 30000; // Cross-chain ~30s
    } else if (buy_config.is_dex || sell_config.is_dex) {
        execution_time_ms = 200; // DEX ~200ms
    } else {
        execution_time_ms = 50; // CEX ~50ms
    }
    
    // Confidence score calculation
    const float spread_score = fminf(spread * 1000.0f, 1.0f);
    const float volume_score = fminf(trade_size_usd / 100000.0f, 1.0f);
    const float slippage_score = 1.0f - slippage_impact;
    const float confidence = (spread_score * 0.4f + volume_score * 0.3f + slippage_score * 0.3f);
    
    // Store opportunity
    const size_t opp_index = opportunity_offset + tid;
    if (opp_index < MAX_OPPORTUNITIES) {
        GPUOpportunity& opp = opportunities[opp_index];
        opp.id = (uint64_t)tid + ((uint64_t)blockIdx.x << 32);
        opp.token_id = price1.token_id;
        opp.buy_exchange_id = buy_exchange;
        opp.sell_exchange_id = sell_exchange;
        opp.buy_chain_id = buy_chain;
        opp.sell_chain_id = sell_chain;
        opp.buy_price = buy_price;
        opp.sell_price = sell_price;
        opp.spread_percentage = spread * 100.0f;
        opp.gross_profit_usd = gross_profit;
        opp.net_profit_usd = net_profit;
        opp.gas_cost_usd = gas_cost + bridge_cost;
        opp.execution_time_ms = execution_time_ms;
        opp.confidence_score = confidence;
        opp.volume_limit_usd = trade_size_usd;
        opp.slippage_percentage = slippage_impact * 100.0f;
        opp.timestamp_ns = clock64();
        opp.strategy_type = 0; // Cross-exchange
        opp.mev_protection = buy_config.is_dex || sell_config.is_dex;
        opp.flash_loan_available = buy_config.supports_flash_loans || sell_config.supports_flash_loans;
    }
}

__global__ void triangular_arbitrage_kernel(
    const GPUPrice* __restrict__ prices,
    GPUOpportunity* __restrict__ opportunities,
    const ExchangeConfig* __restrict__ exchange_configs,
    const ChainConfig* __restrict__ chain_configs,
    size_t price_count,
    float min_profit_usd,
    size_t opportunity_offset
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // For triangular arbitrage, we need groups of 3 prices from same exchange
    if (tid >= price_count) return;
    
    const GPUPrice& base_price = prices[tid];
    const ExchangeConfig& exchange_config = exchange_configs[base_price.exchange_id];
    
    // Only process DEX or major CEX that support triangular arbitrage
    if (!exchange_config.is_dex && base_price.exchange_id > 7) return;
    
    // Find two other tokens on the same exchange
    for (size_t i = 0; i < price_count; ++i) {
        if (i == tid) continue;
        const GPUPrice& price2 = prices[i];
        if (price2.exchange_id != base_price.exchange_id) continue;
        
        for (size_t j = i + 1; j < price_count; ++j) {
            if (j == tid) continue;
            const GPUPrice& price3 = prices[j];
            if (price3.exchange_id != base_price.exchange_id) continue;
            
            // Calculate triangular rates
            const float rate1 = base_price.bid;
            const float rate2 = price2.bid;
            const float rate3 = 1.0f / price3.ask;
            
            const float final_rate = rate1 * rate2 * rate3;
            const float profit_rate = final_rate - 1.0f;
            
            if (profit_rate < 0.001f) continue; // Minimum 0.1% profit
            
            const float trade_size = fminf(fminf(base_price.volume_24h, price2.volume_24h), price3.volume_24h) * 0.01f;
            if (trade_size < 1000.0f) continue;
            
            const float gross_profit = trade_size * profit_rate;
            const float trading_fees = trade_size * exchange_config.fee_bps * 3.0f / 10000.0f; // 3 trades
            
            float gas_cost = 0.0f;
            if (exchange_config.is_dex) {
                const ChainConfig& chain_config = chain_configs[base_price.chain_id];
                gas_cost = (300000.0f * base_price.gas_price_gwei * chain_config.gas_multiplier * 2500.0f) / 1e9f;
            }
            
            const float net_profit = gross_profit - trading_fees - gas_cost;
            if (net_profit < min_profit_usd) continue;
            
            // Store triangular opportunity
            const size_t opp_index = opportunity_offset + tid * 1000 + (i * 100 + j) % 1000;
            if (opp_index < MAX_OPPORTUNITIES) {
                GPUOpportunity& opp = opportunities[opp_index];
                opp.id = (uint64_t)tid + ((uint64_t)i << 16) + ((uint64_t)j << 32);
                opp.token_id = base_price.token_id;
                opp.buy_exchange_id = base_price.exchange_id;
                opp.sell_exchange_id = base_price.exchange_id;
                opp.buy_chain_id = base_price.chain_id;
                opp.sell_chain_id = base_price.chain_id;
                opp.buy_price = rate1;
                opp.sell_price = final_rate;
                opp.spread_percentage = profit_rate * 100.0f;
                opp.gross_profit_usd = gross_profit;
                opp.net_profit_usd = net_profit;
                opp.gas_cost_usd = gas_cost;
                opp.execution_time_ms = exchange_config.is_dex ? 300 : 150;
                opp.confidence_score = fminf(profit_rate * 100.0f, 0.9f);
                opp.volume_limit_usd = trade_size;
                opp.slippage_percentage = 1.0f;
                opp.timestamp_ns = clock64();
                opp.strategy_type = 1; // Triangular
                opp.mev_protection = exchange_config.is_dex;
                opp.flash_loan_available = exchange_config.supports_flash_loans;
            }
            
            break; // Limit one triangular opportunity per base token
        }
    }
}

__global__ void flash_loan_arbitrage_kernel(
    const GPUPrice* __restrict__ prices,
    GPUOpportunity* __restrict__ opportunities,
    const ExchangeConfig* __restrict__ exchange_configs,
    const ChainConfig* __restrict__ chain_configs,
    size_t price_count,
    float min_profit_usd,
    float max_slippage,
    size_t opportunity_offset
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= price_count) return;
    
    const GPUPrice& dex_price = prices[tid];
    const ExchangeConfig& dex_config = exchange_configs[dex_price.exchange_id];
    
    // Only process DEX prices that support flash loans
    if (!dex_config.is_dex || !dex_config.supports_flash_loans) return;
    
    // Find corresponding CEX price for same token
    for (size_t i = 0; i < price_count; ++i) {
        if (i == tid) continue;
        
        const GPUPrice& cex_price = prices[i];
        const ExchangeConfig& cex_config = exchange_configs[cex_price.exchange_id];
        
        // Must be same token and CEX
        if (cex_price.token_id != dex_price.token_id || cex_config.is_dex) continue;
        
        // Calculate spread
        float spread;
        bool dex_to_cex;
        if (dex_price.ask < cex_price.bid) {
            spread = (cex_price.bid - dex_price.ask) / dex_price.ask;
            dex_to_cex = true;
        } else if (cex_price.ask < dex_price.bid) {
            spread = (dex_price.bid - cex_price.ask) / cex_price.ask;
            dex_to_cex = false;
        } else {
            continue;
        }
        
        if (spread < 0.003f) continue; // Minimum 0.3% for flash loans
        
        // Flash loan sizing
        const float loan_amount = fminf(dex_price.liquidity_usd * 0.1f, 1000000.0f);
        if (loan_amount < 50000.0f) continue;
        
        const float gross_profit = loan_amount * spread;
        const float flash_loan_fee = loan_amount * 0.0005f; // 0.05% Aave fee
        const float trading_fees = loan_amount * 0.001f;
        
        // Complex gas calculation for flash loan
        const ChainConfig& chain_config = chain_configs[dex_price.chain_id];
        const float gas_cost = (800000.0f * dex_price.gas_price_gwei * chain_config.gas_multiplier * 2500.0f) / 1e9f;
        
        const float slippage_impact = (dex_price.liquidity_usd > 0.0f) ? 
            fminf(sqrtf(loan_amount / dex_price.liquidity_usd) * 0.15f, 0.05f) : 0.05f;
        
        if (slippage_impact > max_slippage) continue;
        
        const float slippage_cost = loan_amount * slippage_impact;
        const float net_profit = gross_profit - flash_loan_fee - trading_fees - gas_cost - slippage_cost;
        
        if (net_profit < min_profit_usd * 5.0f) continue; // Higher threshold
        
        // Store flash loan opportunity
        const size_t opp_index = opportunity_offset + tid * 200 + i % 200;
        if (opp_index < MAX_OPPORTUNITIES) {
            GPUOpportunity& opp = opportunities[opp_index];
            opp.id = (uint64_t)tid + ((uint64_t)i << 24) + (1ULL << 63); // Flash loan flag
            opp.token_id = dex_price.token_id;
            opp.buy_exchange_id = dex_to_cex ? dex_price.exchange_id : cex_price.exchange_id;
            opp.sell_exchange_id = dex_to_cex ? cex_price.exchange_id : dex_price.exchange_id;
            opp.buy_chain_id = dex_price.chain_id;
            opp.sell_chain_id = 0; // CEX
            opp.buy_price = dex_to_cex ? dex_price.ask : cex_price.ask;
            opp.sell_price = dex_to_cex ? cex_price.bid : dex_price.bid;
            opp.spread_percentage = spread * 100.0f;
            opp.gross_profit_usd = gross_profit;
            opp.net_profit_usd = net_profit;
            opp.gas_cost_usd = flash_loan_fee + gas_cost;
            opp.execution_time_ms = 500;
            opp.confidence_score = fminf(spread * 200.0f, 0.95f);
            opp.volume_limit_usd = loan_amount;
            opp.slippage_percentage = slippage_impact * 100.0f;
            opp.timestamp_ns = clock64();
            opp.strategy_type = 2; // Flash loan
            opp.mev_protection = 1;
            opp.flash_loan_available = 1;
        }
        
        break; // One flash loan opportunity per DEX price
    }
}

__global__ void cross_chain_arbitrage_kernel(
    const GPUPrice* __restrict__ prices,
    GPUOpportunity* __restrict__ opportunities,
    const ExchangeConfig* __restrict__ exchange_configs,
    const ChainConfig* __restrict__ chain_configs,
    size_t price_count,
    float min_profit_usd,
    float max_slippage,
    size_t opportunity_offset
) {
    const size_t total_combinations = (price_count * (price_count - 1)) / 2;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_combinations) return;
    
    // Convert to pair indices
    size_t i = 0, j = 1;
    size_t remaining = tid;
    while (remaining >= price_count - i - 1) {
        remaining -= (price_count - i - 1);
        i++;
        j = i + 1;
    }
    j += remaining;
    
    const GPUPrice& price1 = prices[i];
    const GPUPrice& price2 = prices[j];
    
    // Only process cross-chain DEX opportunities
    if (price1.token_id != price2.token_id || 
        price1.chain_id == price2.chain_id ||
        !exchange_configs[price1.exchange_id].is_dex ||
        !exchange_configs[price2.exchange_id].is_dex) {
        return;
    }
    
    float spread;
    uint16_t buy_exchange, sell_exchange, buy_chain, sell_chain;
    float buy_price, sell_price;
    
    if (price1.ask < price2.bid) {
        spread = (price2.bid - price1.ask) / price1.ask;
        buy_exchange = price1.exchange_id;
        sell_exchange = price2.exchange_id;
        buy_chain = price1.chain_id;
        sell_chain = price2.chain_id;
        buy_price = price1.ask;
        sell_price = price2.bid;
    } else if (price2.ask < price1.bid) {
        spread = (price1.bid - price2.ask) / price2.ask;
        buy_exchange = price2.exchange_id;
        sell_exchange = price1.exchange_id;
        buy_chain = price2.chain_id;
        sell_chain = price1.chain_id;
        buy_price = price2.ask;
        sell_price = price1.bid;
    } else {
        return;
    }
    
    if (spread < 0.008f) return; // Minimum 0.8% for cross-chain
    
    const float trade_size = fminf(fminf(price1.liquidity_usd, price2.liquidity_usd) * 0.05f, 200000.0f);
    if (trade_size < 10000.0f) return;
    
    const float gross_profit = trade_size * spread;
    
    // Cross-chain costs
    const ChainConfig& buy_chain_config = chain_configs[buy_chain];
    const ChainConfig& sell_chain_config = chain_configs[sell_chain];
    
    const float bridge_fee = trade_size * fmaxf(buy_chain_config.bridge_fee_bps, sell_chain_config.bridge_fee_bps) / 10000.0f;
    const float gas_cost_buy = (300000.0f * price1.gas_price_gwei * buy_chain_config.gas_multiplier * 2500.0f) / 1e9f;
    const float gas_cost_sell = (200000.0f * price2.gas_price_gwei * sell_chain_config.gas_multiplier * 2500.0f) / 1e9f;
    const float time_value_cost = trade_size * 0.0001f; // Opportunity cost
    
    const float slippage_impact = (sqrtf(trade_size / fminf(price1.liquidity_usd, price2.liquidity_usd)) * 0.1f);
    if (slippage_impact > max_slippage) return;
    
    const float total_costs = bridge_fee + gas_cost_buy + gas_cost_sell + time_value_cost + (trade_size * slippage_impact);
    const float net_profit = gross_profit - total_costs;
    
    if (net_profit < min_profit_usd * 20.0f) return; // Much higher threshold for cross-chain
    
    // Store cross-chain opportunity
    const size_t opp_index = opportunity_offset + tid;
    if (opp_index < MAX_OPPORTUNITIES) {
        GPUOpportunity& opp = opportunities[opp_index];
        opp.id = (uint64_t)tid + (2ULL << 62); // Cross-chain flag
        opp.token_id = price1.token_id;
        opp.buy_exchange_id = buy_exchange;
        opp.sell_exchange_id = sell_exchange;
        opp.buy_chain_id = buy_chain;
        opp.sell_chain_id = sell_chain;
        opp.buy_price = buy_price;
        opp.sell_price = sell_price;
        opp.spread_percentage = spread * 100.0f;
        opp.gross_profit_usd = gross_profit;
        opp.net_profit_usd = net_profit;
        opp.gas_cost_usd = bridge_fee + gas_cost_buy + gas_cost_sell;
        opp.execution_time_ms = 45000; // 45 seconds
        opp.confidence_score = fminf(spread * 50.0f, 0.8f);
        opp.volume_limit_usd = trade_size;
        opp.slippage_percentage = slippage_impact * 100.0f;
        opp.timestamp_ns = clock64();
        opp.strategy_type = 3; // Cross-chain
        opp.mev_protection = 1;
        opp.flash_loan_available = 0;
    }
}

// Python bindings
extern "C" {
    HyperArbitrageGPU* create_gpu_engine(size_t max_prices, size_t max_opps, int streams) {
        return new HyperArbitrageGPU(max_prices, max_opps, streams);
    }
    
    void destroy_gpu_engine(HyperArbitrageGPU* engine) {
        delete engine;
    }
    
    void update_gpu_prices(HyperArbitrageGPU* engine, GPUPrice* prices, size_t count) {
        std::vector<GPUPrice> price_vector(prices, prices + count);
        engine->update_prices(price_vector);
    }
    
    size_t scan_gpu_opportunities(HyperArbitrageGPU* engine, GPUOpportunity* results, 
                                  size_t max_results, float min_profit, float max_slippage) {
        auto opportunities = engine->scan_all_opportunities(min_profit, max_slippage);
        size_t count = std::min(opportunities.size(), max_results);
        std::copy(opportunities.begin(), opportunities.begin() + count, results);
        return count;
    }
    
    void print_gpu_stats(HyperArbitrageGPU* engine) {
        engine->print_performance_stats();
    }
}
EOF

echo "✅ Generated C++/CUDA GPU Engine"