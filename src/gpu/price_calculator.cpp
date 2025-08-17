#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>

struct PriceData {
    float bid;
    float ask;
    float volume;
    uint64_t timestamp;
    int exchange_id;
    int chain_id;
};

struct ArbitrageResult {
    float spread;
    float profit;
    float gas_cost;
    float net_profit;
    int buy_exchange;
    int sell_exchange;
    int buy_chain;
    int sell_chain;
    float confidence;
};

class GPUArbitrageCalculator {
private:
    cublasHandle_t cublas_handle;
    int max_exchanges;
    int max_symbols;
    
public:
    GPUArbitrageCalculator(int exchanges = 50, int symbols = 10000) 
        : max_exchanges(exchanges), max_symbols(symbols) {
        cublasCreate(&cublas_handle);
        
        // Initialize CUDA
        cudaSetDevice(0);
        
        // Warm up GPU
        warmupGPU();
    }
    
    ~GPUArbitrageCalculator() {
        cublasDestroy(cublas_handle);
    }
    
    void warmupGPU() {
        // Allocate and free memory to warm up GPU
        float* temp;
        cudaMalloc(&temp, 1024 * sizeof(float));
        cudaFree(temp);
        cudaDeviceSynchronize();
    }
    
    std::vector<ArbitrageResult> calculateArbitrageOpportunities(
        const std::vector<PriceData>& prices,
        const std::vector<float>& gas_prices,
        const std::vector<float>& bridge_costs,
        float min_profit_threshold = 10.0f
    ) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Copy data to GPU
        thrust::device_vector<PriceData> d_prices(prices);
        thrust::device_vector<float> d_gas_prices(gas_prices);
        thrust::device_vector<float> d_bridge_costs(bridge_costs);
        
        // Calculate pairwise arbitrage opportunities
        size_t num_combinations = prices.size() * (prices.size() - 1) / 2;
        thrust::device_vector<ArbitrageResult> d_results(num_combinations);
        
        // Launch CUDA kernel
        dim3 blockSize(256);
        dim3 gridSize((num_combinations + blockSize.x - 1) / blockSize.x);
        
        calculateArbitrageKernel<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_prices.data()),
            thrust::raw_pointer_cast(d_gas_prices.data()),
            thrust::raw_pointer_cast(d_bridge_costs.data()),
            thrust::raw_pointer_cast(d_results.data()),
            prices.size(),
            min_profit_threshold
        );
        
        cudaDeviceSynchronize();
        
        // Filter profitable opportunities
        thrust::device_vector<ArbitrageResult> d_profitable;
        thrust::copy_if(d_results.begin(), d_results.end(), 
                       thrust::back_inserter(d_profitable),
                       [](const ArbitrageResult& r) { return r.net_profit > 0; });
        
        // Sort by net profit (descending)
        thrust::sort(d_profitable.begin(), d_profitable.end(),
                    [](const ArbitrageResult& a, const ArbitrageResult& b) {
                        return a.net_profit > b.net_profit;
                    });
        
        // Copy results back to host
        std::vector<ArbitrageResult> results(d_profitable.size());
        thrust::copy(d_profitable.begin(), d_profitable.end(), results.begin());
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "GPU calculation completed in " << duration.count() << " microseconds\n";
        std::cout << "Found " << results.size() << " profitable opportunities\n";
        
        return results;
    }
    
    std::vector<float> calculateTriangularArbitrage(
        const std::vector<std::vector<float>>& rate_matrix,
        int num_currencies
    ) {
        // Floyd-Warshall algorithm on GPU for triangular arbitrage
        thrust::device_vector<float> d_rates(rate_matrix.size() * rate_matrix[0].size());
        
        // Copy matrix to device
        for (size_t i = 0; i < rate_matrix.size(); ++i) {
            thrust::copy(rate_matrix[i].begin(), rate_matrix[i].end(),
                        d_rates.begin() + i * rate_matrix[i].size());
        }
        
        // Launch Floyd-Warshall kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((num_currencies + blockSize.x - 1) / blockSize.x,
                     (num_currencies + blockSize.y - 1) / blockSize.y);
        
        for (int k = 0; k < num_currencies; ++k) {
            floydWarshallKernel<<<gridSize, blockSize>>>(
                thrust::raw_pointer_cast(d_rates.data()),
                num_currencies, k
            );
            cudaDeviceSynchronize();
        }
        
        // Find profitable cycles
        std::vector<float> cycles(num_currencies);
        thrust::copy(d_rates.begin(), d_rates.begin() + num_currencies, cycles.begin());
        
        return cycles;
    }
    
    void optimizeGasStrategy(
        const std::vector<float>& gas_prices,
        const std::vector<float>& profit_estimates,
        std::vector<float>& optimal_gas_prices
    ) {
        // GPU-optimized gas price calculation
        thrust::device_vector<float> d_gas_prices(gas_prices);
        thrust::device_vector<float> d_profits(profit_estimates);
        thrust::device_vector<float> d_optimal(gas_prices.size());
        
        dim3 blockSize(256);
        dim3 gridSize((gas_prices.size() + blockSize.x - 1) / blockSize.x);
        
        optimizeGasKernel<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_gas_prices.data()),
            thrust::raw_pointer_cast(d_profits.data()),
            thrust::raw_pointer_cast(d_optimal.data()),
            gas_prices.size()
        );
        
        cudaDeviceSynchronize();
        
        optimal_gas_prices.resize(gas_prices.size());
        thrust::copy(d_optimal.begin(), d_optimal.end(), optimal_gas_prices.begin());
    }
    
    void parallelPriceUpdate(
        std::vector<PriceData>& prices,
        const std::vector<float>& new_bids,
        const std::vector<float>& new_asks,
        const std::vector<float>& new_volumes
    ) {
        // Parallel price updates using GPU
        thrust::device_vector<PriceData> d_prices(prices);
        thrust::device_vector<float> d_bids(new_bids);
        thrust::device_vector<float> d_asks(new_asks);
        thrust::device_vector<float> d_volumes(new_volumes);
        
        dim3 blockSize(256);
        dim3 gridSize((prices.size() + blockSize.x - 1) / blockSize.x);
        
        updatePricesKernel<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_prices.data()),
            thrust::raw_pointer_cast(d_bids.data()),
            thrust::raw_pointer_cast(d_asks.data()),
            thrust::raw_pointer_cast(d_volumes.data()),
            prices.size()
        );
        
        cudaDeviceSynchronize();
        
        thrust::copy(d_prices.begin(), d_prices.end(), prices.begin());
    }
};

// CUDA Kernels
extern "C" {

__global__ void calculateArbitrageKernel(
    const PriceData* prices,
    const float* gas_prices,
    const float* bridge_costs,
    ArbitrageResult* results,
    int num_prices,
    float min_profit
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_combinations = num_prices * (num_prices - 1) / 2;
    
    if (idx >= total_combinations) return;
    
    // Convert linear index to pair indices
    int i = 0, j = 1;
    int remaining = idx;
    while (remaining >= num_prices - i - 1) {
        remaining -= (num_prices - i - 1);
        i++;
        j = i + 1;
    }
    j += remaining;
    
    const PriceData& price_i = prices[i];
    const PriceData& price_j = prices[j];
    
    // Calculate arbitrage opportunity
    float spread = 0.0f;
    float buy_price = 0.0f;
    float sell_price = 0.0f;
    int buy_exchange = i;
    int sell_exchange = j;
    
    // Check both directions
    if (price_i.ask < price_j.bid) {
        spread = (price_j.bid - price_i.ask) / price_i.ask;
        buy_price = price_i.ask;
        sell_price = price_j.bid;
    } else if (price_j.ask < price_i.bid) {
        spread = (price_i.bid - price_j.ask) / price_j.ask;
        buy_price = price_j.ask;
        sell_price = price_i.bid;
        buy_exchange = j;
        sell_exchange = i;
    }
    
    if (spread > 0.002f) { // Minimum 0.2% spread
        float trade_size = 10000.0f;
        float gross_profit = trade_size * spread;
        
        // Calculate costs
        float gas_cost = 0.0f;
        float bridge_cost = 0.0f;
        
        if (price_i.chain_id != price_j.chain_id) {
            bridge_cost = bridge_costs[price_i.chain_id] + bridge_costs[price_j.chain_id];
        }
        
        if (price_i.chain_id != 0) { // Not CEX
            gas_cost += gas_prices[price_i.chain_id];
        }
        if (price_j.chain_id != 0) { // Not CEX
            gas_cost += gas_prices[price_j.chain_id];
        }
        
        float total_fees = trade_size * 0.002f; // 0.2% trading fees
        float net_profit = gross_profit - total_fees - gas_cost - bridge_cost;
        
        float confidence = 0.8f;
        if (price_i.chain_id != price_j.chain_id) confidence = 0.75f; // Lower for cross-chain
        if (spread > 0.01f) confidence = 0.9f; // Higher for large spreads
        
        results[idx] = {
            spread * 100.0f,
            gross_profit,
            gas_cost + bridge_cost,
            net_profit,
            buy_exchange,
            sell_exchange,
            price_i.chain_id,
            price_j.chain_id,
            confidence
        };
    } else {
        results[idx] = {0.0f, 0.0f, 0.0f, -1.0f, -1, -1, -1, -1, 0.0f};
    }
}

__global__ void floydWarshallKernel(
    float* rates,
    int num_currencies,
    int k
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num_currencies && j < num_currencies) {
        float new_rate = rates[i * num_currencies + k] * rates[k * num_currencies + j];
        if (new_rate > rates[i * num_currencies + j]) {
            rates[i * num_currencies + j] = new_rate;
        }
    }
}

__global__ void optimizeGasKernel(
    const float* gas_prices,
    const float* profits,
    float* optimal_gas,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float profit = profits[idx];
        float base_gas = gas_prices[idx];
        
        // Optimize gas price based on profit potential
        if (profit > 1000.0f) {
            optimal_gas[idx] = base_gas * 2.0f; // Urgent
        } else if (profit > 100.0f) {
            optimal_gas[idx] = base_gas * 1.5f; // High priority
        } else if (profit > 10.0f) {
            optimal_gas[idx] = base_gas * 1.2f; // Normal
        } else {
            optimal_gas[idx] = base_gas * 0.9f; // Low priority
        }
    }
}

__global__ void updatePricesKernel(
    PriceData* prices,
    const float* new_bids,
    const float* new_asks,
    const float* new_volumes,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        prices[idx].bid = new_bids[idx];
        prices[idx].ask = new_asks[idx];
        prices[idx].volume = new_volumes[idx];
        prices[idx].timestamp = clock64();
    }
}

} // extern "C"

// Python binding functions
extern "C" {
    GPUArbitrageCalculator* create_calculator(int exchanges, int symbols) {
        return new GPUArbitrageCalculator(exchanges, symbols);
    }
    
    void destroy_calculator(GPUArbitrageCalculator* calc) {
        delete calc;
    }
    
    int calculate_opportunities(
        GPUArbitrageCalculator* calc,
        PriceData* prices,
        int num_prices,
        float* gas_prices,
        float* bridge_costs,
        ArbitrageResult* results,
        int max_results
    ) {
        std::vector<PriceData> price_vec(prices, prices + num_prices);
        std::vector<float> gas_vec(gas_prices, gas_prices + 10); // Assume 10 chains
        std::vector<float> bridge_vec(bridge_costs, bridge_costs + 10);
        
        auto opportunities = calc->calculateArbitrageOpportunities(price_vec, gas_vec, bridge_vec);
        
        int num_results = std::min(static_cast<int>(opportunities.size()), max_results);
        std::copy(opportunities.begin(), opportunities.begin() + num_results, results);
        
        return num_results;
    }
}