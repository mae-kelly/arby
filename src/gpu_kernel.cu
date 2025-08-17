#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define WARP_SIZE 32
#define MAX_EXCHANGES 100
#define MAX_SYMBOLS 10000
#define MAX_PATHS 100000
#define BLOCK_SIZE 256

struct Market {
    float bid;
    float ask;
    float volume;
    uint32_t exchange_id;
    uint32_t symbol_id;
    uint64_t timestamp;
};

struct Path {
    uint32_t exchanges[8];
    uint32_t symbols[8];
    float profit;
    float confidence;
    uint8_t length;
};

__constant__ float fees[MAX_EXCHANGES];
__constant__ float min_profit = 0.001f;

// Optimized matrix for price differentials
__device__ float price_matrix[MAX_SYMBOLS][MAX_EXCHANGES];

__global__ void update_price_matrix(
    Market* markets,
    int num_markets
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_markets) {
        Market m = markets[idx];
        atomicExch(&price_matrix[m.symbol_id][m.exchange_id], (m.bid + m.ask) / 2.0f);
    }
}

__global__ void find_direct_arbitrage(
    Path* paths,
    int* path_count,
    int num_symbols,
    int num_exchanges
) {
    __shared__ float shared_prices[BLOCK_SIZE][4];
    
    cg::thread_block block = cg::this_thread_block();
    int tid = threadIdx.x;
    int symbol_id = blockIdx.x;
    
    if (symbol_id >= num_symbols) return;
    
    // Load prices to shared memory
    for (int i = tid; i < num_exchanges; i += blockDim.x) {
        shared_prices[tid][i % 4] = price_matrix[symbol_id][i];
    }
    block.sync();
    
    // Find best arbitrage for this symbol
    float max_profit = 0.0f;
    int best_buy = -1, best_sell = -1;
    
    for (int buy_ex = 0; buy_ex < num_exchanges; buy_ex++) {
        float buy_price = price_matrix[symbol_id][buy_ex];
        if (buy_price <= 0) continue;
        
        for (int sell_ex = buy_ex + 1; sell_ex < num_exchanges; sell_ex++) {
            float sell_price = price_matrix[symbol_id][sell_ex];
            if (sell_price <= 0) continue;
            
            float buy_cost = buy_price * (1.0f + fees[buy_ex]);
            float sell_revenue = sell_price * (1.0f - fees[sell_ex]);
            
            float profit = (sell_revenue - buy_cost) / buy_cost;
            
            if (profit > max_profit && profit > min_profit) {
                max_profit = profit;
                best_buy = buy_ex;
                best_sell = sell_ex;
            }
        }
    }
    
    // Write result if profitable
    if (best_buy >= 0 && tid == 0) {
        int idx = atomicAdd(path_count, 1);
        if (idx < MAX_PATHS) {
            Path p;
            p.exchanges[0] = best_buy;
            p.exchanges[1] = best_sell;
            p.symbols[0] = symbol_id;
            p.symbols[1] = symbol_id;
            p.profit = max_profit;
            p.confidence = 0.9f;
            p.length = 2;
            
            paths[idx] = p;
        }
    }
}

__global__ void find_triangular_arbitrage(
    Path* paths,
    int* path_count,
    int num_symbols,
    int num_exchanges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_combinations = num_symbols * num_symbols * num_symbols;
    
    if (tid >= total_combinations) return;
    
    // Decode combination
    int s1 = tid % num_symbols;
    int s2 = (tid / num_symbols) % num_symbols;
    int s3 = (tid / (num_symbols * num_symbols)) % num_symbols;
    
    if (s1 == s2 || s2 == s3 || s1 == s3) return;
    
    // Find best path through three symbols
    float best_profit = 0.0f;
    int best_exchanges[3] = {-1, -1, -1};
    
    for (int e1 = 0; e1 < num_exchanges; e1++) {
        float p1 = price_matrix[s1][e1];
        if (p1 <= 0) continue;
        
        for (int e2 = 0; e2 < num_exchanges; e2++) {
            float p2 = price_matrix[s2][e2];
            if (p2 <= 0) continue;
            
            for (int e3 = 0; e3 < num_exchanges; e3++) {
                float p3 = price_matrix[s3][e3];
                if (p3 <= 0) continue;
                
                // Calculate triangular arbitrage
                float rate1 = p2 / p1 * (1.0f - fees[e1]);
                float rate2 = p3 / p2 * (1.0f - fees[e2]);
                float rate3 = p1 / p3 * (1.0f - fees[e3]);
                
                float total_rate = rate1 * rate2 * rate3;
                float profit = total_rate - 1.0f;
                
                if (profit > best_profit && profit > min_profit) {
                    best_profit = profit;
                    best_exchanges[0] = e1;
                    best_exchanges[1] = e2;
                    best_exchanges[2] = e3;
                }
            }
        }
    }
    
    // Write result
    if (best_profit > min_profit) {
        int idx = atomicAdd(path_count, 1);
        if (idx < MAX_PATHS) {
            Path p;
            p.symbols[0] = s1;
            p.symbols[1] = s2;
            p.symbols[2] = s3;
            p.exchanges[0] = best_exchanges[0];
            p.exchanges[1] = best_exchanges[1];
            p.exchanges[2] = best_exchanges[2];
            p.profit = best_profit;
            p.confidence = 0.75f;
            p.length = 3;
            
            paths[idx] = p;
        }
    }
}

__global__ void calculate_cross_chain_opportunities(
    Path* paths,
    int* path_count,
    float* bridge_fees,
    int num_chains,
    int num_symbols
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_symbols * num_chains * num_chains) return;
    
    int symbol = tid % num_symbols;
    int chain1 = (tid / num_symbols) % num_chains;
    int chain2 = (tid / (num_symbols * num_chains)) % num_chains;
    
    if (chain1 == chain2) return;
    
    // Get prices on different chains
    float price_chain1 = 0.0f;
    float price_chain2 = 0.0f;
    
    // Find best prices on each chain
    for (int ex = chain1 * 10; ex < (chain1 + 1) * 10 && ex < MAX_EXCHANGES; ex++) {
        float p = price_matrix[symbol][ex];
        if (p > price_chain1) price_chain1 = p;
    }
    
    for (int ex = chain2 * 10; ex < (chain2 + 1) * 10 && ex < MAX_EXCHANGES; ex++) {
        float p = price_matrix[symbol][ex];
        if (p > price_chain2) price_chain2 = p;
    }
    
    if (price_chain1 > 0 && price_chain2 > 0) {
        float bridge_fee = bridge_fees[chain1 * num_chains + chain2];
        float profit = (price_chain2 - price_chain1) / price_chain1 - bridge_fee;
        
        if (profit > min_profit) {
            int idx = atomicAdd(path_count, 1);
            if (idx < MAX_PATHS) {
                Path p;
                p.symbols[0] = symbol;
                p.exchanges[0] = chain1;
                p.exchanges[1] = chain2;
                p.profit = profit;
                p.confidence = 0.6f;
                p.length = 2;
                
                paths[idx] = p;
            }
        }
    }
}

__global__ void rank_opportunities(
    Path* paths,
    int num_paths,
    float* scores
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_paths) {
        Path p = paths[idx];
        
        // Score based on profit, confidence, and complexity
        float score = p.profit * 1000.0f;
        score *= p.confidence;
        score /= (1.0f + p.length * 0.1f);
        
        scores[idx] = score;
    }
}

extern "C" {
    void* create_gpu_context() {
        cudaSetDevice(0);
        
        Path* d_paths;
        int* d_path_count;
        float* d_scores;
        
        cudaMalloc(&d_paths, MAX_PATHS * sizeof(Path));
        cudaMalloc(&d_path_count, sizeof(int));
        cudaMalloc(&d_scores, MAX_PATHS * sizeof(float));
        
        cudaMemset(d_path_count, 0, sizeof(int));
        
        // Return context
        void** context = new void*[3];
        context[0] = d_paths;
        context[1] = d_path_count;
        context[2] = d_scores;
        
        return context;
    }
    
    int find_opportunities_gpu(
        void* context,
        Market* markets,
        int num_markets,
        Path* output_paths
    ) {
        void** ctx = (void**)context;
        Path* d_paths = (Path*)ctx[0];
        int* d_path_count = (int*)ctx[1];
        float* d_scores = (float*)ctx[2];
        
        // Reset counter
        cudaMemset(d_path_count, 0, sizeof(int));
        
        // Upload markets
        Market* d_markets;
        cudaMalloc(&d_markets, num_markets * sizeof(Market));
        cudaMemcpy(d_markets, markets, num_markets * sizeof(Market), cudaMemcpyHostToDevice);
        
        // Update price matrix
        int blocks = (num_markets + BLOCK_SIZE - 1) / BLOCK_SIZE;
        update_price_matrix<<<blocks, BLOCK_SIZE>>>(d_markets, num_markets);
        
        // Find direct arbitrage
        find_direct_arbitrage<<<MAX_SYMBOLS, BLOCK_SIZE>>>(
            d_paths, d_path_count, MAX_SYMBOLS, MAX_EXCHANGES
        );
        
        // Find triangular arbitrage
        int tri_blocks = (MAX_SYMBOLS * MAX_SYMBOLS * 10 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        find_triangular_arbitrage<<<tri_blocks, BLOCK_SIZE>>>(
            d_paths, d_path_count, MAX_SYMBOLS, MAX_EXCHANGES
        );
        
        // Get path count
        int h_path_count;
        cudaMemcpy(&h_path_count, d_path_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Rank opportunities
        blocks = (h_path_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        rank_opportunities<<<blocks, BLOCK_SIZE>>>(d_paths, h_path_count, d_scores);
        
        // Sort by score
        thrust::device_ptr<float> scores_ptr(d_scores);
        thrust::device_ptr<Path> paths_ptr(d_paths);
        thrust::sort_by_key(scores_ptr, scores_ptr + h_path_count, paths_ptr, thrust::greater<float>());
        
        // Copy results
        int output_count = (h_path_count < 1000) ? h_path_count : 1000;
        cudaMemcpy(output_paths, d_paths, output_count * sizeof(Path), cudaMemcpyDeviceToHost);
        
        cudaFree(d_markets);
        
        return output_count;
    }
    
    void destroy_gpu_context(void* context) {
        void** ctx = (void**)context;
        cudaFree(ctx[0]);
        cudaFree(ctx[1]);
        cudaFree(ctx[2]);
        delete[] ctx;
    }
}