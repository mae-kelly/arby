#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

#define MAX_TOKENS 1000
#define MAX_MARKETS 10000
#define MAX_PATH_LENGTH 5
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define PROFIT_THRESHOLD 0.001f

namespace cg = cooperative_groups;

struct Market {
    int token_a;
    int token_b;
    float reserve_a;
    float reserve_b;
    float fee;
    int exchange_id;
    int chain_id;
};

struct Path {
    int markets[MAX_PATH_LENGTH];
    int length;
    float profit;
    float gas_cost;
};

struct PathResult {
    Path path;
    float score;
};

__constant__ Market d_markets[MAX_MARKETS];
__constant__ float d_gas_prices[10];  // Per chain

texture<float4, cudaTextureType2D, cudaReadModeElementType> tex_prices;

// Optimized shared memory allocation
template<int BLOCK_DIM>
__device__ void load_markets_shared(Market* s_markets, int market_count) {
    __shared__ float4 price_cache[BLOCK_DIM];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    if (gid < market_count) {
        s_markets[tid] = d_markets[gid];
    }
    __syncthreads();
}

__device__ float calculate_output(float input, float reserve_in, float reserve_out, float fee) {
    float input_with_fee = input * (1.0f - fee);
    float numerator = input_with_fee * reserve_out;
    float denominator = reserve_in + input_with_fee;
    return numerator / denominator;
}

__device__ float evaluate_path(const int* path_markets, int length, float initial_amount) {
    float amount = initial_amount;
    
    #pragma unroll
    for (int i = 0; i < length; i++) {
        if (i >= MAX_PATH_LENGTH) break;
        
        Market m = d_markets[path_markets[i]];
        amount = calculate_output(amount, m.reserve_a, m.reserve_b, m.fee);
        
        if (amount <= 0) return -1.0f;
    }
    
    return (amount - initial_amount) / initial_amount;
}

__global__ void find_arbitrage_paths_kernel(
    PathResult* results,
    int* result_count,
    int market_count,
    int token_count,
    float min_profit
) {
    __shared__ int s_adjacency[MAX_TOKENS][32];
    __shared__ int s_adjacency_count[MAX_TOKENS];
    __shared__ PathResult s_best_paths[BLOCK_SIZE];
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Initialize shared memory
    if (tid < token_count) {
        s_adjacency_count[tid] = 0;
    }
    block.sync();
    
    // Build adjacency list in shared memory
    for (int i = tid; i < market_count; i += blockDim.x) {
        Market m = d_markets[i];
        int idx_a = atomicAdd(&s_adjacency_count[m.token_a], 1);
        if (idx_a < 32) {
            s_adjacency[m.token_a][idx_a] = i;
        }
    }
    block.sync();
    
    // DFS with warp-level parallelism
    curandState state;
    curand_init(clock64() + gid, 0, 0, &state);
    
    PathResult best_path;
    best_path.score = 0;
    
    // Each warp explores different starting tokens
    int start_token = warp_id % token_count;
    
    // Stack for DFS
    __shared__ int stack[BLOCK_SIZE][MAX_PATH_LENGTH];
    __shared__ int stack_depth[BLOCK_SIZE];
    
    if (lane_id == 0) {
        stack_depth[tid] = 0;
    }
    warp.sync();
    
    // Parallel DFS within warp
    for (int depth = 0; depth < MAX_PATH_LENGTH; depth++) {
        int current_token = (depth == 0) ? start_token : -1;
        
        if (current_token >= 0 && current_token < token_count) {
            int adj_count = s_adjacency_count[current_token];
            
            // Each lane in warp explores different market
            for (int j = lane_id; j < adj_count; j += WARP_SIZE) {
                int market_id = s_adjacency[current_token][j];
                Market m = d_markets[market_id];
                
                // Add to path
                stack[tid][depth] = market_id;
                
                // Evaluate if we have a cycle
                if (depth > 0 && m.token_b == start_token) {
                    float profit = evaluate_path(stack[tid], depth + 1, 1000000.0f);
                    
                    if (profit > min_profit && profit > best_path.score) {
                        best_path.score = profit;
                        best_path.path.length = depth + 1;
                        
                        #pragma unroll
                        for (int k = 0; k <= depth; k++) {
                            best_path.path.markets[k] = stack[tid][k];
                        }
                        
                        best_path.path.profit = profit;
                        best_path.path.gas_cost = (depth + 1) * d_gas_prices[m.chain_id];
                    }
                }
            }
        }
        warp.sync();
    }
    
    // Reduce within warp to find best path
    float warp_best_score = best_path.score;
    int best_lane = lane_id;
    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_score = __shfl_down_sync(0xffffffff, warp_best_score, offset);
        int other_lane = __shfl_down_sync(0xffffffff, best_lane, offset);
        
        if (other_score > warp_best_score) {
            warp_best_score = other_score;
            best_lane = other_lane;
        }
    }
    
    // Write result
    if (lane_id == 0 && warp_best_score > min_profit) {
        s_best_paths[warp_id] = best_path;
    }
    block.sync();
    
    // Final reduction and output
    if (tid == 0) {
        for (int i = 0; i < blockDim.x / WARP_SIZE; i++) {
            if (s_best_paths[i].score > min_profit) {
                int idx = atomicAdd(result_count, 1);
                if (idx < MAX_MARKETS) {
                    results[idx] = s_best_paths[i];
                }
            }
        }
    }
}

__global__ void optimize_execution_order(
    PathResult* paths,
    int path_count,
    float* scores
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < path_count) {
        PathResult p = paths[tid];
        
        // Calculate execution score based on profit, gas cost, and complexity
        float score = p.path.profit * 1000000.0f;  // Convert to USD
        score -= p.path.gas_cost;
        score /= (1.0f + p.path.length * 0.1f);  // Penalize complex paths
        
        // Consider market depth and slippage
        float total_liquidity = 0;
        for (int i = 0; i < p.path.length; i++) {
            Market m = d_markets[p.path.markets[i]];
            total_liquidity += m.reserve_a + m.reserve_b;
        }
        
        score *= __logf(total_liquidity + 1.0f);
        scores[tid] = score;
    }
}

extern "C" {
    void* cuda_find_arbitrage(
        float* market_data,
        int market_count,
        int token_count,
        float min_profit,
        int* result_count
    ) {
        // Allocate device memory
        Market* d_market_data;
        PathResult* d_results;
        int* d_result_count;
        float* d_scores;
        
        cudaMalloc(&d_market_data, market_count * sizeof(Market));
        cudaMalloc(&d_results, MAX_MARKETS * sizeof(PathResult));
        cudaMalloc(&d_result_count, sizeof(int));
        cudaMalloc(&d_scores, MAX_MARKETS * sizeof(float));
        
        // Copy data to device
        cudaMemcpy(d_market_data, market_data, market_count * sizeof(Market), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_markets, market_data, market_count * sizeof(Market));
        cudaMemset(d_result_count, 0, sizeof(int));
        
        // Configure kernel launch
        int threads = BLOCK_SIZE;
        int blocks = (market_count + threads - 1) / threads;
        blocks = min(blocks, 65535);
        
        // Launch path finding kernel
        find_arbitrage_paths_kernel<<<blocks, threads>>>(
            d_results,
            d_result_count,
            market_count,
            token_count,
            min_profit
        );
        
        // Launch optimization kernel
        optimize_execution_order<<<blocks, threads>>>(
            d_results,
            market_count,
            d_scores
        );
        
        // Sort results by score
        thrust::device_ptr<float> scores_ptr(d_scores);
        thrust::device_ptr<PathResult> results_ptr(d_results);
        thrust::sort_by_key(scores_ptr, scores_ptr + market_count, results_ptr, thrust::greater<float>());
        
        // Copy results back
        cudaMemcpy(result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Clean up
        cudaFree(d_market_data);
        cudaFree(d_result_count);
        cudaFree(d_scores);
        
        return d_results;  // Return device pointer for further processing
    }
    
    void cuda_free_results(void* results) {
        cudaFree(results);
    }
}