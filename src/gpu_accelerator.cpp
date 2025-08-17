#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <vector>
#include <memory>
#include <iostream>
#include <chrono>
#include <cmath>

struct ArbitrageData {
    float buy_price;
    float sell_price;
    float volume;
    float spread;
    float confidence;
    int exchange_id_buy;
    int exchange_id_sell;
    int token_id;
};

struct OpportunityScore {
    float score;
    int index;
};

class GPUArbitrageAccelerator {
private:
    cublasHandle_t cublas_handle;
    curandGenerator_t curand_gen;
    int device_id;
    size_t available_memory;
    
    thrust::device_vector<ArbitrageData> d_arbitrage_data;
    thrust::device_vector<float> d_price_matrix;
    thrust::device_vector<float> d_volume_matrix;
    thrust::device_vector<OpportunityScore> d_scores;
    
    const int MAX_EXCHANGES = 100;
    const int MAX_TOKENS = 10000;
    const int THREADS_PER_BLOCK = 256;

public:
    GPUArbitrageAccelerator(int device = 0) : device_id(device) {
        cudaSetDevice(device_id);
        
        cublasCreate(&cublas_handle);
        curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(curand_gen, time(NULL));
        
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        available_memory = free_mem * 0.8;
        
        std::cout << "GPU Accelerator initialized on device " << device_id 
                  << " with " << available_memory / (1024*1024*1024) << "GB memory" << std::endl;
    }
    
    ~GPUArbitrageAccelerator() {
        cublasDestroy(cublas_handle);
        curandDestroyGenerator(curand_gen);
    }
    
    void initialize_price_matrix(int num_exchanges, int num_tokens) {
        size_t matrix_size = num_exchanges * num_tokens;
        
        d_price_matrix.resize(matrix_size);
        d_volume_matrix.resize(matrix_size);
        
        thrust::fill(d_price_matrix.begin(), d_price_matrix.end(), 0.0f);
        thrust::fill(d_volume_matrix.begin(), d_volume_matrix.end(), 0.0f);
    }
    
    void update_price_data(const std::vector<float>& prices, 
                          const std::vector<float>& volumes,
                          int exchange_id, int num_tokens) {
        if (prices.size() != num_tokens || volumes.size() != num_tokens) {
            std::cerr << "Invalid data size for exchange " << exchange_id << std::endl;
            return;
        }
        
        size_t offset = exchange_id * num_tokens;
        
        thrust::copy(prices.begin(), prices.end(), 
                    d_price_matrix.begin() + offset);
        thrust::copy(volumes.begin(), volumes.end(), 
                    d_volume_matrix.begin() + offset);
    }
    
    std::vector<ArbitrageData> find_cross_exchange_opportunities(
        int num_exchanges, int num_tokens, float min_spread = 0.001f) {
        
        std::vector<ArbitrageData> opportunities;
        
        thrust::host_vector<float> h_prices = d_price_matrix;
        thrust::host_vector<float> h_volumes = d_volume_matrix;
        
        for (int token = 0; token < num_tokens; ++token) {
            for (int ex1 = 0; ex1 < num_exchanges; ++ex1) {
                for (int ex2 = ex1 + 1; ex2 < num_exchanges; ++ex2) {
                    float price1 = h_prices[ex1 * num_tokens + token];
                    float price2 = h_prices[ex2 * num_tokens + token];
                    
                    if (price1 > 0 && price2 > 0) {
                        float spread = std::abs(price2 - price1) / std::min(price1, price2);
                        
                        if (spread > min_spread) {
                            ArbitrageData opp;
                            opp.buy_price = std::min(price1, price2);
                            opp.sell_price = std::max(price1, price2);
                            opp.spread = spread;
                            opp.volume = std::min(h_volumes[ex1 * num_tokens + token],
                                                h_volumes[ex2 * num_tokens + token]);
                            opp.exchange_id_buy = (price1 < price2) ? ex1 : ex2;
                            opp.exchange_id_sell = (price1 < price2) ? ex2 : ex1;
                            opp.token_id = token;
                            opp.confidence = calculate_confidence(spread, opp.volume);
                            
                            opportunities.push_back(opp);
                        }
                    }
                }
            }
        }
        
        return opportunities;
    }
    
    __device__ float calculate_triangular_profit(float rate1, float rate2, float rate3) {
        return rate1 * rate2 * rate3 - 1.0f;
    }
    
    std::vector<ArbitrageData> find_triangular_opportunities_gpu(
        const std::vector<std::vector<float>>& exchange_rates,
        int num_tokens, float min_profit = 0.002f) {
        
        std::vector<ArbitrageData> opportunities;
        
        for (size_t ex = 0; ex < exchange_rates.size(); ++ex) {
            const auto& rates = exchange_rates[ex];
            
            for (int base1 = 0; base1 < num_tokens; ++base1) {
                for (int base2 = base1 + 1; base2 < num_tokens; ++base2) {
                    for (int quote = 0; quote < num_tokens; ++quote) {
                        if (quote == base1 || quote == base2) continue;
                        
                        int idx1 = base1 * num_tokens + quote;
                        int idx2 = base1 * num_tokens + base2;
                        int idx3 = base2 * num_tokens + quote;
                        
                        if (idx1 < rates.size() && idx2 < rates.size() && idx3 < rates.size()) {
                            float rate1 = rates[idx1];
                            float rate2 = rates[idx2];
                            float rate3 = 1.0f / rates[idx3];
                            
                            if (rate1 > 0 && rate2 > 0 && rate3 > 0) {
                                float profit = calculate_triangular_profit(rate1, rate2, rate3);
                                
                                if (profit > min_profit) {
                                    ArbitrageData opp;
                                    opp.buy_price = rate1;
                                    opp.sell_price = rate1 * rate2 * rate3;
                                    opp.spread = profit * 100.0f;
                                    opp.volume = std::min({rate1, rate2, rate3}) * 10000.0f;
                                    opp.exchange_id_buy = ex;
                                    opp.exchange_id_sell = ex;
                                    opp.token_id = base1;
                                    opp.confidence = calculate_confidence(profit, opp.volume);
                                    
                                    opportunities.push_back(opp);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return opportunities;
    }
    
    __global__ void compute_opportunity_scores_kernel(
        ArbitrageData* data, OpportunityScore* scores, int n) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        
        ArbitrageData& opp = data[idx];
        
        float profit_factor = (opp.sell_price - opp.buy_price) / opp.buy_price;
        float volume_factor = logf(opp.volume + 1.0f) / 10.0f;
        float spread_factor = opp.spread / 100.0f;
        
        float score = profit_factor * 0.4f + volume_factor * 0.3f + 
                     spread_factor * 0.2f + opp.confidence * 0.1f;
        
        scores[idx].score = score;
        scores[idx].index = idx;
    }
    
    std::vector<int> rank_opportunities_gpu(std::vector<ArbitrageData>& opportunities) {
        if (opportunities.empty()) return {};
        
        d_arbitrage_data.resize(opportunities.size());
        d_scores.resize(opportunities.size());
        
        thrust::copy(opportunities.begin(), opportunities.end(), d_arbitrage_data.begin());
        
        int num_blocks = (opportunities.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        compute_opportunity_scores_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_arbitrage_data.data()),
            thrust::raw_pointer_cast(d_scores.data()),
            opportunities.size()
        );
        
        cudaDeviceSynchronize();
        
        thrust::sort(d_scores.begin(), d_scores.end(), 
                    [](const OpportunityScore& a, const OpportunityScore& b) {
                        return a.score > b.score;
                    });
        
        thrust::host_vector<OpportunityScore> h_scores = d_scores;
        
        std::vector<int> ranked_indices;
        for (const auto& score : h_scores) {
            ranked_indices.push_back(score.index);
        }
        
        return ranked_indices;
    }
    
    __global__ void parallel_price_analysis_kernel(
        float* price_matrix, float* volume_matrix, float* results,
        int num_exchanges, int num_tokens, int analysis_type) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = num_exchanges * num_tokens;
        
        if (idx >= total_elements) return;
        
        int exchange = idx / num_tokens;
        int token = idx % num_tokens;
        
        float price = price_matrix[idx];
        float volume = volume_matrix[idx];
        
        if (analysis_type == 0) { // Volatility analysis
            float sum = 0.0f;
            int count = 0;
            
            for (int other_ex = 0; other_ex < num_exchanges; ++other_ex) {
                if (other_ex != exchange) {
                    float other_price = price_matrix[other_ex * num_tokens + token];
                    if (other_price > 0) {
                        sum += fabsf(price - other_price) / price;
                        count++;
                    }
                }
            }
            
            results[idx] = (count > 0) ? sum / count : 0.0f;
            
        } else if (analysis_type == 1) { // Liquidity analysis
            results[idx] = volume * price;
            
        } else if (analysis_type == 2) { // Momentum analysis
            // Simple momentum based on price position relative to others
            float higher_count = 0.0f;
            float total_count = 0.0f;
            
            for (int other_ex = 0; other_ex < num_exchanges; ++other_ex) {
                float other_price = price_matrix[other_ex * num_tokens + token];
                if (other_price > 0) {
                    if (price > other_price) higher_count += 1.0f;
                    total_count += 1.0f;
                }
            }
            
            results[idx] = (total_count > 0) ? higher_count / total_count : 0.5f;
        }
    }
    
    std::vector<float> analyze_market_patterns(int num_exchanges, int num_tokens, int analysis_type) {
        thrust::device_vector<float> d_results(num_exchanges * num_tokens);
        
        int num_blocks = (num_exchanges * num_tokens + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        parallel_price_analysis_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_price_matrix.data()),
            thrust::raw_pointer_cast(d_volume_matrix.data()),
            thrust::raw_pointer_cast(d_results.data()),
            num_exchanges, num_tokens, analysis_type
        );
        
        cudaDeviceSynchronize();
        
        thrust::host_vector<float> h_results = d_results;
        return std::vector<float>(h_results.begin(), h_results.end());
    }
    
    void batch_process_opportunities(
        const std::vector<std::vector<float>>& exchange_data,
        const std::vector<std::vector<float>>& volume_data,
        std::vector<ArbitrageData>& all_opportunities) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int num_exchanges = exchange_data.size();
        int num_tokens = exchange_data.empty() ? 0 : exchange_data[0].size();
        
        initialize_price_matrix(num_exchanges, num_tokens);
        
        for (int ex = 0; ex < num_exchanges; ++ex) {
            update_price_data(exchange_data[ex], volume_data[ex], ex, num_tokens);
        }
        
        auto cross_opps = find_cross_exchange_opportunities(num_exchanges, num_tokens);
        auto triangular_opps = find_triangular_opportunities_gpu(exchange_data, num_tokens);
        
        all_opportunities.insert(all_opportunities.end(), cross_opps.begin(), cross_opps.end());
        all_opportunities.insert(all_opportunities.end(), triangular_opps.begin(), triangular_opps.end());
        
        auto ranked_indices = rank_opportunities_gpu(all_opportunities);
        
        std::vector<ArbitrageData> ranked_opportunities;
        for (int idx : ranked_indices) {
            if (idx < all_opportunities.size()) {
                ranked_opportunities.push_back(all_opportunities[idx]);
            }
        }
        
        all_opportunities = std::move(ranked_opportunities);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "GPU batch processing completed in " << duration.count() 
                  << "ms, found " << all_opportunities.size() << " opportunities" << std::endl;
    }
    
private:
    float calculate_confidence(float spread, float volume) {
        float spread_score = std::min(spread / 0.05f, 1.0f);  // Normalize to 5% max spread
        float volume_score = std::min(std::log(volume + 1.0f) / 15.0f, 1.0f);  // Log scale for volume
        
        return (spread_score * 0.6f + volume_score * 0.4f);
    }
};

extern "C" {
    GPUArbitrageAccelerator* create_accelerator(int device_id) {
        return new GPUArbitrageAccelerator(device_id);
    }
    
    void destroy_accelerator(GPUArbitrageAccelerator* accelerator) {
        delete accelerator;
    }
    
    void process_batch(GPUArbitrageAccelerator* accelerator,
                      float* exchange_data, float* volume_data,
                      int num_exchanges, int num_tokens,
                      ArbitrageData* results, int* num_results) {
        
        std::vector<std::vector<float>> exchanges(num_exchanges);
        std::vector<std::vector<float>> volumes(num_exchanges);
        
        for (int i = 0; i < num_exchanges; ++i) {
            exchanges[i].resize(num_tokens);
            volumes[i].resize(num_tokens);
            
            for (int j = 0; j < num_tokens; ++j) {
                exchanges[i][j] = exchange_data[i * num_tokens + j];
                volumes[i][j] = volume_data[i * num_tokens + j];
            }
        }
        
        std::vector<ArbitrageData> opportunities;
        accelerator->batch_process_opportunities(exchanges, volumes, opportunities);
        
        *num_results = std::min(static_cast<int>(opportunities.size()), 1000);
        for (int i = 0; i < *num_results; ++i) {
            results[i] = opportunities[i];
        }
    }
}