#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>
#include <future>
#include <Python.h>
#include <cuda_runtime.h>
#include \"gpu_kernels.cuh\"

struct MarketData {
    std::string exchange;
    std::string symbol;
    double bid;
    double ask;
    double volume;
    uint64_t timestamp;
    double last_price;
    double price_change_24h;
};

struct TradeExecution {
    std::string strategy;
    std::string symbol;
    std::string buy_exchange;
    std::string sell_exchange;
    double buy_amount;
    double sell_amount;
    double expected_profit;
    double actual_profit;
    uint64_t execution_time_us;
    bool success;
};

class UltraFastExecutionEngine {
private:
    std::unordered_map<std::string, MarketData> market_cache;
    std::mutex cache_mutex;
    std::atomic<uint64_t> total_executions{0};
    std::atomic<double> total_profit{0.0};
    std::queue<TradeExecution> execution_queue;
    std::mutex queue_mutex;
    
    // GPU memory pointers
    float *d_prices;
    float *d_volumes;
    float *d_spreads;
    int *d_indices;
    int max_symbols;
    
    // Exchange connection pools
    std::unordered_map<std::string, std::vector<std::string>> exchange_endpoints;
    std::unordered_map<std::string, double> exchange_fees;
    
public:
    UltraFastExecutionEngine(int max_symbols = 10000) : max_symbols(max_symbols) {
        initialize_gpu_memory();
        initialize_exchange_configs();
    }
    
    ~UltraFastExecutionEngine() {
        cleanup_gpu_memory();
    }
    
    void initialize_gpu_memory() {
        size_t size = max_symbols * sizeof(float);
        cudaMalloc(&d_prices, size * 4); // bid, ask, volume, timestamp
        cudaMalloc(&d_volumes, size);
        cudaMalloc(&d_spreads, size);
        cudaMalloc(&d_indices, max_symbols * sizeof(int));
        
        if (!d_prices || !d_volumes || !d_spreads || !d_indices) {
            throw std::runtime_error(\"Failed to allocate GPU memory\");
        }
    }
    
    void cleanup_gpu_memory() {
        if (d_prices) cudaFree(d_prices);
        if (d_volumes) cudaFree(d_volumes);
        if (d_spreads) cudaFree(d_spreads);
        if (d_indices) cudaFree(d_indices);
    }
    
    void initialize_exchange_configs() {
        // CEX endpoints
        exchange_endpoints[\"binance\"] = {
            \"https://api.binance.com/api/v3/ticker/bookTicker\",
            \"wss://stream.binance.com:9443/ws/!ticker@arr\"
        };
        exchange_endpoints[\"coinbase\"] = {
            \"https://api.exchange.coinbase.com/products\",
            \"wss://ws-feed.exchange.coinbase.com\"
        };
        exchange_endpoints[\"okx\"] = {
            \"https://www.okx.com/api/v5/market/tickers\",
            \"wss://ws.okx.com:8443/ws/v5/public\"
        };
        exchange_endpoints[\"bybit\"] = {
            \"https://api.bybit.com/v5/market/tickers\",
            \"wss://stream.bybit.com/v5/public/spot\"
        };
        exchange_endpoints[\"huobi\"] = {
            \"https://api.huobi.pro/market/tickers\",
            \"wss://api.huobi.pro/ws\"
        };
        exchange_endpoints[\"kucoin\"] = {
            \"https://api.kucoin.com/api/v1/market/allTickers\",
            \"wss://ws-api.kucoin.com/endpoint\"
        };
        exchange_endpoints[\"gate\"] = {
            \"https://api.gateio.ws/api/v4/spot/tickers\",
            \"wss://api.gateio.ws/ws/v4/\"
        };
        exchange_endpoints[\"mexc\"] = {
            \"https://api.mexc.com/api/v3/ticker/24hr\",
            \"wss://wbs.mexc.com/ws\"
        };
        
        // DEX endpoints
        exchange_endpoints[\"uniswap_v3\"] = {
            \"https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3\",
            \"wss://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3\"
        };
        exchange_endpoints[\"sushiswap\"] = {
            \"https://api.thegraph.com/subgraphs/name/sushiswap/exchange\",
            \"wss://api.thegraph.com/subgraphs/name/sushiswap/exchange\"
        };
        exchange_endpoints[\"curve\"] = {
            \"https://api.curve.fi/api/getPools/all\",
            \"wss://curve.fi/ws\"
        };
        exchange_endpoints[\"balancer\"] = {
            \"https://api.thegraph.com/subgraphs/name/balancer-labs/balancer-v2\",
            \"wss://api.thegraph.com/subgraphs/name/balancer-labs/balancer-v2\"
        };
        
        // Exchange fees
        exchange_fees[\"binance\"] = 0.001;
        exchange_fees[\"coinbase\"] = 0.005;
        exchange_fees[\"okx\"] = 0.001;
        exchange_fees[\"bybit\"] = 0.001;
        exchange_fees[\"huobi\"] = 0.002;
        exchange_fees[\"kucoin\"] = 0.001;
        exchange_fees[\"gate\"] = 0.002;
        exchange_fees[\"mexc\"] = 0.002;
        exchange_fees[\"uniswap_v3\"] = 0.003;
        exchange_fees[\"sushiswap\"] = 0.003;
        exchange_fees[\"curve\"] = 0.0004;
        exchange_fees[\"balancer\"] = 0.0005;
    }
    
    void update_market_data(const std::string& exchange, const std::string& symbol, 
                           double bid, double ask, double volume) {
        auto now = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        std::lock_guard<std::mutex> lock(cache_mutex);
        std::string key = exchange + \":\" + symbol;
        
        market_cache[key] = {
            exchange, symbol, bid, ask, volume, 
            static_cast<uint64_t>(now), (bid + ask) / 2.0, 0.0
        };
    }
    
    std::vector<std::string> scan_cross_exchange_opportunities(double min_profit = 1.0) {
        std::vector<std::string> opportunities;
        std::unordered_map<std::string, std::vector<MarketData*>> symbol_map;
        
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            for (auto& [key, data] : market_cache) {
                symbol_map[data.symbol].push_back(&data);
            }
        }
        
        // Prepare GPU data
        std::vector<float> h_bids, h_asks, h_volumes;
        std::vector<std::string> symbol_keys;
        
        for (auto& [symbol, data_list] : symbol_map) {
            if (data_list.size() >= 2) {
                for (auto* data : data_list) {
                    h_bids.push_back(data->bid);
                    h_asks.push_back(data->ask);
                    h_volumes.push_back(data->volume);
                    symbol_keys.push_back(data->exchange + \":\" + symbol);
                }
            }
        }
        
        if (h_bids.size() < 2) return opportunities;
        
        // Copy to GPU
        int n = h_bids.size();
        cudaMemcpy(d_prices, h_bids.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_prices + n, h_asks.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_volumes, h_volumes.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
        // Launch GPU kernel for parallel arbitrage detection
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        
        find_arbitrage_opportunities_kernel<<<grid, block>>>(
            d_prices, d_prices + n, d_volumes, d_spreads, d_indices, n, min_profit
        );
        
        cudaDeviceSynchronize();
        
        // Copy results back
        std::vector<float> h_spreads(n);
        std::vector<int> h_indices(n);
        cudaMemcpy(h_spreads.data(), d_spreads, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_indices.data(), d_indices, n * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Process results
        for (int i = 0; i < n; i++) {
            if (h_spreads[i] > 0.002 && h_indices[i] >= 0) {
                int j = h_indices[i];
                if (j < symbol_keys.size() && i < symbol_keys.size()) {
                    double trade_size = 10000.0;
                    double gross_profit = trade_size * h_spreads[i];
                    
                    // Extract exchange names from keys
                    std::string buy_exchange = symbol_keys[i].substr(0, symbol_keys[i].find(':'));
                    std::string sell_exchange = symbol_keys[j].substr(0, symbol_keys[j].find(':'));
                    std::string symbol = symbol_keys[i].substr(symbol_keys[i].find(':') + 1);
                    
                    double buy_fee = trade_size * exchange_fees[buy_exchange];
                    double sell_fee = trade_size * exchange_fees[sell_exchange];
                    double net_profit = gross_profit - buy_fee - sell_fee;
                    
                    if (net_profit > min_profit) {
                        char buffer[1024];
                        snprintf(buffer, sizeof(buffer), 
                            \"{\\\"strategy\\\":\\\"cross_exchange\\\",\\\"symbol\\\":\\\"%s\\\",\"\
                            \"\\\"buy_exchange\\\":\\\"%s\\\",\\\"sell_exchange\\\":\\\"%s\\\",\"\
                            \"\\\"spread\\\":%.4f,\\\"net_profit\\\":%.2f}\",
                            symbol.c_str(), buy_exchange.c_str(), sell_exchange.c_str(),
                            h_spreads[i] * 100.0, net_profit);
                        opportunities.push_back(std::string(buffer));
                    }
                }
            }
        }
        
        return opportunities;
    }
    
    std::vector<std::string> scan_triangular_opportunities(const std::string& exchange, double min_profit = 1.0) {
        std::vector<std::string> opportunities;
        std::unordered_map<std::string, MarketData*> exchange_data;
        
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            for (auto& [key, data] : market_cache) {
                if (data.exchange == exchange) {
                    exchange_data[data.symbol] = &data;
                }
            }
        }
        
        // Define triangular paths
        std::vector<std::tuple<std::string, std::string, std::string>> triangular_paths = {
            {\"BTCUSDT\", \"ETHBTC\", \"ETHUSDT\"},
            {\"BNBUSDT\", \"ETHBNB\", \"ETHUSDT\"},
            {\"ADAUSDT\", \"ETHADA\", \"ETHUSDT\"},
            {\"DOTUSDT\", \"ETHDOT\", \"ETHUSDT\"},
            {\"LINKUSDT\", \"ETHLINK\", \"ETHUSDT\"},
            {\"LTCUSDT\", \"ETHLTC\", \"ETHUSDT\"},
            {\"XRPUSDT\", \"ETHXRP\", \"ETHUSDT\"},
            {\"SOLUSDT\", \"ETHSOL\", \"ETHUSDT\"}
        };
        
        for (auto& [pair1, pair2, pair3] : triangular_paths) {
            auto it1 = exchange_data.find(pair1);
            auto it2 = exchange_data.find(pair2);
            auto it3 = exchange_data.find(pair3);
            
            if (it1 != exchange_data.end() && it2 != exchange_data.end() && it3 != exchange_data.end()) {
                MarketData* p1 = it1->second;
                MarketData* p2 = it2->second;
                MarketData* p3 = it3->second;
                
                double rate1 = p1->bid;
                double rate2 = p2->bid;
                double rate3 = 1.0 / p3->ask;
                
                double final_rate = rate1 * rate2 * rate3;
                double profit_rate = final_rate - 1.0;
                
                if (profit_rate > 0.001) {
                    double trade_size = 10000.0;
                    double gross_profit = trade_size * profit_rate;
                    double total_fees = trade_size * 0.003; // 3 trades * 0.1% each
                    double net_profit = gross_profit - total_fees;
                    
                    if (net_profit > min_profit) {
                        char buffer[1024];
                        snprintf(buffer, sizeof(buffer),
                            \"{\\\"strategy\\\":\\\"triangular\\\",\\\"symbol\\\":\\\"%s-%s-%s\\\",\"\
                            \"\\\"exchange\\\":\\\"%s\\\",\\\"profit_rate\\\":%.4f,\\\"net_profit\\\":%.2f}\",
                            pair1.c_str(), pair2.c_str(), pair3.c_str(), 
                            exchange.c_str(), profit_rate * 100.0, net_profit);
                        opportunities.push_back(std::string(buffer));
                    }
                }
            }
        }
        
        return opportunities;
    }
    
    bool execute_trade(const std::string& strategy, const std::string& symbol,
                      const std::string& buy_exchange, const std::string& sell_exchange,
                      double amount, double expected_profit) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Simulate ultra-fast trade execution
        std::this_thread::sleep_for(std::chrono::microseconds(100)); // 100 microsecond execution
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();
        
        // Simulate execution success (95% success rate)
        bool success = (rand() % 100) < 95;
        double actual_profit = success ? expected_profit * (0.95 + (rand() % 10) * 0.01) : 0.0;
        
        TradeExecution execution = {
            strategy, symbol, buy_exchange, sell_exchange,
            amount, amount, expected_profit, actual_profit,
            static_cast<uint64_t>(execution_time), success
        };
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            execution_queue.push(execution);
        }
        
        if (success) {
            total_executions++;
            total_profit += actual_profit;
        }
        
        return success;
    }
    
    std::vector<std::string> get_execution_history(int limit = 100) {
        std::vector<std::string> history;
        std::lock_guard<std::mutex> lock(queue_mutex);
        
        std::queue<TradeExecution> temp_queue = execution_queue;
        int count = 0;
        
        while (!temp_queue.empty() && count < limit) {
            TradeExecution& exec = temp_queue.front();
            temp_queue.pop();
            
            char buffer[1024];
            snprintf(buffer, sizeof(buffer),
                \"{\\\"strategy\\\":\\\"%s\\\",\\\"symbol\\\":\\\"%s\\\",\"\
                \"\\\"buy_exchange\\\":\\\"%s\\\",\\\"sell_exchange\\\":\\\"%s\\\",\"\
                \"\\\"expected_profit\\\":%.2f,\\\"actual_profit\\\":%.2f,\"\
                \"\\\"execution_time_us\\\":%lu,\\\"success\\\":%s}\",
                exec.strategy.c_str(), exec.symbol.c_str(),
                exec.buy_exchange.c_str(), exec.sell_exchange.c_str(),
                exec.expected_profit, exec.actual_profit,
                exec.execution_time_us, exec.success ? \"true\" : \"false\");
            
            history.push_back(std::string(buffer));
            count++;
        }
        
        return history;
    }
    
    std::string get_performance_stats() {
        char buffer[512];
        snprintf(buffer, sizeof(buffer),
            \"{\\\"total_executions\\\":%lu,\\\"total_profit\\\":%.2f,\"\
            \"\\\"cache_size\\\":%zu,\\\"queue_size\\\":%zu}\",
            total_executions.load(), total_profit.load(),
            market_cache.size(), execution_queue.size());
        return std::string(buffer);
    }
};

// Python interface
static UltraFastExecutionEngine* engine = nullptr;

extern \"C\" {
    void init_execution_engine(int max_symbols = 10000) {
        if (!engine) {
            engine = new UltraFastExecutionEngine(max_symbols);
        }
    }
    
    void update_market_data(const char* exchange, const char* symbol, 
                           double bid, double ask, double volume) {
        if (engine) {
            engine->update_market_data(std::string(exchange), std::string(symbol), bid, ask, volume);
        }
    }
    
    char* scan_opportunities(double min_profit) {
        if (!engine) return nullptr;
        
        auto cross_opportunities = engine->scan_cross_exchange_opportunities(min_profit);
        auto triangular_binance = engine->scan_triangular_opportunities(\"binance\", min_profit);
        auto triangular_okx = engine->scan_triangular_opportunities(\"okx\", min_profit);
        
        std::string result = \"[\";
        bool first = true;
        
        for (const auto& opp : cross_opportunities) {
            if (!first) result += \",\";
            result += opp;
            first = false;
        }
        for (const auto& opp : triangular_binance) {
            if (!first) result += \",\";
            result += opp;
            first = false;
        }
        for (const auto& opp : triangular_okx) {
            if (!first) result += \",\";
            result += opp;
            first = false;
        }
        
        result += \"]\";
        
        char* c_result = new char[result.length() + 1];
        strcpy(c_result, result.c_str());
        return c_result;
    }
    
    bool execute_trade(const char* strategy, const char* symbol,
                      const char* buy_exchange, const char* sell_exchange,
                      double amount, double expected_profit) {
        if (!engine) return false;
        return engine->execute_trade(std::string(strategy), std::string(symbol),
                                   std::string(buy_exchange), std::string(sell_exchange),
                                   amount, expected_profit);
    }
    
    char* get_performance_stats() {
        if (!engine) return nullptr;
        std::string stats = engine->get_performance_stats();
        char* c_stats = new char[stats.length() + 1];
        strcpy(c_stats, stats.c_str());
        return c_stats;
    }
    
    void cleanup_execution_engine() {
        if (engine) {
            delete engine;
            engine = nullptr;
        }
    }
}