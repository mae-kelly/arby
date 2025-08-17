#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <thread>
#include <atomic>
#include <chrono>
#include <immintrin.h>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_queue.h>
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <rapidjson/document.h>
#include <librdkafka/rdkafkacpp.h>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;

constexpr size_t MAX_ORDERBOOK_DEPTH = 100;
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr double MIN_SPREAD = 0.0001;

struct alignas(CACHE_LINE_SIZE) OrderLevel {
    double price;
    double quantity;
    uint64_t timestamp;
    
    OrderLevel() : price(0), quantity(0), timestamp(0) {}
    OrderLevel(double p, double q, uint64_t t) : price(p), quantity(q), timestamp(t) {}
};

struct alignas(CACHE_LINE_SIZE) OrderBook {
    std::vector<OrderLevel> bids;
    std::vector<OrderLevel> asks;
    std::atomic<uint64_t> last_update;
    std::string symbol;
    std::string exchange;
    
    OrderBook() : last_update(0) {
        bids.reserve(MAX_ORDERBOOK_DEPTH);
        asks.reserve(MAX_ORDERBOOK_DEPTH);
    }
    
    double get_spread() const {
        if (asks.empty() || bids.empty()) return 0.0;
        return asks[0].price - bids[0].price;
    }
    
    double get_mid_price() const {
        if (asks.empty() || bids.empty()) return 0.0;
        return (asks[0].price + bids[0].price) / 2.0;
    }
};

class OrderBookScanner {
private:
    tbb::concurrent_hash_map<std::string, OrderBook> orderbooks;
    tbb::concurrent_queue<std::pair<std::string, std::string>> arbitrage_opportunities;
    std::vector<std::thread> worker_threads;
    std::atomic<bool> running{true};
    net::io_context ioc;
    
    struct ExchangeConnection {
        std::shared_ptr<websocket::stream<tcp::socket>> ws;
        std::string exchange_name;
        std::vector<std::string> symbols;
    };
    
    std::vector<ExchangeConnection> connections;
    
public:
    OrderBookScanner() {
        init_connections();
        start_workers();
    }
    
    ~OrderBookScanner() {
        running = false;
        for (auto& t : worker_threads) {
            if (t.joinable()) t.join();
        }
    }
    
    void init_connections() {
        // Initialize WebSocket connections to exchanges
        std::vector<std::pair<std::string, std::string>> exchanges = {
            {"binance", "wss://stream.binance.com:9443/ws"},
            {"coinbase", "wss://ws-feed.exchange.coinbase.com"},
            {"kraken", "wss://ws.kraken.com"},
            {"bybit", "wss://stream.bybit.com/realtime"},
            {"okx", "wss://ws.okx.com:8443/ws/v5/public"}
        };
        
        for (const auto& [name, url] : exchanges) {
            connect_exchange(name, url);
        }
    }
    
    void connect_exchange(const std::string& name, const std::string& url) {
        std::thread([this, name, url]() {
            try {
                tcp::resolver resolver{ioc};
                auto ws = std::make_shared<websocket::stream<tcp::socket>>(ioc);
                
                auto const results = resolver.resolve("stream.binance.com", "9443");
                auto ep = net::connect(ws->next_layer(), results);
                
                ws->handshake("stream.binance.com", "/ws");
                
                ExchangeConnection conn;
                conn.ws = ws;
                conn.exchange_name = name;
                connections.push_back(conn);
                
                beast::flat_buffer buffer;
                while (running) {
                    ws->read(buffer);
                    process_message(name, beast::buffers_to_string(buffer.data()));
                    buffer.clear();
                }
            } catch (std::exception const& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
        }).detach();
    }
    
    void process_message(const std::string& exchange, const std::string& message) {
        rapidjson::Document doc;
        doc.Parse(message.c_str());
        
        if (doc.HasParseError()) return;
        
        if (doc.HasMember("symbol") && doc.HasMember("bids") && doc.HasMember("asks")) {
            std::string symbol = doc["symbol"].GetString();
            update_orderbook(exchange, symbol, doc);
        }
    }
    
    void update_orderbook(const std::string& exchange, const std::string& symbol, 
                         const rapidjson::Document& doc) {
        std::string key = exchange + ":" + symbol;
        
        tbb::concurrent_hash_map<std::string, OrderBook>::accessor acc;
        orderbooks.insert(acc, key);
        
        OrderBook& ob = acc->second;
        ob.symbol = symbol;
        ob.exchange = exchange;
        
        ob.bids.clear();
        ob.asks.clear();
        
        auto now = std::chrono::system_clock::now().time_since_epoch().count();
        
        if (doc["bids"].IsArray()) {
            for (const auto& bid : doc["bids"].GetArray()) {
                if (bid.IsArray() && bid.Size() >= 2) {
                    double price = std::stod(bid[0].GetString());
                    double qty = std::stod(bid[1].GetString());
                    ob.bids.emplace_back(price, qty, now);
                }
                if (ob.bids.size() >= MAX_ORDERBOOK_DEPTH) break;
            }
        }
        
        if (doc["asks"].IsArray()) {
            for (const auto& ask : doc["asks"].GetArray()) {
                if (ask.IsArray() && ask.Size() >= 2) {
                    double price = std::stod(ask[0].GetString());
                    double qty = std::stod(ask[1].GetString());
                    ob.asks.emplace_back(price, qty, now);
                }
                if (ob.asks.size() >= MAX_ORDERBOOK_DEPTH) break;
            }
        }
        
        ob.last_update = now;
        check_arbitrage(key, ob);
    }
    
    void check_arbitrage(const std::string& key, const OrderBook& ob) {
        __m256d min_spread = _mm256_set1_pd(MIN_SPREAD);
        
        tbb::parallel_for(orderbooks.range(), [&](const auto& range) {
            for (auto it = range.begin(); it != range.end(); ++it) {
                if (it->first == key) continue;
                
                const OrderBook& other = it->second;
                if (ob.symbol != other.symbol) continue;
                
                if (ob.asks.empty() || ob.bids.empty() || 
                    other.asks.empty() || other.bids.empty()) continue;
                
                double spread1 = ob.asks[0].price - other.bids[0].price;
                double spread2 = other.asks[0].price - ob.bids[0].price;
                
                __m256d v_spread1 = _mm256_set1_pd(spread1);
                __m256d v_spread2 = _mm256_set1_pd(spread2);
                
                __m256d cmp1 = _mm256_cmp_pd(v_spread1, min_spread, _CMP_LT_OQ);
                __m256d cmp2 = _mm256_cmp_pd(v_spread2, min_spread, _CMP_LT_OQ);
                
                if (_mm256_movemask_pd(cmp1) || _mm256_movemask_pd(cmp2)) {
                    arbitrage_opportunities.push({key, it->first});
                }
            }
        });
    }
    
    void start_workers() {
        const size_t num_workers = std::thread::hardware_concurrency();
        
        for (size_t i = 0; i < num_workers; ++i) {
            worker_threads.emplace_back([this]() {
                process_arbitrage();
            });
        }
    }
    
    void process_arbitrage() {
        while (running) {
            std::pair<std::string, std::string> opp;
            if (arbitrage_opportunities.try_pop(opp)) {
                execute_arbitrage(opp.first, opp.second);
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    }
    
    void execute_arbitrage(const std::string& book1, const std::string& book2) {
        tbb::concurrent_hash_map<std::string, OrderBook>::const_accessor acc1, acc2;
        
        if (!orderbooks.find(acc1, book1) || !orderbooks.find(acc2, book2)) {
            return;
        }
        
        const OrderBook& ob1 = acc1->second;
        const OrderBook& ob2 = acc2->second;
        
        double profit = calculate_profit(ob1, ob2);
        
        if (profit > 0) {
            std::cout << "Arbitrage opportunity: " << book1 << " <-> " << book2 
                     << " Profit: " << profit << std::endl;
            
            send_to_executor(ob1, ob2, profit);
        }
    }
    
    double calculate_profit(const OrderBook& ob1, const OrderBook& ob2) {
        if (ob1.asks.empty() || ob1.bids.empty() || 
            ob2.asks.empty() || ob2.bids.empty()) {
            return 0.0;
        }
        
        double buy_price = ob1.asks[0].price;
        double sell_price = ob2.bids[0].price;
        double quantity = std::min(ob1.asks[0].quantity, ob2.bids[0].quantity);
        
        double gross_profit = (sell_price - buy_price) * quantity;
        double fees = (buy_price * 0.001 + sell_price * 0.001) * quantity;
        
        return gross_profit - fees;
    }
    
    void send_to_executor(const OrderBook& ob1, const OrderBook& ob2, double profit) {
        // Send to execution engine via IPC or shared memory
        std::stringstream ss;
        ss << "{\"type\":\"arbitrage\",\"exchange1\":\"" << ob1.exchange 
           << "\",\"exchange2\":\"" << ob2.exchange 
           << "\",\"symbol\":\"" << ob1.symbol 
           << "\",\"profit\":" << profit << "}";
        
        // In real implementation, send via IPC
        std::cout << ss.str() << std::endl;
    }
    
    void run() {
        ioc.run();
    }
};

extern "C" {
    OrderBookScanner* create_scanner() {
        return new OrderBookScanner();
    }
    
    void destroy_scanner(OrderBookScanner* scanner) {
        delete scanner;
    }
    
    void run_scanner(OrderBookScanner* scanner) {
        scanner->run();
    }
}