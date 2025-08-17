#include <map>
#include <vector>
#include <mutex>
#include <cstring>

extern "C" {
    struct OrderBook {
        std::map<double, double> bids;
        std::map<double, double> asks;
        std::mutex mutex;
    };
    
    void* create_orderbook() {
        return new OrderBook();
    }
    
    void update_orderbook(void* ob, double price, double volume, int is_bid) {
        OrderBook* book = static_cast<OrderBook*>(ob);
        std::lock_guard<std::mutex> lock(book->mutex);
        
        if (is_bid) {
            book->bids[price] = volume;
        } else {
            book->asks[price] = volume;
        }
    }
    
    double get_best_bid(void* ob) {
        OrderBook* book = static_cast<OrderBook*>(ob);
        std::lock_guard<std::mutex> lock(book->mutex);
        return book->bids.empty() ? 0.0 : book->bids.rbegin()->first;
    }
    
    double get_best_ask(void* ob) {
        OrderBook* book = static_cast<OrderBook*>(ob);
        std::lock_guard<std::mutex> lock(book->mutex);
        return book->asks.empty() ? 0.0 : book->asks.begin()->first;
    }
    
    void destroy_orderbook(void* ob) {
        delete static_cast<OrderBook*>(ob);
    }
}
