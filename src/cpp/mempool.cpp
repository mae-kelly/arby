#include <queue>
#include <mutex>
#include <cstring>

extern "C" {
    struct Transaction {
        char hash[66];
        double gas_price;
        double value;
    };
    
    struct Mempool {
        std::priority_queue<Transaction> pending;
        std::mutex mutex;
    };
    
    void* create_mempool() {
        return new Mempool();
    }
    
    void add_transaction(void* mp, const char* hash, double gas_price, double value) {
        Mempool* mempool = static_cast<Mempool*>(mp);
        std::lock_guard<std::mutex> lock(mempool->mutex);
        
        Transaction tx;
        strncpy(tx.hash, hash, 65);
        tx.gas_price = gas_price;
        tx.value = value;
        
        mempool->pending.push(tx);
    }
    
    int get_mempool_size(void* mp) {
        Mempool* mempool = static_cast<Mempool*>(mp);
        std::lock_guard<std::mutex> lock(mempool->mutex);
        return mempool->pending.size();
    }
    
    void destroy_mempool(void* mp) {
        delete static_cast<Mempool*>(mp);
    }
}
