#include <iostream>
#include <map>
#include <string>

extern "C" {
    void* create_scanner() {
        return new std::map<std::string, double>();
    }
    
    void destroy_scanner(void* scanner) {
        delete static_cast<std::map<std::string, double>*>(scanner);
    }
    
    double check_spread(void* scanner) {
        auto* prices = static_cast<std::map<std::string, double>*>(scanner);
        if (prices->size() < 2) return 0.0;
        
        double min = 1e9, max = 0;
        for (const auto& [exchange, price] : *prices) {
            if (price < min) min = price;
            if (price > max) max = price;
        }
        
        return max > min ? (max - min) / min * 100.0 : 0.0;
    }
}

int main() {
    std::cout << "Arbitrage Scanner Ready" << std::endl;
    return 0;
}
