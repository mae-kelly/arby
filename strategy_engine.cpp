# ===== C++ - Mathematical Strategy Engine =====
# strategy_engine.cpp

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>
#include <curl/curl.h>
#include <json/json.h>
#include <openssl/hmac.h>
#include <openssl/sha.h>

class StrategyEngine {
private:
    std::string apiKey;
    std::string secretKey;
    std::string passphrase;
    std::string baseUrl;
    std::string webhookUrl;
    std::map<std::string, std::vector<double>> priceHistory;
    std::map<std::string, double> positions;

    struct WriteCallback {
        std::string data;
        static size_t WriteData(void* contents, size_t size, size_t nmemb, WriteCallback* userp) {
            userp->data.append((char*)contents, size * nmemb);
            return size * nmemb;
        }
    };

    std::string sign(const std::string& timestamp, const std::string& method, 
                    const std::string& requestPath, const std::string& body) {
        std::string message = timestamp + method + requestPath + body;
        
        unsigned char* digest = HMAC(EVP_sha256(), secretKey.c_str(), secretKey.length(),
                                    (unsigned char*)message.c_str(), message.length(), NULL, NULL);
        
        std::string signature;
        char buf[3];
        for (int i = 0; i < 32; i++) {
            sprintf(buf, "%02x", digest[i]);
            signature += buf;
        }
        
        return signature;
    }

public:
    StrategyEngine() : 
        apiKey("8a760df1-4a2d-471b-ba42-d16893614dab"),
        secretKey("C9F3FC89A6A30226E11DFFD098C7CF3D"),
        passphrase("trading_bot_2024"),
        baseUrl("https://www.okx.com"),
        webhookUrl("https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3") {}

    double calculateSMA(const std::vector<double>& prices, int period) {
        if (prices.size() < period) return 0.0;
        
        double sum = std::accumulate(prices.end() - period, prices.end(), 0.0);
        return sum / period;
    }

    double calculateEMA(const std::vector<double>& prices, int period) {
        if (prices.empty()) return 0.0;
        if (prices.size() == 1) return prices[0];
        
        double multiplier = 2.0 / (period + 1);
        double ema = prices[0];
        
        for (size_t i = 1; i < prices.size(); i++) {
            ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
        }
        
        return ema;
    }

    double calculateRSI(const std::vector<double>& prices, int period = 14) {
        if (prices.size() < period + 1) return 50.0;
        
        std::vector<double> gains, losses;
        
        for (size_t i = 1; i < prices.size(); i++) {
            double change = prices[i] - prices[i-1];
            gains.push_back(change > 0 ? change : 0);
            losses.push_back(change < 0 ? -change : 0);
        }
        
        double avgGain = std::accumulate(gains.end() - period, gains.end(), 0.0) / period;
        double avgLoss = std::accumulate(losses.end() - period, losses.end(), 0.0) / period;
        
        if (avgLoss == 0) return 100.0;
        
        double rs = avgGain / avgLoss;
        return 100.0 - (100.0 / (1.0 + rs));
    }

    std::pair<double, double> calculateBollingerBands(const std::vector<double>& prices, int period = 20, double multiplier = 2.0) {
        if (prices.size() < period) return {0.0, 0.0};
        
        double sma = calculateSMA(prices, period);
        
        double variance = 0.0;
        for (int i = prices.size() - period; i < prices.size(); i++) {
            variance += std::pow(prices[i] - sma, 2);
        }
        variance /= period;
        
        double stdDev = std::sqrt(variance);
        
        return {sma - (multiplier * stdDev), sma + (multiplier * stdDev)};
    }

    std::string generateSignal(const std::string& symbol) {
        auto& prices = priceHistory[symbol];
        if (prices.size() < 50) return "HOLD";
        
        double currentPrice = prices.back();
        double sma20 = calculateSMA(prices, 20);
        double sma50 = calculateSMA(prices, 50);
        double ema12 = calculateEMA(prices, 12);
        double ema26 = calculateEMA(prices, 26);
        double rsi = calculateRSI(prices);
        
        auto bollinger = calculateBollingerBands(prices);
        double lowerBand = bollinger.first;
        double upperBand = bollinger.second;
        
        double macdLine = ema12 - ema26;
        
        int bullishSignals = 0;
        int bearishSignals = 0;
        
        if (sma20 > sma50) bullishSignals++;
        else bearishSignals++;
        
        if (currentPrice > sma20) bullishSignals++;
        else bearishSignals++;
        
        if (rsi < 30) bullishSignals += 2;
        else if (rsi > 70) bearishSignals += 2;
        
        if (currentPrice < lowerBand) bullishSignals += 2;
        else if (currentPrice > upperBand) bearishSignals += 2;
        
        if (macdLine > 0) bullishSignals++;
        else bearishSignals++;
        
        if (bullishSignals > bearishSignals + 1) return "BUY";
        if (bearishSignals > bullishSignals + 1) return "SELL";
        
        return "HOLD";
    }

    void sendDiscordAlert(const std::string& message) {
        CURL* curl = curl_easy_init();
        if (curl) {
            Json::Value payload;
            payload["content"] = "🧮 C++ Strategy Engine: " + message;
            
            Json::StreamWriterBuilder builder;
            std::string jsonString = Json::writeString(builder, payload);
            
            curl_easy_setopt(curl, CURLOPT_URL, webhookUrl.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonString.c_str());
            
            struct curl_slist* headers = NULL;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            
            curl_easy_perform(curl);
            curl_easy_cleanup(curl);
            curl_slist_free_all(headers);
        }
    }

    void updatePriceHistory(const std::string& symbol, double price) {
        priceHistory[symbol].push_back(price);
        
        if (priceHistory[symbol].size() > 200) {
            priceHistory[symbol].erase(priceHistory[symbol].begin());
        }
    }

    void runStrategy() {
        sendDiscordAlert("Mathematical Strategy Engine started");
        
        std::vector<std::string> symbols = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT"};
        
        while (true) {
            for (const auto& symbol : symbols) {
                double mockPrice = 50000 + (rand() % 1000 - 500);
                updatePriceHistory(symbol, mockPrice);
                
                std::string signal = generateSignal(symbol);
                
                if (signal != "HOLD") {
                    std::string message = "Signal generated for " + symbol + ": " + signal + 
                                        " (Price: " + std::to_string(mockPrice) + ")";
                    sendDiscordAlert(message);
                    
                    std::cout << message << std::endl;
                }
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }
};

int main() {
    StrategyEngine engine;
    engine.runStrategy();
    return 0;
}