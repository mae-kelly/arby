# main_orchestrator.py

import asyncio
import aiohttp
import json
import time
import hmac
import hashlib
import base64
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import websocket
import threading
import requests

@dataclass
class ArbitrageOpportunity:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_percentage: float
    volume: float

class OKXClient:
    def __init__(self):
        self.api_key = "8a760df1-4a2d-471b-ba42-d16893614dab"
        self.secret_key = "C9F3FC89A6A30226E11DFFD098C7CF3D"
        self.passphrase = "trading_bot_2024"
        self.base_url = "https://www.okx.com"
        
    def sign(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()
    
    async def get_ticker(self, symbol: str) -> Dict:
        timestamp = str(time.time())
        method = "GET"
        request_path = f"/api/v5/market/ticker?instId={symbol}"
        
        headers = {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': self.sign(timestamp, method, request_path),
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}{request_path}", headers=headers) as response:
                return await response.json()

    async def place_order(self, symbol: str, side: str, amount: str, price: str) -> Dict:
        timestamp = str(time.time())
        method = "POST"
        request_path = "/api/v5/trade/order"
        
        body = json.dumps({
            "instId": symbol,
            "tdMode": "cash",
            "side": side,
            "ordType": "limit",
            "sz": amount,
            "px": price
        })
        
        headers = {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': self.sign(timestamp, method, request_path, body),
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}{request_path}", headers=headers, data=body) as response:
                return await response.json()

class NotificationService:
    def __init__(self):
        self.discord_webhook = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3"
    
    def send_alert(self, message: str):
        payload = {"content": f"🤖 Trading Bot Alert: {message}"}
        requests.post(self.discord_webhook, json=payload)

class ArbitrageBot:
    def __init__(self):
        self.okx_client = OKXClient()
        self.notification_service = NotificationService()
        self.opportunities = []
        self.min_profit_threshold = 0.5
        self.running = False
        
    def calculate_arbitrage(self, prices: Dict) -> List[ArbitrageOpportunity]:
        opportunities = []
        symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT"]
        
        for symbol in symbols:
            if symbol in prices:
                buy_price = float(prices[symbol]["bid"])
                sell_price = float(prices[symbol]["ask"])
                spread = ((sell_price - buy_price) / buy_price) * 100
                
                if spread > self.min_profit_threshold:
                    opportunity = ArbitrageOpportunity(
                        symbol=symbol,
                        buy_exchange="OKX",
                        sell_exchange="OKX",
                        buy_price=buy_price,
                        sell_price=sell_price,
                        profit_percentage=spread,
                        volume=1000
                    )
                    opportunities.append(opportunity)
        
        return opportunities
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity):
        try:
            buy_result = await self.okx_client.place_order(
                opportunity.symbol, "buy", "0.01", str(opportunity.buy_price)
            )
            
            if buy_result.get("code") == "0":
                sell_result = await self.okx_client.place_order(
                    opportunity.symbol, "sell", "0.01", str(opportunity.sell_price)
                )
                
                if sell_result.get("code") == "0":
                    profit = (opportunity.sell_price - opportunity.buy_price) * 0.01
                    message = f"Arbitrage executed for {opportunity.symbol}: Profit ${profit:.4f}"
                    self.notification_service.send_alert(message)
                    
        except Exception as e:
            self.notification_service.send_alert(f"Arbitrage failed: {str(e)}")
    
    async def run(self):
        self.running = True
        symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT"]
        
        while self.running:
            try:
                prices = {}
                for symbol in symbols:
                    ticker = await self.okx_client.get_ticker(symbol)
                    if ticker.get("code") == "0" and ticker.get("data"):
                        data = ticker["data"][0]
                        prices[symbol] = {
                            "bid": data["bidPx"],
                            "ask": data["askPx"]
                        }
                
                opportunities = self.calculate_arbitrage(prices)
                
                for opportunity in opportunities:
                    await self.execute_arbitrage(opportunity)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.notification_service.send_alert(f"Bot error: {str(e)}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = ArbitrageBot()
    asyncio.run(bot.run())

# ===== JAVASCRIPT - High Frequency Trading Engine =====
# hft_engine.js

const WebSocket = require('ws');
const crypto = require('crypto');
const axios = require('axios');

class HFTEngine {
    constructor() {
        this.apiKey = "8a760df1-4a2d-471b-ba42-d16893614dab";
        this.secretKey = "C9F3FC89A6A30226E11DFFD098C7CF3D";
        this.passphrase = "trading_bot_2024";
        this.baseUrl = "https://www.okx.com";
        this.discordWebhook = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3";
        this.orderBook = {};
        this.lastPrices = {};
        this.positions = {};
        this.running = false;
    }

    sign(timestamp, method, requestPath, body = "") {
        const message = timestamp + method + requestPath + body;
        const signature = crypto.createHmac('sha256', this.secretKey).update(message).digest('base64');
        return signature;
    }

    async sendDiscordAlert(message) {
        try {
            await axios.post(this.discordWebhook, {
                content: `⚡ HFT Engine: ${message}`
            });
        } catch (error) {
            console.error("Discord alert failed:", error.message);
        }
    }

    async placeOrder(symbol, side, amount, price) {
        const timestamp = Date.now().toString();
        const method = "POST";
        const requestPath = "/api/v5/trade/order";
        
        const body = JSON.stringify({
            instId: symbol,
            tdMode: "cash",
            side: side,
            ordType: "limit",
            sz: amount,
            px: price
        });

        const headers = {
            'OK-ACCESS-KEY': this.apiKey,
            'OK-ACCESS-SIGN': this.sign(timestamp, method, requestPath, body),
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': this.passphrase,
            'Content-Type': 'application/json'
        };

        try {
            const response = await axios.post(`${this.baseUrl}${requestPath}`, JSON.parse(body), { headers });
            return response.data;
        } catch (error) {
            throw new Error(`Order failed: ${error.message}`);
        }
    }

    analyzeMarketMicrostructure(symbol) {
        const orderBook = this.orderBook[symbol];
        if (!orderBook) return null;

        const bidAskSpread = parseFloat(orderBook.asks[0][0]) - parseFloat(orderBook.bids[0][0]);
        const midPrice = (parseFloat(orderBook.asks[0][0]) + parseFloat(orderBook.bids[0][0])) / 2;
        const spreadPercentage = (bidAskSpread / midPrice) * 100;

        const bidDepth = orderBook.bids.slice(0, 5).reduce((sum, [price, size]) => sum + parseFloat(size), 0);
        const askDepth = orderBook.asks.slice(0, 5).reduce((sum, [price, size]) => sum + parseFloat(size), 0);
        const imbalance = (bidDepth - askDepth) / (bidDepth + askDepth);

        return {
            spread: bidAskSpread,
            spreadPercentage,
            midPrice,
            imbalance,
            bidDepth,
            askDepth
        };
    }

    async executeMomentumStrategy(symbol, analysis) {
        const currentPrice = analysis.midPrice;
        const lastPrice = this.lastPrices[symbol];
        
        if (!lastPrice) {
            this.lastPrices[symbol] = currentPrice;
            return;
        }

        const priceChange = (currentPrice - lastPrice) / lastPrice;
        const threshold = 0.001;

        if (Math.abs(priceChange) > threshold && analysis.spreadPercentage < 0.1) {
            const side = priceChange > 0 ? "buy" : "sell";
            const size = "0.01";
            const price = side === "buy" ? analysis.midPrice * 1.0001 : analysis.midPrice * 0.9999;

            try {
                const result = await this.placeOrder(symbol, side, size, price.toString());
                if (result.code === "0") {
                    await this.sendDiscordAlert(`Momentum trade: ${side} ${symbol} at ${price}`);
                }
            } catch (error) {
                console.error(`Momentum strategy failed: ${error.message}`);
            }
        }

        this.lastPrices[symbol] = currentPrice;
    }

    connectWebSocket() {
        const ws = new WebSocket('wss://ws.okx.com:8443/ws/v5/public');
        
        ws.on('open', () => {
            console.log('HFT WebSocket connected');
            
            const subscriptions = [
                { "op": "subscribe", "args": [{ "channel": "books", "instId": "BTC-USDT" }] },
                { "op": "subscribe", "args": [{ "channel": "books", "instId": "ETH-USDT" }] },
                { "op": "subscribe", "args": [{ "channel": "books", "instId": "SOL-USDT" }] }
            ];

            subscriptions.forEach(sub => ws.send(JSON.stringify(sub)));
        });

        ws.on('message', async (data) => {
            try {
                const message = JSON.parse(data);
                
                if (message.data && message.arg && message.arg.channel === 'books') {
                    const symbol = message.arg.instId;
                    const bookData = message.data[0];
                    
                    this.orderBook[symbol] = {
                        bids: bookData.bids,
                        asks: bookData.asks,
                        timestamp: bookData.ts
                    };

                    const analysis = this.analyzeMarketMicrostructure(symbol);
                    if (analysis) {
                        await this.executeMomentumStrategy(symbol, analysis);
                    }
                }
            } catch (error) {
                console.error('WebSocket message error:', error.message);
            }
        });

        ws.on('error', (error) => {
            console.error('WebSocket error:', error.message);
            setTimeout(() => this.connectWebSocket(), 5000);
        });

        ws.on('close', () => {
            console.log('WebSocket disconnected, reconnecting...');
            setTimeout(() => this.connectWebSocket(), 5000);
        });
    }

    start() {
        this.running = true;
        this.connectWebSocket();
        this.sendDiscordAlert("HFT Engine started");
    }

    stop() {
        this.running = false;
        this.sendDiscordAlert("HFT Engine stopped");
    }
}

const hftEngine = new HFTEngine();
hftEngine.start();

# ===== RUST - Ultra-Low Latency Order Execution =====
# order_executor.rs

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::sleep;
use serde_json::{json, Value};
use reqwest::Client;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use base64;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone)]
pub struct OrderRequest {
    pub symbol: String,
    pub side: String,
    pub amount: String,
    pub price: String,
    pub order_type: String,
}

#[derive(Debug)]
pub struct ExecutionEngine {
    api_key: String,
    secret_key: String,
    passphrase: String,
    base_url: String,
    client: Client,
    discord_webhook: String,
}

impl ExecutionEngine {
    pub fn new() -> Self {
        Self {
            api_key: "8a760df1-4a2d-471b-ba42-d16893614dab".to_string(),
            secret_key: "C9F3FC89A6A30226E11DFFD098C7CF3D".to_string(),
            passphrase: "trading_bot_2024".to_string(),
            base_url: "https://www.okx.com".to_string(),
            client: Client::new(),
            discord_webhook: "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3".to_string(),
        }
    }

    fn sign(&self, timestamp: &str, method: &str, request_path: &str, body: &str) -> String {
        let message = format!("{}{}{}{}", timestamp, method, request_path, body);
        let mut mac = HmacSha256::new_from_slice(self.secret_key.as_bytes()).unwrap();
        mac.update(message.as_bytes());
        base64::encode(mac.finalize().into_bytes())
    }

    pub async fn execute_order(&self, order: OrderRequest) -> Result<Value, Box<dyn std::error::Error>> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis()
            .to_string();

        let body = json!({
            "instId": order.symbol,
            "tdMode": "cash",
            "side": order.side,
            "ordType": order.order_type,
            "sz": order.amount,
            "px": order.price
        });

        let body_str = body.to_string();
        let request_path = "/api/v5/trade/order";
        let signature = self.sign(&timestamp, "POST", request_path, &body_str);

        let response = self.client
            .post(&format!("{}{}", self.base_url, request_path))
            .header("OK-ACCESS-KEY", &self.api_key)
            .header("OK-ACCESS-SIGN", signature)
            .header("OK-ACCESS-TIMESTAMP", timestamp)
            .header("OK-ACCESS-PASSPHRASE", &self.passphrase)
            .header("Content-Type", "application/json")
            .body(body_str)
            .send()
            .await?;

        let result: Value = response.json().await?;
        
        if result["code"] == "0" {
            self.send_discord_alert(&format!(
                "Order executed: {} {} {} at {}",
                order.side, order.amount, order.symbol, order.price
            )).await?;
        }

        Ok(result)
    }

    async fn send_discord_alert(&self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        let payload = json!({
            "content": format!("⚡ Rust Executor: {}", message)
        });

        self.client
            .post(&self.discord_webhook)
            .json(&payload)
            .send()
            .await?;

        Ok(())
    }

    pub async fn batch_execute(&self, orders: Vec<OrderRequest>) -> Vec<Result<Value, Box<dyn std::error::Error>>> {
        let mut results = Vec::new();
        
        for order in orders {
            let result = self.execute_order(order).await;
            results.push(result);
            sleep(Duration::from_millis(10)).await;
        }
        
        results
    }

    pub async fn market_making_strategy(&self, symbol: &str, mid_price: f64, spread: f64) -> Result<(), Box<dyn std::error::Error>> {
        let bid_price = mid_price - (spread / 2.0);
        let ask_price = mid_price + (spread / 2.0);
        
        let buy_order = OrderRequest {
            symbol: symbol.to_string(),
            side: "buy".to_string(),
            amount: "0.01".to_string(),
            price: bid_price.to_string(),
            order_type: "limit".to_string(),
        };
        
        let sell_order = OrderRequest {
            symbol: symbol.to_string(),
            side: "sell".to_string(),
            amount: "0.01".to_string(),
            price: ask_price.to_string(),
            order_type: "limit".to_string(),
        };

        let orders = vec![buy_order, sell_order];
        let results = self.batch_execute(orders).await;
        
        for (i, result) in results.iter().enumerate() {
            match result {
                Ok(res) => println!("Order {} executed: {:?}", i, res),
                Err(e) => println!("Order {} failed: {:?}", i, e),
            }
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let executor = ExecutionEngine::new();
    
    loop {
        let symbols = vec!["BTC-USDT", "ETH-USDT", "SOL-USDT"];
        
        for symbol in symbols {
            executor.market_making_strategy(symbol, 50000.0, 10.0).await?;
            sleep(Duration::from_secs(5)).await;
        }
    }
}

# ===== GO - Market Data Aggregator =====
# market_data_aggregator.go

package main

import (
    "bytes"
    "crypto/hmac"
    "crypto/sha256"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
    "strconv"
    "time"
    "sync"
)

type MarketData struct {
    Symbol    string  `json:"symbol"`
    Bid       float64 `json:"bid"`
    Ask       float64 `json:"ask"`
    LastPrice float64 `json:"last_price"`
    Volume    float64 `json:"volume"`
    Timestamp int64   `json:"timestamp"`
}

type OKXTicker struct {
    InstID   string `json:"instId"`
    Last     string `json:"last"`
    BidPx    string `json:"bidPx"`
    AskPx    string `json:"askPx"`
    Vol24h   string `json:"vol24h"`
    Ts       string `json:"ts"`
}

type OKXResponse struct {
    Code string      `json:"code"`
    Data []OKXTicker `json:"data"`
}

type DataAggregator struct {
    apiKey      string
    secretKey   string
    passphrase  string
    baseURL     string
    webhookURL  string
    mu          sync.RWMutex
    marketData  map[string]MarketData
}

func NewDataAggregator() *DataAggregator {
    return &DataAggregator{
        apiKey:     "8a760df1-4a2d-471b-ba42-d16893614dab",
        secretKey:  "C9F3FC89A6A30226E11DFFD098C7CF3D",
        passphrase: "trading_bot_2024",
        baseURL:    "https://www.okx.com",
        webhookURL: "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3",
        marketData: make(map[string]MarketData),
    }
}

func (da *DataAggregator) sign(timestamp, method, requestPath, body string) string {
    message := timestamp + method + requestPath + body
    h := hmac.New(sha256.New, []byte(da.secretKey))
    h.Write([]byte(message))
    return base64.StdEncoding.EncodeToString(h.Sum(nil))
}

func (da *DataAggregator) sendDiscordAlert(message string) error {
    payload := map[string]string{
        "content": fmt.Sprintf("📊 Go Aggregator: %s", message),
    }
    
    jsonPayload, _ := json.Marshal(payload)
    
    resp, err := http.Post(da.webhookURL, "application/json", bytes.NewBuffer(jsonPayload))
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    return nil
}

func (da *DataAggregator) fetchMarketData(symbol string) (*MarketData, error) {
    timestamp := strconv.FormatInt(time.Now().UnixMilli(), 10)
    method := "GET"
    requestPath := "/api/v5/market/ticker?instId=" + symbol
    
    req, err := http.NewRequest(method, da.baseURL+requestPath, nil)
    if err != nil {
        return nil, err
    }
    
    signature := da.sign(timestamp, method, requestPath, "")
    
    req.Header.Set("OK-ACCESS-KEY", da.apiKey)
    req.Header.Set("OK-ACCESS-SIGN", signature)
    req.Header.Set("OK-ACCESS-TIMESTAMP", timestamp)
    req.Header.Set("OK-ACCESS-PASSPHRASE", da.passphrase)
    req.Header.Set("Content-Type", "application/json")
    
    client := &http.Client{Timeout: 10 * time.Second}
    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        return nil, err
    }
    
    var okxResp OKXResponse
    if err := json.Unmarshal(body, &okxResp); err != nil {
        return nil, err
    }
    
    if len(okxResp.Data) == 0 {
        return nil, fmt.Errorf("no data received for %s", symbol)
    }
    
    ticker := okxResp.Data[0]
    
    bid, _ := strconv.ParseFloat(ticker.BidPx, 64)
    ask, _ := strconv.ParseFloat(ticker.AskPx, 64)
    last, _ := strconv.ParseFloat(ticker.Last, 64)
    volume, _ := strconv.ParseFloat(ticker.Vol24h, 64)
    timestamp_int, _ := strconv.ParseInt(ticker.Ts, 10, 64)
    
    return &MarketData{
        Symbol:    symbol,
        Bid:       bid,
        Ask:       ask,
        LastPrice: last,
        Volume:    volume,
        Timestamp: timestamp_int,
    }, nil
}

func (da *DataAggregator) detectArbitrageOpportunities() {
    da.mu.RLock()
    defer da.mu.RUnlock()
    
    for symbol, data := range da.marketData {
        spread := ((data.Ask - data.Bid) / data.Bid) * 100
        
        if spread > 0.5 {
            message := fmt.Sprintf("Arbitrage opportunity detected: %s spread %.2f%%", symbol, spread)
            go da.sendDiscordAlert(message)
        }
    }
}

func (da *DataAggregator) updateMarketData() {
    symbols := []string{"BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT", "ADA-USDT"}
    
    for _, symbol := range symbols {
        go func(sym string) {
            data, err := da.fetchMarketData(sym)
            if err != nil {
                fmt.Printf("Error fetching data for %s: %v\n", sym, err)
                return
            }
            
            da.mu.Lock()
            da.marketData[sym] = *data
            da.mu.Unlock()
            
            fmt.Printf("Updated %s: Bid=%.2f, Ask=%.2f, Last=%.2f\n", 
                data.Symbol, data.Bid, data.Ask, data.LastPrice)
        }(symbol)
    }
}

func (da *DataAggregator) run() {
    da.sendDiscordAlert("Market Data Aggregator started")
    
    ticker := time.NewTicker(2 * time.Second)
    arbitrageTicker := time.NewTicker(10 * time.Second)
    
    for {
        select {
        case <-ticker.C:
            da.updateMarketData()
        case <-arbitrageTicker.C:
            da.detectArbitrageOpportunities()
        }
    }
}

func main() {
    aggregator := NewDataAggregator()
    aggregator.run()
}

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

# ===== JAVA - Risk Management System =====
# RiskManager.java

import java.util.*;
import java.util.concurrent.*;
import java.io.*;
import java.net.http.*;
import java.net.URI;
import java.time.Instant;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.util.Base64;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;

public class RiskManager {
    private static final String API_KEY = "8a760df1-4a2d-471b-ba42-d16893614dab";
    private static final String SECRET_KEY = "C9F3FC89A6A30226E11DFFD098C7CF3D";
    private static final String PASSPHRASE = "trading_bot_2024";
    private static final String BASE_URL = "https://www.okx.com";
    private static final String DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1398448251933298740/lSnT3iPsfvb87RWdN0XCd3AjdFsCZiTpF-_I1ciV3rB2BqTpIszS6U6tFxAVk5QmM2q3";
    
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final Map<String, Double> positions;
    private final Map<String, Double> maxPositions;
    private final Map<String, Double> stopLosses;
    private final Map<String, Double> profitTargets;
    private double maxDrawdown = 0.05;
    private double maxDailyLoss = 1000.0;
    private double currentDailyPnL = 0.0;
    private boolean tradingEnabled = true;

    public RiskManager() {
        this.httpClient = HttpClient.newHttpClient();
        this.objectMapper = new ObjectMapper();
        this.positions = new ConcurrentHashMap<>();
        this.maxPositions = new ConcurrentHashMap<>();
        this.stopLosses = new ConcurrentHashMap<>();
        this.profitTargets = new ConcurrentHashMap<>();
        
        initializeRiskParameters();
    }

    private void initializeRiskParameters() {
        maxPositions.put("BTC-USDT", 0.1);
        maxPositions.put("ETH-USDT", 1.0);
        maxPositions.put("SOL-USDT", 10.0);
        maxPositions.put("DOGE-USDT", 1000.0);
        
        stopLosses.put("BTC-USDT", 0.02);
        stopLosses.put("ETH-USDT", 0.03);
        stopLosses.put("SOL-USDT", 0.05);
        stopLosses.put("DOGE-USDT", 0.10);
        
        profitTargets.put("BTC-USDT", 0.05);
        profitTargets.put("ETH-USDT", 0.08);
        profitTargets.put("SOL-USDT", 0.15);
        profitTargets.put("DOGE-USDT", 0.25);
    }

    private String sign(String timestamp, String method, String requestPath, String body) throws Exception {
        String message = timestamp + method + requestPath + body;
        Mac mac = Mac.getInstance("HmacSHA256");
        SecretKeySpec secretKeySpec = new SecretKeySpec(SECRET_KEY.getBytes(), "HmacSHA256");
        mac.init(secretKeySpec);
        byte[] digest = mac.doFinal(message.getBytes());
        return Base64.getEncoder().encodeToString(digest);
    }

    public boolean validateOrder(String symbol, String side, double amount, double price) {
        if (!tradingEnabled) {
            sendDiscordAlert("Trading disabled due to risk limits");
            return false;
        }

        if (Math.abs(currentDailyPnL) > maxDailyLoss) {
            tradingEnabled = false;
            sendDiscordAlert("Daily loss limit exceeded: " + currentDailyPnL);
            return false;
        }

        double currentPosition = positions.getOrDefault(symbol, 0.0);
        double newPosition = side.equals("buy") ? currentPosition + amount : currentPosition - amount;
        double maxPosition = maxPositions.getOrDefault(symbol, 0.0);

        if (Math.abs(newPosition) > maxPosition) {
            sendDiscordAlert("Position limit exceeded for " + symbol + ": " + Math.abs(newPosition) + " > " + maxPosition);
            return false;
        }

        double notionalValue = Math.abs(amount) * price;
        if (notionalValue > 10000) {
            sendDiscordAlert("Order size too large: $" + notionalValue);
            return false;
        }

        return true;
    }

    public void updatePosition(String symbol, String side, double amount, double price) {
        double currentPosition = positions.getOrDefault(symbol, 0.0);
        double newPosition = side.equals("buy") ? currentPosition + amount : currentPosition - amount;
        positions.put(symbol, newPosition);
        
        double pnl = calculateUnrealizedPnL(symbol, price);
        currentDailyPnL += pnl;
        
        checkStopLossAndProfitTarget(symbol, price);
    }

    private double calculateUnrealizedPnL(String symbol, double currentPrice) {
        double position = positions.getOrDefault(symbol, 0.0);
        if (position == 0.0) return 0.0;
        
        return position * currentPrice * 0.001;
    }

    private void checkStopLossAndProfitTarget(String symbol, double currentPrice) {
        double position = positions.getOrDefault(symbol, 0.0);
        if (position == 0.0) return;
        
        double entryPrice = 50000.0;
        double priceChange = (currentPrice - entryPrice) / entryPrice;
        
        double stopLoss = stopLosses.getOrDefault(symbol, 0.05);
        double profitTarget = profitTargets.getOrDefault(symbol, 0.10);
        
        boolean shouldClosePosition = false;
        String reason = "";
        
        if (position > 0) {
            if (priceChange <= -stopLoss) {
                shouldClosePosition = true;
                reason = "Stop loss triggered";
            } else if (priceChange >= profitTarget) {
                shouldClosePosition = true;
                reason = "Profit target reached";
            }
        } else if (position < 0) {
            if (priceChange >= stopLoss) {
                shouldClosePosition = true;
                reason = "Stop loss triggered";
            } else if (priceChange <= -profitTarget) {
                shouldClosePosition = true;
                reason = "Profit target reached";
            }
        }
        
        if (shouldClosePosition) {
            try {
                closePosition(symbol, reason);
            } catch (Exception e) {
                sendDiscordAlert("Failed to close position for " + symbol + ": " + e.getMessage());
            }
        }
    }

    private void closePosition(String symbol, String reason) throws Exception {
        double position = positions.get(symbol);
        if (position == 0.0) return;
        
        String side = position > 0 ? "sell" : "buy";
        double amount = Math.abs(position);
        
        placeOrder(symbol, side, amount, 50000.0);
        positions.put(symbol, 0.0);
        
        sendDiscordAlert("Position closed for " + symbol + ": " + reason);
    }

    private void placeOrder(String symbol, String side, double amount, double price) throws Exception {
        String timestamp = String.valueOf(Instant.now().toEpochMilli());
        String method = "POST";
        String requestPath = "/api/v5/trade/order";
        
        Map<String, Object> orderData = new HashMap<>();
        orderData.put("instId", symbol);
        orderData.put("tdMode", "cash");
        orderData.put("side", side);
        orderData.put("ordType", "limit");
        orderData.put("sz", String.valueOf(amount));
        orderData.put("px", String.valueOf(price));
        
        String body = objectMapper.writeValueAsString(orderData);
        String signature = sign(timestamp, method, requestPath, body);
        
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(BASE_URL + requestPath))
            .header("OK-ACCESS-KEY", API_KEY)
            .header("OK-ACCESS-SIGN", signature)
            .header("OK-ACCESS-TIMESTAMP", timestamp)
            .header("OK-ACCESS-PASSPHRASE", PASSPHRASE)
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body))
            .build();
        
        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        
        if (response.statusCode() == 200) {
            JsonNode responseJson = objectMapper.readTree(response.body());
            if ("0".equals(responseJson.get("code").asText())) {
                sendDiscordAlert("Risk management order executed: " + side + " " + amount + " " + symbol);
            }
        }
    }

    private void sendDiscordAlert(String message) {
        try {
            Map<String, String> payload = new HashMap<>();
            payload.put("content", "🛡️ Java Risk Manager: " + message);
            
            String jsonPayload = objectMapper.writeValueAsString(payload);
            
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(DISCORD_WEBHOOK))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                .build();
            
            httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString());
        } catch (Exception e) {
            System.err.println("Failed to send Discord alert: " + e.getMessage());
        }
    }

    public void generateRiskReport() {
        StringBuilder report = new StringBuilder();
        report.append("**Risk Management Report**\n");
        report.append("Daily PnL: $").append(String.format("%.2f", currentDailyPnL)).append("\n");
        report.append("Trading Enabled: ").append(tradingEnabled).append("\n");
        report.append("Active Positions:\n");
        
        for (Map.Entry<String, Double> entry : positions.entrySet()) {
            if (entry.getValue() != 0.0) {
                report.append("- ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
            }
        }
        
        sendDiscordAlert(report.toString());
    }

    public void runRiskMonitoring() {
        sendDiscordAlert("Risk Management System started");
        
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);
        
        scheduler.scheduleAtFixedRate(() -> {
            try {
                for (String symbol : Arrays.asList("BTC-USDT", "ETH-USDT", "SOL-USDT")) {
                    double mockPrice = 50000 + (Math.random() * 1000 - 500);
                    checkStopLossAndProfitTarget(symbol, mockPrice);
                }
            } catch (Exception e) {
                sendDiscordAlert("Risk monitoring error: " + e.getMessage());
            }
        }, 0, 5, TimeUnit.SECONDS);
        
        scheduler.scheduleAtFixedRate(this::generateRiskReport, 0, 30, TimeUnit.MINUTES);
    }

    public static void main(String[] args) {
        RiskManager riskManager = new RiskManager();
        riskManager.runRiskMonitoring();
    }
}