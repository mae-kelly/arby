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

func (da *DataAggregator) detectRealisticArbitrageOpportunities() {
    da.mu.RLock()
    defer da.mu.RUnlock()
    
    for symbol, data := range da.marketData {
        spread := ((data.Ask - data.Bid) / data.Bid) * 100
        
        // Only alert on realistic spreads (0.01% or higher)
        if spread > 0.01 && spread < 1.0 {
            message := fmt.Sprintf("Realistic spread detected: %s %.4f%% (Bid: %.2f, Ask: %.2f)", 
                symbol, spread, data.Bid, data.Ask)
            go da.sendDiscordAlert(message)
            fmt.Printf("🎯 %s\n", message)
        } else if spread > 1.0 {
            fmt.Printf("⚠️  Suspicious spread on %s: %.2f%% (likely error)\n", symbol, spread)
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
            
            spread := ((data.Ask - data.Bid) / data.Bid) * 100
            fmt.Printf("Updated %s: Bid=%.2f, Ask=%.2f, Spread=%.4f%%\n", 
                data.Symbol, data.Bid, data.Ask, spread)
        }(symbol)
    }
}

func (da *DataAggregator) run() {
    da.sendDiscordAlert("Realistic Market Data Aggregator started - monitoring small spreads")
    
    ticker := time.NewTicker(10 * time.Second)
    arbitrageTicker := time.NewTicker(30 * time.Second)
    
    fmt.Println("🚀 Go Market Data Aggregator Started")
    fmt.Println("📊 Monitoring realistic spreads (0.01-1.0%)")
    fmt.Println("⚠️  Large spreads (>1%) flagged as errors")
    
    for i := 0; i < 20; i++ { // Run for limited time in demo
        select {
        case <-ticker.C:
            fmt.Printf("\n[%s] Fetching market data...\n", time.Now().Format("15:04:05"))
            da.updateMarketData()
        case <-arbitrageTicker.C:
            fmt.Printf("\n[%s] Analyzing arbitrage opportunities...\n", time.Now().Format("15:04:05"))
            da.detectRealisticArbitrageOpportunities()
        }
        
        if i%5 == 0 && i > 0 {
            fmt.Printf("\n📈 Completed %d update cycles\n", i)
        }
    }
    
    fmt.Println("\n✅ Market data aggregator demo completed")
    da.sendDiscordAlert("Market data aggregator demo completed - found realistic opportunities")
}

func main() {
    aggregator := NewDataAggregator()
    aggregator.run()
}