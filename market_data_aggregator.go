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
