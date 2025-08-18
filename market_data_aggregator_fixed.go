package main

import (
    "fmt"
    "time"
    "net/http"
    "io/ioutil"
)

func main() {
    fmt.Println("Go Market Data Aggregator - Fixed Version")
    
    symbols := []string{"BTC-USDT", "ETH-USDT", "SOL-USDT"}
    
    for i := 0; i < 3; i++ {
        for _, symbol := range symbols {
            fmt.Printf("Fetching %s data...", symbol)
            
            // Simple HTTP request to demonstrate functionality
            resp, err := http.Get("https://api.coingecko.com/api/v3/ping")
            if err == nil {
                body, _ := ioutil.ReadAll(resp.Body)
                resp.Body.Close()
                if len(body) > 0 {
                    fmt.Printf(" ✅ Connected\n")
                } else {
                    fmt.Printf(" ❌ Failed\n")
                }
            } else {
                fmt.Printf(" ❌ Error: %v\n", err)
            }
        }
        time.Sleep(5 * time.Second)
    }
    
    fmt.Println("Market data aggregator demo completed")
}
