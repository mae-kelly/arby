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

# ===== GO - Market