//! Bybit Exchange Client
//!
//! Provides integration with Bybit cryptocurrency exchange API.

use super::{ExchangeClient, OrderBook, OrderBookLevel, Trade, Ticker, TradeSide};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bybit client configuration
#[derive(Debug, Clone)]
pub struct BybitConfig {
    /// API key (optional for public endpoints)
    pub api_key: Option<String>,
    /// API secret (optional for public endpoints)
    pub api_secret: Option<String>,
    /// Use testnet
    pub testnet: bool,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            api_secret: None,
            testnet: true,
        }
    }
}

/// Bybit exchange client
#[derive(Debug, Clone)]
pub struct BybitClient {
    config: BybitConfig,
    client: reqwest::Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new(config: BybitConfig) -> Self {
        let base_url = if config.testnet {
            "https://api-testnet.bybit.com".to_string()
        } else {
            "https://api.bybit.com".to_string()
        };

        Self {
            config,
            client: reqwest::Client::new(),
            base_url,
        }
    }

    /// Generate signature for authenticated requests
    fn sign(&self, params: &str, timestamp: i64) -> Option<String> {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        let secret = self.config.api_secret.as_ref()?;
        let api_key = self.config.api_key.as_ref()?;

        let sign_str = format!("{}{}{}", timestamp, api_key, params);

        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).ok()?;
        mac.update(sign_str.as_bytes());
        let result = mac.finalize();

        Some(hex::encode(result.into_bytes()))
    }

    /// Make a GET request
    async fn get<T: for<'de> Deserialize<'de>>(&self, endpoint: &str, params: Option<&HashMap<String, String>>) -> Result<T> {
        let url = format!("{}{}", self.base_url, endpoint);

        let mut request = self.client.get(&url);

        if let Some(p) = params {
            request = request.query(p);
        }

        let response = request
            .send()
            .await
            .context("Failed to send request")?;

        let text = response.text().await.context("Failed to read response")?;

        serde_json::from_str(&text)
            .context(format!("Failed to parse response: {}", text))
    }
}

#[async_trait::async_trait]
impl ExchangeClient for BybitClient {
    async fn get_orderbook(&self, symbol: &str, depth: usize) -> Result<OrderBook> {
        #[derive(Deserialize)]
        struct Response {
            #[serde(rename = "retCode")]
            ret_code: i32,
            #[serde(rename = "retMsg")]
            ret_msg: String,
            result: OrderBookResult,
        }

        #[derive(Deserialize)]
        struct OrderBookResult {
            s: String,
            b: Vec<Vec<String>>,
            a: Vec<Vec<String>>,
            ts: u64,
            u: u64,
        }

        let mut params = HashMap::new();
        params.insert("category".to_string(), "linear".to_string());
        params.insert("symbol".to_string(), symbol.to_string());
        params.insert("limit".to_string(), depth.to_string());

        let response: Response = self.get("/v5/market/orderbook", Some(&params)).await?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let bids = response.result.b.iter()
            .map(|level| OrderBookLevel {
                price: level[0].parse().unwrap_or(0.0),
                quantity: level[1].parse().unwrap_or(0.0),
            })
            .collect();

        let asks = response.result.a.iter()
            .map(|level| OrderBookLevel {
                price: level[0].parse().unwrap_or(0.0),
                quantity: level[1].parse().unwrap_or(0.0),
            })
            .collect();

        Ok(OrderBook {
            symbol: response.result.s,
            bids,
            asks,
            timestamp: DateTime::from_timestamp_millis(response.result.ts as i64)
                .unwrap_or_else(Utc::now),
            sequence: response.result.u,
        })
    }

    async fn get_trades(&self, symbol: &str, limit: usize) -> Result<Vec<Trade>> {
        #[derive(Deserialize)]
        struct Response {
            #[serde(rename = "retCode")]
            ret_code: i32,
            #[serde(rename = "retMsg")]
            ret_msg: String,
            result: TradesResult,
        }

        #[derive(Deserialize)]
        struct TradesResult {
            list: Vec<TradeItem>,
        }

        #[derive(Deserialize)]
        struct TradeItem {
            #[serde(rename = "execId")]
            exec_id: String,
            symbol: String,
            price: String,
            size: String,
            side: String,
            time: String,
        }

        let mut params = HashMap::new();
        params.insert("category".to_string(), "linear".to_string());
        params.insert("symbol".to_string(), symbol.to_string());
        params.insert("limit".to_string(), limit.to_string());

        let response: Response = self.get("/v5/market/recent-trade", Some(&params)).await?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let trades = response.result.list.iter()
            .map(|t| Trade {
                symbol: t.symbol.clone(),
                trade_id: t.exec_id.clone(),
                price: t.price.parse().unwrap_or(0.0),
                quantity: t.size.parse().unwrap_or(0.0),
                side: if t.side == "Buy" { TradeSide::Buy } else { TradeSide::Sell },
                timestamp: DateTime::from_timestamp_millis(t.time.parse().unwrap_or(0))
                    .unwrap_or_else(Utc::now),
            })
            .collect();

        Ok(trades)
    }

    async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        #[derive(Deserialize)]
        struct Response {
            #[serde(rename = "retCode")]
            ret_code: i32,
            #[serde(rename = "retMsg")]
            ret_msg: String,
            result: TickerResult,
        }

        #[derive(Deserialize)]
        struct TickerResult {
            list: Vec<TickerItem>,
        }

        #[derive(Deserialize)]
        struct TickerItem {
            symbol: String,
            #[serde(rename = "lastPrice")]
            last_price: String,
            #[serde(rename = "highPrice24h")]
            high_24h: String,
            #[serde(rename = "lowPrice24h")]
            low_24h: String,
            #[serde(rename = "volume24h")]
            volume_24h: String,
            #[serde(rename = "price24hPcnt")]
            price_change_pct: String,
        }

        let mut params = HashMap::new();
        params.insert("category".to_string(), "linear".to_string());
        params.insert("symbol".to_string(), symbol.to_string());

        let response: Response = self.get("/v5/market/tickers", Some(&params)).await?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let item = response.result.list.first()
            .context("No ticker data found")?;

        Ok(Ticker {
            symbol: item.symbol.clone(),
            last_price: item.last_price.parse().unwrap_or(0.0),
            high_24h: item.high_24h.parse().unwrap_or(0.0),
            low_24h: item.low_24h.parse().unwrap_or(0.0),
            volume_24h: item.volume_24h.parse().unwrap_or(0.0),
            price_change_pct: item.price_change_pct.parse::<f64>().unwrap_or(0.0) * 100.0,
            timestamp: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bybit_config_default() {
        let config = BybitConfig::default();
        assert!(config.testnet);
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_bybit_client_creation() {
        let client = BybitClient::new(BybitConfig::default());
        assert!(client.base_url.contains("testnet"));
    }

    #[test]
    fn test_bybit_client_mainnet() {
        let client = BybitClient::new(BybitConfig {
            testnet: false,
            ..Default::default()
        });
        assert!(!client.base_url.contains("testnet"));
    }
}
