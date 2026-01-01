//! Bybit HTTP API Client

use crate::api::models::*;
use anyhow::{anyhow, Result};
use chrono::Utc;
use hmac::{Hmac, Mac};
use reqwest::Client;
use sha2::Sha256;
use std::collections::HashMap;

/// Bybit API client
pub struct BybitClient {
    /// HTTP client
    client: Client,
    /// Base URL for API
    base_url: String,
    /// API key (optional, for authenticated endpoints)
    api_key: Option<String>,
    /// API secret (optional, for authenticated endpoints)
    api_secret: Option<String>,
    /// Receive window for authenticated requests
    recv_window: u64,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new client for public endpoints
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://api.bybit.com".to_string(),
            api_key: None,
            api_secret: None,
            recv_window: 5000,
        }
    }

    /// Create a client with API credentials
    pub fn with_credentials(api_key: String, api_secret: String) -> Self {
        Self {
            api_key: Some(api_key),
            api_secret: Some(api_secret),
            ..Self::new()
        }
    }

    /// Use testnet instead of mainnet
    pub fn testnet(mut self) -> Self {
        self.base_url = "https://api-testnet.bybit.com".to_string();
        self
    }

    /// Fetch kline/candlestick data
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
        start_time: Option<i64>,
    ) -> Result<Vec<Kline>> {
        let mut params = vec![
            ("category", "linear".to_string()),
            ("symbol", symbol.to_string()),
            ("interval", interval.to_string()),
            ("limit", limit.to_string()),
        ];

        if let Some(start) = start_time {
            params.push(("start", start.to_string()));
        }

        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self.client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        let api_response: ApiResponse<KlineListResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(anyhow!("API error: {}", api_response.ret_msg));
        }

        // Parse kline data from string arrays
        let klines = api_response.result.list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Kline {
                        start_time: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                        turnover: row[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(klines)
    }

    /// Fetch historical klines with pagination
    pub async fn get_historical_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<Kline>> {
        let mut all_klines = Vec::new();
        let mut current_start = start_time;

        // Interval to milliseconds
        let interval_ms = Self::interval_to_ms(interval)?;

        loop {
            let klines = self.get_klines(symbol, interval, 200, Some(current_start)).await?;

            if klines.is_empty() {
                break;
            }

            let last_time = klines.first().map(|k| k.start_time).unwrap_or(0);
            all_klines.extend(klines);

            // Move start time forward
            current_start = last_time + interval_ms;

            if current_start >= end_time {
                break;
            }

            // Rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        // Filter to requested range and sort
        all_klines.retain(|k| k.start_time >= start_time && k.start_time <= end_time);
        all_klines.sort_by_key(|k| k.start_time);
        all_klines.reverse(); // Most recent first

        Ok(all_klines)
    }

    /// Get order book depth
    pub async fn get_orderbook(&self, symbol: &str, limit: usize) -> Result<OrderBook> {
        let params = vec![
            ("category", "linear"),
            ("symbol", symbol),
            ("limit", &limit.to_string()),
        ];

        let url = format!("{}/v5/market/orderbook", self.base_url);

        let response = self.client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        let api_response: ApiResponse<OrderBookResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(anyhow!("API error: {}", api_response.ret_msg));
        }

        let result = api_response.result;

        // Parse bids and asks
        let bids: Vec<(f64, f64)> = result.bids
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some((row[0].parse().ok()?, row[1].parse().ok()?))
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<(f64, f64)> = result.asks
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some((row[0].parse().ok()?, row[1].parse().ok()?))
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            symbol: result.symbol,
            bids,
            asks,
            timestamp: result.timestamp,
        })
    }

    /// Get ticker for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let params = vec![
            ("category", "linear"),
            ("symbol", symbol),
        ];

        let url = format!("{}/v5/market/tickers", self.base_url);

        let response = self.client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        let response_text = response.text().await?;
        let value: serde_json::Value = serde_json::from_str(&response_text)?;

        if value["retCode"].as_i64() != Some(0) {
            return Err(anyhow!("API error: {}", value["retMsg"]));
        }

        let list = value["result"]["list"].as_array()
            .ok_or_else(|| anyhow!("No ticker data"))?;

        if list.is_empty() {
            return Err(anyhow!("Ticker not found for {}", symbol));
        }

        let t = &list[0];

        Ok(Ticker {
            symbol: t["symbol"].as_str().unwrap_or("").to_string(),
            last_price: t["lastPrice"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            high_24h: t["highPrice24h"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            low_24h: t["lowPrice24h"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            volume_24h: t["volume24h"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            turnover_24h: t["turnover24h"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            price_change_percent: t["price24hPcnt"].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0) * 100.0,
            timestamp: Utc::now().timestamp_millis(),
        })
    }

    /// Get recent trades
    pub async fn get_recent_trades(&self, symbol: &str, limit: usize) -> Result<Vec<Trade>> {
        let params = vec![
            ("category", "linear"),
            ("symbol", symbol),
            ("limit", &limit.to_string()),
        ];

        let url = format!("{}/v5/market/recent-trade", self.base_url);

        let response = self.client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        let response_text = response.text().await?;
        let value: serde_json::Value = serde_json::from_str(&response_text)?;

        if value["retCode"].as_i64() != Some(0) {
            return Err(anyhow!("API error: {}", value["retMsg"]));
        }

        let list = value["result"]["list"].as_array()
            .ok_or_else(|| anyhow!("No trade data"))?;

        let trades: Vec<Trade> = list
            .iter()
            .filter_map(|t| {
                Some(Trade {
                    id: t["execId"].as_str()?.to_string(),
                    symbol: t["symbol"].as_str()?.to_string(),
                    price: t["price"].as_str()?.parse().ok()?,
                    quantity: t["size"].as_str()?.parse().ok()?,
                    side: if t["side"].as_str()? == "Buy" {
                        TradeSide::Buy
                    } else {
                        TradeSide::Sell
                    },
                    timestamp: t["time"].as_str()?.parse().ok()?,
                })
            })
            .collect();

        Ok(trades)
    }

    /// Get funding rate
    pub async fn get_funding_rate(&self, symbol: &str) -> Result<FundingRate> {
        let params = vec![
            ("category", "linear"),
            ("symbol", symbol),
        ];

        let url = format!("{}/v5/market/funding/history", self.base_url);

        let response = self.client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        let response_text = response.text().await?;
        let value: serde_json::Value = serde_json::from_str(&response_text)?;

        if value["retCode"].as_i64() != Some(0) {
            return Err(anyhow!("API error: {}", value["retMsg"]));
        }

        let list = value["result"]["list"].as_array()
            .ok_or_else(|| anyhow!("No funding data"))?;

        if list.is_empty() {
            return Err(anyhow!("No funding rate data for {}", symbol));
        }

        let f = &list[0];

        Ok(FundingRate {
            symbol: f["symbol"].as_str().unwrap_or("").to_string(),
            funding_rate: f["fundingRate"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            funding_rate_timestamp: f["fundingRateTimestamp"].as_str().unwrap_or("0").parse().unwrap_or(0),
        })
    }

    /// Get server time
    pub async fn get_server_time(&self) -> Result<i64> {
        let url = format!("{}/v5/market/time", self.base_url);

        let response = self.client
            .get(&url)
            .send()
            .await?;

        let response_text = response.text().await?;
        let value: serde_json::Value = serde_json::from_str(&response_text)?;

        if value["retCode"].as_i64() != Some(0) {
            return Err(anyhow!("API error: {}", value["retMsg"]));
        }

        value["result"]["timeSecond"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| anyhow!("Failed to parse server time"))
    }

    /// Convert interval string to milliseconds
    fn interval_to_ms(interval: &str) -> Result<i64> {
        let ms = match interval {
            "1" => 60 * 1000,
            "3" => 3 * 60 * 1000,
            "5" => 5 * 60 * 1000,
            "15" => 15 * 60 * 1000,
            "30" => 30 * 60 * 1000,
            "60" => 60 * 60 * 1000,
            "120" => 2 * 60 * 60 * 1000,
            "240" => 4 * 60 * 60 * 1000,
            "360" => 6 * 60 * 60 * 1000,
            "720" => 12 * 60 * 60 * 1000,
            "D" => 24 * 60 * 60 * 1000,
            "W" => 7 * 24 * 60 * 60 * 1000,
            "M" => 30 * 24 * 60 * 60 * 1000,
            _ => return Err(anyhow!("Unknown interval: {}", interval)),
        };
        Ok(ms)
    }

    /// Generate signature for authenticated requests
    #[allow(dead_code)]
    fn generate_signature(&self, timestamp: i64, params: &str) -> Result<String> {
        let api_key = self.api_key.as_ref().ok_or_else(|| anyhow!("API key not set"))?;
        let api_secret = self.api_secret.as_ref().ok_or_else(|| anyhow!("API secret not set"))?;

        let sign_str = format!("{}{}{}{}", timestamp, api_key, self.recv_window, params);

        let mut mac = Hmac::<Sha256>::new_from_slice(api_secret.as_bytes())
            .map_err(|e| anyhow!("HMAC error: {}", e))?;
        mac.update(sign_str.as_bytes());
        let result = mac.finalize();

        Ok(hex::encode(result.into_bytes()))
    }

    /// Create signed headers for authenticated requests
    #[allow(dead_code)]
    fn create_auth_headers(&self, params: &str) -> Result<HashMap<String, String>> {
        let api_key = self.api_key.as_ref().ok_or_else(|| anyhow!("API key not set"))?;
        let timestamp = Utc::now().timestamp_millis();
        let signature = self.generate_signature(timestamp, params)?;

        let mut headers = HashMap::new();
        headers.insert("X-BAPI-API-KEY".to_string(), api_key.clone());
        headers.insert("X-BAPI-TIMESTAMP".to_string(), timestamp.to_string());
        headers.insert("X-BAPI-SIGN".to_string(), signature);
        headers.insert("X-BAPI-RECV-WINDOW".to_string(), self.recv_window.to_string());
        headers.insert("Content-Type".to_string(), "application/json".to_string());

        Ok(headers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_to_ms() {
        assert_eq!(BybitClient::interval_to_ms("1").unwrap(), 60_000);
        assert_eq!(BybitClient::interval_to_ms("60").unwrap(), 3_600_000);
        assert_eq!(BybitClient::interval_to_ms("D").unwrap(), 86_400_000);
    }

    #[tokio::test]
    async fn test_client_creation() {
        let client = BybitClient::new();
        assert!(client.api_key.is_none());

        let auth_client = BybitClient::with_credentials(
            "test_key".to_string(),
            "test_secret".to_string(),
        );
        assert!(auth_client.api_key.is_some());
    }
}
