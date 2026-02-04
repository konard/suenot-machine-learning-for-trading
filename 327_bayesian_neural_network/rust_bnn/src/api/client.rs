//! Bybit API client for fetching market data.

use anyhow::{anyhow, Result};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use std::time::Duration;

use super::types::{ApiResponse, Kline, KlinesResult, OrderBook, Ticker};

/// Bybit API endpoints.
const BYBIT_API_URL: &str = "https://api.bybit.com";
const BYBIT_TESTNET_URL: &str = "https://api-testnet.bybit.com";

/// Bybit API client.
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new Bybit client for mainnet.
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: BYBIT_API_URL.to_string(),
        }
    }

    /// Create a new Bybit client for testnet.
    pub fn testnet() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: BYBIT_TESTNET_URL.to_string(),
        }
    }

    /// Fetch OHLCV kline data.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Timeframe (e.g., "1", "5", "15", "60", "D")
    /// * `limit` - Number of candles to fetch (max 200)
    ///
    /// # Returns
    /// Vector of Kline data
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: ApiResponse<KlinesResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!("API error: {}", response.ret_msg));
        }

        let klines: Vec<Kline> = response
            .result
            .list
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    let timestamp_ms: i64 = item[0].parse().ok()?;
                    let timestamp = Utc.timestamp_millis_opt(timestamp_ms).single()?;

                    Some(Kline {
                        timestamp,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                        turnover: item.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Reverse to get chronological order (API returns newest first)
        let mut klines = klines;
        klines.reverse();

        Ok(klines)
    }

    /// Fetch klines with pagination for longer history.
    pub async fn get_klines_extended(
        &self,
        symbol: &str,
        interval: &str,
        total_limit: u32,
    ) -> Result<Vec<Kline>> {
        let mut all_klines = Vec::new();
        let batch_size = 200;
        let mut end_time: Option<i64> = None;

        while all_klines.len() < total_limit as usize {
            let remaining = total_limit as usize - all_klines.len();
            let limit = remaining.min(batch_size);

            let mut url = format!(
                "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
                self.base_url, symbol, interval, limit
            );

            if let Some(end) = end_time {
                url.push_str(&format!("&end={}", end));
            }

            let response: ApiResponse<KlinesResult> = self
                .client
                .get(&url)
                .send()
                .await?
                .json()
                .await?;

            if response.ret_code != 0 {
                return Err(anyhow!("API error: {}", response.ret_msg));
            }

            let klines: Vec<Kline> = response
                .result
                .list
                .into_iter()
                .filter_map(|item| {
                    if item.len() >= 6 {
                        let timestamp_ms: i64 = item[0].parse().ok()?;
                        let timestamp = Utc.timestamp_millis_opt(timestamp_ms).single()?;

                        Some(Kline {
                            timestamp,
                            open: item[1].parse().ok()?,
                            high: item[2].parse().ok()?,
                            low: item[3].parse().ok()?,
                            close: item[4].parse().ok()?,
                            volume: item[5].parse().ok()?,
                            turnover: item.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                        })
                    } else {
                        None
                    }
                })
                .collect();

            if klines.is_empty() {
                break;
            }

            // Get earliest timestamp for next batch
            if let Some(earliest) = klines.last() {
                end_time = Some(earliest.timestamp.timestamp_millis() - 1);
            }

            all_klines.extend(klines);

            // Rate limiting
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Reverse to chronological order
        all_klines.reverse();
        Ok(all_klines)
    }

    /// Fetch ticker data for a symbol.
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        #[derive(serde::Deserialize)]
        struct TickerResult {
            list: Vec<TickerItem>,
        }

        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct TickerItem {
            symbol: String,
            last_price: String,
            high_price_24h: String,
            low_price_24h: String,
            volume_24h: String,
            turnover_24h: String,
            price_24h_pcnt: String,
            bid1_price: String,
            ask1_price: String,
        }

        let response: ApiResponse<TickerResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!("API error: {}", response.ret_msg));
        }

        let item = response
            .result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No ticker data"))?;

        Ok(Ticker {
            symbol: item.symbol,
            last_price: item.last_price.parse()?,
            high_24h: item.high_price_24h.parse()?,
            low_24h: item.low_price_24h.parse()?,
            volume_24h: item.volume_24h.parse()?,
            turnover_24h: item.turnover_24h.parse()?,
            price_change_pct: item.price_24h_pcnt.parse()?,
            bid_price: item.bid1_price.parse()?,
            ask_price: item.ask1_price.parse()?,
            timestamp: Utc::now(),
        })
    }

    /// Fetch order book for a symbol.
    pub async fn get_order_book(&self, symbol: &str, limit: u32) -> Result<OrderBook> {
        let url = format!(
            "{}/v5/market/orderbook?category=linear&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        #[derive(serde::Deserialize)]
        struct OrderBookResult {
            s: String,
            b: Vec<Vec<String>>,
            a: Vec<Vec<String>>,
        }

        let response: ApiResponse<OrderBookResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!("API error: {}", response.ret_msg));
        }

        let result = response.result;

        let bids: Vec<(f64, f64)> = result
            .b
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 2 {
                    Some((item[0].parse().ok()?, item[1].parse().ok()?))
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<(f64, f64)> = result
            .a
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 2 {
                    Some((item[0].parse().ok()?, item[1].parse().ok()?))
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            symbol: result.s,
            bids,
            asks,
            timestamp: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_ticker() {
        let client = BybitClient::new();
        let result = client.get_ticker("BTCUSDT").await;
        // This test requires network access
        if let Ok(ticker) = result {
            assert_eq!(ticker.symbol, "BTCUSDT");
            assert!(ticker.last_price > 0.0);
        }
    }
}
