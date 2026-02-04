//! Bybit API client for fetching market data.

use anyhow::{anyhow, Result};
use reqwest::Client;
use std::time::Duration;
use tracing::{info, warn};

use super::types::*;

const BYBIT_BASE_URL: &str = "https://api.bybit.com";

/// Bybit API client
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: BYBIT_BASE_URL.to_string(),
        }
    }

    /// Create client with custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.to_string(),
        }
    }

    /// Fetch OHLCV klines data
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

        info!("Fetching klines for {} ({})", symbol, interval);

        let response = self.client.get(&url).send().await?;
        let body: BybitResponse<KlinesResult> = response.json().await?;

        if body.ret_code != 0 {
            return Err(anyhow!("API error: {}", body.ret_msg));
        }

        let klines: Vec<Kline> = body
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Kline {
                        timestamp: row[0].parse().unwrap_or(0),
                        open: row[1].parse().unwrap_or(0.0),
                        high: row[2].parse().unwrap_or(0.0),
                        low: row[3].parse().unwrap_or(0.0),
                        close: row[4].parse().unwrap_or(0.0),
                        volume: row[5].parse().unwrap_or(0.0),
                        symbol: symbol.to_string(),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Reverse to get chronological order
        let mut klines = klines;
        klines.reverse();

        info!("Fetched {} klines for {}", klines.len(), symbol);
        Ok(klines)
    }

    /// Fetch current ticker
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response = self.client.get(&url).send().await?;
        let body: BybitResponse<TickerResult> = response.json().await?;

        if body.ret_code != 0 {
            return Err(anyhow!("API error: {}", body.ret_msg));
        }

        let ticker_info = body
            .result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No ticker found for {}", symbol))?;

        let last_price: f64 = ticker_info.last_price.parse().unwrap_or(0.0);
        let price_change_pct: f64 = ticker_info.price_change_pct.parse().unwrap_or(0.0);
        let high_24h: f64 = ticker_info.high_24h.parse().unwrap_or(0.0);
        let low_24h: f64 = ticker_info.low_24h.parse().unwrap_or(0.0);

        Ok(Ticker {
            symbol: ticker_info.symbol,
            last_price,
            bid_price: ticker_info.bid_price.parse().unwrap_or(0.0),
            ask_price: ticker_info.ask_price.parse().unwrap_or(0.0),
            volume_24h: ticker_info.volume_24h.parse().unwrap_or(0.0),
            price_change_24h: last_price * price_change_pct,
            price_change_pct_24h: price_change_pct * 100.0,
            high_24h,
            low_24h,
            timestamp: chrono::Utc::now().timestamp_millis(),
        })
    }

    /// Fetch order book
    pub async fn get_order_book(&self, symbol: &str, limit: u32) -> Result<OrderBook> {
        let url = format!(
            "{}/v5/market/orderbook?category=linear&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let response = self.client.get(&url).send().await?;
        let body: BybitResponse<OrderBookResult> = response.json().await?;

        if body.ret_code != 0 {
            return Err(anyhow!("API error: {}", body.ret_msg));
        }

        let bids: Vec<OrderBookLevel> = body
            .result
            .b
            .iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some(OrderBookLevel {
                        price: row[0].parse().unwrap_or(0.0),
                        quantity: row[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<OrderBookLevel> = body
            .result
            .a
            .iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some(OrderBookLevel {
                        price: row[0].parse().unwrap_or(0.0),
                        quantity: row[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            symbol: body.result.s,
            timestamp: body.result.ts,
            bids,
            asks,
        })
    }

    /// Fetch klines for multiple symbols
    pub async fn get_multiple_klines(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: u32,
    ) -> Result<Vec<(String, Vec<Kline>)>> {
        let mut results = Vec::new();

        for symbol in symbols {
            match self.get_klines(symbol, interval, limit).await {
                Ok(klines) => {
                    results.push((symbol.to_string(), klines));
                }
                Err(e) => {
                    warn!("Failed to fetch klines for {}: {}", symbol, e);
                }
            }
            // Rate limiting
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(results)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, BYBIT_BASE_URL);
    }
}
