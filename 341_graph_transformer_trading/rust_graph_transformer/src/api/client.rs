//! Bybit HTTP API client
//!
//! Provides methods for fetching market data from Bybit exchange.

use super::types::*;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use std::collections::HashMap;
use std::time::Duration;

/// Bybit API base URL
const BYBIT_API_BASE: &str = "https://api.bybit.com";

/// Bybit API client for fetching market data
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
    /// Create a new Bybit client
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: BYBIT_API_BASE.to_string(),
        }
    }

    /// Create a new client with custom base URL
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

    /// Get all tickers for spot market
    pub async fn get_tickers(&self) -> Result<Vec<Ticker>> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[("category", "spot")])
            .send()
            .await
            .context("Failed to fetch tickers")?;

        let api_response: ApiResponse<TickerListResult> = response
            .json()
            .await
            .context("Failed to parse ticker response")?;

        if api_response.ret_code != 0 {
            anyhow::bail!("API error: {}", api_response.ret_msg);
        }

        Ok(api_response.result.list)
    }

    /// Get ticker for a specific symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[("category", "spot"), ("symbol", symbol)])
            .send()
            .await
            .context("Failed to fetch ticker")?;

        let api_response: ApiResponse<TickerListResult> = response
            .json()
            .await
            .context("Failed to parse ticker response")?;

        if api_response.ret_code != 0 {
            anyhow::bail!("API error: {}", api_response.ret_msg);
        }

        api_response
            .result
            .list
            .into_iter()
            .next()
            .context("Ticker not found")
    }

    /// Get klines (candlestick data) for a symbol
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Candlestick interval (e.g., "1", "5", "15", "60", "D")
    /// * `limit` - Number of candles to fetch (max 1000)
    pub async fn get_klines(&self, symbol: &str, interval: &str, limit: u32) -> Result<Vec<Kline>> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await
            .context("Failed to fetch klines")?;

        let api_response: ApiResponse<KlineListResult> = response
            .json()
            .await
            .context("Failed to parse kline response")?;

        if api_response.ret_code != 0 {
            anyhow::bail!("API error: {}", api_response.ret_msg);
        }

        let klines: Vec<Kline> = api_response
            .result
            .list
            .iter()
            .filter_map(|data| Kline::from_api_response(data))
            .collect();

        Ok(klines)
    }

    /// Get order book for a symbol
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol
    /// * `depth` - Order book depth (5, 10, 20, 50, 100, 200)
    pub async fn get_orderbook(&self, symbol: &str, depth: u32) -> Result<OrderBook> {
        let url = format!("{}/v5/market/orderbook", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("limit", &depth.to_string()),
            ])
            .send()
            .await
            .context("Failed to fetch orderbook")?;

        let api_response: ApiResponse<OrderBookResult> = response
            .json()
            .await
            .context("Failed to parse orderbook response")?;

        if api_response.ret_code != 0 {
            anyhow::bail!("API error: {}", api_response.ret_msg);
        }

        let result = api_response.result;

        let bids: Vec<OrderBookLevel> = result
            .b
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(OrderBookLevel {
                        price: level[0].parse().ok()?,
                        size: level[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<OrderBookLevel> = result
            .a
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(OrderBookLevel {
                        price: level[0].parse().ok()?,
                        size: level[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            symbol: result.s,
            bids,
            asks,
            timestamp: result.ts,
            update_id: result.u,
        })
    }

    /// Get recent trades for a symbol
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol
    /// * `limit` - Number of trades to fetch (max 1000)
    pub async fn get_recent_trades(&self, symbol: &str, limit: u32) -> Result<Vec<Trade>> {
        let url = format!("{}/v5/market/recent-trade", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await
            .context("Failed to fetch trades")?;

        #[derive(serde::Deserialize)]
        struct TradeResult {
            list: Vec<TradeRaw>,
        }

        #[derive(serde::Deserialize)]
        struct TradeRaw {
            #[serde(rename = "execId")]
            exec_id: String,
            symbol: String,
            price: String,
            size: String,
            side: String,
            time: String,
            #[serde(rename = "isBlockTrade")]
            is_block_trade: bool,
        }

        let api_response: ApiResponse<TradeResult> = response
            .json()
            .await
            .context("Failed to parse trade response")?;

        if api_response.ret_code != 0 {
            anyhow::bail!("API error: {}", api_response.ret_msg);
        }

        let trades: Vec<Trade> = api_response
            .result
            .list
            .into_iter()
            .filter_map(|t| {
                Some(Trade {
                    id: t.exec_id,
                    symbol: t.symbol,
                    price: t.price.parse().ok()?,
                    size: t.size.parse().ok()?,
                    is_buyer_maker: t.side == "Sell",
                    timestamp: t.time.parse().ok()?,
                })
            })
            .collect();

        Ok(trades)
    }

    /// Get klines for multiple symbols in parallel
    pub async fn get_klines_batch(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: u32,
    ) -> Result<HashMap<String, Vec<Kline>>> {
        let mut results = HashMap::new();

        // Fetch in parallel using tokio::join_all
        let futures: Vec<_> = symbols
            .iter()
            .map(|symbol| self.get_klines(symbol, interval, limit))
            .collect();

        let responses = futures::future::join_all(futures).await;

        for (symbol, response) in symbols.iter().zip(responses) {
            if let Ok(klines) = response {
                results.insert(symbol.to_string(), klines);
            }
        }

        Ok(results)
    }

    /// Get tickers for specific symbols
    pub async fn get_tickers_filtered(&self, symbols: &[&str]) -> Result<Vec<Ticker>> {
        let all_tickers = self.get_tickers().await?;

        let filtered: Vec<Ticker> = all_tickers
            .into_iter()
            .filter(|t| symbols.contains(&t.symbol.as_str()))
            .collect();

        Ok(filtered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, BYBIT_API_BASE);
    }
}
