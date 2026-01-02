//! Bybit API Client
//!
//! Client for fetching cryptocurrency market data from Bybit exchange.

use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::candle::Candle;
use super::orderbook::{OrderBook, OrderBookLevel};

/// Bybit API client for fetching market data
#[derive(Debug, Clone)]
pub struct BybitClient {
    /// HTTP client
    client: Client,

    /// Base URL for API
    base_url: String,

    /// Request timeout
    timeout: Duration,
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline (candlestick) response
#[derive(Debug, Deserialize)]
struct KlineResult {
    #[allow(dead_code)]
    category: String,
    #[allow(dead_code)]
    symbol: String,
    list: Vec<Vec<String>>,
}

/// Order book response
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    s: String, // symbol
    b: Vec<Vec<String>>, // bids
    a: Vec<Vec<String>>, // asks
    ts: u64, // timestamp
    #[allow(dead_code)]
    u: u64, // update id
}

impl BybitClient {
    /// Create a new Bybit client (mainnet)
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://api.bybit.com".to_string(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Create a client for testnet
    pub fn testnet() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://api-testnet.bybit.com".to_string(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Create a client with custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: base_url.to_string(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candle interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
    /// * `limit` - Number of candles (max 1000)
    /// * `start` - Start time in milliseconds (optional)
    /// * `end` - End time in milliseconds (optional)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
        start: Option<u64>,
        end: Option<u64>,
    ) -> Result<Vec<Candle>> {
        let mut url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol,
            interval,
            limit.min(1000)
        );

        if let Some(s) = start {
            url.push_str(&format!("&start={}", s));
        }
        if let Some(e) = end {
            url.push_str(&format!("&end={}", e));
        }

        let response: BybitResponse<KlineResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!("Bybit API error: {}", response.ret_msg));
        }

        let candles: Vec<Candle> = response
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Candle {
                        timestamp: row[0].parse().unwrap_or(0),
                        open: row[1].parse().unwrap_or(0.0),
                        high: row[2].parse().unwrap_or(0.0),
                        low: row[3].parse().unwrap_or(0.0),
                        close: row[4].parse().unwrap_or(0.0),
                        volume: row[5].parse().unwrap_or(0.0),
                        turnover: row[6].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse to get chronological order
        let mut sorted_candles = candles;
        sorted_candles.reverse();

        Ok(sorted_candles)
    }

    /// Fetch klines for multiple symbols
    pub async fn get_multi_klines(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Result<Vec<(String, Vec<Candle>)>> {
        let mut results = Vec::new();

        for symbol in symbols {
            let candles = self.get_klines(symbol, interval, limit, None, None).await?;
            results.push((symbol.to_string(), candles));
        }

        Ok(results)
    }

    /// Fetch order book
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `limit` - Depth limit (1, 25, 50, 100, 200)
    pub async fn get_orderbook(&self, symbol: &str, limit: usize) -> Result<OrderBook> {
        let url = format!(
            "{}/v5/market/orderbook?category=linear&symbol={}&limit={}",
            self.base_url,
            symbol,
            limit.min(200)
        );

        let response: BybitResponse<OrderBookResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!("Bybit API error: {}", response.ret_msg));
        }

        let result = response.result;

        let bids: Vec<OrderBookLevel> = result
            .b
            .into_iter()
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

        let asks: Vec<OrderBookLevel> = result
            .a
            .into_iter()
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

        Ok(OrderBook::new(result.s, bids, asks, result.ts))
    }

    /// Fetch ticker information
    pub async fn get_ticker(&self, symbol: &str) -> Result<TickerInfo> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response: serde_json::Value = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        let ret_code = response["retCode"].as_i64().unwrap_or(-1);
        if ret_code != 0 {
            return Err(anyhow!("Bybit API error: {:?}", response["retMsg"]));
        }

        let list = &response["result"]["list"];
        if let Some(ticker) = list.get(0) {
            Ok(TickerInfo {
                symbol: ticker["symbol"].as_str().unwrap_or("").to_string(),
                last_price: ticker["lastPrice"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                bid_price: ticker["bid1Price"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                ask_price: ticker["ask1Price"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                volume_24h: ticker["volume24h"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                turnover_24h: ticker["turnover24h"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                high_24h: ticker["highPrice24h"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                low_24h: ticker["lowPrice24h"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                price_change_24h: ticker["price24hPcnt"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                funding_rate: ticker["fundingRate"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                open_interest: ticker["openInterest"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            })
        } else {
            Err(anyhow!("No ticker data found"))
        }
    }

    /// Fetch tickers for multiple symbols
    pub async fn get_tickers(&self, symbols: &[&str]) -> Result<Vec<TickerInfo>> {
        let mut tickers = Vec::new();

        for symbol in symbols {
            match self.get_ticker(symbol).await {
                Ok(ticker) => tickers.push(ticker),
                Err(e) => log::warn!("Failed to fetch ticker for {}: {}", symbol, e),
            }
        }

        Ok(tickers)
    }

    /// Fetch historical funding rate
    pub async fn get_funding_rate_history(
        &self,
        symbol: &str,
        limit: usize,
    ) -> Result<Vec<FundingRate>> {
        let url = format!(
            "{}/v5/market/funding/history?category=linear&symbol={}&limit={}",
            self.base_url,
            symbol,
            limit.min(200)
        );

        let response: serde_json::Value = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        let ret_code = response["retCode"].as_i64().unwrap_or(-1);
        if ret_code != 0 {
            return Err(anyhow!("Bybit API error: {:?}", response["retMsg"]));
        }

        let list = response["result"]["list"].as_array();
        let funding_rates: Vec<FundingRate> = list
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        Some(FundingRate {
                            symbol: item["symbol"].as_str()?.to_string(),
                            funding_rate: item["fundingRate"].as_str()?.parse().ok()?,
                            funding_rate_timestamp: item["fundingRateTimestamp"]
                                .as_str()?
                                .parse()
                                .ok()?,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(funding_rates)
    }

    /// Get list of popular trading symbols
    pub fn popular_symbols() -> Vec<&'static str> {
        vec![
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "ADAUSDT",
            "DOGEUSDT",
            "AVAXUSDT",
            "DOTUSDT",
            "LINKUSDT",
            "MATICUSDT",
            "LTCUSDT",
            "ATOMUSDT",
            "UNIUSDT",
            "NEARUSDT",
        ]
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Ticker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerInfo {
    pub symbol: String,
    pub last_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub volume_24h: f64,
    pub turnover_24h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub price_change_24h: f64,
    pub funding_rate: f64,
    pub open_interest: f64,
}

impl TickerInfo {
    /// Calculate the spread as percentage
    pub fn spread_pct(&self) -> f64 {
        if self.bid_price > 0.0 {
            (self.ask_price - self.bid_price) / self.bid_price
        } else {
            0.0
        }
    }

    /// Calculate mid price
    pub fn mid_price(&self) -> f64 {
        (self.bid_price + self.ask_price) / 2.0
    }
}

/// Funding rate data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    pub symbol: String,
    pub funding_rate: f64,
    pub funding_rate_timestamp: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert!(client.base_url.contains("bybit.com"));
    }

    #[test]
    fn test_testnet_client() {
        let client = BybitClient::testnet();
        assert!(client.base_url.contains("testnet"));
    }

    #[test]
    fn test_popular_symbols() {
        let symbols = BybitClient::popular_symbols();
        assert!(symbols.contains(&"BTCUSDT"));
        assert!(symbols.contains(&"ETHUSDT"));
        assert!(symbols.len() >= 10);
    }

    #[test]
    fn test_ticker_spread() {
        let ticker = TickerInfo {
            symbol: "BTCUSDT".to_string(),
            last_price: 50000.0,
            bid_price: 49999.0,
            ask_price: 50001.0,
            volume_24h: 1000000.0,
            turnover_24h: 50000000000.0,
            high_24h: 51000.0,
            low_24h: 49000.0,
            price_change_24h: 0.02,
            funding_rate: 0.0001,
            open_interest: 100000.0,
        };

        assert!((ticker.spread_pct() - 0.00004).abs() < 1e-8);
        assert!((ticker.mid_price() - 50000.0).abs() < 1e-8);
    }
}
