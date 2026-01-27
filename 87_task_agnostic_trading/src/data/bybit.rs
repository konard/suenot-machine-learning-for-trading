//! Bybit API client for fetching market data

use super::types::{FundingRate, Kline, OrderBook, OrderBookLevel, Ticker};
use chrono::{TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Bybit API errors
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("API error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Rate limited")]
    RateLimited,
}

/// Bybit API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitConfig {
    /// Base URL for the API
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Whether to use testnet
    pub testnet: bool,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            timeout_secs: 30,
            testnet: false,
        }
    }
}

impl BybitConfig {
    /// Create config for testnet
    pub fn testnet() -> Self {
        Self {
            base_url: "https://api-testnet.bybit.com".to_string(),
            testnet: true,
            ..Default::default()
        }
    }
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: Option<T>,
}

/// Kline response data
#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

/// Ticker response data
#[derive(Debug, Deserialize)]
struct TickerResult {
    list: Vec<TickerData>,
}

#[derive(Debug, Deserialize)]
struct TickerData {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "highPrice24h")]
    high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    low_price_24h: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "turnover24h")]
    turnover_24h: String,
    #[serde(rename = "price24hPcnt")]
    price_24h_pcnt: String,
}

/// Order book response data
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    b: Vec<Vec<String>>, // bids
    a: Vec<Vec<String>>, // asks
}

/// Funding rate response data
#[derive(Debug, Deserialize)]
struct FundingResult {
    list: Vec<FundingData>,
}

#[derive(Debug, Deserialize)]
struct FundingData {
    #[allow(dead_code)]
    symbol: String,
    #[serde(rename = "fundingRate")]
    funding_rate: String,
    #[serde(rename = "fundingRateTimestamp")]
    funding_rate_timestamp: String,
}

/// Bybit API client for fetching market data
pub struct BybitClient {
    client: Client,
    config: BybitConfig,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new(config: BybitConfig) -> Result<Self, BybitError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()?;

        Ok(Self { client, config })
    }

    /// Create a client with default configuration
    pub fn default_client() -> Result<Self, BybitError> {
        Self::new(BybitConfig::default())
    }

    /// Fetch kline (candlestick) data
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>, BybitError> {
        let url = format!("{}/v5/market/kline", self.config.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let api_response: ApiResponse<KlineResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let result = api_response
            .result
            .ok_or_else(|| BybitError::ParseError("No result in response".to_string()))?;

        let mut klines: Vec<Kline> = result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 7 {
                    let timestamp_ms: i64 = item[0].parse().ok()?;
                    Some(Kline {
                        timestamp: Utc.timestamp_millis_opt(timestamp_ms).single()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                        turnover: item[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse for chronological order
        klines.reverse();
        Ok(klines)
    }

    /// Fetch ticker data for a symbol
    pub async fn fetch_ticker(&self, symbol: &str) -> Result<Ticker, BybitError> {
        let url = format!("{}/v5/market/tickers", self.config.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[("category", "linear"), ("symbol", symbol)])
            .send()
            .await?;

        let api_response: ApiResponse<TickerResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let result = api_response
            .result
            .ok_or_else(|| BybitError::ParseError("No result".to_string()))?;

        let data = result
            .list
            .first()
            .ok_or_else(|| BybitError::ParseError("Empty ticker list".to_string()))?;

        Ok(Ticker {
            symbol: data.symbol.clone(),
            last_price: data.last_price.parse().unwrap_or(0.0),
            high_24h: data.high_price_24h.parse().unwrap_or(0.0),
            low_24h: data.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: data.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: data.turnover_24h.parse().unwrap_or(0.0),
            price_change_pct: data.price_24h_pcnt.parse().unwrap_or(0.0),
        })
    }

    /// Fetch order book data
    pub async fn fetch_orderbook(
        &self,
        symbol: &str,
        depth: usize,
    ) -> Result<OrderBook, BybitError> {
        let url = format!("{}/v5/market/orderbook", self.config.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("limit", &depth.to_string()),
            ])
            .send()
            .await?;

        let api_response: ApiResponse<OrderBookResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let result = api_response
            .result
            .ok_or_else(|| BybitError::ParseError("No result".to_string()))?;

        let bids = result
            .b
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(OrderBookLevel {
                        price: level[0].parse().ok()?,
                        quantity: level[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks = result
            .a
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(OrderBookLevel {
                        price: level[0].parse().ok()?,
                        quantity: level[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            timestamp: Utc::now(),
            bids,
            asks,
        })
    }

    /// Fetch funding rate
    pub async fn fetch_funding_rate(&self, symbol: &str) -> Result<FundingRate, BybitError> {
        let url = format!("{}/v5/market/funding/history", self.config.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("limit", "1"),
            ])
            .send()
            .await?;

        let api_response: ApiResponse<FundingResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let result = api_response
            .result
            .ok_or_else(|| BybitError::ParseError("No result".to_string()))?;

        let data = result
            .list
            .first()
            .ok_or_else(|| BybitError::ParseError("Empty funding list".to_string()))?;

        let ts: i64 = data
            .funding_rate_timestamp
            .parse()
            .unwrap_or(Utc::now().timestamp_millis());

        Ok(FundingRate {
            timestamp: Utc
                .timestamp_millis_opt(ts)
                .single()
                .unwrap_or_else(Utc::now),
            rate: data.funding_rate.parse().unwrap_or(0.0),
        })
    }

    /// Fetch klines for multiple symbols concurrently
    pub async fn fetch_multi_klines(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Vec<(String, Result<Vec<Kline>, BybitError>)> {
        let futures: Vec<_> = symbols
            .iter()
            .map(|&symbol| {
                let sym = symbol.to_string();
                let client = self.client.clone();
                let url = format!("{}/v5/market/kline", self.config.base_url);
                let interval = interval.to_string();

                async move {
                    let response = client
                        .get(&url)
                        .query(&[
                            ("category", "linear"),
                            ("symbol", sym.as_str()),
                            ("interval", interval.as_str()),
                            ("limit", &limit.to_string()),
                        ])
                        .send()
                        .await;

                    let result = match response {
                        Ok(resp) => {
                            let api_resp: Result<ApiResponse<KlineResult>, _> = resp.json().await;
                            match api_resp {
                                Ok(api) => {
                                    if api.ret_code != 0 {
                                        Err(BybitError::ApiError {
                                            code: api.ret_code,
                                            message: api.ret_msg,
                                        })
                                    } else if let Some(result) = api.result {
                                        let mut klines: Vec<Kline> = result
                                            .list
                                            .iter()
                                            .filter_map(|item| {
                                                if item.len() >= 7 {
                                                    let ts: i64 = item[0].parse().ok()?;
                                                    Some(Kline {
                                                        timestamp: Utc
                                                            .timestamp_millis_opt(ts)
                                                            .single()?,
                                                        open: item[1].parse().ok()?,
                                                        high: item[2].parse().ok()?,
                                                        low: item[3].parse().ok()?,
                                                        close: item[4].parse().ok()?,
                                                        volume: item[5].parse().ok()?,
                                                        turnover: item[6].parse().ok()?,
                                                    })
                                                } else {
                                                    None
                                                }
                                            })
                                            .collect();
                                        klines.reverse();
                                        Ok(klines)
                                    } else {
                                        Err(BybitError::ParseError("No result".to_string()))
                                    }
                                }
                                Err(e) => Err(BybitError::HttpError(e)),
                            }
                        }
                        Err(e) => Err(BybitError::HttpError(e)),
                    };

                    (sym, result)
                }
            })
            .collect();

        futures::future::join_all(futures).await
    }
}
