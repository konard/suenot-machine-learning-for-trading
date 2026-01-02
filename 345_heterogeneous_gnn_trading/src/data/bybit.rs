//! Bybit API client for fetching market data

use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use thiserror::Error;

use super::{FundingRate, Kline, OpenInterest, OrderBook, OrderBookLevel, Ticker, Trade};

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

/// Bybit client configuration
#[derive(Debug, Clone)]
pub struct BybitConfig {
    /// Base URL for API
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retries
    pub max_retries: u32,
    /// Whether to use testnet
    pub testnet: bool,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            timeout_secs: 10,
            max_retries: 3,
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

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    /// HTTP client
    client: Client,
    /// Configuration
    pub config: BybitConfig,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new(config: BybitConfig) -> Result<Self, BybitError> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()?;

        Ok(Self { client, config })
    }

    /// Create with default config
    pub fn default_client() -> Result<Self, BybitError> {
        Self::new(BybitConfig::default())
    }

    /// Get klines for a symbol
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>, BybitError> {
        let url = format!("{}/v5/market/kline", self.config.base_url);

        let response: ApiResponse<KlineResponse> = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?
            .json()
            .await?;

        self.check_response(&response)?;

        let klines = response
            .result
            .list
            .into_iter()
            .map(|k| Kline {
                timestamp: k[0].parse().unwrap_or(0),
                open: k[1].parse().unwrap_or(0.0),
                high: k[2].parse().unwrap_or(0.0),
                low: k[3].parse().unwrap_or(0.0),
                close: k[4].parse().unwrap_or(0.0),
                volume: k[5].parse().unwrap_or(0.0),
                turnover: k[6].parse().unwrap_or(0.0),
            })
            .collect();

        Ok(klines)
    }

    /// Get ticker for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker, BybitError> {
        let url = format!("{}/v5/market/tickers", self.config.base_url);

        let response: ApiResponse<TickerResponse> = self
            .client
            .get(&url)
            .query(&[("category", "linear"), ("symbol", symbol)])
            .send()
            .await?
            .json()
            .await?;

        self.check_response(&response)?;

        let ticker_data = response
            .result
            .list
            .first()
            .ok_or_else(|| BybitError::ParseError("No ticker data".to_string()))?;

        Ok(Ticker {
            symbol: ticker_data.symbol.clone(),
            last_price: ticker_data.last_price.parse().unwrap_or(0.0),
            high_24h: ticker_data.high_price_24h.parse().unwrap_or(0.0),
            low_24h: ticker_data.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: ticker_data.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: ticker_data.turnover_24h.parse().unwrap_or(0.0),
            price_change_24h: ticker_data.price_24h_pcnt.parse::<f64>().unwrap_or(0.0) * 100.0,
            bid_price: ticker_data.bid1_price.parse().unwrap_or(0.0),
            ask_price: ticker_data.ask1_price.parse().unwrap_or(0.0),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        })
    }

    /// Get tickers for multiple symbols
    pub async fn get_tickers(&self, symbols: &[&str]) -> Result<HashMap<String, Ticker>, BybitError> {
        let url = format!("{}/v5/market/tickers", self.config.base_url);

        let response: ApiResponse<TickerResponse> = self
            .client
            .get(&url)
            .query(&[("category", "linear")])
            .send()
            .await?
            .json()
            .await?;

        self.check_response(&response)?;

        let symbol_set: std::collections::HashSet<_> = symbols.iter().copied().collect();

        let tickers = response
            .result
            .list
            .into_iter()
            .filter(|t| symbol_set.contains(t.symbol.as_str()))
            .map(|t| {
                let ticker = Ticker {
                    symbol: t.symbol.clone(),
                    last_price: t.last_price.parse().unwrap_or(0.0),
                    high_24h: t.high_price_24h.parse().unwrap_or(0.0),
                    low_24h: t.low_price_24h.parse().unwrap_or(0.0),
                    volume_24h: t.volume_24h.parse().unwrap_or(0.0),
                    turnover_24h: t.turnover_24h.parse().unwrap_or(0.0),
                    price_change_24h: t.price_24h_pcnt.parse::<f64>().unwrap_or(0.0) * 100.0,
                    bid_price: t.bid1_price.parse().unwrap_or(0.0),
                    ask_price: t.ask1_price.parse().unwrap_or(0.0),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0),
                };
                (t.symbol, ticker)
            })
            .collect();

        Ok(tickers)
    }

    /// Get order book
    pub async fn get_orderbook(&self, symbol: &str, limit: u32) -> Result<OrderBook, BybitError> {
        let url = format!("{}/v5/market/orderbook", self.config.base_url);

        let response: ApiResponse<OrderBookResponse> = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?
            .json()
            .await?;

        self.check_response(&response)?;

        let bids: Vec<OrderBookLevel> = response
            .result
            .b
            .into_iter()
            .map(|level| OrderBookLevel {
                price: level[0].parse().unwrap_or(0.0),
                size: level[1].parse().unwrap_or(0.0),
            })
            .collect();

        let asks: Vec<OrderBookLevel> = response
            .result
            .a
            .into_iter()
            .map(|level| OrderBookLevel {
                price: level[0].parse().unwrap_or(0.0),
                size: level[1].parse().unwrap_or(0.0),
            })
            .collect();

        Ok(OrderBook::new(symbol, bids, asks))
    }

    /// Get recent trades
    pub async fn get_trades(&self, symbol: &str, limit: u32) -> Result<Vec<Trade>, BybitError> {
        let url = format!("{}/v5/market/recent-trade", self.config.base_url);

        let response: ApiResponse<TradeResponse> = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?
            .json()
            .await?;

        self.check_response(&response)?;

        let trades = response
            .result
            .list
            .into_iter()
            .map(|t| Trade {
                id: t.exec_id,
                symbol: t.symbol,
                price: t.price.parse().unwrap_or(0.0),
                size: t.size.parse().unwrap_or(0.0),
                side: t.side,
                timestamp: t.time.parse().unwrap_or(0),
            })
            .collect();

        Ok(trades)
    }

    /// Get funding rate
    pub async fn get_funding_rate(&self, symbol: &str) -> Result<FundingRate, BybitError> {
        let url = format!("{}/v5/market/tickers", self.config.base_url);

        let response: ApiResponse<TickerResponse> = self
            .client
            .get(&url)
            .query(&[("category", "linear"), ("symbol", symbol)])
            .send()
            .await?
            .json()
            .await?;

        self.check_response(&response)?;

        let ticker = response
            .result
            .list
            .first()
            .ok_or_else(|| BybitError::ParseError("No ticker data".to_string()))?;

        Ok(FundingRate {
            symbol: ticker.symbol.clone(),
            funding_rate: ticker.funding_rate.parse().unwrap_or(0.0),
            funding_rate_timestamp: 0,
            next_funding_time: ticker.next_funding_time.parse().unwrap_or(0),
        })
    }

    /// Get open interest
    pub async fn get_open_interest(&self, symbol: &str) -> Result<OpenInterest, BybitError> {
        let url = format!("{}/v5/market/open-interest", self.config.base_url);

        let response: ApiResponse<OpenInterestResponse> = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("intervalTime", "5min"),
                ("limit", "1"),
            ])
            .send()
            .await?
            .json()
            .await?;

        self.check_response(&response)?;

        let oi = response
            .result
            .list
            .first()
            .ok_or_else(|| BybitError::ParseError("No OI data".to_string()))?;

        Ok(OpenInterest {
            symbol: symbol.to_string(),
            open_interest: oi.open_interest.parse().unwrap_or(0.0),
            timestamp: oi.timestamp.parse().unwrap_or(0),
        })
    }

    /// Check API response for errors
    fn check_response<T>(&self, response: &ApiResponse<T>) -> Result<(), BybitError> {
        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg.clone(),
            });
        }
        Ok(())
    }
}

// API response structures

#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

#[derive(Debug, Deserialize)]
struct KlineResponse {
    list: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct TickerResponse {
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
    #[serde(rename = "bid1Price")]
    bid1_price: String,
    #[serde(rename = "ask1Price")]
    ask1_price: String,
    #[serde(rename = "fundingRate", default)]
    funding_rate: String,
    #[serde(rename = "nextFundingTime", default)]
    next_funding_time: String,
}

#[derive(Debug, Deserialize)]
struct OrderBookResponse {
    b: Vec<Vec<String>>,
    a: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct TradeResponse {
    list: Vec<TradeData>,
}

#[derive(Debug, Deserialize)]
struct TradeData {
    #[serde(rename = "execId")]
    exec_id: String,
    symbol: String,
    price: String,
    size: String,
    side: String,
    time: String,
}

#[derive(Debug, Deserialize)]
struct OpenInterestResponse {
    list: Vec<OpenInterestData>,
}

#[derive(Debug, Deserialize)]
struct OpenInterestData {
    #[serde(rename = "openInterest")]
    open_interest: String,
    timestamp: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BybitConfig::default();
        assert!(!config.testnet);
        assert!(config.base_url.contains("api.bybit.com"));
    }

    #[test]
    fn test_config_testnet() {
        let config = BybitConfig::testnet();
        assert!(config.testnet);
        assert!(config.base_url.contains("testnet"));
    }

    #[tokio::test]
    async fn test_client_creation() {
        let client = BybitClient::default_client();
        assert!(client.is_ok());
    }
}
