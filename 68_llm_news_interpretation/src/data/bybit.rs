//! Bybit exchange client implementation

use super::{MarketData, MarketDataError, OHLCV, OrderBook, Ticker};
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

/// Bybit API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitConfig {
    /// API key (for authenticated endpoints)
    pub api_key: Option<String>,
    /// API secret (for authenticated endpoints)
    pub api_secret: Option<String>,
    /// Use testnet
    pub testnet: bool,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            api_secret: None,
            testnet: false,
            timeout_ms: 10000,
        }
    }
}

impl BybitConfig {
    /// Create config for testnet
    pub fn testnet() -> Self {
        Self {
            testnet: true,
            ..Default::default()
        }
    }

    /// Create config with API credentials
    pub fn with_credentials(api_key: &str, api_secret: &str) -> Self {
        Self {
            api_key: Some(api_key.to_string()),
            api_secret: Some(api_secret.to_string()),
            ..Default::default()
        }
    }

    /// Get base URL
    pub fn base_url(&self) -> &str {
        if self.testnet {
            "https://api-testnet.bybit.com"
        } else {
            "https://api.bybit.com"
        }
    }
}

/// Bybit exchange client
pub struct BybitClient {
    config: BybitConfig,
    http_client: reqwest::Client,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new(config: BybitConfig) -> Result<Self, MarketDataError> {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(config.timeout_ms))
            .build()
            .map_err(|e| MarketDataError::ConnectionError(e.to_string()))?;

        Ok(Self {
            config,
            http_client,
        })
    }

    /// Create client for mainnet with no authentication
    pub fn mainnet() -> Result<Self, MarketDataError> {
        Self::new(BybitConfig::default())
    }

    /// Create client for testnet with no authentication
    pub fn testnet() -> Result<Self, MarketDataError> {
        Self::new(BybitConfig::testnet())
    }

    /// Get server time
    pub async fn get_server_time(&self) -> Result<DateTime<Utc>, MarketDataError> {
        let url = format!("{}/v5/market/time", self.config.base_url());

        let response: BybitResponse<TimeResponse> = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| MarketDataError::ConnectionError(e.to_string()))?
            .json()
            .await
            .map_err(|e| MarketDataError::ParseError(e.to_string()))?;

        if response.ret_code != 0 {
            return Err(MarketDataError::ApiError(response.ret_msg));
        }

        let timestamp_ms: i64 = response
            .result
            .time_second
            .parse()
            .map_err(|_| MarketDataError::ParseError("Invalid timestamp".to_string()))?;

        Utc.timestamp_opt(timestamp_ms, 0)
            .single()
            .ok_or_else(|| MarketDataError::ParseError("Invalid timestamp".to_string()))
    }

    /// Parse Bybit kline data to OHLCV
    fn parse_kline(&self, kline: &[String]) -> Result<OHLCV, MarketDataError> {
        if kline.len() < 6 {
            return Err(MarketDataError::ParseError(
                "Invalid kline data".to_string(),
            ));
        }

        let timestamp_ms: i64 = kline[0]
            .parse()
            .map_err(|_| MarketDataError::ParseError("Invalid timestamp".to_string()))?;

        let timestamp = Utc
            .timestamp_millis_opt(timestamp_ms)
            .single()
            .ok_or_else(|| MarketDataError::ParseError("Invalid timestamp".to_string()))?;

        Ok(OHLCV {
            timestamp,
            open: kline[1]
                .parse()
                .map_err(|_| MarketDataError::ParseError("Invalid open".to_string()))?,
            high: kline[2]
                .parse()
                .map_err(|_| MarketDataError::ParseError("Invalid high".to_string()))?,
            low: kline[3]
                .parse()
                .map_err(|_| MarketDataError::ParseError("Invalid low".to_string()))?,
            close: kline[4]
                .parse()
                .map_err(|_| MarketDataError::ParseError("Invalid close".to_string()))?,
            volume: kline[5]
                .parse()
                .map_err(|_| MarketDataError::ParseError("Invalid volume".to_string()))?,
        })
    }
}

#[async_trait::async_trait]
impl MarketData for BybitClient {
    async fn get_ticker(&self, symbol: &str) -> Result<Ticker, MarketDataError> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.config.base_url(),
            symbol
        );

        let response: BybitResponse<TickersResponse> = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| MarketDataError::ConnectionError(e.to_string()))?
            .json()
            .await
            .map_err(|e| MarketDataError::ParseError(e.to_string()))?;

        if response.ret_code != 0 {
            return Err(MarketDataError::ApiError(response.ret_msg));
        }

        let ticker_data = response
            .result
            .list
            .first()
            .ok_or_else(|| MarketDataError::SymbolNotFound(symbol.to_string()))?;

        Ok(Ticker {
            symbol: ticker_data.symbol.clone(),
            last_price: ticker_data
                .last_price
                .parse()
                .unwrap_or(0.0),
            bid_price: ticker_data
                .bid1_price
                .parse()
                .unwrap_or(0.0),
            ask_price: ticker_data
                .ask1_price
                .parse()
                .unwrap_or(0.0),
            volume_24h: ticker_data
                .volume24h
                .parse()
                .unwrap_or(0.0),
            change_24h: ticker_data
                .price24h_pcnt
                .parse::<f64>()
                .unwrap_or(0.0)
                * 100.0,
            timestamp: Utc::now(),
        })
    }

    async fn get_ohlcv(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<OHLCV>, MarketDataError> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.config.base_url(),
            symbol,
            interval,
            limit
        );

        let response: BybitResponse<KlineResponse> = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| MarketDataError::ConnectionError(e.to_string()))?
            .json()
            .await
            .map_err(|e| MarketDataError::ParseError(e.to_string()))?;

        if response.ret_code != 0 {
            return Err(MarketDataError::ApiError(response.ret_msg));
        }

        let mut candles = Vec::new();
        for kline in &response.result.list {
            candles.push(self.parse_kline(kline)?);
        }

        // Reverse to get oldest first
        candles.reverse();

        Ok(candles)
    }

    async fn get_orderbook(&self, symbol: &str, depth: usize) -> Result<OrderBook, MarketDataError> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.config.base_url(),
            symbol,
            depth
        );

        let response: BybitResponse<OrderBookResponse> = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| MarketDataError::ConnectionError(e.to_string()))?
            .json()
            .await
            .map_err(|e| MarketDataError::ParseError(e.to_string()))?;

        if response.ret_code != 0 {
            return Err(MarketDataError::ApiError(response.ret_msg));
        }

        let parse_level = |level: &[String]| -> Option<(f64, f64)> {
            if level.len() >= 2 {
                let price = level[0].parse().ok()?;
                let qty = level[1].parse().ok()?;
                Some((price, qty))
            } else {
                None
            }
        };

        let bids: Vec<(f64, f64)> = response
            .result
            .b
            .iter()
            .filter_map(|l| parse_level(l))
            .collect();

        let asks: Vec<(f64, f64)> = response
            .result
            .a
            .iter()
            .filter_map(|l| parse_level(l))
            .collect();

        Ok(OrderBook {
            symbol: symbol.to_string(),
            bids,
            asks,
            timestamp: Utc::now(),
        })
    }

    fn is_connected(&self) -> bool {
        // Simple check - could be enhanced with actual connectivity test
        true
    }
}

// Bybit API response structures

#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

#[derive(Debug, Deserialize)]
struct TimeResponse {
    #[serde(rename = "timeSecond")]
    time_second: String,
}

#[derive(Debug, Deserialize)]
struct TickersResponse {
    list: Vec<TickerData>,
}

#[derive(Debug, Deserialize)]
struct TickerData {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "bid1Price")]
    bid1_price: String,
    #[serde(rename = "ask1Price")]
    ask1_price: String,
    #[serde(rename = "volume24h")]
    volume24h: String,
    #[serde(rename = "price24hPcnt")]
    price24h_pcnt: String,
}

#[derive(Debug, Deserialize)]
struct KlineResponse {
    list: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct OrderBookResponse {
    /// Bids
    b: Vec<Vec<String>>,
    /// Asks
    a: Vec<Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_urls() {
        let mainnet = BybitConfig::default();
        assert_eq!(mainnet.base_url(), "https://api.bybit.com");

        let testnet = BybitConfig::testnet();
        assert_eq!(testnet.base_url(), "https://api-testnet.bybit.com");
    }

    #[test]
    fn test_parse_kline() {
        let client = BybitClient::new(BybitConfig::default()).unwrap();
        let kline_data = vec![
            "1704067200000".to_string(),
            "42000.0".to_string(),
            "42500.0".to_string(),
            "41800.0".to_string(),
            "42300.0".to_string(),
            "100.5".to_string(),
        ];

        let result = client.parse_kline(&kline_data);
        assert!(result.is_ok());

        let ohlcv = result.unwrap();
        assert_eq!(ohlcv.open, 42000.0);
        assert_eq!(ohlcv.high, 42500.0);
        assert_eq!(ohlcv.low, 41800.0);
        assert_eq!(ohlcv.close, 42300.0);
        assert_eq!(ohlcv.volume, 100.5);
    }
}
