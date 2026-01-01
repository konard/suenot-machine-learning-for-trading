//! Bybit Exchange API Client
//!
//! Provides access to Bybit's public API for market data.

use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::{Candle, DataError, OrderBook, OrderBookLevel, Ticker, Timeframe, Trade};

/// Bybit API base URLs
const BYBIT_MAINNET: &str = "https://api.bybit.com";
const BYBIT_TESTNET: &str = "https://api-testnet.bybit.com";

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    /// HTTP client
    client: Client,
    /// Base URL
    base_url: String,
    /// Request timeout
    timeout: Duration,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new Bybit client for mainnet
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: BYBIT_MAINNET.to_string(),
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
            base_url: BYBIT_TESTNET.to_string(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Set custom timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self.client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to create HTTP client");
        self
    }

    /// Fetch kline/candlestick data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Timeframe (e.g., "1", "5", "15", "60", "D")
    /// * `limit` - Number of candles (max 1000)
    ///
    /// # Returns
    /// Vector of candles, oldest first
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>, DataError> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol,
            interval,
            limit.min(1000)
        );

        let response: BybitResponse<KlineResponse> = self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(DataError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let timeframe = Timeframe::from_bybit_interval(interval)
            .ok_or_else(|| DataError::InvalidTimeframe(interval.to_string()))?;

        let mut candles: Vec<Candle> = response
            .result
            .list
            .into_iter()
            .map(|k| k.to_candle(symbol, timeframe))
            .collect();

        // API returns newest first, reverse to get oldest first
        candles.reverse();

        Ok(candles)
    }

    /// Fetch klines for a specific time range
    pub async fn get_klines_range(
        &self,
        symbol: &str,
        interval: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Candle>, DataError> {
        let start_ms = start.timestamp_millis();
        let end_ms = end.timestamp_millis();

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&start={}&end={}&limit=1000",
            self.base_url, symbol, interval, start_ms, end_ms
        );

        let response: BybitResponse<KlineResponse> = self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(DataError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let timeframe = Timeframe::from_bybit_interval(interval)
            .ok_or_else(|| DataError::InvalidTimeframe(interval.to_string()))?;

        let mut candles: Vec<Candle> = response
            .result
            .list
            .into_iter()
            .map(|k| k.to_candle(symbol, timeframe))
            .collect();

        candles.reverse();

        Ok(candles)
    }

    /// Fetch order book
    ///
    /// # Arguments
    /// * `symbol` - Trading pair
    /// * `limit` - Depth (1, 25, 50, 100, 200)
    pub async fn get_orderbook(&self, symbol: &str, limit: usize) -> Result<OrderBook, DataError> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let response: BybitResponse<OrderBookResponse> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(DataError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response.result;

        let bids: Vec<OrderBookLevel> = result
            .b
            .into_iter()
            .map(|level| OrderBookLevel {
                price: level[0].parse().unwrap_or(0.0),
                quantity: level[1].parse().unwrap_or(0.0),
            })
            .collect();

        let asks: Vec<OrderBookLevel> = result
            .a
            .into_iter()
            .map(|level| OrderBookLevel {
                price: level[0].parse().unwrap_or(0.0),
                quantity: level[1].parse().unwrap_or(0.0),
            })
            .collect();

        let timestamp = Utc
            .timestamp_millis_opt(result.ts.parse().unwrap_or(0))
            .single()
            .unwrap_or_else(Utc::now);

        Ok(OrderBook {
            symbol: symbol.to_string(),
            timestamp,
            bids,
            asks,
        })
    }

    /// Fetch recent trades
    pub async fn get_trades(&self, symbol: &str, limit: usize) -> Result<Vec<Trade>, DataError> {
        let url = format!(
            "{}/v5/market/recent-trade?category=spot&symbol={}&limit={}",
            self.base_url,
            symbol,
            limit.min(1000)
        );

        let response: BybitResponse<TradesResponse> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(DataError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let trades: Vec<Trade> = response
            .result
            .list
            .into_iter()
            .map(|t| Trade {
                id: t.exec_id,
                symbol: symbol.to_string(),
                price: t.price.parse().unwrap_or(0.0),
                quantity: t.size.parse().unwrap_or(0.0),
                timestamp: Utc
                    .timestamp_millis_opt(t.time.parse().unwrap_or(0))
                    .single()
                    .unwrap_or_else(Utc::now),
                is_buyer_maker: t.side == "Sell",
            })
            .collect();

        Ok(trades)
    }

    /// Fetch ticker information
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker, DataError> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        let response: BybitResponse<TickersResponse> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(DataError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let ticker = response
            .result
            .list
            .into_iter()
            .next()
            .ok_or(DataError::NoData)?;

        Ok(Ticker {
            symbol: ticker.symbol,
            last_price: ticker.last_price.parse().unwrap_or(0.0),
            price_change_pct: ticker.price_24h_pcnt.parse().unwrap_or(0.0) * 100.0,
            high_24h: ticker.high_price_24h.parse().unwrap_or(0.0),
            low_24h: ticker.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: ticker.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: ticker.turnover_24h.parse().unwrap_or(0.0),
        })
    }

    /// Get available trading pairs
    pub async fn get_symbols(&self) -> Result<Vec<String>, DataError> {
        let url = format!("{}/v5/market/instruments-info?category=spot", self.base_url);

        let response: BybitResponse<InstrumentsResponse> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(DataError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let symbols: Vec<String> = response
            .result
            .list
            .into_iter()
            .filter(|i| i.status == "Trading")
            .map(|i| i.symbol)
            .collect();

        Ok(symbols)
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
struct KlineResponse {
    list: Vec<KlineData>,
}

#[derive(Debug, Deserialize)]
struct KlineData {
    #[serde(rename = "0")]
    start_time: String,
    #[serde(rename = "1")]
    open: String,
    #[serde(rename = "2")]
    high: String,
    #[serde(rename = "3")]
    low: String,
    #[serde(rename = "4")]
    close: String,
    #[serde(rename = "5")]
    volume: String,
    #[serde(rename = "6")]
    turnover: String,
}

impl KlineData {
    fn to_candle(&self, symbol: &str, timeframe: Timeframe) -> Candle {
        let timestamp_ms: i64 = self.start_time.parse().unwrap_or(0);
        let timestamp = Utc
            .timestamp_millis_opt(timestamp_ms)
            .single()
            .unwrap_or_else(Utc::now);

        Candle {
            symbol: symbol.to_string(),
            timestamp,
            timeframe,
            open: self.open.parse().unwrap_or(0.0),
            high: self.high.parse().unwrap_or(0.0),
            low: self.low.parse().unwrap_or(0.0),
            close: self.close.parse().unwrap_or(0.0),
            volume: self.volume.parse().unwrap_or(0.0),
        }
    }
}

#[derive(Debug, Deserialize)]
struct OrderBookResponse {
    s: String,
    b: Vec<Vec<String>>,
    a: Vec<Vec<String>>,
    ts: String,
}

#[derive(Debug, Deserialize)]
struct TradesResponse {
    list: Vec<TradeData>,
}

#[derive(Debug, Deserialize)]
struct TradeData {
    #[serde(rename = "execId")]
    exec_id: String,
    price: String,
    size: String,
    side: String,
    time: String,
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
    #[serde(rename = "price24hPcnt")]
    price_24h_pcnt: String,
    #[serde(rename = "highPrice24h")]
    high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    low_price_24h: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "turnover24h")]
    turnover_24h: String,
}

#[derive(Debug, Deserialize)]
struct InstrumentsResponse {
    list: Vec<InstrumentData>,
}

#[derive(Debug, Deserialize)]
struct InstrumentData {
    symbol: String,
    status: String,
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
}
