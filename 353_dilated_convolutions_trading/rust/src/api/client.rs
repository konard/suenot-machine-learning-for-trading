//! Bybit API client

use reqwest::Client;
use tracing::{debug, info};

use super::error::ApiError;
use super::types::*;
use super::{Category, Interval, BYBIT_API_URL, BYBIT_TESTNET_URL};

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    /// HTTP client
    client: Client,
    /// Base URL
    base_url: String,
    /// Default category
    default_category: Category,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new client with default settings
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: BYBIT_API_URL.to_string(),
            default_category: Category::Linear,
        }
    }

    /// Create a client for testnet
    pub fn testnet() -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: BYBIT_TESTNET_URL.to_string(),
            default_category: Category::Linear,
        }
    }

    /// Set the base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the default category
    pub fn with_category(mut self, category: Category) -> Self {
        self.default_category = category;
        self
    }

    /// Get klines (candlesticks)
    ///
    /// # Arguments
    /// - `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// - `interval` - Candle interval
    /// - `limit` - Number of candles (max 1000)
    ///
    /// # Example
    /// ```rust,no_run
    /// use dilated_conv_trading::{BybitClient, api::Interval};
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let client = BybitClient::new();
    ///     let klines = client.get_klines_with_interval("BTCUSDT", Interval::Hour1, 100).await?;
    ///     println!("Last candle: close = {}", klines.last().unwrap().close);
    ///     Ok(())
    /// }
    /// ```
    pub async fn get_klines_with_interval(
        &self,
        symbol: &str,
        interval: Interval,
        limit: u32,
    ) -> Result<Vec<Kline>, ApiError> {
        self.get_klines(symbol, interval.as_str(), limit).await
    }

    /// Get klines with interval as string
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>, ApiError> {
        let limit = limit.min(1000);

        let url = format!("{}/v5/market/kline", self.base_url);

        debug!(symbol, interval, limit, "Fetching klines");

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", self.default_category.as_str()),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let data: BybitResponse<KlineResult> = response.json().await?;

        if !data.is_ok() {
            return Err(ApiError::bybit_error(data.ret_code, data.ret_msg));
        }

        // Bybit returns klines in reverse order (newest first)
        let mut klines: Vec<Kline> = data
            .result
            .list
            .into_iter()
            .map(|k| {
                Kline::new(
                    k.timestamp(),
                    symbol,
                    interval,
                    k.open(),
                    k.high(),
                    k.low(),
                    k.close(),
                    k.volume(),
                    k.turnover(),
                )
            })
            .collect();

        // Sort by time (oldest first)
        klines.reverse();

        info!(symbol, count = klines.len(), "Fetched klines");

        Ok(klines)
    }

    /// Get klines for a specific time range
    pub async fn get_klines_range(
        &self,
        symbol: &str,
        interval: Interval,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<Kline>, ApiError> {
        let url = format!("{}/v5/market/kline", self.base_url);

        debug!(symbol, ?interval, start_time, end_time, "Fetching klines range");

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", self.default_category.as_str()),
                ("symbol", symbol),
                ("interval", interval.as_str()),
                ("start", &start_time.to_string()),
                ("end", &end_time.to_string()),
                ("limit", "1000"),
            ])
            .send()
            .await?;

        let data: BybitResponse<KlineResult> = response.json().await?;

        if !data.is_ok() {
            return Err(ApiError::bybit_error(data.ret_code, data.ret_msg));
        }

        let mut klines: Vec<Kline> = data
            .result
            .list
            .into_iter()
            .map(|k| {
                Kline::new(
                    k.timestamp(),
                    symbol,
                    interval.as_str(),
                    k.open(),
                    k.high(),
                    k.low(),
                    k.close(),
                    k.volume(),
                    k.turnover(),
                )
            })
            .collect();

        klines.reverse();

        Ok(klines)
    }

    /// Get ticker for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker, ApiError> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        debug!(symbol, "Fetching ticker");

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", self.default_category.as_str()),
                ("symbol", symbol),
            ])
            .send()
            .await?;

        let data: BybitResponse<TickerResult> = response.json().await?;

        if !data.is_ok() {
            return Err(ApiError::bybit_error(data.ret_code, data.ret_msg));
        }

        let ticker_data = data
            .result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| ApiError::no_data(symbol))?;

        Ok(Ticker {
            symbol: ticker_data.symbol,
            last_price: ticker_data.last_price.parse().unwrap_or(0.0),
            bid_price: ticker_data.bid1_price.parse().unwrap_or(0.0),
            bid_size: ticker_data.bid1_size.parse().unwrap_or(0.0),
            ask_price: ticker_data.ask1_price.parse().unwrap_or(0.0),
            ask_size: ticker_data.ask1_size.parse().unwrap_or(0.0),
            price_change_24h: 0.0,
            price_change_percent_24h: ticker_data.price_24h_pcnt.parse().unwrap_or(0.0) * 100.0,
            high_24h: ticker_data.high_price_24h.parse().unwrap_or(0.0),
            low_24h: ticker_data.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: ticker_data.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: ticker_data.turnover_24h.parse().unwrap_or(0.0),
            timestamp: data.time,
        })
    }

    /// Get order book
    pub async fn get_orderbook(&self, symbol: &str, depth: u32) -> Result<OrderBook, ApiError> {
        let depth = depth.min(500);
        let url = format!("{}/v5/market/orderbook", self.base_url);

        debug!(symbol, depth, "Fetching orderbook");

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", self.default_category.as_str()),
                ("symbol", symbol),
                ("limit", &depth.to_string()),
            ])
            .send()
            .await?;

        let data: BybitResponse<OrderBookResult> = response.json().await?;

        if !data.is_ok() {
            return Err(ApiError::bybit_error(data.ret_code, data.ret_msg));
        }

        let result = data.result;

        let bids: Vec<OrderBookLevel> = result
            .bids
            .into_iter()
            .map(|[price, size]| {
                OrderBookLevel::new(
                    price.parse().unwrap_or(0.0),
                    size.parse().unwrap_or(0.0),
                )
            })
            .collect();

        let asks: Vec<OrderBookLevel> = result
            .asks
            .into_iter()
            .map(|[price, size]| {
                OrderBookLevel::new(
                    price.parse().unwrap_or(0.0),
                    size.parse().unwrap_or(0.0),
                )
            })
            .collect();

        Ok(OrderBook::new(symbol, bids, asks, result.timestamp))
    }

    /// Get server time
    pub async fn get_server_time(&self) -> Result<i64, ApiError> {
        let url = format!("{}/v5/market/time", self.base_url);

        let response = self.client.get(&url).send().await?;

        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct TimeResponse {
            time_second: String,
            #[allow(dead_code)]
            time_nano: String,
        }

        let data: BybitResponse<TimeResponse> = response.json().await?;

        if !data.is_ok() {
            return Err(ApiError::bybit_error(data.ret_code, data.ret_msg));
        }

        let time_ms = data.result.time_second.parse::<i64>().unwrap_or(0) * 1000;

        Ok(time_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, BYBIT_API_URL);
    }

    #[test]
    fn test_testnet_client() {
        let client = BybitClient::testnet();
        assert_eq!(client.base_url, BYBIT_TESTNET_URL);
    }

    #[test]
    fn test_with_category() {
        let client = BybitClient::new().with_category(Category::Spot);
        assert_eq!(client.default_category, Category::Spot);
    }
}
