//! Bybit API client implementation

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;

use super::types::*;

/// Bybit API client
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
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create a client with custom base URL (for testnet)
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
        }
    }

    /// Create a testnet client
    pub fn testnet() -> Self {
        Self::with_base_url("https://api-testnet.bybit.com")
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Time interval
    /// * `limit` - Number of candles to fetch (max 1000)
    /// * `start` - Start timestamp (optional)
    /// * `end` - End timestamp (optional)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: TimeFrame,
        limit: Option<u32>,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<MarketData> {
        let mut url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}",
            self.base_url,
            symbol,
            interval.as_str()
        );

        if let Some(l) = limit {
            url.push_str(&format!("&limit={}", l.min(1000)));
        }

        if let Some(s) = start {
            url.push_str(&format!("&start={}", s.timestamp_millis()));
        }

        if let Some(e) = end {
            url.push_str(&format!("&end={}", e.timestamp_millis()));
        }

        let response: BybitResponse<KlineResult> = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send request")?
            .json()
            .await
            .context("Failed to parse response")?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {} - {}", response.ret_code, response.ret_msg);
        }

        let candles = response
            .result
            .list
            .into_iter()
            .filter_map(|item| {
                if item.len() < 7 {
                    return None;
                }

                let timestamp_ms: i64 = item[0].parse().ok()?;
                let timestamp = Utc.timestamp_millis_opt(timestamp_ms).single()?;

                Some(Candle::new(
                    timestamp,
                    item[1].parse().ok()?,
                    item[2].parse().ok()?,
                    item[3].parse().ok()?,
                    item[4].parse().ok()?,
                    item[5].parse().ok()?,
                    item[6].parse().ok()?,
                ))
            })
            .collect::<Vec<_>>();

        // Bybit returns newest first, we want oldest first
        let mut candles = candles;
        candles.reverse();

        let mut data = MarketData::new(symbol, candles);
        data.timeframe = Some(interval);

        Ok(data)
    }

    /// Fetch order book snapshot
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol
    /// * `limit` - Depth limit (1, 25, 50, 100, 200)
    pub async fn get_orderbook(
        &self,
        symbol: &str,
        limit: Option<u32>,
    ) -> Result<OrderBookSnapshot> {
        let limit = limit.unwrap_or(50);
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let response: BybitResponse<OrderBookResult> = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send request")?
            .json()
            .await
            .context("Failed to parse response")?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {} - {}", response.ret_code, response.ret_msg);
        }

        let result = response.result;

        let bids = result
            .b
            .into_iter()
            .filter_map(|item| {
                if item.len() < 2 {
                    return None;
                }
                Some((item[0].parse().ok()?, item[1].parse().ok()?))
            })
            .collect();

        let asks = result
            .a
            .into_iter()
            .filter_map(|item| {
                if item.len() < 2 {
                    return None;
                }
                Some((item[0].parse().ok()?, item[1].parse().ok()?))
            })
            .collect();

        let timestamp = Utc.timestamp_millis_opt(result.ts).single()
            .unwrap_or_else(Utc::now);

        Ok(OrderBookSnapshot {
            timestamp,
            symbol: symbol.to_string(),
            bids,
            asks,
        })
    }

    /// Fetch ticker information
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        let response: BybitResponse<TickerResult> = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send request")?
            .json()
            .await
            .context("Failed to parse response")?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {} - {}", response.ret_code, response.ret_msg);
        }

        let item = response
            .result
            .list
            .into_iter()
            .next()
            .context("No ticker data returned")?;

        Ok(Ticker {
            symbol: item.symbol,
            last_price: item.last_price.parse().unwrap_or(0.0),
            bid_price: item.bid1_price.parse().unwrap_or(0.0),
            ask_price: item.ask1_price.parse().unwrap_or(0.0),
            volume_24h: item.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: item.turnover_24h.parse().unwrap_or(0.0),
            high_24h: item.high_price_24h.parse().unwrap_or(0.0),
            low_24h: item.low_price_24h.parse().unwrap_or(0.0),
            price_change_24h: 0.0, // Calculated below
            price_change_pct_24h: item.price_24h_pcnt.parse().unwrap_or(0.0) * 100.0,
        })
    }

    /// Fetch multiple timeframes of kline data
    pub async fn get_multi_timeframe(
        &self,
        symbol: &str,
        timeframes: &[TimeFrame],
        limit: u32,
    ) -> Result<Vec<MarketData>> {
        let mut results = Vec::new();

        for tf in timeframes {
            let data = self.get_klines(symbol, *tf, Some(limit), None, None).await?;
            results.push(data);
        }

        Ok(results)
    }

    /// Fetch historical data with pagination
    pub async fn get_historical_klines(
        &self,
        symbol: &str,
        interval: TimeFrame,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<MarketData> {
        let mut all_candles = Vec::new();
        let mut current_end = end;

        while current_end > start {
            let data = self
                .get_klines(symbol, interval, Some(1000), Some(start), Some(current_end))
                .await?;

            if data.candles.is_empty() {
                break;
            }

            let oldest_timestamp = data.candles.first().map(|c| c.timestamp).unwrap();

            // Prepend candles (they're already in chronological order)
            let mut new_candles = data.candles;
            new_candles.append(&mut all_candles);
            all_candles = new_candles;

            // Move end to before the oldest candle we got
            current_end = oldest_timestamp - chrono::Duration::milliseconds(1);

            // Avoid rate limiting
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        let mut data = MarketData::new(symbol, all_candles);
        data.timeframe = Some(interval);

        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = BybitClient::new();
        assert!(client.base_url.contains("bybit.com"));

        let testnet = BybitClient::testnet();
        assert!(testnet.base_url.contains("testnet"));
    }

    // Integration tests require network access
    #[tokio::test]
    #[ignore = "requires network access"]
    async fn test_get_klines() {
        let client = BybitClient::new();
        let data = client
            .get_klines("BTCUSDT", TimeFrame::Hour1, Some(10), None, None)
            .await
            .unwrap();

        assert!(!data.candles.is_empty());
        assert!(data.candles.len() <= 10);
    }

    #[tokio::test]
    #[ignore = "requires network access"]
    async fn test_get_orderbook() {
        let client = BybitClient::new();
        let orderbook = client.get_orderbook("BTCUSDT", Some(25)).await.unwrap();

        assert!(!orderbook.bids.is_empty());
        assert!(!orderbook.asks.is_empty());
        assert!(orderbook.best_bid().unwrap() < orderbook.best_ask().unwrap());
    }

    #[tokio::test]
    #[ignore = "requires network access"]
    async fn test_get_ticker() {
        let client = BybitClient::new();
        let ticker = client.get_ticker("BTCUSDT").await.unwrap();

        assert_eq!(ticker.symbol, "BTCUSDT");
        assert!(ticker.last_price > 0.0);
    }
}
