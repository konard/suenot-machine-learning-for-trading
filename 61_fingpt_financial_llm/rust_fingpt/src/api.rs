//! API clients for external services
//!
//! This module provides clients for interacting with external APIs,
//! including cryptocurrency exchanges like Bybit.

use crate::error::{Error, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Bybit ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitTicker {
    /// Symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: String,
    /// 24h price change percentage
    pub price_24h_pcnt: String,
    /// 24h trading volume
    pub volume_24h: String,
    /// Best bid price
    pub bid_price: String,
    /// Best ask price
    pub ask_price: String,
}

/// Bybit kline/candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitKline {
    /// Start timestamp
    pub start_time: DateTime<Utc>,
    /// Open price
    pub open: String,
    /// High price
    pub high: String,
    /// Low price
    pub low: String,
    /// Close price
    pub close: String,
    /// Volume
    pub volume: String,
}

/// Bybit orderbook level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookLevel {
    /// Price
    pub price: String,
    /// Quantity
    pub qty: String,
}

/// Bybit orderbook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitOrderbook {
    /// Symbol
    pub symbol: String,
    /// Bid levels
    pub bids: Vec<OrderbookLevel>,
    /// Ask levels
    pub asks: Vec<OrderbookLevel>,
    /// Update timestamp
    pub timestamp: DateTime<Utc>,
}

/// Bybit API client
pub struct BybitClient {
    use_mock: bool,
    #[allow(dead_code)]
    api_key: Option<String>,
    #[allow(dead_code)]
    api_secret: Option<String>,
    #[allow(dead_code)]
    base_url: String,
}

impl BybitClient {
    /// Create a new mock Bybit client for demos
    pub fn new_mock() -> Self {
        Self {
            use_mock: true,
            api_key: None,
            api_secret: None,
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create a new Bybit client with API credentials
    pub fn new(api_key: String, api_secret: String, testnet: bool) -> Self {
        let base_url = if testnet {
            "https://api-testnet.bybit.com".to_string()
        } else {
            "https://api.bybit.com".to_string()
        };

        Self {
            use_mock: false,
            api_key: Some(api_key),
            api_secret: Some(api_secret),
            base_url,
        }
    }

    /// Get ticker data for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<BybitTicker> {
        if self.use_mock {
            return Ok(self.mock_ticker(symbol));
        }

        Err(Error::Api("Real API not implemented. Use new_mock() for demos.".into()))
    }

    /// Get kline/candlestick data
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<BybitKline>> {
        if self.use_mock {
            return Ok(self.mock_klines(symbol, interval, limit));
        }

        Err(Error::Api("Real API not implemented. Use new_mock() for demos.".into()))
    }

    /// Get orderbook data
    pub async fn get_orderbook(&self, symbol: &str, depth: usize) -> Result<BybitOrderbook> {
        if self.use_mock {
            return Ok(self.mock_orderbook(symbol, depth));
        }

        Err(Error::Api("Real API not implemented. Use new_mock() for demos.".into()))
    }

    /// Generate mock ticker data
    fn mock_ticker(&self, symbol: &str) -> BybitTicker {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let base_price = match symbol {
            s if s.starts_with("BTC") => 45000.0 + rng.gen_range(-2000.0..2000.0),
            s if s.starts_with("ETH") => 2500.0 + rng.gen_range(-100.0..100.0),
            s if s.starts_with("SOL") => 100.0 + rng.gen_range(-10.0..10.0),
            _ => 100.0 + rng.gen_range(-5.0..5.0),
        };

        let spread = base_price * 0.0001;

        BybitTicker {
            symbol: symbol.to_string(),
            last_price: format!("{:.2}", base_price),
            price_24h_pcnt: format!("{:.4}", rng.gen_range(-0.05..0.05)),
            volume_24h: format!("{:.0}", rng.gen_range(100_000_000.0..500_000_000.0)),
            bid_price: format!("{:.2}", base_price - spread),
            ask_price: format!("{:.2}", base_price + spread),
        }
    }

    /// Generate mock kline data
    fn mock_klines(&self, symbol: &str, _interval: &str, limit: usize) -> Vec<BybitKline> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut price = match symbol {
            s if s.starts_with("BTC") => 45000.0,
            s if s.starts_with("ETH") => 2500.0,
            s if s.starts_with("SOL") => 100.0,
            _ => 100.0,
        };

        let volatility = match symbol {
            s if s.starts_with("BTC") => 0.02,
            s if s.starts_with("ETH") => 0.025,
            _ => 0.03,
        };

        let now = Utc::now();
        let mut klines = Vec::with_capacity(limit);

        for i in 0..limit {
            let start_time = now - chrono::Duration::hours((limit - i) as i64);

            let change = rng.gen_range(-volatility..volatility);
            price *= 1.0 + change;

            let open: f64 = price * (1.0 + rng.gen_range(-0.005..0.005));
            let close: f64 = price * (1.0 + rng.gen_range(-0.005..0.005));
            let high: f64 = price * (1.0 + rng.gen_range(0.0..volatility));
            let low: f64 = price * (1.0 - rng.gen_range(0.0..volatility));
            let volume = rng.gen_range(1000.0..10000.0);

            klines.push(BybitKline {
                start_time,
                open: format!("{:.2}", open),
                high: format!("{:.2}", high.max(open).max(close)),
                low: format!("{:.2}", low.min(open).min(close)),
                close: format!("{:.2}", close),
                volume: format!("{:.2}", volume),
            });
        }

        klines
    }

    /// Generate mock orderbook data
    fn mock_orderbook(&self, symbol: &str, depth: usize) -> BybitOrderbook {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mid_price = match symbol {
            s if s.starts_with("BTC") => 45000.0,
            s if s.starts_with("ETH") => 2500.0,
            s if s.starts_with("SOL") => 100.0,
            _ => 100.0,
        };

        let tick_size = mid_price * 0.0001;
        let mut bids = Vec::with_capacity(depth);
        let mut asks = Vec::with_capacity(depth);

        for i in 0..depth {
            let bid_price = mid_price - (i as f64 + 1.0) * tick_size;
            let ask_price = mid_price + (i as f64 + 1.0) * tick_size;
            let qty = rng.gen_range(0.1..10.0);

            bids.push(OrderbookLevel {
                price: format!("{:.2}", bid_price),
                qty: format!("{:.4}", qty),
            });

            asks.push(OrderbookLevel {
                price: format!("{:.2}", ask_price),
                qty: format!("{:.4}", qty),
            });
        }

        BybitOrderbook {
            symbol: symbol.to_string(),
            bids,
            asks,
            timestamp: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_ticker() {
        let client = BybitClient::new_mock();
        let ticker = client.get_ticker("BTCUSDT").await.unwrap();
        assert_eq!(ticker.symbol, "BTCUSDT");
        assert!(!ticker.last_price.is_empty());
    }

    #[tokio::test]
    async fn test_mock_klines() {
        let client = BybitClient::new_mock();
        let klines = client.get_klines("ETHUSDT", "1h", 24).await.unwrap();
        assert_eq!(klines.len(), 24);
    }
}
