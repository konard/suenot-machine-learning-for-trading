//! Market data loading module
//!
//! This module provides data loaders for various market data sources.

use crate::error::{Error, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Data source type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataSource {
    /// Stock market data
    Stock,
    /// Cryptocurrency data
    Crypto,
    /// Forex data
    Forex,
}

/// OHLCV (Open, High, Low, Close, Volume) bar
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVBar {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
}

impl OHLCVBar {
    /// Create a new OHLCV bar
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Calculate the bar's return (close / open - 1)
    pub fn return_pct(&self) -> f64 {
        self.close / self.open - 1.0
    }

    /// Check if this is a bullish bar (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate the bar's range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }
}

/// Market data loader
pub struct MarketDataLoader {
    use_mock: bool,
}

impl MarketDataLoader {
    /// Create a new mock data loader
    pub fn new_mock() -> Self {
        Self { use_mock: true }
    }

    /// Load historical data for a symbol
    pub async fn load(
        &self,
        symbol: &str,
        source: DataSource,
        days: usize,
    ) -> Result<Vec<OHLCVBar>> {
        if !self.use_mock {
            return Err(Error::Api("Real data loading not implemented".into()));
        }

        Ok(self.generate_mock_data(symbol, source, days))
    }

    /// Generate mock OHLCV data
    fn generate_mock_data(
        &self,
        _symbol: &str,
        source: DataSource,
        days: usize,
    ) -> Vec<OHLCVBar> {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let mut bars = Vec::with_capacity(days);

        // Base price depends on asset type
        let mut price = match source {
            DataSource::Stock => 150.0,
            DataSource::Crypto => 45000.0,
            DataSource::Forex => 1.10,
        };

        let volatility = match source {
            DataSource::Stock => 0.02,
            DataSource::Crypto => 0.04,
            DataSource::Forex => 0.005,
        };

        let base_volume = match source {
            DataSource::Stock => 10_000_000.0,
            DataSource::Crypto => 1_000_000_000.0,
            DataSource::Forex => 100_000_000.0,
        };

        let now = Utc::now();

        for i in 0..days {
            let timestamp = now - chrono::Duration::days((days - i - 1) as i64);

            // Random walk with drift
            let change = rng.gen_range(-volatility..volatility);
            let drift = 0.0001; // Slight upward bias
            price *= 1.0 + change + drift;

            let open = price * (1.0 + rng.gen_range(-0.005..0.005));
            let close = price * (1.0 + rng.gen_range(-0.005..0.005));
            let high = price * (1.0 + rng.gen_range(0.0..volatility));
            let low = price * (1.0 - rng.gen_range(0.0..volatility));
            let volume = base_volume * rng.gen_range(0.5..2.0);

            bars.push(OHLCVBar::new(
                timestamp,
                open,
                high.max(open).max(close),
                low.min(open).min(close),
                close,
                volume,
            ));
        }

        bars
    }
}

/// News data item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsData {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// News headline
    pub headline: String,
    /// News source
    pub source: String,
    /// Related symbols
    pub symbols: Vec<String>,
}

/// News data loader
pub struct NewsDataLoader {
    use_mock: bool,
}

impl NewsDataLoader {
    /// Create a new mock news loader
    pub fn new_mock() -> Self {
        Self { use_mock: true }
    }

    /// Load news for a symbol
    pub async fn load(&self, symbol: &str, days: usize) -> Result<Vec<NewsData>> {
        if !self.use_mock {
            return Err(Error::Api("Real news loading not implemented".into()));
        }

        Ok(self.generate_mock_news(symbol, days))
    }

    /// Generate mock news data
    fn generate_mock_news(&self, symbol: &str, days: usize) -> Vec<NewsData> {
        use rand::Rng;

        let positive_headlines = [
            "reports record quarterly revenue, beating expectations",
            "announces major partnership deal",
            "stock surges on strong earnings",
            "receives analyst upgrade to buy rating",
            "expands into new markets with strong demand",
        ];

        let negative_headlines = [
            "misses earnings expectations, shares fall",
            "announces layoffs amid restructuring",
            "faces regulatory investigation",
            "warns of slowing demand in key markets",
            "stock drops on disappointing guidance",
        ];

        let neutral_headlines = [
            "schedules investor day next month",
            "appoints new board member",
            "maintains quarterly dividend",
            "announces conference participation",
            "files routine SEC documents",
        ];

        let mut rng = rand::thread_rng();
        let mut news = Vec::new();
        let now = Utc::now();
        let sources = ["Reuters", "Bloomberg", "CNBC", "WSJ"];

        // Generate 1-3 news items per day
        for day in 0..days {
            let num_items = rng.gen_range(0..4);

            for _ in 0..num_items {
                let timestamp = now
                    - chrono::Duration::days((days - day - 1) as i64)
                    + chrono::Duration::hours(rng.gen_range(0..24));

                let headlines = match rng.gen_range(0..10) {
                    0..=4 => &positive_headlines[..],
                    5..=7 => &negative_headlines[..],
                    _ => &neutral_headlines[..],
                };

                let headline_template = headlines[rng.gen_range(0..headlines.len())];
                let headline = format!("{} {}", symbol, headline_template);
                let source = sources[rng.gen_range(0..sources.len())].to_string();

                news.push(NewsData {
                    timestamp,
                    headline,
                    source,
                    symbols: vec![symbol.to_string()],
                });
            }
        }

        news.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        news
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_load_stock_data() {
        let loader = MarketDataLoader::new_mock();
        let data = loader.load("AAPL", DataSource::Stock, 30).await.unwrap();
        assert_eq!(data.len(), 30);
    }

    #[tokio::test]
    async fn test_load_crypto_data() {
        let loader = MarketDataLoader::new_mock();
        let data = loader.load("BTCUSDT", DataSource::Crypto, 30).await.unwrap();
        assert_eq!(data.len(), 30);
        // Crypto prices should be higher
        assert!(data[0].close > 1000.0);
    }
}
