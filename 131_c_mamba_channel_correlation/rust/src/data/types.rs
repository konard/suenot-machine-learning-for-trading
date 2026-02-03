//! Data types for market data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV candle data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: Option<f64>,
}

impl Candle {
    /// Calculate typical price (HLC average).
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate range.
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if bullish candle.
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Price series for a single asset.
#[derive(Debug, Clone)]
pub struct PriceSeries {
    pub symbol: String,
    pub interval: String,
    pub candles: Vec<Candle>,
}

impl PriceSeries {
    pub fn new(symbol: String, interval: String) -> Self {
        Self {
            symbol,
            interval,
            candles: Vec::new(),
        }
    }

    pub fn push(&mut self, candle: Candle) {
        self.candles.push(candle);
    }

    pub fn len(&self) -> usize {
        self.candles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }

    pub fn first(&self) -> Option<&Candle> {
        self.candles.first()
    }

    pub fn last(&self) -> Option<&Candle> {
        self.candles.last()
    }

    /// Sort candles by timestamp.
    pub fn sort_by_time(&mut self) {
        self.candles.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    }

    /// Get close prices as vector.
    pub fn close_prices(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Get returns (percentage changes).
    pub fn returns(&self) -> Vec<f64> {
        let closes = self.close_prices();
        closes.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Calculate volatility (standard deviation of returns).
    pub fn volatility(&self) -> f64 {
        let returns = self.returns();
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }
}

/// Multi-asset price data.
#[derive(Debug)]
pub struct MultiAssetData {
    pub symbols: Vec<String>,
    pub series: Vec<PriceSeries>,
}

impl MultiAssetData {
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
            series: Vec::new(),
        }
    }

    pub fn add(&mut self, series: PriceSeries) {
        self.symbols.push(series.symbol.clone());
        self.series.push(series);
    }

    pub fn n_assets(&self) -> usize {
        self.symbols.len()
    }

    /// Convert to price matrix [timesteps, n_assets].
    pub fn to_price_matrix(&self) -> ndarray::Array2<f64> {
        if self.series.is_empty() {
            return ndarray::Array2::zeros((0, 0));
        }

        let n_timesteps = self.series.iter().map(|s| s.len()).min().unwrap_or(0);
        let n_assets = self.n_assets();

        let mut matrix = ndarray::Array2::zeros((n_timesteps, n_assets));

        for (j, series) in self.series.iter().enumerate() {
            for (i, candle) in series.candles.iter().take(n_timesteps).enumerate() {
                matrix[[i, j]] = candle.close;
            }
        }

        matrix
    }
}

impl Default for MultiAssetData {
    fn default() -> Self {
        Self::new()
    }
}
