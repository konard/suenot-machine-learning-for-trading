//! Market data handling module.
//!
//! Provides data structures and utilities for loading and processing market data.

use chrono::{DateTime, Duration, NaiveDate, Utc};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, TradingError};

/// OHLCV candlestick data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    /// Create a new candle.
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

    /// Calculate the range (high - low).
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if the candle is bullish (close > open).
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate body size.
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }
}

/// Container for market data.
#[derive(Debug, Clone)]
pub struct MarketData {
    pub symbol: String,
    pub candles: Vec<Candle>,
    pub source: String,
    pub metadata: HashMap<String, String>,
}

impl MarketData {
    /// Create new market data container.
    pub fn new(symbol: &str, candles: Vec<Candle>, source: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            candles,
            source: source.to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Get the number of candles.
    pub fn len(&self) -> usize {
        self.candles.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }

    /// Get close prices as a vector.
    pub fn close_prices(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Get the latest close price.
    pub fn latest_close(&self) -> Option<f64> {
        self.candles.last().map(|c| c.close)
    }

    /// Get a slice of the most recent candles.
    pub fn tail(&self, n: usize) -> &[Candle] {
        let start = self.candles.len().saturating_sub(n);
        &self.candles[start..]
    }

    /// Calculate returns.
    pub fn returns(&self) -> Vec<f64> {
        self.candles
            .windows(2)
            .map(|w| w[1].close / w[0].close - 1.0)
            .collect()
    }

    /// Calculate log returns.
    pub fn log_returns(&self) -> Vec<f64> {
        self.candles
            .windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect()
    }

    /// Calculate simple moving average.
    pub fn sma(&self, period: usize) -> Vec<f64> {
        let closes = self.close_prices();
        if closes.len() < period {
            return vec![];
        }

        closes
            .windows(period)
            .map(|w| w.iter().sum::<f64>() / period as f64)
            .collect()
    }

    /// Calculate exponential moving average.
    pub fn ema(&self, period: usize) -> Vec<f64> {
        let closes = self.close_prices();
        if closes.is_empty() {
            return vec![];
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = vec![closes[0]];

        for i in 1..closes.len() {
            let new_ema = (closes[i] - ema[i - 1]) * multiplier + ema[i - 1];
            ema.push(new_ema);
        }

        ema
    }

    /// Calculate RSI (Relative Strength Index).
    pub fn rsi(&self, period: usize) -> Vec<f64> {
        let returns = self.returns();
        if returns.len() < period {
            return vec![];
        }

        let mut gains: Vec<f64> = returns.iter().map(|&r| r.max(0.0)).collect();
        let mut losses: Vec<f64> = returns.iter().map(|&r| (-r).max(0.0)).collect();

        let mut rsi_values = vec![];

        // Simple moving average for initial values
        for i in period..=returns.len() {
            let avg_gain: f64 = gains[i - period..i].iter().sum::<f64>() / period as f64;
            let avg_loss: f64 = losses[i - period..i].iter().sum::<f64>() / period as f64;

            let rs = if avg_loss == 0.0 {
                100.0
            } else {
                avg_gain / avg_loss
            };

            let rsi = 100.0 - (100.0 / (1.0 + rs));
            rsi_values.push(rsi);
        }

        rsi_values
    }

    /// Calculate volatility (standard deviation of returns).
    pub fn volatility(&self, period: usize) -> Vec<f64> {
        let returns = self.returns();
        if returns.len() < period {
            return vec![];
        }

        returns
            .windows(period)
            .map(|w| {
                let mean = w.iter().sum::<f64>() / period as f64;
                let variance = w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / period as f64;
                variance.sqrt() * (252.0_f64).sqrt() // Annualized
            })
            .collect()
    }

    /// Calculate maximum drawdown.
    pub fn max_drawdown(&self) -> f64 {
        let closes = self.close_prices();
        if closes.is_empty() {
            return 0.0;
        }

        let mut peak = closes[0];
        let mut max_dd = 0.0;

        for &price in &closes {
            if price > peak {
                peak = price;
            }
            let drawdown = (peak - price) / peak;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        max_dd
    }
}

/// Create mock market data for testing.
pub fn create_mock_data(symbol: &str, days: usize, initial_price: f64) -> MarketData {
    let mut rng = rand::thread_rng();
    let start_date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

    let mut candles = Vec::with_capacity(days);
    let mut price = initial_price;

    for i in 0..days {
        let date = start_date + Duration::days(i as i64);
        let timestamp = date.and_hms_opt(0, 0, 0).unwrap().and_utc();

        // Random walk with slight upward drift
        let change = rng.gen_range(-0.03..0.035);
        price *= 1.0 + change;

        let open = price * (1.0 + rng.gen_range(-0.005..0.005));
        let high = price * (1.0 + rng.gen_range(0.0..0.02));
        let low = price * (1.0 - rng.gen_range(0.0..0.02));
        let close = price;
        let volume = rng.gen_range(1_000_000.0..10_000_000.0);

        candles.push(Candle::new(
            timestamp,
            open,
            high.max(open).max(close),
            low.min(open).min(close),
            close,
            volume,
        ));
    }

    MarketData::new(symbol, candles, "mock")
}

/// Create mock cryptocurrency data with higher volatility.
pub fn create_crypto_mock_data(symbol: &str, hours: usize, initial_price: f64) -> MarketData {
    let mut rng = rand::thread_rng();
    let start_date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

    let mut candles = Vec::with_capacity(hours);
    let mut price = initial_price;

    for i in 0..hours {
        let timestamp = (start_date.and_hms_opt(0, 0, 0).unwrap() + Duration::hours(i as i64)).and_utc();

        // Higher volatility for crypto
        let change = rng.gen_range(-0.02..0.025);
        price *= 1.0 + change;

        let open = price * (1.0 + rng.gen_range(-0.003..0.003));
        let high = price * (1.0 + rng.gen_range(0.0..0.01));
        let low = price * (1.0 - rng.gen_range(0.0..0.01));
        let close = price;
        let volume = rng.gen_range(100.0..1000.0) * price / 1000.0;

        candles.push(Candle::new(
            timestamp,
            open,
            high.max(open).max(close),
            low.min(open).min(close),
            close,
            volume,
        ));
    }

    MarketData::new(symbol, candles, "mock_crypto")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_mock_data() {
        let data = create_mock_data("TEST", 100, 100.0);
        assert_eq!(data.symbol, "TEST");
        assert_eq!(data.len(), 100);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_sma() {
        let data = create_mock_data("TEST", 50, 100.0);
        let sma = data.sma(20);
        assert_eq!(sma.len(), 31); // 50 - 20 + 1
    }

    #[test]
    fn test_rsi() {
        let data = create_mock_data("TEST", 50, 100.0);
        let rsi = data.rsi(14);
        assert!(!rsi.is_empty());
        assert!(rsi.iter().all(|&r| (0.0..=100.0).contains(&r)));
    }

    #[test]
    fn test_max_drawdown() {
        let data = create_mock_data("TEST", 100, 100.0);
        let dd = data.max_drawdown();
        assert!(dd >= 0.0 && dd <= 1.0);
    }
}
