//! Data types for market data
//!
//! This module defines the core data structures for representing
//! cryptocurrency market data.

use chrono::{DateTime, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// OHLCV candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
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
    /// Quote volume (in USDT)
    pub quote_volume: f64,
}

impl Candle {
    /// Create a new candle
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        quote_volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            quote_volume,
        }
    }

    /// Calculate returns from previous close
    pub fn returns(&self, prev_close: f64) -> f64 {
        if prev_close > 0.0 {
            (self.close - prev_close) / prev_close
        } else {
            0.0
        }
    }

    /// Calculate intraday volatility (high-low range)
    pub fn intraday_volatility(&self) -> f64 {
        if self.low > 0.0 {
            (self.high - self.low) / self.low
        } else {
            0.0
        }
    }

    /// Calculate true range
    pub fn true_range(&self, prev_close: f64) -> f64 {
        let hl = self.high - self.low;
        let hc = (self.high - prev_close).abs();
        let lc = (self.low - prev_close).abs();
        hl.max(hc).max(lc)
    }
}

/// Time series of price data
#[derive(Debug, Clone)]
pub struct PriceSeries {
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Candles in chronological order
    pub candles: Vec<Candle>,
}

impl PriceSeries {
    /// Create a new price series
    pub fn new(symbol: String, candles: Vec<Candle>) -> Self {
        Self { symbol, candles }
    }

    /// Get the number of candles
    pub fn len(&self) -> usize {
        self.candles.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }

    /// Get closing prices as array
    pub fn close_prices(&self) -> Array1<f64> {
        Array1::from_vec(self.candles.iter().map(|c| c.close).collect())
    }

    /// Get volumes as array
    pub fn volumes(&self) -> Array1<f64> {
        Array1::from_vec(self.candles.iter().map(|c| c.volume).collect())
    }

    /// Calculate returns
    pub fn returns(&self) -> Array1<f64> {
        let closes = self.close_prices();
        let mut returns = Array1::zeros(closes.len());
        for i in 1..closes.len() {
            returns[i] = (closes[i] - closes[i - 1]) / closes[i - 1];
        }
        returns
    }

    /// Calculate rolling volatility
    pub fn rolling_volatility(&self, window: usize) -> Array1<f64> {
        let returns = self.returns();
        let n = returns.len();
        let mut vol = Array1::zeros(n);

        for i in window..n {
            let window_returns = returns.slice(ndarray::s![i - window..i]);
            let mean = window_returns.mean().unwrap_or(0.0);
            let variance: f64 = window_returns
                .iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>()
                / window as f64;
            vol[i] = variance.sqrt() * (365.0_f64).sqrt(); // Annualized
        }
        vol
    }

    /// Get the latest candle
    pub fn latest(&self) -> Option<&Candle> {
        self.candles.last()
    }
}

/// Feature set for model input
#[derive(Debug, Clone)]
pub struct Features {
    /// Symbol
    pub symbol: String,
    /// Feature matrix (time_steps x features)
    pub data: ndarray::Array2<f64>,
    /// Feature names
    pub feature_names: Vec<String>,
}

impl Features {
    /// Create features from price series
    pub fn from_price_series(series: &PriceSeries, lookback: usize) -> Self {
        let n = series.len();

        // Calculate raw features
        let returns = series.returns();
        let volatility = series.rolling_volatility(20);
        let closes = series.close_prices();
        let volumes = series.volumes();

        // Log volume
        let log_volume: Array1<f64> = volumes.mapv(|v| (v + 1.0).ln());
        let volume_change: Array1<f64> = {
            let mut vc = Array1::zeros(n);
            for i in 1..n {
                if log_volume[i - 1] > 0.0 {
                    vc[i] = log_volume[i] - log_volume[i - 1];
                }
            }
            vc
        };

        // RSI
        let rsi = calculate_rsi(&returns, 14);

        // Price momentum (various periods)
        let mom_7 = calculate_momentum(&closes, 7);
        let mom_14 = calculate_momentum(&closes, 14);
        let mom_30 = calculate_momentum(&closes, 30);

        // Number of features
        let num_features = 7;
        let valid_len = n.saturating_sub(lookback);

        // Build feature matrix
        let mut data = ndarray::Array2::zeros((valid_len, num_features));

        for i in 0..valid_len {
            let idx = i + lookback;
            data[[i, 0]] = returns[idx];
            data[[i, 1]] = volatility[idx];
            data[[i, 2]] = volume_change[idx];
            data[[i, 3]] = rsi[idx];
            data[[i, 4]] = mom_7[idx];
            data[[i, 5]] = mom_14[idx];
            data[[i, 6]] = mom_30[idx];
        }

        let feature_names = vec![
            "returns".to_string(),
            "volatility".to_string(),
            "volume_change".to_string(),
            "rsi".to_string(),
            "momentum_7".to_string(),
            "momentum_14".to_string(),
            "momentum_30".to_string(),
        ];

        Features {
            symbol: series.symbol.clone(),
            data,
            feature_names,
        }
    }
}

/// Calculate RSI (Relative Strength Index)
fn calculate_rsi(returns: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = returns.len();
    let mut rsi = Array1::zeros(n);

    for i in period..n {
        let window = returns.slice(ndarray::s![i - period..i]);
        let gains: f64 = window.iter().filter(|&&r| r > 0.0).sum();
        let losses: f64 = window.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();

        if losses > 0.0 {
            let rs = gains / losses;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        } else {
            rsi[i] = 100.0;
        }
    }
    rsi
}

/// Calculate price momentum
fn calculate_momentum(prices: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = prices.len();
    let mut momentum = Array1::zeros(n);

    for i in period..n {
        if prices[i - period] > 0.0 {
            momentum[i] = (prices[i] - prices[i - period]) / prices[i - period];
        }
    }
    momentum
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_candle_returns() {
        let candle = Candle::new(
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            100.0,
            110.0,
            95.0,
            105.0,
            1000.0,
            100000.0,
        );
        let returns = candle.returns(100.0);
        assert!((returns - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_price_series_returns() {
        let candles = vec![
            Candle::new(
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                100.0, 105.0, 98.0, 100.0, 1000.0, 100000.0,
            ),
            Candle::new(
                Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
                100.0, 110.0, 99.0, 105.0, 1200.0, 120000.0,
            ),
        ];
        let series = PriceSeries::new("BTCUSDT".to_string(), candles);
        let returns = series.returns();
        assert!((returns[1] - 0.05).abs() < 1e-10);
    }
}
