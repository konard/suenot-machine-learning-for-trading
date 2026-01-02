//! Technical indicators and feature extraction
//!
//! Computes features from OHLCV data for ML models.

use crate::api::Candle;
use ndarray::{Array1, Array2};

/// Feature extractor for market data
pub struct FeatureExtractor {
    /// Lookback periods for various indicators
    periods: Vec<usize>,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    /// Create with default periods
    pub fn new() -> Self {
        Self {
            periods: vec![5, 10, 20, 50],
        }
    }

    /// Create with custom periods
    pub fn with_periods(periods: Vec<usize>) -> Self {
        Self { periods }
    }

    /// Extract all features from candles
    pub fn extract(&self, candles: &[Candle]) -> Array2<f64> {
        let n = candles.len();
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
        let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        let mut features = Vec::new();

        // Returns for different periods
        for &period in &self.periods {
            features.push(self.compute_returns(&closes, period));
        }

        // Volatility
        for &period in &self.periods {
            features.push(self.compute_volatility(&closes, period));
        }

        // RSI
        features.push(self.compute_rsi(&closes, 14));

        // MACD
        let (macd, signal) = self.compute_macd(&closes);
        features.push(macd);
        features.push(signal);

        // Bollinger Bands position
        features.push(self.compute_bb_position(&closes, 20));

        // Volume change
        features.push(self.compute_returns(&volumes, 1));

        // Price position relative to high-low range
        features.push(self.compute_range_position(&closes, &highs, &lows));

        // Stack features into matrix
        let num_features = features.len();
        let mut result = Array2::zeros((n, num_features));

        for (j, feature) in features.iter().enumerate() {
            for (i, &val) in feature.iter().enumerate() {
                result[[i, j]] = if val.is_finite() { val } else { 0.0 };
            }
        }

        result
    }

    /// Compute returns
    fn compute_returns(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut returns = vec![0.0; n];

        for i in period..n {
            if prices[i - period] != 0.0 {
                returns[i] = (prices[i] / prices[i - period]) - 1.0;
            }
        }

        returns
    }

    /// Compute rolling volatility
    fn compute_volatility(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let returns = self.compute_returns(prices, 1);
        let n = returns.len();
        let mut vol = vec![0.0; n];

        for i in period..n {
            let window = &returns[i - period + 1..=i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 = window.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / period as f64;
            vol[i] = variance.sqrt();
        }

        vol
    }

    /// Compute RSI
    fn compute_rsi(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut rsi = vec![50.0; n]; // Neutral default

        if n < period + 1 {
            return rsi;
        }

        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        for i in 1..n {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Initial averages
        let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        for i in period..n {
            if i > period {
                avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
                avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
            }

            if avg_loss == 0.0 {
                rsi[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        // Normalize to [-1, 1]
        rsi.iter().map(|&r| (r - 50.0) / 50.0).collect()
    }

    /// Compute MACD
    fn compute_macd(&self, prices: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let ema12 = self.compute_ema(prices, 12);
        let ema26 = self.compute_ema(prices, 26);

        let macd: Vec<f64> = ema12
            .iter()
            .zip(ema26.iter())
            .map(|(a, b)| a - b)
            .collect();

        let signal = self.compute_ema(&macd, 9);

        // Normalize
        let max_macd = macd.iter().cloned().fold(0.0_f64, f64::max).max(1e-8);
        let macd_norm: Vec<f64> = macd.iter().map(|&m| m / max_macd).collect();
        let signal_norm: Vec<f64> = signal.iter().map(|&s| s / max_macd).collect();

        (macd_norm, signal_norm)
    }

    /// Compute EMA
    fn compute_ema(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut ema = vec![0.0; n];
        let multiplier = 2.0 / (period + 1) as f64;

        if n == 0 {
            return ema;
        }

        ema[0] = prices[0];

        for i in 1..n {
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
        }

        ema
    }

    /// Compute Bollinger Bands position
    fn compute_bb_position(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut position = vec![0.0; n];

        for i in period..n {
            let window = &prices[i - period + 1..=i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let std: f64 = (window.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / period as f64).sqrt();

            if std > 0.0 {
                let upper = mean + 2.0 * std;
                let lower = mean - 2.0 * std;
                position[i] = (prices[i] - lower) / (upper - lower) * 2.0 - 1.0;
            }
        }

        position
    }

    /// Compute position in high-low range
    fn compute_range_position(&self, closes: &[f64], highs: &[f64], lows: &[f64]) -> Vec<f64> {
        closes
            .iter()
            .zip(highs.iter())
            .zip(lows.iter())
            .map(|((&c, &h), &l)| {
                if h - l > 0.0 {
                    (c - l) / (h - l) * 2.0 - 1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Extract features for multiple assets
    pub fn extract_multi(&self, candles_multi: &[Vec<Candle>]) -> Array2<f64> {
        if candles_multi.is_empty() {
            return Array2::zeros((0, 0));
        }

        // Get minimum length across all assets
        let min_len = candles_multi.iter().map(|c| c.len()).min().unwrap_or(0);

        // Extract features for each asset
        let features: Vec<Array2<f64>> = candles_multi
            .iter()
            .map(|candles| {
                let trimmed: Vec<Candle> = candles.iter().rev().take(min_len).rev().cloned().collect();
                self.extract(&trimmed)
            })
            .collect();

        if features.is_empty() || features[0].is_empty() {
            return Array2::zeros((0, 0));
        }

        // Take the last row from each (current features)
        let num_features = features[0].ncols();
        let num_assets = features.len();
        let mut result = Array2::zeros((num_assets, num_features));

        for (i, feat) in features.iter().enumerate() {
            let last_row = feat.row(feat.nrows() - 1);
            result.row_mut(i).assign(&last_row);
        }

        result
    }

    /// Compute returns matrix from candles
    pub fn compute_returns_matrix(&self, candles_multi: &[Vec<Candle>]) -> Array2<f64> {
        if candles_multi.is_empty() {
            return Array2::zeros((0, 0));
        }

        let min_len = candles_multi.iter().map(|c| c.len()).min().unwrap_or(0);
        let num_assets = candles_multi.len();

        let mut returns = Array2::zeros((min_len, num_assets));

        for (j, candles) in candles_multi.iter().enumerate() {
            let closes: Vec<f64> = candles.iter().rev().take(min_len).rev().map(|c| c.close).collect();
            let rets = self.compute_returns(&closes, 1);

            for (i, &r) in rets.iter().enumerate() {
                returns[[i, j]] = r;
            }
        }

        returns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| Candle {
                timestamp: i as i64 * 3600000,
                open: 100.0 + (i as f64 * 0.1).sin(),
                high: 101.0 + (i as f64 * 0.1).sin(),
                low: 99.0 + (i as f64 * 0.1).sin(),
                close: 100.5 + (i as f64 * 0.1).sin(),
                volume: 1000.0 + i as f64,
                turnover: 100000.0,
            })
            .collect()
    }

    #[test]
    fn test_extract_features() {
        let candles = sample_candles(100);
        let extractor = FeatureExtractor::new();
        let features = extractor.extract(&candles);

        assert_eq!(features.nrows(), 100);
        assert!(features.ncols() > 0);
    }

    #[test]
    fn test_rsi() {
        let candles = sample_candles(50);
        let extractor = FeatureExtractor::new();
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let rsi = extractor.compute_rsi(&closes, 14);

        assert_eq!(rsi.len(), 50);
        for &r in &rsi[14..] {
            assert!(r >= -1.0 && r <= 1.0);
        }
    }

    #[test]
    fn test_multi_asset() {
        let candles1 = sample_candles(100);
        let candles2 = sample_candles(100);
        let candles_multi = vec![candles1, candles2];

        let extractor = FeatureExtractor::new();
        let features = extractor.extract_multi(&candles_multi);

        assert_eq!(features.nrows(), 2);
    }
}
