//! Feature engineering for GLOW trading model
//!
//! Extracts market features from OHLCV data for model input

use crate::data::Candle;
use ndarray::{Array1, Array2};
use std::collections::VecDeque;

/// Number of features extracted per timestep
pub const NUM_FEATURES: usize = 16;

/// Market features extracted from candle data
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    /// Return over 1 period
    pub return_1: f64,
    /// Return over 5 periods
    pub return_5: f64,
    /// Return over 10 periods
    pub return_10: f64,
    /// Return over 20 periods
    pub return_20: f64,
    /// Volatility over 5 periods
    pub volatility_5: f64,
    /// Volatility over 20 periods
    pub volatility_20: f64,
    /// Volatility ratio (5/20)
    pub vol_ratio: f64,
    /// Momentum over 10 periods
    pub momentum_10: f64,
    /// Momentum over 20 periods
    pub momentum_20: f64,
    /// Volume ratio (current/MA20)
    pub volume_ratio: f64,
    /// Price position in 20-period range (0-1)
    pub price_position: f64,
    /// Candle body ratio
    pub body_ratio: f64,
    /// Upper shadow ratio
    pub upper_shadow: f64,
    /// Lower shadow ratio
    pub lower_shadow: f64,
    /// Average True Range normalized
    pub atr_norm: f64,
    /// RSI-like indicator
    pub rsi: f64,
}

impl MarketFeatures {
    /// Convert features to array
    pub fn to_array(&self) -> Array1<f64> {
        Array1::from_vec(vec![
            self.return_1,
            self.return_5,
            self.return_10,
            self.return_20,
            self.volatility_5,
            self.volatility_20,
            self.vol_ratio,
            self.momentum_10,
            self.momentum_20,
            self.volume_ratio,
            self.price_position,
            self.body_ratio,
            self.upper_shadow,
            self.lower_shadow,
            self.atr_norm,
            self.rsi,
        ])
    }

    /// Create from array
    pub fn from_array(arr: &Array1<f64>) -> Self {
        Self {
            return_1: arr[0],
            return_5: arr[1],
            return_10: arr[2],
            return_20: arr[3],
            volatility_5: arr[4],
            volatility_20: arr[5],
            vol_ratio: arr[6],
            momentum_10: arr[7],
            momentum_20: arr[8],
            volume_ratio: arr[9],
            price_position: arr[10],
            body_ratio: arr[11],
            upper_shadow: arr[12],
            lower_shadow: arr[13],
            atr_norm: arr[14],
            rsi: arr[15],
        }
    }
}

/// Feature extractor for market data
pub struct FeatureExtractor {
    lookback: usize,
    buffer: VecDeque<Candle>,
}

impl FeatureExtractor {
    /// Create new feature extractor
    ///
    /// # Arguments
    /// * `lookback` - Number of periods to look back for feature calculation
    pub fn new(lookback: usize) -> Self {
        Self {
            lookback,
            buffer: VecDeque::with_capacity(lookback + 1),
        }
    }

    /// Add a new candle and optionally compute features
    ///
    /// Returns `Some(features)` if we have enough data
    pub fn add_candle(&mut self, candle: Candle) -> Option<MarketFeatures> {
        self.buffer.push_back(candle);

        // Keep only lookback + 1 candles
        while self.buffer.len() > self.lookback + 1 {
            self.buffer.pop_front();
        }

        // Need at least lookback + 1 candles to compute features
        if self.buffer.len() < self.lookback + 1 {
            return None;
        }

        Some(self.compute_features())
    }

    /// Compute features from current buffer
    fn compute_features(&self) -> MarketFeatures {
        let candles: Vec<&Candle> = self.buffer.iter().collect();
        let n = candles.len();
        let current = candles[n - 1];

        // Compute returns
        let returns: Vec<f64> = candles
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();

        // Returns at different scales
        let return_1 = *returns.last().unwrap_or(&0.0);
        let return_5 = returns.iter().rev().take(5).sum();
        let return_10 = returns.iter().rev().take(10).sum();
        let return_20 = returns.iter().sum();

        // Volatility
        let volatility_5 = Self::std_dev(&returns.iter().rev().take(5).copied().collect::<Vec<_>>());
        let volatility_20 = Self::std_dev(&returns);
        let vol_ratio = if volatility_20 > 1e-8 {
            volatility_5 / volatility_20
        } else {
            1.0
        };

        // Momentum
        let momentum_10 = if n > 10 {
            current.close / candles[n - 11].close - 1.0
        } else {
            0.0
        };
        let momentum_20 = current.close / candles[0].close - 1.0;

        // Volume ratio
        let avg_volume: f64 = candles.iter().map(|c| c.volume).sum::<f64>() / n as f64;
        let volume_ratio = if avg_volume > 1e-8 {
            current.volume / avg_volume
        } else {
            1.0
        };

        // Price position in range
        let high_20: f64 = candles.iter().map(|c| c.high).fold(f64::MIN, f64::max);
        let low_20: f64 = candles.iter().map(|c| c.low).fold(f64::MAX, f64::min);
        let range = high_20 - low_20;
        let price_position = if range > 1e-8 {
            (current.close - low_20) / range
        } else {
            0.5
        };

        // Candle patterns
        let candle_range = current.range();
        let body_ratio = if candle_range > 1e-8 {
            (current.close - current.open) / candle_range
        } else {
            0.0
        };

        let upper_shadow = if candle_range > 1e-8 {
            (current.high - current.close.max(current.open)) / candle_range
        } else {
            0.0
        };

        let lower_shadow = if candle_range > 1e-8 {
            (current.close.min(current.open) - current.low) / candle_range
        } else {
            0.0
        };

        // ATR normalized
        let true_ranges: Vec<f64> = candles
            .windows(2)
            .map(|w| {
                let high_low = w[1].high - w[1].low;
                let high_prev_close = (w[1].high - w[0].close).abs();
                let low_prev_close = (w[1].low - w[0].close).abs();
                high_low.max(high_prev_close).max(low_prev_close)
            })
            .collect();

        let atr = if !true_ranges.is_empty() {
            true_ranges.iter().sum::<f64>() / true_ranges.len() as f64
        } else {
            0.0
        };
        let atr_norm = if current.close > 1e-8 {
            atr / current.close
        } else {
            0.0
        };

        // RSI-like indicator
        let gains: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let losses: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let rsi = if gains + losses > 1e-8 {
            gains / (gains + losses)
        } else {
            0.5
        };

        MarketFeatures {
            return_1,
            return_5,
            return_10,
            return_20,
            volatility_5,
            volatility_20,
            vol_ratio,
            momentum_10,
            momentum_20,
            volume_ratio,
            price_position,
            body_ratio,
            upper_shadow,
            lower_shadow,
            atr_norm,
            rsi,
        }
    }

    /// Compute standard deviation
    fn std_dev(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    /// Reset the extractor state
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Extract features from a slice of candles
    pub fn extract_features_batch(candles: &[Candle], lookback: usize) -> Array2<f64> {
        let mut extractor = FeatureExtractor::new(lookback);
        let mut features = Vec::new();

        for candle in candles {
            if let Some(f) = extractor.add_candle(candle.clone()) {
                features.push(f.to_array().to_vec());
            }
        }

        if features.is_empty() {
            return Array2::zeros((0, NUM_FEATURES));
        }

        let rows = features.len();
        let flat: Vec<f64> = features.into_iter().flatten().collect();
        Array2::from_shape_vec((rows, NUM_FEATURES), flat)
            .expect("Failed to create feature matrix")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| Candle {
                timestamp: i as i64 * 3600000,
                open: 100.0 + i as f64,
                high: 102.0 + i as f64,
                low: 98.0 + i as f64,
                close: 101.0 + i as f64,
                volume: 1000.0 + i as f64 * 10.0,
                turnover: 100000.0,
            })
            .collect()
    }

    #[test]
    fn test_feature_extractor() {
        let candles = create_test_candles(30);
        let mut extractor = FeatureExtractor::new(20);

        let mut feature_count = 0;
        for candle in candles {
            if let Some(_features) = extractor.add_candle(candle) {
                feature_count += 1;
            }
        }

        // Should have 30 - 20 = 10 feature sets (first 20 are used for initialization)
        assert_eq!(feature_count, 9); // Actually lookback requires lookback+1 candles
    }

    #[test]
    fn test_batch_extraction() {
        let candles = create_test_candles(50);
        let features = FeatureExtractor::extract_features_batch(&candles, 20);

        assert_eq!(features.ncols(), NUM_FEATURES);
        assert!(features.nrows() > 0);
    }

    #[test]
    fn test_market_features_conversion() {
        let features = MarketFeatures {
            return_1: 0.01,
            return_5: 0.05,
            return_10: 0.08,
            return_20: 0.10,
            volatility_5: 0.02,
            volatility_20: 0.03,
            vol_ratio: 0.67,
            momentum_10: 0.05,
            momentum_20: 0.08,
            volume_ratio: 1.2,
            price_position: 0.75,
            body_ratio: 0.5,
            upper_shadow: 0.2,
            lower_shadow: 0.3,
            atr_norm: 0.015,
            rsi: 0.6,
        };

        let arr = features.to_array();
        let restored = MarketFeatures::from_array(&arr);

        assert!((restored.return_1 - features.return_1).abs() < 1e-10);
        assert!((restored.rsi - features.rsi).abs() < 1e-10);
    }
}
