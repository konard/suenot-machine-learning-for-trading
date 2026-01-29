//! # Feature Engineering Module
//!
//! Transforms raw OHLCV kline data into numerical feature vectors suitable for
//! machine learning models. Computes a comprehensive set of technical indicators
//! including returns, volatility, momentum, RSI-like oscillators, volume ratios,
//! and moving average ratios.
//!
//! All features are normalized to zero mean and unit variance for stable training.
//!
//! ## Features Computed
//!
//! 1. **Log Returns** - `ln(close[t] / close[t-1])`
//! 2. **Volatility** - Rolling standard deviation of returns
//! 3. **Momentum** - Price change over the lookback window
//! 4. **RSI-like Oscillator** - Ratio of average gains to average losses
//! 5. **Volume Ratio** - Current volume relative to rolling average
//! 6. **Moving Average Ratio** - Close price relative to its moving average
//!
//! ## Example
//!
//! ```rust
//! use domain_adaptation_trading::data::bybit::BybitClient;
//! use domain_adaptation_trading::data::features::FeatureGenerator;
//!
//! let client = BybitClient::new();
//! let klines = client.simulated_klines("BTCUSDT", 100);
//! let feature_gen = FeatureGenerator::new(20);
//!
//! let features = feature_gen.generate_features(&klines);
//! let labels = feature_gen.generate_labels(&klines);
//!
//! // Labels have one fewer element (need next candle for direction label)
//! assert_eq!(features.len(), labels.len() + 1);
//! ```

use crate::data::bybit::Kline;

/// Generates feature vectors and labels from raw kline data.
///
/// Uses a rolling window approach to compute technical indicators.
/// The `window_size` parameter controls the lookback period for all
/// rolling calculations.
pub struct FeatureGenerator {
    /// Number of periods to look back for rolling calculations
    pub window_size: usize,
}

impl FeatureGenerator {
    /// Creates a new `FeatureGenerator` with the specified window size.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of periods for rolling window calculations.
    ///   Typical values: 10-30 for intraday, 20-50 for daily.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is zero.
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "Window size must be positive");
        Self { window_size }
    }

    /// Generates normalized feature vectors from kline data.
    ///
    /// Each feature vector contains 6 normalized technical indicators.
    /// The first `window_size` klines are consumed as lookback and do not
    /// produce output features.
    ///
    /// # Arguments
    ///
    /// * `klines` - Slice of [`Kline`] data sorted chronologically
    ///
    /// # Returns
    ///
    /// A vector of feature vectors, where each inner vector has 6 elements:
    /// `[return, volatility, momentum, rsi, volume_ratio, ma_ratio]`
    ///
    /// Returns an empty vector if there are fewer klines than `window_size + 1`.
    pub fn generate_features(&self, klines: &[Kline]) -> Vec<Vec<f64>> {
        if klines.len() < self.window_size + 1 {
            return Vec::new();
        }

        // First compute raw log returns for the entire series
        let returns: Vec<f64> = klines
            .windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect();

        let n_features = klines.len() - self.window_size;
        let mut features = Vec::with_capacity(n_features);

        for i in self.window_size..klines.len() {
            let window_start = i - self.window_size;

            // Feature 1: Current log return
            let current_return = returns[i - 1]; // returns is 0-indexed from klines[1]

            // Feature 2: Rolling volatility (std dev of returns)
            let window_returns: Vec<f64> = returns[window_start..i].to_vec();
            let volatility = Self::std_dev(&window_returns);

            // Feature 3: Momentum (cumulative return over window)
            let momentum = (klines[i].close / klines[window_start].close).ln();

            // Feature 4: RSI-like oscillator
            let rsi = Self::compute_rsi(&window_returns);

            // Feature 5: Volume ratio (current volume / average volume)
            let avg_volume: f64 = klines[window_start..i]
                .iter()
                .map(|k| k.volume)
                .sum::<f64>()
                / self.window_size as f64;
            let volume_ratio = if avg_volume > 0.0 {
                klines[i].volume / avg_volume
            } else {
                1.0
            };

            // Feature 6: Moving average ratio (close / SMA)
            let sma: f64 = klines[window_start..i]
                .iter()
                .map(|k| k.close)
                .sum::<f64>()
                / self.window_size as f64;
            let ma_ratio = if sma > 0.0 {
                klines[i].close / sma - 1.0
            } else {
                0.0
            };

            features.push(vec![
                current_return,
                volatility,
                momentum,
                rsi,
                volume_ratio,
                ma_ratio,
            ]);
        }

        // Normalize features column-wise to zero mean, unit variance
        Self::normalize_features(&mut features);

        features
    }

    /// Generates binary labels for supervised training.
    ///
    /// Labels encode future price direction:
    /// - `1.0` if the next candle's close is higher than the current close
    /// - `0.0` otherwise
    ///
    /// # Arguments
    ///
    /// * `klines` - Slice of [`Kline`] data sorted chronologically
    ///
    /// # Returns
    ///
    /// A vector of labels aligned with the feature vectors from
    /// [`generate_features`](Self::generate_features).
    pub fn generate_labels(&self, klines: &[Kline]) -> Vec<f64> {
        if klines.len() < self.window_size + 2 {
            return Vec::new();
        }

        let n_labels = klines.len() - self.window_size - 1;
        let mut labels = Vec::with_capacity(n_labels);

        for i in self.window_size..(klines.len() - 1) {
            // Label: 1.0 if next candle goes up, 0.0 otherwise
            let label = if klines[i + 1].close > klines[i].close {
                1.0
            } else {
                0.0
            };
            labels.push(label);
        }

        labels
    }

    /// Computes the number of features produced per data point.
    ///
    /// Currently always returns 6 (the number of technical indicators).
    pub fn feature_count(&self) -> usize {
        6
    }

    /// Computes the standard deviation of a slice of f64 values.
    ///
    /// Uses the population standard deviation formula (dividing by N).
    fn std_dev(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        variance.sqrt()
    }

    /// Computes an RSI-like oscillator from a window of returns.
    ///
    /// Returns a value in [0, 1] where:
    /// - Values near 1.0 indicate strong upward momentum (overbought)
    /// - Values near 0.0 indicate strong downward momentum (oversold)
    /// - Values near 0.5 indicate balanced momentum
    fn compute_rsi(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.5;
        }

        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;
        let n = returns.len() as f64;

        for &r in returns {
            if r > 0.0 {
                avg_gain += r;
            } else {
                avg_loss += r.abs();
            }
        }

        avg_gain /= n;
        avg_loss /= n;

        if avg_loss == 0.0 {
            return 1.0;
        }

        let rs = avg_gain / avg_loss;
        1.0 - 1.0 / (1.0 + rs)
    }

    /// Normalizes features in-place to zero mean and unit variance (column-wise).
    ///
    /// Each feature column is independently standardized. Columns with zero
    /// variance are left unchanged (set to 0.0).
    fn normalize_features(features: &mut [Vec<f64>]) {
        if features.is_empty() {
            return;
        }

        let n = features.len() as f64;
        let n_features = features[0].len();

        for col in 0..n_features {
            // Compute mean
            let mean: f64 = features.iter().map(|row| row[col]).sum::<f64>() / n;

            // Compute standard deviation
            let variance: f64 = features
                .iter()
                .map(|row| (row[col] - mean).powi(2))
                .sum::<f64>()
                / n;
            let std = variance.sqrt();

            // Normalize: (x - mean) / std
            if std > 1e-10 {
                for row in features.iter_mut() {
                    row[col] = (row[col] - mean) / std;
                }
            } else {
                // Zero variance column: set all to 0.0
                for row in features.iter_mut() {
                    row[col] = 0.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::BybitClient;

    #[test]
    fn test_feature_generation() {
        let client = BybitClient::new();
        let klines = client.simulated_klines("BTCUSDT", 100);
        let gen = FeatureGenerator::new(20);

        let features = gen.generate_features(&klines);
        assert_eq!(features.len(), 80); // 100 - 20 = 80
        assert_eq!(features[0].len(), 6); // 6 features per sample
    }

    #[test]
    fn test_label_generation() {
        let client = BybitClient::new();
        let klines = client.simulated_klines("BTCUSDT", 100);
        let gen = FeatureGenerator::new(20);

        let labels = gen.generate_labels(&klines);
        assert_eq!(labels.len(), 79); // 100 - 20 - 1 = 79

        // Labels should be binary
        for &label in &labels {
            assert!(label == 0.0 || label == 1.0);
        }
    }

    #[test]
    fn test_features_normalized() {
        let client = BybitClient::new();
        let klines = client.simulated_klines("BTCUSDT", 200);
        let gen = FeatureGenerator::new(20);

        let features = gen.generate_features(&klines);

        // Check approximate zero mean for each feature
        for col in 0..6 {
            let mean: f64 =
                features.iter().map(|row| row[col]).sum::<f64>() / features.len() as f64;
            assert!(
                mean.abs() < 0.01,
                "Feature {} mean {} not near zero",
                col,
                mean
            );
        }
    }

    #[test]
    fn test_insufficient_data() {
        let client = BybitClient::new();
        let klines = client.simulated_klines("BTCUSDT", 10);
        let gen = FeatureGenerator::new(20);

        let features = gen.generate_features(&klines);
        assert!(features.is_empty());
    }

    #[test]
    fn test_rsi_bounds() {
        let returns = vec![0.01, -0.02, 0.015, -0.005, 0.003];
        let rsi = FeatureGenerator::compute_rsi(&returns);
        assert!((0.0..=1.0).contains(&rsi), "RSI {} out of bounds", rsi);
    }

    #[test]
    fn test_feature_count() {
        let gen = FeatureGenerator::new(20);
        assert_eq!(gen.feature_count(), 6);
    }

    #[test]
    #[should_panic(expected = "Window size must be positive")]
    fn test_zero_window_panics() {
        let _gen = FeatureGenerator::new(0);
    }
}
