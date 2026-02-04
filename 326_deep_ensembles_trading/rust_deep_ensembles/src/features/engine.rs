//! Feature engineering engine

use super::indicators::TechnicalIndicators;
use crate::api::types::Kline;
use ndarray::{Array1, Array2};

/// Feature engineering engine
pub struct FeatureEngine {
    /// Feature names
    pub feature_names: Vec<String>,
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureEngine {
    /// Create a new feature engine
    pub fn new() -> Self {
        Self {
            feature_names: vec![
                "returns_1".to_string(),
                "returns_5".to_string(),
                "returns_10".to_string(),
                "returns_20".to_string(),
                "volatility_10".to_string(),
                "volatility_20".to_string(),
                "rsi_14".to_string(),
                "macd_diff".to_string(),
                "bb_position".to_string(),
                "volume_ratio".to_string(),
            ],
        }
    }

    /// Compute features from klines
    pub fn compute_features(klines: &[Kline]) -> Array2<f64> {
        let n = klines.len();
        let n_features = 10;

        let mut features = Array2::zeros((n, n_features));

        // Extract price and volume series
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Returns at different horizons
        let returns_1 = Self::compute_returns(&closes, 1);
        let returns_5 = Self::compute_returns(&closes, 5);
        let returns_10 = Self::compute_returns(&closes, 10);
        let returns_20 = Self::compute_returns(&closes, 20);

        // Volatility
        let volatility_10 = TechnicalIndicators::volatility(&closes, 10);
        let volatility_20 = TechnicalIndicators::volatility(&closes, 20);

        // RSI
        let rsi = TechnicalIndicators::rsi(&closes, 14);

        // MACD
        let (macd_line, signal_line, _) = TechnicalIndicators::macd(&closes, 12, 26, 9);
        let macd_diff = &macd_line - &signal_line;

        // Bollinger Band position
        let bb_pos = TechnicalIndicators::bb_position(&closes, 20, 2.0);

        // Volume ratio
        let vol_ratio = TechnicalIndicators::volume_ratio(&volumes, 20);

        // Assign to feature matrix
        for i in 0..n {
            features[[i, 0]] = returns_1[i];
            features[[i, 1]] = returns_5[i];
            features[[i, 2]] = returns_10[i];
            features[[i, 3]] = returns_20[i];
            features[[i, 4]] = volatility_10[i];
            features[[i, 5]] = volatility_20[i];
            features[[i, 6]] = (rsi[i] - 50.0) / 50.0; // Normalize RSI to [-1, 1]
            features[[i, 7]] = macd_diff[i] / closes[i] * 100.0; // Normalize MACD
            features[[i, 8]] = bb_pos[i] * 2.0 - 1.0; // Normalize to [-1, 1]
            features[[i, 9]] = vol_ratio[i].ln(); // Log volume ratio
        }

        // Handle NaN/Inf values
        for i in 0..n {
            for j in 0..n_features {
                if !features[[i, j]].is_finite() {
                    features[[i, j]] = 0.0;
                }
            }
        }

        features
    }

    /// Compute targets (future returns)
    pub fn compute_targets(klines: &[Kline], horizon: usize) -> Array1<f64> {
        let n = klines.len();
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let mut targets = Array1::zeros(n);

        for i in 0..n.saturating_sub(horizon) {
            targets[i] = (closes[i + horizon] - closes[i]) / closes[i];
        }

        // Fill last values with zeros
        for i in n.saturating_sub(horizon)..n {
            targets[i] = 0.0;
        }

        targets
    }

    /// Compute returns with given horizon
    fn compute_returns(closes: &[f64], horizon: usize) -> Array1<f64> {
        let n = closes.len();
        let mut returns = Array1::zeros(n);

        for i in horizon..n {
            returns[i] = (closes[i] - closes[i - horizon]) / closes[i - horizon];
        }

        returns
    }

    /// Standardize features (zero mean, unit variance)
    pub fn standardize(features: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let n_features = features.ncols();

        let mut means = Array1::zeros(n_features);
        let mut stds = Array1::zeros(n_features);

        for j in 0..n_features {
            let col = features.column(j);
            means[j] = col.mean().unwrap_or(0.0);
            stds[j] = col.std(0.0);
            if stds[j] < 1e-10 {
                stds[j] = 1.0;
            }
        }

        let mut standardized = features.clone();
        for j in 0..n_features {
            for i in 0..features.nrows() {
                standardized[[i, j]] = (features[[i, j]] - means[j]) / stds[j];
            }
        }

        (standardized, means, stds)
    }

    /// Apply standardization with given mean and std
    pub fn apply_standardization(
        features: &Array2<f64>,
        means: &Array1<f64>,
        stds: &Array1<f64>,
    ) -> Array2<f64> {
        let mut standardized = features.clone();

        for j in 0..features.ncols() {
            for i in 0..features.nrows() {
                standardized[[i, j]] = (features[[i, j]] - means[j]) / stds[j];
            }
        }

        standardized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        let mut klines = Vec::with_capacity(n);
        let mut price = 100.0;

        for i in 0..n {
            let change = 0.01 * ((i as f64 * 0.1).sin());
            price *= 1.0 + change;

            klines.push(Kline {
                start_time: i as i64 * 3600000,
                open: price,
                high: price * 1.01,
                low: price * 0.99,
                close: price,
                volume: 1000.0 + (i as f64 * 0.5).cos() * 500.0,
                turnover: price * 1000.0,
            });
        }

        klines
    }

    #[test]
    fn test_compute_features() {
        let klines = create_test_klines(100);
        let features = FeatureEngine::compute_features(&klines);

        assert_eq!(features.nrows(), 100);
        assert_eq!(features.ncols(), 10);

        // Check no NaN values
        for i in 0..features.nrows() {
            for j in 0..features.ncols() {
                assert!(features[[i, j]].is_finite());
            }
        }
    }

    #[test]
    fn test_compute_targets() {
        let klines = create_test_klines(50);
        let targets = FeatureEngine::compute_targets(&klines, 5);

        assert_eq!(targets.len(), 50);
    }

    #[test]
    fn test_standardize() {
        let features = Array2::from_shape_fn((100, 5), |_| rand::random::<f64>() * 10.0);

        let (standardized, means, stds) = FeatureEngine::standardize(&features);

        // Check standardized features have approximately zero mean and unit variance
        for j in 0..5 {
            let col = standardized.column(j);
            let mean = col.mean().unwrap();
            let std = col.std(0.0);

            assert!(mean.abs() < 0.1);
            assert!((std - 1.0).abs() < 0.1);
        }
    }
}
