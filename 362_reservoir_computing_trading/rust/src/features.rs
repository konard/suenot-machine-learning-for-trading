//! # Feature Engineering
//!
//! Feature extraction and normalization for trading signals.
//!
//! ## Features
//!
//! - Log returns
//! - Realized volatility
//! - RSI (Relative Strength Index)
//! - Volume imbalance
//! - Order flow imbalance
//! - Momentum indicators

use crate::bybit::{Kline, OrderBook};
use ndarray::{Array1, Array2, s};
use std::collections::VecDeque;

/// Market features for model input
#[derive(Debug, Clone, Default)]
pub struct MarketFeatures {
    /// Log return
    pub log_return: f64,

    /// Realized volatility (rolling window)
    pub realized_volatility: f64,

    /// Normalized RSI [-1, 1]
    pub rsi_normalized: f64,

    /// Volume ratio (current / average)
    pub volume_ratio: f64,

    /// Price momentum (5-period)
    pub momentum_5: f64,

    /// Price momentum (10-period)
    pub momentum_10: f64,

    /// Normalized spread (spread / mid)
    pub spread_normalized: f64,

    /// Order book imbalance [-1, 1]
    pub order_imbalance: f64,
}

impl MarketFeatures {
    /// Convert to array for model input
    pub fn to_array(&self) -> Array1<f64> {
        Array1::from_vec(vec![
            self.log_return,
            self.realized_volatility,
            self.rsi_normalized,
            self.volume_ratio,
            self.momentum_5,
            self.momentum_10,
            self.spread_normalized,
        ])
    }

    /// Feature names for logging/debugging
    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "log_return",
            "realized_volatility",
            "rsi_normalized",
            "volume_ratio",
            "momentum_5",
            "momentum_10",
            "spread_normalized",
        ]
    }

    /// Number of features
    pub fn n_features() -> usize {
        7
    }
}

/// Feature extractor with rolling window calculations
pub struct FeatureExtractor {
    /// Window size for volatility calculation
    volatility_window: usize,

    /// Window size for RSI calculation
    rsi_window: usize,

    /// Window size for volume average
    volume_window: usize,

    /// Price history
    prices: VecDeque<f64>,

    /// Return history
    returns: VecDeque<f64>,

    /// Volume history
    volumes: VecDeque<f64>,

    /// Gains for RSI
    gains: VecDeque<f64>,

    /// Losses for RSI
    losses: VecDeque<f64>,

    /// Last processed price
    last_price: Option<f64>,

    /// Minimum samples before valid output
    min_samples: usize,

    /// Current sample count
    sample_count: usize,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    ///
    /// # Arguments
    ///
    /// * `volatility_window` - Window size for volatility (default 20)
    /// * `rsi_window` - Window size for RSI (default 14)
    /// * `volume_window` - Window size for volume average (default 20)
    pub fn new(volatility_window: usize, rsi_window: usize, volume_window: usize) -> Self {
        let min_samples = volatility_window.max(rsi_window).max(volume_window) + 1;

        Self {
            volatility_window,
            rsi_window,
            volume_window,
            prices: VecDeque::with_capacity(volatility_window + 10),
            returns: VecDeque::with_capacity(volatility_window + 1),
            volumes: VecDeque::with_capacity(volume_window + 1),
            gains: VecDeque::with_capacity(rsi_window + 1),
            losses: VecDeque::with_capacity(rsi_window + 1),
            last_price: None,
            min_samples,
            sample_count: 0,
        }
    }

    /// Create with default parameters
    pub fn default_params() -> Self {
        Self::new(20, 14, 20)
    }

    /// Update with new kline data and extract features
    pub fn update(&mut self, kline: &Kline, orderbook: Option<&OrderBook>) -> Option<MarketFeatures> {
        let price = kline.close;
        let volume = kline.volume;

        // Calculate log return
        let log_return = if let Some(last) = self.last_price {
            (price / last).ln()
        } else {
            0.0
        };

        // Update price history
        self.prices.push_back(price);
        if self.prices.len() > self.volatility_window + 10 {
            self.prices.pop_front();
        }

        // Update return history
        if self.last_price.is_some() {
            self.returns.push_back(log_return);
            if self.returns.len() > self.volatility_window {
                self.returns.pop_front();
            }

            // Update gains/losses for RSI
            if log_return > 0.0 {
                self.gains.push_back(log_return);
                self.losses.push_back(0.0);
            } else {
                self.gains.push_back(0.0);
                self.losses.push_back(-log_return);
            }

            if self.gains.len() > self.rsi_window {
                self.gains.pop_front();
                self.losses.pop_front();
            }
        }

        // Update volume history
        self.volumes.push_back(volume);
        if self.volumes.len() > self.volume_window {
            self.volumes.pop_front();
        }

        self.last_price = Some(price);
        self.sample_count += 1;

        // Check if we have enough data
        if self.sample_count < self.min_samples {
            return None;
        }

        // Calculate features
        let realized_volatility = self.calculate_volatility();
        let rsi = self.calculate_rsi();
        let rsi_normalized = (rsi - 50.0) / 50.0; // Scale to [-1, 1]
        let volume_ratio = self.calculate_volume_ratio(volume);
        let momentum_5 = self.calculate_momentum(5);
        let momentum_10 = self.calculate_momentum(10);

        // Order book features
        let (spread_normalized, order_imbalance) = if let Some(ob) = orderbook {
            let spread_norm = ob.spread_bps().unwrap_or(0.0) / 100.0; // Normalize to ~[-1, 1]
            let imbalance = ob.imbalance(5);
            (spread_norm, imbalance)
        } else {
            (0.0, 0.0)
        };

        Some(MarketFeatures {
            log_return,
            realized_volatility,
            rsi_normalized,
            volume_ratio,
            momentum_5,
            momentum_10,
            spread_normalized,
            order_imbalance,
        })
    }

    /// Calculate realized volatility
    fn calculate_volatility(&self) -> f64 {
        if self.returns.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self.returns.iter().cloned().collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / (returns.len() - 1) as f64;

        variance.sqrt()
    }

    /// Calculate RSI
    fn calculate_rsi(&self) -> f64 {
        if self.gains.len() < self.rsi_window {
            return 50.0; // Neutral
        }

        let avg_gain: f64 = self.gains.iter().sum::<f64>() / self.rsi_window as f64;
        let avg_loss: f64 = self.losses.iter().sum::<f64>() / self.rsi_window as f64;

        if avg_loss < 1e-10 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Calculate volume ratio
    fn calculate_volume_ratio(&self, current_volume: f64) -> f64 {
        if self.volumes.len() < 2 {
            return 1.0;
        }

        let avg_volume: f64 = self.volumes.iter().sum::<f64>() / self.volumes.len() as f64;

        if avg_volume < 1e-10 {
            return 1.0;
        }

        (current_volume / avg_volume).min(5.0).max(0.0) // Cap at 5x
    }

    /// Calculate price momentum
    fn calculate_momentum(&self, periods: usize) -> f64 {
        if self.prices.len() <= periods {
            return 0.0;
        }

        let current = self.prices.back().unwrap();
        let past = self.prices.get(self.prices.len() - periods - 1).unwrap();

        if *past < 1e-10 {
            return 0.0;
        }

        (current / past - 1.0).max(-1.0).min(1.0)
    }

    /// Reset the extractor state
    pub fn reset(&mut self) {
        self.prices.clear();
        self.returns.clear();
        self.volumes.clear();
        self.gains.clear();
        self.losses.clear();
        self.last_price = None;
        self.sample_count = 0;
    }

    /// Check if the extractor has enough data
    pub fn is_ready(&self) -> bool {
        self.sample_count >= self.min_samples
    }

    /// Get the number of samples processed
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }
}

/// Batch feature extraction from historical data
pub fn extract_features_batch(klines: &[Kline]) -> (Array2<f64>, Array1<f64>) {
    let mut extractor = FeatureExtractor::default_params();
    let mut features_list = Vec::new();
    let mut targets_list = Vec::new();

    for i in 0..klines.len() {
        if let Some(features) = extractor.update(&klines[i], None) {
            // Target: next return (if available)
            if i + 1 < klines.len() {
                let next_return = (klines[i + 1].close / klines[i].close).ln();
                features_list.push(features.to_array());
                targets_list.push(next_return);
            }
        }
    }

    let n_samples = features_list.len();
    let n_features = MarketFeatures::n_features();

    let mut features = Array2::zeros((n_samples, n_features));
    for (i, f) in features_list.iter().enumerate() {
        features.row_mut(i).assign(f);
    }

    let targets = Array1::from_vec(targets_list);

    (features, targets)
}

/// Feature scaler for normalization
pub struct FeatureScaler {
    /// Mean of each feature
    means: Array1<f64>,

    /// Standard deviation of each feature
    stds: Array1<f64>,

    /// Whether the scaler has been fitted
    fitted: bool,
}

impl FeatureScaler {
    /// Create a new scaler
    pub fn new() -> Self {
        Self {
            means: Array1::zeros(0),
            stds: Array1::ones(0),
            fitted: false,
        }
    }

    /// Fit the scaler on training data
    pub fn fit(&mut self, data: &Array2<f64>) {
        let n_features = data.ncols();

        self.means = data.mean_axis(ndarray::Axis(0)).unwrap();

        self.stds = Array1::zeros(n_features);
        for j in 0..n_features {
            let col = data.column(j);
            let mean = self.means[j];
            let variance = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (col.len() - 1) as f64;
            self.stds[j] = variance.sqrt().max(1e-8);
        }

        self.fitted = true;
    }

    /// Transform data using fitted parameters
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        if !self.fitted {
            return data.clone();
        }

        let mut result = data.clone();
        for j in 0..data.ncols() {
            let col = data.column(j);
            let transformed: Vec<f64> = col
                .iter()
                .map(|x| (x - self.means[j]) / self.stds[j])
                .collect();
            result.column_mut(j).assign(&Array1::from_vec(transformed));
        }

        result
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }

    /// Transform a single feature vector
    pub fn transform_one(&self, features: &Array1<f64>) -> Array1<f64> {
        if !self.fitted {
            return features.clone();
        }

        Array1::from_iter(
            features
                .iter()
                .zip(self.means.iter())
                .zip(self.stds.iter())
                .map(|((x, mean), std)| (x - mean) / std),
        )
    }
}

impl Default for FeatureScaler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_kline(close: f64, volume: f64) -> Kline {
        Kline {
            start_time: 0,
            open: close - 1.0,
            high: close + 1.0,
            low: close - 2.0,
            close,
            volume,
            turnover: close * volume,
        }
    }

    #[test]
    fn test_feature_extractor() {
        let mut extractor = FeatureExtractor::new(5, 5, 5);

        // Add enough data to get features
        for i in 0..10 {
            let price = 100.0 + (i as f64) * 0.5;
            let kline = create_test_kline(price, 1000.0);
            let features = extractor.update(&kline, None);

            if i >= 5 {
                assert!(features.is_some(), "Should have features after warmup");
            }
        }

        assert!(extractor.is_ready());
    }

    #[test]
    fn test_market_features() {
        let features = MarketFeatures {
            log_return: 0.01,
            realized_volatility: 0.02,
            rsi_normalized: 0.3,
            volume_ratio: 1.5,
            momentum_5: 0.02,
            momentum_10: 0.04,
            spread_normalized: 0.001,
            order_imbalance: 0.2,
        };

        let array = features.to_array();
        assert_eq!(array.len(), MarketFeatures::n_features());
        assert!((array[0] - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_feature_scaler() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
        )
        .unwrap();

        let mut scaler = FeatureScaler::new();
        let scaled = scaler.fit_transform(&data);

        // Check that scaled data has mean ~0 and std ~1
        let col0: Vec<f64> = scaled.column(0).to_vec();
        let mean0 = col0.iter().sum::<f64>() / col0.len() as f64;
        assert!(mean0.abs() < 0.01);
    }

    #[test]
    fn test_batch_extraction() {
        let klines: Vec<Kline> = (0..50)
            .map(|i| create_test_kline(100.0 + (i as f64) * 0.1, 1000.0))
            .collect();

        let (features, targets) = extract_features_batch(&klines);

        // Should have some valid samples after warmup
        assert!(features.nrows() > 0);
        assert!(targets.len() > 0);
        assert_eq!(features.nrows(), targets.len());
    }
}
