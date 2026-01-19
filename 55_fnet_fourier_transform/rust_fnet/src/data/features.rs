//! Feature engineering for trading data.

use ndarray::{Array1, Array2, Axis};

use crate::api::types::Kline;

/// Configuration for feature calculation.
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// RSI period
    pub rsi_period: usize,
    /// Bollinger Bands period
    pub bb_period: usize,
    /// Bollinger Bands standard deviations
    pub bb_std: f64,
    /// Short momentum period
    pub momentum_short: usize,
    /// Medium momentum period
    pub momentum_medium: usize,
    /// Long momentum period
    pub momentum_long: usize,
    /// Volatility window
    pub volatility_window: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            bb_period: 20,
            bb_std: 2.0,
            momentum_short: 5,
            momentum_medium: 10,
            momentum_long: 20,
            volatility_window: 20,
        }
    }
}

/// Trading features computed from OHLCV data.
#[derive(Debug, Clone)]
pub struct TradingFeatures {
    /// Log returns
    pub log_returns: Vec<f64>,
    /// Rolling volatility
    pub volatility: Vec<f64>,
    /// Volume ratio (current / rolling average)
    pub volume_ratio: Vec<f64>,
    /// Short-term momentum
    pub momentum_short: Vec<f64>,
    /// Medium-term momentum
    pub momentum_medium: Vec<f64>,
    /// Long-term momentum
    pub momentum_long: Vec<f64>,
    /// RSI normalized to [-1, 1]
    pub rsi_normalized: Vec<f64>,
    /// Bollinger Band position (normalized)
    pub bb_position: Vec<f64>,
    /// Close prices
    pub close_prices: Vec<f64>,
    /// Timestamps
    pub timestamps: Vec<i64>,
}

impl TradingFeatures {
    /// Get number of valid samples (after removing NaN).
    pub fn len(&self) -> usize {
        self.log_returns.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.log_returns.is_empty()
    }

    /// Convert to feature matrix.
    pub fn to_matrix(&self) -> Array2<f64> {
        let n = self.len();
        let mut matrix = Array2::zeros((n, 8));

        for i in 0..n {
            matrix[[i, 0]] = self.log_returns[i];
            matrix[[i, 1]] = self.volatility[i];
            matrix[[i, 2]] = self.volume_ratio[i];
            matrix[[i, 3]] = self.momentum_short[i];
            matrix[[i, 4]] = self.momentum_medium[i];
            matrix[[i, 5]] = self.momentum_long[i];
            matrix[[i, 6]] = self.rsi_normalized[i];
            matrix[[i, 7]] = self.bb_position[i];
        }

        matrix
    }

    /// Get feature names.
    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "log_return",
            "volatility",
            "volume_ratio",
            "momentum_short",
            "momentum_medium",
            "momentum_long",
            "rsi_normalized",
            "bb_position",
        ]
    }
}

/// Calculate trading features from kline data.
pub fn calculate_features(klines: &[Kline], config: &FeatureConfig) -> TradingFeatures {
    let n = klines.len();

    // Extract base data
    let close: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let volume: Vec<f64> = klines.iter().map(|k| k.volume).collect();
    let timestamps: Vec<i64> = klines.iter().map(|k| k.timestamp).collect();

    // Calculate log returns
    let log_returns = calculate_log_returns(&close);

    // Calculate volatility
    let volatility = calculate_rolling_std(&log_returns, config.volatility_window);

    // Calculate volume ratio
    let volume_ratio = calculate_volume_ratio(&volume, config.volatility_window);

    // Calculate momentum at different periods
    let momentum_short = calculate_momentum(&close, config.momentum_short);
    let momentum_medium = calculate_momentum(&close, config.momentum_medium);
    let momentum_long = calculate_momentum(&close, config.momentum_long);

    // Calculate RSI
    let rsi = calculate_rsi(&close, config.rsi_period);
    let rsi_normalized: Vec<f64> = rsi.iter().map(|&r| (r - 50.0) / 50.0).collect();

    // Calculate Bollinger Band position
    let bb_position = calculate_bb_position(&close, config.bb_period, config.bb_std);

    // Find start index where all features are valid
    let start_idx = config
        .momentum_long
        .max(config.bb_period)
        .max(config.rsi_period)
        .max(config.volatility_window);

    // Trim all features to valid range
    TradingFeatures {
        log_returns: log_returns[start_idx..].to_vec(),
        volatility: volatility[start_idx..].to_vec(),
        volume_ratio: volume_ratio[start_idx..].to_vec(),
        momentum_short: momentum_short[start_idx..].to_vec(),
        momentum_medium: momentum_medium[start_idx..].to_vec(),
        momentum_long: momentum_long[start_idx..].to_vec(),
        rsi_normalized: rsi_normalized[start_idx..].to_vec(),
        bb_position: bb_position[start_idx..].to_vec(),
        close_prices: close[start_idx..].to_vec(),
        timestamps: timestamps[start_idx..].to_vec(),
    }
}

/// Calculate log returns.
fn calculate_log_returns(close: &[f64]) -> Vec<f64> {
    let mut returns = vec![0.0];
    for i in 1..close.len() {
        if close[i - 1] > 0.0 {
            returns.push((close[i] / close[i - 1]).ln());
        } else {
            returns.push(0.0);
        }
    }
    returns
}

/// Calculate rolling standard deviation.
fn calculate_rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];

    for i in window..n {
        let slice = &data[i - window..i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let variance: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window as f64;
        result[i] = variance.sqrt();
    }

    result
}

/// Calculate volume ratio.
fn calculate_volume_ratio(volume: &[f64], window: usize) -> Vec<f64> {
    let n = volume.len();
    let mut result = vec![1.0; n];

    for i in window..n {
        let avg: f64 = volume[i - window..i].iter().sum::<f64>() / window as f64;
        if avg > 0.0 {
            result[i] = volume[i] / avg;
        }
    }

    result
}

/// Calculate momentum (percentage change over period).
fn calculate_momentum(close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![0.0; n];

    for i in period..n {
        if close[i - period] > 0.0 {
            result[i] = (close[i] / close[i - period]) - 1.0;
        }
    }

    result
}

/// Calculate RSI (Relative Strength Index).
fn calculate_rsi(close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut rsi = vec![50.0; n];

    // Calculate price changes
    let mut gains = vec![0.0; n];
    let mut losses = vec![0.0; n];

    for i in 1..n {
        let change = close[i] - close[i - 1];
        if change > 0.0 {
            gains[i] = change;
        } else {
            losses[i] = -change;
        }
    }

    // Calculate initial averages
    if n <= period {
        return rsi;
    }

    let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

    for i in period..n {
        if i > period {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
        }

        if avg_loss > 0.0 {
            let rs = avg_gain / avg_loss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        } else if avg_gain > 0.0 {
            rsi[i] = 100.0;
        } else {
            rsi[i] = 50.0;
        }
    }

    rsi
}

/// Calculate Bollinger Band position (normalized).
fn calculate_bb_position(close: &[f64], period: usize, num_std: f64) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![0.0; n];

    for i in period..n {
        let slice = &close[i - period..i];
        let mean: f64 = slice.iter().sum::<f64>() / period as f64;
        let std: f64 = (slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64)
            .sqrt();

        if std > 0.0 {
            let upper = mean + num_std * std;
            let lower = mean - num_std * std;
            result[i] = (close[i] - lower) / (upper - lower) * 2.0 - 1.0;
        }
    }

    result
}

/// Normalize features using z-score normalization.
pub fn normalize_features(
    features: &Array2<f64>,
    mean: Option<&Array1<f64>>,
    std: Option<&Array1<f64>>,
) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let computed_mean = mean.cloned().unwrap_or_else(|| features.mean_axis(Axis(0)).unwrap());

    let computed_std = std.cloned().unwrap_or_else(|| {
        let n = features.nrows() as f64;
        let variance = features.map_axis(Axis(0), |col| {
            let m = col.mean().unwrap_or(0.0);
            col.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n
        });
        variance.mapv(|v| v.sqrt().max(1e-8))
    });

    let normalized = (features - &computed_mean) / &computed_std;

    (normalized, computed_mean, computed_std)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines() -> Vec<Kline> {
        let mut klines = Vec::new();
        let mut price = 100.0;

        for i in 0..100 {
            // Simulate price movement
            price *= 1.0 + 0.01 * (i as f64 * 0.1).sin();

            klines.push(Kline {
                timestamp: 1700000000 + i * 3600,
                open: price * 0.999,
                high: price * 1.01,
                low: price * 0.99,
                close: price,
                volume: 1000.0 * (1.0 + 0.5 * (i as f64 * 0.2).cos()),
                turnover: price * 1000.0,
            });
        }

        klines
    }

    #[test]
    fn test_calculate_features() {
        let klines = create_test_klines();
        let config = FeatureConfig::default();
        let features = calculate_features(&klines, &config);

        assert!(!features.is_empty());
        assert_eq!(features.log_returns.len(), features.volatility.len());
        assert_eq!(features.log_returns.len(), features.rsi_normalized.len());
    }

    #[test]
    fn test_feature_matrix() {
        let klines = create_test_klines();
        let config = FeatureConfig::default();
        let features = calculate_features(&klines, &config);

        let matrix = features.to_matrix();
        assert_eq!(matrix.ncols(), 8);
        assert_eq!(matrix.nrows(), features.len());
    }

    #[test]
    fn test_rsi_bounds() {
        let klines = create_test_klines();
        let config = FeatureConfig::default();
        let features = calculate_features(&klines, &config);

        for &rsi in &features.rsi_normalized {
            assert!(rsi >= -1.0 && rsi <= 1.0, "RSI should be in [-1, 1]");
        }
    }

    #[test]
    fn test_normalize_features() {
        let klines = create_test_klines();
        let config = FeatureConfig::default();
        let features = calculate_features(&klines, &config);
        let matrix = features.to_matrix();

        let (normalized, mean, std) = normalize_features(&matrix, None, None);

        // Check that normalized mean is close to 0
        let norm_mean = normalized.mean_axis(Axis(0)).unwrap();
        for &m in norm_mean.iter() {
            assert!(m.abs() < 1e-10, "Normalized mean should be ~0");
        }
    }
}
