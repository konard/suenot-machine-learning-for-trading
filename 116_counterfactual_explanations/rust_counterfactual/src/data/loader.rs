//! Data loading utilities.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// OHLCV candle data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Create trading labels from returns.
///
/// # Arguments
///
/// * `returns` - Slice of return values
/// * `threshold` - Threshold for buy/sell signals
///
/// # Returns
///
/// Vector of labels (0=SELL, 1=HOLD, 2=BUY)
pub fn create_labels(returns: &[f64], threshold: f64) -> Vec<usize> {
    returns
        .iter()
        .map(|&r| {
            if r > threshold {
                2 // BUY
            } else if r < -threshold {
                0 // SELL
            } else {
                1 // HOLD
            }
        })
        .collect()
}

/// Get sample data for testing.
///
/// Generates synthetic trading data with correlated features.
///
/// # Returns
///
/// Tuple of (features, labels, feature_names)
pub fn get_sample_data() -> (Vec<Vec<f64>>, Vec<usize>, Vec<String>) {
    let mut rng = rand::thread_rng();
    let n_samples = 500;

    let mut features = Vec::with_capacity(n_samples);
    let mut labels = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // Generate correlated features
        let rsi: f64 = (rng.gen::<f64>() * 30.0 + 35.0).clamp(0.0, 100.0); // 35-65 range
        let rsi_normalized = (rsi - 50.0) / 50.0;

        let macd: f64 = rng.gen::<f64>() * 0.02 - 0.01; // -0.01 to 0.01
        let volume_ratio: f64 = (rng.gen::<f64>() * 0.6 + 0.7); // 0.7 to 1.3
        let volatility: f64 = rng.gen::<f64>() * 0.04; // 0 to 0.04
        let bb_position: f64 = rng.gen::<f64>() * 2.0 - 1.0; // -1 to 1
        let momentum: f64 = rng.gen::<f64>() * 0.1 - 0.05; // -0.05 to 0.05

        let sample = vec![
            rsi_normalized,
            macd,
            volume_ratio - 1.0,
            volatility,
            bb_position,
            momentum,
        ];

        // Create label based on features
        let label = if rsi_normalized < -0.4 && momentum > 0.0 {
            2 // BUY
        } else if rsi_normalized > 0.4 && momentum < 0.0 {
            0 // SELL
        } else {
            1 // HOLD
        };

        features.push(sample);
        labels.push(label);
    }

    let feature_names = vec![
        "rsi_normalized".to_string(),
        "macd".to_string(),
        "volume_ratio".to_string(),
        "volatility".to_string(),
        "bb_position".to_string(),
        "momentum".to_string(),
    ];

    (features, labels, feature_names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_labels() {
        let returns = vec![0.01, -0.01, 0.002, -0.002, 0.0];
        let labels = create_labels(&returns, 0.005);
        assert_eq!(labels, vec![2, 0, 1, 1, 1]);
    }

    #[test]
    fn test_get_sample_data() {
        let (features, labels, names) = get_sample_data();
        assert_eq!(features.len(), 500);
        assert_eq!(labels.len(), 500);
        assert_eq!(names.len(), 6);
    }
}
