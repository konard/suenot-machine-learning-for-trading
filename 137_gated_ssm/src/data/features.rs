//! Feature engineering for financial time series.

use super::bybit::BybitKline;

/// Compute trading features from kline data.
///
/// Features per bar:
/// 1. Log return
/// 2. Normalized range (high - low) / close
/// 3. Body ratio (close - open) / (high - low + epsilon)
/// 4. Log volume change
///
/// # Returns
/// Vector of feature vectors (one per bar, skipping the first).
pub fn compute_features(klines: &[BybitKline]) -> Vec<Vec<f64>> {
    if klines.len() < 2 {
        return Vec::new();
    }

    let mut features = Vec::with_capacity(klines.len() - 1);

    for i in 1..klines.len() {
        let prev = &klines[i - 1];
        let curr = &klines[i];

        // Log return
        let log_ret = (curr.close / (prev.close + 1e-10)).ln();

        // Normalized range
        let range = (curr.high - curr.low) / (curr.close + 1e-10);

        // Body ratio
        let body = (curr.close - curr.open) / (curr.high - curr.low + 1e-10);

        // Log volume change
        let vol_change = ((curr.volume + 1.0) / (prev.volume + 1.0)).ln();

        features.push(vec![log_ret, range, body, vol_change]);
    }

    features
}

/// Compute rolling statistics for feature normalization.
pub fn rolling_mean(values: &[f64], window: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        if i < window - 1 {
            result.push(f64::NAN);
        } else {
            let sum: f64 = values[i + 1 - window..=i].iter().sum();
            result.push(sum / window as f64);
        }
    }
    result
}

/// Compute rolling standard deviation.
pub fn rolling_std(values: &[f64], window: usize) -> Vec<f64> {
    let means = rolling_mean(values, window);
    let mut result = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        if i < window - 1 {
            result.push(f64::NAN);
        } else {
            let mean = means[i];
            let var: f64 = values[i + 1 - window..=i]
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / window as f64;
            result.push(var.sqrt());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_features() {
        let klines = vec![
            BybitKline {
                open_time: 1000,
                open: 100.0,
                high: 105.0,
                low: 98.0,
                close: 103.0,
                volume: 1000.0,
            },
            BybitKline {
                open_time: 2000,
                open: 103.0,
                high: 108.0,
                low: 101.0,
                close: 106.0,
                volume: 1200.0,
            },
        ];
        let features = compute_features(&klines);
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].len(), 4);
        assert!(features[0].iter().all(|f| f.is_finite()));
    }

    #[test]
    fn test_rolling_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let means = rolling_mean(&values, 3);
        assert!(means[0].is_nan());
        assert!(means[1].is_nan());
        assert!((means[2] - 2.0).abs() < 1e-10);
        assert!((means[3] - 3.0).abs() < 1e-10);
        assert!((means[4] - 4.0).abs() < 1e-10);
    }
}
