//! Return calculation utilities.

use crate::api::types::Kline;

/// Calculate log returns from kline data.
///
/// log_return_t = ln(close_t / close_{t-1})
pub fn calculate_log_returns(klines: &[Kline]) -> Vec<f64> {
    if klines.len() < 2 {
        return Vec::new();
    }

    klines
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect()
}

/// Calculate simple (percentage) returns from kline data.
///
/// simple_return_t = (close_t - close_{t-1}) / close_{t-1}
pub fn calculate_simple_returns(klines: &[Kline]) -> Vec<f64> {
    if klines.len() < 2 {
        return Vec::new();
    }

    klines
        .windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect()
}

/// Calculate returns from a price vector.
pub fn returns_from_prices(prices: &[f64], log_returns: bool) -> Vec<f64> {
    if prices.len() < 2 {
        return Vec::new();
    }

    prices
        .windows(2)
        .map(|w| {
            if log_returns {
                (w[1] / w[0]).ln()
            } else {
                (w[1] - w[0]) / w[0]
            }
        })
        .collect()
}

/// Standardize returns (zero mean, unit variance).
pub fn standardize(returns: &[f64]) -> Vec<f64> {
    if returns.is_empty() {
        return Vec::new();
    }

    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    if std < 1e-10 {
        return returns.iter().map(|x| x - mean).collect();
    }

    returns.iter().map(|x| (x - mean) / std).collect()
}

/// Winsorize returns (clip extreme values).
pub fn winsorize(returns: &[f64], percentile: f64) -> Vec<f64> {
    if returns.is_empty() {
        return Vec::new();
    }

    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let lower_idx = (percentile * n as f64) as usize;
    let upper_idx = ((1.0 - percentile) * n as f64) as usize;

    let lower = sorted[lower_idx];
    let upper = sorted[upper_idx.min(n - 1)];

    returns.iter().map(|&x| x.clamp(lower, upper)).collect()
}

/// Generate synthetic returns with fat tails (Student-t distribution).
pub fn generate_synthetic_returns(n_samples: usize, df: f64) -> Vec<f64> {
    use rand::Rng;
    use rand_distr::{ChiSquared, StandardNormal};

    let mut rng = rand::thread_rng();
    let chi_sq = ChiSquared::new(df).unwrap();

    (0..n_samples)
        .map(|_| {
            let z: f64 = rng.sample(StandardNormal);
            let v: f64 = rng.sample(chi_sq);
            z / (v / df).sqrt() * 0.02 // Scale to ~2% daily vol
        })
        .collect()
}

/// Generate mixture distribution returns (regime switching).
pub fn generate_mixture_returns(n_samples: usize, volatile_prob: f64) -> Vec<f64> {
    use rand::Rng;
    use rand_distr::StandardNormal;

    let mut rng = rand::thread_rng();

    (0..n_samples)
        .map(|_| {
            let z: f64 = rng.sample(StandardNormal);
            if rng.gen::<f64>() < volatile_prob {
                // Volatile regime
                z * 0.04 - 0.002
            } else {
                // Calm regime
                z * 0.01 + 0.001
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_returns() {
        let klines = vec![
            Kline::new(0, 100.0, 102.0, 99.0, 101.0, 1000.0),
            Kline::new(1, 101.0, 103.0, 100.0, 102.0, 1000.0),
            Kline::new(2, 102.0, 104.0, 101.0, 103.0, 1000.0),
        ];

        let returns = calculate_log_returns(&klines);
        assert_eq!(returns.len(), 2);

        // First return: ln(102/101) ≈ 0.00985
        assert!((returns[0] - (102.0_f64 / 101.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_standardize() {
        let returns = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std_returns = standardize(&returns);

        // Mean should be ~0
        let mean: f64 = std_returns.iter().sum::<f64>() / std_returns.len() as f64;
        assert!(mean.abs() < 1e-10);

        // Std should be ~1
        let variance: f64 = std_returns.iter().map(|x| x.powi(2)).sum::<f64>() / std_returns.len() as f64;
        assert!((variance - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_winsorize() {
        let returns = vec![-0.5, -0.1, 0.0, 0.1, 0.5];
        let winsorized = winsorize(&returns, 0.2);

        // Extreme values should be clipped
        assert!(winsorized.iter().all(|&x| x >= -0.1 && x <= 0.1));
    }
}
