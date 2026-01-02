//! Mathematical utility functions

/// Calculate Pearson correlation coefficient between two series
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|v| v * v).sum();
    let sum_y2: f64 = y.iter().map(|v| v * v).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x.powi(2)) * (n * sum_y2 - sum_y.powi(2))).sqrt();

    if denominator.abs() < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Apply softmax to a slice
pub fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }

    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();

    exp_vals.iter().map(|&v| v / sum).collect()
}

/// Normalize array to [0, 1] range
pub fn normalize(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }

    let min_val = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    if range.abs() < 1e-10 {
        vec![0.5; x.len()]
    } else {
        x.iter().map(|&v| (v - min_val) / range).collect()
    }
}

/// Compute z-score normalization
pub fn zscore(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }

    let mean: f64 = x.iter().sum::<f64>() / x.len() as f64;
    let variance: f64 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
    let std = variance.sqrt();

    if std.abs() < 1e-10 {
        vec![0.0; x.len()]
    } else {
        x.iter().map(|&v| (v - mean) / std).collect()
    }
}

/// Calculate rolling mean
pub fn rolling_mean(x: &[f64], window: usize) -> Vec<f64> {
    if x.len() < window || window == 0 {
        return vec![];
    }

    let mut result = Vec::with_capacity(x.len() - window + 1);
    let mut sum: f64 = x[..window].iter().sum();
    result.push(sum / window as f64);

    for i in window..x.len() {
        sum = sum - x[i - window] + x[i];
        result.push(sum / window as f64);
    }

    result
}

/// Calculate rolling standard deviation
pub fn rolling_std(x: &[f64], window: usize) -> Vec<f64> {
    if x.len() < window || window == 0 {
        return vec![];
    }

    let mut result = Vec::with_capacity(x.len() - window + 1);

    for i in 0..=(x.len() - window) {
        let slice = &x[i..i + window];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let var: f64 = slice.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / window as f64;
        result.push(var.sqrt());
    }

    result
}

/// Calculate log returns
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Calculate simple returns
pub fn simple_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10);

        let z = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr_neg = correlation(&x, &z);
        assert!((corr_neg + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax() {
        let x = vec![1.0, 2.0, 3.0];
        let result = softmax(&x);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let x = vec![0.0, 5.0, 10.0];
        let result = normalize(&x);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.5);
        assert_eq!(result[2], 1.0);
    }

    #[test]
    fn test_zscore() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = zscore(&x);
        let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
        assert!(mean.abs() < 1e-10);
    }
}
