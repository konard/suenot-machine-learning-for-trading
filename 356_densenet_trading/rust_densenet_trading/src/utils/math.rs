//! Math Utilities
//!
//! Common mathematical functions for neural networks and trading.

/// Normalize values to [0, 1] range
pub fn normalize(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range == 0.0 {
        return vec![0.5; values.len()];
    }

    values.iter().map(|&v| (v - min) / range).collect()
}

/// Standardize values (z-score normalization)
pub fn standardize(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    if std == 0.0 {
        return vec![0.0; values.len()];
    }

    values.iter().map(|&v| (v - mean) / std).collect()
}

/// Softmax activation function
pub fn softmax(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = values.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exp_values.iter().sum();

    if sum == 0.0 {
        return vec![1.0 / values.len() as f64; values.len()];
    }

    exp_values.iter().map(|&v| v / sum).collect()
}

/// ReLU activation function
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Leaky ReLU activation function
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * x }
}

/// Sigmoid activation function
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Tanh activation function
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Clip values to range
pub fn clip(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

/// Calculate mean
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Calculate standard deviation
pub fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let m = mean(values);
    let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

/// Calculate correlation coefficient
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = mean(x);
    let mean_y = mean(y);

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 {
        return 0.0;
    }

    cov / denom
}

/// Calculate rolling mean
pub fn rolling_mean(values: &[f64], window: usize) -> Vec<f64> {
    if values.len() < window || window == 0 {
        return vec![f64::NAN; values.len()];
    }

    let mut result = vec![f64::NAN; window - 1];

    for i in (window - 1)..values.len() {
        let sum: f64 = values[(i + 1 - window)..=i].iter().sum();
        result.push(sum / window as f64);
    }

    result
}

/// Calculate rolling standard deviation
pub fn rolling_std(values: &[f64], window: usize) -> Vec<f64> {
    if values.len() < window || window == 0 {
        return vec![f64::NAN; values.len()];
    }

    let means = rolling_mean(values, window);
    let mut result = vec![f64::NAN; window - 1];

    for i in (window - 1)..values.len() {
        let m = means[i];
        let variance: f64 = values[(i + 1 - window)..=i]
            .iter()
            .map(|v| (v - m).powi(2))
            .sum::<f64>() / window as f64;
        result.push(variance.sqrt());
    }

    result
}

/// Exponential weighted moving average
pub fn ewma(values: &[f64], alpha: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut result = vec![values[0]];

    for i in 1..values.len() {
        let ema = alpha * values[i] + (1.0 - alpha) * result[i - 1];
        result.push(ema);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let values = vec![0.0, 50.0, 100.0];
        let normalized = normalize(&values);

        assert!((normalized[0] - 0.0).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
        assert!((normalized[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_standardize() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let standardized = standardize(&values);

        let mean: f64 = standardized.iter().sum::<f64>() / standardized.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_softmax() {
        let values = vec![1.0, 2.0, 3.0];
        let probs = softmax(&values);

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Larger values should have higher probabilities
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_activations() {
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(1.0), 1.0);

        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);

        assert!((tanh(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let corr = correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10);

        let z = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let neg_corr = correlation(&x, &z);
        assert!((neg_corr - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rm = rolling_mean(&values, 3);

        assert!(rm[0].is_nan());
        assert!(rm[1].is_nan());
        assert!((rm[2] - 2.0).abs() < 1e-10);
        assert!((rm[3] - 3.0).abs() < 1e-10);
        assert!((rm[4] - 4.0).abs() < 1e-10);
    }
}
