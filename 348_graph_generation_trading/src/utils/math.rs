//! Mathematical utility functions.

/// Calculate mean of a slice
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Calculate variance of a slice
pub fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let n = values.len() as f64;
    values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)
}

/// Calculate standard deviation
pub fn std_dev(values: &[f64]) -> f64 {
    variance(values).sqrt()
}

/// Calculate covariance between two slices
pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let mean_x = mean(&x[..n]);
    let mean_y = mean(&y[..n]);

    let cov: f64 = x[..n]
        .iter()
        .zip(y[..n].iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    cov / (n - 1) as f64
}

/// Normalize values to [0, 1] range
pub fn normalize_minmax(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let range = max - min;
    if range < 1e-10 {
        return vec![0.5; values.len()];
    }

    values.iter().map(|x| (x - min) / range).collect()
}

/// Z-score normalization
pub fn normalize_zscore(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let m = mean(values);
    let s = std_dev(values);

    if s < 1e-10 {
        return vec![0.0; values.len()];
    }

    values.iter().map(|x| (x - m) / s).collect()
}

/// Calculate exponential moving average
pub fn ema(values: &[f64], period: usize) -> Vec<f64> {
    if values.is_empty() || period == 0 {
        return Vec::new();
    }

    let alpha = 2.0 / (period + 1) as f64;
    let mut result = Vec::with_capacity(values.len());

    // First value is just the first data point
    let mut current_ema = values[0];
    result.push(current_ema);

    for &value in &values[1..] {
        current_ema = alpha * value + (1.0 - alpha) * current_ema;
        result.push(current_ema);
    }

    result
}

/// Calculate simple moving average
pub fn sma(values: &[f64], period: usize) -> Vec<f64> {
    if values.len() < period || period == 0 {
        return Vec::new();
    }

    values
        .windows(period)
        .map(|window| window.iter().sum::<f64>() / period as f64)
        .collect()
}

/// Sigmoid function
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Softmax function
pub fn softmax(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = values.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();

    exp_vals.iter().map(|x| x / sum).collect()
}

/// Matrix multiplication
pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() || b.is_empty() || a[0].is_empty() {
        return Vec::new();
    }

    let m = a.len();
    let k = a[0].len();
    let n = b[0].len();

    if b.len() != k {
        return Vec::new();
    }

    let mut result = vec![vec![0.0; n]; m];

    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                result[i][j] += a[i][l] * b[l][j];
            }
        }
    }

    result
}

/// Transpose a matrix
pub fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Vec::new();
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    (0..cols)
        .map(|j| (0..rows).map(|i| matrix[i][j]).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&values) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = variance(&values);
        assert!(var > 0.0);
    }

    #[test]
    fn test_covariance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let cov = covariance(&x, &y);
        assert!(cov > 0.0); // Positive covariance
    }

    #[test]
    fn test_normalize_minmax() {
        let values = vec![0.0, 50.0, 100.0];
        let normalized = normalize_minmax(&values);

        assert!((normalized[0] - 0.0).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
        assert!((normalized[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema_values = ema(&values, 3);

        assert_eq!(ema_values.len(), 5);
        assert!(ema_values[4] > ema_values[0]);
    }

    #[test]
    fn test_softmax() {
        let values = vec![1.0, 2.0, 3.0];
        let probs = softmax(&values);

        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let c = matmul(&a, &b);

        assert_eq!(c[0][0], 19.0); // 1*5 + 2*7
        assert_eq!(c[1][1], 50.0); // 3*6 + 4*8
    }
}
