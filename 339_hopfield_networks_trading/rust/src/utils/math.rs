//! Mathematical utility functions
//!
//! This module provides common mathematical operations used in
//! Hopfield Networks and trading calculations.

use std::f64::consts::E;

/// Sigmoid activation function
///
/// σ(x) = 1 / (1 + e^(-x))
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

/// Derivative of sigmoid function
///
/// σ'(x) = σ(x) * (1 - σ(x))
pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// Hyperbolic tangent activation
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Derivative of tanh
pub fn tanh_derivative(x: f64) -> f64 {
    1.0 - x.tanh().powi(2)
}

/// ReLU activation function
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Leaky ReLU activation
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * x }
}

/// Softmax function for a vector
pub fn softmax(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = values.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();

    exp_vals.iter().map(|&x| x / sum).collect()
}

/// Log-sum-exp trick for numerical stability
pub fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = values.iter().map(|&x| (x - max_val).exp()).sum();

    max_val + sum.ln()
}

/// Sign function
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Clip value to range
pub fn clip(x: f64, min: f64, max: f64) -> f64 {
    x.max(min).min(max)
}

/// Linear interpolation
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Dot product of two vectors
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm (Euclidean length) of a vector
pub fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
}

/// L1 norm (Manhattan distance) of a vector
pub fn l1_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).sum()
}

/// Normalize a vector to unit length
pub fn normalize_vector(v: &[f64]) -> Vec<f64> {
    let norm = l2_norm(v);
    if norm < 1e-10 {
        return v.to_vec();
    }
    v.iter().map(|&x| x / norm).collect()
}

/// Element-wise vector addition
pub fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Element-wise vector subtraction
pub fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Element-wise vector multiplication
pub fn vec_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Scalar multiplication of a vector
pub fn vec_scale(v: &[f64], scalar: f64) -> Vec<f64> {
    v.iter().map(|&x| x * scalar).collect()
}

/// Mean of a vector
pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Variance of a vector
pub fn variance(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
}

/// Standard deviation of a vector
pub fn std_dev(v: &[f64]) -> f64 {
    variance(v).sqrt()
}

/// Covariance between two vectors
pub fn covariance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    if a.len() < 2 {
        return 0.0;
    }

    let mean_a = mean(a);
    let mean_b = mean(b);

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum::<f64>()
        / a.len() as f64
}

/// Pearson correlation coefficient
pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let cov = covariance(a, b);
    let std_a = std_dev(a);
    let std_b = std_dev(b);

    if std_a < 1e-10 || std_b < 1e-10 {
        return 0.0;
    }

    cov / (std_a * std_b)
}

/// Outer product of two vectors (creates a matrix as flattened vector)
pub fn outer_product(a: &[f64], b: &[f64]) -> Vec<Vec<f64>> {
    a.iter().map(|&x| b.iter().map(|&y| x * y).collect()).collect()
}

/// Calculate Sharpe ratio
///
/// # Arguments
/// * `returns` - Vector of returns
/// * `risk_free_rate` - Annualized risk-free rate
/// * `periods_per_year` - Number of periods in a year (e.g., 252 for daily, 52 for weekly)
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mean_return = mean(returns);
    let std_return = std_dev(returns);

    if std_return < 1e-10 {
        return 0.0;
    }

    let excess_return = mean_return - risk_free_rate / periods_per_year;
    let annualized_return = excess_return * periods_per_year;
    let annualized_std = std_return * periods_per_year.sqrt();

    annualized_return / annualized_std
}

/// Calculate Sortino ratio (uses downside deviation)
pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mean_return = mean(returns);
    let target = risk_free_rate / periods_per_year;

    // Calculate downside deviation
    let downside_returns: Vec<f64> = returns
        .iter()
        .filter(|&&r| r < target)
        .map(|&r| (r - target).powi(2))
        .collect();

    if downside_returns.is_empty() {
        return f64::INFINITY;
    }

    let downside_dev = (downside_returns.iter().sum::<f64>() / downside_returns.len() as f64).sqrt();

    if downside_dev < 1e-10 {
        return f64::INFINITY;
    }

    let excess_return = mean_return - target;
    let annualized_return = excess_return * periods_per_year;
    let annualized_downside = downside_dev * periods_per_year.sqrt();

    annualized_return / annualized_downside
}

/// Calculate maximum drawdown
pub fn max_drawdown(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let mut peak = values[0];
    let mut max_dd = 0.0;

    for &value in values.iter() {
        if value > peak {
            peak = value;
        }
        let dd = (peak - value) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

/// Calculate cumulative returns from simple returns
pub fn cumulative_returns(returns: &[f64]) -> Vec<f64> {
    let mut cumulative = Vec::with_capacity(returns.len() + 1);
    cumulative.push(1.0);

    for &r in returns {
        let prev = *cumulative.last().unwrap();
        cumulative.push(prev * (1.0 + r));
    }

    cumulative
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_softmax() {
        let values = vec![1.0, 2.0, 3.0];
        let result = softmax(&values);

        // Sum should be 1
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Last element should be largest
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Perfect positive correlation
        assert!((correlation(&a, &b) - 1.0).abs() < 1e-10);

        // Perfect negative correlation
        let c = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((correlation(&a, &c) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.03, 0.01];
        let sharpe = sharpe_ratio(&returns, 0.02, 252.0);
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_max_drawdown() {
        let values = vec![100.0, 110.0, 105.0, 95.0, 100.0, 90.0, 95.0];
        let dd = max_drawdown(&values);

        // Max drawdown should be from 110 to 90 = 18.18%
        assert!((dd - 0.1818).abs() < 0.01);
    }

    #[test]
    fn test_cumulative_returns() {
        let returns = vec![0.1, -0.05, 0.15];
        let cumulative = cumulative_returns(&returns);

        assert_eq!(cumulative.len(), 4);
        assert!((cumulative[0] - 1.0).abs() < 1e-10);
        assert!((cumulative[1] - 1.1).abs() < 1e-10);
        assert!((cumulative[2] - 1.045).abs() < 1e-10);
    }
}
