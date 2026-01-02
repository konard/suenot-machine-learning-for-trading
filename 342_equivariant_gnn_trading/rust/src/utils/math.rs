//! Math Utilities

pub use crate::features::normalizer::{normalize, standardize};

/// Calculate correlation matrix
pub fn correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut corr = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            corr[i][j] = if i == j { 1.0 } else { pearson_correlation(&data[i], &data[j]) };
        }
    }
    corr
}

/// Calculate Pearson correlation
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return 0.0; }
    let mean_x = x.iter().take(n).sum::<f64>() / n as f64;
    let mean_y = y.iter().take(n).sum::<f64>() / n as f64;
    let (mut cov, mut var_x, mut var_y) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (dx, dy) = (x[i] - mean_x, y[i] - mean_y);
        cov += dx * dy; var_x += dx * dx; var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 { 0.0 } else { cov / denom }
}
