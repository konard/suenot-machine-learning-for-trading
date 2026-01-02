//! Correlation calculation methods for financial time series.

use super::builder::CorrelationMethod;

/// A correlation matrix storing pairwise correlations
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    /// The correlation values (symmetric matrix)
    data: Vec<Vec<f64>>,
    /// Number of assets
    size: usize,
}

impl CorrelationMatrix {
    /// Create a new correlation matrix of given size
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![vec![0.0; size]; size],
            size,
        }
    }

    /// Create correlation matrix from raw data
    pub fn from_matrix(data: Vec<Vec<f64>>) -> Self {
        let size = data.len();
        Self { data, size }
    }

    /// Calculate correlation matrix from returns
    pub fn from_returns(returns: &[Vec<f64>], method: CorrelationMethod) -> Self {
        let n = returns.len();
        let mut matrix = Self::new(n);

        for i in 0..n {
            matrix.data[i][i] = 1.0; // Diagonal is always 1

            for j in (i + 1)..n {
                let corr = match method {
                    CorrelationMethod::Pearson => pearson_correlation(&returns[i], &returns[j]),
                    CorrelationMethod::Spearman => spearman_correlation(&returns[i], &returns[j]),
                    CorrelationMethod::Kendall => kendall_correlation(&returns[i], &returns[j]),
                };

                matrix.data[i][j] = corr;
                matrix.data[j][i] = corr; // Symmetric
            }
        }

        matrix
    }

    /// Get correlation between assets i and j
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i < self.size && j < self.size {
            self.data[i][j]
        } else {
            0.0
        }
    }

    /// Set correlation between assets i and j
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        if i < self.size && j < self.size {
            self.data[i][j] = value;
            self.data[j][i] = value;
        }
    }

    /// Get the size of the matrix
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the raw data
    pub fn data(&self) -> &Vec<Vec<f64>> {
        &self.data
    }

    /// Calculate average correlation (excluding diagonal)
    pub fn average_correlation(&self) -> f64 {
        if self.size < 2 {
            return 0.0;
        }

        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..self.size {
            for j in (i + 1)..self.size {
                sum += self.data[i][j];
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    /// Get row as vector
    pub fn row(&self, i: usize) -> Vec<f64> {
        if i < self.size {
            self.data[i].clone()
        } else {
            Vec::new()
        }
    }
}

/// Calculate Pearson correlation coefficient
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let n_f = n as f64;

    let mean_x: f64 = x[..n].iter().sum::<f64>() / n_f;
    let mean_y: f64 = y[..n].iter().sum::<f64>() / n_f;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    cov / denom
}

/// Calculate Spearman rank correlation
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    // Convert to ranks
    let rank_x = to_ranks(&x[..n]);
    let rank_y = to_ranks(&y[..n]);

    // Calculate Pearson correlation on ranks
    pearson_correlation(&rank_x, &rank_y)
}

/// Calculate Kendall tau correlation
pub fn kendall_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let mut concordant = 0i64;
    let mut discordant = 0i64;

    for i in 0..n {
        for j in (i + 1)..n {
            let x_diff = x[j] - x[i];
            let y_diff = y[j] - y[i];
            let product = x_diff * y_diff;

            if product > 0.0 {
                concordant += 1;
            } else if product < 0.0 {
                discordant += 1;
            }
            // Ties are ignored
        }
    }

    let total = concordant + discordant;
    if total == 0 {
        return 0.0;
    }

    (concordant - discordant) as f64 / total as f64
}

/// Convert values to ranks (average rank for ties)
fn to_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();

    // Sort by value
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;

    while i < n {
        let mut j = i;

        // Find all elements with the same value (ties)
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
            j += 1;
        }

        // Calculate average rank for ties
        let avg_rank = (i + j + 1) as f64 / 2.0;

        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }

        i = j;
    }

    ranks
}

/// Calculate rolling correlation
pub fn rolling_correlation(x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
    let n = x.len().min(y.len());
    if n < window || window < 2 {
        return Vec::new();
    }

    (0..=(n - window))
        .map(|i| pearson_correlation(&x[i..i + window], &y[i..i + window]))
        .collect()
}

/// Calculate exponentially weighted correlation
pub fn ewm_correlation(x: &[f64], y: &[f64], span: f64) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let alpha = 2.0 / (span + 1.0);

    // Calculate weighted means
    let mut weight_sum = 0.0;
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;

    for i in 0..n {
        let w = (1.0 - alpha).powi((n - 1 - i) as i32);
        weight_sum += w;
        mean_x += w * x[i];
        mean_y += w * y[i];
    }

    mean_x /= weight_sum;
    mean_y /= weight_sum;

    // Calculate weighted covariance and variances
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let w = (1.0 - alpha).powi((n - 1 - i) as i32);
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += w * dx * dy;
        var_x += w * dx * dx;
        var_y += w * dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    cov / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let corr = pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];

        let corr = pearson_correlation(&x, &y);
        assert!((corr + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_uncorrelated() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 1.0, 4.0, 1.0, 5.0]; // Random-ish

        let corr = pearson_correlation(&x, &y);
        assert!(corr.abs() < 0.5);
    }

    #[test]
    fn test_spearman_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 6.0, 7.0, 8.0, 7.0];

        let corr = spearman_correlation(&x, &y);
        assert!(corr > 0.5); // Should be positive
    }

    #[test]
    fn test_to_ranks() {
        let values = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let ranks = to_ranks(&values);

        // 1.0 appears twice, should have rank 1.5
        assert!((ranks[1] - 1.5).abs() < 1e-10);
        assert!((ranks[3] - 1.5).abs() < 1e-10);

        // 3.0 is the middle value, rank 3
        assert!((ranks[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_matrix() {
        let returns = vec![
            vec![0.01, 0.02, -0.01, 0.015],
            vec![0.012, 0.018, -0.008, 0.014],
            vec![-0.005, 0.03, 0.01, -0.02],
        ];

        let matrix = CorrelationMatrix::from_returns(&returns, CorrelationMethod::Pearson);

        // Diagonal should be 1
        assert!((matrix.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((matrix.get(1, 1) - 1.0).abs() < 1e-10);

        // Should be symmetric
        assert!((matrix.get(0, 1) - matrix.get(1, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let rolling = rolling_correlation(&x, &y, 3);

        // All should be ~1.0 since both series are identical
        for corr in rolling {
            assert!((corr - 1.0).abs() < 1e-10);
        }
    }
}
