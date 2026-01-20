//! Information Coefficient (IC) Calculator
//!
//! Calculates IC and related metrics for factor evaluation.

use statrs::statistics::{Data, Distribution, OrderStatistics};

/// Information Coefficient calculator
pub struct IcCalculator {
    /// Forward return periods for IC calculation
    forward_periods: Vec<usize>,
}

impl IcCalculator {
    /// Create a new IC calculator with default forward periods
    pub fn new() -> Self {
        Self {
            forward_periods: vec![1, 5, 10, 20],
        }
    }

    /// Create with custom forward periods
    pub fn with_periods(periods: Vec<usize>) -> Self {
        Self {
            forward_periods: periods,
        }
    }

    /// Calculate Pearson correlation (used for IC)
    pub fn pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return None;
        }

        let n = x.len() as f64;

        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;

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

        if var_x == 0.0 || var_y == 0.0 {
            return None;
        }

        Some(cov / (var_x.sqrt() * var_y.sqrt()))
    }

    /// Calculate Spearman rank correlation (RankIC)
    pub fn spearman_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return None;
        }

        let rank_x = Self::rank(x);
        let rank_y = Self::rank(y);

        Self::pearson_correlation(&rank_x, &rank_y)
    }

    /// Calculate ranks for a series
    fn rank(values: &[f64]) -> Vec<f64> {
        let n = values.len();
        let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();

        // Sort by value
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign ranks (handling ties by averaging)
        let mut ranks = vec![0.0; n];
        let mut i = 0;

        while i < n {
            let mut j = i;
            // Find all elements with the same value
            while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
                j += 1;
            }

            // Average rank for ties
            let avg_rank = (i + j) as f64 / 2.0 + 0.5;
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }

            i = j;
        }

        ranks
    }

    /// Calculate IC for factor values and forward returns
    pub fn calculate_ic(&self, factor_values: &[f64], forward_returns: &[f64]) -> Option<f64> {
        Self::pearson_correlation(factor_values, forward_returns)
    }

    /// Calculate Rank IC for factor values and forward returns
    pub fn calculate_rank_ic(&self, factor_values: &[f64], forward_returns: &[f64]) -> Option<f64> {
        Self::spearman_correlation(factor_values, forward_returns)
    }

    /// Calculate rolling IC time series
    pub fn rolling_ic(
        &self,
        factor_values: &[f64],
        forward_returns: &[f64],
        window: usize,
    ) -> Vec<f64> {
        if factor_values.len() != forward_returns.len() || factor_values.len() < window {
            return vec![];
        }

        let mut ic_series = Vec::with_capacity(factor_values.len() - window + 1);

        for i in 0..=(factor_values.len() - window) {
            let factor_window = &factor_values[i..i + window];
            let return_window = &forward_returns[i..i + window];

            if let Some(ic) = Self::pearson_correlation(factor_window, return_window) {
                ic_series.push(ic);
            } else {
                ic_series.push(0.0);
            }
        }

        ic_series
    }

    /// Calculate IC statistics (mean, std, IR)
    pub fn ic_statistics(&self, ic_series: &[f64]) -> IcStats {
        if ic_series.is_empty() {
            return IcStats::default();
        }

        let n = ic_series.len() as f64;
        let mean = ic_series.iter().sum::<f64>() / n;

        let variance: f64 = ic_series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();

        let ir = if std > 0.0 { mean / std } else { 0.0 };

        // Calculate hit rate (% of positive IC)
        let positive_count = ic_series.iter().filter(|&&x| x > 0.0).count();
        let hit_rate = positive_count as f64 / n;

        // Calculate t-statistic
        let t_stat = if std > 0.0 {
            mean / (std / n.sqrt())
        } else {
            0.0
        };

        IcStats {
            mean,
            std,
            ir,
            hit_rate,
            t_stat,
            count: ic_series.len(),
        }
    }

    /// Calculate forward returns from prices
    pub fn calculate_forward_returns(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() <= period {
            return vec![];
        }

        prices
            .windows(period + 1)
            .map(|w| (w[period] - w[0]) / w[0])
            .collect()
    }

    /// Get forward periods
    pub fn forward_periods(&self) -> &[usize] {
        &self.forward_periods
    }
}

impl Default for IcCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// IC statistics
#[derive(Debug, Clone, Default)]
pub struct IcStats {
    /// Mean IC
    pub mean: f64,
    /// Standard deviation of IC
    pub std: f64,
    /// Information Ratio (IC/std)
    pub ir: f64,
    /// Hit rate (% positive IC)
    pub hit_rate: f64,
    /// T-statistic
    pub t_stat: f64,
    /// Number of observations
    pub count: usize,
}

impl IcStats {
    /// Check if IC is statistically significant (|t| > 2)
    pub fn is_significant(&self) -> bool {
        self.t_stat.abs() > 2.0
    }

    /// Check if factor meets quality threshold
    pub fn meets_threshold(&self, min_ic: f64, min_ir: f64) -> bool {
        self.mean.abs() >= min_ic && self.ir.abs() >= min_ir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_correlation_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let corr = IcCalculator::pearson_correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_correlation_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];

        let corr = IcCalculator::pearson_correlation(&x, &y).unwrap();
        assert!((corr - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_correlation_zero() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, -1.0, 1.0, -1.0, 1.0];

        let corr = IcCalculator::pearson_correlation(&x, &y).unwrap();
        // Should be close to zero (not exactly due to finite samples)
        assert!(corr.abs() < 0.5);
    }

    #[test]
    fn test_rank() {
        let values = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let ranks = IcCalculator::rank(&values);

        // 1.0 appears twice at positions 1 and 3, average rank = 1.5
        // 3.0 is at position 2, rank = 3
        // 4.0 is at position 3, rank = 4
        // 5.0 is at position 4, rank = 5
        assert!((ranks[0] - 3.0).abs() < 1e-10); // 3.0 -> rank 3
        assert!((ranks[1] - 1.5).abs() < 1e-10); // 1.0 -> rank 1.5
        assert!((ranks[2] - 4.0).abs() < 1e-10); // 4.0 -> rank 4
        assert!((ranks[3] - 1.5).abs() < 1e-10); // 1.0 -> rank 1.5
        assert!((ranks[4] - 5.0).abs() < 1e-10); // 5.0 -> rank 5
    }

    #[test]
    fn test_spearman_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 6.0, 7.0, 8.0, 7.0]; // Monotonic relationship

        let corr = IcCalculator::spearman_correlation(&x, &y).unwrap();
        assert!(corr > 0.8); // Strong positive correlation
    }

    #[test]
    fn test_forward_returns() {
        let prices = vec![100.0, 110.0, 105.0, 115.0, 120.0];
        let returns = IcCalculator::calculate_forward_returns(&prices, 1);

        assert_eq!(returns.len(), 4);
        assert!((returns[0] - 0.1).abs() < 1e-10); // (110-100)/100
        assert!((returns[1] - (-0.0454545)).abs() < 0.0001); // (105-110)/110
    }

    #[test]
    fn test_ic_statistics() {
        let calc = IcCalculator::new();
        let ic_series = vec![0.03, 0.02, -0.01, 0.04, 0.02, 0.03, 0.01, 0.02];

        let stats = calc.ic_statistics(&ic_series);

        assert!(stats.mean > 0.01 && stats.mean < 0.03);
        assert!(stats.ir > 0.0);
        assert!(stats.hit_rate > 0.7);
    }
}
