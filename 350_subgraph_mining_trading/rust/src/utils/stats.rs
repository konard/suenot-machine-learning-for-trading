//! Statistical functions for financial analysis.

use std::collections::HashMap;

/// Statistical utilities
pub struct Stats;

impl Stats {
    /// Calculate mean of a slice
    pub fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    /// Calculate variance
    pub fn variance(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = Self::mean(values);
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
    }

    /// Calculate standard deviation
    pub fn std_dev(values: &[f64]) -> f64 {
        Self::variance(values).sqrt()
    }

    /// Calculate Pearson correlation coefficient
    pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = Self::mean(x);
        let mean_y = Self::mean(y);

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
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

    /// Calculate correlation matrix for multiple series
    pub fn correlation_matrix(series: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = series.len();
        let mut matrix = vec![vec![1.0; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let corr = Self::correlation(&series[i], &series[j]);
                matrix[i][j] = corr;
                matrix[j][i] = corr;
            }
        }

        matrix
    }

    /// Calculate log returns from prices
    pub fn log_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return Vec::new();
        }

        prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }

    /// Calculate simple returns from prices
    pub fn simple_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return Vec::new();
        }

        prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Calculate Sharpe ratio
    pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = Self::mean(returns);
        let std = Self::std_dev(returns);

        if std == 0.0 {
            return 0.0;
        }

        // Annualized (assuming daily returns)
        let annual_factor = (252.0_f64).sqrt();
        (mean_return - risk_free_rate / 252.0) * annual_factor / std
    }

    /// Calculate Sortino ratio
    pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = Self::mean(returns);
        let downside: Vec<_> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

        let downside_std = if downside.is_empty() {
            1.0
        } else {
            let downside_var: f64 = downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
            downside_var.sqrt()
        };

        if downside_std == 0.0 {
            return 0.0;
        }

        let annual_factor = (252.0_f64).sqrt();
        (mean_return - risk_free_rate / 252.0) * annual_factor / downside_std
    }

    /// Calculate maximum drawdown
    pub fn max_drawdown(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut peak = values[0];
        let mut max_dd = 0.0;

        for &value in values {
            peak = peak.max(value);
            let dd = (peak - value) / peak;
            max_dd = max_dd.max(dd);
        }

        max_dd
    }

    /// Calculate cumulative returns
    pub fn cumulative_returns(returns: &[f64]) -> Vec<f64> {
        let mut cum = Vec::with_capacity(returns.len());
        let mut product = 1.0;

        for &r in returns {
            product *= 1.0 + r;
            cum.push(product - 1.0);
        }

        cum
    }

    /// Calculate rolling mean
    pub fn rolling_mean(values: &[f64], window: usize) -> Vec<f64> {
        if values.len() < window || window == 0 {
            return Vec::new();
        }

        values
            .windows(window)
            .map(|w| Self::mean(w))
            .collect()
    }

    /// Calculate rolling standard deviation
    pub fn rolling_std(values: &[f64], window: usize) -> Vec<f64> {
        if values.len() < window || window == 0 {
            return Vec::new();
        }

        values
            .windows(window)
            .map(|w| Self::std_dev(w))
            .collect()
    }

    /// Calculate rolling correlation
    pub fn rolling_correlation(x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
        if x.len() != y.len() || x.len() < window || window == 0 {
            return Vec::new();
        }

        (0..=x.len() - window)
            .map(|i| Self::correlation(&x[i..i + window], &y[i..i + window]))
            .collect()
    }

    /// Calculate percentile
    pub fn percentile(values: &[f64], p: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = (p * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Calculate skewness
    pub fn skewness(values: &[f64]) -> f64 {
        if values.len() < 3 {
            return 0.0;
        }

        let mean = Self::mean(values);
        let std = Self::std_dev(values);

        if std == 0.0 {
            return 0.0;
        }

        let n = values.len() as f64;
        let sum_cubed: f64 = values.iter().map(|v| ((v - mean) / std).powi(3)).sum();

        sum_cubed / n
    }

    /// Calculate kurtosis
    pub fn kurtosis(values: &[f64]) -> f64 {
        if values.len() < 4 {
            return 0.0;
        }

        let mean = Self::mean(values);
        let std = Self::std_dev(values);

        if std == 0.0 {
            return 0.0;
        }

        let n = values.len() as f64;
        let sum_fourth: f64 = values.iter().map(|v| ((v - mean) / std).powi(4)).sum();

        sum_fourth / n - 3.0 // Excess kurtosis
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((Stats::mean(&values) - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((Stats::correlation(&x, &y) - 1.0).abs() < 0.001);

        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert!((Stats::correlation(&x, &y_neg) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 105.0, 110.0];
        let returns = Stats::log_returns(&prices);
        assert_eq!(returns.len(), 2);
    }

    #[test]
    fn test_max_drawdown() {
        let values = vec![100.0, 110.0, 90.0, 95.0, 80.0, 85.0];
        let dd = Stats::max_drawdown(&values);
        // From 110 to 80 = 27.27% drawdown
        assert!((dd - 0.2727).abs() < 0.01);
    }

    #[test]
    fn test_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((Stats::percentile(&values, 0.5) - 3.0).abs() < 0.001);
    }
}
