//! Returns calculation and statistics utilities.
//!
//! Provides tools for calculating various types of returns and their statistics
//! from price data.

use super::Kline;

/// Wrapper for return series with statistical utilities
#[derive(Debug, Clone)]
pub struct Returns {
    /// Raw return values
    pub values: Vec<f64>,
    /// Timestamps corresponding to returns
    pub timestamps: Vec<i64>,
}

impl Returns {
    /// Create returns from kline data (close-to-close returns)
    pub fn from_klines(klines: &[Kline]) -> Self {
        if klines.len() < 2 {
            return Self {
                values: vec![],
                timestamps: vec![],
            };
        }

        let mut values = Vec::with_capacity(klines.len() - 1);
        let mut timestamps = Vec::with_capacity(klines.len() - 1);

        for i in 1..klines.len() {
            let ret = (klines[i].close - klines[i - 1].close) / klines[i - 1].close;
            values.push(ret);
            timestamps.push(klines[i].timestamp);
        }

        Self { values, timestamps }
    }

    /// Create log returns from kline data
    pub fn log_returns_from_klines(klines: &[Kline]) -> Self {
        if klines.len() < 2 {
            return Self {
                values: vec![],
                timestamps: vec![],
            };
        }

        let mut values = Vec::with_capacity(klines.len() - 1);
        let mut timestamps = Vec::with_capacity(klines.len() - 1);

        for i in 1..klines.len() {
            let ret = (klines[i].close / klines[i - 1].close).ln();
            values.push(ret);
            timestamps.push(klines[i].timestamp);
        }

        Self { values, timestamps }
    }

    /// Calculate the mean return
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Calculate the standard deviation
    pub fn std(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f64 = self
            .values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (self.values.len() - 1) as f64;
        variance.sqrt()
    }

    /// Calculate the variance
    pub fn variance(&self) -> f64 {
        self.std().powi(2)
    }

    /// Calculate the Sharpe ratio (assuming risk-free rate = 0)
    /// Annualization factor depends on the data frequency
    pub fn sharpe_ratio(&self, annualization_factor: f64) -> f64 {
        let mean = self.mean();
        let std = self.std();
        if std == 0.0 {
            return 0.0;
        }
        (mean * annualization_factor) / (std * annualization_factor.sqrt())
    }

    /// Calculate skewness
    pub fn skewness(&self) -> f64 {
        if self.values.len() < 3 {
            return 0.0;
        }
        let mean = self.mean();
        let std = self.std();
        if std == 0.0 {
            return 0.0;
        }
        let n = self.values.len() as f64;
        let skew: f64 = self
            .values
            .iter()
            .map(|x| ((x - mean) / std).powi(3))
            .sum::<f64>()
            / n;
        skew
    }

    /// Calculate kurtosis (excess kurtosis, normal = 0)
    pub fn kurtosis(&self) -> f64 {
        if self.values.len() < 4 {
            return 0.0;
        }
        let mean = self.mean();
        let std = self.std();
        if std == 0.0 {
            return 0.0;
        }
        let n = self.values.len() as f64;
        let kurt: f64 = self
            .values
            .iter()
            .map(|x| ((x - mean) / std).powi(4))
            .sum::<f64>()
            / n;
        kurt - 3.0 // Excess kurtosis
    }

    /// Count positive returns
    pub fn count_positive(&self) -> usize {
        self.values.iter().filter(|&&x| x > 0.0).count()
    }

    /// Count negative returns
    pub fn count_negative(&self) -> usize {
        self.values.iter().filter(|&&x| x < 0.0).count()
    }

    /// Calculate win rate (proportion of positive returns)
    pub fn win_rate(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.count_positive() as f64 / self.values.len() as f64
    }

    /// Get the maximum return
    pub fn max(&self) -> f64 {
        self.values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get the minimum return
    pub fn min(&self) -> f64 {
        self.values.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Calculate Value at Risk (VaR) at given confidence level
    pub fn var(&self, confidence: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    /// Calculate Conditional VaR (Expected Shortfall)
    pub fn cvar(&self, confidence: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let var = self.var(confidence);
        let tail: Vec<f64> = self.values.iter().filter(|&&x| x <= var).copied().collect();
        if tail.is_empty() {
            return var;
        }
        tail.iter().sum::<f64>() / tail.len() as f64
    }

    /// Calculate cumulative returns
    pub fn cumulative(&self) -> Vec<f64> {
        let mut cum = Vec::with_capacity(self.values.len());
        let mut total = 1.0;
        for &ret in &self.values {
            total *= 1.0 + ret;
            cum.push(total - 1.0);
        }
        cum
    }

    /// Calculate maximum drawdown
    pub fn max_drawdown(&self) -> f64 {
        let cumulative = self.cumulative();
        let mut max_dd = 0.0;
        let mut peak = 0.0_f64;

        for cum_ret in cumulative {
            if cum_ret > peak {
                peak = cum_ret;
            }
            let dd = (peak - cum_ret) / (1.0 + peak);
            if dd > max_dd {
                max_dd = dd;
            }
        }
        max_dd
    }

    /// Get number of observations
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get a slice of returns
    pub fn slice(&self, start: usize, end: usize) -> Self {
        let end = end.min(self.values.len());
        let start = start.min(end);
        Self {
            values: self.values[start..end].to_vec(),
            timestamps: self.timestamps[start..end].to_vec(),
        }
    }

    /// Calculate rolling statistics
    pub fn rolling_mean(&self, window: usize) -> Vec<f64> {
        if window == 0 || window > self.values.len() {
            return vec![];
        }
        let mut result = Vec::with_capacity(self.values.len() - window + 1);
        for i in 0..=(self.values.len() - window) {
            let sum: f64 = self.values[i..i + window].iter().sum();
            result.push(sum / window as f64);
        }
        result
    }

    /// Calculate rolling standard deviation
    pub fn rolling_std(&self, window: usize) -> Vec<f64> {
        if window < 2 || window > self.values.len() {
            return vec![];
        }
        let mut result = Vec::with_capacity(self.values.len() - window + 1);
        for i in 0..=(self.values.len() - window) {
            let slice = &self.values[i..i + window];
            let mean: f64 = slice.iter().sum::<f64>() / window as f64;
            let variance: f64 =
                slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window - 1) as f64;
            result.push(variance.sqrt());
        }
        result
    }
}

/// Calculate correlation between two return series
pub fn correlation(returns1: &Returns, returns2: &Returns) -> f64 {
    let n = returns1.len().min(returns2.len());
    if n < 2 {
        return 0.0;
    }

    let mean1: f64 = returns1.values[..n].iter().sum::<f64>() / n as f64;
    let mean2: f64 = returns2.values[..n].iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var1 = 0.0;
    let mut var2 = 0.0;

    for i in 0..n {
        let d1 = returns1.values[i] - mean1;
        let d2 = returns2.values[i] - mean2;
        cov += d1 * d2;
        var1 += d1 * d1;
        var2 += d2 * d2;
    }

    if var1 == 0.0 || var2 == 0.0 {
        return 0.0;
    }

    cov / (var1.sqrt() * var2.sqrt())
}

/// Calculate covariance between two return series
pub fn covariance(returns1: &Returns, returns2: &Returns) -> f64 {
    let n = returns1.len().min(returns2.len());
    if n < 2 {
        return 0.0;
    }

    let mean1: f64 = returns1.values[..n].iter().sum::<f64>() / n as f64;
    let mean2: f64 = returns2.values[..n].iter().sum::<f64>() / n as f64;

    let cov: f64 = (0..n)
        .map(|i| (returns1.values[i] - mean1) * (returns2.values[i] - mean2))
        .sum::<f64>()
        / (n - 1) as f64;

    cov
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_returns_statistics() {
        let returns = Returns {
            values: vec![0.01, -0.02, 0.03, -0.01, 0.02],
            timestamps: vec![1, 2, 3, 4, 5],
        };

        assert!((returns.mean() - 0.006).abs() < 1e-10);
        assert!(returns.std() > 0.0);
        assert_eq!(returns.count_positive(), 3);
        assert_eq!(returns.count_negative(), 2);
    }

    #[test]
    fn test_correlation() {
        let r1 = Returns {
            values: vec![0.01, 0.02, 0.03],
            timestamps: vec![1, 2, 3],
        };
        let r2 = Returns {
            values: vec![0.01, 0.02, 0.03],
            timestamps: vec![1, 2, 3],
        };

        let corr = correlation(&r1, &r2);
        assert!((corr - 1.0).abs() < 1e-10);
    }
}
